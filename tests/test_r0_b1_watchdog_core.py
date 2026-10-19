"""Deterministic CPU-only tests for the R0-B1 watchdog core."""

import unittest

from scripts.r0_b1_watchdog_core import (
    GiB,
    Observation,
    WatchdogStop,
    enforce_observation,
    run_monitor,
    validate_observation,
)


def sample(**overrides):
    fields = {
        "observed_monotonic": 100.0,
        "gpu_model": "NVIDIA H200",
        "gpu_count": 1,
        "compute_capability": (9, 0),
        "lock_verified": True,
        "lock_identity": "mock-lock-1",
        "lock_owner_matches": True,
        "physical_hbm_free_bytes": 44 * GiB,
        "job_hbm_baseline_bytes": 10 * GiB,
        "job_hbm_current_bytes": 98 * GiB,
        "allocator_bytes": 96 * GiB,
        "disk_free_bytes": 24 * GiB,
    }
    fields.update(overrides)
    return Observation(**fields)


class FakeOwnedTree:
    def __init__(self):
        self.reasons = []

    def terminate_owned_tree(self, reason):
        self.reasons.append(reason)


class WatchdogTests(unittest.TestCase):
    def test_exact_contract_boundaries_pass(self):
        validate_observation(sample(), now_monotonic=105.0, started_monotonic=0.0)
        validate_observation(
            sample(observed_monotonic=1795.0),
            now_monotonic=1800.0,
            started_monotonic=0.0,
        )

    def test_each_invalid_gate_fails_closed(self):
        cases = [
            ({"observed_monotonic": 94.9}, 100.0, 0.0, "RESOURCE_SAMPLE_STALE_OR_FUTURE"),
            ({"observed_monotonic": 100.1}, 100.0, 0.0, "RESOURCE_SAMPLE_STALE_OR_FUTURE"),
            ({"observed_monotonic": 1796.0}, 1800.01, 0.0, "DEADLINE_EXCEEDED"),
            ({"gpu_count": 2}, 100.0, 0.0, "GPU_IDENTITY_MISMATCH"),
            ({"compute_capability": (8, 0)}, 100.0, 0.0, "GPU_IDENTITY_MISMATCH"),
            ({"lock_verified": False}, 100.0, 0.0, "GPU_LOCK_MISSING_OR_LOST"),
            ({"lock_identity": None}, 100.0, 0.0, "GPU_LOCK_MISSING_OR_LOST"),
            ({"lock_owner_matches": False}, 100.0, 0.0, "GPU_LOCK_OWNER_MISMATCH"),
            ({"physical_hbm_free_bytes": 44 * GiB - 1}, 100.0, 0.0, "PHYSICAL_HBM_RESERVE_BREACHED"),
            ({"job_hbm_current_bytes": 98 * GiB + 1}, 100.0, 0.0, "JOB_HBM_GROWTH_LIMIT_EXCEEDED"),
            ({"allocator_bytes": 96 * GiB + 1}, 100.0, 0.0, "ALLOCATOR_LIMIT_EXCEEDED"),
            ({"disk_free_bytes": 24 * GiB - 1}, 100.0, 0.0, "DISK_RESERVE_BREACHED"),
            ({"job_hbm_current_bytes": 9 * GiB}, 100.0, 0.0, "JOB_HBM_COUNTER_REGRESSION"),
            ({"disk_free_bytes": True}, 100.0, 0.0, "RESOURCE_COUNTER_INVALID"),
            ({"observed_monotonic": float("nan")}, 100.0, 0.0, "MONOTONIC_TIMESTAMP_INVALID"),
        ]
        for overrides, now, started, expected in cases:
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(WatchdogStop, expected):
                    validate_observation(sample(**overrides), now_monotonic=now, started_monotonic=started)

    def test_enforcement_calls_only_injected_owned_tree(self):
        owned = FakeOwnedTree()
        with self.assertRaisesRegex(WatchdogStop, "GPU_LOCK_OWNER_MISMATCH"):
            enforce_observation(
                sample(lock_owner_matches=False),
                now_monotonic=100.0,
                started_monotonic=0.0,
                owned_child_tree=owned,
            )
        self.assertEqual(owned.reasons, ["GPU_LOCK_OWNER_MISMATCH"])

    def test_missing_sample_stops_owned_tree(self):
        owned = FakeOwnedTree()
        with self.assertRaisesRegex(WatchdogStop, "RESOURCE_SAMPLE_MISSING"):
            run_monitor(
                read_observation=lambda: None,
                is_child_running=lambda: True,
                child_exit_code=lambda: None,
                owned_child_tree=owned,
                started_monotonic=0.0,
                monotonic=lambda: 1.0,
                sleep=lambda _: self.fail("must not sleep after missing sample"),
            )
        self.assertEqual(owned.reasons, ["RESOURCE_SAMPLE_MISSING"])

    def test_provider_error_stops_owned_tree(self):
        owned = FakeOwnedTree()

        def broken_provider():
            raise OSError("private provider detail")

        with self.assertRaisesRegex(WatchdogStop, "RESOURCE_PROVIDER_ERROR:OSError"):
            run_monitor(
                read_observation=broken_provider,
                is_child_running=lambda: True,
                child_exit_code=lambda: None,
                owned_child_tree=owned,
                started_monotonic=0.0,
                monotonic=lambda: 1.0,
                sleep=lambda _: self.fail("must not sleep after provider error"),
            )
        self.assertEqual(owned.reasons, ["RESOURCE_PROVIDER_ERROR:OSError"])

    def test_healthy_monitor_samples_once_per_second_until_child_exits(self):
        owned = FakeOwnedTree()
        running = iter((True, True, False))
        times = iter((100.0, 101.0))
        sleeps = []
        run_monitor(
            read_observation=lambda: sample(observed_monotonic=100.0),
            is_child_running=lambda: next(running),
            child_exit_code=lambda: 0,
            owned_child_tree=owned,
            started_monotonic=99.0,
            monotonic=lambda: next(times),
            sleep=sleeps.append,
        )
        self.assertEqual(sleeps, [1.0, 1.0])
        self.assertEqual(owned.reasons, [])

    def test_worker_failure_and_oom_are_stop_conditions(self):
        for field, reason in (("worker_failed", "WORKER_FAILURE_REPORTED"), ("oom_detected", "OOM_DETECTED")):
            with self.subTest(reason=reason):
                owned = FakeOwnedTree()
                with self.assertRaisesRegex(WatchdogStop, reason):
                    enforce_observation(
                        sample(**{field: True}),
                        now_monotonic=100.0,
                        started_monotonic=0.0,
                        owned_child_tree=owned,
                    )
                self.assertEqual(owned.reasons, [reason])

    def test_nonzero_child_exit_is_reported_without_process_discovery(self):
        owned = FakeOwnedTree()
        with self.assertRaisesRegex(WatchdogStop, "WORKER_EXIT_NONZERO"):
            run_monitor(
                read_observation=lambda: self.fail("no sample required after child exit"),
                is_child_running=lambda: False,
                child_exit_code=lambda: 3,
                owned_child_tree=owned,
                started_monotonic=0.0,
                monotonic=lambda: 0.0,
                sleep=lambda _: self.fail("must not sleep after child exit"),
            )
        self.assertEqual(owned.reasons, [])


if __name__ == "__main__":
    unittest.main()

