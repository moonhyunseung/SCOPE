"""CPU-only fail-closed watchdog core for SCOPE R0-B1.

This module does not query CUDA/NVML, acquire a GPU lock, launch workloads, or
terminate operating-system processes. A deployment must inject a trusted
resource provider and an owner-scoped child-tree controller before it can be
considered operationally ready.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Protocol


GiB = 1024**3
MIN_PHYSICAL_FREE_BYTES = 44 * GiB
MAX_JOB_HBM_GROWTH_BYTES = 88 * GiB
MAX_ALLOCATOR_BYTES = 96 * GiB
MIN_DISK_FREE_BYTES = 24 * GiB
MAX_WALL_SECONDS = 1800.0
MAX_SAMPLE_AGE_SECONDS = 5.0


@dataclass(frozen=True)
class Observation:
    """One timestamped sample produced by a trusted external provider."""

    observed_monotonic: float
    gpu_model: str
    gpu_count: int
    compute_capability: tuple[int, int]
    lock_verified: bool
    lock_identity: str | None
    lock_owner_matches: bool
    physical_hbm_free_bytes: int
    job_hbm_baseline_bytes: int
    job_hbm_current_bytes: int
    allocator_bytes: int
    disk_free_bytes: int
    worker_failed: bool = False
    oom_detected: bool = False


class OwnedChildTree(Protocol):
    """Controller bound to exactly the child tree created by this run."""

    def terminate_owned_tree(self, reason: str) -> None: ...


class WatchdogStop(RuntimeError):
    """A fail-closed stop condition, suitable for private run evidence."""


def validate_observation(
    observation: Observation,
    *,
    now_monotonic: float,
    started_monotonic: float,
) -> None:
    """Raise WatchdogStop unless every R0-B1 resource gate is satisfied."""

    for value in (observation.observed_monotonic, now_monotonic, started_monotonic):
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            raise WatchdogStop("MONOTONIC_TIMESTAMP_INVALID")
    byte_counts = (
        observation.physical_hbm_free_bytes,
        observation.job_hbm_baseline_bytes,
        observation.job_hbm_current_bytes,
        observation.allocator_bytes,
        observation.disk_free_bytes,
    )
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in byte_counts):
        raise WatchdogStop("RESOURCE_COUNTER_INVALID")
    age = now_monotonic - observation.observed_monotonic
    if age < 0 or age > MAX_SAMPLE_AGE_SECONDS:
        raise WatchdogStop("RESOURCE_SAMPLE_STALE_OR_FUTURE")
    if now_monotonic - started_monotonic > MAX_WALL_SECONDS:
        raise WatchdogStop("DEADLINE_EXCEEDED")
    if observation.oom_detected:
        raise WatchdogStop("OOM_DETECTED")
    if observation.worker_failed:
        raise WatchdogStop("WORKER_FAILURE_REPORTED")
    if (
        observation.gpu_model != "NVIDIA H200"
        or observation.gpu_count != 1
        or observation.compute_capability != (9, 0)
    ):
        raise WatchdogStop("GPU_IDENTITY_MISMATCH")
    if not observation.lock_verified or not observation.lock_identity:
        raise WatchdogStop("GPU_LOCK_MISSING_OR_LOST")
    if not observation.lock_owner_matches:
        raise WatchdogStop("GPU_LOCK_OWNER_MISMATCH")
    if observation.job_hbm_current_bytes < observation.job_hbm_baseline_bytes:
        raise WatchdogStop("JOB_HBM_COUNTER_REGRESSION")
    if observation.physical_hbm_free_bytes < MIN_PHYSICAL_FREE_BYTES:
        raise WatchdogStop("PHYSICAL_HBM_RESERVE_BREACHED")
    if (
        observation.job_hbm_current_bytes - observation.job_hbm_baseline_bytes
        > MAX_JOB_HBM_GROWTH_BYTES
    ):
        raise WatchdogStop("JOB_HBM_GROWTH_LIMIT_EXCEEDED")
    if observation.allocator_bytes > MAX_ALLOCATOR_BYTES:
        raise WatchdogStop("ALLOCATOR_LIMIT_EXCEEDED")
    if observation.disk_free_bytes < MIN_DISK_FREE_BYTES:
        raise WatchdogStop("DISK_RESERVE_BREACHED")


def enforce_observation(
    observation: Observation,
    *,
    now_monotonic: float,
    started_monotonic: float,
    owned_child_tree: OwnedChildTree,
) -> None:
    """Check one sample and stop only the injected, run-owned child tree.

    The controller is deliberately mandatory and injected. This function has
    no fallback that discovers or kills processes by name, PID, or GPU use.
    """

    try:
        validate_observation(
            observation,
            now_monotonic=now_monotonic,
            started_monotonic=started_monotonic,
        )
    except WatchdogStop as exc:
        owned_child_tree.terminate_owned_tree(str(exc))
        raise


def run_monitor(
    *,
    read_observation: Callable[[], Observation | None],
    is_child_running: Callable[[], bool],
    child_exit_code: Callable[[], int | None],
    owned_child_tree: OwnedChildTree,
    started_monotonic: float,
    monotonic: Callable[[], float],
    sleep: Callable[[float], None],
) -> None:
    """Poll once per second until the owned child exits; fail closed on gaps.

    This loop is hardware-agnostic. Deployments must provide their own trusted
    sampling and child-tree adapters; neither adapter is implemented here.
    """

    while True:
        try:
            child_running = is_child_running()
        except Exception as exc:
            reason = f"CHILD_STATE_PROVIDER_ERROR:{type(exc).__name__}"
            owned_child_tree.terminate_owned_tree(reason)
            raise WatchdogStop(reason) from exc
        if not child_running:
            try:
                exit_code = child_exit_code()
            except Exception as exc:
                reason = f"CHILD_EXIT_STATUS_ERROR:{type(exc).__name__}"
                raise WatchdogStop(reason) from exc
            if exit_code is None:
                raise WatchdogStop("CHILD_EXIT_STATUS_UNKNOWN")
            if exit_code != 0:
                raise WatchdogStop("WORKER_EXIT_NONZERO")
            return
        try:
            observation = read_observation()
        except Exception as exc:
            reason = f"RESOURCE_PROVIDER_ERROR:{type(exc).__name__}"
            owned_child_tree.terminate_owned_tree(reason)
            raise WatchdogStop(reason) from exc
        if observation is None:
            owned_child_tree.terminate_owned_tree("RESOURCE_SAMPLE_MISSING")
            raise WatchdogStop("RESOURCE_SAMPLE_MISSING")
        now = monotonic()
        enforce_observation(
            observation,
            now_monotonic=now,
            started_monotonic=started_monotonic,
            owned_child_tree=owned_child_tree,
        )
        try:
            sleep(1.0)
        except Exception as exc:
            reason = f"MONITOR_SLEEP_ERROR:{type(exc).__name__}"
            owned_child_tree.terminate_owned_tree(reason)
            raise WatchdogStop(reason) from exc

