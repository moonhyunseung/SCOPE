import copy
import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.r0_b0_validate import (
    ContractError,
    PINNED_BACKEND_SOURCE_BLOBS,
    sha256_json,
    validate_backend_evidence,
    validate_contract,
    validate_request_set,
    validate_resource_preflight,
    validate_r0b1_readiness,
    validate_server_tokenization,
    validate_tokenizer_manifest,
    validate_warmup_sets,
)
from scripts.r0_b0_prepare_requests import MODEL_REVISION, tokenizer_snapshot_manifest


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "r0_b0_contract.json"
PREFLIGHT = ROOT / "configs" / "r0_b1_preflight_template.json"
FLASH_BACKEND = "vllm.v1.attention.backends.flash_attn.FlashAttentionBackend"
FLASH_IMPL = "vllm.v1.attention.backends.flash_attn.FlashAttentionImpl"
TRITON_BACKEND = "vllm.v1.attention.backends.triton_attn.TritonAttentionBackend"
TRITON_IMPL = "vllm.v1.attention.backends.triton_attn.TritonAttentionImpl"


def backend_record(backend_id="FLASH_ATTN", backend=FLASH_BACKEND, impl=FLASH_IMPL):
    entrypoint = ("vllm.v1.attention.backends.fa_utils.flash_attn_varlen_func"
                  if backend_id == "FLASH_ATTN"
                  else "vllm.v1.attention.ops.triton_unified_attention.unified_attention")
    layer = {"layer_name": "model.layers.0.self_attn", "effective_backend_class": backend,
             "effective_impl_class": impl, "effective_entrypoint": entrypoint}
    if backend_id == "FLASH_ATTN":
        layer["flash_attention_version"] = 4
    elif backend_id == "TRITON_ATTN":
        layer["effective_kernel_dispatch_id"] = "mocked_kernel_dispatch"
    return {
        "vllm_version": "0.32.0",
        "vllm_commit": "5df63a7203cdb9328d824a27b682e77c7b9928be",
        "backend_source_blobs": dict(PINNED_BACKEND_SOURCE_BLOBS),
        "runtime_override_active": False,
        "workers": [{
            "worker_id": "worker-0",
            "startup_backend_id": backend_id,
            "layers": [layer],
        }],
    }


def resource_record():
    return {
        "gpu_count": 1,
        "gpu_model": "NVIDIA H200",
        "exclusive_lock_verified": True,
        "physical_hbm_free_gib": 44.0,
        "workload_hbm_increase_gib": 88.0,
        "allocator_gib": 96.0,
        "disk_free_gib": 24.0,
        "wall_time_seconds": 1800.0,
    }


def request(rid, prompt_hash, token_hash=None, prompt_tokens=512, output_tokens=128):
    return {
        "request_id": rid,
        "prompt_sha256": prompt_hash,
        "token_ids_sha256": token_hash or prompt_hash,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
    }


class R0B0PreparationTests(unittest.TestCase):
    def setUp(self):
        self.contract = json.loads(CONTRACT.read_text(encoding="utf-8"))

    def test_frozen_contract_and_not_execution_authority(self):
        validate_contract(self.contract)
        bad = copy.deepcopy(self.contract)
        bad["workloads"][0]["prompt_tokens"] += 1
        with self.assertRaises(ContractError):
            validate_contract(bad)
        bad = copy.deepcopy(self.contract)
        bad["execution"]["r0b1_authorized"] = True
        with self.assertRaises(ContractError):
            validate_contract(bad)

    def test_canonical_hash_is_stable_but_order_sensitive(self):
        self.assertEqual(sha256_json({"a": 1, "b": [2, 3]}), sha256_json({"b": [2, 3], "a": 1}))
        self.assertNotEqual(sha256_json([1, 2]), sha256_json([2, 1]))

    def test_backend_requires_exact_version_pinned_worker_layer_identity(self):
        self.assertEqual(validate_backend_evidence(backend_record()), "BACKEND_EVIDENCE_CONSISTENT")
        self.assertEqual(validate_backend_evidence(backend_record("TRITON_ATTN", TRITON_BACKEND, TRITON_IMPL)), "BACKEND_EVIDENCE_CONSISTENT")
        self.assertEqual(validate_backend_evidence(backend_record("TRITON_FLASH_ATTN", FLASH_BACKEND, FLASH_IMPL)), "BACKEND_UNVERIFIED")
        self.assertEqual(validate_backend_evidence(backend_record(backend="SomeFlashAttentionBackendAlias")), "BACKEND_UNVERIFIED")
        self.assertEqual(validate_backend_evidence(backend_record(impl="vllm.custom.FlashAttentionImpl")), "BACKEND_UNVERIFIED")
        self.assertEqual(validate_backend_evidence({**backend_record(), "vllm_commit": "other"}), "BACKEND_UNVERIFIED")
        mixed = backend_record()
        mixed["workers"].append(copy.deepcopy(mixed["workers"][0]))
        self.assertEqual(validate_backend_evidence(mixed), "BACKEND_UNVERIFIED")
        no_entrypoint = backend_record()
        no_entrypoint["workers"][0]["layers"][0]["effective_entrypoint"] = ""
        self.assertEqual(validate_backend_evidence(no_entrypoint), "BACKEND_UNVERIFIED")
        overridden = backend_record()
        overridden["runtime_override_active"] = True
        self.assertEqual(validate_backend_evidence(overridden), "BACKEND_UNVERIFIED")
        wrong_source = backend_record()
        wrong_source["backend_source_blobs"]["vllm/v1/attention/backends/registry.py"] = "0" * 40
        self.assertEqual(validate_backend_evidence(wrong_source), "BACKEND_UNVERIFIED")

    def test_resource_record_boundaries_and_invalid_numeric_values_fail_closed(self):
        record = resource_record()
        self.assertEqual(validate_resource_preflight(record), [])
        for field, value in (
            ("physical_hbm_free_gib", float("inf")),
            ("workload_hbm_increase_gib", float("nan")),
            ("allocator_gib", -1),
            ("disk_free_gib", True),
            ("wall_time_seconds", -0.1),
            ("gpu_count", True),
        ):
            invalid = {**record, field: value}
            self.assertTrue(validate_resource_preflight(invalid), (field, value))

    def test_request_identity_order_uniqueness_and_lengths(self):
        rows = [request("W1-0", "a" * 64), request("W1-1", "b" * 64)]
        validate_request_set(rows, request_prefix="W1-", count=2, prompt_tokens=512, output_tokens=128)
        for bad in ([rows[1], rows[0]], [rows[0], rows[0]], [rows[0]], [rows[0], request("W1-1", "b" * 64, prompt_tokens=511)]):
            with self.assertRaises(ContractError):
                validate_request_set(bad, request_prefix="W1-", count=2, prompt_tokens=512, output_tokens=128)

    def test_warmups_are_separate_unique_and_not_in_measured_set(self):
        measured = [request("W1-0", "a" * 64)]
        warmups = [request("WU-W1-R0-0", "b" * 64), request("WU-W1-R1-0", "c" * 64)]
        validate_warmup_sets(measured, warmups, 2)
        with self.assertRaises(ContractError):
            validate_warmup_sets(measured, [warmups[0], request("WU-W1-R1-0", "a" * 64)], 2)

    def test_tokenizer_manifest_requires_sorted_safe_content_hashes(self):
        good = {"revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306", "files": [
            {"path": "tokenizer.json", "size_bytes": 0, "sha256": "a" * 64},
            {"path": "tokenizer_config.json", "size_bytes": 7, "sha256": "b" * 64},
        ]}
        validate_tokenizer_manifest(good)
        for files in ([{"path": "../secret", "size_bytes": 1, "sha256": "a" * 64}], list(reversed(good["files"]))):
            with self.assertRaises(ContractError):
                validate_tokenizer_manifest({**good, "files": files})

    def test_tokenizer_snapshot_content_hash_is_deterministic_without_loading_tokenizer(self):
        with TemporaryDirectory() as tmp:
            snapshot = Path(tmp) / MODEL_REVISION
            snapshot.mkdir()
            (snapshot / "tokenizer.json").write_text('{"vocab": [1, 2]}', encoding="utf-8")
            first = tokenizer_snapshot_manifest(snapshot)
            second = tokenizer_snapshot_manifest(snapshot)
            self.assertEqual(first, second)
            validate_tokenizer_manifest(first)
            self.assertEqual(first["files"][0]["sha256"], hashlib.sha256(b'{"vocab": [1, 2]}').hexdigest())

    def test_server_tokenization_requires_exact_order_ids_hash_and_lengths(self):
        local = [request("W1-0", "a" * 64), request("W1-1", "b" * 64)]
        validate_server_tokenization(local, copy.deepcopy(local))
        altered = copy.deepcopy(local)
        altered[0]["token_ids_sha256"] = "c" * 64
        with self.assertRaises(ContractError):
            validate_server_tokenization(local, altered)
        reordered = list(reversed(local))
        with self.assertRaises(ContractError):
            validate_server_tokenization(local, reordered)

    def test_readiness_template_is_explicitly_blocked_and_never_authorizes(self):
        packet = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
        blockers = validate_r0b1_readiness(packet)
        self.assertIn("R0B1_NOT_AUTHORIZED", blockers)
        self.assertIn("INPUT_MANIFEST_SHA256_PENDING", blockers)
        self.assertTrue(blockers)
        fully_filled = copy.deepcopy(packet)
        digest = hashlib.sha256(b"fixture").hexdigest()
        for key in ("source_contract_sha256", "input_manifest_sha256", "tokenizer_manifest_sha256"):
            fully_filled[key] = digest
        self.assertIn("R0B1_NOT_AUTHORIZED", validate_r0b1_readiness(fully_filled))


if __name__ == "__main__":
    unittest.main()
