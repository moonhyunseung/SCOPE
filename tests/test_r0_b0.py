import json
import unittest
from pathlib import Path

from scripts.r0_b0_validate import (
    ContractError,
    sha256_json,
    validate_backend_evidence,
    validate_contract,
    validate_resource_preflight,
)


CONTRACT = Path(__file__).resolve().parents[1] / "configs" / "r0_b0_contract.json"


class R0B0ContractTests(unittest.TestCase):
    def setUp(self):
        self.contract = json.loads(CONTRACT.read_text(encoding="utf-8"))

    def test_contract_is_frozen_and_not_execution_authority(self):
        validate_contract(self.contract)

    def test_contract_rejects_changed_workload_or_authorization(self):
        bad = json.loads(json.dumps(self.contract))
        bad["workloads"][0]["prompt_tokens"] += 1
        with self.assertRaises(ContractError):
            validate_contract(bad)
        bad = json.loads(json.dumps(self.contract))
        bad["execution"]["r0b1_authorized"] = True
        with self.assertRaises(ContractError):
            validate_contract(bad)

    def test_canonical_hash_is_stable_and_ordered(self):
        self.assertEqual(sha256_json({"a": 1, "b": [2, 3]}), sha256_json({"b": [2, 3], "a": 1}))
        self.assertNotEqual(sha256_json([1, 2]), sha256_json([2, 1]))

    def test_backend_verification_fails_closed(self):
        self.assertEqual(validate_backend_evidence(None, ["FlashAttentionBackend"]), "BACKEND_UNVERIFIED")
        self.assertEqual(validate_backend_evidence("FLASH_ATTN", ["FlashAttentionBackend"]), "VERIFIED")
        self.assertEqual(validate_backend_evidence("FLASH_ATTN", ["FlashAttentionBackend", "FlashAttentionBackend"]), "VERIFIED")
        self.assertEqual(validate_backend_evidence("FLASH_ATTN", ["TritonAttentionBackend"]), "BACKEND_UNVERIFIED")
        self.assertEqual(validate_backend_evidence("FLASH_ATTN", ["FlashAttentionBackend", ""]), "BACKEND_UNVERIFIED")

    def test_resource_preflight_requires_all_reserves_and_lock(self):
        record = {
            "gpu_count": 1, "gpu_model": "NVIDIA H200", "exclusive_lock_verified": True,
            "physical_hbm_free_gib": 45, "workload_hbm_increase_gib": 80,
            "allocator_gib": 96, "disk_free_gib": 24, "wall_time_seconds": 1800,
        }
        self.assertEqual(validate_resource_preflight(record), [])
        record["exclusive_lock_verified"] = False
        record["physical_hbm_free_gib"] = 43
        self.assertEqual(validate_resource_preflight(record), ["RESOURCE_OWNERSHIP_UNKNOWN", "HBM_RESERVE_BREACH"])


if __name__ == "__main__":
    unittest.main()

