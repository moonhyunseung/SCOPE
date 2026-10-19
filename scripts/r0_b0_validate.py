"""CPU-only fail-closed validators for the SCOPE R0-B0 contract."""

from __future__ import annotations

import hashlib
import json
from typing import Any


class ContractError(ValueError):
    pass


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def validate_contract(contract: dict[str, Any]) -> None:
    if contract.get("status") != "PREPARATION_ONLY_NOT_EXECUTED":
        raise ContractError("status must remain PREPARATION_ONLY_NOT_EXECUTED")
    if contract.get("execution", {}).get("r0b1_authorized") is not False:
        raise ContractError("R0-B1 must remain unauthorized")
    workloads = contract.get("workloads")
    if not isinstance(workloads, list) or [w.get("id") for w in workloads] != ["W1", "W2"]:
        raise ContractError("workloads must contain W1 and W2 in order")
    expected = [(8192, 128, 1, 64, 3, 4), (512, 128, 16, 128, 3, 16)]
    for item, row in zip(workloads, expected, strict=True):
        observed = tuple(item.get(key) for key in (
            "prompt_tokens", "output_tokens", "concurrency", "unique_requests",
            "repetitions", "warmup_requests_per_repetition"))
        if observed != row or item.get("prefix_cache") is not False:
            raise ContractError(f"workload mismatch: {item.get('id')}")
    hashes = contract.get("request_generation", {}).get("hashes", {})
    if any(hashes.get(key) != "PENDING_LOCAL_PREPARATION" for key in (
        "request_source_sha256", "ordered_token_ids_sha256", "tokenizer_snapshot_manifest_sha256")):
        raise ContractError("not-yet-generated request hashes must be pending")


def validate_backend_evidence(startup_backend: str | None, effective_worker_classes: list[str] | None) -> str:
    if not startup_backend or not effective_worker_classes:
        return "BACKEND_UNVERIFIED"
    if any(not isinstance(item, str) or not item.strip() for item in effective_worker_classes):
        return "BACKEND_UNVERIFIED"
    def canonical_backend(value: str) -> str:
        compact = "".join(ch for ch in value.casefold() if ch.isalnum())
        if "flashattn" in compact or "flashattention" in compact:
            return "flashattention"
        if "triton" in compact:
            return "triton"
        if "xformers" in compact:
            return "xformers"
        if "torchsdpa" in compact or "pytorchsdpa" in compact:
            return "torchsdpa"
        return compact

    selected = canonical_backend(startup_backend)
    effective = {canonical_backend(item) for item in effective_worker_classes}
    if not selected or effective != {selected}:
        return "BACKEND_UNVERIFIED"
    return "VERIFIED"


def validate_resource_preflight(record: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    checks = (
        (record.get("gpu_count") == 1, "GPU_COUNT_NOT_ONE"),
        (record.get("gpu_model") == "NVIDIA H200", "GPU_MODEL_MISMATCH"),
        (record.get("exclusive_lock_verified") is True, "RESOURCE_OWNERSHIP_UNKNOWN"),
        (isinstance(record.get("physical_hbm_free_gib"), (int, float)) and record["physical_hbm_free_gib"] >= 44, "HBM_RESERVE_BREACH"),
        (isinstance(record.get("workload_hbm_increase_gib"), (int, float)) and 0 <= record["workload_hbm_increase_gib"] <= 88, "HBM_DELTA_BREACH"),
        (isinstance(record.get("allocator_gib"), (int, float)) and record["allocator_gib"] <= 96, "ALLOCATOR_LIMIT_BREACH"),
        (isinstance(record.get("disk_free_gib"), (int, float)) and record["disk_free_gib"] >= 24, "DISK_RESERVE_BREACH"),
        (isinstance(record.get("wall_time_seconds"), (int, float)) and 0 <= record["wall_time_seconds"] <= 1800, "WALL_TIME_LIMIT_BREACH"),
    )
    failures.extend(code for passed, code in checks if not passed)
    return failures

