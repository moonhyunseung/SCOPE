"""CPU-only contracts for the SCOPE R0-B preparation package.

These functions validate evidence records. They do not reserve hardware,
authenticate a lock, monitor resources, or authorize execution.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from typing import Any


class ContractError(ValueError):
    """A frozen experiment or evidence contract is malformed or changed."""


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


# Explicit identities observed in the pinned vLLM build. This intentionally
# recognizes only canonical IDs and class paths; aliases, substring matches,
# and unlisted/composite paths remain UNKNOWN. Extend only with source-backed
# additions for this exact vLLM revision.
PINNED_VLLM_VERSION = "0.32.0"
PINNED_BACKEND_SOURCE_BLOBS = {
    "vllm/v1/attention/backends/registry.py": "207c112a384c1e1450362c7c91ac31893fd783bd",
    "vllm/v1/attention/backends/flash_attn.py": "6e906bc520c49e90bc6658c4a189b54215e5e28e",
    "vllm/v1/attention/backends/triton_attn.py": "a95b0c1d0e13e44154f9d63950769823b0abc47e",
}
PINNED_BACKEND_IDENTITIES: dict[str, tuple[str, str, str]] = {
    "FLASH_ATTN": (
        "vllm.v1.attention.backends.flash_attn.FlashAttentionBackend",
        "vllm.v1.attention.backends.flash_attn.FlashAttentionImpl",
        "vllm.v1.attention.backends.fa_utils.flash_attn_varlen_func",
    ),
    "TRITON_ATTN": (
        "vllm.v1.attention.backends.triton_attn.TritonAttentionBackend",
        "vllm.v1.attention.backends.triton_attn.TritonAttentionImpl",
        "vllm.v1.attention.ops.triton_unified_attention.unified_attention",
    ),
}
TOKENIZER_REQUIRED_FILES = ("tokenizer.json", "tokenizer_config.json")
TOKENIZER_OPTIONAL_FILES = (
    "added_tokens.json", "chat_template.jinja", "merges.txt",
    "special_tokens_map.json", "tokenizer.model", "vocab.json",
)


def validate_backend_evidence(record: dict[str, Any]) -> str:
    """Return VERIFIED only for a complete exact identity match per worker/layer.

    Evidence is not runtime attestation: the caller must preserve the records
    from the actual server/worker and this CPU check cannot prove their origin.
    """
    if not isinstance(record, dict) or record.get("vllm_version") != PINNED_VLLM_VERSION:
        return "BACKEND_UNVERIFIED"
    if record.get("vllm_commit") != "5df63a7203cdb9328d824a27b682e77c7b9928be":
        return "BACKEND_UNVERIFIED"
    if record.get("backend_source_blobs") != PINNED_BACKEND_SOURCE_BLOBS:
        return "BACKEND_UNVERIFIED"
    if record.get("runtime_override_active") is not False:
        return "BACKEND_UNVERIFIED"
    if record.get("gpu_model") != "NVIDIA H200" or record.get("compute_capability") != "SM90":
        return "BACKEND_UNVERIFIED"
    if record.get("hardware_execution_observed") is not False:
        return "BACKEND_UNVERIFIED"
    expected_worker_ids = record.get("expected_worker_ids")
    expected_layer_names = record.get("expected_attention_layer_names")
    if (not isinstance(expected_worker_ids, list) or not expected_worker_ids
            or any(not isinstance(value, str) or not value for value in expected_worker_ids)
            or len(set(expected_worker_ids)) != len(expected_worker_ids)
            or not isinstance(expected_layer_names, list) or not expected_layer_names
            or any(not isinstance(value, str) or not value for value in expected_layer_names)
            or len(set(expected_layer_names)) != len(expected_layer_names)):
        return "BACKEND_UNVERIFIED"
    workers = record.get("workers")
    if not isinstance(workers, list) or not workers:
        return "BACKEND_UNVERIFIED"
    seen_worker_ids: set[str] = set()
    for worker in workers:
        if not isinstance(worker, dict):
            return "BACKEND_UNVERIFIED"
        worker_id = worker.get("worker_id")
        startup = worker.get("startup_backend_id")
        layers = worker.get("layers")
        if not isinstance(worker_id, str) or not worker_id or worker_id in seen_worker_ids:
            return "BACKEND_UNVERIFIED"
        seen_worker_ids.add(worker_id)
        if not isinstance(startup, str) or startup not in PINNED_BACKEND_IDENTITIES:
            return "BACKEND_UNVERIFIED"
        if not isinstance(layers, list) or not layers:
            return "BACKEND_UNVERIFIED"
        expected_backend, expected_impl, expected_entrypoint = PINNED_BACKEND_IDENTITIES[startup]
        seen_layers: set[str] = set()
        for layer in layers:
            if not isinstance(layer, dict):
                return "BACKEND_UNVERIFIED"
            layer_name = layer.get("layer_name")
            class_path = layer.get("effective_backend_class")
            implementation_path = layer.get("effective_impl_class")
            entrypoint_path = layer.get("effective_entrypoint")
            if (not isinstance(layer_name, str) or not layer_name or layer_name in seen_layers
                    or not isinstance(class_path, str) or not isinstance(implementation_path, str)
                    or not isinstance(entrypoint_path, str)
                    or not class_path or not implementation_path or not entrypoint_path):
                return "BACKEND_UNVERIFIED"
            seen_layers.add(layer_name)
            if (class_path != expected_backend or implementation_path != expected_impl
                    or entrypoint_path != expected_entrypoint):
                return "BACKEND_UNVERIFIED"
            if startup == "FLASH_ATTN":
                flash_version = layer.get("flash_attention_version")
                # vLLM's pinned source supports FA3 on Hopper/SM90. FA2 and
                # FA4 remain distinct explicit versions; this is source-level
                # consistency only, not evidence that a kernel actually ran.
                if type(flash_version) is not int or flash_version not in (2, 3, 4):
                    return "BACKEND_UNVERIFIED"
                if flash_version == 3 and record.get("compute_capability") != "SM90":
                    return "BACKEND_UNVERIFIED"
            if startup == "TRITON_ATTN" and not (isinstance(layer.get("effective_kernel_dispatch_id"), str)
                                                   and layer["effective_kernel_dispatch_id"]):
                return "BACKEND_UNVERIFIED"
        if seen_layers != set(expected_layer_names):
            return "BACKEND_UNVERIFIED"
    if seen_worker_ids != set(expected_worker_ids):
        return "BACKEND_UNVERIFIED"
    # This validates the explicit source identity only. It does not attest
    # hardware execution or resolve Triton's JIT kernel choice.
    return "BACKEND_EVIDENCE_CONSISTENT"


def _finite_number(value: Any, *, positive: bool = False) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and (value > 0 if positive else value >= 0))


def validate_resource_preflight(
    record: dict[str, Any], *, now_monotonic_ns: int | None = None,
    max_sample_age_seconds: float = 5.0,
) -> list[str]:
    """Validate a measurement record's shape and frozen bounds, never readiness."""
    if not isinstance(record, dict):
        return ["RESOURCE_RECORD_INVALID"]
    failures: list[str] = []
    if now_monotonic_ns is None:
        now_monotonic_ns = time.monotonic_ns()
    observed = record.get("observed_monotonic_ns")
    if type(observed) is not int or observed <= 0 or observed > now_monotonic_ns:
        failures.append("RESOURCE_TIMESTAMP_INVALID")
    elif (not _finite_number(max_sample_age_seconds, positive=True)
          or now_monotonic_ns - observed > max_sample_age_seconds * 1_000_000_000):
        failures.append("RESOURCE_SAMPLE_STALE")
    count = record.get("gpu_count")
    if type(count) is not int or count != 1:
        failures.append("GPU_COUNT_NOT_ONE")
    if record.get("gpu_model") != "NVIDIA H200":
        failures.append("GPU_MODEL_MISMATCH")
    if record.get("exclusive_lock_verified") is not True:
        failures.append("RESOURCE_OWNERSHIP_UNKNOWN")
    if not _finite_number(record.get("physical_hbm_free_gib"), positive=True) or record["physical_hbm_free_gib"] < 44:
        failures.append("HBM_RESERVE_BREACH")
    if not _finite_number(record.get("workload_hbm_increase_gib")) or record["workload_hbm_increase_gib"] > 88:
        failures.append("HBM_DELTA_BREACH")
    if not _finite_number(record.get("allocator_gib"), positive=True) or record["allocator_gib"] > 96:
        failures.append("ALLOCATOR_LIMIT_BREACH")
    if not _finite_number(record.get("disk_free_gib"), positive=True) or record["disk_free_gib"] < 24:
        failures.append("DISK_RESERVE_BREACH")
    if not _finite_number(record.get("wall_time_seconds"), positive=True) or record["wall_time_seconds"] > 1800:
        failures.append("WALL_TIME_LIMIT_BREACH")
    return failures


def validate_request_set(
    requests: list[dict[str, Any]], *, request_prefix: str, count: int,
    prompt_tokens: int, output_tokens: int, expected_ids: list[str] | None = None,
) -> None:
    """Check exact ID/order, prompt uniqueness, exact lengths and output lengths."""
    if len(requests) != count:
        raise ContractError(f"request count mismatch: {len(requests)} != {count}")
    ids = [item.get("request_id") for item in requests]
    expected = expected_ids or [f"{request_prefix}{i}" for i in range(count)]
    if ids != expected or len(set(ids)) != count:
        raise ContractError("request IDs are duplicated, missing, or out of order")
    prompt_hashes: list[str] = []
    for item in requests:
        if item.get("prompt_tokens") != prompt_tokens or item.get("output_tokens") != output_tokens:
            raise ContractError(f"length mismatch for {item.get('request_id')}")
        prompt_hash = item.get("prompt_sha256")
        token_hash = item.get("token_ids_sha256")
        if not isinstance(prompt_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", prompt_hash):
            raise ContractError("missing or invalid prompt hash")
        if not isinstance(token_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", token_hash):
            raise ContractError("missing or invalid token-ID hash")
        prompt_hashes.append(prompt_hash)
    if len(set(prompt_hashes)) != count:
        raise ContractError("duplicate prompts in request set")


def validate_warmup_sets(
    measured: list[dict[str, Any]], warmups: list[dict[str, Any]],
    per_rep: int, repetitions: int = 1,
) -> None:
    if not measured or per_rep <= 0 or repetitions <= 0:
        raise ContractError("warm-up schedule requires measured requests and positive dimensions")
    if len(warmups) != per_rep * repetitions:
        raise ContractError("warm-up request count mismatch")
    measured_ids = {item.get("request_id") for item in measured}
    measured_hashes = {item.get("prompt_sha256") for item in measured}
    expected_ids = [f"WU-{measured[0]['request_id'].split('-', 1)[0]}-R{rep}-{index}"
                    for rep in range(repetitions) for index in range(per_rep)]
    warmup_ids = [item.get("request_id") for item in warmups]
    warmup_hashes = [item.get("prompt_sha256") for item in warmups]
    assignments = [item.get("repetition") for item in warmups]
    expected_assignments = [rep for rep in range(repetitions) for _ in range(per_rep)]
    if (warmup_ids != expected_ids or assignments != expected_assignments
            or any(type(value) is not int for value in assignments)):
        raise ContractError("warm-up IDs/order/repetition assignments do not match the frozen schedule")
    if len(set(warmup_ids)) != len(warmup_ids) or len(set(warmup_hashes)) != per_rep * repetitions:
        raise ContractError("warm-up identities must be unique")
    if measured_ids.intersection(warmup_ids) or measured_hashes.intersection(warmup_hashes):
        raise ContractError("warm-up requests must be disjoint from measured requests")


def validate_tokenizer_manifest(manifest: dict[str, Any]) -> None:
    if not isinstance(manifest, dict) or manifest.get("revision") != "989aa7980e4cf806f80c7fef2b1adb7bc71aa306":
        raise ContractError("tokenizer manifest revision mismatch")
    entries = manifest.get("files")
    if manifest.get("required_files") != list(TOKENIZER_REQUIRED_FILES):
        raise ContractError("tokenizer required-file policy is missing or changed")
    if not isinstance(entries, list) or not entries:
        raise ContractError("tokenizer file manifest is empty")
    paths: list[str] = []
    seen_required: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ContractError("invalid tokenizer file manifest entry")
        path, digest, size = entry.get("path"), entry.get("sha256"), entry.get("size_bytes")
        if not isinstance(path, str) or not path or path.startswith("/") or ".." in path.split("/"):
            raise ContractError("tokenizer manifest path must be safe and relative")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ContractError("invalid tokenizer file digest")
        if type(size) is not int or size < 0:
            raise ContractError("invalid tokenizer file size")
        if path not in TOKENIZER_REQUIRED_FILES + TOKENIZER_OPTIONAL_FILES:
            raise ContractError("tokenizer manifest contains an unapproved non-tokenizer artifact")
        if entry.get("source_kind") not in ("snapshot_file", "hf_cache_blob_symlink"):
            raise ContractError("tokenizer artifact provenance is missing or unsupported")
        target = entry.get("cache_blob_relpath")
        if entry["source_kind"] == "hf_cache_blob_symlink":
            if (not isinstance(target, str) or not target.startswith("blobs/")
                    or len(target.split("/")) != 2 or ".." in target.split("/")):
                raise ContractError("tokenizer cache symlink provenance is unsafe")
        elif target is not None:
            raise ContractError("regular tokenizer files cannot claim a cache symlink target")
        paths.append(path)
        if path in TOKENIZER_REQUIRED_FILES:
            seen_required.add(path)
    if paths != sorted(paths) or len(set(paths)) != len(paths):
        raise ContractError("tokenizer manifest paths must be sorted and unique")
    if seen_required != set(TOKENIZER_REQUIRED_FILES):
        raise ContractError("required tokenizer files are missing")


def validate_server_tokenization(local_requests: list[dict[str, Any]], server_requests: list[dict[str, Any]]) -> None:
    """Check /tokenize preflight parity; this is not inference-path evidence."""
    if not local_requests or len(local_requests) != len(server_requests):
        raise ContractError("server request count differs from sealed local set")
    for index, (local, server) in enumerate(zip(local_requests, server_requests, strict=True)):
        if local.get("request_id") != server.get("request_id"):
            raise ContractError(f"server request order/identity mismatch at index {index}")
        if local.get("token_ids_sha256") != server.get("token_ids_sha256"):
            raise ContractError(f"server token IDs differ at index {index}")
        if local.get("prompt_tokens") != server.get("prompt_tokens"):
            raise ContractError(f"server prompt length differs at index {index}")


def validate_inference_token_identity(
    local_requests: list[dict[str, Any]], inference_requests: list[dict[str, Any]],
    *, evidence_stage: str,
) -> None:
    """Validate token IDs observed at engine ingestion, not /tokenize output."""
    if evidence_stage != "inference_path_observed":
        raise ContractError("/tokenize parity is preflight only, not inference-path identity")
    if not local_requests or len(local_requests) != len(inference_requests):
        raise ContractError("inference-path request count differs from sealed local set")
    for index, (local, observed) in enumerate(zip(local_requests, inference_requests, strict=True)):
        if not isinstance(observed, dict):
            raise ContractError(f"invalid inference-path evidence record at index {index}")
        if (observed.get("request_id") != local.get("request_id")
                or type(observed.get("sequence_index")) is not int
                or observed["sequence_index"] != index
                or observed.get("token_ids_sha256") != local.get("token_ids_sha256")
                or type(observed.get("token_count")) is not int
                or observed["token_count"] != local.get("prompt_tokens")
                or type(observed.get("evidence_monotonic_ns")) is not int
                or observed["evidence_monotonic_ns"] <= 0):
            raise ContractError(f"inference-path token identity evidence mismatch at index {index}")


def validate_result_record(record: dict[str, Any]) -> None:
    """Validate one CPU-visible request result against the frozen schema."""
    if not isinstance(record, dict):
        raise ContractError("result record must be an object")
    required = {
        "condition_id", "request_id", "repetition", "phase", "prompt_tokens",
        "output_tokens_requested", "output_tokens_observed", "token_ids_sha256",
        "server_token_ids_sha256", "token_identity_evidence_stage", "ttft_ms",
        "itl_ms", "tpot_ms", "e2e_ms", "enqueue_monotonic_ns",
        "dispatch_monotonic_ns", "queue_wait_ms", "status", "error_code",
        "backend_evidence_ref", "resource_samples_ref",
    }
    if not required.issubset(record):
        raise ContractError("result record omits required schema fields")
    if record.get("condition_id") not in ("W1", "W2"):
        raise ContractError("unknown result condition")
    if not isinstance(record.get("request_id"), str) or not record["request_id"]:
        raise ContractError("result request identity is missing")
    if type(record.get("repetition")) is not int or record["repetition"] not in (0, 1, 2):
        raise ContractError("invalid result repetition")
    if record.get("phase") not in ("warmup", "measured"):
        raise ContractError("invalid result phase")
    for key in ("prompt_tokens", "output_tokens_requested"):
        if type(record.get(key)) is not int or record[key] <= 0:
            raise ContractError(f"invalid result {key}")
    observed_output = record.get("output_tokens_observed")
    if observed_output is not None and (type(observed_output) is not int or observed_output < 0):
        raise ContractError("invalid observed output length")
    for key in ("token_ids_sha256",):
        if not isinstance(record.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", record[key]):
            raise ContractError(f"invalid result {key}")
    for key in ("server_token_ids_sha256",):
        value = record.get(key)
        if value is not None and (not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)):
            raise ContractError(f"invalid result {key}")
    if record.get("token_identity_evidence_stage") not in (
        "tokenize_preflight_only", "inference_path_observed", "unverified"
    ):
        raise ContractError("invalid token identity evidence stage")
    if record.get("status") not in ("success", "error", "timeout", "cancelled"):
        raise ContractError("invalid request completion status")
    for key in ("ttft_ms", "tpot_ms", "e2e_ms", "queue_wait_ms"):
        value = record.get(key)
        if value is not None and not _finite_number(value):
            raise ContractError(f"invalid result metric {key}")
    itl = record.get("itl_ms")
    if itl is not None and (not isinstance(itl, list) or any(not _finite_number(value) for value in itl)):
        raise ContractError("invalid ITL series")
    for key in ("enqueue_monotonic_ns", "dispatch_monotonic_ns"):
        value = record.get(key)
        if value is not None and (type(value) is not int or value <= 0):
            raise ContractError(f"invalid result timestamp {key}")
    for key in ("error_code", "backend_evidence_ref", "resource_samples_ref"):
        if record.get(key) is not None and not isinstance(record[key], str):
            raise ContractError(f"invalid result reference {key}")


def validate_r0b1_readiness(packet: dict[str, Any]) -> list[str]:
    """Return blockers. This validator cannot authorize any execution."""
    blockers: list[str] = []
    if not isinstance(packet, dict):
        return ["PACKET_INVALID"]
    # This preparation checker never returns an authorization result. Preserve
    # the blocker even if all evidence fields are later populated.
    blockers.append("R0B1_NOT_AUTHORIZED")
    if packet.get("status") != "NOT_AUTHORIZED" or packet.get("researcher_approval") is not False:
        blockers.append("AUTHORIZATION_STATE_INVALID")
    for key in ("input_manifest_sha256", "tokenizer_manifest_sha256", "source_contract_sha256"):
        value = packet.get(key)
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            blockers.append(f"{key.upper()}_PENDING")
    if packet.get("server_tokenization_identity") != "VERIFIED":
        blockers.append("SERVER_TOKENIZATION_NOT_VERIFIED")
    if packet.get("selected_backend") != "VERIFIED":
        blockers.append("BACKEND_NOT_VERIFIED")
    if not isinstance(packet.get("gpu_lock_owner"), str) or packet.get("gpu_lock_owner") in ("", "UNKNOWN"):
        blockers.append("GPU_LOCK_OWNER_UNKNOWN_REQUIRED")
    if packet.get("live_monitor") != "IMPLEMENTED_VERIFIED":
        blockers.append("LIVE_RESOURCE_MONITOR_NOT_IMPLEMENTED")
    if not isinstance(packet.get("dispatch_policy"), dict):
        blockers.append("DISPATCH_POLICY_MISSING")
    else:
        policy = packet["dispatch_policy"]
        for key in ("mode", "arrival_profile", "concurrency_cap", "client_queue_metrics"):
            if key not in policy or policy[key] in (None, "", "PENDING", "UNVERIFIED"):
                blockers.append(f"DISPATCH_{key.upper()}_PENDING")
    result_schema = packet.get("result_schema")
    if (not isinstance(result_schema, dict) or not {
        "condition_id", "request_id", "repetition", "phase", "prompt_tokens",
        "output_tokens_requested", "output_tokens_observed", "token_ids_sha256",
        "server_token_ids_sha256", "token_identity_evidence_stage", "ttft_ms",
        "itl_ms", "tpot_ms", "e2e_ms", "enqueue_monotonic_ns",
        "dispatch_monotonic_ns", "queue_wait_ms", "status", "error_code",
        "backend_evidence_ref", "resource_samples_ref",
    }.issubset(result_schema)):
        blockers.append("RESULT_SCHEMA_MISSING")
    slo = packet.get("slo_contract")
    expected_slo = {
        "W1": {"ttft_ms_max": 8000, "tpot_ms_max": 120},
        "W2": {"ttft_ms_max": 3000, "tpot_ms_max": 120},
    }
    if (not isinstance(slo, dict) or slo.get("workloads") != expected_slo
            or slo.get("status") != "PROPOSED_REQUIRES_SCIENTIFIC_APPROVAL"):
        blockers.append("SLO_CONTRACT_MISSING_OR_CHANGED")
    measurement = packet.get("measurement_contract", {})
    if not isinstance(measurement, dict):
        blockers.append("MEASUREMENT_CONTRACT_INVALID")
        measurement = {}
    for key in ("temperature", "eos_policy", "prefix_caching_enabled", "goodput_definition",
                "failure_denominator", "manifest_canonicalization"):
        if not isinstance(measurement.get(key), (str, int, float, bool)):
            blockers.append(f"MEASUREMENT_{key.upper()}_MISSING")
    if type(measurement.get("temperature")) not in (int, float) or measurement.get("temperature") != 0:
        blockers.append("TEMPERATURE_MISSING_OR_CHANGED")
    if measurement.get("eos_policy") != "stop_on_first_model_eos_or_output_token_cap_128; record observed output token count":
        blockers.append("EOS_POLICY_MISSING_OR_CHANGED")
    if measurement.get("goodput_definition") != (
        "SLO-satisfying successful measured unique requests divided by measured wall-clock seconds; "
        "no warm-up requests in numerator or denominator."
    ):
        blockers.append("GOODPUT_DEFINITION_MISSING_OR_CHANGED")
    if measurement.get("failure_denominator") != (
        "All planned measured unique requests remain in the denominator; report success, error, timeout, "
        "cancelled and SLO-miss counts separately; never replace."
    ):
        blockers.append("FAILURE_DENOMINATOR_MISSING_OR_CHANGED")
    if measurement.get("manifest_canonicalization") != (
        "UTF-8 JSON; object keys sorted lexicographically; separators comma-colon without spaces; "
        "ensure_ascii=false; one trailing LF; SHA256 over exact canonical bytes."
    ):
        blockers.append("MANIFEST_CANONICALIZATION_MISSING_OR_CHANGED")
    if measurement.get("prefix_caching_enabled") is not False:
        blockers.append("PREFIX_CACHING_MUST_BE_DISABLED")
    if measurement.get("arrival_policy") != "saturated_closed_loop_concurrency_limited":
        blockers.append("ARRIVAL_POLICY_MISSING_OR_CHANGED")
    identity = packet.get("token_identity_evidence", {})
    if not isinstance(identity, dict):
        blockers.append("TOKEN_IDENTITY_EVIDENCE_INVALID")
        identity = {}
    if identity.get("tokenize_endpoint_role") != "preflight_only_not_inference_proof":
        blockers.append("TOKENIZE_EVIDENCE_SCOPE_MISSING")
    if identity.get("inference_path_evidence") != "request_correlated_token_id_hash_at_engine_ingestion":
        blockers.append("INFERENCE_IDENTITY_EVIDENCE_MISSING")
    if not isinstance(packet.get("readiness_checklist"), list) or not packet["readiness_checklist"]:
        blockers.append("READINESS_CHECKLIST_MISSING")
    return blockers

