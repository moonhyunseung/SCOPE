#!/usr/bin/env python3
"""Prepare a private CPU-only synthetic request pool via pinned vLLM source.

This is a future R0-B0 preparation utility. It does not run automatically,
load model weights, start a server, invoke CUDA APIs, or perform inference. It
requires an already installed vLLM 0.32.0 and an already cached tokenizer
snapshot. No downloads or fallback installs are attempted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = REPOSITORY_ROOT / "configs" / "r0_b0_contract.json"
EXPECTED_VLLM_VERSION = "0.32.0"
MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
EXPECTED_SOURCE_BLOBS = {
    "vllm/benchmarks/datasets/datasets.py": "3746ac727e0f6da859d250ad52882ece820e5e39",
    "vllm/benchmarks/datasets/utils.py": "a2dd5093c50b2e7a08ebd33d1467c20f47df5894",
}


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def tokenizer_snapshot_manifest(snapshot: Path) -> dict[str, Any]:
    """Hash tokenizer artifacts in the pinned, local HF cache snapshot.

    Standard snapshot-file symlinks are accepted only when they resolve to a
    regular content blob under this model's own ``blobs`` directory. Hashing
    streams bytes and never reads model weights. This does not load tokenizer
    code or access the network.
    """
    from scripts.r0_b0_validate import (
        TOKENIZER_OPTIONAL_FILES,
        TOKENIZER_REQUIRED_FILES,
        validate_tokenizer_manifest,
    )

    root = Path(snapshot).absolute()
    if (root.is_symlink() or root.name != MODEL_REVISION or root.parent.name != "snapshots"
            or root.parent.parent.name != "models--Qwen--Qwen2.5-1.5B-Instruct"
            or not root.is_dir()):
        raise RuntimeError("tokenizer path must be the pinned local Hugging Face snapshot")
    cache_repo = root.parent.parent.resolve(strict=True)
    resolved_root = root.resolve(strict=True)
    if resolved_root != root.resolve():
        raise RuntimeError("snapshot directory must not resolve through an unexpected path")
    allowed = set(TOKENIZER_REQUIRED_FILES) | set(TOKENIZER_OPTIONAL_FILES)
    for name in TOKENIZER_REQUIRED_FILES:
        candidate = root / name
        if not candidate.exists() and not candidate.is_symlink():
            raise RuntimeError(f"required tokenizer file is missing: {name}")
    entries = []
    for name in sorted(allowed):
        path = root / name
        if not path.exists() and not path.is_symlink():
            continue
        if path.is_symlink():
            try:
                resolved = path.resolve(strict=True)
            except (OSError, RuntimeError) as exc:
                raise RuntimeError(f"broken or unsafe tokenizer symlink: {name}") from exc
            try:
                resolved.relative_to(cache_repo)
            except ValueError as exc:
                raise RuntimeError(f"tokenizer symlink escapes the pinned cache repository: {name}") from exc
            if (resolved.parent != (cache_repo / "blobs").resolve(strict=True)
                    or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", resolved.name)
                    or not resolved.is_file()):
                raise RuntimeError(f"tokenizer symlink must target a content blob: {name}")
            source_kind = "hf_cache_blob_symlink"
            link_target = resolved.relative_to(cache_repo).as_posix()
        else:
            resolved = path.resolve(strict=True)
            if resolved.parent != resolved_root or not resolved.is_file():
                raise RuntimeError(f"tokenizer artifact must be a regular snapshot file: {name}")
            source_kind = "snapshot_file"
            link_target = None
        digest = hashlib.sha256()
        size_bytes = 0
        try:
            with resolved.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    size_bytes += len(chunk)
                    digest.update(chunk)
        except OSError as exc:
            raise RuntimeError(f"cannot read tokenizer artifact: {name}") from exc
        entries.append({
            "path": name,
            "size_bytes": size_bytes,
            "sha256": digest.hexdigest(),
            "source_kind": source_kind,
            "cache_blob_relpath": link_target,
        })
    manifest = {
        "revision": MODEL_REVISION,
        "required_files": list(TOKENIZER_REQUIRED_FILES),
        "files": entries,
    }
    validate_tokenizer_manifest(manifest)
    return manifest


def check_vllm_source() -> tuple[Any, Any]:
    import vllm
    from vllm.benchmarks.datasets.datasets import RandomDataset

    if getattr(vllm, "__version__", None) != EXPECTED_VLLM_VERSION:
        raise RuntimeError(f"expected vLLM {EXPECTED_VLLM_VERSION}; found {getattr(vllm, '__version__', 'unknown')}")
    module_paths = {
        "vllm/benchmarks/datasets/datasets.py": Path(sys.modules[RandomDataset.__module__].__file__),
    }
    import vllm.benchmarks.datasets.utils as dataset_utils
    module_paths["vllm/benchmarks/datasets/utils.py"] = Path(dataset_utils.__file__)
    for relative, expected in EXPECTED_SOURCE_BLOBS.items():
        actual = git_blob_sha1(module_paths[relative])
        if actual != expected:
            raise RuntimeError(f"pinned source mismatch for {relative}: expected {expected}, got {actual}")
    return vllm, RandomDataset


def generate(workload: dict[str, Any], tokenizer_snapshot: Path) -> tuple[list[dict[str, Any]], list[list[int]], dict[str, Any]]:
    resolved = tokenizer_snapshot.resolve(strict=True)
    if MODEL_REVISION not in resolved.parts:
        raise RuntimeError("tokenizer path must be an existing local snapshot directory named by the pinned revision")
    manifest_before = tokenizer_snapshot_manifest(resolved)
    from transformers import AutoTokenizer

    _, RandomDataset = check_vllm_source()
    tokenizer = AutoTokenizer.from_pretrained(
        str(resolved), local_files_only=True, trust_remote_code=False
    )
    requested_prompt_len = int(workload["prompt_tokens"])
    # RandomDataset subtracts num_special_tokens_to_add() from input_len.
    # Compensate so the resulting prompt has the contract's requested length.
    sampler_input_len = requested_prompt_len + int(tokenizer.num_special_tokens_to_add())
    warmup_count = int(workload["warmup_requests_per_repetition"]) * int(workload["repetitions"])
    measured_count = int(workload["unique_requests"])
    # One deterministic sample gives a shared RNG stream. The first measured
    # requests retain the frozen W1-i/W2-i IDs; the tail is partitioned into
    # repetition-specific warm-up identities.
    sampled = RandomDataset(random_seed=20261019).sample(
        tokenizer=tokenizer,
        num_requests=measured_count + warmup_count,
        request_id_prefix=f"{workload['id']}-",
        prefix_len=0,
        range_ratio=0.0,
        input_len=sampler_input_len,
        output_len=int(workload["output_tokens"]),
    )
    requests: list[dict[str, Any]] = []
    warmups: list[dict[str, Any]] = []
    all_token_ids: list[list[int]] = []
    for index, req in enumerate(sampled):
        if not isinstance(req.prompt, str):
            raise RuntimeError(f"non-text prompt at request {index}")
        token_ids = tokenizer.encode(req.prompt, add_special_tokens=False)
        if len(token_ids) != requested_prompt_len or req.prompt_len != requested_prompt_len:
            raise RuntimeError(
                f"prompt length mismatch at {index}: encoded={len(token_ids)}, sampler={req.prompt_len}, expected={requested_prompt_len}"
            )
        if req.expected_output_len != int(workload["output_tokens"]):
            raise RuntimeError(f"output length mismatch at request {index}")
        all_token_ids.append(token_ids)
        target = requests if index < measured_count else warmups
        if index < measured_count:
            request_id = f"{workload['id']}-{index}"
        else:
            warm_index = index - measured_count
            per_rep = int(workload["warmup_requests_per_repetition"])
            request_id = f"WU-{workload['id']}-R{warm_index // per_rep}-{warm_index % per_rep}"
        target.append({
            "request_id": request_id,
            **({"repetition": warm_index // per_rep} if index >= measured_count else {}),
            "prompt": req.prompt,
            "prompt_tokens": len(token_ids),
            "output_tokens": req.expected_output_len,
            "prompt_sha256": hashlib.sha256(req.prompt.encode("utf-8")).hexdigest(),
            "token_ids_sha256": hashlib.sha256(canonical_json(token_ids)).hexdigest(),
        })
    from scripts.r0_b0_validate import validate_request_set, validate_warmup_sets

    validate_request_set(
        requests,
        request_prefix=f"{workload['id']}-",
        count=measured_count,
        prompt_tokens=requested_prompt_len,
        output_tokens=int(workload["output_tokens"]),
    )
    validate_warmup_sets(
        requests, warmups,
        int(workload["warmup_requests_per_repetition"]),
        repetitions=int(workload["repetitions"]),
    )
    manifest_after = tokenizer_snapshot_manifest(resolved)
    if manifest_before != manifest_after:
        raise RuntimeError("tokenizer snapshot changed while requests were being prepared")
    return requests + warmups, all_token_ids, manifest_after


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", choices=("W1", "W2"), required=True)
    parser.add_argument("--tokenizer-snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="private local JSONL path; do not commit")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    workload = next(item for item in contract["workloads"] if item["id"] == args.workload)
    requests, all_token_ids, tokenizer_manifest = generate(workload, args.tokenizer_snapshot)
    payload = b"".join(canonical_json(row) for row in requests)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    summary = {
        "status": "LOCAL_REQUESTS_PREPARED_HASHES_PENDING_REVIEW",
        "workload": args.workload,
        "request_count": len(requests),
        "measured_request_count": int(workload["unique_requests"]),
        "warmup_request_count": int(workload["warmup_requests_per_repetition"]) * int(workload["repetitions"]),
        "request_source_sha256": hashlib.sha256(payload).hexdigest(),
        "ordered_token_ids_sha256": hashlib.sha256(canonical_json(all_token_ids)).hexdigest(),
        "tokenizer_snapshot_manifest_sha256": hashlib.sha256(canonical_json(tokenizer_manifest)).hexdigest(),
        "prompt_lengths": sorted({r["prompt_tokens"] for r in requests}),
        "output_lengths": sorted({r["output_tokens"] for r in requests}),
        "tokenizer_revision": MODEL_REVISION,
        "vllm_version": EXPECTED_VLLM_VERSION,
        "gpu_used": False,
        "model_loaded": False,
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

