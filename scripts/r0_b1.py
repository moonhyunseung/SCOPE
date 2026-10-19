#!/usr/bin/env python3
"""R0-B1: actual vLLM OpenAI-completions serving baseline client.

Run only against an approved, dedicated, already-started vLLM 0.32.0 server.
No GPU process is started/stopped here. CPU-only --check does not import torch.
Generated prompts and token IDs are never written to result artifacts.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
from pathlib import Path
import time
from urllib import request, error, parse

from scripts.r0_b0_validate import (ContractError, validate_contract,
                                    validate_request_set, validate_warmup_sets,
                                    validate_result_record)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "r0_b0_contract.json"
MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"


def canonical(obj):
    return (json.dumps(obj, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":")) + "\n").encode("utf-8")


def digest(obj):
    return hashlib.sha256(canonical(obj)).hexdigest()


def load_requests(path, workload):
    raw = path.read_bytes()
    lines = [json.loads(line) for line in raw.splitlines() if line.strip()]
    n = workload["unique_requests"]
    measured = lines[:n]
    warmups = lines[n:]
    validate_request_set(measured, request_prefix=workload["id"] + "-",
                         count=n, prompt_tokens=workload["prompt_tokens"],
                         output_tokens=workload["output_tokens"])
    validate_warmup_sets(measured, warmups,
                         workload["warmup_requests_per_repetition"],
                         repetitions=workload["repetitions"])
    for row in lines:
        if not isinstance(row.get("prompt"), str):
            raise ContractError("prepared request must contain a private text prompt")
        if hashlib.sha256(row["prompt"].encode()).hexdigest() != row.get("prompt_sha256"):
            raise ContractError("prompt hash mismatch: " + str(row.get("request_id")))
    return measured, warmups, hashlib.sha256(raw).hexdigest()


def check_and_tokenize(rows, snapshot):
    # This is a local-only CPU operation; no download or model weight loading.
    if (snapshot.name != MODEL_REVISION or not snapshot.is_dir()
            or snapshot.parent.name != "snapshots"
            or snapshot.parent.parent.name != "models--Qwen--Qwen2.5-1.5B-Instruct"):
        raise ContractError("tokenizer must be the pinned local HF snapshot")
    from scripts.r0_b0_prepare_requests import tokenizer_snapshot_manifest
    from transformers import AutoTokenizer

    before = tokenizer_snapshot_manifest(snapshot)
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True,
                                              trust_remote_code=False)
    for row in rows:
        ids = tokenizer.encode(row["prompt"], add_special_tokens=False)
        if len(ids) != row["prompt_tokens"] or digest(ids) != row["token_ids_sha256"]:
            raise ContractError("token ID mismatch: " + row["request_id"])
        row["_ids"] = ids  # in memory only; never serialize rows
    if before != tokenizer_snapshot_manifest(snapshot):
        raise ContractError("tokenizer snapshot changed during preparation")
    return digest(before)


def get_json(base, route, timeout=10):
    with request.urlopen(base + route, timeout=timeout) as response:
        return json.loads(response.read())


def server_preflight(base, model):
    version = get_json(base, "/version")
    if version.get("version") != "0.32.0":
        raise ContractError("server must report vLLM 0.32.0, got " + repr(version))
    models = get_json(base, "/v1/models")
    if model not in [item.get("id") for item in models.get("data", [])]:
        raise ContractError("served model ID does not match explicit model selection")
    return {"server_vllm_version": version["version"],
            "served_model_id": model,
            "server_commit": "NOT_ATTESTED_FROM_API",
            "effective_attention_backend": "NOT_ATTESTED_FROM_API"}


def percentiles(values):
    if not values:
        return {"n": 0, "p50": None, "p95": None, "p99": None}
    s = sorted(values)
    def at(p):
        i = (len(s) - 1) * p
        a = math.floor(i)
        return round(s[a] + (s[math.ceil(i)] - s[a]) * (i - a), 4)
    return {"n": len(s), "p50": at(.50), "p95": at(.95), "p99": at(.99)}


def new_record(row, workload, rep, phase, queued_ns, dispatched_ns):
    return {
        "condition_id": workload["id"], "request_id": row["request_id"],
        "repetition": rep, "phase": phase,
        "prompt_tokens": row["prompt_tokens"],
        "output_tokens_requested": row["output_tokens"],
        "output_tokens_observed": None,
        "token_ids_sha256": row["token_ids_sha256"],
        "server_token_ids_sha256": None,
        "token_identity_evidence_stage": "unverified",
        "ttft_ms": None, "itl_ms": None, "tpot_ms": None, "e2e_ms": None,
        "enqueue_monotonic_ns": queued_ns,
        "dispatch_monotonic_ns": dispatched_ns,
        "queue_wait_ms": (dispatched_ns - queued_ns) / 1e6,
        "status": "error", "error_code": None,
        "backend_evidence_ref": None, "resource_samples_ref": None,
        "stream_token_chunks": 0, "stream_tokens_accounted": 0,
        "finish_reason": None,
    }


def send_one(row, workload, rep, phase, queued_ns, base, model, timeout):
    dispatched_ns = time.monotonic_ns()
    record = new_record(row, workload, rep, phase, queued_ns, dispatched_ns)
    body = {
        "model": model, "prompt": row["_ids"], "max_tokens": row["output_tokens"],
        "temperature": 0, "seed": 20261019, "n": 1, "echo": False,
        "stream": True, "stream_interval": 1, "return_token_ids": True,
        "stream_options": {"include_usage": True}, "ignore_eos": False,
    }
    payload = json.dumps(body, separators=(",", ":")).encode()
    req = request.Request(base + "/v1/completions", data=payload,
                          headers={"Content-Type": "application/json"}, method="POST")
    first_ns = None
    token_times = []
    chunk_count = 0
    token_count = 0
    final_usage = None
    prompt_seen = False
    done = False
    finish_reason = None
    try:
        with request.urlopen(req, timeout=timeout) as stream:
            data_lines = []
            while True:
                line = stream.readline()
                if not line:
                    break
                line = line.decode("utf-8").rstrip("\r\n")
                if line.startswith("data:"):
                    data_lines.append(line[5:].strip())
                elif not line and data_lines:
                    datum = "\n".join(data_lines)
                    data_lines = []
                    if datum == "[DONE]":
                        done = True
                        break
                    event = json.loads(datum)
                    if "error" in event:
                        raise ValueError("SERVER_STREAM_ERROR")
                    usage = event.get("usage")
                    if usage is not None:
                        final_usage = usage
                    for choice in event.get("choices", []):
                        if choice.get("index", 0) != 0:
                            raise ValueError("UNEXPECTED_CHOICE_INDEX")
                        server_prompt = choice.get("prompt_token_ids")
                        if server_prompt is not None:
                            if prompt_seen or server_prompt != row["_ids"]:
                                raise ValueError("SERVER_PROMPT_TOKEN_IDS_MISMATCH")
                            prompt_seen = True
                            record["server_token_ids_sha256"] = digest(server_prompt)
                            record["token_identity_evidence_stage"] = "tokenize_preflight_only"
                        tokens = choice.get("token_ids")
                        if tokens is None:
                            raise ValueError("MISSING_STREAM_TOKEN_IDS")
                        if tokens:
                            now_ns = time.monotonic_ns()
                            if first_ns is None:
                                first_ns = now_ns
                            token_count += len(tokens)
                            chunk_count += 1
                            if len(tokens) == 1:
                                token_times.append(now_ns)
                        if choice.get("finish_reason") is not None:
                            finish_reason = choice["finish_reason"]
        record["stream_token_chunks"] = chunk_count
        record["stream_tokens_accounted"] = token_count
        record["finish_reason"] = finish_reason
        record["e2e_ms"] = (time.monotonic_ns() - dispatched_ns) / 1e6
        if first_ns is not None:
            record["ttft_ms"] = (first_ns - dispatched_ns) / 1e6
        if not done or first_ns is None or finish_reason is None:
            raise ValueError("INCOMPLETE_STREAM")
        if not prompt_seen:
            raise ValueError("MISSING_SERVER_PROMPT_IDS")
        if final_usage is None or final_usage.get("prompt_tokens") != row["prompt_tokens"]:
            raise ValueError("MISSING_OR_MISMATCHED_PROMPT_USAGE")
        if final_usage.get("completion_tokens") != token_count or token_count <= 0:
            raise ValueError("MISSING_OR_MISMATCHED_COMPLETION_USAGE")
        if token_count > row["output_tokens"]:
            raise ValueError("OUTPUT_CAP_EXCEEDED")
        record["output_tokens_observed"] = token_count
        # A stream chunk is not necessarily a token. No interpolated per-token timings.
        if len(token_times) == token_count and token_count >= 2:
            record["itl_ms"] = [(b - a) / 1e6 for a, b in zip(token_times, token_times[1:])]
            record["tpot_ms"] = (token_times[-1] - token_times[0]) / (token_count - 1) / 1e6
        record["status"] = "success"
    except error.HTTPError as ex:
        record["error_code"] = "HTTP_" + str(ex.code)
    except (error.URLError, OSError) as ex:
        record["error_code"] = "TRANSPORT_" + type(ex).__name__.upper()
        if isinstance(ex, TimeoutError):
            record["status"] = "timeout"
    except (ValueError, json.JSONDecodeError, UnicodeDecodeError) as ex:
        record["error_code"] = str(ex) if str(ex) in {
            "SERVER_STREAM_ERROR", "UNEXPECTED_CHOICE_INDEX", "SERVER_PROMPT_TOKEN_IDS_MISMATCH",
            "MISSING_STREAM_TOKEN_IDS", "INCOMPLETE_STREAM", "MISSING_SERVER_PROMPT_IDS",
            "MISSING_OR_MISMATCHED_PROMPT_USAGE", "MISSING_OR_MISMATCHED_COMPLETION_USAGE",
            "OUTPUT_CAP_EXCEEDED"} else "INVALID_STREAM_PAYLOAD"
    except Exception as ex:
        # No raw prompts, HTTP bodies or token IDs in public-facing errors.
        record["error_code"] = "UNEXPECTED_" + type(ex).__name__.upper()
    finally:
        if record["e2e_ms"] is None:
            record["e2e_ms"] = (time.monotonic_ns() - dispatched_ns) / 1e6
        if record["status"] != "success":
            record["output_tokens_observed"] = token_count
        validate_result_record(record)
    return record


def run_wave(rows, workload, rep, phase, base, model, timeout, out):
    t0 = time.monotonic_ns()
    records = []
    with ThreadPoolExecutor(max_workers=workload["concurrency"]) as pool:
        futures = []
        for row in rows:
            queued_ns = time.monotonic_ns()
            futures.append(pool.submit(send_one, row, workload, rep, phase,
                                       queued_ns, base, model, timeout))
        for future in as_completed(futures):
            record = future.result()
            # Durable incremental output: no private prompt or token array in record.
            out.write(canonical(record))
            out.flush()
            records.append(record)
    t1 = time.monotonic_ns()
    return sorted(records, key=lambda rec: rec["request_id"]), (t1 - t0) / 1e9


def summarize(repetitions, workload):
    all_rows = [row for rep in repetitions for row in rep["records"]]
    successful = [row for row in all_rows if row["status"] == "success"]
    def meeting(row):
        return (row["ttft_ms"] is not None and row["tpot_ms"] is not None
                and row["ttft_ms"] <= workload["ttft_slo_ms"]
                and row["tpot_ms"] <= workload["tpot_slo_ms"])
    duration = sum(rep["measured_wall_s"] for rep in repetitions)
    n_good = sum(meeting(row) for row in successful)
    total_tokens = sum(row["output_tokens_observed"] for row in successful)
    return {
        "planned_measured_n": workload["unique_requests"] * workload["repetitions"],
        "observed_measured_n": len(all_rows),
        "success_n": len(successful),
        "error_n": sum(r["status"] == "error" for r in all_rows),
        "timeout_n": sum(r["status"] == "timeout" for r in all_rows),
        "cancelled_n": sum(r["status"] == "cancelled" for r in all_rows),
        "slo_pass_n": n_good,
        "slo_fail_or_unmeasurable_n": len(all_rows) - n_good,
        "measured_wall_seconds": duration,
        "output_tokens_per_second": total_tokens / duration if duration > 0 else None,
        "requests_per_second": len(successful) / duration if duration > 0 else None,
        "slo_goodput_requests_per_second": n_good / duration if duration > 0 else None,
        "TTFT_ms": percentiles([r["ttft_ms"] for r in successful if r["ttft_ms"] is not None]),
        "TPOT_ms": percentiles([r["tpot_ms"] for r in successful if r["tpot_ms"] is not None]),
        "E2E_ms": percentiles([r["e2e_ms"] for r in successful if r["e2e_ms"] is not None]),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workload", required=True, choices=("W1", "W2"))
    ap.add_argument("--requests", required=True, type=Path)
    ap.add_argument("--check", action="store_true", help="CPU schema/hash preflight only")
    ap.add_argument("--run", action="store_true", help="send real GPU-backed serving requests")
    ap.add_argument("--gpu-approved", action="store_true", help="explicit operator authorization; not a lock")
    ap.add_argument("--tokenizer-snapshot", type=Path)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--served-model-name", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--results-dir", type=Path)
    ap.add_argument("--limit-measured", type=int, default=None)
    ap.add_argument("--repetitions", type=int, default=None)
    ap.add_argument("--request-timeout-s", type=float, default=180.0)
    args = ap.parse_args(argv)
    if args.check == args.run:
        ap.error("select exactly one of --check or --run")
    contract = json.loads(CONTRACT.read_text())
    validate_contract(contract)
    workload = next(w for w in contract["workloads"] if w["id"] == args.workload)
    measured, warmups, source_sha = load_requests(args.requests, workload)
    if args.check:
        print(json.dumps({"status": "CPU_INPUT_CHECK_OK", "workload": args.workload,
                          "measured_n": len(measured), "warmup_n": len(warmups),
                          "input_sha256": source_sha}, sort_keys=True))
        return 0
    if not args.gpu_approved:
        ap.error("--run requires externally approved GPU ownership and --gpu-approved")
    if args.tokenizer_snapshot is None or args.results_dir is None:
        ap.error("--run needs --tokenizer-snapshot and --results-dir")
    url = parse.urlparse(args.base_url)
    if url.scheme != "http" or url.hostname not in ("127.0.0.1", "localhost", "::1") or url.path not in ("", "/"):
        ap.error("only a dedicated local HTTP server is supported")
    if args.request_timeout_s <= 0 or not math.isfinite(args.request_timeout_s):
        ap.error("request timeout must be finite and positive")
    reps = workload["repetitions"] if args.repetitions is None else args.repetitions
    limit = workload["unique_requests"] if args.limit_measured is None else args.limit_measured
    if not 1 <= reps <= workload["repetitions"] or not 1 <= limit <= workload["unique_requests"]:
        ap.error("requested pilot cannot exceed frozen run dimensions")
    rows = measured[:limit] + [w for w in warmups if w["repetition"] < reps]
    tokenizer_sha = check_and_tokenize(rows, args.tokenizer_snapshot)
    base = args.base_url.rstrip("/")
    server = server_preflight(base, args.served_model_name)
    args.results_dir.mkdir(parents=True, exist_ok=False)
    execution = []
    # Observed throughput excludes warmups. No retries and no synthetic latency values.
    with (args.results_dir / "requests.jsonl").open("wb") as results_file:
        for rep in range(reps):
            wu = [row for row in rows if row["request_id"].startswith("WU-") and row["repetition"] == rep]
            warmup_result, _ = run_wave(wu, workload, rep, "warmup", base,
                                        args.served_model_name, args.request_timeout_s, results_file)
            if any(row["status"] != "success" for row in warmup_result):
                raise RuntimeError("warmup failed; partial records preserved in requests.jsonl")
            record_set, elapsed = run_wave(measured[:limit], workload, rep, "measured", base,
                                           args.served_model_name, args.request_timeout_s, results_file)
            execution.append({"repetition": rep, "records": record_set,
                              "measured_wall_s": elapsed})
    stats = summarize(execution, workload)
    summary = {
        "status": "SMOKE_ONLY" if reps != workload["repetitions"] or limit != workload["unique_requests"]
        else "FULL_CLIENT_SERVING_BASELINE_UNATTESTED",
        "client_observed": True, "gpu_kernel_time_measured": False,
        "effective_backend_attested": False, "resource_ownership_attested_by_script": False,
        "token_identity_evidence": "tokenize_preflight_only; not engine ingestion",
        "input_sha256": source_sha, "tokenizer_manifest_sha256": tokenizer_sha,
        "workload": args.workload, "repetitions_executed": reps,
        "measured_per_repetition": limit,
        "server": server, "stats": stats,
        "per_repetition": [{"repetition": item["repetition"],
                            "wall_s": item["measured_wall_s"],
                            "summary": summarize([item], {**workload,"repetitions":1})}
                           for item in execution],
        "not_measured": ["attention_gpu_time", "whole_model_gpu_time", "allocator_bytes", "physical_hbm"],
    }
    (args.results_dir / "summary.json").write_bytes(canonical(summary))
    print(json.dumps({"status": summary["status"], "stats": stats}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
