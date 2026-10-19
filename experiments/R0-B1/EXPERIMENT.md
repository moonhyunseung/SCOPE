# R0-B1 — vLLM streaming serving baseline client

**Implemented, CPU transport-tested; GPU benchmark NOT run.** This experiment does not authorize GPU use and does not claim that a new attention engine exists.

## Why this measurement

Measure client-observed TTFT, token-level ITL/TPOT when observable, end-to-end latency, achieved throughput, failures, and SLO goodput on **real vLLM `/v1/completions` streaming requests**. Reuse sealed R0-B0 synthetic W1 and W2 inputs, local pinned tokenizer and fixed contract. W1 (8192/128, C1) versus W2 (512/128, C16) is a *contrast*, not a controlled estimate of prompt-length or concurrency effects. Follow-up factorial tests are needed to isolate those variables.

## Version / workload identity

- SCOPE input contract: `configs/r0_b0_contract.json` (immutable for this experiment).
- Model and tokenizer: `Qwen/Qwen2.5-1.5B-Instruct`, revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.
- vLLM: `0.32.0`, source commit `5df63a7203cdb9328d824a27b682e77c7b9928be`.
- W1: 8192 input, up to 128 output, concurrency 1, 64 measured unique requests ×3 repetitions, 4 warmups/repetition.
- W2: 512 input, up to 128 output, concurrency 16, 128 measured unique requests ×3 repetitions, 16 warmups/repetition.
- Frozen generation: temperature 0, seed 20261019, stop on EOS or 128 output tokens, prefix caching **disabled**.
- Dispatch: saturated closed-loop client pool bounded by the workload concurrency; warmups drain before each measured repetition; repeated measured prompts are retained in all three repetitions; no retries/replacements.

## Runbook — CPU now / GPU only after explicit approval

Commands from the repository root. `SNAPSHOT` must be a **locally cached** path with the pinned revision. `PRIVATE` and `OUTPUT` paths must remain outside tracked directories (e.g. `/tmp/scope-r0b1-...`). The generator requires pinned vLLM source and `transformers` already installed, and does not download files.

~~~bash
# CPU-only verification, no GPU inference:
python -m unittest tests.test_r0_b1 -v
python -m unittest discover -s tests -v

# Example *local* private R0-B0 preparation; only if existing sealed files
# do not already exist (do not overwrite prepared inputs):
python -m scripts.r0_b0_prepare_requests --workload W1 --tokenizer-snapshot "$SNAPSHOT" --output "$PRIVATE/W1.jsonl"
python -m scripts.r0_b0_prepare_requests --workload W2 --tokenizer-snapshot "$SNAPSHOT" --output "$PRIVATE/W2.jsonl"
python -m scripts.r0_b1 --check --workload W1 --requests "$PRIVATE/W1.jsonl"
python -m scripts.r0_b1 --check --workload W2 --requests "$PRIVATE/W2.jsonl"
~~~

No installation, model download, GPU session acquisition, unrelated process termination, or port takeover is delegated here. **Do not run the remaining commands until the researcher has explicitly approved ownership of a dedicated H200, the server process, and an experiment budget.** Reuse an approved server if available. Confirm `vllm serve --help` on that exact installation and launch the pinned model from the local weights/tokenizer snapshot with these required effective settings:

~~~bash
# OPERATOR ONLY, AFTER EXPLICIT GPU APPROVAL AND VERIFIED EXCLUSIVE OWNERSHIP.
# Check actual CLI flags in pinned vLLM first. This command does NOT acquire a lock.
CUDA_VISIBLE_DEVICES="$APPROVED_GPU" vllm serve "$SNAPSHOT" \
  --served-model-name Qwen/Qwen2.5-1.5B-Instruct \
  --tokenizer "$SNAPSHOT" \
  --dtype bfloat16 --max-model-len 16384 \
  --gpu-memory-utilization 0.55 \
  --no-enable-prefix-caching --host 127.0.0.1 --port 8000
~~~

Preserve actual startup logs, `vllm --version`, package/source commit, worker/layer backend identity and effective flags **privately**. Inspect resource ownership and free HBM independently; this client neither reserves HBM nor enforces an exclusive GPUMON lock. Do not run if effective backend is unknown, model is unsupported, or the specified resource boundary cannot be maintained. Backend source-consistency checks in R0-B0 are *not* runtime attestation.

If approved, start a **two-request per repetition** pilot *before* the full pre-registered run:

~~~bash
python -m scripts.r0_b1 --run --gpu-approved --workload W1 \
  --requests "$PRIVATE/W1.jsonl" --tokenizer-snapshot "$SNAPSHOT" \
  --limit-measured 2 --repetitions 1 --results-dir "$OUTPUT/W1-pilot"
~~~

Then, after assessing the pilot (and only with the same approval and backend/resource evidence), run the full workloads separately:

~~~bash
python -m scripts.r0_b1 --run --gpu-approved --workload W1 \
  --requests "$PRIVATE/W1.jsonl" --tokenizer-snapshot "$SNAPSHOT" \
  --results-dir "$OUTPUT/W1-full"
python -m scripts.r0_b1 --run --gpu-approved --workload W2 \
  --requests "$PRIVATE/W2.jsonl" --tokenizer-snapshot "$SNAPSHOT" \
  --results-dir "$OUTPUT/W2-full"
~~~

## Measurement semantics / failure modes

`return_token_ids=true` (verified in pinned vLLM source) returns the input token IDs on the first response and token-ID **deltas** on subsequent streamed completions. The client compares server-returned prompt IDs with locally pinned token IDs. This is **API-path parity**, *not evidence that the worker ingested an identical token stream*; do not mark `inference_path_observed` without a separate engine-ingestion trace. Output token counts must also match final streaming usage.

TTFT is the time from client dispatch to first streamed token-ID delta; not GPU prefill time. ITL is computed from observed per-token stream timestamps, and TPOT is the average of the `N−1` intervals after the first token. When the server coalesces multiple tokens into a chunk, **ITL and TPOT are set to null** instead of fabricating per-token timing. Single-token EOS responses also have undefined TPOT. SLO classification is conservative: missing TPOT is *unmeasurable*, never a passing SLO. Usage is required, and failed/short/incomplete streams remain in results/denominators.

For each condition/repetition, measured wall time excludes warmups; achieved output-tokens/s, completed-requests/s, and goodput use measured wall time. Quantiles (p50/p95/p99) are descriptive client-observed samples, not confidence intervals. Outputs are `requests.jsonl` (all warmup and measured request outcomes) and `summary.json` (per-repetition and aggregate statistics). Neither includes prompt text nor token arrays. Keep even these files private until operational safety review; publication may use filtered summaries and hashes.

**Not provided by this first client:** kernel-level attention time, GPU-wide timing attribution, allocator/physical HBM residency, automatically attested effective worker backend, trusted resource watchdog, actual GPU session ownership, inference-ingestion token proof. Those require separate instrumentation. The implementation is a real serving-path client, but its uninstrumented measurements cannot by themselves establish the Attention-specific bottleneck or pass R0 evidence gates.
