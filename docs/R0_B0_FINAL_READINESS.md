# R0-B0 final preparation readiness contract

Status: preparation only. R0-B1 is `NOT_AUTHORIZED`.

This report records the final CPU-only corrections requested after review of
main commit `5d2950d369d04ae56b39a918fe61b4cf818db034`. It does not establish
runtime readiness, GPU ownership, actual backend execution, or model behavior.

## Tokenizer and request identity

The tokenizer manifest is limited to the pinned snapshot's required
`tokenizer.json` and `tokenizer_config.json`, plus an explicit allowlist of
optional tokenizer artifacts. Model weights and unrelated snapshot files are
not opened. Bytes are hashed in 1 MiB streaming chunks. Standard Hugging Face
snapshot links are accepted only when they resolve to regular files inside
that model repository's `blobs/` directory. Broken, external, unsafe, missing,
or non-content targets fail closed. Each manifest entry records whether it
was a regular snapshot file or an in-repository cache-blob symlink.

Measured request IDs, prompt/token hashes, order and lengths remain fixed.
Warm-up IDs use `WU-<workload>-R<repetition>-<index>` and must match the exact
per-repetition schedule. Local tokenization and a server `/tokenize` response
are preflight parity evidence only. Inference-path identity requires a
request-correlated token-ID hash and token count observed at engine ingestion;
the CPU validator cannot produce that runtime evidence.

## Backend and resource evidence

The source-pinned backend consistency check accepts FA2, FA3, or FA4 only as
distinct integer versions; source-supported FA3 is eligible on H200/SM90.
Expected worker IDs and the full expected attention-layer name set must match
the evidence records exactly. `BACKEND_EVIDENCE_CONSISTENT` remains a CPU
record-consistency result. Class identity, implementation, dynamic dispatch,
and hardware execution are separate evidence categories; no CPU fixture is
hardware attestation.

Resource records include a monotonic observation timestamp and expire after
five seconds by default. A stale sample or lost lock fails closed. The
validator checks a record and bounds; a trusted lock provider, live monitor,
watchdog, and child-process termination policy remain design prerequisites,
not implemented protections.

## Frozen proposed measurement contract

| Workload | Prompt / output cap | Concurrency | Measured requests | Repetitions | TTFT max | TPOT max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| W1 | 8192 / 128 | 1 | 64 | 3 | 8000 ms | 120 ms |
| W2 | 512 / 128 | 16 | 128 | 3 | 3000 ms | 120 ms |

Temperature is 0; stop at model EOS or the 128-token output cap and record
actual output length; prefix caching is disabled. The proposal uses saturated
closed-loop dispatch bounded by concurrency and reports observed offered and
completed rates instead of inventing a fixed open-loop arrival rate.

Goodput is the count of successful measured unique requests meeting both
TTFT and TPOT thresholds divided by measured wall-clock seconds. Every planned
measured request remains in the denominator; success, error, timeout,
cancellation, and SLO-miss counts are reported separately, with no automatic
replacement. Warm-ups are excluded from measured goodput and its denominator.

Manifests use UTF-8 JSON, lexicographically sorted object keys, compact
comma/colon separators, `ensure_ascii=false`, and exactly one trailing LF;
SHA256 covers those exact bytes. The result schema retains request IDs,
repetition, status, observed output length, token identity evidence stage,
timestamps, latency metrics, and backend/resource evidence references.

## Remaining blockers

- Tokenizer files and request manifest hashes are still pending; no tokenizer
  was loaded and no real request prompts were generated.
- Server inference-path token identity, effective backend, complete worker and
  layer enumeration, GPU lock owner, live monitor, and isolated vLLM runtime
  remain unknown or unverified.
- SLOs and dispatch policy remain proposed until scientific approval.
- `validate_r0b1_readiness()` always returns `R0B1_NOT_AUTHORIZED`.

No package installation/download, server startup, model inference, GPU
benchmark, or profiling is part of this preparation package.
