# SCOPE R0-B0: Baseline preparation (not executed)

Status: `PREPARATION_ONLY_NOT_EXECUTED`. This package freezes the proposed
request-generation and safety contract. It does not demonstrate a working
vLLM environment, verify a runtime attention backend, or authorize R0-B1.

## Pinned inputs and workloads

- vLLM source: v0.32.0, commit
  `5df63a7203cdb9328d824a27b682e77c7b9928be`.
- Synthetic model/tokenizer candidate: `Qwen/Qwen2.5-1.5B-Instruct`, revision
  `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.
- W1: requested prompt 8,192 tokens, output 128, concurrency 1, 64 unique
  requests, 3 repetitions, 4 warm-up requests per repetition.
- W2: requested prompt 512 tokens, output 128, concurrency 16, 128 unique
  requests, 3 repetitions, 16 warm-up requests per repetition.
- Synthetic performance-request seed: `20261019`; prefix caching disabled;
  deterministic sampling (`temperature=0`); EOS ignored only when enforcing
  the performance sample's output length.
- The synthetic pool is not a natural-language workload. A separate set of at
  least 16 fixed natural-language diagnostic requests must be prepared, hashed,
  and reviewed before any execution. It must not be mixed into performance
  claims.

## Request generation and identity

`scripts/r0_b0_prepare_requests.py` calls the pinned vLLM `RandomDataset`
implementation, and only loads a tokenizer from an explicitly supplied local
snapshot (`local_files_only=True`). It checks the installed vLLM version and
Git-blob hashes of the two source files that define sampling. It never invokes
the model, downloads files, or uses a GPU. It passes requested prompt length
plus the tokenizer's special-token count because vLLM subtracts that count
before sampling; it then independently checks the re-encoded prompt length.
A mismatch is a hard preparation failure and no replacement sample is chosen.

Generated prompt text is written only to the caller-specified local JSONL file.
The public contract contains no generated prompt text, token IDs, or model
weights. The private local JSONL, ordered request-source SHA-256, ordered
token-ID SHA-256, per-request lengths, tokenizer revision, and generator
provenance remain `PENDING` until a permitted local-only preparation actually
runs. Do not commit those generated inputs. The generator refuses a tokenizer
directory whose path does not identify the pinned snapshot revision.

## Backend verification contract

No backend is verified in R0-B0. For a later separately approved run, preserve
both (1) the startup backend-selection record and (2) effective worker-layer
attention backend class names from every worker. Require a non-empty, unique
effective class set consistent with the startup record. Configuration alone,
available-kernel lists, or a package import do not establish the selected
backend. Missing, ambiguous, or inconsistent evidence is
`BACKEND_UNVERIFIED`; do not measure or report performance as a backend result.
The CPU validator implements this fail-closed comparison for supplied
evidence records, but does not inspect or start a server.

## Future-run safety preflight (checks only; no reservation is made)

Before any separately approved R0-B1 run, require a verified exclusive
single-H200 reservation/lock, at least 44 GiB physical HBM free, at least
24 GiB disk free, allocator ceiling no more than 96 GiB, projected workload
physical HBM increase no more than 88 GiB, and a 1,800-second wall-clock
deadline. `gpu_memory_utilization=0.55` is only a proposed vLLM setting, not a
physical cap. Never terminate other processes or change services/containers.
Stop without retry on unknown ownership, backend mismatch, identity mismatch,
resource-limit breach, OOM, worker failure, timeout, or missing/error requests.
Preserve failures and denominators. An SLO miss alone is an observation, not a
stop condition.

Diagnostic SLO candidates, not external service guarantees: W1 request TTFT
<= 8,000 ms and TPOT <= 120 ms; W2 TTFT <= 3,000 ms and TPOT <= 120 ms.
Planned observations include per-request TTFT/ITL/TPOT/E2E, output tokens/s,
requests/s, goodput, prefill/decode/attention/whole-model time, persistent KV
and padding, allocator active/allocated/reserved bytes, workspace peak, and
sampled physical HBM with measurement caveats. W1's 64 unique requests do not
support a reliable tail-P99 claim; repeated token steps are not independent
requests.

## CPU validation and blockers

`python -m unittest discover -s tests -p 'test_r0_b0.py'` tests manifest
consistency, canonical hashing, fail-closed backend evidence, and resource
preflight logic using synthetic fixtures only. It does not import vLLM,
Transformers, PyTorch, or CUDA and does not create benchmark prompts.

Known blockers carried forward from the R0-A audit: system Python had no
PyTorch/vLLM; the older research environment had PyTorch but no vLLM; the
server-wide GPU lock/ownership state is unknown; NVIDIA-SMI and PyTorch memory
totals differed by 644,939,776 bytes; actual tokenizer cache identity and
vLLM compatibility have not been revalidated in the target isolated runtime;
and no effective attention backend can yet be reported. The attached
code-server checkout is not current main and has unrelated pending local
changes, so this preparation does not use or alter that checkout.

R0-B1 (installation, model loading, serving, profiling, GPU inference, or
benchmarking) remains outside this package and requires a separate explicit
approval after scientific review.

