# Production-relevant benchmark and evidence contract (proposal)

**Status: no measurements in this new track.** This file specifies what must be frozen *before* any benchmark. A sample workload list is under `benchmarks/workload_matrix.example.json`, not actual results.

## 0 — Establish baseline identity

- vLLM commit/version, Python/PyTorch/CUDA/compiler, driver, actual GPU SKU, power/clocks, model/tokenizer revision, KV dtype, attention backend actually selected, scheduler/paged allocator configuration.
- Specify prompt-source and token-window hashes, sampling/teacher forcing, requested output lengths, seed, prefix-cache hit/miss, chunked prefill, CUDA Graph, eager mode, batching and concurrent request policy.
- Verify functioning supported production baselines first. Compare the best compatible baseline (for the exact workload) and a diagnostic reference when needed; explain unsupported modes instead of using an artificially weak baseline.
- Capture service/API overhead separately where possible. Keep tooling/profiler overhead out of timing evidence.

## 1 — Workload structure

Freeze at least low-concurrency/long-context, high-concurrency/short-context, realistic mixed traffic, plus prefill-sensitive and decode-sensitive scenarios. Do not infer serving efficiency from a four-token B1 microbenchmark. Record independent requests/source families and per-request prompt/output tokens. Do not overcount correlated repetitions.

## 2 — Performance and physical costs

- TTFT, ITL/TPOT distribution (median, P95, P99 with sample counts), end-to-end request latency, output tokens/s, requests/s, stable throughput at a fixed SLO, goodput and GPU-hour per useful output token.
- Decode attention GPU time vs whole-model GPU time; prefill, projection, append, cache transforms, scheduler/launch, synchronization and LM head attribution where measurable.
- Persistent actual KV bytes, page/block padding, allocator active/allocated/reserved, peak transient workspace, *measured or sampled* physical HBM with caveats, weights/runtime overhead and fragmentation. Theoretical payload is reported separately.
- Repetition order, warmup, clocks, outliers and measurement uncertainty. Matched interleaved A/B runs mitigate drifting baselines; do not silently drop slow runs.

## 3 — Correctness and quality

Keep three distinct axes:
1. **Execution fidelity:** candidate vs same-representation reference for scores/softmax/output/append as appropriate.
2. **Representation/model fidelity:** when state is lossy, candidate vs original full-model and cached baselines; report original cached/full mismatch separately.
3. **Behavior/quality:** preregistered prompt-completion tasks, NLL/perplexity only when measured over an appropriate corpus, free-running generated trajectories, answer utility and tail regressions.

A historical SCOPE relative-logit-L2 <= 0.025 was a fixed **legacy numerical experiment gate**. Do not copy it as the universal acceptance criterion for a new approximate engine. Freeze new numerical and behavior tolerances, workloads and acceptance thresholds *before* seeing new candidate results. Do not equate different generated text with automatic user-visible quality regression, or matching top-1 with certified quality.

## 4 — Comparison/decision

Only compare feasible points: same quality contract, same workload, same hardware/settings and complete cost accounting. An implementation failing a numerical or behavior gate cannot be a performance winner. Track regressions and useful subregions; a targeted conditional-dispatch improvement can matter even without a universally faster kernel.

The experiment output must contain: frozen manifest, invocation/configs (without secrets), measurement table with denominators and failures, hardware/software metadata, raw timing samples or permitted non-sensitive summaries, uncertainty assessment, SHA256s and concise stop/go recommendation.

## 5 — Safety and cost

Dedicated resource consent, allocator/HBM/disk reserve, own-worker lock, runtime cap, and no interruption of unrelated serving/processes. The review conversation chooses next scientific work only after completed results are sealed. Do not run a GPU campaign merely because this document exists.
