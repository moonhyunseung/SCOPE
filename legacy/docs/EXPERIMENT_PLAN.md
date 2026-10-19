# Reproducible execution plan

## P0 — Correct reference and observable state

- Freeze a single-token decode attention definition: GQA grouping, dtype, score scaling, mask/position handling, cache inclusion, and softmax precision.
- Implement an instrumented *decoded-quantized* PyTorch reference with code/scale metadata.
- Run synthetic smoke and unit tests first. Compare reference output to PyTorch SDPA on the same decoded K,V.
- Build a read-only capture adapter for real decode Q,K,V and per-layer/head context. Confirm its selected tensors match the original model under the declared tolerances; fused attention implementations may require a separate diagnostic reference pass.
- Save a small, bounded trace and provenance manifest. No timing claim from the traced path.

**P0 acceptance:** trace has correct axes; GQA/head mapping verified; finite/nonfinite reporting; deterministic tests; numerical reference check; synthetic/real provenance labels correct.

## P1 — Distributions and relationships

- Stratify by model layer, query/KV head, decode position, context length, prompt class, and workload.
- Analyze signed magnitudes, exponent ranges, softmax concentration, top-gap, code saturation, scale dispersion, `W=P*S_V`, and output cancellation.
- Report confidence intervals or uncertainty from finite samples, not absolute generalizations. Preserve rare outlier blocks.
- Pre-register transformations only **after** a descriptive exploration set; evaluate on held-out traces.

**P1 acceptance:** reproducible statistics and stratified exports; joint-distribution views and tail coverage; no kernel speed claims.

## P2 — Candidate transformations

For each candidate, document `observation → conjecture → semantic proof obligation → numerical gate → failure conditions`.

Examples: offset-to-signed recoding; token-major vs group-major; direct nibble operand formation; FP32 coefficient decompositions; FP64/TF32 paths; per-tile expansion; bounded-split streaming reductions; optional global materialization. No candidate is assumed superior.

**P2 acceptance:** a reference-equivalent implementation or quantified approximation, with error profile on *held-out* observations.

## P3 — H200 realization

Measure fixed decode workloads with profiling disabled:
- kernel and end-to-end decode latency, throughput, p50/p95
- persistent KV allocation, allocator metadata/padding, peak transient HBM
- HBM bytes, register pressure, occupancy, tensor-core utilization, launch/synchronization cost as available
- precision/error contract against the same decoded-quantized reference and full-model quality comparison

**P3 acceptance:** improvement on a fixed resident-KV vs decode-efficiency Pareto frontier with comparable workloads and confidence intervals. Results are not reported as validated before measurement.

## Experiment artifact contract

Each run stores config, seed, code commit, environment manifest, reference checks, observed metrics, and a concise conclusion. Failed hypotheses remain recorded; never silently retune the reference or dataset to save a hypothesis.
