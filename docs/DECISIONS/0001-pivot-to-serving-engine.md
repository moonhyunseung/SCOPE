# ADR 0001 — Rebase SCOPE on serving cost, not a quantizer

- **Status:** Accepted research-direction decision; no performance claims.
- **Date:** 2026-10-19
- **Archive reference:** `7b3a1ca7b8c890b9da5b12a7ab5ad98193eb874f`.

## Context

The original SCOPE campaign produced reproducible K/V sensitivity, floating-point path differences, quantization negative results and selected H200 microbenchmarks. It did not demonstrate model-quality-constrained end-to-end throughput, latency or resident-KV capacity improvements. A fixed groupwise quantizer was never the desired product.

## Decision

Preserve *all* historical source/data/results as an immutable `legacy/` tree and frozen branch. The active repository instead targets a maintainable multi-model, multi-hardware attention execution strategy motivated by **measured** vLLM serving bottlenecks. The engine is not yet implemented. Quantization, state reduction and different attention semantics are possible interventions to be evaluated, not required architecture.

## Consequences

- New research starts by pinning and profiling current vLLM baselines, not an automatic INT4/INT8 sweep.
- A narrow conditional-dispatch win can be worthwhile; blanket chip/model speedups are unnecessary.
- Backend correctness, original-model behavior, and realistic serving costs are separate eligibility gates.
- A new quality contract may tolerate small numeric deviations if measured model behavior is protected; never relabel old failed .025 conditions.
- Upstream PR is a goal subject to code/benchmark validation and external reviewer acceptance.
- Existing commits, source SHA, experiments, and pre-pivot CI are preserved under `legacy/` rather than executed by new root CI.
