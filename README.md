# SCOPE — Attention Systems Engineering

**Purpose:** Discover, explain, and remove measurable inefficiencies in real LLM attention serving. Target an adoptable, cross-model and hardware-aware attention execution engine, with a credible path to an upstream vLLM contribution.

**Current status: research pivot / baseline profiling not yet run.** There is no new engine, demonstrated production speedup, or measured serving-capacity improvement in this new track.

SCOPE is **not** an INT4 quantization project. Our central question is whether current attention state, representation, data movement, and execution plans are necessary and efficient for real workloads. Approximation and new state representations are candidates, *not* prerequisites. Start with real serving bottlenecks and an explicit model-quality contract; let evidence choose the transformation.

## Workstream

1. **Observe production-relevant cost:** Pin vLLM + actual model/backend versions. Profile prefill, decode, KV movement and allocation, launch/scheduling effects, batching and serving SLOs on H200. Reproduce with compatible Blackwell hardware next.
2. **Identify an avoidable cost:** Distinguish attention compute, memory traffic, KV residency, runtime overhead and quality constraints. Propose **one falsifiable** end-to-end efficiency hypothesis at a time.
3. **Build the smallest compatible implementation:** A semantics-preserving backend or a separately gated approximate representation. Avoid assuming that quantization, a custom kernel, or a new attention formula must win.
4. **Measure end-to-end:** Against strong, version-pinned vLLM backends, compare whole-request performance, model behavior, peak physical/allocator memory, and cost per useful token.
5. **Earn upstream readiness:** Model/head layout compatibility, paged KV semantics, decoding correctness, graph/scheduler interaction, fallbacks, deterministic regression tests and maintenance documentation.

## Repository layout

| Path | Role |
| --- | --- |
| [docs/VISION.md](docs/VISION.md) | Problem, ambition, scope and explicit non-goals |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Planned engine boundaries, compatibility and capability-driven dispatch |
| [docs/BENCHMARK_CONTRACT.md](docs/BENCHMARK_CONTRACT.md) | Version-locked serving cost, quality, fairness and acceptance methodology |
| [docs/RESEARCH_ROADMAP.md](docs/RESEARCH_ROADMAP.md) | Evidence gates and stop/go decisions |
| [docs/LEGACY_EVIDENCE.md](docs/LEGACY_EVIDENCE.md) | Original SCOPE evidence, failed candidates and preserved provenance |
| [benchmarks/README.md](benchmarks/README.md) | Workload matrix template — **not benchmark results** |
| [scope_engine/](scope_engine/) | Stdlib-only workload/shape contract scaffolding — **not an attention backend** |
| [legacy/](legacy/) | Exact pre-pivot root tree: prior source, docs, tests, results and workflows |
| [AGENTS.md](AGENTS.md) | Instructions for Codex and other coding agents |

**Archive boundary:** All pre-pivot files remain at [legacy/](legacy/) and at the immutable commit [`7b3a1ca`](https://github.com/moonhyunseung/SCOPE/commit/7b3a1ca7b8c890b9da5b12a7ab5ad98193eb874f). An additional preservation branch is [`legacy/scope-v4-2026-10-19`](https://github.com/moonhyunseung/SCOPE/tree/legacy/scope-v4-2026-10-19). Nothing in the archive is silently retested, improved or relabelled by this restructure.

## Immediate next step

**Do not begin another quantization sweep or GPU frontier.** First produce a version-pinned, reproducible vLLM baseline cost profile. See [benchmark contract](docs/BENCHMARK_CONTRACT.md). Run new experiments only after explicit scope/budget approval. CPU scaffold validation: `python -m unittest discover -s tests -v`.

## Attribution and evidence

Historical results and previously authored research are archived, not attributed to this restructuring. New prose and scaffolding here constitute a planning pivot, **not measured findings**. Positive and negative results remain distinct; never infer a GPU speedup from smaller tensors or an execution-reference pass from model-level fidelity.
