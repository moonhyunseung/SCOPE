# SCOPE agent contract — new attention-engine track

The root is a **new research-planning track**. Prior data/source under `legacy/` is **immutable evidence**, not the active engine implementation.

## Scientific objective
Find an avoidable cost in real attention serving, construct a model-compatible or explicitly quality-gated execution alternative, and demonstrate end-to-end performance/quality value suitable for a potential vLLM upstream contribution.

## Required practice
1. Read `docs/VISION.md`, `docs/BENCHMARK_CONTRACT.md`, `docs/RESEARCH_ROADMAP.md` and `docs/LEGACY_EVIDENCE.md` before modifying code.
2. Observe and explain the bottleneck before proposing quantization, state reduction, a new attention formula or a new kernel.
3. New hypotheses, model/checkpoint/tokenizer revisions, workloads, metrics, budgets and stop conditions must be frozen **before** looking at new outcomes.
4. Compare a strong version-locked vLLM production baseline; document which backend actually ran. Whole-model and serving numbers never inherit conclusions from microbenchmarks.
5. Maintain separate reference contracts: execution correctness, original-model behavior, recurrent decoding behavior, whole-system cost.
6. When testing approximate computation, do **not** require bitwise equality or automatically reuse historical 0.025 logit L2 as the sole quality objective. Freeze a new task/trajectory quality contract *before* execution; leave all historical thresholds unchanged.
7. Keep all failed, unsupported, missing and non-comparable conditions with denominators. Repeated steps, heads, runs and models sharing a document are not independent source units.
8. Preserve `legacy/` and the archive commit; never retroactively update its metrics, hashes or experiment definitions. Publish additive, audited changes only.
9. No GPU execution, new model downloads, external process termination, changes to credentials, or system-wide services without explicit approval. Respect user-owned H200/Blackwell resource and lock policies when authorized.
10. No model weights, raw activation tensors, exact sensitive prompts/tokens, credentials or operational private logs in public commits; no LLM inside tensor loops.
11. Make one small, falsifiable engineering decision at a time. Stop on failed correctness/quality gates; do not promote failed kernels as performance winners.
12. Codex executes and verifies approved blocks; scientific interpretation and approval of new directions remain with the user and the main SCOPE review conversation.

A code scaffold, a synthetic test and a planning benchmark matrix are not research findings.
