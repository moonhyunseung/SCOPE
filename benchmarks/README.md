# Baseline workload templates — not results

No new serving benchmarks have been run. `workload_matrix.example.json` provides *illustrative* workload axes and empty provenance placeholders. The benchmark runner, model selection, and hardware allocation require a separate approved experiment contract.

For each actual run freeze model/tokenizer revision, vLLM commit and backend actually selected, hardware/software identity, source hashes, prompt/generation length distributions, batch/concurrency/prefix policy, seeds, run ordering, SLO and resource budget. Use a production-serving endpoint plus GPU profiler, not only isolated `attention()` timing.

Results should record TTFT, ITL/TPOT tails and throughput/goodput, attention fraction of end-to-end cost, allocation/padding/physical HBM limitations, and correctness/behavioral endpoints. Unsupported or failing cases remain visible. Consult `../docs/BENCHMARK_CONTRACT.md`.
