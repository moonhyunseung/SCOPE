# Hypothesis registry — candidates, not results

| ID | Observation to verify | Proposed transformation | Falsification criterion |
|---|---|---|---|
| H-K1 | Signed/offset recoding preserves decoded codes | Exact recode before contraction | Any decoded-value mismatch or loss of end-to-end fidelity |
| H-K2 | K accesses favor a grouped stride pattern at long L | Group-major physical packed K | No reproducible decode-level improvement after layout/conversion costs |
| H-V1 | `W=P*S_V` has exploitable precision structure | One FP64 MMA or split TF32 operands | Error gate fails, or extra conversion exceeds native contraction gains |
| H-V2 | Token-axis reduction workspaces limit residency | Bounded-split streaming reduction | Peak HBM or latency worsens on matched workload |
| H-ATT1 | Joint Q/K/P/V statistics permit lower-cost online attention | Packed input with tile-local expansion and fused online reduction | Output/next-state mismatch or failure to improve frontier |
| H-MEM1 | Hopper-specific movement changes conversion tradeoffs | TMA/shared-memory staged packed data | Memory transaction, occupancy, or end-to-end cost fails to improve |

Every hypothesis needs **measured baseline, scope, input distribution, quantization model, precision tolerance, and an independent held-out evaluation** before promotion to a result.

## Required record per hypothesis

- Statement, origin and evidence source (real-model trace or synthetic only).
- Mathematical identity or explicit approximation, including state-update semantics.
- Program variants and memory hierarchy transitions; no intermediate representation is excluded by fiat.
- Expected wins and reasons it might fail on H200.
- Negative-control baseline; pre-registered workloads and acceptance metrics.
- Results, uncertainty, commit SHA, and postmortem.

## Relationship to OFD

OFD can consume a **validated** SCOPE hypothesis, but past OFD timings are not automatically SCOPE results. Cross-repository claims require the exact benchmark contract and reproducible evidence.
