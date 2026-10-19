# Roadmap — evidence first, architecture second

Each stage is conditional. A completed stage is not implied by creating its folder or this document.

| Gate | Question | Required evidence | Failure/stop |
| --- | --- | --- | --- |
| R0 Profile | Where is real serving time/HBM spent? | Pinned vLLM baselines with workload + model + backend identity; service and GPU accounting | Stop if incompatible baselines, unrepeatable timing or unknown backend |
| R1 Hypothesis | Is the cost avoidable? | One explained cost center and predicted measurable win/cost tradeoff | Stop if projected gain disappears after full cost accounting |
| R2 Correctness | Can one small intervention preserve semantics/quality? | Independent execution reference, KV append and multi-step behavior; approximate path has preregistered quality | Stop or quarantine failing methods |
| R3 Systems | Does it win in realistic serving? | Uninstrumented matched A/B, TTFT/ITL/throughput/HBM/goodput; negative cases kept | Stop if non-feasible or no meaningful cost win |
| R4 Portability | Is it useful beyond a single setup? | At least another model/head layout and GPU generation, safe fallback | Call it chip-specific if not general |
| R5 Upstream | Is it maintainable? | vLLM-supported contract, tests, layout/paging/scheduler/graph checks, documentation, reviewer feedback | Never imply integration before acceptance |

## Initial small step (requires explicit approval)

Create **R0 measurement contract and one small vLLM baseline run** on an authorized single GPU: one pinned model, one fixed prompt set and two contrasting workload shapes. First verify which attention backend vLLM actually uses. Only then profile GPU and serving costs; propose **no new kernel before R0 evidence**.

Do not import 800-condition legacy tests as active benchmark dependencies. Their results are preserved for reuse as hypotheses and warnings, not proofs of modern vLLM speed.

## Research governance

Every scientific campaign pre-registers inputs/thresholds/control logic. Code agents do deterministic execution, verification and publication; the user's analysis discussion determines the next scientific direction. Published negative results are immutable. Limit GPU budgets and never silently broaden a grid.
