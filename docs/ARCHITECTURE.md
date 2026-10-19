# Planned system architecture (design only)

Nothing in this file describes a working new kernel or an integrated vLLM backend.

```text
Actual served workload (models, tokens, concurrency, KV policy, SLO)
                 |
          Capability / cost profiler
                 |
    One falsifiable bottleneck hypothesis
                 |
  +--------------+----------------+
  |                               |
Dense/model-compatible          Optional new state
execution candidate             representation candidate
  |                               |
Full semantic + numeric          Independent model-quality +
checks                           multi-step trajectory gate
  +--------------+----------------+
                 |
            Execution plan
   [chip backend + memory layout + fallback]
                 |
    Real vLLM integration (later, gated)
                 |
  Matched quality / latency / HBM / throughput
```

## Conceptual interfaces

- **Model capabilities:** query and KV head counts, head dimension, positional encoding, causal/mask semantics, dtype, layer layout, sliding window/MLA distinctions.
- **Request/workload:** active sequence lengths, prompt/decode sizes, concurrency, prefix cache hit/miss, batching and scheduler configuration.
- **State:** persistent representation plus append/update and a reader. Snapshot attention equality does not certify future decode equivalence.
- **Executor:** available chip/driver/dtype/layout capabilities, attention operations and declared numerical reference.
- **Dispatcher:** chooses a verified supported path or safely falls back; never selects an unsupported "optimized" path.
- **Instrumentation:** kernel timing, bandwidth/IO evidence, allocator/residency/padding, runtime and serving-level metrics.

## Architecture coverage

Target test axes include MHA/GQA/MQA, BF16/FP16, causal decode and prefill, dynamic/paged KV and physically separate KV storage. These are a **planned coverage matrix**, not supported implementations. MLA, sliding-window, speculative decoding and cross-layer sharing require explicit semantics; never pretend they are handled by a generic tensor shape.

Chip targets are H200 (Hopper) for first benchmark and RTX PRO 6000 Blackwell for transfer. V100/other GPUs are compatibility/fallback targets only when their supported dtypes and serving stack are verified. No fabricated portability.

## Quantization is optional

Do not force one format: dense, native layouts, recoded representations, tile-local reconstruction, lossy compression and changed computational schedule are competing hypotheses. Count encode, append, paging, dequant, scratch, prefill amortization and scheduling when relevant. Hardware lowering follows eligibility, not the reverse.

See `BENCHMARK_CONTRACT.md` for measurements and `RESEARCH_ROADMAP.md` for stop/go gates.
