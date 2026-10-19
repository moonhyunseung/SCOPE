# Vision: SCOPE as an attention systems research program

## The actual problem

Modern attention serving may expend avoidable work retaining, moving or reading state, performing reductions, formatting operands, or coordinating heterogeneous requests. **Do not assume in advance which source of cost dominates**. Determine where deployed stacks spend GPU time/HBM/bandwidth and whether the current representation and execution plan is necessary under model-behavior constraints.

**End goal:** a model- and hardware-aware attention engine that delivers reproducible useful improvements on production-relevant inference and is maintainable as an upstream vLLM backend/optimization path.

The engine is a research target; it does **not yet exist**.

## Questions, in order

1. In pinned vLLM serving experiments, what dominates TTFT/ITL/TPOT, throughput, actual allocation and cost per output token? How do those costs change with prompt length, generation length, batch, concurrency, KV head layout and architecture?
2. Which avoidable cost is due to redundant persistent state vs execution layout, movement, scheduling, extra work or numerical overprecision?
3. Is there a more efficient execution/dataflow while retaining the original model? If not, is a separately validated approximate state representation worth its quality and maintenance cost?
4. Is the improvement sustained under real scheduling, paging, graph capture, cache append, realistic prompts and multi-model/hardware tests?

## Two distinct tracks

**Drop-in compatibility (first):** Existing model weights and attention semantics, no training, correct KV append/positions/masks/GQA, explicit finite precision tolerance, capability-based fallbacks.

**State/attention redesign (later, independently gated):** Represent information differently, remove redundant states, use approximate attention or trained mechanisms only with prespecified quality/trajectory validation. This is not automatically a drop-in vLLM backend.

## Success definition

A change is useful only if it is eligible on numerical/behavioral quality **and** demonstrates an end-to-end benefit under comparable, strong production baselines. Report trade-offs, not only best-case percent speedups. Different hardware can use different kernels under one stable interface.

## Explicit non-goals right now

- Claiming that KV cache is provably redundant.
- Treating groupwise INT4/INT8 as the default representation.
- Treating small operator residual or lower byte count as serving improvement.
- Retrofitting the old fixed .025 gate to make rejected candidates pass.
- Writing a general new attention mechanism before establishing the real serving bottleneck.
- Claiming vLLM acceptance or chip portability without actual evidence.
