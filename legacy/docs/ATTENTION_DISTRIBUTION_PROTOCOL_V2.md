# Native attention distribution protocol v2

Frozen before collecting these sketches. Two screening sources, prose and code, five locally audited pinned models, BF16, L512/L1536, decode steps0/7, first/middle/last layers, all query heads. These are repeated measures on two source units, and descriptive engineering screens. No population-wide uncertainty or independently multiplied head/sample n claim.

Observe the installed native eager attention unchanged. Capture actual Q/K/V inputs, actual input to its softmax (Z), returned native probabilities P, and returned attention output Y. Rerun the same cache snapshot and require exact complete logits and cache equality. Never silently replace native logits or label reconstructed scores as Z. Quantization runs after the native call and cannot alter its output. Assert INT4 decode equals the frozen quantizer reconstruction.

Publish marginal quantiles, zero/sign fractions, frexp exponent counts and eight abs-mantissa bins on deterministic bounded coordinate samples (up to8192 values each). Counts identify the sample and population size. Quantized signed C and FP32 S are explicitly reconstructed representations. Keep P aligned by token with each V scale group; W=P*S is reconstructed. Save paired log2(P),log2(S) bins, W/S exponent spans, code zero/saturation, top32 native probability mass, and reconstructed sum(P*S*C) cancellation. No raw activation tensors are public. Small JSONL shards bound each publication file.

Predeclared descriptive falsification checks, evaluated per source/model/length before cross-head summaries:

1. A narrow shared-exponent opportunity requires 90% W exponent span <=8 for at least90% of head/layer/step observations in **each** source/model/length cell. A cell that fails rejects this strong screen; it does not prove another arithmetic scheme impossible. W's span is also compared with S; a large scale range cannot be inferred from P alone.
2. A strong concentration screen requires top32 native P mass >=0.9 for at least80% of observations in each cell. Failure rejects this screen. Success does not authorize token pruning or a quality claim.
3. Severe cancellation is defined as reconstructed L1(sum_t P*S*C)/sum_t L1(P*S*C) <0.1. Its frequency is recorded in every cell. Any proposed reduced accumulator must later pass independent end-to-end Gate D; histograms alone cannot validate precision.

These thresholds are engineering screens, not optimized precision settings or claims of achievable speed. Follow-up kernels require hardware feasibility, full execution-path error gating, and measured full-model timing. Cached/full native discrepancies remain a separate diagnosis. Remaining topic/project lineage dependence, bounded licensed shards, deterministic subsampling and limited source coverage are explicit limitations.
