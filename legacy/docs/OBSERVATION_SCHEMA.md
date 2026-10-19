# Observation schema: attention state and provenance

## Required tensor semantics

For one decode step, specify shapes explicitly:
- `Q`: [B, Hq, 1, D], after the model-specific positional transform.
- `K,V`: [B, Hkv, L, D], including previous tokens and the current token according to the reference contract.
- `Z`: [B, Hq, 1, L], **scaled and masked** attention scores.
- `P`: [B, Hq, 1, L], the reference softmax probabilities.
- `Y`: [B, Hq, 1, D], attention output before output projection.
- If **groupwise quantized V** is active, `S_V`: [B, Hkv, L, G]; define `W[b,h,t,g] = P[b,h,0,t] S_V[b, kv_head(h), t, g]`. `W` is diagnostic, not a universal native transformer tensor.
- Quantized `C_K,C_V` and scales must document code alphabet, group size, signed/offset convention, rounding, clipping, zero-scale policy, and bit interpretation.

Do not mix token/head/group axes. GQA must specify `Hq % Hkv == 0` and `kv_head(h) = floor(h / (Hq/Hkv))` for the assumed contiguous grouping; verify model-specific mapping.

## Minimum provenance

A result must record: `schema_version`, `run_id`, `input_kind` (synthetic or real-model), seed, model ID/revision, tokenizer/revision, attention implementation, prefill/decode step, L, B, layer/head, dtype, RoPE/mask/GQA contract, quantization contract, hardware name, GPU/driver/CUDA/Torch versions, Git SHA, prompt/dataset identifier, trace sampling rule and known limitations.

Use `null` for unavailable values; **never invent metadata**. Only publish permitted, de-identified samples. Raw prompts/weights/traces may require access controls.

## Statistics

Per tensor (conditional on layer/head/context length and input distribution):
- finite and nonfinite counts, signed min/max, mean/std, RMS
- abs quantiles and exponent/significand range when useful
- zeros/near-zeros, saturation/clipping of quantized codes
- P entropy, top-k mass, score top-gap, and tail/extreme-case counts
- scale/code coupling; Q–K contraction behavior; P–V and W=PS_V coupling
- accumulation/cancellation sensitivity, output error by precision choice

Retain bounded samples of joint tuples (e.g. `P,S_V,W,C_V`) and short *block* slices as needed; independent marginal histograms cannot establish joint structure. Document any sampling approximation and its bias.

## Measurement separation

1. Reference trace computes and observes full `Z,P` deliberately. This is acceptable **for instrumentation only**.
2. Hardware performance is measured without trace hooks, with synchronizations and allocator/peak HBM measured consistently.
3. Validate output against the exact reference under an explicit error tolerance.
4. Synthetic smoke tests validate shapes, semantics and sampling plumbing; their distributions are not findings about actual model data.

## P0 smoke output

`scripts/p0_smoke.py` writes a JSON report marked `input_kind="synthetic"` with shapes, finite-valued summary statistics, code frequencies, P entropy, and a consistency check against PyTorch SDPA. It is not a vLLM trace or a production benchmark.
