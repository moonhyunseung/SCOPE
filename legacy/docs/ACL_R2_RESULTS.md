# ACL round2 additive results

Run `d599462d38ba9322`; preregistered repair1 source/inputs at main `5a69c6661330879a852da9bc4315ad925779adb3`. Original failed measurement run `478ce1357b85b881` and its CONTROL_FAIL remain preserved. Repair1 materialized integer codes before independent dequantization to resolve floating negative-zero raw-hash disagreement. The synthetic CPU fixture is bookkeeping evidence, not model-quality evidence. All numerical thresholds and workloads stayed fixed.

## Original representation screen

Terminal status: **REPRESENTATION_FIDELITY_FAIL**; baseline status **BASELINE_DIVERGENCE**. One screening prose source, Qwen1.5B/B1/L512, four teacher-forced steps, two exact replays. Steps/replays do not increase independent n.

|Comparison|Maximum observed relative logit L2|
|---|---:|
|baseline|0.0272492026|
|representation_cached|0.9169751984|
|representation_total|0.9207611394|
|execution|0.0000000000|

The .025 local logit screen failed for representation versus original full. Packed global/chunked versus the independent dense quantized reference were exact, including decoded cache hashes. This separates storage/execution fidelity from original-model numerical fidelity. Relative distances are not additive and are not task-accuracy loss rates. Natural continuation NLL, KL and top1 agreement are included per comparison; no corpus perplexity claim.

## First-divergence diagnosis

The same four-step source was replayed twice. Native trace references matched quality-block logits and caches exactly. Packed decode and append parity passed. SDPA default, explicit MATH SDPA, and the original Triton path all still failed .025 Gate D. No optimized kernel repair or performance benchmark was performed.

|Method|First same-operand Y difference: step/layer|Observed diagnostic boundary|
|---|---|---|
|static_sdpa|0/0|Y|
|static_sdpa_math|0/0|Y|
|int4_triton|0/22|Z|

Triton diagnostic-clone Y controls: 224/224 exact. Z/P observations are accepted only for exact clones. This does not establish compiler-internal instruction identity. Fused SDPA internal Z/P and internal cuBLAS algorithm IDs remain UNKNOWN. Input position-ID/cache-position equality flags compare signatures; UNKNOWN signature equality is not independent validation. Position-embedding and cache-append observations must be read separately.

## Operand-format / PV residual screen

Two existing screening source IDs; fixed layers0/14/27, heads0/6/11, steps0/7.36 repeated head/layer/step units,108 format rows, two exact replays and an unobserved native observer control. Native P, INT4 codes/scales and BF16-decoded V. CPU FP32 format emulation only.

|Operand representation|Max W relative L2|Max PV relative L2 against P*BF16(decoded V)|Max PV relative L2 against factorized FP32 W*C|
|---|---:|---:|---:|
|fp32|0|0.00158339247|0|
|tf32_rne_single|0.0003507116224|0.001525410758|0.0002971872679|
|tf32_rne_hi_plus_lo|8.024154903e-08|0.001583350123|2.559164107e-07|

Two-part TF32-format reconstruction greatly reduces the format residual against factorized FP32, while the separately retained BF16-decode rounding gap remains. No TF32 MMA throughput or full-model quality gain follows from this emulation. Norm floors, zero-channel handling and channel cancellation quantiles were frozen before execution.

## Evidence

- [lowering_diagnosis_v2 summary](../results/lowering_diagnosis_v2/d599462d38ba9322/blocks/first-divergence/summary.json) · [sealed manifest](../results/lowering_diagnosis_v2/d599462d38ba9322/blocks/first-divergence/manifest.json)
- [operand_screen_v2 summary](../results/operand_screen_v2/d599462d38ba9322/blocks/operand/summary.json) · [sealed manifest](../results/operand_screen_v2/d599462d38ba9322/blocks/operand/manifest.json)
- [quality_gate_v2 summary](../results/quality_gate_v2/d599462d38ba9322/blocks/quality/summary.json) · [sealed manifest](../results/quality_gate_v2/d599462d38ba9322/blocks/quality/manifest.json)

All three new block manifests were independently byte/SHA256 verified. The fixed800 campaign, original resident pilot, prior distribution screens and original .025 Gate D remain unchanged. No new source population, frontier, weights, token IDs or raw activation tensors are published.
