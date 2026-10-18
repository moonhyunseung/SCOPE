# Completed continuation campaign: evidence index

The frozen 800-condition campaign ended with 798 COLLECTED and 2 UNSUPPORTED; no runtime-failed conditions. Screening: 238 collected/2 unsupported. Confirmation: all560 collected, five models on the same112 source units (56/domain). Models and steps are repeated measures, not additional independent sources.

## Confirmatory source audit

The additive CPU audit independently reproduced all8 frozen Pythia/OPT primary estimates and intervals from sealed step files. Dense/full fidelity failures:338/560 confirmation conditions, retained without primary filtering. Qwen1.5B55 PASS/57 FAIL; Pythia110/2; OPT0/112; Qwen7B0/112; Qwen14B57/55.

Endpoint: K-only minus V-only mean relative logit L2 against native cached logits over8 correlated decode steps. Positive estimates indicate larger K perturbation under this intervention. They do not establish original full-reference quality.

|Model|Domain|Regime|n|Mean K−V|99.375% source bootstrap CI|
|---|---|---|---:|---:|---|
|pythia1b|prose|isolated|56|0.03322932|[0.02857416, 0.03924119]|
|pythia1b|prose|recurrent|56|0.03604326|[0.03165061, 0.04101257]|
|pythia1b|code|isolated|56|0.04352955|[0.03713438, 0.05194240]|
|pythia1b|code|recurrent|56|0.04967538|[0.04269175, 0.05821383]|
|opt1_3b|prose|isolated|56|0.11527201|[0.10450258, 0.12716474]|
|opt1_3b|prose|recurrent|56|0.11751234|[0.10649920, 0.12965591]|
|opt1_3b|code|isolated|56|0.21073237|[0.18767168, 0.23495297]|
|opt1_3b|code|recurrent|56|0.21411398|[0.19116458, 0.24032872]|

All intervals use20,000 source draws. Cross-model exploratory contrasts resample shared source IDs jointly. Source maxima and exact observed threshold counts are included; no Q99 or rare-population precision claim.

## Resident pilot

One Qwen1.5B prose prefix, B1/L512, four teacher-forced decode steps. Gate D checks same-representation execution; original-model quantization quality is unmeasured. Passed points use two warmups and ten timing repetitions; this is one homogeneous workload. No serving frontier or maximum capacity is established.

|Method|Terminal status|Max step relative L2|Initial resident bytes|Median four-step CUDA ms|
|---|---|---:|---:|---:|
|dynamic_eager|COLLECTED|0.000000000|14680064|54.45968|
|int4_chunked|COLLECTED|0.000000000|5734400|117.30568|
|int4_global|COLLECTED|0.000000000|5734400|170.99673|
|int4_triton|GATE_D_FAIL|0.076934038|5734400|NOT_RUN_GATE_FAILURE|
|static_eager|COLLECTED|0.000000000|18350080|63.26064|
|static_sdpa|GATE_D_FAIL|0.076202266|18350080|NOT_RUN_GATE_FAILURE|

Static cache is page-rounded contiguous storage, not page-table paged serving. Actual allocator peaks, reserve/slack, persistent codes/scales/padding and physical snapshots are in each sealed summary. Physical snapshots are not independently sampled physical-HBM peaks. Timing excludes prefill/materialization; inspect raw trial order before steady-state comparisons.

## Native distributions and BF16 shape probes

All20 distribution cells collected; native observer controls exact. Five models ×2 source documents ×2 lengths. Only2 independent source IDs; heads/layers/steps are repeated observations. Native Q/K/V/Z/P/Y and explicitly reconstructed C/S/W carry separate labels. Neither narrow-W nor concentration screen passed in any of20 cells. This does not select a hardware lowering.

The3 preselected BF16 operations replayed upstream and native cached/full proxies exactly. At fixed operands, M1 versus full M reproduces small native differences. All15 true/false reduction-flag pairs were bitwise identical. CUDA kernel names were observed, but internal cuBLAS algorithm IDs remain UNKNOWN. Same-weight FP32 is diagnostic, not a backend-bug proof or an exhaustive explanation of historical failures.

## Sealed artifact locations

- [Additive audit](../results/confirmation_audit_v2/ea42f05db641537d/blocks/confirmation/report.md)
- [BF16 shape probe](../results/shape_cause_v2/763655a288dfbaa8/blocks/probe/report.md)
- [Frozen campaign terminal checkpoint](../results/replication_v2/2aa605c55475af9d/public_checkpoint.json)

|Family|Block|Manifest|
|---|---|---|
|confirmation_audit_v2|confirmation|[sealed manifest](../results/confirmation_audit_v2/ea42f05db641537d/blocks/confirmation/manifest.json)|
|distributions_v2|459394cf61cc3dc8|[sealed manifest](../results/distributions_v2/12fbc139be256644/blocks/459394cf61cc3dc8/manifest.json)|
|distributions_v2|491e08f4b325b089|[sealed manifest](../results/distributions_v2/12fbc139be256644/blocks/491e08f4b325b089/manifest.json)|
|distributions_v2|5d57a978068a90f7|[sealed manifest](../results/distributions_v2/12fbc139be256644/blocks/5d57a978068a90f7/manifest.json)|
|distributions_v2|5ef519dd6fd984bd|[sealed manifest](../results/distributions_v2/12fbc139be256644/blocks/5ef519dd6fd984bd/manifest.json)|
|distributions_v2|712bf6bc0c538e45|[sealed manifest](../results/distributions_v2/12fbc139be256644/blocks/712bf6bc0c538e45/manifest.json)|
|distributions_v2|82706424b6afede7|[sealed manifest](../results/distributions_v2/12fbc139be256644/blocks/82706424b6afede7/manifest.json)|
|distributions_v2|8477a9163eacc8cf|[sealed manifest](../results/distributions_v2/12fbc139be256644/blocks/8477a9163eacc8cf/manifest.json)|
|distributions_v2|88fc97e684d8d6db|[sealed manifest](../results/distributions_v2/12fbc139be256644/blocks/88fc97e684d8d6db/manifest.json)|
|distributions_v2|a052ee62a6e77c6e|[sealed manifest](../results/distributions_v2/12fbc139be256644/blocks/a052ee62a6e77c6e/manifest.json)|
|distributions_v2|b99d9c22bf0422d9|[sealed manifest](../results/distributions_v2/12fbc139be256644/blocks/b99d9c22bf0422d9/manifest.json)|
|distributions_v2|ba5d915808456886|[sealed manifest](../results/distributions_v2/12fbc139be256644/blocks/ba5d915808456886/manifest.json)|
|distributions_v2|bcd7931658034f81|[sealed manifest](../results/distributions_v2/12fbc139be256644/blocks/bcd7931658034f81/manifest.json)|
|distributions_v2|c871d82358e654bf|[sealed manifest](../results/distributions_v2/12fbc139be256644/blocks/c871d82358e654bf/manifest.json)|
|distributions_v2|d78cc115af9f3f11|[sealed manifest](../results/distributions_v2/12fbc139be256644/blocks/d78cc115af9f3f11/manifest.json)|
|distributions_v2|d88387753849fcce|[sealed manifest](../results/distributions_v2/12fbc139be256644/blocks/d88387753849fcce/manifest.json)|
|distributions_v2|dacd39749a8606f5|[sealed manifest](../results/distributions_v2/12fbc139be256644/blocks/dacd39749a8606f5/manifest.json)|
|distributions_v2|e77004abc582d115|[sealed manifest](../results/distributions_v2/12fbc139be256644/blocks/e77004abc582d115/manifest.json)|
|distributions_v2|f29ad8df4016429e|[sealed manifest](../results/distributions_v2/12fbc139be256644/blocks/f29ad8df4016429e/manifest.json)|
|distributions_v2|ff392ebc80134bbb|[sealed manifest](../results/distributions_v2/12fbc139be256644/blocks/ff392ebc80134bbb/manifest.json)|
|distributions_v2|ff3cdcad75a24b0d|[sealed manifest](../results/distributions_v2/12fbc139be256644/blocks/ff3cdcad75a24b0d/manifest.json)|
|replication_v2|confirmatory-opt1_3b|[sealed manifest](../results/replication_v2/2aa605c55475af9d/blocks/confirmatory-opt1_3b/manifest.json)|
|replication_v2|confirmatory-qwen14b|[sealed manifest](../results/replication_v2/2aa605c55475af9d/blocks/confirmatory-qwen14b/manifest.json)|
|replication_v2|confirmatory-qwen7b|[sealed manifest](../results/replication_v2/2aa605c55475af9d/blocks/confirmatory-qwen7b/manifest.json)|
|resident_v2|2f2a54947331ab7d|[sealed manifest](../results/resident_v2/4ec6746ef06d4ca5/points/2f2a54947331ab7d/manifest.json)|
|resident_v2|52acb86f229ab752|[sealed manifest](../results/resident_v2/4ec6746ef06d4ca5/points/52acb86f229ab752/manifest.json)|
|resident_v2|86be1fead9392cc3|[sealed manifest](../results/resident_v2/4ec6746ef06d4ca5/points/86be1fead9392cc3/manifest.json)|
|resident_v2|94221bf46249728f|[sealed manifest](../results/resident_v2/4ec6746ef06d4ca5/points/94221bf46249728f/manifest.json)|
|resident_v2|b3045dfa99c1a60c|[sealed manifest](../results/resident_v2/4ec6746ef06d4ca5/points/b3045dfa99c1a60c/manifest.json)|
|resident_v2|cd3ea20613f779f6|[sealed manifest](../results/resident_v2/4ec6746ef06d4ca5/points/cd3ea20613f779f6/manifest.json)|
|shape_cause_v2|probe|[sealed manifest](../results/shape_cause_v2/763655a288dfbaa8/blocks/probe/manifest.json)|

All31 newly published blocks were byte-size/SHA256 checked before publication, including every terminal resident failure. Older blocks and protocols are preserved. No source token IDs, raw activation tensors, weights, credentials or private operational logs are included.
