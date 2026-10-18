# K/V role fidelity screen v3 results

Preregistered on main `63a9e606a670aae0746a2afbc19dc9d81020387c` before GPU execution. Run `6c2c9a058382dcb0`; source SHA256 `b8ce1f2721f71b5772f13394dd28ad5ab44f5e3c7b0fbb1f7b6b617b17e8dd7d`. CPU controls:9 tests passed. Dedicated H200, allocator ceiling at most96GiB adjusted to available headroom,44GiB physical reserve checkpoints,24GiB disk reserve,1500s worker limit. No scientific repair used.

Both screening sources, Qwen2.5-1.5B pinned revision, BF16 eager, B1/L512, four teacher-forced steps, two exact replays. Eight source-representation conditions; two original-full references are shared and replayed. Independent n=2 sources, no CI/population test. Relative logit L2 is not task error rate.

## All conditions and unchanged .025 gates

R0=BF16/BF16;R1=INT4/BF16;R2=BF16/INT4;R3=INT4/INT4. INT4 group32, FP32 scales; only role changes.

|Source|Role K/V|Max L2 vs cached|Cached gate|Max L2 vs full|Full gate|Baseline|Local disposition|
|---|---|---:|---|---:|---|---|---|
|prose `7190bc75f0028b8ef4e2`|R0|0.0000000000|PASS|0.0272492026|FAIL|BASELINE_DIVERGENCE|FAIL|
|prose `7190bc75f0028b8ef4e2`|R1|0.8751218649|FAIL|0.8795554741|FAIL|BASELINE_DIVERGENCE|FAIL|
|prose `7190bc75f0028b8ef4e2`|R2|0.0672726547|FAIL|0.0552491784|FAIL|BASELINE_DIVERGENCE|FAIL|
|prose `7190bc75f0028b8ef4e2`|R3|0.9169751984|FAIL|0.9207611394|FAIL|BASELINE_DIVERGENCE|FAIL|
|code `95459423d3b8f7c779e3`|R0|0.0000000000|PASS|0.0221552555|PASS|BASELINE_SCREEN_PASS|LOCAL_SCREEN_PASS|
|code `95459423d3b8f7c779e3`|R1|1.7654984350|FAIL|1.7664143333|FAIL|BASELINE_SCREEN_PASS|FAIL|
|code `95459423d3b8f7c779e3`|R2|0.1275159497|FAIL|0.1275619046|FAIL|BASELINE_SCREEN_PASS|FAIL|
|code `95459423d3b8f7c779e3`|R3|1.7516419422|FAIL|1.7526229011|FAIL|BASELINE_SCREEN_PASS|FAIL|

No compressed role is accepted by the preregistered local fidelity contract across both sources. Baseline divergence is retained and cannot be cured by relabelling a quantized cancellation as fidelity. The outcome is restricted to these two sources/model/windows/quantizer. It does not reject all INT4 or all models.

## NLL, KL and top1 diagnostics

Natural next-token NLL delta is output minus the named reference. A negative delta is not a corpus-quality gain; extrema span the four repeated steps (replay duplicates are exact). KL is reference||output.

|Source|Role|Reference|NLL delta min|NLL delta max|Max KL|Top1 disagreement steps /4|
|---|---|---|---:|---:|---:|---:|
|prose|R0|native_cached|0|0|0|0/4|
|prose|R0|original_full|-0.011935795|0.001279074|0.000773516|0/4|
|prose|R1|native_cached|-0.09102855|10.017362|9.9318448|3/4|
|prose|R1|original_full|-0.089749476|10.017775|9.9342596|3/4|
|prose|R2|native_cached|2.7392995e-06|0.034699463|0.002621733|0/4|
|prose|R2|original_full|-0.006889731|0.035978537|0.0027957883|0/4|
|prose|R3|native_cached|-0.093441482|10.664163|10.580431|3/4|
|prose|R3|original_full|-0.092162408|10.664577|10.582847|3/4|
|code|R0|native_cached|0|0|0|0/4|
|code|R0|original_full|-0.00075688402|0.013200544|0.0010175076|0/4|
|code|R1|native_cached|-0.073979511|9.7988271|13.144296|2/4|
|code|R1|original_full|-0.074736395|9.8031009|13.157437|2/4|
|code|R2|native_cached|-0.15681596|0.098464029|0.0073604122|0/4|
|code|R2|original_full|-0.15757285|0.10273783|0.007549827|0/4|
|code|R3|native_cached|-0.22229179|10.994808|13.918514|2/4|
|code|R3|original_full|-0.22304868|10.999081|13.933381|2/4|

## Verification and limits

All128 metric rows have the complete source/role/reference/step/replay key grid and finite metrics. Both replays are exact for every metric, logits hash, cache hash and1792 layer-append controls. Selected-role integer-oracle parity, unselected BF16 bytes, unchanged cache prefix and sequence lengths passed. R0 matches unmodified native cached on both sources; prose R3 matches prior sealed R2 quantized trajectory exactly.

The independent mixed cache persists **dense dequantized BF16** values to isolate representation semantics. This is not proof of packed physical HBM savings, throughput, task accuracy or serving capacity. No new kernel family, lowerings repair, INT8 run or frontier expansion was started. INT8 remains conditional on a subsequent advisor decision. Old800 primary results and failure history are immutable.

[All step metrics](../results/role_fidelity_v3/6c2c9a058382dcb0/blocks/role/role_comparisons.csv) · [source summaries](../results/role_fidelity_v3/6c2c9a058382dcb0/blocks/role/source_summary.json) · [controls](../results/role_fidelity_v3/6c2c9a058382dcb0/blocks/role/replay_controls.json) · [sealed manifest](../results/role_fidelity_v3/6c2c9a058382dcb0/blocks/role/manifest.json) · [execution contract](ROLE_FIDELITY_V3_CONTRACT.md)
