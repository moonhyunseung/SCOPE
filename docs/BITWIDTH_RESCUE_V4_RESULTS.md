# INT8 bitwidth rescue v4 results

Run `8eb785804303f773`; preregistered on main `a2b1dca6e135156b5a15ec6467048a289e281906` and5 remote source/config/prepared blobs verified before GPU. Source SHA256 `2ec0788c2aeded8e8214873561626ab8df0f5c7ec504c10ec4c156680f61b0a3`.12targeted CPU tests passed; no scientific repair.

Qwen2.5-1.5B pinned revision, same2 screening sources/windows, BF16 eager/B1/L512/four teacher steps/two exact replays. Six new source-representation conditions,96finite complete-grid metric rows. n=2sources; no CI/population test. No packed memory, task accuracy, latency or resident capacity claim.

## Both unchanged .025 gates and baseline

R4=K BF16/V INT8;R5=K INT8/V BF16;R6=K INT8/V INT8. Only bitwidth changes from prior INT4, so the same scale policy uses divisor127 instead of7.

|Source|Role|Max L2 vs cached|Cached gate|Max L2 vs full|Full gate|Baseline|Disposition|
|---|---|---:|---|---:|---|---|---|
|prose `7190bc75f0028b8ef4e2`|R4|0.0206134389|PASS|0.0200800833|PASS|BASELINE_DIVERGENCE|BASELINE_DIVERGENCE_UNRESOLVED|
|prose `7190bc75f0028b8ef4e2`|R5|0.5838921716|FAIL|0.5736341368|FAIL|BASELINE_DIVERGENCE|REPRESENTATION_CACHED_FAIL|
|prose `7190bc75f0028b8ef4e2`|R6|0.5756723994|FAIL|0.5656447006|FAIL|BASELINE_DIVERGENCE|REPRESENTATION_CACHED_FAIL|
|code `95459423d3b8f7c779e3`|R4|0.0290645334|FAIL|0.0308203458|FAIL|BASELINE_SCREEN_PASS|REPRESENTATION_CACHED_FAIL|
|code `95459423d3b8f7c779e3`|R5|0.1377942237|FAIL|0.1388858358|FAIL|BASELINE_SCREEN_PASS|REPRESENTATION_CACHED_FAIL|
|code `95459423d3b8f7c779e3`|R6|0.1259404743|FAIL|0.1267743785|FAIL|BASELINE_SCREEN_PASS|REPRESENTATION_CACHED_FAIL|

Baseline cached/full maxima are unchanged: prose .0272492025951(BASELINE_DIVERGENCE), code .0221552555301(BASELINE_SCREEN_PASS). Relative distances are not subtracted into error contributions. Any accidental prose both-gate pass remains baseline-unresolved.

## Minimum eligibility for further scientific review

The preregistered minimum is cachedPASS on both sources plus code fullPASS. This does not authorize system adoption or hide prose full-reference failures.

|Candidate|Eligible for further review|
|---|---|
|R4|False|
|R5|False|
|R6|False|

Predefined stop-uniform-family condition: **True**. It triggers only if all3 candidates fail cached .025 on at least one source. No new grid, kernel, group/scale policy, source recruitment or resident frontier was automatically started.

## Output distribution diagnostics

Signed natural-next-token NLL delta is output minus reference; extrema are four repeated steps, not population estimates. KL is reference||output; L2 is not task error rate.

|Source|Role|Reference|NLL delta min|NLL delta max|Max KL|Top1 disagreement steps /4|
|---|---|---|---:|---:|---:|---:|
|prose|R4|native_cached|-0.0038347056|0.01373723|0.00075369813|0/4|
|prose|R4|original_full|-0.0025556316|0.0018014355|0.00050361641|0/4|
|prose|R5|native_cached|-0.0092229459|0.54445615|0.35504308|0/4|
|prose|R5|original_full|-0.011166867|0.53252035|0.33841201|0/4|
|prose|R6|native_cached|-0.0045939767|0.52386092|0.3409622|0/4|
|prose|R6|original_full|-0.0065378975|0.51192513|0.32511642|0/4|
|code|R4|native_cached|-0.079336657|0.00063818281|0.0018005149|0/4|
|code|R4|original_full|-0.080093541|0.0037382316|0.0018045096|0/4|
|code|R5|native_cached|-0.078696258|0.14368607|0.0070419753|0/4|
|code|R5|original_full|-0.079453142|0.14795988|0.0077050078|0/4|
|code|R6|native_cached|-0.0067112155|0.14416603|0.0078014347|0/4|
|code|R6|original_full|-0.0064336382|0.14843983|0.0084766063|0/4|

## Reference and control verification

Historical prefill, original-full logits and native cached logits/cache/R0 hashes exactly match both previous V3 replays. Baseline metrics and Python/torch/Transformers/CUDA/GPU/precision runtime fields are exact. All1344layer-append role/oracle/prefix/length controls and new candidate metrics/cache/logits replay exactly. Direct signed int8 code dequantization matches existing packed INT8 decode bytewise; selected/unselected role policy applies both at initialization and append.

The cache persists dense BF16 decoded values. Earlier R0-R3, original800conditions and prior failures remain immutable. All six candidate outcomes are shown, and the old .025 gates are unchanged.

[Complete metrics](../results/bitwidth_rescue_v4/8eb785804303f773/blocks/rescue/bitwidth_comparisons.csv) · [source summaries](../results/bitwidth_rescue_v4/8eb785804303f773/blocks/rescue/source_summary.json) · [replay controls](../results/bitwidth_rescue_v4/8eb785804303f773/blocks/rescue/replay_controls.json) · [sealed manifest](../results/bitwidth_rescue_v4/8eb785804303f773/blocks/rescue/manifest.json) · [contract](BITWIDTH_RESCUE_V4_CONTRACT.md)
