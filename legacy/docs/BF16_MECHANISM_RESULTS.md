# Bounded native BF16 mechanism diagnosis

The stratified Qwen2.5-1.5B diagnosis completed 32 conditions on 15 distinct original source units: 16 paired BF16/FP32 cases, selected to balance historical PASS/FAIL, prose/code and L512/L2048. Every BF16 step metric exactly reproduced the historical result. All 288 instrumentation/explicit-position-mask comparisons passed, as did appended cache length, RoPE, effective mask row and scaling checks. BF16 kept its selected 8 PASS/8 FAIL split; FP32 passed 16/16. This selected failure fraction is not a prevalence estimate.

The follow-up completed 32 conditions and 1,728 operation-control records. At the first exact nonzero BF16 stage in all 48 prespecified traced forwards (steps 0, 3, 7), native input parity and same-input shape controls demonstrated a numerical shape effect: first-layer MLP down projection in 36 traces, QK scores in 8, and PV attention output in 4. Native-operation proxies replayed the original cached operation exactly. The fixed-vector/operand M=1 versus M=L+step+1 relative differences at those stages ranged from 1.8951e-9 to 6.3189e-4. All 864 recorded shape-error metric pairs remained unchanged when the BF16 reduced-precision-reduction option was toggled. This is a negative result for that option; raw outputs across its two settings were not directly compared for bitwise equality.

These are native BF16 operations with no quantization intervention. The measured first-stage differences can therefore arise before quantization. They propagate through subsequent layers: final BF16 relative logit errors ranged from 0.007215 to 0.063097 (median 0.014214 over 128 correlated steps); FP32 ranged from 6.406e-7 to 4.418e-6. The unchanged 0.025 gate continues to mark failures. Neither intermediate-stage differences nor eight-token continuation NLL are model-wide accuracy measurements.

The first-stage classification has zero unresolved traces within this bounded sample. Complete contribution to final logits, untraced decode positions, the internal GEMM algorithm, and the other 445 of 453 historical failed conditions remain unestablished. No implementation defect is inferred. Cross-dtype FP32 norm/projection first differences remain small under the final fidelity bound. Structural parity rules out only the tested mismatches in these selected cases.

Reproduce one demonstrated operation on the server with the private frozen plan:

```bash
python -m scope_v2.minimal --plan configs/continuation_diagnosis_v2.json --case-id 04638e2cd2ce202c --step 3 --operation mlp_down
python -m scope_v2.minimal --plan configs/continuation_diagnosis_v2.json --case-id 04f82ec90543c3fc --step 3 --operation QK
python -m scope_v2.minimal --plan configs/continuation_diagnosis_v2.json --case-id 04638e2cd2ce202c --step 0 --operation PV
```

The script reports input parity, native proxy replay, changed-shape output versus native full output and final logit error. It uses the shared exclusive H200 lock and local checkpoint only. Exact inputs and raw model tensors stay on the server. Public `results/mechanism_v2/41432250ead13a05/blocks/{stratified32,classification}/` retain statistical evidence and input-block hashes. Original diagnostic classifications and results are preserved; this follow-up is additive. No performance or persistent-cache capacity claim is made.
