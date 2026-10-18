# SCOPE continuation v2

Historical baseline: `3f06d6f5e8f86959f6e1a1bc9fc67ccdd05c67a8`.
Historical source, input and result files are immutable. New modules live in `scope_v2/`.

## Native numerical diagnosis

The frozen mechanism sample selects two cases in each domain × context length × historical PASS/FAIL stratum from the previous dtype-audit population: 16 cases, 15 distinct source units, each measured in BF16 and FP32. This is an outcome-stratified diagnostic sample; its failure fraction cannot estimate population prevalence. It reuses the original natural teacher tokens and checkpoint revision. A first pilot runs one historical passing and one failing L512 case in both dtypes.

All eight decode steps retain the original relative-L2 ≤ 0.025 full/cached gate. Steps 0, 3 and 7 additionally trace the last query through native embeddings, norm, Q/K/V projection, RoPE, full appended K/V, actual pre-softmax scores, FP32 softmax output, cast probabilities, attention output, projection, residual/MLP and logits. Structural checks compare positions, cos/sin, effective mask row, scale and cache length. Instrumentation must reproduce the uninstrumented logits exactly. Tracing calls the installed original eager attention function. Only its actual softmax input/output are observed; Q/K/V are captured at the native attention boundary. No reconstructed scores are represented as native observations.

Interventions:

- Explicit position IDs and all-valid attention mask versus the default native contract.
- Recompute the entire prefix before each selected decode step versus recurrent cache history. This also changes prefix GEMM shapes; it does not isolate history from all shape effects.
- Fix one exact first-layer input vector, projection weight and bias; run M=1 and M=L+1 GEMMs. Compare both to FP32 computation with the same current-dtype weights cast to FP32. This establishes shape sensitivity when nonzero, but does not identify a particular cuBLAS algorithm or completely attribute downstream errors.

FP32 checkpoint loading and FP32 arithmetic on BF16-rounded weights are separate controls. First exact nonzero error and first stage exceeding the old 0.025 logit tolerance are both recorded; applying that logit threshold to intermediate stages is descriptive, not a new correctness gate.

```bash
python -m pytest -q tests_v2
python -m scope_v2.diagnose prepare \
  --population /path/to/frozen/results/population/46f7c2fa18e3d9c7 \
  --input /path/to/frozen/data/population_v1/manifest.json \
  --plan configs/continuation_diagnosis_v2.json
python -m scope_v2.diagnose run --plan configs/continuation_diagnosis_v2.json --limit 2
python -m scope_v2.diagnose run --plan configs/continuation_diagnosis_v2.json
```

The private frozen plan includes exact input IDs. Public artifacts expose their hashes and source-case identities; raw model states and projection vectors remain on the server. To reproduce projection sensitivity, run `python -m scope_v2.reproduce /path/to/projection_q_proj.pt --rows 513` on the server. Model downloads are disabled (`local_files_only=True`).

SQLite checkpoints record every condition and attempt; per-condition manifests seal files before aggregation. Completed or failed attempts are never silently rerun. A source/config change creates a different deterministic run ID. A shared exclusive GPU lock and H200 preflight protect unrelated work. GPU tracing timing is not performance evidence.

## Publication and dependencies

Server noninteractive push failed with exit 128 because no GitHub username/authentication was available. The runner transactionally queues verified analysis manifests as `PENDING`. An interactive assistant can publish a compact block through the connected GitHub app, but that is not autonomous worker authentication. A block becomes `PUBLISHED` only after remote commit verification.

Gate A remains bounded to the observed model, inputs, dtype, backend and shapes. A demonstrated numerical mechanism does not certify all 453 prior failures. Additional-model K/V confirmation, transformed execution correctness and real persistent-cache capacity must be reported separately. No hardware performance or capacity result is implied by this diagnostic suite.
