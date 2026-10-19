# H200 campaign 479dde6421f18d68

Frozen code: ba0b4bbdea12be44aa7b6686c766e6173da0e85a

Execution blocks: {"PASS": 11, "FAIL": 6, "SKIPPED_WITH_REASON": 12}

Publication: {"PENDING": 29}

Real-model conditions passed: 2

## Actual full-model dense baselines

| Model | L | Input | Median decode ms | Native replay max relative L2 |
|---|---:|---|---:|---:|
| Qwen/Qwen2.5-14B-Instruct | 512 | code | 25.0375 | 0.0 |
| Qwen/Qwen2.5-14B-Instruct | 2048 | prose | 24.7628 | 0.0 |

Full-model values are uninstrumented eager BF16 forward plus greedy token choice, excluding prefill; median of three six-token timing windows. Exact contexts grow from the initial length. These are dense-cache baselines.

## Operator screening

| Condition | Variant | Status | Median ms | Execution relative L2 |
|---|---|---|---:|---:|
| P4-1aa300813eba | dense_eager | PASS | 0.15206 | 0.0 |
| P4-1aa300813eba | dense_sdpa | PASS | 0.07294 | 0.014676 |
| P4-1aa300813eba | packed_restore_eager | PASS | 0.42402 | 0.0 |
| P4-1aa300813eba | packed_restore_sdpa | PASS | 0.3341 | 0.015422 |
| P4-b4492dcfad67 | dense_eager | PASS | 0.1683 | 0.0 |
| P4-b4492dcfad67 | dense_sdpa | PASS | 0.08224 | 0.010673 |
| P4-b4492dcfad67 | packed_restore_eager | PASS | 0.45474 | 0.0 |
| P4-b4492dcfad67 | packed_restore_sdpa | PASS | 0.3637 | 0.009079 |

Operator medians summarize three CUDA-event timing windows. Failed variants were not timed. Storage bytes count actual K/V tensor storage including scales, rather than a maximum resident-capacity experiment.

## Limits

- No full-model compressed-cache integration or end-model quality evaluation
- No measured maximum resident-KV/decode frontier
- No real MQA, hybrid/MLA, custom Triton, FlashAttention or vLLM execution
- Fixed repeated public prose/code population; no confirmatory held-out replication
- P7 is artifact integrity only; block PASS does not mean every screened variant passed

## Reproduction

Use the frozen source commit and exact configuration in campaign.json with the recorded CUDA environment. On a dedicated H200:

```bash
python -m scope.runner preflight --config configs/h200_campaign.yaml
python -m scope.runner plan --config configs/h200_campaign.yaml
python -m scope.runner run --config configs/h200_campaign.yaml --resume --publish-each-block
python -m scope.runner reproduce --run-id 479dde6421f18d68
```

The reproduce command validates hashes. Independent GPU replication requires a fresh checkout and a fresh result store; do not copy previous campaign results into it. Publication requires existing server Git authentication. Raw tensors/logs remain at /home/inugpu/workspace/scope_h200_20261018_9EfrwM/SCOPE_v2/results/campaigns/479dde6421f18d68.
