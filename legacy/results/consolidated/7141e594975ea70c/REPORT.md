# H200 campaign 7141e594975ea70c

Frozen code: 7d3c442bc1c3e154d64b274c830cea522da391c1

Execution blocks: {"PASS": 23, "UNSUPPORTED": 4, "SKIPPED_WITH_REASON": 12, "FAIL": 2}

Publication: {"PENDING": 41}

Real-model conditions passed: 6

## Actual full-model dense baselines

| Model | L | Input | Median decode ms | Native replay max relative L2 |
|---|---:|---|---:|---:|
| Qwen/Qwen2.5-1.5B-Instruct | 512 | prose | 13.2692 | 0.0 |
| Qwen/Qwen2.5-1.5B-Instruct | 512 | code | 13.2767 | 0.0 |
| Qwen/Qwen2.5-1.5B-Instruct | 2048 | prose | 14.129 | 0.0 |
| Qwen/Qwen2.5-1.5B-Instruct | 2048 | code | 13.6436 | 0.0 |
| EleutherAI/pythia-1b | 512 | prose | 6.2933 | 0.0 |
| EleutherAI/pythia-1b | 512 | code | 5.4746 | 0.0 |

Full-model values are uninstrumented eager BF16 forward plus greedy token choice, excluding prefill; median of three six-token timing windows. Exact contexts grow from the initial length. These are dense-cache baselines.

## Operator screening

| Condition | Variant | Status | Median ms | Execution relative L2 |
|---|---|---|---:|---:|
| P4-04fb19d9f6c5 | dense_eager | PASS | 0.15597 | 0.0 |
| P4-04fb19d9f6c5 | dense_sdpa | FAIL | None | 0.150582 |
| P4-04fb19d9f6c5 | packed_restore_eager | PASS | 0.4312 | 0.0 |
| P4-04fb19d9f6c5 | packed_restore_sdpa | FAIL | None | 0.103643 |
| P4-728b54fc0999 | dense_eager | PASS | 0.08635 | 0.0 |
| P4-728b54fc0999 | dense_sdpa | FAIL | None | 0.063311 |
| P4-728b54fc0999 | packed_restore_eager | PASS | 0.24296 | 0.0 |
| P4-728b54fc0999 | packed_restore_sdpa | FAIL | None | 0.122218 |
| P4-149a05ffeab0 | dense_eager | PASS | 0.15205 | 0.0 |
| P4-149a05ffeab0 | dense_sdpa | FAIL | None | 0.244084 |
| P4-149a05ffeab0 | packed_restore_eager | PASS | 0.41803 | 0.0 |
| P4-149a05ffeab0 | packed_restore_sdpa | FAIL | None | 0.417089 |
| P4-c1cc3e592af8 | dense_eager | PASS | 0.08638 | 0.0 |
| P4-c1cc3e592af8 | dense_sdpa | FAIL | None | 0.302649 |
| P4-c1cc3e592af8 | packed_restore_eager | PASS | 0.24339 | 0.0 |
| P4-c1cc3e592af8 | packed_restore_sdpa | FAIL | None | 0.386079 |
| P4-9e912262597e | dense_eager | PASS | 0.15072 | 0.0 |
| P4-9e912262597e | dense_sdpa | PASS | 0.07106 | 0.004059 |
| P4-9e912262597e | packed_restore_eager | PASS | 0.41405 | 0.0 |
| P4-9e912262597e | packed_restore_sdpa | PASS | 0.32374 | 0.003849 |
| P4-bc18bc673b85 | dense_eager | PASS | 0.0865 | 0.0 |
| P4-bc18bc673b85 | dense_sdpa | PASS | 0.04117 | 0.005075 |
| P4-bc18bc673b85 | packed_restore_eager | PASS | 0.23774 | 0.0 |
| P4-bc18bc673b85 | packed_restore_sdpa | PASS | 0.18768 | 0.003565 |

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
python -m scope.runner reproduce --run-id 7141e594975ea70c
```

The reproduce command validates hashes. Independent GPU replication requires a fresh checkout and a fresh result store; do not copy previous campaign results into it. Publication requires existing server Git authentication. Raw tensors/logs remain at /home/inugpu/workspace/scope_h200_20261018_9EfrwM/SCOPE_v2/results/campaigns/7141e594975ea70c.
