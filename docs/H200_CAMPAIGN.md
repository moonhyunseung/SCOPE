# H200 exploratory campaign

This is a bounded pilot, not a completed systems evaluation. The upstream P0 scaffold is preserved. P0-P7 follow the execution blocks in the supplied research brief, extending the bootstrap naming in README.

Use Python 3.12 with CUDA PyTorch, Triton, Transformers 5.19.0, Hugging Face Hub, PyYAML, pytest. Freeze/check the versions recorded by preflight. Installation example in a new environment:

```bash
python -m pip install -e ".[test]" transformers huggingface_hub pyyaml triton
python -m scope.runner preflight --config configs/h200_campaign.yaml
python -m scope.runner plan --config configs/h200_campaign.yaml
tmux new-session -d -s scope-h200 'python -u -m scope.runner run --config configs/h200_campaign.yaml --resume --publish-each-block > campaign.log 2>&1'
python -m scope.runner status --config configs/h200_campaign.yaml
python -m scope.runner reproduce --run-id RUN_ID
```

The example assumes the intended CUDA PyTorch runtime is installed and this repository is on PYTHONPATH or installed. Do not replace an existing serving environment. The observed deployment uses a private venv with a read-only .pth reference to an existing CUDA dependency environment; exact runtime versions are recorded, with no changes to that environment.

P0 validates synthetic MHA/GQA/MQA and append, mask, packing and runner integrity tests. P1 loads three public pinned checkpoint revisions, samples first/middle/last actual eager attention layers across three decode steps, and checks native attention output by exact same-dtype replay and cached versus recomputed logits. FP32 diagnostic deviation is reported as numerical sensitivity, separately from executor error. Real Q/K/V include the model's own position transforms and appended cache. Contexts that exceed the checkpoint's position limit are explicit UNSUPPORTED conditions. Input populations are fixed repeated public prose/code, not a quality dataset.

P2 reports bounded conditional/joint statistics. P3 tests one illustrative signed INT4 offset/nibble layout and reports representation error separately. P4 screens dense eager/SDPA and unpack+dequantize+eager/SDPA on real captured inputs, refuses to time variants that violate the original BF16 eager execution contract, and, uses three timing windows, alternating path order and CUDA events. It is a single-layer microbenchmark; repeated windows are not independent workload replications. It measures tensor-storage bytes and allocator peaks, not maximum resident KV or full-model throughput. P5 reports architecture coverage. P6 aggregates exploratory paired timings. P7 audits hashes; it is not independent replication.

SQLite, atomic manifests, a campaign lock, parent-death worker watchdog and heartbeat provide durable progress. Resume checks a frozen source fingerprint; do not edit code during a campaign. Publication is separate from execution: failed pushes are PENDING and later resume retries them without GPU reruns. Three bounded publication attempts use existing credentials, never credentials embedded in the config. Remote SHA read-back is required for PUBLISHED. No force push. Explicit allowlists omit weights, raw samples, stdout, SQLite and caches.

The controller enforces dedicated H200, available memory/disk, per-process 96 GiB allocator cap, 12 GiB reserve, job/campaign timeouts, and never stops other processes. Raw samples and complete logs remain on the execution host with SHA256. Only compact manifests/summaries/CSV/reports/checksums are eligible for Git publication. A raw artifact's local location is not a public download URL.

Untested: medium/large model quality, MQA real-model traces, hybrid/MLA, long-context/batch sweeps, custom direct-packed/Triton/FlashAttention/vLLM backends, traffic/SM/utilization profiling, full-model latency, resident-capacity frontier, confirmatory factorial designs and independent GPU replication. Successful P0/P1/operator replays do not establish these claims.

## Protocol correction v2

The v1 run incorrectly gated original BF16 model attention against a full FP32 diagnostic. Large differences were observed for Qwen and OPT and remain failed v1 conditions. v2 fixes the reference contract without increasing tolerances: exact BF16 QK/scale/mask, FP32 softmax cast to BF16, BF16 PV must match native output bit-for-bit. Diagnostic FP32 differences remain reported. Captured native returned probabilities/output are distinguished from reconstructed diagnostics. Three additional tests compare this replay with installed upstream Qwen2/GPT-NeoX/OPT functions. SDPA paths that violate the same-representation BF16 reference remain failed variants and are never timed.
