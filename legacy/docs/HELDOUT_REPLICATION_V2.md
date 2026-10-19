# Held-out native K/V factorial replication

This continuation preserves the original 1.5B population. The new sampler excludes all 512 historical article/project-name units, normalized content duplicates and token-5gram Jaccard ≥0.8 window duplicates. It selects 64 new units per domain by a new deterministic hash rank from the same pinned licensed dataset shards. The frame remains bounded; related Wikipedia topics and upstream project lineages may remain dependent.

Each model sees the same raw character segment and endpoint, tokenized by its pinned tokenizer. Each consumes the final natural 2057-token window. Cross-vocabulary token targets differ, so effects are evaluated within each model. No repetition/padding is used. Eight units/domain form screening; the other 56/domain are frozen confirmation before GPU outcomes. Models are Qwen2.5-1.5B/7B/14B, Pythia-1B and OPT-1.3B where their pinned assets are already cached. No new weights are downloaded.

The operator wrapper changes K/V operands passed into the installed original native eager attention function. It preserves the family's scaling, mask, positional transformations and accumulation behavior. The full 2×2 factorial is K0V0, K1V0, K0V1, K1V1; original passthrough and packed/unpacked K1V1 add exact controls. Both isolated cloned-history and recurrent regimes consume identical natural teacher tokens. Exact no-op, clone, unchanged-snapshot, packing, hook coverage, cache append and first-step-history checks are mandatory. The old full/cached 0.025 gate is retained and reported separately, including every failure.

Staging avoids a Cartesian sweep:

1. Pilot: one source/domain × two dtypes × each available model (20 conditions for five models).
2. Screening: all 16 screening sources × BF16/FP32 × L512 core. Each source has one additional paired BF16 condition, assigned before outcomes: 8-bit, group64, first/middle/last layer, KV-head0, L1536, or group128. Unsupported group/head combinations remain in the registry.
3. Confirmation: 112 held-out sources × each model, BF16 L1536 INT4 g32, all layers and KV heads. L1536 plus eight decode positions respects Pythia/OPT's 2048-position contract.

Five models yield 800 unique planned conditions; pilot conditions are reused by screening. The experimental unit remains the source, not conditions × models × heads × steps. Eight primary contrasts are prespecified for the two additional architecture families Pythia/OPT, separately domain and isolated/recurrent: per-source mean of eight relative-L2 steps, K-only minus V-only. Source-block percentile bootstrap uses 20,000 resamples and 99.375% intervals for nominal Bonferroni 95% family coverage. Other endpoints, Qwen sizes and parameter screens are exploratory. NLL covers eight natural targets per source and does not establish corpus perplexity or generation quality.

The runner freezes code and input hashes, byte-audits cached model checkpoints, uses SQLite WAL checkpoints, batches at most eight conditions per isolated worker, and limits each batch to 45 minutes. A shared exclusive lock, dedicated-H200 check, 96GiB allocator cap and disk/HBM reserves protect unrelated work. Control failures stop the campaign; unsupported combinations remain explicit. No LLM calls occur in the loop. Complete per-model stage blocks are sealed and transactionally queued for publication. Server Git authentication is unavailable, so autonomous publication remains `PENDING`; connected-app publication requires remote commit verification.

```bash
python -m scope_v2.heldout --historical /path/to/original/manifest.json --out data/heldout_v2/manifest.json
python -m scope_v2.asset_audit --manifest data/heldout_v2/manifest.json --out data/heldout_v2/asset_audit.json
python -m pytest -q tests_v2
python -m scope_v2.replication run --manifest data/heldout_v2/manifest.json --stage pilot
python -m scope_v2.replication run --manifest data/heldout_v2/manifest.json --stage all
```

This stage retains a dense persistent cache. It contributes real-model numerical generalization, not systems capacity, latency or persistent-INT4 evidence. Runtime weight conversions are declared separately from original checkpoint dtypes.
