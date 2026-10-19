# Screening progress v2

Dated checkpoint: 2026-10-18 16:27 UTC. Campaign `2aa605c55475af9d` had 180 collected, two unsupported and 618 planned conditions. This is a checkpoint, not a completed confirmation campaign.

Three new sealed screening blocks:

| Model | Planned | Collected | Unsupported | Native cached/full pass | Native cached/full fail |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-1.5B-Instruct | 48 | 48 | 0 | 32 | 16 |
| Pythia-1B | 48 | 48 | 0 | 48 | 0 |
| OPT-1.3B | 48 | 46 | 2 | 16 | 30 |

Each block covers the same **16 screening source units**, eight per domain. Dtypes, parameter probes, models and eight decode steps are repeated measurements. The two unsupported OPT conditions use group128 with native head dimension64; they are not execution failures. All collected conditions passed the exact no-op, packing, first-step history and cache snapshot controls. Compact result hashes, row counts and denominators were independently verified after transfer.

Native cached/full relative-L2 uses the unchanged 0.025 gate. Its failures remain failures. Factorial intervention errors are measured against the same native cached trajectory; this controlled contrast does not turn a failing cached trajectory into a faithful full-reference reproduction. Screening effects and intervals are exploratory. In particular, OPT's substantial native discrepancies limit any precision-independent or full-reference quality interpretation. The separate held-out confirmation population remains frozen; these screening outcomes do not tune its source selection, dtypes, endpoints or primary contrasts.

Pilot and screening blocks overlap by design. Do not add their collected counts as unique conditions. Their common registry identifies unique planned specifications and separate terminal statuses. The 128-source population contains 16 screening and 112 confirmatory sources; n is not multiplied by the five models.

The added resident-KV and native joint-distribution modules passed **16 CPU tests**, including exact native logits/cache observation on a small actual Qwen model. Their H200 execution is queued behind the current campaign. CPU tests establish implementation controls, not H200 capacity, latency or successful packed-kernel compilation. Performance/capacity remains unestablished at this checkpoint.

Server noninteractive Git push is unavailable. Completed blocks remain durably sealed and transactionally queued; active-session publication uses the connected GitHub app and is marked published only after remote blob/main-ref verification. Future unattended blocks can remain PENDING while GPU work continues.
