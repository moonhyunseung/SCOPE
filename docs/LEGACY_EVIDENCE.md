# Historical SCOPE — immutable evidence map

The entire pre-pivot repository root is mounted at [`legacy/`](../legacy/) by referencing its original Git tree, without rewriting its file contents. The immutable pre-pivot commit is [`7b3a1ca7b8c890b9da5b12a7ab5ad98193eb874f`](https://github.com/moonhyunseung/SCOPE/commit/7b3a1ca7b8c890b9da5b12a7ab5ad98193eb874f); preservation branch: [`legacy/scope-v4-2026-10-19`](https://github.com/moonhyunseung/SCOPE/tree/legacy/scope-v4-2026-10-19). Its original `scope/`, `scope_v2/`, `docs/`, `data/`, `results/`, `scripts/`, `tests/`, `tests_v2/`, `pyproject.toml` and historical CI workflows are still present there.

## Key verified historical outcomes

| Campaign | Evidence | Limits |
| --- | --- | --- |
| Population v1 | 512 source units, 1,152 collected conditions; K-only exceeded V-only in four preregistered contrasts; 453/1,024 BF16 cached/full gate failures | One population model; no serving capacity claim |
| Frozen multi-model continuation | 800 planned, 798 collected + 2 unsupported; 560/560 confirmatory; all eight Pythia/OPT K-minus-V intervals positive; dense cached/full failed 338/560 | 112 shared sources total, 56/domain; primary compares to native cached, not original full model quality |
| Native BF16 diagnosis | 48 first-divergence traces and three shape-sensitive minimal reproductions | Internal algorithm identity and full error attribution unresolved |
| Resident pilot | One B1/L512 source, 6 paths; INT4 payload 5,734,400 bytes vs static dense 18,350,080; chunked median 117.31 ms vs dynamic dense 54.46 ms; Triton/SDPA fidelity-gate failures | 4 decode steps, no serving frontier; quality of representation initially untested |
| Attention distribution | 20 cells over two independent sources; preset narrow-W and concentration screens not met | Too few independent source units for general claims |
| Role/bitwidth screens | K/V INT4/INT8 group32 role variants tested on two sources, all compression candidates ineligible under frozen 0.025 screen | B1/L512, 4 teacher-forced steps, one model; not a general quantization impossibility |

Historical contract was stricter about operator-logit L2 than a production-quality study may require. **Keep that original criterion unchanged in archived reporting.** New research must independently pre-register serving/behavioral quality criteria if studying approximation.

## Entry points

- [Population report](../legacy/docs/POPULATION_RESULTS_20261018.md)
- [Multi-model confirmation and actual resident measurements](../legacy/docs/CONTINUATION_TERMINAL_EVIDENCE_V2.md)
- [BF16 mechanism report](../legacy/docs/BF16_MECHANISM_RESULTS.md)
- [Execution and representation difference](../legacy/docs/ACL_R2_RESULTS.md)
- [Role-based INT4 evidence](../legacy/docs/ROLE_FIDELITY_V3_RESULTS.md)
- [Terminal INT8 negative result](../legacy/docs/BITWIDTH_RESCUE_V4_RESULTS.md)

The archive is not deprecated because its results are false. It is retired from **active project layout** because the new question is broader than quantization: where does industrial attention serving waste cost, and what is the smallest verifiably valuable replacement?
