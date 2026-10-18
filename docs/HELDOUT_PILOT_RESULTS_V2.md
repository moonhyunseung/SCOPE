# Held-out native-model pilot v2

Run `2aa605c55475af9d` completed 20 of 800 frozen conditions: five pinned models, two runtime dtypes, one prose and one code source, L512, INT4, group32. These are screening sources. The 112 confirmatory sources have not been used to tune this pilot.

All 20 conditions completed. Each passed 32 exact no-op checks, 16 exact packed/dequantized checks, six first-step isolated/recurrent history checks, and unchanged-cache snapshot checks. The compact blocks contain 240 document-level treatment rows and 40 factorial rows. They are repeated measurements on **two source units**, not 240 independent observations.

The unchanged cached/full dense relative-L2 gate (0.025) passed 13/20 and failed 7/20: Pythia 4/4 pass; OPT 2/4; Qwen1.5B 3/4; Qwen7B 2/4; Qwen14B 2/4. These are native dense discrepancies, preserved separately from the quantization intervention against the same native cached baseline. Completion and intervention control validity do not make a dense-fidelity failure a pass.

No inferential result is claimed from n=1 per domain. Main effects, interaction contrasts, natural next-token NLL changes, KL, top-1 agreement and relative-L2 are provided for inspection; interpretation awaits the frozen source-level confirmation campaign. Runtime dtype conversion and on-disk checkpoint dtype are recorded separately in the asset audit. Cross-dtype comparisons do not isolate arithmetic precision when checkpoint values are converted.

This pilot uses dense resident intervention caches. It establishes neither persistent packed-KV capacity nor decode speed. The separate three-operation minimal BF16 reproducers reproduce the native first-stage shape effects exactly, without establishing the complete cause of all 453 historical final-logit failures.

Private exact input token IDs and weights stay on the execution server. Public input metadata, normalized-content hashes, token-window hashes, pinned checkpoint revisions and audited cached-weight SHA256 values accompany the results. Server publication queue entries remain pending until a GitHub tree/head read-back verifies publication.
