# ACL review round 1: additive execution contract

Review baseline: `8619a74c1c8b2648e41ffcc75ab9131a6a649353`. The execution identities of earlier runs remain unchanged. This contract implements the received research review without changing the frozen 800-condition campaign, source populations, primary endpoints, dtype/length settings or the .025 numerical gate.

1. Complete the original campaign and retain all terminal statuses. An additive CPU audit reads raw sealed step files, validates planned source IDs and controls, reproduces the eight original primary bootstrap contrasts (20,000 source draws,99.375% intervals,n56/domain), and adds descriptive source maxima/tail counts. Exploratory cross-model effect contrasts resample shared source IDs jointly. Missing inputs make a contrast non-estimable; no post-hoc dense-pass-only primary subset is permitted. A dense failure is retained separately from quantization effects against native cached logits.
2. Reuse only the three sealed BF16 minimal reproducers. After the existing GPU queue has terminated, allow at most30 additional GPU minutes. Vary contraction row count at fixed operands, check native cached/full proxies exactly, compare true/false reduction flags **bitwise**, and collect optional CUDA profiler kernel names. Same-weight FP32 is a diagnostic control. Kernel names are not verified cuBLAS algorithm IDs; unobserved IDs remain UNKNOWN. Do not replay all445 previously untraced failures or claim an exhaustive cause of453 failures.
3. Execute the unchanged resident6 and native-distribution20 pilots already queued. Publish failures as well as successes. No large frontier, new precision, kernel-family or source-population expansion before reviewing these results.

Interpretation corrections, without retrospective protocol edits:

- The current static cache is a **page-rounded contiguous allocation**, not a page-table serving allocator or paged-attention system. It rounds capacity to128 slots; no serving-scheduler claim is supported.
- Same-representation execution Gate D is distinct from original-model representation/quality fidelity. The latter is not measured in the current resident pilot. Both would be required for a quality-constrained system-capacity claim; a future quality gate must be an additive block.
- A physical HBM snapshot is not an independently measured physical-HBM high-water mark. Allocator peaks and physical snapshots must be labelled separately.
- Channelwise cancellation and actual operand-format residual measurements are potential follow-ups after existing distribution screens are observed, not authorization to alter the frozen distribution pilot.

Completed screening results later added at `ab5adc2dbd50edfa70966324e9b4fb1cdaab2f79` do not modify the review's frozen confirmation contract. Timing trials are repeats of one homogeneous workload, not independent source samples. No systems-performance or maximum-capacity claim is established by this contract or CPU tests.
