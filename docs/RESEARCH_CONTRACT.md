# Research contract (draft, pre-theorem)

## 1. What is held fixed?

Fix a workload family Ω, hardware H, reference model revision, layer/head specification, attention mask/position/GQA semantics, and a declared numerical comparison contract. Define a logical decode state space as a **disjoint union** S = ⨆_{L≥0} S_L. For a given external input η, the transition is T_{η,L}: S_L → S_{L+1}; equivalently the typed union induces an endomap T_η: S → S. Equality of tensor shapes or membership in the same input/output set is **not** equality of transition outputs.

A snapshot attention read F_{η,L}: (Q,K,V) → Y is not a whole autoregressive state transition. Equal snapshot attention outputs do **not** imply equivalent future behavior after append or recomputation of K/V.

## 2. Reference layers (two different fidelity comparisons)

- **Full/reference model:** T_full; quality must be compared to this when compression schemes themselves are being optimized.
- **Decoded-quantized reference:** for fixed compressed persistent state c and semantic reader D_L(c), T_quant defines the reference transition used for **execution fidelity** comparisons. The precise append/encoding policy must be identical.
- **Hardware executor:** G_{H,η,L} acts on a particular physical memory representation. A faithful executor satisfies a commuting condition of the form D_{L+1}∘G_{H,η,L}(c) ≈ T_quant,η,L(D_L(c)), on the declared reachable states and with a separately specified output-observation condition.
- **Quality gate:** compare T_quant trajectories against T_full trajectories on an independent validation workload. Execution error ≈0 **cannot certify compression quality**.

All diagrams must type-check, including the domain/codomain of the compressed next-state representation. Floating-point ≈ requires a declared norm, denominator, units, tolerance, and evaluation scope. Exact real-arithmetic identities are not finite-precision GPU guarantees.

## 3. Sufficiency / equivalence

For a fixed snapshot operator family F_L, X₁ ~_L X₂ iff ∀f∈F_L, f(X₁)=f(X₂). This is an equivalence relation for a **fixed, state-independent** family. R_L is sufficient iff R_L(X₁)=R_L(X₂) implies X₁~_L X₂, i.e. every f factors through R_L on its image. The quotient provides the coarsest exact *partition*, **not** smallest storage in bytes or cheapest machine representation.

Whole-trajectory equivalence requires an additional transition-compatible condition: X~X′ implies corresponding immediate outputs agree and U_η(X)~U_η(X′) for every admissible input η, with length-typed relations. This is a simulation/bisimulation-style obligation; snapshot sufficiency alone does not provide it.

## 4. Implementation freedoms

The execution plan may recode, transpose, expand selected tiles, stage data on-chip, or temporarily reconstruct dense tensors. None is **prohibited by definition**. All candidates are judged against the same output/next-state contract and full costs: persistent allocation, peak transient workspace, data movement, synchronization, numeric error, latency, and throughput. “Direct”/“closed” algebra is not itself a performance result.

Physical storage: a layout π_L maps logical compressed state C_L to valid memory images M_{π,L}; a reader ρ_{π,L} must satisfy ρ∘π = id on the valid states if a lossless layout is claimed. Compression itself may be lossy relative to the original dense model.

## 5. Optimization objective

For workload ω=(B,L₁,…,L_B), resident tokens N(ω)=Σ_b L_b. Use measured allocator cost M_KV(Θ,ω) (pages, padding, metadata included), not just theoretical N·bytes/token. Require

M_fixed + M_KV(Θ,ω) + W_peak(Θ,ω) ≤ M_HBM,

plus independently specified execution fidelity, end-model quality and decode latency/throughput gates. Study the resident-capacity / decode-latency **Pareto frontier** at fixed workload policies. Synthetic reference tests establish none of these system-level claims.

## 6. Terminology

- **Trace mode:** materializes selected reference intermediates to measure distributions; its speed is not benchmark evidence.
- **Replay/performance mode:** instrumentation disabled, same frozen workload.
- **P0:** validate observation correctness, not speedup.
- **P1:** estimate conditional distributions and rare-case coverage; never treat a sample histogram as a universal invariant.
- **P2/P3:** propose then falsify hardware-specific transformation hypotheses.
