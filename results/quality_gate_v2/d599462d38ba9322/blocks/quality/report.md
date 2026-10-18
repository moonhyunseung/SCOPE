# ACL round2 additive diagnostic

{
  "status": "REPRESENTATION_FIDELITY_FAIL",
  "stage": "quality_gate_v2",
  "baseline_status": "BASELINE_DIVERGENCE",
  "representation_max_relative_l2": 0.9207611394367784,
  "baseline_max_relative_l2": 0.02724920259505381,
  "execution_max_relative_l2": 0.0,
  "independent_cache_parity_exact": true,
  "replays_exact": true,
  "independent_source_units": 1,
  "repeated_steps": 4,
  "replays": 2,
  "execution_gate": 0.025,
  "representation_screen": 0.025,
  "scope": "Local logit fidelity screen and natural next-token NLL only. No corpus perplexity/task accuracy claim. Distances are not additive error components.",
  "runtime": {
    "python": "3.12.3",
    "torch": "2.14.1+cu130",
    "transformers": "5.19.0",
    "cuda": "13.0",
    "gpu": "NVIDIA H200",
    "tf32": false,
    "bf16_reduction_flag": true
  },
  "source_sha256": "f422b649a112d1a7573cfb6363dd03fcf07ff24f52cf4ddaa5034c457b3ecba9",
  "preregistration_sha256": "d599462d38ba9322f39636844f3b35baecd792d0e2f6a02d5fd43f698266c617",
  "publication": "PENDING"
}

Raw logits, token IDs and activation tensors were not saved to this block. Historical sources, result identities, original failures and .025 Gate D are preserved.
