# ACL round2 additive diagnostic

{
  "status": "COLLECTED",
  "stage": "operand_screen_v2",
  "source_units": 2,
  "selected_head_layer_step_units": 36,
  "format_rows": 108,
  "replays": 2,
  "observer_controls_exact": true,
  "residuals": {
    "fp32": {
      "max_w_relative_l2": 0.0,
      "max_pv_relative_l2": 0.001583392470494166,
      "max_pv_vs_factorized_fp32_relative_l2": 0.0
    },
    "tf32_rne_single": {
      "max_w_relative_l2": 0.0003507116223988912,
      "max_pv_relative_l2": 0.0015254107577216875,
      "max_pv_vs_factorized_fp32_relative_l2": 0.0002971872678558857
    },
    "tf32_rne_hi_plus_lo": {
      "max_w_relative_l2": 8.024154903000968e-08,
      "max_pv_relative_l2": 0.0015833501225887307,
      "max_pv_vs_factorized_fp32_relative_l2": 2.5591641069263777e-07
    }
  },
  "scope": "Two screening sources only. Heads/layers/steps are repeated observations. CPU FP32 format emulation with native P and actual INT4-dequantized V. Dequantization rounding gap retained; not MMA throughput or model-quality evidence.",
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
