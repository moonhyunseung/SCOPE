# Native attention distribution block

{
  "status": "COLLECTED",
  "spec": {
    "model": "qwen14b",
    "document_id": "7190bc75f0028b8ef4e2",
    "domain": "prose",
    "length": 512
  },
  "model": {
    "key": "qwen14b",
    "id": "Qwen/Qwen2.5-14B-Instruct",
    "revision": "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8",
    "model_type": "qwen2",
    "num_attention_heads": 40,
    "num_key_value_heads": 8,
    "max_position_embeddings": 32768,
    "attention_backend": "eager"
  },
  "code_sha256": "3795f87fc49d572fc26e6bbcf1a43aeba756052a5ac91e0cb0ae359ed0ec933e",
  "input_manifest_sha256": "fbdd626cb95eac1992fbf76406d871f726f15cf0aa48fd3bf5817a46a6038ec5",
  "runtime": {
    "python": "3.12.3",
    "torch": "2.14.1+cu130",
    "transformers": "5.19.0",
    "cuda": "13.0",
    "attention_backend": "native eager",
    "dtype": "bfloat16",
    "tf32": false,
    "deterministic_algorithms": true
  },
  "sampling": "First/middle/last layers, all query heads, decode steps0/7; marginal deterministic evenly spaced flatten coordinates up to8192; P-S joints use all valid token/group coordinates; no random outcome selection.",
  "token_window_sha256": "ef4c98a279004658f531ded6a09c0573b766ec07cf08d5e6e27dfe562817b858",
  "native_controls_exact": [
    true,
    true
  ],
  "original_source_count": 1,
  "head_observations": 240,
  "source_units": 1,
  "publication": "PENDING",
  "prespecified_screens": {
    "narrow_w_fraction": 0.44583333333333336,
    "concentration_fraction": 0.23333333333333334,
    "severe_cancellation_fraction": 0.0,
    "interpretation": "Within one source/model/length cell; descriptive repeated head/layer/step observations, no source-level CI.",
    "narrow_w_pass": false,
    "concentration_pass": false
  }
}

Head/layer/step observations are repeated measures, not independent sources. Native and reconstructed variables are explicitly labelled. Fixed bounded coordinate samples and selected screening sources limit inference. No speed or quality benefit follows from a distribution alone.
