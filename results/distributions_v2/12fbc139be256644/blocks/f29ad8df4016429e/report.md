# Native attention distribution block

{
  "status": "COLLECTED",
  "spec": {
    "model": "opt1_3b",
    "document_id": "95459423d3b8f7c779e3",
    "domain": "code",
    "length": 512
  },
  "model": {
    "key": "opt1_3b",
    "id": "facebook/opt-1.3b",
    "revision": "3f5c25d0bc631cb57ac65913f76e22c2dfb61d62",
    "model_type": "opt",
    "num_attention_heads": 32,
    "num_key_value_heads": 32,
    "max_position_embeddings": 2048,
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
  "token_window_sha256": "cb647a124ce7affcdb5006b7da8b6df4f1a12bb333bff665ba1d8e3da5a46e8e",
  "native_controls_exact": [
    true,
    true
  ],
  "original_source_count": 1,
  "head_observations": 192,
  "source_units": 1,
  "publication": "PENDING",
  "prespecified_screens": {
    "narrow_w_fraction": 0.4791666666666667,
    "concentration_fraction": 0.4322916666666667,
    "severe_cancellation_fraction": 0.0,
    "interpretation": "Within one source/model/length cell; descriptive repeated head/layer/step observations, no source-level CI.",
    "narrow_w_pass": false,
    "concentration_pass": false
  }
}

Head/layer/step observations are repeated measures, not independent sources. Native and reconstructed variables are explicitly labelled. Fixed bounded coordinate samples and selected screening sources limit inference. No speed or quality benefit follows from a distribution alone.
