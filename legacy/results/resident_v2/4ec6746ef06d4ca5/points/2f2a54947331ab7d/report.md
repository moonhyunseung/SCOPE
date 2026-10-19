# Actual resident-KV full-model point

{
  "status": "COLLECTED",
  "spec": {
    "document_id": "7190bc75f0028b8ef4e2",
    "domain": "prose",
    "length": 512,
    "batch": 1,
    "method": "int4_global",
    "steps": 4
  },
  "source_sha256": "4f434b13600b00ff63250cf0bcb57877de245f6e4aad8f3cd81594e6543af340",
  "model": {
    "key": "qwen1_5b",
    "id": "Qwen/Qwen2.5-1.5B-Instruct",
    "revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
    "model_type": "qwen2",
    "num_attention_heads": 12,
    "num_key_value_heads": 2,
    "max_position_embeddings": 32768,
    "attention_backend": "eager"
  },
  "input_manifest_sha256": "fbdd626cb95eac1992fbf76406d871f726f15cf0aa48fd3bf5817a46a6038ec5",
  "gate_d": 0.025,
  "representation": "symmetric group32 INT4 codes plus FP32 scales; BF16 reconstructions",
  "workload": "Physically distinct copies of one real prefix per batch; homogeneous requests, no independent source-n claim. Teacher-forced four-token decode; excludes prefill and scheduling.",
  "runtime": {
    "python": "3.12.3",
    "torch": "2.14.1+cu130",
    "transformers": "5.19.0",
    "triton": "3.8.0",
    "cuda": "13.0",
    "gpu": "NVIDIA H200",
    "deterministic_algorithms": true,
    "tf32": false,
    "bf16_reduced_precision_reduction": true
  },
  "model_storage_bytes": 3087429120,
  "token_window_sha256": "ef4c98a279004658f531ded6a09c0573b766ec07cf08d5e6e27dfe562817b858",
  "model_runtime_baseline": {
    "allocated_bytes": 3121901568,
    "reserved_bytes": 3319791616,
    "peak_allocated_bytes": 3179843584,
    "peak_reserved_bytes": 3376414720,
    "active_block_bytes": 3121901568,
    "active_requested_bytes": 3120983552,
    "active_block_rounding_bytes": 918016,
    "reserved_minus_active_bytes": 197890048,
    "inactive_split_bytes": 197890048,
    "physical_used_bytes": 4006608896,
    "physical_free_bytes": 146104385536,
    "physical_minus_allocator_reserved_bytes": 686817280,
    "host_allocator_metadata_bytes": null
  },
  "budget_prediction": {
    "payload_bytes": 5734400,
    "workspace_upper_estimate_bytes": 542332416,
    "available_device_budget_bytes": 98858827776,
    "host_reference_bytes": 1215488
  },
  "cache_materialization_wall_s": 0.03057929500937462,
  "resident_initial": {
    "storage_bytes": 5734400,
    "layers": [
      {
        "layer": 0,
        "valid_tokens": 512,
        "fields": {
          "pk": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "pv": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "sk": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          },
          "sv": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          }
        }
      },
      {
        "layer": 1,
        "valid_tokens": 512,
        "fields": {
          "pk": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "pv": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "sk": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          },
          "sv": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          }
        }
      },
      {
        "layer": 2,
        "valid_tokens": 512,
        "fields": {
          "pk": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "pv": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "sk": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          },
          "sv": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          }
        }
      },
      {
        "layer": 3,
        "valid_tokens": 512,
        "fields": {
          "pk": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "pv": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "sk": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          },
          "sv": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          }
        }
      },
      {
        "layer": 4,
        "valid_tokens": 512,
        "fields": {
          "pk": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "pv": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "sk": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          },
          "sv": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          }
        }
      },
      {
        "layer": 5,
        "valid_tokens": 512,
        "fields": {
          "pk": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "pv": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "sk": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          },
          "sv": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          }
        }
      },
      {
        "layer": 6,
        "valid_tokens": 512,
        "fields": {
          "pk": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "pv": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "sk": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          },
          "sv": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          }
        }
      },
      {
        "layer": 7,
        "valid_tokens": 512,
        "fields": {
          "pk": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "pv": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "sk": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          },
          "sv": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          }
        }
      },
      {
        "layer": 8,
        "valid_tokens": 512,
        "fields": {
          "pk": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "pv": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "sk": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          },
          "sv": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          }
        }
      },
      {
        "layer": 9,
        "valid_tokens": 512,
        "fields": {
          "pk": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "pv": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "sk": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          },
          "sv": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          }
        }
      },
      {
        "layer": 10,
        "valid_tokens": 512,
        "fields": {
          "pk": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "pv": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "sk": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          },
          "sv": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          }
        }
      },
      {
        "layer": 11,
        "valid_tokens": 512,
        "fields": {
          "pk": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "pv": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "sk": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          },
          "sv": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          }
        }
      },
      {
        "layer": 12,
        "valid_tokens": 512,
        "fields": {
          "pk": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "pv": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "sk": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          },
          "sv": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          }
        }
      },
      {
        "layer": 13,
        "valid_tokens": 512,
        "fields": {
          "pk": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "pv": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "sk": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          },
          "sv": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          }
        }
      },
      {
        "layer": 14,
        "valid_tokens": 512,
        "fields": {
          "pk": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "pv": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "sk": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          },
          "sv": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          }
        }
      },
      {
        "layer": 15,
        "valid_tokens": 512,
        "fields": {
          "pk": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "pv": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "sk": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          },
          "sv": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          }
        }
      },
      {
        "layer": 16,
        "valid_tokens": 512,
        "fields": {
          "pk": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "pv": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "sk": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          },
          "sv": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          }
        }
      },
      {
        "layer": 17,
        "valid_tokens": 512,
        "fields": {
          "pk": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "pv": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "sk": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          },
          "sv": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          }
        }
      },
      {
        "layer": 18,
        "valid_tokens": 512,
        "fields": {
          "pk": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "pv": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "sk": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          },
          "sv": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          }
        }
      },
      {
        "layer": 19,
        "valid_tokens": 512,
        "fields": {
          "pk": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "pv": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "sk": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          },
          "sv": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          }
        }
      },
      {
        "layer": 20,
        "valid_tokens": 512,
        "fields": {
          "pk": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "pv": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "sk": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          },
          "sv": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          }
        }
      },
      {
        "layer": 21,
        "valid_tokens": 512,
        "fields": {
          "pk": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "pv": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "sk": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          },
          "sv": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          }
        }
      },
      {
        "layer": 22,
        "valid_tokens": 512,
        "fields": {
          "pk": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "pv": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "sk": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          },
          "sv": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          }
        }
      },
      {
        "layer": 23,
        "valid_tokens": 512,
        "fields": {
          "pk": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "pv": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "sk": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          },
          "sv": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          }
        }
      },
      {
        "layer": 24,
        "valid_tokens": 512,
        "fields": {
          "pk": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "pv": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "sk": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          },
          "sv": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          }
        }
      },
      {
        "layer": 25,
        "valid_tokens": 512,
        "fields": {
          "pk": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "pv": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "sk": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          },
          "sv": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          }
        }
      },
      {
        "layer": 26,
        "valid_tokens": 512,
        "fields": {
          "pk": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "pv": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "sk": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          },
          "sv": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          }
        }
      },
      {
        "layer": 27,
        "valid_tokens": 512,
        "fields": {
          "pk": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "pv": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "sk": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          },
          "sv": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          }
        }
      }
    ],
    "unique_storages": 112,
    "dense_persistent_fields": 0
  },
  "memory_initial": {
    "allocated_bytes": 3127638016,
    "reserved_bytes": 3319791616,
    "peak_allocated_bytes": 3179843584,
    "peak_reserved_bytes": 3376414720,
    "active_block_bytes": 3127638016,
    "active_requested_bytes": 3126717984,
    "active_block_rounding_bytes": 920032,
    "reserved_minus_active_bytes": 192153600,
    "inactive_split_bytes": 192153600,
    "physical_used_bytes": 4006608896,
    "physical_free_bytes": 146104385536,
    "physical_minus_allocator_reserved_bytes": 686817280,
    "host_allocator_metadata_bytes": null
  },
  "gate_d_steps": [
    {
      "max_request_relative_l2": 0.0,
      "mean_request_relative_l2": 0.0,
      "max_abs": 0.0,
      "top1_agreement": 1.0
    },
    {
      "max_request_relative_l2": 0.0,
      "mean_request_relative_l2": 0.0,
      "max_abs": 0.0,
      "top1_agreement": 1.0
    },
    {
      "max_request_relative_l2": 0.0,
      "mean_request_relative_l2": 0.0,
      "max_abs": 0.0,
      "top1_agreement": 1.0
    },
    {
      "max_request_relative_l2": 0.0,
      "mean_request_relative_l2": 0.0,
      "max_abs": 0.0,
      "top1_agreement": 1.0
    }
  ],
  "memory_after_gate": {
    "allocated_bytes": 3127638016,
    "reserved_bytes": 3321888768,
    "peak_allocated_bytes": 3179843584,
    "peak_reserved_bytes": 3376414720,
    "active_block_bytes": 3127638016,
    "active_requested_bytes": 3126717984,
    "active_block_rounding_bytes": 920032,
    "reserved_minus_active_bytes": 194250752,
    "inactive_split_bytes": 192153600,
    "physical_used_bytes": 4008706048,
    "physical_free_bytes": 146102288384,
    "physical_minus_allocator_reserved_bytes": 686817280,
    "host_allocator_metadata_bytes": null
  },
  "resident_timed_start": {
    "storage_bytes": 5734400,
    "layers": [
      {
        "layer": 0,
        "valid_tokens": 512,
        "fields": {
          "pk": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "pv": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "sk": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          },
          "sv": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          }
        }
      },
      {
        "layer": 1,
        "valid_tokens": 512,
        "fields": {
          "pk": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "pv": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "sk": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          },
          "sv": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          }
        }
      },
      {
        "layer": 2,
        "valid_tokens": 512,
        "fields": {
          "pk": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "pv": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "sk": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          },
          "sv": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          }
        }
      },
      {
        "layer": 3,
        "valid_tokens": 512,
        "fields": {
          "pk": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "pv": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "sk": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          },
          "sv": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          }
        }
      },
      {
        "layer": 4,
        "valid_tokens": 512,
        "fields": {
          "pk": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "pv": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "sk": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          },
          "sv": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          }
        }
      },
      {
        "layer": 5,
        "valid_tokens": 512,
        "fields": {
          "pk": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "pv": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "sk": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          },
          "sv": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          }
        }
      },
      {
        "layer": 6,
        "valid_tokens": 512,
        "fields": {
          "pk": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "pv": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "sk": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          },
          "sv": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          }
        }
      },
      {
        "layer": 7,
        "valid_tokens": 512,
        "fields": {
          "pk": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "pv": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "sk": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          },
          "sv": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          }
        }
      },
      {
        "layer": 8,
        "valid_tokens": 512,
        "fields": {
          "pk": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "pv": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "sk": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          },
          "sv": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          }
        }
      },
      {
        "layer": 9,
        "valid_tokens": 512,
        "fields": {
          "pk": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "pv": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "sk": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          },
          "sv": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          }
        }
      },
      {
        "layer": 10,
        "valid_tokens": 512,
        "fields": {
          "pk": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "pv": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "sk": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          },
          "sv": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          }
        }
      },
      {
        "layer": 11,
        "valid_tokens": 512,
        "fields": {
          "pk": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "pv": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "sk": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          },
          "sv": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          }
        }
      },
      {
        "layer": 12,
        "valid_tokens": 512,
        "fields": {
          "pk": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "pv": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "sk": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          },
          "sv": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          }
        }
      },
      {
        "layer": 13,
        "valid_tokens": 512,
        "fields": {
          "pk": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "pv": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "sk": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          },
          "sv": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          }
        }
      },
      {
        "layer": 14,
        "valid_tokens": 512,
        "fields": {
          "pk": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "pv": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "sk": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          },
          "sv": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          }
        }
      },
      {
        "layer": 15,
        "valid_tokens": 512,
        "fields": {
          "pk": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "pv": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "sk": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          },
          "sv": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          }
        }
      },
      {
        "layer": 16,
        "valid_tokens": 512,
        "fields": {
          "pk": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "pv": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "sk": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          },
          "sv": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          }
        }
      },
      {
        "layer": 17,
        "valid_tokens": 512,
        "fields": {
          "pk": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "pv": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "sk": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          },
          "sv": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          }
        }
      },
      {
        "layer": 18,
        "valid_tokens": 512,
        "fields": {
          "pk": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "pv": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "sk": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          },
          "sv": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          }
        }
      },
      {
        "layer": 19,
        "valid_tokens": 512,
        "fields": {
          "pk": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "pv": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "sk": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          },
          "sv": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          }
        }
      },
      {
        "layer": 20,
        "valid_tokens": 512,
        "fields": {
          "pk": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "pv": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "sk": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          },
          "sv": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          }
        }
      },
      {
        "layer": 21,
        "valid_tokens": 512,
        "fields": {
          "pk": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "pv": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "sk": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          },
          "sv": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          }
        }
      },
      {
        "layer": 22,
        "valid_tokens": 512,
        "fields": {
          "pk": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "pv": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "sk": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          },
          "sv": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          }
        }
      },
      {
        "layer": 23,
        "valid_tokens": 512,
        "fields": {
          "pk": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "pv": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "sk": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          },
          "sv": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          }
        }
      },
      {
        "layer": 24,
        "valid_tokens": 512,
        "fields": {
          "pk": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "pv": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "sk": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          },
          "sv": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          }
        }
      },
      {
        "layer": 25,
        "valid_tokens": 512,
        "fields": {
          "pk": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "pv": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "sk": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          },
          "sv": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          }
        }
      },
      {
        "layer": 26,
        "valid_tokens": 512,
        "fields": {
          "pk": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "pv": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "sk": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          },
          "sv": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          }
        }
      },
      {
        "layer": 27,
        "valid_tokens": 512,
        "fields": {
          "pk": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "pv": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 65536,
            "padding_bytes": 16384
          },
          "sk": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          },
          "sv": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16384,
            "padding_bytes": 4096
          }
        }
      }
    ],
    "unique_storages": 112,
    "dense_persistent_fields": 0
  },
  "timing_repetitions": [
    {
      "cuda_ms": 179.19203186035156,
      "wall_s": 0.17931172996759415,
      "aggregate_tokens_per_s": 22.322421139335543,
      "wall_tokens_per_s": 22.30751998613194
    },
    {
      "cuda_ms": 168.05715942382812,
      "wall_s": 0.16809435840696096,
      "aggregate_tokens_per_s": 23.801425739395526,
      "wall_tokens_per_s": 23.79615852612907
    },
    {
      "cuda_ms": 170.21023559570312,
      "wall_s": 0.1702538151293993,
      "aggregate_tokens_per_s": 23.50034935326168,
      "wall_tokens_per_s": 23.494334015128235
    },
    {
      "cuda_ms": 172.44863891601562,
      "wall_s": 0.17247870285063982,
      "aggregate_tokens_per_s": 23.195312094913337,
      "wall_tokens_per_s": 23.191269031422692
    },
    {
      "cuda_ms": 175.6623077392578,
      "wall_s": 0.17569463327527046,
      "aggregate_tokens_per_s": 22.77096351220292,
      "wall_tokens_per_s": 22.766773949964538
    },
    {
      "cuda_ms": 170.8748779296875,
      "wall_s": 0.17090033367276192,
      "aggregate_tokens_per_s": 23.408941375489622,
      "wall_tokens_per_s": 23.40545459471809
    },
    {
      "cuda_ms": 167.96954345703125,
      "wall_s": 0.16799198184162378,
      "aggregate_tokens_per_s": 23.813840995663902,
      "wall_tokens_per_s": 23.810660224075708
    },
    {
      "cuda_ms": 170.24044799804688,
      "wall_s": 0.1702673127874732,
      "aggregate_tokens_per_s": 23.49617876972393,
      "wall_tokens_per_s": 23.492471540869268
    },
    {
      "cuda_ms": 175.1294403076172,
      "wall_s": 0.17516664694994688,
      "aggregate_tokens_per_s": 22.840248863777255,
      "wall_tokens_per_s": 22.83539743238325
    },
    {
      "cuda_ms": 171.11859130859375,
      "wall_s": 0.17115680500864983,
      "aggregate_tokens_per_s": 23.375601501922347,
      "wall_tokens_per_s": 23.370382496903062
    }
  ],
  "memory_timed": {
    "allocated_bytes": 3127638016,
    "reserved_bytes": 3321888768,
    "peak_allocated_bytes": 3131914752,
    "peak_reserved_bytes": 3321888768,
    "active_block_bytes": 3127638016,
    "active_requested_bytes": 3126717984,
    "active_block_rounding_bytes": 920032,
    "reserved_minus_active_bytes": 194250752,
    "inactive_split_bytes": 192153600,
    "physical_used_bytes": 4008706048,
    "physical_free_bytes": 146102288384,
    "physical_minus_allocator_reserved_bytes": 686817280,
    "host_allocator_metadata_bytes": null
  },
  "resident_final": {
    "storage_bytes": 5734400,
    "layers": [
      {
        "layer": 0,
        "valid_tokens": 516,
        "fields": {
          "pk": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 66048,
            "padding_bytes": 15872
          },
          "pv": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 66048,
            "padding_bytes": 15872
          },
          "sk": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16512,
            "padding_bytes": 3968
          },
          "sv": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16512,
            "padding_bytes": 3968
          }
        }
      },
      {
        "layer": 1,
        "valid_tokens": 516,
        "fields": {
          "pk": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 66048,
            "padding_bytes": 15872
          },
          "pv": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 66048,
            "padding_bytes": 15872
          },
          "sk": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16512,
            "padding_bytes": 3968
          },
          "sv": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16512,
            "padding_bytes": 3968
          }
        }
      },
      {
        "layer": 2,
        "valid_tokens": 516,
        "fields": {
          "pk": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 66048,
            "padding_bytes": 15872
          },
          "pv": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 66048,
            "padding_bytes": 15872
          },
          "sk": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16512,
            "padding_bytes": 3968
          },
          "sv": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16512,
            "padding_bytes": 3968
          }
        }
      },
      {
        "layer": 3,
        "valid_tokens": 516,
        "fields": {
          "pk": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 66048,
            "padding_bytes": 15872
          },
          "pv": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 66048,
            "padding_bytes": 15872
          },
          "sk": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16512,
            "padding_bytes": 3968
          },
          "sv": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16512,
            "padding_bytes": 3968
          }
        }
      },
      {
        "layer": 4,
        "valid_tokens": 516,
        "fields": {
          "pk": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 66048,
            "padding_bytes": 15872
          },
          "pv": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 66048,
            "padding_bytes": 15872
          },
          "sk": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16512,
            "padding_bytes": 3968
          },
          "sv": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16512,
            "padding_bytes": 3968
          }
        }
      },
      {
        "layer": 5,
        "valid_tokens": 516,
        "fields": {
          "pk": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 66048,
            "padding_bytes": 15872
          },
          "pv": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 66048,
            "padding_bytes": 15872
          },
          "sk": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16512,
            "padding_bytes": 3968
          },
          "sv": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16512,
            "padding_bytes": 3968
          }
        }
      },
      {
        "layer": 6,
        "valid_tokens": 516,
        "fields": {
          "pk": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 66048,
            "padding_bytes": 15872
          },
          "pv": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 66048,
            "padding_bytes": 15872
          },
          "sk": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16512,
            "padding_bytes": 3968
          },
          "sv": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16512,
            "padding_bytes": 3968
          }
        }
      },
      {
        "layer": 7,
        "valid_tokens": 516,
        "fields": {
          "pk": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 66048,
            "padding_bytes": 15872
          },
          "pv": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 66048,
            "padding_bytes": 15872
          },
          "sk": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16512,
            "padding_bytes": 3968
          },
          "sv": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16512,
            "padding_bytes": 3968
          }
        }
      },
      {
        "layer": 8,
        "valid_tokens": 516,
        "fields": {
          "pk": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 66048,
            "padding_bytes": 15872
          },
          "pv": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 66048,
            "padding_bytes": 15872
          },
          "sk": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16512,
            "padding_bytes": 3968
          },
          "sv": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16512,
            "padding_bytes": 3968
          }
        }
      },
      {
        "layer": 9,
        "valid_tokens": 516,
        "fields": {
          "pk": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 66048,
            "padding_bytes": 15872
          },
          "pv": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 66048,
            "padding_bytes": 15872
          },
          "sk": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16512,
            "padding_bytes": 3968
          },
          "sv": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16512,
            "padding_bytes": 3968
          }
        }
      },
      {
        "layer": 10,
        "valid_tokens": 516,
        "fields": {
          "pk": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 66048,
            "padding_bytes": 15872
          },
          "pv": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 66048,
            "padding_bytes": 15872
          },
          "sk": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16512,
            "padding_bytes": 3968
          },
          "sv": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16512,
            "padding_bytes": 3968
          }
        }
      },
      {
        "layer": 11,
        "valid_tokens": 516,
        "fields": {
          "pk": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 66048,
            "padding_bytes": 15872
          },
          "pv": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 66048,
            "padding_bytes": 15872
          },
          "sk": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16512,
            "padding_bytes": 3968
          },
          "sv": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16512,
            "padding_bytes": 3968
          }
        }
      },
      {
        "layer": 12,
        "valid_tokens": 516,
        "fields": {
          "pk": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 66048,
            "padding_bytes": 15872
          },
          "pv": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 66048,
            "padding_bytes": 15872
          },
          "sk": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16512,
            "padding_bytes": 3968
          },
          "sv": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16512,
            "padding_bytes": 3968
          }
        }
      },
      {
        "layer": 13,
        "valid_tokens": 516,
        "fields": {
          "pk": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 66048,
            "padding_bytes": 15872
          },
          "pv": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 66048,
            "padding_bytes": 15872
          },
          "sk": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16512,
            "padding_bytes": 3968
          },
          "sv": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16512,
            "padding_bytes": 3968
          }
        }
      },
      {
        "layer": 14,
        "valid_tokens": 516,
        "fields": {
          "pk": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 66048,
            "padding_bytes": 15872
          },
          "pv": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 66048,
            "padding_bytes": 15872
          },
          "sk": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16512,
            "padding_bytes": 3968
          },
          "sv": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16512,
            "padding_bytes": 3968
          }
        }
      },
      {
        "layer": 15,
        "valid_tokens": 516,
        "fields": {
          "pk": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 66048,
            "padding_bytes": 15872
          },
          "pv": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 66048,
            "padding_bytes": 15872
          },
          "sk": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16512,
            "padding_bytes": 3968
          },
          "sv": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16512,
            "padding_bytes": 3968
          }
        }
      },
      {
        "layer": 16,
        "valid_tokens": 516,
        "fields": {
          "pk": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 66048,
            "padding_bytes": 15872
          },
          "pv": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 66048,
            "padding_bytes": 15872
          },
          "sk": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16512,
            "padding_bytes": 3968
          },
          "sv": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16512,
            "padding_bytes": 3968
          }
        }
      },
      {
        "layer": 17,
        "valid_tokens": 516,
        "fields": {
          "pk": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 66048,
            "padding_bytes": 15872
          },
          "pv": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 66048,
            "padding_bytes": 15872
          },
          "sk": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16512,
            "padding_bytes": 3968
          },
          "sv": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16512,
            "padding_bytes": 3968
          }
        }
      },
      {
        "layer": 18,
        "valid_tokens": 516,
        "fields": {
          "pk": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 66048,
            "padding_bytes": 15872
          },
          "pv": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 66048,
            "padding_bytes": 15872
          },
          "sk": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16512,
            "padding_bytes": 3968
          },
          "sv": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16512,
            "padding_bytes": 3968
          }
        }
      },
      {
        "layer": 19,
        "valid_tokens": 516,
        "fields": {
          "pk": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 66048,
            "padding_bytes": 15872
          },
          "pv": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 66048,
            "padding_bytes": 15872
          },
          "sk": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16512,
            "padding_bytes": 3968
          },
          "sv": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16512,
            "padding_bytes": 3968
          }
        }
      },
      {
        "layer": 20,
        "valid_tokens": 516,
        "fields": {
          "pk": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 66048,
            "padding_bytes": 15872
          },
          "pv": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 66048,
            "padding_bytes": 15872
          },
          "sk": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16512,
            "padding_bytes": 3968
          },
          "sv": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16512,
            "padding_bytes": 3968
          }
        }
      },
      {
        "layer": 21,
        "valid_tokens": 516,
        "fields": {
          "pk": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 66048,
            "padding_bytes": 15872
          },
          "pv": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 66048,
            "padding_bytes": 15872
          },
          "sk": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16512,
            "padding_bytes": 3968
          },
          "sv": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16512,
            "padding_bytes": 3968
          }
        }
      },
      {
        "layer": 22,
        "valid_tokens": 516,
        "fields": {
          "pk": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 66048,
            "padding_bytes": 15872
          },
          "pv": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 66048,
            "padding_bytes": 15872
          },
          "sk": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16512,
            "padding_bytes": 3968
          },
          "sv": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16512,
            "padding_bytes": 3968
          }
        }
      },
      {
        "layer": 23,
        "valid_tokens": 516,
        "fields": {
          "pk": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 66048,
            "padding_bytes": 15872
          },
          "pv": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 66048,
            "padding_bytes": 15872
          },
          "sk": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16512,
            "padding_bytes": 3968
          },
          "sv": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16512,
            "padding_bytes": 3968
          }
        }
      },
      {
        "layer": 24,
        "valid_tokens": 516,
        "fields": {
          "pk": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 66048,
            "padding_bytes": 15872
          },
          "pv": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 66048,
            "padding_bytes": 15872
          },
          "sk": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16512,
            "padding_bytes": 3968
          },
          "sv": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16512,
            "padding_bytes": 3968
          }
        }
      },
      {
        "layer": 25,
        "valid_tokens": 516,
        "fields": {
          "pk": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 66048,
            "padding_bytes": 15872
          },
          "pv": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 66048,
            "padding_bytes": 15872
          },
          "sk": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16512,
            "padding_bytes": 3968
          },
          "sv": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16512,
            "padding_bytes": 3968
          }
        }
      },
      {
        "layer": 26,
        "valid_tokens": 516,
        "fields": {
          "pk": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 66048,
            "padding_bytes": 15872
          },
          "pv": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 66048,
            "padding_bytes": 15872
          },
          "sk": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16512,
            "padding_bytes": 3968
          },
          "sv": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16512,
            "padding_bytes": 3968
          }
        }
      },
      {
        "layer": 27,
        "valid_tokens": 516,
        "fields": {
          "pk": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 66048,
            "padding_bytes": 15872
          },
          "pv": {
            "shape": [
              1,
              2,
              640,
              64
            ],
            "dtype": "torch.uint8",
            "storage_bytes": 81920,
            "valid_bytes": 66048,
            "padding_bytes": 15872
          },
          "sk": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16512,
            "padding_bytes": 3968
          },
          "sv": {
            "shape": [
              1,
              2,
              640,
              4
            ],
            "dtype": "torch.float32",
            "storage_bytes": 20480,
            "valid_bytes": 16512,
            "padding_bytes": 3968
          }
        }
      }
    ],
    "unique_storages": 112,
    "dense_persistent_fields": 0
  },
  "resident_tokens_start": 512,
  "resident_tokens_end": 516,
  "timing_contract": "Whole model, four decode steps, KV append/encode and attention included. Two warmups then ten timed repetitions. Cache reset outside interval. Model prefill and cache materialization excluded and reported separately.",
  "interpretation": "A feasible measured grid point, not a maximum-OOM capacity claim or independently sampled serving workload.",
  "peak_workspace_and_growth_upper_bytes": 4276736
}

Only COLLECTED points passed full-model Gate D. Repeated homogeneous inputs and repeated timing trials are not independent quality samples. No serving scheduler or maximum-capacity claim.
