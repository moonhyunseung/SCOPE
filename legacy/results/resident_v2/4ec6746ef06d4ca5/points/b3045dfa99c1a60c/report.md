# Actual resident-KV full-model point

{
  "status": "GATE_D_FAIL",
  "spec": {
    "document_id": "7190bc75f0028b8ef4e2",
    "domain": "prose",
    "length": 512,
    "batch": 1,
    "method": "int4_triton",
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
    "workspace_upper_estimate_bytes": 537607680,
    "available_device_budget_bytes": 98858827776,
    "host_reference_bytes": 1215488
  },
  "cache_materialization_wall_s": 0.02123323082923889,
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
      "max_request_relative_l2": 0.06828393425246819,
      "mean_request_relative_l2": 0.06828393425246819,
      "max_abs": 1.140625,
      "top1_agreement": 1.0
    },
    {
      "max_request_relative_l2": 0.07693403830873781,
      "mean_request_relative_l2": 0.07693403830873781,
      "max_abs": 1.328125,
      "top1_agreement": 1.0
    }
  ]
}

Only COLLECTED points passed full-model Gate D. Repeated homogeneous inputs and repeated timing trials are not independent quality samples. No serving scheduler or maximum-capacity claim.
