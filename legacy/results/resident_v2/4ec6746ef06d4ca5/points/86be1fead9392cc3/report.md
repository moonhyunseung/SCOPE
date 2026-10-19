# Actual resident-KV full-model point

{
  "status": "COLLECTED",
  "spec": {
    "document_id": "7190bc75f0028b8ef4e2",
    "domain": "prose",
    "length": 512,
    "batch": 1,
    "method": "dynamic_eager",
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
  "representation": "BF16 native dense",
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
    "payload_bytes": 14794752,
    "workspace_upper_estimate_bytes": 542332416,
    "available_device_budget_bytes": 98858827776,
    "host_reference_bytes": 1215488
  },
  "cache_materialization_wall_s": 0.004602998495101929,
  "resident_initial": {
    "storage_bytes": 14680064,
    "layers": [
      {
        "layer": 0,
        "valid_tokens": 512,
        "fields": {
          "keys": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 262144,
            "valid_bytes": 262144,
            "padding_bytes": 0
          },
          "values": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 262144,
            "valid_bytes": 262144,
            "padding_bytes": 0
          }
        }
      },
      {
        "layer": 1,
        "valid_tokens": 512,
        "fields": {
          "keys": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 262144,
            "valid_bytes": 262144,
            "padding_bytes": 0
          },
          "values": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 262144,
            "valid_bytes": 262144,
            "padding_bytes": 0
          }
        }
      },
      {
        "layer": 2,
        "valid_tokens": 512,
        "fields": {
          "keys": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 262144,
            "valid_bytes": 262144,
            "padding_bytes": 0
          },
          "values": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 262144,
            "valid_bytes": 262144,
            "padding_bytes": 0
          }
        }
      },
      {
        "layer": 3,
        "valid_tokens": 512,
        "fields": {
          "keys": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 262144,
            "valid_bytes": 262144,
            "padding_bytes": 0
          },
          "values": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 262144,
            "valid_bytes": 262144,
            "padding_bytes": 0
          }
        }
      },
      {
        "layer": 4,
        "valid_tokens": 512,
        "fields": {
          "keys": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 262144,
            "valid_bytes": 262144,
            "padding_bytes": 0
          },
          "values": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 262144,
            "valid_bytes": 262144,
            "padding_bytes": 0
          }
        }
      },
      {
        "layer": 5,
        "valid_tokens": 512,
        "fields": {
          "keys": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 262144,
            "valid_bytes": 262144,
            "padding_bytes": 0
          },
          "values": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 262144,
            "valid_bytes": 262144,
            "padding_bytes": 0
          }
        }
      },
      {
        "layer": 6,
        "valid_tokens": 512,
        "fields": {
          "keys": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 262144,
            "valid_bytes": 262144,
            "padding_bytes": 0
          },
          "values": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 262144,
            "valid_bytes": 262144,
            "padding_bytes": 0
          }
        }
      },
      {
        "layer": 7,
        "valid_tokens": 512,
        "fields": {
          "keys": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 262144,
            "valid_bytes": 262144,
            "padding_bytes": 0
          },
          "values": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 262144,
            "valid_bytes": 262144,
            "padding_bytes": 0
          }
        }
      },
      {
        "layer": 8,
        "valid_tokens": 512,
        "fields": {
          "keys": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 262144,
            "valid_bytes": 262144,
            "padding_bytes": 0
          },
          "values": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 262144,
            "valid_bytes": 262144,
            "padding_bytes": 0
          }
        }
      },
      {
        "layer": 9,
        "valid_tokens": 512,
        "fields": {
          "keys": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 262144,
            "valid_bytes": 262144,
            "padding_bytes": 0
          },
          "values": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 262144,
            "valid_bytes": 262144,
            "padding_bytes": 0
          }
        }
      },
      {
        "layer": 10,
        "valid_tokens": 512,
        "fields": {
          "keys": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 262144,
            "valid_bytes": 262144,
            "padding_bytes": 0
          },
          "values": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 262144,
            "valid_bytes": 262144,
            "padding_bytes": 0
          }
        }
      },
      {
        "layer": 11,
        "valid_tokens": 512,
        "fields": {
          "keys": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 262144,
            "valid_bytes": 262144,
            "padding_bytes": 0
          },
          "values": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 262144,
            "valid_bytes": 262144,
            "padding_bytes": 0
          }
        }
      },
      {
        "layer": 12,
        "valid_tokens": 512,
        "fields": {
          "keys": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 262144,
            "valid_bytes": 262144,
            "padding_bytes": 0
          },
          "values": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 262144,
            "valid_bytes": 262144,
            "padding_bytes": 0
          }
        }
      },
      {
        "layer": 13,
        "valid_tokens": 512,
        "fields": {
          "keys": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 262144,
            "valid_bytes": 262144,
            "padding_bytes": 0
          },
          "values": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 262144,
            "valid_bytes": 262144,
            "padding_bytes": 0
          }
        }
      },
      {
        "layer": 14,
        "valid_tokens": 512,
        "fields": {
          "keys": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 262144,
            "valid_bytes": 262144,
            "padding_bytes": 0
          },
          "values": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 262144,
            "valid_bytes": 262144,
            "padding_bytes": 0
          }
        }
      },
      {
        "layer": 15,
        "valid_tokens": 512,
        "fields": {
          "keys": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 262144,
            "valid_bytes": 262144,
            "padding_bytes": 0
          },
          "values": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 262144,
            "valid_bytes": 262144,
            "padding_bytes": 0
          }
        }
      },
      {
        "layer": 16,
        "valid_tokens": 512,
        "fields": {
          "keys": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 262144,
            "valid_bytes": 262144,
            "padding_bytes": 0
          },
          "values": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 262144,
            "valid_bytes": 262144,
            "padding_bytes": 0
          }
        }
      },
      {
        "layer": 17,
        "valid_tokens": 512,
        "fields": {
          "keys": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 262144,
            "valid_bytes": 262144,
            "padding_bytes": 0
          },
          "values": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 262144,
            "valid_bytes": 262144,
            "padding_bytes": 0
          }
        }
      },
      {
        "layer": 18,
        "valid_tokens": 512,
        "fields": {
          "keys": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 262144,
            "valid_bytes": 262144,
            "padding_bytes": 0
          },
          "values": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 262144,
            "valid_bytes": 262144,
            "padding_bytes": 0
          }
        }
      },
      {
        "layer": 19,
        "valid_tokens": 512,
        "fields": {
          "keys": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 262144,
            "valid_bytes": 262144,
            "padding_bytes": 0
          },
          "values": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 262144,
            "valid_bytes": 262144,
            "padding_bytes": 0
          }
        }
      },
      {
        "layer": 20,
        "valid_tokens": 512,
        "fields": {
          "keys": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 262144,
            "valid_bytes": 262144,
            "padding_bytes": 0
          },
          "values": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 262144,
            "valid_bytes": 262144,
            "padding_bytes": 0
          }
        }
      },
      {
        "layer": 21,
        "valid_tokens": 512,
        "fields": {
          "keys": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 262144,
            "valid_bytes": 262144,
            "padding_bytes": 0
          },
          "values": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 262144,
            "valid_bytes": 262144,
            "padding_bytes": 0
          }
        }
      },
      {
        "layer": 22,
        "valid_tokens": 512,
        "fields": {
          "keys": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 262144,
            "valid_bytes": 262144,
            "padding_bytes": 0
          },
          "values": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 262144,
            "valid_bytes": 262144,
            "padding_bytes": 0
          }
        }
      },
      {
        "layer": 23,
        "valid_tokens": 512,
        "fields": {
          "keys": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 262144,
            "valid_bytes": 262144,
            "padding_bytes": 0
          },
          "values": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 262144,
            "valid_bytes": 262144,
            "padding_bytes": 0
          }
        }
      },
      {
        "layer": 24,
        "valid_tokens": 512,
        "fields": {
          "keys": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 262144,
            "valid_bytes": 262144,
            "padding_bytes": 0
          },
          "values": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 262144,
            "valid_bytes": 262144,
            "padding_bytes": 0
          }
        }
      },
      {
        "layer": 25,
        "valid_tokens": 512,
        "fields": {
          "keys": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 262144,
            "valid_bytes": 262144,
            "padding_bytes": 0
          },
          "values": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 262144,
            "valid_bytes": 262144,
            "padding_bytes": 0
          }
        }
      },
      {
        "layer": 26,
        "valid_tokens": 512,
        "fields": {
          "keys": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 262144,
            "valid_bytes": 262144,
            "padding_bytes": 0
          },
          "values": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 262144,
            "valid_bytes": 262144,
            "padding_bytes": 0
          }
        }
      },
      {
        "layer": 27,
        "valid_tokens": 512,
        "fields": {
          "keys": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 262144,
            "valid_bytes": 262144,
            "padding_bytes": 0
          },
          "values": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 262144,
            "valid_bytes": 262144,
            "padding_bytes": 0
          }
        }
      }
    ],
    "unique_storages": 56,
    "dense_persistent_fields": 56
  },
  "memory_initial": {
    "allocated_bytes": 3136583680,
    "reserved_bytes": 3321888768,
    "peak_allocated_bytes": 3179843584,
    "peak_reserved_bytes": 3376414720,
    "active_block_bytes": 3136583680,
    "active_requested_bytes": 3135663648,
    "active_block_rounding_bytes": 920032,
    "reserved_minus_active_bytes": 185305088,
    "inactive_split_bytes": 185305088,
    "physical_used_bytes": 4008706048,
    "physical_free_bytes": 146102288384,
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
    "allocated_bytes": 3136698368,
    "reserved_bytes": 3330277376,
    "peak_allocated_bytes": 3179843584,
    "peak_reserved_bytes": 3376414720,
    "active_block_bytes": 3136698368,
    "active_requested_bytes": 3135778336,
    "active_block_rounding_bytes": 920032,
    "reserved_minus_active_bytes": 193579008,
    "inactive_split_bytes": 193579008,
    "physical_used_bytes": 4017094656,
    "physical_free_bytes": 146093899776,
    "physical_minus_allocator_reserved_bytes": 686817280,
    "host_allocator_metadata_bytes": null
  },
  "resident_timed_start": {
    "storage_bytes": 14794752,
    "layers": [
      {
        "layer": 0,
        "valid_tokens": 512,
        "fields": {
          "keys": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 262144,
            "padding_bytes": 2048
          },
          "values": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 262144,
            "padding_bytes": 2048
          }
        }
      },
      {
        "layer": 1,
        "valid_tokens": 512,
        "fields": {
          "keys": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 262144,
            "padding_bytes": 2048
          },
          "values": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 262144,
            "padding_bytes": 2048
          }
        }
      },
      {
        "layer": 2,
        "valid_tokens": 512,
        "fields": {
          "keys": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 262144,
            "padding_bytes": 2048
          },
          "values": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 262144,
            "padding_bytes": 2048
          }
        }
      },
      {
        "layer": 3,
        "valid_tokens": 512,
        "fields": {
          "keys": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 262144,
            "padding_bytes": 2048
          },
          "values": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 262144,
            "padding_bytes": 2048
          }
        }
      },
      {
        "layer": 4,
        "valid_tokens": 512,
        "fields": {
          "keys": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 262144,
            "padding_bytes": 2048
          },
          "values": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 262144,
            "padding_bytes": 2048
          }
        }
      },
      {
        "layer": 5,
        "valid_tokens": 512,
        "fields": {
          "keys": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 262144,
            "padding_bytes": 2048
          },
          "values": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 262144,
            "padding_bytes": 2048
          }
        }
      },
      {
        "layer": 6,
        "valid_tokens": 512,
        "fields": {
          "keys": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 262144,
            "padding_bytes": 2048
          },
          "values": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 262144,
            "padding_bytes": 2048
          }
        }
      },
      {
        "layer": 7,
        "valid_tokens": 512,
        "fields": {
          "keys": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 262144,
            "padding_bytes": 2048
          },
          "values": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 262144,
            "padding_bytes": 2048
          }
        }
      },
      {
        "layer": 8,
        "valid_tokens": 512,
        "fields": {
          "keys": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 262144,
            "padding_bytes": 2048
          },
          "values": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 262144,
            "padding_bytes": 2048
          }
        }
      },
      {
        "layer": 9,
        "valid_tokens": 512,
        "fields": {
          "keys": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 262144,
            "padding_bytes": 2048
          },
          "values": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 262144,
            "padding_bytes": 2048
          }
        }
      },
      {
        "layer": 10,
        "valid_tokens": 512,
        "fields": {
          "keys": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 262144,
            "padding_bytes": 2048
          },
          "values": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 262144,
            "padding_bytes": 2048
          }
        }
      },
      {
        "layer": 11,
        "valid_tokens": 512,
        "fields": {
          "keys": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 262144,
            "padding_bytes": 2048
          },
          "values": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 262144,
            "padding_bytes": 2048
          }
        }
      },
      {
        "layer": 12,
        "valid_tokens": 512,
        "fields": {
          "keys": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 262144,
            "padding_bytes": 2048
          },
          "values": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 262144,
            "padding_bytes": 2048
          }
        }
      },
      {
        "layer": 13,
        "valid_tokens": 512,
        "fields": {
          "keys": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 262144,
            "padding_bytes": 2048
          },
          "values": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 262144,
            "padding_bytes": 2048
          }
        }
      },
      {
        "layer": 14,
        "valid_tokens": 512,
        "fields": {
          "keys": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 262144,
            "padding_bytes": 2048
          },
          "values": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 262144,
            "padding_bytes": 2048
          }
        }
      },
      {
        "layer": 15,
        "valid_tokens": 512,
        "fields": {
          "keys": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 262144,
            "padding_bytes": 2048
          },
          "values": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 262144,
            "padding_bytes": 2048
          }
        }
      },
      {
        "layer": 16,
        "valid_tokens": 512,
        "fields": {
          "keys": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 262144,
            "padding_bytes": 2048
          },
          "values": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 262144,
            "padding_bytes": 2048
          }
        }
      },
      {
        "layer": 17,
        "valid_tokens": 512,
        "fields": {
          "keys": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 262144,
            "padding_bytes": 2048
          },
          "values": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 262144,
            "padding_bytes": 2048
          }
        }
      },
      {
        "layer": 18,
        "valid_tokens": 512,
        "fields": {
          "keys": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 262144,
            "padding_bytes": 2048
          },
          "values": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 262144,
            "padding_bytes": 2048
          }
        }
      },
      {
        "layer": 19,
        "valid_tokens": 512,
        "fields": {
          "keys": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 262144,
            "padding_bytes": 2048
          },
          "values": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 262144,
            "padding_bytes": 2048
          }
        }
      },
      {
        "layer": 20,
        "valid_tokens": 512,
        "fields": {
          "keys": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 262144,
            "padding_bytes": 2048
          },
          "values": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 262144,
            "padding_bytes": 2048
          }
        }
      },
      {
        "layer": 21,
        "valid_tokens": 512,
        "fields": {
          "keys": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 262144,
            "padding_bytes": 2048
          },
          "values": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 262144,
            "padding_bytes": 2048
          }
        }
      },
      {
        "layer": 22,
        "valid_tokens": 512,
        "fields": {
          "keys": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 262144,
            "padding_bytes": 2048
          },
          "values": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 262144,
            "padding_bytes": 2048
          }
        }
      },
      {
        "layer": 23,
        "valid_tokens": 512,
        "fields": {
          "keys": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 262144,
            "padding_bytes": 2048
          },
          "values": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 262144,
            "padding_bytes": 2048
          }
        }
      },
      {
        "layer": 24,
        "valid_tokens": 512,
        "fields": {
          "keys": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 262144,
            "padding_bytes": 2048
          },
          "values": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 262144,
            "padding_bytes": 2048
          }
        }
      },
      {
        "layer": 25,
        "valid_tokens": 512,
        "fields": {
          "keys": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 262144,
            "padding_bytes": 2048
          },
          "values": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 262144,
            "padding_bytes": 2048
          }
        }
      },
      {
        "layer": 26,
        "valid_tokens": 512,
        "fields": {
          "keys": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 262144,
            "padding_bytes": 2048
          },
          "values": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 262144,
            "padding_bytes": 2048
          }
        }
      },
      {
        "layer": 27,
        "valid_tokens": 512,
        "fields": {
          "keys": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 262144,
            "padding_bytes": 2048
          },
          "values": {
            "shape": [
              1,
              2,
              512,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 262144,
            "padding_bytes": 2048
          }
        }
      }
    ],
    "unique_storages": 56,
    "dense_persistent_fields": 56
  },
  "timing_repetitions": [
    {
      "cuda_ms": 58.53180694580078,
      "wall_s": 0.05870301928371191,
      "aggregate_tokens_per_s": 68.338911930464,
      "wall_tokens_per_s": 68.13959569384302
    },
    {
      "cuda_ms": 64.80464172363281,
      "wall_s": 0.06482441816478968,
      "aggregate_tokens_per_s": 61.723973678590504,
      "wall_tokens_per_s": 61.70514311183834
    },
    {
      "cuda_ms": 54.62694549560547,
      "wall_s": 0.05464528780430555,
      "aggregate_tokens_per_s": 73.22393671675793,
      "wall_tokens_per_s": 73.19935827449035
    },
    {
      "cuda_ms": 54.40099334716797,
      "wall_s": 0.05441885534673929,
      "aggregate_tokens_per_s": 73.52806913788153,
      "wall_tokens_per_s": 73.5039348864157
    },
    {
      "cuda_ms": 54.32992172241211,
      "wall_s": 0.054347275756299496,
      "aggregate_tokens_per_s": 73.62425479714845,
      "wall_tokens_per_s": 73.60074528733581
    },
    {
      "cuda_ms": 54.3969612121582,
      "wall_s": 0.054414535872638226,
      "aggregate_tokens_per_s": 73.53351935228993,
      "wall_tokens_per_s": 73.509769693935
    },
    {
      "cuda_ms": 54.47385787963867,
      "wall_s": 0.05449140630662441,
      "aggregate_tokens_per_s": 73.42971758743613,
      "wall_tokens_per_s": 73.40607026164652
    },
    {
      "cuda_ms": 54.410945892333984,
      "wall_s": 0.0544282253831625,
      "aggregate_tokens_per_s": 73.51461979571218,
      "wall_tokens_per_s": 73.4912808904001
    },
    {
      "cuda_ms": 54.59132766723633,
      "wall_s": 0.05460863746702671,
      "aggregate_tokens_per_s": 73.27171129418511,
      "wall_tokens_per_s": 73.24848568901291
    },
    {
      "cuda_ms": 54.44550323486328,
      "wall_s": 0.054463460110127926,
      "aggregate_tokens_per_s": 73.4679590111432,
      "wall_tokens_per_s": 73.44373625751639
    }
  ],
  "memory_timed": {
    "allocated_bytes": 3136698368,
    "reserved_bytes": 3330277376,
    "peak_allocated_bytes": 3140446720,
    "peak_reserved_bytes": 3330277376,
    "active_block_bytes": 3136698368,
    "active_requested_bytes": 3135778336,
    "active_block_rounding_bytes": 920032,
    "reserved_minus_active_bytes": 193579008,
    "inactive_split_bytes": 193579008,
    "physical_used_bytes": 4017094656,
    "physical_free_bytes": 146093899776,
    "physical_minus_allocator_reserved_bytes": 686817280,
    "host_allocator_metadata_bytes": null
  },
  "resident_final": {
    "storage_bytes": 14794752,
    "layers": [
      {
        "layer": 0,
        "valid_tokens": 516,
        "fields": {
          "keys": {
            "shape": [
              1,
              2,
              516,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 264192,
            "padding_bytes": 0
          },
          "values": {
            "shape": [
              1,
              2,
              516,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 264192,
            "padding_bytes": 0
          }
        }
      },
      {
        "layer": 1,
        "valid_tokens": 516,
        "fields": {
          "keys": {
            "shape": [
              1,
              2,
              516,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 264192,
            "padding_bytes": 0
          },
          "values": {
            "shape": [
              1,
              2,
              516,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 264192,
            "padding_bytes": 0
          }
        }
      },
      {
        "layer": 2,
        "valid_tokens": 516,
        "fields": {
          "keys": {
            "shape": [
              1,
              2,
              516,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 264192,
            "padding_bytes": 0
          },
          "values": {
            "shape": [
              1,
              2,
              516,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 264192,
            "padding_bytes": 0
          }
        }
      },
      {
        "layer": 3,
        "valid_tokens": 516,
        "fields": {
          "keys": {
            "shape": [
              1,
              2,
              516,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 264192,
            "padding_bytes": 0
          },
          "values": {
            "shape": [
              1,
              2,
              516,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 264192,
            "padding_bytes": 0
          }
        }
      },
      {
        "layer": 4,
        "valid_tokens": 516,
        "fields": {
          "keys": {
            "shape": [
              1,
              2,
              516,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 264192,
            "padding_bytes": 0
          },
          "values": {
            "shape": [
              1,
              2,
              516,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 264192,
            "padding_bytes": 0
          }
        }
      },
      {
        "layer": 5,
        "valid_tokens": 516,
        "fields": {
          "keys": {
            "shape": [
              1,
              2,
              516,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 264192,
            "padding_bytes": 0
          },
          "values": {
            "shape": [
              1,
              2,
              516,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 264192,
            "padding_bytes": 0
          }
        }
      },
      {
        "layer": 6,
        "valid_tokens": 516,
        "fields": {
          "keys": {
            "shape": [
              1,
              2,
              516,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 264192,
            "padding_bytes": 0
          },
          "values": {
            "shape": [
              1,
              2,
              516,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 264192,
            "padding_bytes": 0
          }
        }
      },
      {
        "layer": 7,
        "valid_tokens": 516,
        "fields": {
          "keys": {
            "shape": [
              1,
              2,
              516,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 264192,
            "padding_bytes": 0
          },
          "values": {
            "shape": [
              1,
              2,
              516,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 264192,
            "padding_bytes": 0
          }
        }
      },
      {
        "layer": 8,
        "valid_tokens": 516,
        "fields": {
          "keys": {
            "shape": [
              1,
              2,
              516,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 264192,
            "padding_bytes": 0
          },
          "values": {
            "shape": [
              1,
              2,
              516,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 264192,
            "padding_bytes": 0
          }
        }
      },
      {
        "layer": 9,
        "valid_tokens": 516,
        "fields": {
          "keys": {
            "shape": [
              1,
              2,
              516,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 264192,
            "padding_bytes": 0
          },
          "values": {
            "shape": [
              1,
              2,
              516,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 264192,
            "padding_bytes": 0
          }
        }
      },
      {
        "layer": 10,
        "valid_tokens": 516,
        "fields": {
          "keys": {
            "shape": [
              1,
              2,
              516,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 264192,
            "padding_bytes": 0
          },
          "values": {
            "shape": [
              1,
              2,
              516,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 264192,
            "padding_bytes": 0
          }
        }
      },
      {
        "layer": 11,
        "valid_tokens": 516,
        "fields": {
          "keys": {
            "shape": [
              1,
              2,
              516,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 264192,
            "padding_bytes": 0
          },
          "values": {
            "shape": [
              1,
              2,
              516,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 264192,
            "padding_bytes": 0
          }
        }
      },
      {
        "layer": 12,
        "valid_tokens": 516,
        "fields": {
          "keys": {
            "shape": [
              1,
              2,
              516,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 264192,
            "padding_bytes": 0
          },
          "values": {
            "shape": [
              1,
              2,
              516,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 264192,
            "padding_bytes": 0
          }
        }
      },
      {
        "layer": 13,
        "valid_tokens": 516,
        "fields": {
          "keys": {
            "shape": [
              1,
              2,
              516,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 264192,
            "padding_bytes": 0
          },
          "values": {
            "shape": [
              1,
              2,
              516,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 264192,
            "padding_bytes": 0
          }
        }
      },
      {
        "layer": 14,
        "valid_tokens": 516,
        "fields": {
          "keys": {
            "shape": [
              1,
              2,
              516,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 264192,
            "padding_bytes": 0
          },
          "values": {
            "shape": [
              1,
              2,
              516,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 264192,
            "padding_bytes": 0
          }
        }
      },
      {
        "layer": 15,
        "valid_tokens": 516,
        "fields": {
          "keys": {
            "shape": [
              1,
              2,
              516,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 264192,
            "padding_bytes": 0
          },
          "values": {
            "shape": [
              1,
              2,
              516,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 264192,
            "padding_bytes": 0
          }
        }
      },
      {
        "layer": 16,
        "valid_tokens": 516,
        "fields": {
          "keys": {
            "shape": [
              1,
              2,
              516,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 264192,
            "padding_bytes": 0
          },
          "values": {
            "shape": [
              1,
              2,
              516,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 264192,
            "padding_bytes": 0
          }
        }
      },
      {
        "layer": 17,
        "valid_tokens": 516,
        "fields": {
          "keys": {
            "shape": [
              1,
              2,
              516,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 264192,
            "padding_bytes": 0
          },
          "values": {
            "shape": [
              1,
              2,
              516,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 264192,
            "padding_bytes": 0
          }
        }
      },
      {
        "layer": 18,
        "valid_tokens": 516,
        "fields": {
          "keys": {
            "shape": [
              1,
              2,
              516,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 264192,
            "padding_bytes": 0
          },
          "values": {
            "shape": [
              1,
              2,
              516,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 264192,
            "padding_bytes": 0
          }
        }
      },
      {
        "layer": 19,
        "valid_tokens": 516,
        "fields": {
          "keys": {
            "shape": [
              1,
              2,
              516,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 264192,
            "padding_bytes": 0
          },
          "values": {
            "shape": [
              1,
              2,
              516,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 264192,
            "padding_bytes": 0
          }
        }
      },
      {
        "layer": 20,
        "valid_tokens": 516,
        "fields": {
          "keys": {
            "shape": [
              1,
              2,
              516,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 264192,
            "padding_bytes": 0
          },
          "values": {
            "shape": [
              1,
              2,
              516,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 264192,
            "padding_bytes": 0
          }
        }
      },
      {
        "layer": 21,
        "valid_tokens": 516,
        "fields": {
          "keys": {
            "shape": [
              1,
              2,
              516,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 264192,
            "padding_bytes": 0
          },
          "values": {
            "shape": [
              1,
              2,
              516,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 264192,
            "padding_bytes": 0
          }
        }
      },
      {
        "layer": 22,
        "valid_tokens": 516,
        "fields": {
          "keys": {
            "shape": [
              1,
              2,
              516,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 264192,
            "padding_bytes": 0
          },
          "values": {
            "shape": [
              1,
              2,
              516,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 264192,
            "padding_bytes": 0
          }
        }
      },
      {
        "layer": 23,
        "valid_tokens": 516,
        "fields": {
          "keys": {
            "shape": [
              1,
              2,
              516,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 264192,
            "padding_bytes": 0
          },
          "values": {
            "shape": [
              1,
              2,
              516,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 264192,
            "padding_bytes": 0
          }
        }
      },
      {
        "layer": 24,
        "valid_tokens": 516,
        "fields": {
          "keys": {
            "shape": [
              1,
              2,
              516,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 264192,
            "padding_bytes": 0
          },
          "values": {
            "shape": [
              1,
              2,
              516,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 264192,
            "padding_bytes": 0
          }
        }
      },
      {
        "layer": 25,
        "valid_tokens": 516,
        "fields": {
          "keys": {
            "shape": [
              1,
              2,
              516,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 264192,
            "padding_bytes": 0
          },
          "values": {
            "shape": [
              1,
              2,
              516,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 264192,
            "padding_bytes": 0
          }
        }
      },
      {
        "layer": 26,
        "valid_tokens": 516,
        "fields": {
          "keys": {
            "shape": [
              1,
              2,
              516,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 264192,
            "padding_bytes": 0
          },
          "values": {
            "shape": [
              1,
              2,
              516,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 264192,
            "padding_bytes": 0
          }
        }
      },
      {
        "layer": 27,
        "valid_tokens": 516,
        "fields": {
          "keys": {
            "shape": [
              1,
              2,
              516,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 264192,
            "padding_bytes": 0
          },
          "values": {
            "shape": [
              1,
              2,
              516,
              128
            ],
            "dtype": "torch.bfloat16",
            "storage_bytes": 264192,
            "valid_bytes": 264192,
            "padding_bytes": 0
          }
        }
      }
    ],
    "unique_storages": 56,
    "dense_persistent_fields": 56
  },
  "resident_tokens_start": 512,
  "resident_tokens_end": 516,
  "timing_contract": "Whole model, four decode steps, KV append/encode and attention included. Two warmups then ten timed repetitions. Cache reset outside interval. Model prefill and cache materialization excluded and reported separately.",
  "interpretation": "A feasible measured grid point, not a maximum-OOM capacity claim or independently sampled serving workload.",
  "peak_workspace_and_growth_upper_bytes": 3748352
}

Only COLLECTED points passed full-model Gate D. Repeated homogeneous inputs and repeated timing trials are not independent quality samples. No serving scheduler or maximum-capacity claim.
