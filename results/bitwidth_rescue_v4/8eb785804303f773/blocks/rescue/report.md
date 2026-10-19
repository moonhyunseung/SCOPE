# INT8 bitwidth rescue v4

{
  "status": "COLLECTED",
  "independent_source_n": 2,
  "source_representation_conditions": 6,
  "steps": 4,
  "replays": 2,
  "metric_rows": 96,
  "all_controls_exact": true,
  "prior_R0_R3_immutable": true,
  "followup_review_eligibility": {
    "R4": false,
    "R5": false,
    "R6": false
  },
  "stop_uniform_family": true,
  "eligibility_is_not_system_adoption": true,
  "original_failures_preserved": true,
  "scope": "Two-source local INT8 representation screen,96metric rows. Dense dequantized BF16 storage, no packed HBM/latency/capacity/task-accuracy claim, no CI/population test. New followup requires advisor interpretation.",
  "runtime": {
    "python": "3.12.3",
    "torch": "2.14.1+cu130",
    "transformers": "5.19.0",
    "cuda": "13.0",
    "gpu": "NVIDIA H200",
    "tf32": false,
    "deterministic": true,
    "allocator_limit_bytes": 102312837120,
    "physical_reserve_gib": 44,
    "bf16_reduced_precision_reduction": true
  }
}

Every source/role, both full and cached references, and all failures retained. Steps/replays are repeated observations, no CI. Dense mixed-cache reference is not a memory-saving measurement. No token IDs, raw logits or activations are included. Subsequent research remains deferred pending advisor review.
