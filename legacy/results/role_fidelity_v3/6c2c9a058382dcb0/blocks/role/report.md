# K/V role fidelity screen v3

{
  "status": "COLLECTED",
  "independent_source_n": 2,
  "source_representation_conditions": 8,
  "steps": 4,
  "replays": 2,
  "metric_rows": 128,
  "all_controls_exact": true,
  "candidate_roles": {
    "R1": "NOT_ACCEPTED",
    "R2": "NOT_ACCEPTED",
    "R3": "NOT_ACCEPTED"
  },
  "original_failures_preserved": true,
  "scope": "Exploratory two-source local logit fidelity screen, no CI/population test. Dense independent mixed representation reference persists dequantized BF16 values; no physical cache saving/capacity or optimized kernel claim. INT8 and lowering repair are deferred pending advisor review.",
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

Every source/role, both full and cached references, and all failures retained. Steps/replays are repeated observations, no CI. Dense mixed-cache reference is not a memory-saving measurement. No token IDs, raw logits or activations are included. INT8 remains deferred.
