# Controlled ablation findings

{
  "run_id": "e5998af63b11ad5d",
  "source_sha": "396177b9bd682b84d7cefe3cddab1a2f3a55a1b2",
  "processed": 8,
  "status_counts": {
    "DATA_COLLECTED": 8
  },
  "all_noop_controls_pass": true,
  "all_native_snapshots_unchanged": true,
  "all_first_step_history_controls_exact": true,
  "first_step_history_checks": 112,
  "packing_added_relative_l2": {
    "n": 128,
    "min": 0.0,
    "median": 0.0,
    "max": 0.0,
    "mean": 0.0
  },
  "trajectory_rows": 1856,
  "treatment_rows": 1792,
  "factorial_rows": 128,
  "cross_dtype_identity": {
    "('prose', 512)": true,
    "('code', 512)": true,
    "('prose', 2048)": true,
    "('code', 2048)": true
  },
  "artifact_integrity": "PASS",
  "limits": "One model, two repeated authored input families, four context populations. Dependent observations; no significance or general language-model-quality claim. Native prefill and dense caches retained; no timing/capacity claim."
}

Per-treatment numerical summaries are in ENGINEERING_FINDINGS.json. Exact paired rows are in treatment_numeric.csv and factorial_numeric.csv. Raw paired logits remain in per-condition server directories.
