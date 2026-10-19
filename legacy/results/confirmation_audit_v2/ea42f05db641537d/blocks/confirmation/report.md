# Additive confirmation audit, ACL-R1

{
  "status": "VERIFIED_ADDITIVE_AUDIT",
  "upstream_run": "2aa605c55475af9d",
  "input_manifest_sha256": "fbdd626cb95eac1992fbf76406d871f726f15cf0aa48fd3bf5817a46a6038ec5",
  "upstream_source_sha256": "d23c432a036062cedeaf5199f293b7efdbab6a27f5443ebff1513f52c7a44b90",
  "planned_conditions": 560,
  "terminal_counts": {
    "COLLECTED": 560
  },
  "planned_independent_source_units": 112,
  "planned_independent_sources_per_domain": 56,
  "observed_source_units_by_model": {
    "qwen1_5b": 112,
    "pythia1b": 112,
    "opt1_3b": 112,
    "qwen7b": 112,
    "qwen14b": 112
  },
  "models_are_repeated_measures": true,
  "primary_contrasts": 8,
  "primary_reproduced": 8,
  "exploratory_cross_model_contrasts": 40,
  "source_tail_rows": 6720,
  "source_mean_rows": 6720,
  "dense_failure_conditions": 338,
  "observed_threshold_scope": "The .025 intervention error threshold in tail tables is descriptive against native cached logits; it is NOT an execution Gate D result or a model-quality pass/fail gate.",
  "uncertainty": "Frozen eight primaries: 20k resamples,99.375% intervals. Cross-model contrasts: jointly paired source draws,20k resamples,exploratory95% intervals; no familywise claim. No Q99/rare-population precision claim.",
  "cross_model_scope": "Differences between within-model normalized logit perturbation effects across distinct vocabularies/tokenizers are exploratory, not model-quality rankings.",
  "publication": "PENDING"
}

Primary endpoints, populations and thresholds are unchanged. Native cached intervention is not original full-reference quality parity. All head/token/step/model repetitions share source units. Observed tail counts are not estimates of extremely rare population events.
