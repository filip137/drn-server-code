# Perfect-Diode Conv3 Learning-Rate Screen at T=8, K=8 (Receipt-Safe)

This approved operational successor preserves v4's user-selected `T=8, K=8`
science, complete upward 3x3 rho grid, and minimal-core smoke at
`rho_conv=0.009, rho_dense=0.03`. It adds a hash-strict, read-only receipt
validation path for the outer scheduled-run audit, which executes outside the
bound tmux lanes. Numerical execution still requires the exact live lane
environment. Fresh manifest-bound residual and gradient audits at exactly
`8/8` and fresh smokes are required before either lane may launch.

```json
{
  "schema_version": "experiment-run-plan/v1",
  "experiment_id": "perfectdiode-conv3-lr-ordinary-mnist-t8k8-20260727-v5",
  "title": "Perfect-Diode Conv3 Learning-Rate Screen at T=8, K=8 (Receipt-Safe)",
  "evidence_scope": "diagnostic",
  "hypothesis": {
    "statement": "At the fixed T=8, K=8 operating point, each Conv3 perfect-diode scheme and optimizer has an independently selectable safety-admissible parameter-relative learning-rate cell within its approved core and single conditional boundary expansion.",
    "control": "Within each surface, every rho cell restarts from the same hash-bound seed-0 Conv3 initialization and uses the same ordinary-MNIST 55000/5000 split, three epoch batch orders, fixed T=8 and K=8 operating point, and optimizer-specific probe; cells differ only in their declared rho targets and resulting parameter-specific learning-rate vector.",
    "treatments": [
      "conv3_baseline_v1_c1--sgd on tmux main",
      "conv3_baseline_v1_c1--adam on tmux main",
      "conv3_ours_v4_c1--sgd on tmux akibscomputer",
      "conv3_ours_v4_c1--adam on tmux akibscomputer",
      "conv3_legacy_v4_c0p25--sgd on tmux main",
      "conv3_legacy_v4_c0p25--adam on tmux main"
    ],
    "expected_direction": "The upward factor-three baseline and ours grids are expected to expose a safety-admissible passing plateau, while legacy may require its independently evaluated safer adaptive center; each scheme-by-optimizer surface is interpreted independently.",
    "decision_rule": {
      "metric": "Minimum final validation loss among safety-admissible candidates completing exactly three epochs and passing epochwise gradient viability, final validation accuracy at least 0.90, and the post-training T/K audit; form the inclusive 2% relative-loss plateau and apply the frozen deterministic tie-breakers.",
      "split": "Hash-bound ordinary-MNIST seed-0 55000-example training and deterministic stratified 5000-example validation partitions; the official test set is not read.",
      "checkpoint_role": "Final checkpoint after exactly 10314 successful optimizer steps for each restarted candidate, with all three epoch checkpoints retained for post-training T/K replay.",
      "seed_aggregation": "One prespecified model, split, and shuffle seed, seed 0; no cross-seed aggregation.",
      "comparison": "Select a learning-rate cell independently within each of the six scheme-by-optimizer surfaces; do not transfer rho targets, optimizer units, raw learning rates, or conclusions across surfaces, and do not make cross-scheme superiority claims because host is confounded with scheme.",
      "minimum_meaningful_effect": 0.02,
      "effect_units": "Relative final-validation-loss fraction defining the inclusive plateau."
    },
    "supports_if": "All six surfaces publish a safety-admissible selected three-epoch cell after at most one approved expansion wave and pass their post-training T/K audits, after all three initialization residual/gradient gates pass at exactly T=8, K=8.",
    "does_not_support_if": "A terminal surface has no passing candidate, exhausts the safer legacy center search, remains boundary-confined after its one expansion, or fails the post-training T/K audit.",
    "inconclusive_if": "Any required fixed T=8/K=8 gate, surface, canary, candidate, probe, epochwise viability check, replay, environment record, completion marker, transfer receipt, or artifact hash is missing, partial, mixed across hosts, non-finite, or unverifiable."
  },
  "sweep": {
    "config_path": "results/perfectdiode_conv3_lr_t8k8_20260727_v5/study.resolved.json",
    "config_sha256": "810fffb9a30357b9ddba66d142e93b4db6eca8364964845832a2897e433c001d",
    "manifest_path": "results/perfectdiode_conv3_lr_t8k8_20260727_v5/surface_manifest.json",
    "manifest_sha256": "b50d535065a044084638cf8cee59d000c2296b6a87467a278180eb5f27b9b987",
    "manifest_job_count_pointer": "/surfaces",
    "axes": [
      {
        "name": "operational_tk",
        "config_path": "/operating_point",
        "ordered_values": [
          {
            "inference_iterations": 8,
            "training_iterations": 8
          }
        ]
      },
      {
        "name": "scheme_optimizer_surface",
        "config_path": "/execution/default_surface_routes",
        "ordered_values": [
          "conv3_baseline_v1_c1--sgd",
          "conv3_baseline_v1_c1--adam",
          "conv3_ours_v4_c1--sgd",
          "conv3_ours_v4_c1--adam",
          "conv3_legacy_v4_c0p25--sgd",
          "conv3_legacy_v4_c0p25--adam"
        ]
      },
      {
        "name": "baseline_ours_rho_conv",
        "config_path": "/rho_search/surface_policies/baseline/rho_conv",
        "ordered_values": [
          0.009,
          0.027,
          0.081
        ]
      },
      {
        "name": "baseline_ours_rho_dense",
        "config_path": "/rho_search/surface_policies/baseline/rho_dense",
        "ordered_values": [
          0.03,
          0.09,
          0.27
        ]
      },
      {
        "name": "legacy_adaptive_center",
        "config_path": "/rho_search/surface_policies/legacy",
        "ordered_values": [
          {
            "initial_rho_conv": 0.003,
            "initial_rho_dense": 0.01,
            "factor": 3.0,
            "maximum_center_attempts": 6
          }
        ]
      }
    ],
    "case_ids": [
      "conv3_baseline_v1_c1--sgd",
      "conv3_baseline_v1_c1--adam",
      "conv3_ours_v4_c1--sgd",
      "conv3_ours_v4_c1--adam",
      "conv3_legacy_v4_c0p25--sgd",
      "conv3_legacy_v4_c0p25--adam"
    ],
    "seeds": [
      0
    ],
    "conditional_expansion": {
      "enabled": true,
      "rule": "Within each surface, add one next factor-of-three value only on outer axes implicated by a passing boundary plateau, include the corner when both axes expand, and stop after one wave at no more than 16 cells. No passing core candidate does not expand; no second expansion or fallback is allowed.",
      "maximum_additional_jobs": 0
    },
    "expected_initial_job_count": 6,
    "maximum_total_job_count": 6,
    "undeclared_fields_fixed_by": [
      "docs/perfectdiode_conv3_learning_protocol.md",
      "configs/conv/perfectdiode_conv3_lr_operating_point_t8_k8_20260727_v1.json",
      "results/perfectdiode_conv3_lr_t8k8_20260727_v5/study.resolved.json",
      "results/perfectdiode_conv3_lr_t8k8_20260727_v5/surface_manifest.json"
    ]
  },
  "execution": {
    "launcher": [
      "/home/filip/server_code_conv_learning_rate_protocol/experiments/run_mnist_conv_perfectdiode_conv3_hparam_t8k8_local_20260727_v4.sh",
      "--dispatch"
    ],
    "collector": [
      "/home/filip/miniconda3/envs/py312/bin/python",
      "/tmp/pd_conv3_lr_t8k8_20260727_source_v5/experiments/run_mnist_conv_perfectdiode_hparam_v2_orchestration.py",
      "finalize-study",
      "--root",
      "/home/filip/server_code_conv_learning_rate_protocol/results/perfectdiode_conv3_lr_t8k8_20260727_v5"
    ],
    "validator": [
      "/home/filip/miniconda3/envs/py312/bin/python",
      "/tmp/pd_conv3_lr_t8k8_20260727_source_v5/experiments/run_mnist_conv_perfectdiode_hparam_v2_orchestration.py",
      "validate-study",
      "--root",
      "/home/filip/server_code_conv_learning_rate_protocol/results/perfectdiode_conv3_lr_t8k8_20260727_v5"
    ],
    "preflight": {
      "smoke_required": true,
      "tk_reference_required": true,
      "scheduled_run_preflight_required": true,
      "receipt_path": "results/perfectdiode_conv3_lr_t8k8_20260727_v5/preflight/scheduled-run-preflight.json"
    }
  },
  "storage": {
    "root_alias": "REPO_ROOT",
    "local_results_root": "results",
    "local_bundle_path": "results/perfectdiode_conv3_lr_t8k8_20260727_v5",
    "remote_staging": [
      {
        "host": "akibscomputer",
        "path": "/home/filiposana/results/perfectdiode_conv3_lr_t8k8_20260727_v5",
        "intended_local_destination": "results/perfectdiode_conv3_lr_t8k8_20260727_v5"
      }
    ]
  },
  "reporting": {
    "progress_tracker": "docs/current_experiments.md",
    "comparison_cards": [
      {
        "comparison_id": "perfectdiode-conv3-lr-ordinary-mnist-t8k8-20260727-v5",
        "card_schema_version": "drn-result-card/v1",
        "card_path": "result_registry/cards/diagnostics/perfectdiode-conv3-lr-ordinary-mnist-t8k8-20260727-v5.json",
        "review_path": "result_registry/reviews/perfectdiode-conv3-lr-ordinary-mnist-t8k8-20260727-v5.json",
        "final_results_anchor": "docs/results/index.md#perfectdiode-conv3-lr-ordinary-mnist-t8k8-20260727-v5"
      }
    ],
    "final_results_page": "docs/results/index.md"
  },
  "completion": {
    "required_coverage": "Exactly three passing initialization residual/gradient gates at T=8,K=8 and six hash-verified terminal surface outcomes; every promoted candidate completes exactly three epochs and 10314 successful steps, every selected epoch checkpoint passes T/K replay, and the complete Akib shard is repatriated and locally validated.",
    "allowed_exclusions": [
      "The single expansion wave is omitted when a passing core plateau does not implicate an outer edge, or when the core has no passing candidate.",
      "A full candidate is omitted only after its own structured safety-canary failure; failure of all six permitted legacy center attempts produces explicit zero candidate work for that surface.",
      "No official-test measurement, long confirmation, second expansion, fallback learning rate, cross-surface learning-rate transfer, paper-facing claim, or cross-scheme superiority claim is allowed."
    ],
    "failure_handling": "Fail closed on any plan, source, archive, runner, environment, initializer, split, cohort, batch-order, fixed T=8/K=8 gate, probe, canary, candidate, epochwise viability, post-training audit, completion, transfer, or output-hash mismatch. Resume only hash-verified immutable complete entries; restart every incomplete Adam entry from the shared initialization.",
    "local_validation_required": true,
    "review_required": true
  },
  "approval": {
    "status": "approved",
    "approved_by": "Filip",
    "approved_at": "2026-07-27T17:59:21+02:00",
    "amendment_of": "perfectdiode-conv3-lr-ordinary-mnist-t8k8-20260727-v4"
  }
}
```

The immutable execution source is commit
`0035d1485205c8e4efdabebeaa7524bad4fa9827`, with source-archive SHA-256
`69cf709fc775aa31028b52bc179dc29745dbcfb2d99d66eb35a3394c70729274`,
effective-code fingerprint
`8eb471f232a6ed2978feb6ec53b1b51b170931274d05362d473a795e074fa623`,
worker SHA-256
`02ec4cb71b7a556b7f615d1eecfa5e788dcb5e56c0b0308e8296fbb513d06786`,
and runner SHA-256
`f6e192b1c73ba36f09f0ea7ebf695c121f35f07698dee55b6236fd6377ba898c`.
The upstream automatic selector remains `T=4, K=4`; the operational point is
the distinct user-fixed `T=8, K=8` contract and must pass three fresh audits.
