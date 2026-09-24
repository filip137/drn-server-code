# Conv3 p99 read-noise diagnostic

```json
{
  "schema_version": "experiment-run-plan/v1",
  "experiment_id": "eqprop-conv3-p99-read-noise-5em4-20260922-v1",
  "title": "Conv3 p99 legacy and ours read noise 5e-4",
  "evidence_scope": "diagnostic",
  "hypothesis": {
    "statement": "Assess full-budget training stability at the largest measured clean p99 beta under read noise 5e-4.",
    "control": "Independent completion per case; p95 parent is contextual evidence.",
    "treatments": [
      "legacy",
      "ours"
    ],
    "expected_direction": "No assumed superiority; assess finite training and validation trajectory.",
    "decision_rule": {
      "metric": "Final validation drop from own best",
      "split": "Fixed ordinary-MNIST 55000/5000 split",
      "checkpoint_role": "final and best validation",
      "seed_aggregation": "seed0 only",
      "comparison": "Complete30 finite epochs and final drop strictly less than5pp",
      "minimum_meaningful_effect": 5,
      "effect_units": "percentage points"
    },
    "supports_if": "Both cases complete30 finite epochs and pass final-drop screen.",
    "does_not_support_if": "Nonfinite training or final drop at least5pp; stop both on any failure.",
    "inconclusive_if": "Missing or unverifiable coverage; cancelled sibling remains partial."
  },
  "sweep": {
    "config_path": "results/eqprop-conv3-p99-read-noise-5em4-20260922-v1/cases.json",
    "config_sha256": "517fe4203bc9af0d25eff545cd4045c54ba5ef9a437fefe4099ba389ef68ba43",
    "manifest_path": "results/eqprop-conv3-p99-read-noise-5em4-20260922-v1/cases.json",
    "manifest_sha256": "517fe4203bc9af0d25eff545cd4045c54ba5ef9a437fefe4099ba389ef68ba43",
    "manifest_job_count_pointer": "/entries",
    "axes": [
      {
        "name": "scheme",
        "config_path": "/entries",
        "ordered_values": [
          "legacy",
          "ours"
        ]
      }
    ],
    "case_ids": [
      "conv3_legacy_p99_sigma5em4_seed0",
      "conv3_ours_p99_sigma5em4_seed0"
    ],
    "seeds": [
      0
    ],
    "conditional_expansion": {
      "enabled": false,
      "rule": null,
      "maximum_additional_jobs": 0
    },
    "expected_initial_job_count": 2,
    "maximum_total_job_count": 2,
    "undeclared_fields_fixed_by": [
      "docs/conv_paper_one_seed_bptt_eqprop_protocol.md",
      "configs/conv/eqprop_conv3_p99_read_noise_5em4_20260922_v1/conv3_legacy_p99_sigma5em4_seed0.json",
      "configs/conv/eqprop_conv3_p99_read_noise_5em4_20260922_v1/conv3_ours_p99_sigma5em4_seed0.json"
    ]
  },
  "execution": {
    "launcher": [
      "bash",
      "results/eqprop-conv3-p99-read-noise-5em4-20260922-v1/run_pair.sh"
    ],
    "collector": [
      "python",
      "/home/filip/server_code_conv_learning_rate_protocol/skills/sync-remote-results/scripts/sync_remote_results.py"
    ],
    "validator": [
      "python",
      "results/eqprop-conv3-p99-read-noise-5em4-20260922-v1/validate_results.py"
    ],
    "preflight": {
      "smoke_required": true,
      "tk_reference_required": false,
      "scheduled_run_preflight_required": true,
      "receipt_path": "results/eqprop-conv3-p99-read-noise-5em4-20260922-v1/preflight-aggregate.json",
      "receipt_schemas": {
        "smoke": "jean-zay-pre-submit-gate/v1",
        "tk_reference": null,
        "scheduled_run_preflight": "scheduled-run-preflight/v1"
      },
      "receipt_bindings": {
        "smoke": {
          "receipt_path": "results/eqprop-conv3-p99-read-noise-5em4-20260922-v1/canary-validation-02-gate.json",
          "subject_id": "eqprop-conv3-p99-read-noise-5em4-20260922-v1",
          "producer_source_id": "022098cfe6eb151dff3f7403d3f4b358709152b9"
        },
        "tk_reference": null,
        "scheduled_run_preflight": {
          "receipt_path": "results/eqprop-conv3-p99-read-noise-5em4-20260922-v1/scheduled-preflight-01.json",
          "subject_id": null,
          "producer_source_id": null
        }
      }
    }
  },
  "storage": {
    "root_alias": "REPO_ROOT",
    "local_results_root": "results",
    "local_bundle_path": "results/eqprop-conv3-p99-read-noise-5em4-20260922-v1",
    "remote_staging": [
      {
        "host": "jean-zay",
        "path": "/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/eqprop-conv3-p99-read-noise-5em4-20260922-v1",
        "intended_local_destination": "results/eqprop-conv3-p99-read-noise-5em4-20260922-v1"
      }
    ]
  },
  "reporting": {
    "progress_tracker": "docs/current_experiments.md",
    "comparison_cards": [
      {
        "comparison_id": "eqprop-conv3-p99-read-noise-5em4-20260922-v1",
        "card_schema_version": "drn-result-card/v1",
        "card_path": "result_registry/cards/diagnostics/eqprop-conv3-p99-read-noise-5em4-20260922-v1.json",
        "review_path": "result_registry/reviews/eqprop-conv3-p99-read-noise-5em4-20260922-v1.json",
        "final_results_anchor": "docs/results/index.md#eqprop-conv3-p99-read-noise-5em4-20260922-v1"
      }
    ],
    "final_results_page": "docs/results/index.md"
  },
  "completion": {
    "required_coverage": "Exactly legacy and ours, seed0,30epochs at read_noise_std0.0005; terminal failure and cancelled sibling are retained.",
    "allowed_exclusions": [],
    "failure_handling": "Slurm srun --kill-on-bad-exit=1 stops both workers on any worker failure; no numerical retries. Operational armed-attempt errors stop and report.",
    "local_validation_required": true,
    "review_required": true
  },
  "approval": {
    "status": "approved",
    "approved_by": "Filip; yes after resolved scientific summary and two-V100 execution proposal",
    "approved_at": "2026-09-22T10:15:46.923486+00:00",
    "amendment_of": null
  }
}
```

Inherit parent p95 sigma5e-4 configs exactly except beta and descriptive metadata.
Legacy injectedB0.1, base2.44140625e-5; ours injectedB0.987333678708, base0.0154270887298125.
Seed0, same saved initializer, fresh Adam and exact weight rates; frozen zero biases,
weights[0,100], inputgain360, perfectdiode, float64 centered frozen-current EqProp,
reset each batch, T=K=8. Independent Gaussian noninput endpoint readout noise only;
noise seed2026081601. Relaxation and validation clean. No official test.
Reuse accepted shared T/K authority in matching protocol; no changed initializer,
architecture, amplification, inputgain, relaxation lengths or gradient algorithm.
Retain finite-T residual caveat. p99 is clean calibration, no guarantee under noise.
One six-hour fmu@v100/gpu_p13/qos_gpu-t3/v100-16g allocation, two tasks, each1GPU/4CPUs.
Canary uses same two-worker allocation and one training/validation batch per case.
Production has no epoch10/20 stops. Expected5-6h plus queue. Cap12GPUh production,
1GPUh canary; hard monitor deadline2026-09-23T12:00:00Z, scheduler-start cap900s.
User request and yes authorize scientific summary and execution unchanged.
