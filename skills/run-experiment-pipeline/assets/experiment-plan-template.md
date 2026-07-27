# Experiment Plan: REPLACE_ME_TITLE

The JSON contract below is the reviewed study design. The linked executable
config and immutable manifest are the numeric execution authority. Progress is
tracked separately in `docs/current_experiments.md`. Approval freezes this
plan but does not make it launch-ready: before production launch, the matching
three-field tracker entry must pass the pipeline's exact-ID gate. Never write
mutable run status into this plan.

```json
{
  "schema_version": "experiment-run-plan/v1",
  "experiment_id": "replace-me-experiment-id",
  "title": "REPLACE_ME_TITLE",
  "evidence_scope": "diagnostic",
  "hypothesis": {
    "statement": "REPLACE_ME",
    "control": "REPLACE_ME",
    "treatments": [
      "REPLACE_ME"
    ],
    "expected_direction": "REPLACE_ME",
    "decision_rule": {
      "metric": "REPLACE_ME",
      "split": "REPLACE_ME",
      "checkpoint_role": "REPLACE_ME",
      "seed_aggregation": "REPLACE_ME",
      "comparison": "REPLACE_ME",
      "minimum_meaningful_effect": 0.0,
      "effect_units": "REPLACE_ME"
    },
    "supports_if": "REPLACE_ME",
    "does_not_support_if": "REPLACE_ME",
    "inconclusive_if": "REPLACE_ME"
  },
  "sweep": {
    "config_path": "configs/REPLACE_ME.json",
    "config_sha256": null,
    "manifest_path": "results/REPLACE_ME/manifest.json",
    "manifest_sha256": null,
    "manifest_job_count_pointer": "/entries",
    "axes": [
      {
        "name": "REPLACE_ME",
        "config_path": "/REPLACE_ME",
        "ordered_values": [
          "REPLACE_ME"
        ]
      }
    ],
    "case_ids": [
      "REPLACE_ME"
    ],
    "seeds": [
      0
    ],
    "conditional_expansion": {
      "enabled": false,
      "rule": null,
      "maximum_additional_jobs": 0
    },
    "expected_initial_job_count": 1,
    "maximum_total_job_count": 1,
    "undeclared_fields_fixed_by": [
      "docs/REPLACE_ME_PROTOCOL.md",
      "configs/REPLACE_ME.json"
    ]
  },
  "execution": {
    "launcher": [
      "python",
      "-m",
      "experiments.REPLACE_ME"
    ],
    "collector": [
      "python",
      "-m",
      "experiments.REPLACE_ME",
      "collect"
    ],
    "validator": [
      "python",
      "REPLACE_ME_VALIDATOR.py"
    ],
    "preflight": {
      "smoke_required": true,
      "tk_reference_required": true,
      "scheduled_run_preflight_required": false,
      "receipt_path": "results/REPLACE_ME/preflight/receipt.json"
    }
  },
  "storage": {
    "root_alias": "REPO_ROOT",
    "local_results_root": "results",
    "local_bundle_path": "results/REPLACE_ME",
    "remote_staging": []
  },
  "reporting": {
    "progress_tracker": "docs/current_experiments.md",
    "comparison_cards": [
      {
        "comparison_id": "replace-me-comparison-id",
        "card_schema_version": "REPLACE_ME_SUPPORTED_CARD_SCHEMA",
        "card_path": "result_registry/cards/diagnostics/replace-me-comparison-id.json",
        "review_path": "result_registry/reviews/replace-me-comparison-id.json",
        "final_results_anchor": "docs/results/index.md#replace-me-comparison-id"
      }
    ],
    "final_results_page": "docs/results/index.md"
  },
  "completion": {
    "required_coverage": "REPLACE_ME",
    "allowed_exclusions": [],
    "failure_handling": "REPLACE_ME",
    "local_validation_required": true,
    "review_required": true
  },
  "approval": {
    "status": "draft",
    "approved_by": null,
    "approved_at": null,
    "amendment_of": null
  }
}
```

## Optional non-decision-making notes

Add scheduling preferences, secondary plots, or context here. Do not put
scientific values here if they are absent from the contract and canonical
config.
