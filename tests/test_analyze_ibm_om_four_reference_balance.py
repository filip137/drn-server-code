from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

import experiments.mnist_relu_drn.analyze_ibm_om_four_reference_balance as analysis
from experiments.artifacts import sha256_file
from experiments.mnist_relu_drn.ibm_om_four_reference_balance import (
    BALANCED_POLICY,
    POLICIES,
    RANDOM_POLICY,
)
from experiments.study_workflow import load_study_plan


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _metric(correct: int, *, prediction: str) -> dict:
    examples = 10000
    teacher_correct = 9726
    agreement = 9000
    return {
        "examples": examples,
        "student_correct": correct,
        "teacher_correct": teacher_correct,
        "teacher_agreement_count": agreement,
        "kl_teacher_student": 0.25,
        "raw_kl_teacher_student": 0.5,
        "student_accuracy": correct / examples,
        "teacher_accuracy": teacher_correct / examples,
        "teacher_agreement": agreement / examples,
        "raw_score_rms": 0.1,
        "calibrated_score_rms": 1.0,
        "teacher_logit_rms": 8.0,
        "fixed_logit_gain": 10.0,
        "prediction_sha256": prediction,
        "voltage": [
            {
                "layer": layer,
                "values": values,
                "mean": 0.0,
                "rms": 0.2,
                "standard_deviation": 0.2,
                "minimum": -0.3,
                "maximum": 0.3,
            }
            for layer, values in enumerate((15_680_000, 1_000_000, 200_000))
        ],
    }


def _summary() -> dict:
    heldout = []
    optimized_values = {policy: [] for policy in POLICIES}
    fixed_values = {policy: [] for policy in POLICIES}
    for assignment_index, seed in enumerate(analysis.EXPECTED_ASSIGNMENTS):
        arms = []
        metrics = {}
        for binding_index, binding in enumerate(POLICIES):
            for calibration_index, calibration in enumerate(POLICIES):
                correct = 7000 + 100 * assignment_index + 30 * binding_index + calibration_index
                metric = _metric(
                    correct,
                    prediction=f"{seed + 10 * binding_index + calibration_index:064x}",
                )
                metrics[(binding, calibration)] = metric
                arms.append(
                    {
                        "assignment_seed": seed,
                        "binding_policy": binding,
                        "calibration_policy": calibration,
                        "ideal_bounded_continuous_test": metric,
                    }
                )
        paired = {}
        for calibration in POLICIES:
            random = metrics[(RANDOM_POLICY, calibration)]
            balanced = metrics[(BALANCED_POLICY, calibration)]
            delta = balanced["student_accuracy"] - random["student_accuracy"]
            paired[calibration] = {
                "balanced_minus_random_accuracy": delta,
                "prediction_flip_count": 25,
                "prediction_flip_fraction": 0.0025,
            }
            fixed_values[calibration].append(delta)
        for policy in POLICIES:
            optimized_values[policy].append(
                metrics[(policy, policy)]["student_accuracy"]
            )
        heldout.append(
            {
                "assignment_seed": seed,
                "arms": arms,
                "paired_binding_effect": paired,
            }
        )
    aggregate = {
        "scheme_optimized": {
            policy: analysis._accuracy_summary(values)
            for policy, values in optimized_values.items()
        },
        "fixed_calibration": {
            policy: {
                "balanced_minus_random_accuracy": analysis._accuracy_summary(values),
                "per_assignment": values,
            }
            for policy, values in fixed_values.items()
        },
    }
    balanced_mean = aggregate["scheme_optimized"][BALANCED_POLICY]["mean"]
    aggregate["balanced_accuracy_gate"] = {
        "threshold": 0.90,
        "observed_mean": balanced_mean,
        "passed": balanced_mean >= 0.90,
    }
    return {
        "metric_definition_id": "ibm_om.reference_balanced_continuous_init.v1",
        "contract": {"device": {"evidence_class": "model_based_aihwkit_preset"}},
        "heldout": heldout,
        "aggregate": aggregate,
        "exclusions": {
            "optimizer_updates": 0,
            "quantization_enabled": False,
            "program_verify_enabled": False,
            "hwa_enabled": False,
            "deployment_write_noise": 0.0,
            "inference_read_noise": 0.0,
            "retention_or_drift": False,
        },
    }


def test_metrics_recompute_exact_counts_and_both_aggregate_families() -> None:
    summary = _summary()
    result = analysis._metrics(summary)

    first = result["scheme_optimized"][RANDOM_POLICY]["per_assignment"][0]
    assert first["correct"] == 7000
    assert first["examples"] == 10000
    assert result["fixed_calibration"][RANDOM_POLICY]["per_assignment"][0][
        "prediction_flip_count"
    ] == 25

    summary["aggregate"]["fixed_calibration"][RANDOM_POLICY][
        "per_assignment"
    ][0] += 0.01
    with pytest.raises(ValueError, match="fixed-calibration aggregate"):
        analysis._metrics(summary)


def test_result_metrics_must_equal_the_scientific_summary_compaction() -> None:
    summary = _summary()
    expected = analysis._expected_result_metrics(summary)
    bundle = {"result": {"metrics": deepcopy(expected)}}
    analysis._validate_result_metrics(bundle, summary)

    bundle["result"]["metrics"]["coverage_valid"] = False
    with pytest.raises(ValueError, match="result.json metrics"):
        analysis._validate_result_metrics(bundle, summary)


def _study(tmp_path: Path) -> tuple[Path, dict]:
    study_root = tmp_path / analysis.EXPECTED_STUDY_ID
    source_plan = tmp_path / "source-plan.json"
    source_config = tmp_path / "config.json"
    _write_json(source_config, {"frozen": True})
    plan = {
        "schema_version": 1,
        "study_id": analysis.EXPECTED_STUDY_ID,
        "title": "Analyzer provenance fixture",
        "hypothesis": "The exact declared plan is retained.",
        "motivation": "Exercise fail-closed study provenance.",
        "evidence_class": analysis.EXPECTED_EVIDENCE_CLASS,
        "completion_criteria": ["The analyzer accepts only the declared runs."],
        "analysis_plan": ["Validate every provenance hash."],
        "arms": [
            {
                "arm_id": arm_id,
                "description": f"Canonical {arm_id} arm.",
                "experiment_id": analysis.EXPECTED_EXPERIMENT_ID,
                "mode": "validate",
                "configs": [source_config.name],
            }
            for arm_id in analysis.EXPECTED_ARM_IDS
        ],
    }
    _write_json(source_plan, plan)
    study = {
        "schema": "ebl.study",
        **load_study_plan(source_plan),
        "prepared_at": "2026-08-28T00:00:00+00:00",
    }
    study_path = study_root / "study.json"
    _write_json(study_path, study)
    return study_root, study


def test_study_context_requires_exact_canonical_arm_and_hashes(tmp_path: Path) -> None:
    study_root, study = _study(tmp_path)
    run_dir = study_root / "runs" / "main" / "run-main"
    run_dir.mkdir(parents=True)
    context = {
        "study_id": analysis.EXPECTED_STUDY_ID,
        "arm_id": "main",
        "evidence_class": analysis.EXPECTED_EVIDENCE_CLASS,
        "study_sha256": sha256_file(study_root / "study.json"),
        "source_plan_sha256": study["source_plan"]["sha256"],
        "source_config_sha256": study["arms"][0]["configs"][0]["sha256"],
    }
    observed, _ = analysis._validate_study_context(
        run_dir, {"study": context}, expected_arm_id="main"
    )
    assert observed == context

    context["arm_id"] = "replay"
    with pytest.raises(ValueError, match="Manifest study provenance"):
        analysis._validate_study_context(
            run_dir, {"study": context}, expected_arm_id="main"
        )


def test_analyze_executes_arm_specific_validation_and_rejects_same_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "same-run"
    run_dir.mkdir()
    calls = []

    def fake_bundle(path: Path, *, expected_arm_id: str):
        calls.append(expected_arm_id)
        return ({"manifest": {"run_id": expected_arm_id}}, {})

    monkeypatch.setattr(analysis, "_one_run", lambda _path: run_dir)
    monkeypatch.setattr(analysis, "_validate_bundle", fake_bundle)

    with pytest.raises(ValueError, match="distinct native run directories"):
        analysis.analyze(Path("main"), Path("replay"))
    assert calls == ["main", "replay"]


def test_development_selection_recomputes_cartesian_argmin() -> None:
    grid = [0.125, 0.25, 0.5, 1.0]
    candidates = []
    for index, (left, right) in enumerate(
        (pair for left in grid for right in grid for pair in [(left, right)])
    ):
        candidates.append(
            {
                "scale_fractions": [left, right],
                "calibration": {
                    "student_accuracy": 0.5 + index / 1000.0,
                    "calibrated_kl": 1.0 - index / 1000.0,
                    "gain": 2.0,
                },
                "target_hashes": ["2" * 64, "3" * 64],
                "baseline_contrast_rms": [0.1, 0.2],
            }
        )
    selected_index = 15
    summary = {
        "development": {
            policy: {
                "binding_policy": policy,
                "selection_domain": "development_assignment_86001_calibration_subset",
                "selection_metric": "mapped_accuracy_then_calibrated_kl_then_scale_pair",
                "selected_index": selected_index,
                "selected": candidates[selected_index],
                "selected_mapping": {
                    "target_hashes": candidates[selected_index]["target_hashes"],
                    "scale_fractions": candidates[selected_index]["scale_fractions"],
                },
                "candidates": deepcopy(candidates),
            }
            for policy in POLICIES
        }
    }
    config = {"student": {"mapping": {"scale_fractions": grid}}}
    analysis._validate_development(summary, config)

    summary["development"][RANDOM_POLICY]["selected_index"] = 0
    with pytest.raises(ValueError, match="selection does not recompute"):
        analysis._validate_development(summary, config)
