from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

import experiments.mnist_relu_drn.analyze_ibm_om_local_reference_compensation as analysis
from experiments.artifacts import sha256_file
from experiments.mnist_relu_drn.ibm_om_local_reference_compensation import (
    CONTROL_POLICY,
    POLICIES,
    TREATMENT_POLICY,
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
    scheme_values = {policy: [] for policy in POLICIES}
    fixed_effects = {policy: [] for policy in POLICIES}
    for assignment_index, seed in enumerate(analysis.EXPECTED_ASSIGNMENTS):
        arms = []
        metrics = {}
        offsets = {
            calibration: [
                f"{seed + 100 * calibration_index:064x}",
                f"{seed + 100 * calibration_index + 1:064x}",
            ]
            for calibration_index, calibration in enumerate(POLICIES)
        }
        for baseline_index, baseline in enumerate(POLICIES):
            for calibration_index, calibration in enumerate(POLICIES):
                correct = (
                    7000
                    + 100 * assignment_index
                    + 30 * baseline_index
                    + calibration_index
                )
                metric = _metric(
                    correct,
                    prediction=(
                        f"{seed + 10 * baseline_index + calibration_index:064x}"
                    ),
                )
                metrics[(baseline, calibration)] = metric
                arms.append(
                    {
                        "assignment_seed": seed,
                        "baseline_policy": baseline,
                        "calibration_policy": calibration,
                        "mapping": {"offset_hashes": offsets[calibration]},
                        "ideal_bounded_continuous_test": metric,
                    }
                )
        paired = {}
        for calibration in POLICIES:
            control = metrics[(CONTROL_POLICY, calibration)]
            treatment = metrics[(TREATMENT_POLICY, calibration)]
            correct_delta = treatment["student_correct"] - control["student_correct"]
            effect = {
                "treatment_minus_control_correct": correct_delta,
                "examples": 10000,
                "treatment_minus_control_accuracy": correct_delta / 10000,
                "prediction_flip_count": 25,
                "prediction_flip_fraction": 0.0025,
                "identical_offset_hashes": offsets[calibration],
            }
            paired[calibration] = effect
            fixed_effects[calibration].append(effect)
        for policy in POLICIES:
            scheme_values[policy].append(
                metrics[(policy, policy)]["student_accuracy"]
            )
        heldout.append(
            {
                "assignment_seed": seed,
                "arms": arms,
                "paired_baseline_effect": paired,
            }
        )

    scheme = {
        policy: analysis._accuracy_summary(values)
        for policy, values in scheme_values.items()
    }
    fixed = {
        calibration: {
            "treatment_minus_control_accuracy": analysis._accuracy_summary(
                [item["treatment_minus_control_accuracy"] for item in effects]
            ),
            "per_assignment": effects,
        }
        for calibration, effects in fixed_effects.items()
    }
    primary_deltas = [
        item["treatment_minus_control_accuracy"]
        for item in fixed_effects[CONTROL_POLICY]
    ]
    primary_summary = analysis._accuracy_summary(primary_deltas)
    secondary_deltas = []
    secondary_correct = []
    for block in heldout:
        control = next(
            arm["ideal_bounded_continuous_test"]
            for arm in block["arms"]
            if arm["baseline_policy"] == CONTROL_POLICY
            and arm["calibration_policy"] == CONTROL_POLICY
        )
        treatment = next(
            arm["ideal_bounded_continuous_test"]
            for arm in block["arms"]
            if arm["baseline_policy"] == TREATMENT_POLICY
            and arm["calibration_policy"] == TREATMENT_POLICY
        )
        delta = treatment["student_correct"] - control["student_correct"]
        secondary_correct.append(delta)
        secondary_deltas.append(delta / 10000)
    treatment_mean = scheme[TREATMENT_POLICY]["mean"]
    aggregate = {
        "scheme_optimized": scheme,
        "fixed_calibration": fixed,
        "primary_fixed_control_calibration": {
            "calibration_policy": CONTROL_POLICY,
            "treatment_minus_control_accuracy": primary_summary,
            "total_treatment_minus_control_correct": sum(
                item["treatment_minus_control_correct"]
                for item in fixed_effects[CONTROL_POLICY]
            ),
            "total_examples": 30000,
        },
        "primary_effect_gate": {
            "threshold": 0.05,
            "observed_mean": primary_summary["mean"],
            "passed": primary_summary["mean"] >= 0.05,
        },
        "secondary_scheme_optimized": {
            "control_accuracy": scheme[CONTROL_POLICY],
            "treatment_accuracy": scheme[TREATMENT_POLICY],
            "treatment_minus_control_accuracy": analysis._accuracy_summary(
                secondary_deltas
            ),
            "total_treatment_minus_control_correct": sum(secondary_correct),
            "total_examples": 30000,
        },
        "treatment_accuracy_gate": {
            "threshold": 0.90,
            "observed_mean": treatment_mean,
            "passed": treatment_mean >= 0.90,
        },
    }
    return {
        "metric_definition_id": analysis.EXPECTED_METRIC_DEFINITION,
        "contract": {"device": {"evidence_class": analysis.EXPECTED_EVIDENCE_CLASS}},
        "artifact_contract": {
            "population_artifacts": 4,
            "baseline_plan_artifacts": 8,
            "mapping_artifacts": 14,
            "identity_reassignment_artifacts": 0,
            "scientific_summary_artifacts": 1,
        },
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
            "identity_reassignment": False,
        },
    }


def test_metrics_recompute_exact_primary_secondary_counts_and_gates() -> None:
    result = analysis._metrics(_summary())

    first = result["scheme_optimized"][CONTROL_POLICY]["per_assignment"][0]
    assert first["correct"] == 7000
    assert first["examples"] == 10000
    assert result["primary_fixed_control_calibration"]["per_assignment"][0][
        "treatment_minus_control_correct"
    ] == 30
    assert result["secondary_scheme_optimized"]["per_assignment"][0][
        "treatment_minus_control_correct"
    ] == 31
    assert result["gates"]["primary_effect"]["passed"] is False


def test_metrics_reject_tampered_aggregate_and_offset_matching() -> None:
    summary = _summary()
    summary["aggregate"]["primary_fixed_control_calibration"][
        "total_treatment_minus_control_correct"
    ] += 1
    with pytest.raises(ValueError, match="primary fixed-calibration aggregate"):
        analysis._metrics(summary)

    summary = _summary()
    summary["heldout"][0]["arms"][2]["mapping"]["offset_hashes"] = [
        "f" * 64,
        summary["heldout"][0]["arms"][2]["mapping"]["offset_hashes"][1],
    ]
    with pytest.raises(ValueError, match="paired fixed-calibration effect"):
        analysis._metrics(summary)


def test_result_metrics_must_equal_scientific_summary_compaction() -> None:
    summary = _summary()
    bundle = {
        "result": {"metrics": deepcopy(analysis._expected_result_metrics(summary))}
    }
    analysis._validate_result_metrics(bundle, summary)

    bundle["result"]["metrics"]["artifact_contract"]["mapping_artifacts"] = 13
    with pytest.raises(ValueError, match="result.json metrics"):
        analysis._validate_result_metrics(bundle, summary)


def test_artifact_contract_rejects_any_extra_or_missing_material() -> None:
    paths = [Path("x")]
    canonical = {
        "ibm_om_population": paths * 4,
        "ibm_om_population_receipt": paths * 4,
        "ibm_om_local_baseline_plan": paths * 8,
        "ibm_om_local_baseline_plan_receipt": paths * 8,
        "ibm_om_full_g_mapping": paths * 14,
        "ibm_om_full_g_mapping_receipt": paths * 14,
        "scientific_summary": paths,
    }
    analysis._require_artifact_contract(canonical)
    tampered = dict(canonical)
    tampered["ibm_om_full_g_mapping"] = paths * 13
    with pytest.raises(ValueError, match="kind/count contract"):
        analysis._require_artifact_contract(tampered)


def test_development_selection_uses_declared_tie_break() -> None:
    candidates = []
    for left in (0.125, 0.25, 0.5, 1.0):
        for right in (0.125, 0.25, 0.5, 1.0):
            candidates.append(
                {
                    "scale_fractions": [left, right],
                    "calibration": {
                        "student_accuracy": 0.8,
                        "calibrated_kl": 0.2,
                    },
                }
            )
    assert analysis._selected_candidate_index(candidates) == 0
    candidates[-1]["calibration"]["student_accuracy"] = 0.81
    assert analysis._selected_candidate_index(candidates) == 15


def _study(tmp_path: Path) -> tuple[Path, dict]:
    study_root = tmp_path / analysis.EXPECTED_STUDY_ID
    source_plan = tmp_path / "source-plan.json"
    source_config = tmp_path / "continuous.json"
    _write_json(source_config, {"frozen": True})
    plan = {
        "schema_version": 1,
        "study_id": analysis.EXPECTED_STUDY_ID,
        "title": "Analyzer provenance fixture",
        "hypothesis": "The exact declared plan is retained.",
        "motivation": "Exercise fail-closed study provenance.",
        "evidence_class": analysis.EXPECTED_EVIDENCE_CLASS,
        "completion_criteria": ["The analyzer accepts only declared runs."],
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
    _write_json(study_root / "study.json", study)
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


def test_analyze_rejects_same_native_run_even_with_mocked_bundles(
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
