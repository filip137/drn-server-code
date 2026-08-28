from __future__ import annotations

import pytest
import torch

from experiments.mnist_relu_drn.ibm_om_local_reference_compensation import (
    CONTROL_POLICY,
)
from experiments.mnist_relu_drn.ibm_om_local_reference_compensation_runtime import (
    _assert_matched_offsets,
    _compact_mapping,
    _material_artifact_contract,
    _paired_effect,
)


def _mapping(*, policy: str = CONTROL_POLICY, offset: str = "a") -> dict:
    digest = offset * 64
    return {
        "baseline_policy": policy,
        "assignment_seed": 87001,
        "identity_binding": "sampled_order_no_permutation",
        "scale_fractions": [0.25, 0.5],
        "target_hashes": ["b" * 64, "c" * 64],
        "shared_matching": {
            "matched_variable": "offset_d",
            "offset_hashes_equal": True,
        },
        "layers": [
            {
                "layer": index,
                "baseline_contrast": {"rms": 0.1 + index},
                "offset_contrast": {"rms": 0.2 + index},
                "realized_contrast": {"rms": 0.3 + index},
                "baseline_loading": {"mean": 1.0},
                "quad_loading": {"mean": 1.2},
                "logical_sign_flip_count": index,
                "logical_sign_flip_fraction_nonzero": index / 10.0,
                "full_g_decomposition_max_abs_residual": 0.0,
                "bound_violation_count": 0,
                "hashes": {
                    "baseline": str(index) * 64,
                    "offset": digest,
                    "conductance": str(index + 2) * 64,
                },
            }
            for index in range(2)
        ],
    }


def _metric(*, correct: int, examples: int = 10) -> dict:
    return {
        "examples": examples,
        "student_correct": correct,
        "student_accuracy": correct / examples,
    }


def test_paired_effect_is_derived_from_exact_counts_and_predictions() -> None:
    control_prediction = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    treatment_prediction = torch.tensor([0, 2, 2, 3, 0, 5, 6, 7, 8, 9])

    effect = _paired_effect(
        _metric(correct=7),
        _metric(correct=9),
        control_prediction,
        treatment_prediction,
    )

    assert effect == {
        "treatment_minus_control_correct": 2,
        "examples": 10,
        "treatment_minus_control_accuracy": 0.2,
        "prediction_flip_count": 2,
        "prediction_flip_fraction": 0.2,
    }


def test_paired_effect_rejects_a_different_cohort_or_inexact_accuracy() -> None:
    predictions = torch.arange(10)
    with pytest.raises(ValueError, match="same nonempty cohort"):
        _paired_effect(
            _metric(correct=7),
            _metric(correct=8, examples=9),
            predictions,
            predictions,
        )

    treatment = _metric(correct=8)
    treatment["student_accuracy"] = 0.81
    with pytest.raises(ValueError, match="Exact correct-count delta"):
        _paired_effect(
            _metric(correct=7), treatment, predictions, predictions
        )


def test_matched_offset_guard_is_bit_exact_per_layer() -> None:
    control = _mapping(offset="a")
    treatment = _mapping(policy="local_min_l2_exact_zero", offset="a")
    assert _assert_matched_offsets(
        control, treatment, context="fixture"
    ) == ("a" * 64, "a" * 64)

    treatment["layers"][1]["hashes"]["offset"] = "d" * 64
    with pytest.raises(RuntimeError, match="Matched-offset invariant"):
        _assert_matched_offsets(control, treatment, context="fixture")


def test_compact_mapping_preserves_full_g_and_matching_invariants() -> None:
    mapping = _mapping()
    compact = _compact_mapping(mapping)

    assert compact["baseline_policy"] == CONTROL_POLICY
    assert compact["identity_binding"] == "sampled_order_no_permutation"
    assert compact["offset_hashes"] == ["a" * 64, "a" * 64]
    assert compact["shared_matching"]["matched_variable"] == "offset_d"
    assert compact["layers"][0]["full_g_decomposition_max_abs_residual"] == 0.0
    assert compact["layers"][0]["hashes"]["conductance"] == "2" * 64

    del mapping["baseline_policy"]
    with pytest.raises(KeyError):
        _compact_mapping(mapping)


def test_native_artifact_contract_has_no_identity_reassignment_artifact() -> None:
    contract = _material_artifact_contract(
        development_assignments=1, heldout_assignments=3
    )
    assert contract == {
        "population_artifacts": 4,
        "baseline_plan_artifacts": 8,
        "mapping_artifacts": 14,
        "identity_reassignment_artifacts": 0,
        "scientific_summary_artifacts": 1,
    }

    with pytest.raises(ValueError, match="one development assignment"):
        _material_artifact_contract(
            development_assignments=2, heldout_assignments=3
        )
