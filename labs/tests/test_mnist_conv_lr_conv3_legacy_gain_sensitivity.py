from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from experiments.mnist_conv.lr_conv3_legacy_gain_sensitivity import (
    GainSensitivitySpec,
    _verify_minibatch_prefix,
)
from experiments.mnist_conv.lr_stages import execute_candidate_entry
from experiments.mnist_conv.lr_study_spec import LRStudySpec


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    REPO_ROOT
    / "configs/conv/"
    "hardsigmoid_lr_conv3_legacy_gain200_rho5e5_3e4_bs16_v1.json"
)
V7_CONFIG = (
    REPO_ROOT
    / "configs/conv/"
    "hardsigmoid_lr_conv3_scheme_two_rho_constant_sgd_bs16_v7.json"
)


def test_gain_sensitivity_has_stable_separate_identity() -> None:
    spec = GainSensitivitySpec.from_path(CONFIG)
    assert spec.sensitivity_id == (
        "lrgain_0dc9dc42f5cb2044a9c4d7e4d4a1ab8f"
        "ab4e3f4b4f356837fde4de1c3f74e303"
    )
    assert spec.data["controlled_change"] == {
        "field": "input_gain",
        "value": 200.0,
        "reason": "recorded_best_legacy_low_gain_diagnostic",
        "reuse_exact_source_learning_rate_vector": True,
        "reprobe_learning_rates": False,
        "rho_labels_are_source_provenance_only": True,
        "all_other_row_fields_unchanged": True,
    }
    assert spec.data["artifacts"]["mutate_parent_study"] is False
    assert spec.data["artifacts"]["mutate_source_rescue"] is False


def test_gain_sensitivity_rejects_an_unplanned_gain() -> None:
    value = json.loads(CONFIG.read_text())
    value["controlled_change"]["value"] = 250.0
    with pytest.raises(ValueError, match="controlled_change.value"):
        GainSensitivitySpec.from_dict(value)


def test_controlled_candidate_requires_both_row_and_run_spec(
    tmp_path: Path,
) -> None:
    study = LRStudySpec.from_path(V7_CONFIG)
    row = next(
        item
        for item in study.rows
        if item["row_id"] == "conv3_legacy_v4_c0p25"
    )
    controlled = copy.deepcopy(row)
    controlled["input_gain"] = 200.0
    with pytest.raises(
        ValueError, match="controlled_runtime_row and controlled_run_spec"
    ):
        execute_candidate_entry(
            study.data,
            tmp_path,
            row["row_id"],
            "controlled",
            data_root=tmp_path,
            download=False,
            device="cpu",
            controlled_runtime_row=controlled,
        )


def test_controlled_candidate_can_change_only_input_gain(
    tmp_path: Path,
) -> None:
    study = LRStudySpec.from_path(V7_CONFIG)
    row = next(
        item
        for item in study.rows
        if item["row_id"] == "conv3_legacy_v4_c0p25"
    )
    controlled = copy.deepcopy(row)
    controlled["input_gain"] = 200.0
    controlled["training_iterations"] = 8
    with pytest.raises(ValueError, match="differ only"):
        execute_candidate_entry(
            study.data,
            tmp_path,
            row["row_id"],
            "controlled",
            data_root=tmp_path,
            download=False,
            device="cpu",
            controlled_runtime_row=controlled,
            controlled_run_spec={"schema_version": "test/v1"},
            controlled_summary_schema="test-result/v1",
        )


def test_early_safety_stop_accepts_only_an_exact_minibatch_prefix() -> None:
    source = {"batches": [[1, 2], [3, 4], [5, 6]]}
    observed = {"batches": [[1, 2], [3, 4]]}
    assert _verify_minibatch_prefix(
        candidate_minibatches=observed,
        source_minibatches=source,
        training_completed=False,
    ) == {
        "observed_steps": 2,
        "source_steps": 3,
        "prefix_equal": True,
        "complete_order_equal": False,
    }
    with pytest.raises(RuntimeError, match="exact source minibatch"):
        _verify_minibatch_prefix(
            candidate_minibatches=observed,
            source_minibatches=source,
            training_completed=True,
        )
    with pytest.raises(RuntimeError, match="exact source minibatch"):
        _verify_minibatch_prefix(
            candidate_minibatches={"batches": [[1, 2], [9, 9]]},
            source_minibatches=source,
            training_completed=False,
        )
