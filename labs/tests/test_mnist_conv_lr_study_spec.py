from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from experiments.mnist_conv.lr_study_spec import (
    FROZEN_ROWS,
    LRStudySpec,
    study_fingerprint,
)
from experiments.mnist_conv.specs import SpecValidationError


REPO_ROOT = Path(__file__).resolve().parents[2]
STUDY_CONFIG = REPO_ROOT / "configs/conv/hardsigmoid_lr_study_sgd_bs16_v1.json"


def study_value() -> dict:
    return json.loads(STUDY_CONFIG.read_text())


def test_frozen_study_config_loads_all_six_rows() -> None:
    spec = LRStudySpec.from_file(STUDY_CONFIG)

    assert spec.data["schema_version"] == "mnist-conv-lr-study/v1"
    assert spec.rows == list(FROZEN_ROWS)
    assert len(spec.rows) == 6
    assert [(row["architecture"], row["scheme"]) for row in spec.rows] == [
        ("conv1", "baseline"),
        ("conv1", "ours"),
        ("conv1", "legacy"),
        ("conv2", "baseline"),
        ("conv2", "ours"),
        ("conv2", "legacy"),
    ]
    assert [row["input_gain"] for row in spec.rows] == [
        75.6030807495,
        84.8402175903,
        31.8188591003,
        253.302230835,
        716.3439331055,
        661.4369506836,
    ]
    assert [
        (row["inference_iterations"], row["training_iterations"])
        for row in spec.rows
    ] == [(4, 4), (4, 4), (4, 4), (16, 6), (24, 6), (8, 4)]
    assert spec.data["range_test"]["validation_steps"] == [
        0,
        100,
        200,
        300,
        400,
        500,
        600,
        700,
    ]


def test_study_fingerprint_is_content_addressed_and_name_independent() -> None:
    original = LRStudySpec.from_dict(study_value())
    renamed_value = study_value()
    renamed_value["name"] = "a different display name"
    renamed = LRStudySpec.from_dict(renamed_value)

    assert original.study_id == study_fingerprint(original)
    assert original.study_id == renamed.study_id
    assert original.study_id.startswith("lrstudy_")
    assert len(original.study_id) == len("lrstudy_") + 64


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["dataset"]["train"].update({"batch_size": 32}),
        lambda value: value["dataset"]["validation"].update({"samples_per_class": 499}),
        lambda value: value["dataset"]["validation"].update({"selection_method": "global_randperm"}),
        lambda value: value["dataset"]["official_test"].update({"read_allowed": True}),
        lambda value: value["model"].update({"conductance_bounds": [-1.0, 1.0]}),
        lambda value: value["model"].update({"quadratic_diode_param": {"alpha": 1.0}}),
        lambda value: value["model"].pop("exponential_diode_param"),
        lambda value: value["model"]["hard_sigmoid"].update({"v_off": 1.5}),
        lambda value: value["solver"]["minimizer"].update({"adaptive_equilibrium": True}),
        lambda value: value["optimizer"].update({"momentum": 0.9}),
        lambda value: value["probe"].update({"minibatches": 31}),
        lambda value: value["range_test"]["gates"]["loss_ema"].update(
            {"factor_over_prior_minimum": 1.5}
        ),
        lambda value: value["candidate_training"].update({"total_steps": 17189}),
        lambda value: value["selection"].update({"plateau_relative_to_minimum": 0.03}),
        lambda value: value["rows"][4].update({"input_gain": 716.0}),
        lambda value: value["rows"][5].update({"training_iterations": 6}),
    ],
)
def test_study_rejects_any_scientific_contract_drift(mutation) -> None:
    value = study_value()
    mutation(value)

    with pytest.raises(SpecValidationError):
        LRStudySpec.from_dict(value)


def test_study_rejects_missing_and_unknown_fields() -> None:
    missing = study_value()
    missing.pop("probe")
    with pytest.raises(SpecValidationError, match="Expected study"):
        LRStudySpec.from_dict(missing)

    extra = study_value()
    extra["executor"] = "local"
    with pytest.raises(SpecValidationError, match="Expected study"):
        LRStudySpec.from_dict(extra)


def test_study_returns_defensive_row_copies() -> None:
    spec = LRStudySpec.from_dict(study_value())
    rows = spec.rows
    rows[0]["input_gain"] = -1
    assert spec.rows[0]["input_gain"] == 75.6030807495


def test_file_parser_rejects_duplicate_keys_and_non_finite_numbers(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":"mnist-conv-lr-study/v1","schema_version":"x"}')
    with pytest.raises(SpecValidationError, match="unique keys"):
        LRStudySpec.from_file(duplicate)

    non_finite = copy.deepcopy(study_value())
    text = json.dumps(non_finite).replace('"nominal_lr": 1.0', '"nominal_lr": NaN')
    nan_path = tmp_path / "nan.json"
    nan_path.write_text(text)
    with pytest.raises(SpecValidationError, match="finite JSON number"):
        LRStudySpec.from_file(nan_path)
