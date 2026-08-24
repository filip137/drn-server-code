from __future__ import annotations

from dataclasses import replace
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ebl.cli import CommandHandlers, TrainRequest, ValidateRequest, main
from experiments.definitions import resolve_experiment_config
from experiments.mnist_relu_drn.components import build_student_stack
from experiments.mnist_relu_drn.runtime import (
    _ibm_target_mapping_preflights,
    _selected_payload,
    _validate_array_specific_modifier_population_parity,
)
from experiments.schema import ConfigError, RunMode
from experiments.study_workflow import load_study_plan
from training.ibm_reram_hwa import IbmReramHwaParameterModifier
from training.modifier import SplitParameterModifier


_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_ROOT = _ROOT / "examples" / "mnist_relu_drn" / "ibm_om_hwa_pilot"
_COMMON_WINDOW_CONFIG_ROOT = (
    _ROOT
    / "examples"
    / "mnist_relu_drn"
    / "ibm_om_common_window_hwa_pilot"
)
_STUDY = (
    _ROOT
    / "studies"
    / "mnist-ibm-om-hwa-program-verify-pilot-20260822-v1.json"
)


def test_four_training_arms_share_common_physical_selection() -> None:
    expected_training = {
        "clean.json": "none",
        "gaussian_3pct.json": "add_normal",
        "om_repaired.json": "ibm_reram_om_program_verify",
        "om_published.json": "ibm_reram_om_program_verify",
    }
    selections = []
    for name, modifier_type in expected_training.items():
        _definition, spec = resolve_experiment_config(
            _CONFIG_ROOT / name,
            RunMode.TRAIN,
        )
        assert spec.settings.weight_modifier.type == modifier_type
        selection = spec.settings.selection_weight_modifier
        assert selection.type == "ibm_reram_om_program_verify"
        assert selection.parameters["execution"] == "pulse_resolved"
        assert selection.parameters["corruption_policy"] == "published"
        assert selection.parameters["assignment_seed"] == 83003
        assert selection.parameters["endpoint_seed"] == 83003
        assert selection.parameters["target_mapping"] == "literal_global"
        assert selection.parameters["dual_rail_layout_by_parameter"] is None
        assert selection.parameters["common_window_margin_fraction"] == 0.0
        selections.append(dict(selection.parameters))
    assert selections[1:] == selections[:-1]


def test_heldout_assignments_and_endpoint_seeds_are_predeclared() -> None:
    for offset in range(4):
        assignment = 83101 + offset
        _definition, spec = resolve_experiment_config(
            _CONFIG_ROOT / f"heldout_assignment_{assignment}.json",
            RunMode.VALIDATE,
        )
        modifier = spec.settings.weight_modifier
        assert modifier.parameters["assignment_seed"] == assignment
        assert modifier.parameters["endpoint_seed"] == 83201 + offset
        assert modifier.parameters["execution"] == "pulse_resolved"
        assert modifier.parameters["corruption_policy"] == "published"


def test_study_declares_each_checkpoint_assignment_validation_as_one_run() -> None:
    study = load_study_plan(_STUDY)
    training_arms = [arm for arm in study["arms"] if arm["mode"] == "train"]
    validation_arms = [
        arm for arm in study["arms"] if arm["mode"] == "validate"
    ]

    assert len(training_arms) == 4
    assert len(validation_arms) == 16
    assert {
        arm["arm_id"] for arm in validation_arms
    } == {
        f"test-{training}-{assignment}"
        for training in (
            "clean",
            "gaussian-3pct",
            "om-repaired",
            "om-published",
        )
        for assignment in range(83101, 83105)
    }


def test_training_ibm_modifier_must_be_compact(tmp_path: Path) -> None:
    payload = json.loads((_CONFIG_ROOT / "om_repaired.json").read_text())
    payload["modes"]["train"]["weight_modifier"]["parameters"][
        "execution"
    ] = "pulse_resolved"
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ConfigError, match="compact_endpoint"):
        resolve_experiment_config(path, RunMode.TRAIN)


def test_selection_ibm_modifier_must_keep_published_corruption(
    tmp_path: Path,
) -> None:
    payload = json.loads((_CONFIG_ROOT / "clean.json").read_text())
    payload["modes"]["train"]["selection_weight_modifier"]["parameters"][
        "corruption_policy"
    ] = "counterfactual_repaired"
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ConfigError, match="published corruption"):
        resolve_experiment_config(path, RunMode.TRAIN)


def _set_quad_mapping(parameters: dict, *, margin: float = 0.25) -> None:
    parameters.update(
        {
            "target_mapping": "dual_rail_quad_common_window",
            "dual_rail_layout_by_parameter": {
                "base.dense_weight.0": "halves",
                "base.dense_weight.1": "paired",
            },
            "common_window_margin_fraction": margin,
        }
    )


def _set_differential_pair_mapping(
    parameters: dict,
    *,
    margin: float = 0.25,
) -> None:
    parameters.update(
        {
            "target_mapping": "differential_pair_common_window",
            "dual_rail_layout_by_parameter": None,
            "common_window_margin_fraction": margin,
        }
    )


def _differential_pair_hwa_payload() -> dict:
    payload = json.loads(
        (_ROOT / "examples/mnist_relu_drn/ideal_differential.json").read_text()
    )
    source = json.loads((_CONFIG_ROOT / "om_repaired.json").read_text())
    training = json.loads(
        json.dumps(
            source["modes"]["train"]["weight_modifier"]
        )
    )
    selection = json.loads(
        json.dumps(
            source["modes"]["train"]["selection_weight_modifier"]
        )
    )
    train_parameters = training["parameters"]
    selection_parameters = selection["parameters"]
    _set_differential_pair_mapping(train_parameters)
    _set_differential_pair_mapping(selection_parameters)
    train_parameters.update(
        {
            "assignment_seed": 84001,
            "endpoint_seed": 84002,
            "corruption_policy": "counterfactual_repaired",
        }
    )
    selection_parameters.update(
        {
            "assignment_seed": 84001,
            "endpoint_seed": 84003,
            "corruption_policy": "counterfactual_repaired",
        }
    )
    payload["modes"]["train"].update(
        {
            "weight_modifier": training,
            "selection_evaluation": "modifier",
            "selection_noise_repeats": 1,
            "selection_weight_modifier": selection,
        }
    )
    return payload


def test_quad_selection_requires_repaired_corruption_and_exact_layout(
    tmp_path: Path,
) -> None:
    payload = json.loads((_CONFIG_ROOT / "clean.json").read_text())
    parameters = payload["modes"]["train"]["selection_weight_modifier"][
        "parameters"
    ]
    _set_quad_mapping(parameters)
    parameters["corruption_policy"] = "counterfactual_repaired"
    path = tmp_path / "quad-clean.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    _definition, spec = resolve_experiment_config(path, RunMode.TRAIN)
    parsed = spec.settings.selection_weight_modifier.parameters
    assert parsed["target_mapping"] == "dual_rail_quad_common_window"
    assert parsed["corruption_policy"] == "counterfactual_repaired"
    assert parsed["common_window_margin_fraction"] == 0.25
    assert dict(parsed["dual_rail_layout_by_parameter"]) == {
        "base.dense_weight.0": "halves",
        "base.dense_weight.1": "paired",
    }

    payload["modes"]["train"]["selection_weight_modifier"]["parameters"][
        "dual_rail_layout_by_parameter"
    ]["base.dense_weight.1"] = "halves"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ConfigError, match="canonical W1/W2 layout"):
        resolve_experiment_config(path, RunMode.TRAIN)


def test_quad_hwa_training_and_selection_must_share_fixed_array_mapping(
    tmp_path: Path,
) -> None:
    payload = json.loads((_CONFIG_ROOT / "om_repaired.json").read_text())
    train_parameters = payload["modes"]["train"]["weight_modifier"][
        "parameters"
    ]
    selection_parameters = payload["modes"]["train"][
        "selection_weight_modifier"
    ]["parameters"]
    _set_quad_mapping(train_parameters)
    _set_quad_mapping(selection_parameters)
    selection_parameters["assignment_seed"] = train_parameters["assignment_seed"]
    selection_parameters["corruption_policy"] = "counterfactual_repaired"
    path = tmp_path / "quad-hwa.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    _definition, spec = resolve_experiment_config(path, RunMode.TRAIN)
    assert (
        spec.settings.weight_modifier.parameters["assignment_seed"]
        == spec.settings.selection_weight_modifier.parameters["assignment_seed"]
    )

    selection_parameters["assignment_seed"] += 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ConfigError, match="fixed array"):
        resolve_experiment_config(path, RunMode.TRAIN)


def test_differential_pair_mapping_requires_differential_encoding_and_null_layout(
    tmp_path: Path,
) -> None:
    payload = _differential_pair_hwa_payload()
    path = tmp_path / "pair-hwa.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    _definition, spec = resolve_experiment_config(path, RunMode.TRAIN)
    training = spec.settings.weight_modifier.parameters
    selection = spec.settings.selection_weight_modifier.parameters
    assert spec.model.encoding == "differential"
    assert training["target_mapping"] == "differential_pair_common_window"
    assert training["dual_rail_layout_by_parameter"] is None
    assert training["common_window_margin_fraction"] == 0.25
    assert selection["corruption_policy"] == "counterfactual_repaired"
    assert training["endpoint_seed"] == 84002
    assert selection["endpoint_seed"] == 84003
    assert training["execution"] == "compact_endpoint"
    assert selection["execution"] == "pulse_resolved"

    payload["model"]["encoding"] = "single"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ConfigError, match="model.encoding='differential'"):
        resolve_experiment_config(path, RunMode.TRAIN)

    payload = _differential_pair_hwa_payload()
    payload["modes"]["train"]["weight_modifier"]["parameters"][
        "dual_rail_layout_by_parameter"
    ] = {"base.dense_weight.0": "halves"}
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ConfigError, match="to be null"):
        resolve_experiment_config(path, RunMode.TRAIN)


def test_quad_mapping_rejects_differential_model_encoding(
    tmp_path: Path,
) -> None:
    payload = _differential_pair_hwa_payload()
    for role in ("weight_modifier", "selection_weight_modifier"):
        parameters = payload["modes"]["train"][role]["parameters"]
        _set_quad_mapping(parameters)
    path = tmp_path / "differential-quad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ConfigError, match="model.encoding='single'"):
        resolve_experiment_config(path, RunMode.TRAIN)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("assignment_seed", 84009),
        ("corruption_policy", "published"),
        ("target_mapping", "literal_global"),
        ("common_window_margin_fraction", 0.2),
    ],
)
def test_differential_pair_hwa_requires_matched_fixed_array_fields(
    field: str,
    replacement,
    tmp_path: Path,
) -> None:
    payload = _differential_pair_hwa_payload()
    selection = payload["modes"]["train"]["selection_weight_modifier"][
        "parameters"
    ]
    selection[field] = replacement
    if field == "target_mapping":
        selection["common_window_margin_fraction"] = 0.0
        selection["corruption_policy"] = "published"
    path = tmp_path / f"mismatch-{field}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    expected = "fixed array" if field != "corruption_policy" else "counterfactual_repaired"
    with pytest.raises(ConfigError, match=expected):
        resolve_experiment_config(path, RunMode.TRAIN)


def test_differential_pair_margin_rejects_negative_value(
    tmp_path: Path,
) -> None:
    payload = _differential_pair_hwa_payload()
    for role in ("weight_modifier", "selection_weight_modifier"):
        payload["modes"]["train"][role]["parameters"][
            "common_window_margin_fraction"
        ] = -0.01
    path = tmp_path / "negative-margin.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ConfigError, match="finite number"):
        resolve_experiment_config(path, RunMode.TRAIN)


def test_differential_pair_validation_requires_repaired_pulse_deployment(
    tmp_path: Path,
) -> None:
    payload = _differential_pair_hwa_payload()
    selection = payload["modes"]["train"]["selection_weight_modifier"]
    payload["modes"] = {
        "validate": {
            "split": "test",
            "sample_limit": 1600,
            "weight_modifier": selection,
            "noise_repeats": 1,
        }
    }
    path = tmp_path / "pair-validate.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    _definition, spec = resolve_experiment_config(path, RunMode.VALIDATE)
    modifier = spec.settings.weight_modifier
    assert modifier.parameters["execution"] == "pulse_resolved"
    assert modifier.parameters["corruption_policy"] == "counterfactual_repaired"
    assert (
        modifier.parameters["target_mapping"]
        == "differential_pair_common_window"
    )

    selection["parameters"]["corruption_policy"] = "published"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ConfigError, match="counterfactual_repaired"):
        resolve_experiment_config(path, RunMode.VALIDATE)


def _fake_ibm_modifier(
    *,
    target_mapping: str,
    fingerprint: str,
    report: dict,
) -> IbmReramHwaParameterModifier:
    modifier = IbmReramHwaParameterModifier.__new__(
        IbmReramHwaParameterModifier
    )
    modifier._config = SimpleNamespace(target_mapping=target_mapping)
    modifier._population = SimpleNamespace(fingerprint=fingerprint)
    modifier.preflight_target_mapping = lambda: (None, report)
    return modifier


def test_runtime_pair_mapping_requires_population_parity_and_persists_preflights() -> None:
    training_report = {"target_mapping": "differential_pair_common_window"}
    selection_report = {
        "target_mapping": "differential_pair_common_window",
        "role": "selection",
    }
    training = _fake_ibm_modifier(
        target_mapping="differential_pair_common_window",
        fingerprint="fixed-pair-array",
        report=training_report,
    )
    selection = _fake_ibm_modifier(
        target_mapping="differential_pair_common_window",
        fingerprint="fixed-pair-array",
        report=selection_report,
    )
    split = SplitParameterModifier(
        training=training,
        evaluation=selection,
    )

    _validate_array_specific_modifier_population_parity(split)
    assert _ibm_target_mapping_preflights(split) == {
        "training": training_report,
        "selection": selection_report,
    }

    selection._population = SimpleNamespace(fingerprint="different-array")
    with pytest.raises(RuntimeError, match="same fixed population fingerprint"):
        _validate_array_specific_modifier_population_parity(split)


def test_runtime_literal_mapping_remains_outside_array_specific_preflights() -> None:
    literal = _fake_ibm_modifier(
        target_mapping="literal_global",
        fingerprint="literal-array",
        report={"unexpected": True},
    )
    _validate_array_specific_modifier_population_parity(literal)
    assert _ibm_target_mapping_preflights(literal) == {}


@pytest.mark.parametrize("config_name", ["clean.json", "exact_bounds_hwa.json"])
def test_common_window_selected_checkpoint_thaws_frozen_modifier_metadata(
    config_name: str,
    tmp_path: Path,
) -> None:
    _definition, spec = resolve_experiment_config(
        _COMMON_WINDOW_CONFIG_ROOT / config_name,
        RunMode.TRAIN,
    )
    frozen_layout = spec.settings.selection_weight_modifier.parameters[
        "dual_rail_layout_by_parameter"
    ]
    with pytest.raises(TypeError):
        frozen_layout["base.dense_weight.0"] = "paired"

    cpu_spec = replace(spec, runtime=replace(spec.runtime, device="cpu"))
    stack = build_student_stack(cpu_spec, enable_measured=False)
    selected = _selected_payload(
        stack,
        spec=cpu_spec,
        teacher_path=tmp_path / "teacher.pt",
        teacher_sha256="teacher-sha",
        mapping=None,
        deployment_source=None,
        device_model_sha256="device-model-sha",
        epoch=-1,
        validation={
            "kl_teacher_student": 0.5,
            "student_accuracy": 0.25,
            "teacher_agreement": 0.3,
        },
    )

    metadata = selected["metadata"]
    json.dumps(metadata, allow_nan=False)
    serialized_layout = metadata["selection_weight_modifier"]["parameters"][
        "dual_rail_layout_by_parameter"
    ]
    assert type(serialized_layout) is dict
    assert serialized_layout == dict(frozen_layout)
    if config_name == "exact_bounds_hwa.json":
        assert type(
            metadata["weight_modifier"]["parameters"][
                "dual_rail_layout_by_parameter"
            ]
        ) is dict


def test_quad_mapping_fields_are_all_or_none_and_margin_is_strict(
    tmp_path: Path,
) -> None:
    payload = json.loads((_CONFIG_ROOT / "clean.json").read_text())
    parameters = payload["modes"]["train"]["selection_weight_modifier"][
        "parameters"
    ]
    parameters["target_mapping"] = "dual_rail_quad_common_window"
    path = tmp_path / "partial.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ConfigError, match="all three target-mapping fields"):
        resolve_experiment_config(path, RunMode.TRAIN)

    _set_quad_mapping(parameters, margin=0.5)
    parameters["corruption_policy"] = "counterfactual_repaired"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ConfigError, match=r"\[0, 0.5\)"):
        resolve_experiment_config(path, RunMode.TRAIN)


def test_cli_forwards_explicit_device_model_to_train_and_validate(
    tmp_path: Path,
) -> None:
    train_seen: list[TrainRequest] = []
    validate_seen: list[ValidateRequest] = []
    teacher = tmp_path / "teacher.pt"
    weights = tmp_path / "weights.pt"
    device_model = tmp_path / "device.json"
    for path in (teacher, weights, device_model):
        path.write_bytes(b"fixture")

    assert main(
        [
            "train",
            "--config",
            str(_CONFIG_ROOT / "clean.json"),
            "--output-dir",
            str(tmp_path / "train"),
            "--teacher-weights",
            str(teacher),
            "--device-model",
            str(device_model),
        ],
        handlers=CommandHandlers(train=lambda request: train_seen.append(request) or 0),
        stdout=io.StringIO(),
    ) == 0
    assert train_seen[0].device_model == device_model

    assert main(
        [
            "validate",
            "--config",
            str(_CONFIG_ROOT / "heldout_assignment_83101.json"),
            "--output-dir",
            str(tmp_path / "validate"),
            "--weights",
            str(weights),
            "--teacher-weights",
            str(teacher),
            "--device-model",
            str(device_model),
        ],
        handlers=CommandHandlers(
            validate=lambda request: validate_seen.append(request) or 0
        ),
        stdout=io.StringIO(),
    ) == 0
    assert validate_seen[0].device_model == device_model
