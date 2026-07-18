from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.mnist_conv.backend import (
    ExecutionContext,
    build_engine_config,
    train_with_mnist_backend,
)
from experiments.mnist_conv.protocol import (
    APPROVED_FINAL_PROTOCOL_IDS,
    ProtocolExecutionError,
    ensure_run_executable,
)
from experiments.mnist_conv.layout import ResultLayout
from experiments.mnist_conv.runner import execute_run
from experiments.mnist_conv.specs import RunSpec, SpecValidationError, SweepSpec
from labs.mnist_train import (
    _checkpoint_best_then_callback,
    _resolve_optimizer_settings,
    _validate_optimizer_rate_consistency,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = REPO_ROOT / "configs/conv/run_v1.diagnostic.example.json"


def _run_value() -> dict:
    return json.loads(EXAMPLE.read_text())


def _sweep_value(base: dict) -> dict:
    return {
        "schema_version": "mnist-conv-sweep/v1",
        "name": "protocol-schema-test",
        "base_run": base,
        "axes": [{"path": "/seed", "values": [0, 1]}],
        "cases": [{"id": "baseline", "set": {}}],
        "varying_fields": ["/seed"],
        "collection": {
            "expected_seeds": [0, 1],
            "required_cases": ["baseline"],
            "group_by": ["/run/model/voltage_amp"],
        },
    }


def test_optimizer_is_required_and_v1_is_exact_sgd() -> None:
    base = _run_value()
    spec = RunSpec.from_dict(base)
    assert spec.data["run"]["training"]["optimizer"] == {
        "name": "SGD",
        "momentum": 0.0,
        "weight_decay": 0.0,
    }

    missing = copy.deepcopy(base)
    missing["run"]["training"].pop("optimizer")
    with pytest.raises(SpecValidationError, match="optimizer"):
        RunSpec.from_dict(missing)

    for field, value in (("name", "Adam"), ("momentum", 0.9), ("weight_decay", 1e-4)):
        invalid = copy.deepcopy(base)
        invalid["run"]["training"]["optimizer"][field] = value
        with pytest.raises(SpecValidationError, match=field):
            RunSpec.from_dict(invalid)


def test_backend_maps_every_optimizer_setting_explicitly() -> None:
    spec = RunSpec.from_dict(_run_value())
    engine = build_engine_config(spec, ExecutionContext("/datasets/mnist", device="cpu"))

    assert engine["optimizer"] == {
        "name": "SGD",
        "learning_rate": [0.01, 0.01, 0.01],
        "lr_decay": 1.0,
        "momentum": 0.0,
        "weight_decay": 0.0,
    }


def test_training_backend_requires_explicit_optimizer_and_rejects_config_drift() -> None:
    config = {
        "optimizer": {
            "name": "SGD",
            "learning_rate": [0.01, 0.02],
            "lr_decay": 0.5,
            "momentum": 0.0,
            "weight_decay": 0.0,
        }
    }
    assert _resolve_optimizer_settings(
        config,
        optimizer_name="SGD",
        momentum=0.0,
        weight_decay=0.0,
    ) == ("SGD", 0.0, 0.0)
    _validate_optimizer_rate_consistency(config, [0.01, 0.02], 0.5)

    with pytest.raises(ValueError, match="momentum.*equal"):
        _resolve_optimizer_settings(
            config,
            optimizer_name="SGD",
            momentum=0.1,
            weight_decay=0.0,
        )
    with pytest.raises(ValueError, match="learning rates.*equal"):
        _validate_optimizer_rate_consistency(config, [0.01, 0.03], 0.5)
    with pytest.raises(ValueError, match="optimizer name"):
        _resolve_optimizer_settings({}, momentum=0.0, weight_decay=0.0)


def test_final_specs_can_plan_but_execution_is_fail_closed(tmp_path: Path) -> None:
    value = _run_value()
    value["category"] = "final"
    value["protocol_id"] = "proposed-not-approved"
    spec = RunSpec.from_dict(value)

    assert APPROVED_FINAL_PROTOCOL_IDS == frozenset()
    build_engine_config(spec, ExecutionContext(tmp_path / "data", device="cpu"))
    with pytest.raises(ProtocolExecutionError, match="numerical execution is blocked"):
        ensure_run_executable(spec)
    with pytest.raises(ProtocolExecutionError, match="proposed-not-approved"):
        train_with_mnist_backend(
            spec=spec,
            context=ExecutionContext(tmp_path / "data", device="cpu"),
            engine_config_path=tmp_path / "unused.json",
            output_dir=tmp_path / "unused-output",
        )
    results = tmp_path / "results"
    with pytest.raises(ProtocolExecutionError, match="proposed-not-approved"):
        execute_run(
            spec,
            ExecutionContext(tmp_path / "data", device="cpu"),
            ResultLayout(results),
            {
                "git_revision": "1" * 40,
                "dirty_source_digest": "2" * 64,
                "effective_code_fingerprint": "3" * 64,
            },
            backend=pytest.fail,
        )
    assert not results.exists()

    diagnostic = RunSpec.from_dict(_run_value())
    ensure_run_executable(diagnostic)


def test_best_checkpoint_is_strictly_validated_before_pruning_callback(tmp_path: Path) -> None:
    callback_called = False

    class CorruptingEnergy:
        _params = []

        def save(self, path):
            torch.save(
                {
                    "format": "drn.function.parameters",
                    "version": 1,
                    "schema": [],
                    "states": [torch.ones(1)],
                },
                path,
            )

    def callback(epoch_info, history):
        nonlocal callback_called
        callback_called = True
        return True

    with pytest.raises(ValueError, match="checkpoint"):
        _checkpoint_best_then_callback(
            energy_fn=CorruptingEnergy(),
            best_model_path=tmp_path / "best.pt",
            epoch=1,
            train_loss=0.7,
            train_accuracy=0.6,
            test_loss=0.5,
            test_accuracy=0.8,
            best_accuracy=float("-inf"),
            best_epoch=None,
            optimizer=SimpleNamespace(param_groups=[{"lr": 0.25}]),
            history={"learning_rate": []},
            epoch_callback=callback,
        )
    assert callback_called is False


@pytest.mark.parametrize(
    ("non_linearity", "field", "value"),
    [
        ("hard_sigmoid", "quadratic_diode_param", {"ignored": 1}),
        ("hard_sigmoid", "exponential_diode_param", {"ignored": 1}),
        ("perfect_diode", "quadratic_diode_param", {"ignored": 1}),
        ("perfect_diode", "exponential_diode_param", {"ignored": 1}),
        ("perfect_diode", "hard_sigmoid_param", {"ignored": 1}),
    ],
)
def test_unused_diode_parameter_dictionaries_are_rejected(
    non_linearity: str,
    field: str,
    value: dict,
) -> None:
    run = _run_value()
    run["run"]["model"]["non_linearity"] = non_linearity
    if non_linearity == "perfect_diode":
        run["run"]["model"]["hard_sigmoid_param"] = {}
        run["run"]["calibration"] = {
            "kind": "perfect_diode_clamped_occupancy",
            "calibration_id": "pd-calibration",
            "scope": "first_hidden_layer",
            "sample_count": 256,
            "batch_size": 64,
            "model_seed": 0,
            "affine_seed": 1729,
            "settling_iterations": 64,
            "adaptive_equilibrium": False,
            "target_initial_occupancy": 0.3,
            "measured_initial_occupancy": 0.3,
            "clamp_epsilon": 1e-8,
            "layer_measurements": [
                {"layer_index": 1, "measured_clamped_occupancy": 0.3}
            ],
        }
    run["run"]["model"][field] = value
    with pytest.raises(SpecValidationError, match=field):
        RunSpec.from_dict(run)


def test_strict_json_rejects_nonfinite_numbers_and_duplicate_keys(tmp_path: Path) -> None:
    text = EXAMPLE.read_text()
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text(text.replace('"input_gain": 1.0', '"input_gain": NaN', 1))
    with pytest.raises(SpecValidationError, match="finite JSON number"):
        RunSpec.from_path(nonfinite)

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(text.replace('"seed": 0,', '"seed": 0, "seed": 1,', 1))
    with pytest.raises(SpecValidationError, match="unique keys"):
        RunSpec.from_path(duplicate)


def test_checkpoint_declarations_are_constrained() -> None:
    base = _run_value()
    base["run"]["initialization"]["checkpoint"] = {
        "path": "runs/source/checkpoints/best.pt",
        "sha256": "a" * 64,
        "format": "drn.function.parameters/v1",
        "source_run_id": "run_" + "b" * 64,
        "role": "best",
    }
    RunSpec.from_dict(base)

    for field, value in (
        ("format", "pickle"),
        ("role", "latest"),
        ("source_run_id", "not-a-run-id"),
    ):
        invalid = copy.deepcopy(base)
        invalid["run"]["initialization"]["checkpoint"][field] = value
        with pytest.raises(SpecValidationError, match=field):
            RunSpec.from_dict(invalid)


@pytest.mark.parametrize("group_by", [["/seed"], ["/replicate_id"], ["/run/model/input_gain"] * 2])
def test_collection_grouping_cannot_split_seed_coverage(group_by: list[str]) -> None:
    sweep = _sweep_value(_run_value())
    sweep["collection"]["group_by"] = group_by
    with pytest.raises(SpecValidationError, match="group_by"):
        SweepSpec.from_dict(sweep)
