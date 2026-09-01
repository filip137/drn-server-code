from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.definitions import resolve_experiment_config
from experiments.mnist_relu_drn import (
    ibm_om_exact_p0_open_vs_closed_loop_adam_runtime as runtime,
)
from experiments.schema import RunMode


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "examples/mnist_relu_drn/ibm_om_exact_p0_open_vs_closed_loop_adam/paired_recovery.json"
)


def test_runtime_contract_is_fail_closed_and_helpers_are_connected() -> None:
    _definition, spec = resolve_experiment_config(CONFIG_PATH, RunMode.TRAIN)
    runtime._strict_protocol(spec.protocol)
    assert callable(runtime.run_train.__globals__["_load_exact_p0"])
    assert callable(runtime.run_train.__globals__["_load_device_model"])
    assert tuple(arm.arm_id for arm in spec.protocol.recovery.arms) == runtime.ARM_IDS


def test_open_loop_historical_parity_gate_is_exact() -> None:
    _definition, spec = resolve_experiment_config(CONFIG_PATH, RunMode.TRAIN)
    historical = spec.protocol.historical_open_loop_parity
    report = {
        "epoch": 1,
        "validation": {
            "student_correct": historical.validation_correct_by_epoch[0],
            "kl_teacher_student": historical.validation_kl_by_epoch[0],
        },
        "validation_prediction_sha256": (
            historical.validation_prediction_sha256_by_epoch[0]
        ),
        "issued_commands_cumulative": (
            historical.cumulative_issued_commands_by_epoch[0]
        ),
        "effective_state_change_events_cumulative": (
            historical.cumulative_effective_state_change_events_by_epoch[0]
        ),
        "train_generator_state_sha256": (
            historical.train_generator_state_sha256_by_epoch[0]
        ),
    }
    gate = runtime._historical_open_loop_epoch_gate(report, protocol=spec.protocol)
    assert gate["cumulative_issued_commands_exact"] is True
    with pytest.raises(RuntimeError, match="historical parity"):
        runtime._historical_open_loop_epoch_gate(
            {**report, "issued_commands_cumulative": 1}, protocol=spec.protocol
        )
    with pytest.raises(RuntimeError, match="historical parity"):
        runtime._historical_open_loop_epoch_gate(
            {**report, "effective_state_change_events_cumulative": 1},
            protocol=spec.protocol,
        )


def test_selected_checkpoint_restore_replays_validation_before_test(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prediction = torch.tensor([2, 1, 0], dtype=torch.int64)
    prediction_sha256 = runtime._tensor_sha256(prediction)
    payload = {
        "plant_continuation_state": {"state_coordinate": "native_raw_active_a"},
        "validation": {"student_correct": 2, "kl_teacher_student": 0.25},
        "validation_prediction_sha256": prediction_sha256,
    }
    plant = SimpleNamespace(persistent=torch.zeros(3))
    seen: list[object] = []
    monkeypatch.setattr(runtime, "_restore_deployment_plant", lambda *args, **kwargs: plant)
    monkeypatch.setattr(
        runtime,
        "sync_catalog_from_persistent",
        lambda *args, **kwargs: seen.append("sync"),
    )
    monkeypatch.setattr(
        runtime,
        "_evaluate_detailed",
        lambda stack, teacher, loader, sample_limit: (
            {"student_correct": 2, "kl_teacher_student": 0.25},
            prediction,
        ),
    )
    stack = SimpleNamespace(
        device=torch.device("cpu"),
        bundle=SimpleNamespace(catalog=object()),
    )
    population = SimpleNamespace(binding_shapes=((2, 1), (1, 1)))
    gate, restored = runtime._restore_selected_validation(
        payload,
        population=population,
        pulse_population=object(),
        stack=stack,
        teacher=object(),
        validation_loader="validation_only",
        maximum_random_draws=385,
    )
    assert restored is plant
    assert seen == ["sync"]
    assert gate["restored_before_any_test_access"] is True

    bad = {**payload, "validation": {"student_correct": 2, "kl_teacher_student": 0.3}}
    with pytest.raises(RuntimeError, match="changed its validation"):
        runtime._restore_selected_validation(
            bad,
            population=population,
            pulse_population=object(),
            stack=stack,
            teacher=object(),
            validation_loader="validation_only",
            maximum_random_draws=385,
        )
