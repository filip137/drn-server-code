from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest

from ebl.cli import TrainRequest, main
from experiments.definitions import EXPERIMENT_REGISTRY, parse_experiment_config
from experiments.mnist_relu_drn.ibm_om_winsorized_qat_config import (
    DEVELOPMENT_ENDPOINT_SEEDS,
    EVALUATION_ENDPOINT_SEEDS,
    EXPECTED_TEACHER_WEIGHTS_PATH,
    EXPECTED_TEACHER_WEIGHTS_SHA256,
    EXPERIMENT_ID,
    FIXED_LOGIT_GAIN,
    FIXED_SCALE_FRACTIONS,
    LEARNING_RATES,
    SPACING_DELTA_X_MULTIPLIERS,
    WinsorizedQatTrainSpec,
    parse_winsorized_qat_config,
    resolve_winsorized_qat_spec,
)
from experiments.schema import ConfigError, RunMode


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "examples" / "mnist_relu_drn" / "ibm_om_winsorized_qat"
RUNTIME_MODULE = "experiments.mnist_relu_drn.ibm_om_winsorized_qat_runtime"


def _paths() -> tuple[Path, ...]:
    return tuple(
        CONFIG_ROOT / f"alpha_000_spacing_{spacing}delta.json"
        for spacing in SPACING_DELTA_X_MULTIPLIERS
    )


def _payload(path: Path | None = None) -> dict:
    selected = path or _paths()[0]
    return json.loads(selected.read_text(encoding="utf-8"))


def test_exact_three_cuda_qat_configs_resolve() -> None:
    assert set(CONFIG_ROOT.glob("*.json")) == set(_paths())
    observed = set()
    for path in _paths():
        spec = resolve_winsorized_qat_spec(
            parse_winsorized_qat_config(_payload(path)),
            RunMode.TRAIN,
        )
        assert isinstance(spec, WinsorizedQatTrainSpec)
        assert spec.student.runtime.device == "cuda"
        assert spec.student.runtime.seed == 42
        assert spec.student.runtime.data_seed == 42
        assert spec.student.data.batch_size == 16
        assert spec.student.settings.num_epochs == 10
        assert spec.student.settings.learning_rates == LEARNING_RATES
        assert spec.protocol.fixed_scale_fractions == FIXED_SCALE_FRACTIONS
        assert spec.protocol.fixed_logit_gain == FIXED_LOGIT_GAIN
        observed.add(spec.protocol.spacing_delta_x_multiplier)
    assert observed == set(SPACING_DELTA_X_MULTIPLIERS)


def test_protocol_freezes_source_assignments_and_noise_boundary() -> None:
    spec = resolve_winsorized_qat_spec(
        parse_winsorized_qat_config(_payload()), RunMode.TRAIN
    )
    protocol = spec.protocol

    assert protocol.source.weights_role == (
        "frozen_relu_source_teacher_and_logical_master_initializer"
    )
    assert protocol.source.expected_teacher_weights_path == (
        EXPECTED_TEACHER_WEIGHTS_PATH
    )
    assert protocol.source.expected_teacher_weights_sha256 == (
        EXPECTED_TEACHER_WEIGHTS_SHA256
    )
    assert protocol.development_assignment_seed == 86001
    assert protocol.evaluation_assignment_seed == 87001
    assert protocol.development_endpoint_seeds == DEVELOPMENT_ENDPOINT_SEEDS
    assert protocol.evaluation_endpoint_seeds == EVALUATION_ENDPOINT_SEEDS
    assert protocol.artifacts.development_hardware_instance_id != (
        protocol.artifacts.evaluation_hardware_instance_id
    )
    assert protocol.training.program_verify is False
    assert protocol.training.read_noise is False
    assert protocol.execution.training_program_verify is False
    assert protocol.evaluation.deployment_program_verify is True
    assert protocol.evaluation.inference_read_noise is False
    assert protocol.mapping.circuit_accounting == (
        "full_G_in_numerator_and_denominator"
    )
    assert protocol.mapping.reference_policy == "absent_no_reference_subtraction"


@pytest.mark.parametrize(
    ("section", "field", "replacement"),
    [
        ("source", "expected_teacher_weights_path", "data/other.pt"),
        ("mapping", "fixed_logit_gain", 14.0),
        ("training", "program_verify", True),
        ("training", "epochs", 9),
        ("assignments", "evaluation_assignment_seed", 87002),
    ],
)
def test_qat_protocol_rejects_contract_drift(
    section: str, field: str, replacement: object
) -> None:
    payload = deepcopy(_payload())
    payload["winsorized_qat"][section][field] = replacement
    with pytest.raises(ConfigError, match=field):
        parse_winsorized_qat_config(payload)


def test_registry_is_additive_and_train_only() -> None:
    definition = EXPERIMENT_REGISTRY[EXPERIMENT_ID]
    assert definition.supported_modes == (RunMode.TRAIN,)
    selected, document = parse_experiment_config(_payload())
    assert selected is definition
    assert document.experiment_id == EXPERIMENT_ID
    with pytest.raises(ConfigError, match="to be one of train"):
        definition.resolve(document, RunMode.VALIDATE)


def test_default_cli_lazily_dispatches_qat_train(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = ModuleType(RUNTIME_MODULE)
    seen: list[TrainRequest] = []

    def run_train(request: TrainRequest) -> int:
        seen.append(request)
        return 37

    runtime.run_train = run_train  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, RUNTIME_MODULE, runtime)
    config_path = _paths()[0]
    teacher_weights = ROOT / EXPECTED_TEACHER_WEIGHTS_PATH
    result = main(
        [
            "train",
            "--config",
            str(config_path),
            "--output-dir",
            str(tmp_path / "run"),
            "--teacher-weights",
            str(teacher_weights),
        ]
    )

    assert result == 37
    assert len(seen) == 1
    request = seen[0]
    assert request.definition is EXPERIMENT_REGISTRY[EXPERIMENT_ID]
    assert isinstance(request.spec, WinsorizedQatTrainSpec)
    assert request.teacher_weights == teacher_weights
    assert request.weights is None
    assert request.base_weights is None
    assert request.resume is None
    assert request.device_data is None
    assert request.device_model is None
