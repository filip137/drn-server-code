from __future__ import annotations

from pathlib import Path
import sys
from types import ModuleType

import pytest

from ebl.cli import TrainRequest, main
from experiments.definitions import (
    EXPERIMENT_REGISTRY,
    resolve_experiment_config,
)
from experiments.mnist_relu_drn.ibm_om_winsorized_endpoint_optimizer_recovery_config import (
    DEVELOPMENT_ENDPOINT_SEED,
    EVALUATION_ENDPOINT_SEEDS,
    EXPERIMENT_ID,
    EndpointOptimizerRecoveryTrainSpec,
    parse_endpoint_optimizer_recovery_config,
)
from experiments.schema import ConfigError, RunMode


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "examples/mnist_relu_drn/ibm_om_winsorized_endpoint_optimizer_recovery/"
    "target_87004.json"
)
RUNTIME_MODULE = (
    "experiments.mnist_relu_drn."
    "ibm_om_winsorized_endpoint_optimizer_recovery_runtime"
)


def _payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "target_assignment_seed": 87004,
        "development_endpoint_seed": DEVELOPMENT_ENDPOINT_SEED,
        "evaluation_endpoint_seeds": list(EVALUATION_ENDPOINT_SEEDS),
    }


def test_endpoint_optimizer_recovery_registry_resolves_frozen_cuda_protocol() -> None:
    definition = EXPERIMENT_REGISTRY[EXPERIMENT_ID]
    assert definition.supported_modes == (RunMode.TRAIN,)

    selected, spec = resolve_experiment_config(CONFIG_PATH, RunMode.TRAIN)
    assert selected is definition
    assert isinstance(spec, EndpointOptimizerRecoveryTrainSpec)
    assert spec.student.runtime.device == "cuda"
    assert spec.protocol.development.endpoint_seed == 89401
    assert spec.protocol.development.test_opened is False
    assert spec.protocol.evaluation.endpoint_seeds == (89402, 89403, 89404, 89405)
    assert spec.protocol.evaluation.independent_population_count == 1
    assert spec.protocol.optimizers.names == (
        "open_loop_sgd",
        "open_loop_adam",
        "tt_v1",
        "tt_v2",
    )
    assert spec.protocol.optimizers.tt_gamma == 0.0
    assert spec.protocol.optimizers.tt_transfer_axis == (
        "canonical_logical_[out,in]_input_columns"
    )
    assert spec.protocol.optimizers.verify_reads_during_training == 0

    with pytest.raises(ConfigError, match="to be one of train"):
        definition.resolve(definition.parse(_payload()), RunMode.VALIDATE)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("target_assignment_seed", 87003),
        ("development_endpoint_seed", 89402),
        ("evaluation_endpoint_seeds", [89401, 89402, 89403, 89404]),
    ),
)
def test_endpoint_optimizer_recovery_config_rejects_protocol_drift(
    field: str, value: object
) -> None:
    payload = _payload()
    payload[field] = value
    with pytest.raises(ConfigError):
        parse_endpoint_optimizer_recovery_config(payload)


def test_default_cli_lazily_dispatches_endpoint_optimizer_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = ModuleType(RUNTIME_MODULE)
    seen: list[TrainRequest] = []

    def run_train(request: TrainRequest) -> int:
        seen.append(request)
        return 43

    runtime.run_train = run_train  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, RUNTIME_MODULE, runtime)

    weights = tmp_path / "logical_source.pt"
    teacher = ROOT / "data/mnist_relu_teacher_fixed_init_20260816.pt"
    result = main(
        [
            "train",
            "--config",
            str(CONFIG_PATH),
            "--output-dir",
            str(tmp_path / "run"),
            "--weights",
            str(weights),
            "--teacher-weights",
            str(teacher),
        ]
    )

    assert result == 43
    assert len(seen) == 1
    request = seen[0]
    assert request.definition is EXPERIMENT_REGISTRY[EXPERIMENT_ID]
    assert isinstance(request.spec, EndpointOptimizerRecoveryTrainSpec)
    assert request.config_path == CONFIG_PATH
    assert request.output_dir == tmp_path / "run"
    assert request.weights == weights
    assert request.teacher_weights == teacher
