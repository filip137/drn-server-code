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
from experiments.mnist_relu_drn.ibm_om_winsorized_tt_output_transfer_followup_config import (
    EXPERIMENT_ID,
    FROZEN_SELECTED_LAMBDAS,
    TtOutputTransferFollowupTrainSpec,
    W2_MULTIPLIER_GRID,
    parse_tt_output_transfer_followup_config,
)
from experiments.schema import ConfigError, RunMode


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "examples/mnist_relu_drn/ibm_om_winsorized_tt_output_transfer_followup/"
    "target_87004.json"
)
RUNTIME_MODULE = (
    "experiments.mnist_relu_drn."
    "ibm_om_winsorized_tt_output_transfer_followup_runtime"
)


def _payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "target_assignment_seed": 87004,
        "development_endpoint_seed": 89401,
        "evaluation_endpoint_seeds": [89402, 89403, 89404, 89405],
        "w2_multiplier_grid": list(W2_MULTIPLIER_GRID),
    }


def test_tt_output_followup_registry_resolves_frozen_cuda_protocol() -> None:
    definition = EXPERIMENT_REGISTRY[EXPERIMENT_ID]
    assert definition.supported_modes == (RunMode.TRAIN,)

    selected, spec = resolve_experiment_config(CONFIG_PATH, RunMode.TRAIN)
    assert selected is definition
    assert isinstance(spec, TtOutputTransferFollowupTrainSpec)
    assert spec.student.runtime.device == "cuda"
    assert spec.protocol.development.endpoint_seed == 89401
    assert spec.protocol.development.test_opened is False
    assert spec.protocol.development.w2_multiplier_grid == (10.0, 30.0, 100.0)
    assert spec.protocol.evaluation.endpoint_seeds == (89402, 89403, 89404, 89405)
    assert spec.protocol.evaluation.independent_population_count == 1
    assert spec.protocol.evaluation.optimizers == ("tt_v1", "tt_v2")
    assert spec.protocol.evaluation.epochs == 1
    assert spec.protocol.evaluation.selection_after_freeze is False
    assert spec.protocol.verify_reads_during_training == 0
    assert dict(
        spec.protocol.source_optimizer_recovery.selected_effective_lambdas
    ) == FROZEN_SELECTED_LAMBDAS

    with pytest.raises(ConfigError, match="to be one of train"):
        definition.resolve(definition.parse(_payload()), RunMode.VALIDATE)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("target_assignment_seed", 87003),
        ("development_endpoint_seed", 89402),
        ("evaluation_endpoint_seeds", [89401, 89402, 89403, 89404]),
        ("w2_multiplier_grid", [3.0, 10.0, 30.0]),
    ),
)
def test_tt_output_followup_config_rejects_protocol_drift(
    field: str, value: object
) -> None:
    payload = _payload()
    payload[field] = value
    with pytest.raises(ConfigError):
        parse_tt_output_transfer_followup_config(payload)


def test_default_cli_lazily_dispatches_tt_output_transfer_followup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = ModuleType(RUNTIME_MODULE)
    seen: list[TrainRequest] = []

    def run_train(request: TrainRequest) -> int:
        seen.append(request)
        return 47

    runtime.run_train = run_train  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, RUNTIME_MODULE, runtime)

    source = (
        ROOT
        / "results/mnist-ibm-om-winsorized-endpoint-optimizer-recovery-"
        "exploratory-20260829-v1/"
        "20260829T142952.088023Z-0ae69031-ec21506d/"
        "artifacts/scientific_summary.json"
    )
    teacher = ROOT / "data/mnist_relu_teacher_fixed_init_20260816.pt"
    result = main(
        [
            "train",
            "--config",
            str(CONFIG_PATH),
            "--output-dir",
            str(tmp_path / "run"),
            "--weights",
            str(source),
            "--teacher-weights",
            str(teacher),
        ]
    )

    assert result == 47
    assert len(seen) == 1
    request = seen[0]
    assert request.definition is EXPERIMENT_REGISTRY[EXPERIMENT_ID]
    assert isinstance(request.spec, TtOutputTransferFollowupTrainSpec)
    assert request.config_path == CONFIG_PATH
    assert request.output_dir == tmp_path / "run"
    assert request.weights == source
    assert request.teacher_weights == teacher
