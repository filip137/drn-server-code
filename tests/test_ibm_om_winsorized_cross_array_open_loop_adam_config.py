from __future__ import annotations

from pathlib import Path
import sys
from types import ModuleType

import pytest

from ebl.cli import TrainRequest, main
from experiments.definitions import EXPERIMENT_REGISTRY
from experiments.mnist_relu_drn.ibm_om_winsorized_cross_array_open_loop_adam_config import (
    EXPERIMENT_ID,
    parse_winsorized_cross_array_open_loop_adam_config,
    resolve_winsorized_cross_array_open_loop_adam_spec,
)
from experiments.schema import ConfigError, RunMode


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_MODULE = (
    "experiments.mnist_relu_drn."
    "ibm_om_winsorized_cross_array_open_loop_adam_runtime"
)


def _payload() -> dict:
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "source_assignment_seed": 87003,
        "target_assignment_seed": 87004,
        "target_endpoint_seeds": [89401, 89402, 89403, 89404, 89405],
        "fine_tune_endpoint_seed": 89401,
    }


def test_cross_array_config_resolves_fresh_fixed_protocol() -> None:
    document = parse_winsorized_cross_array_open_loop_adam_config(_payload())
    spec = resolve_winsorized_cross_array_open_loop_adam_spec(
        document, RunMode.TRAIN
    )
    assert spec.protocol.source.assignment_seed == 87003
    assert spec.protocol.target.assignment_seed == 87004
    assert spec.protocol.target.endpoint_seeds == (89401, 89402, 89403, 89404, 89405)
    assert spec.protocol.deployment.target_clipping is False
    assert spec.protocol.recovery.learning_rate_raw_x == 3e-5
    assert spec.protocol.recovery.verify_reads_during_updates == 0
    assert spec.protocol.recovery.conceptual_column_phases_per_minibatch == (100, 20)
    assert spec.student.runtime.device == "cuda"


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("target_assignment_seed", 87002),
        ("fine_tune_endpoint_seed", 89402),
        ("target_endpoint_seeds", [89401]),
    ),
)
def test_cross_array_config_rejects_protocol_drift(field: str, value: object) -> None:
    payload = _payload()
    payload[field] = value
    with pytest.raises(ConfigError):
        parse_winsorized_cross_array_open_loop_adam_config(payload)


def test_registry_and_default_cli_dispatch_are_additive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    definition = EXPERIMENT_REGISTRY[EXPERIMENT_ID]
    assert definition.supported_modes == (RunMode.TRAIN,)
    runtime = ModuleType(RUNTIME_MODULE)
    seen: list[TrainRequest] = []

    def run_train(request: TrainRequest) -> int:
        seen.append(request)
        return 43

    runtime.run_train = run_train  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, RUNTIME_MODULE, runtime)
    config = (
        ROOT
        / "examples/mnist_relu_drn/ibm_om_winsorized_cross_array_open_loop_adam/"
        "fresh_87004.json"
    )
    source = (
        ROOT
        / "results/mnist-ibm-om-winsorized-onchip-adam-exploratory-20260829-v1/"
        "runs/alpha_000_spacing_1delta/"
        "20260828T224530.737807Z-1ae7917a-10efc98a/"
        "checkpoints/stage2_direct_physical_rail_lr_3e-05.pt"
    )
    teacher = ROOT / "data/mnist_relu_teacher_fixed_init_20260816.pt"
    result = main(
        [
            "train",
            "--config",
            str(config),
            "--output-dir",
            str(tmp_path / "run"),
            "--weights",
            str(source),
            "--teacher-weights",
            str(teacher),
        ]
    )
    assert result == 43
    assert len(seen) == 1
    assert seen[0].definition is definition
    assert seen[0].weights == source
    assert seen[0].teacher_weights == teacher
