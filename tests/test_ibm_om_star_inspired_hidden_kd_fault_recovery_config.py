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
from experiments.mnist_relu_drn.ibm_om_star_inspired_hidden_kd_fault_recovery_config import (
    CORRUPT_DEVICES_RANGE_RAW_A,
    EXPERIMENT_ID,
    FAULT_PROBABILITY,
    StarInspiredHiddenKdFaultRecoveryTrainSpec,
    parse_star_inspired_hidden_kd_fault_recovery_config,
)
from experiments.schema import ConfigError, RunMode


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "examples/mnist_relu_drn/ibm_om_star_inspired_hidden_kd_fault_recovery/"
    "fault_only_stochastic.json"
)
RUNTIME_MODULE = (
    "experiments.mnist_relu_drn."
    "ibm_om_star_inspired_hidden_kd_fault_recovery_runtime"
)


def test_star_config_resolves_corrected_fault_only_protocol() -> None:
    definition, spec = resolve_experiment_config(CONFIG_PATH, RunMode.TRAIN)

    assert definition is EXPERIMENT_REGISTRY[EXPERIMENT_ID]
    assert definition.supported_modes == (RunMode.TRAIN,)
    assert isinstance(spec, StarInspiredHiddenKdFaultRecoveryTrainSpec)

    faults = spec.protocol.faults
    assert faults.policy == "aihwkit_corrupt_device_compatible"
    assert faults.probability == pytest.approx(FAULT_PROBABILITY)
    assert faults.probability == pytest.approx(0.1348)
    assert faults.independent_per_crosspoint is True
    assert faults.corrupt_devices_range_raw_a == pytest.approx(
        CORRUPT_DEVICES_RANGE_RAW_A
    )
    assert faults.corrupt_devices_range_raw_a == pytest.approx(0.01)
    assert faults.stuck_value_distribution == "uniform_support_intersection"
    assert faults.pulse_directions_disabled is True
    assert faults.mask_exposed_to_learner is False

    execution = spec.protocol.execution
    assert execution.raw_device_coordinate == "a_in_minus1_plus1"
    assert (
        execution.circuit_conductance_coordinate
        == "G_equals_a_plus_1_equals_2x"
    )
    assert execution.negative_conductance_allowed is False
    assert spec.student.model.conductance_min == 0.0
    assert spec.student.model.conductance_max == 2.0

    # A raw stuck coordinate a in [-0.01, 0.01] is shifted into a strictly
    # positive circuit conductance G=a+1 in [0.99, 1.01].
    assert 1.0 - faults.corrupt_devices_range_raw_a == pytest.approx(0.99)
    assert 1.0 + faults.corrupt_devices_range_raw_a == pytest.approx(1.01)

    assert execution.program_verify is False
    assert execution.verify_reads == 0
    assert execution.inference_read_noise is False
    assert execution.cycle_to_cycle_noise is False
    assert execution.apparent_write_noise is False


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("target_assignment_seed", 87003),
        ("development_fault_seed", 91502),
        ("evaluation_fault_seeds", [91502]),
    ),
)
def test_star_config_rejects_protocol_drift(field: str, value: object) -> None:
    payload = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "target_assignment_seed": 87004,
        "development_fault_seed": 91501,
        "evaluation_fault_seeds": [91502, 91503, 91504, 91505],
    }
    payload[field] = value
    with pytest.raises(ConfigError):
        parse_star_inspired_hidden_kd_fault_recovery_config(payload)


def test_star_default_cli_dispatch_is_lazy_and_additive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = ModuleType(RUNTIME_MODULE)
    seen: list[TrainRequest] = []

    def run_train(request: TrainRequest) -> int:
        seen.append(request)
        return 47

    runtime.run_train = run_train  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, RUNTIME_MODULE, runtime)

    teacher = ROOT / "data/mnist_relu_teacher_fixed_init_20260816.pt"
    result = main(
        [
            "train",
            "--config",
            str(CONFIG_PATH),
            "--output-dir",
            str(tmp_path / "run"),
            "--teacher-weights",
            str(teacher),
        ]
    )

    assert result == 47
    assert len(seen) == 1
    request = seen[0]
    assert request.definition is EXPERIMENT_REGISTRY[EXPERIMENT_ID]
    assert isinstance(request.spec, StarInspiredHiddenKdFaultRecoveryTrainSpec)
    assert request.config_path == CONFIG_PATH
    assert request.output_dir == tmp_path / "run"
    assert request.weights is None
    assert request.teacher_weights == teacher
