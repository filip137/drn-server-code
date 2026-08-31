from __future__ import annotations

from pathlib import Path
import sys
from types import ModuleType

import pytest

from ebl.cli import TrainRequest, main
from experiments.definitions import EXPERIMENT_REGISTRY, resolve_experiment_config
from experiments.mnist_relu_drn.ibm_om_corrupt_source_hwa_cross_array_pulse_adam_config import (
    EXPERIMENT_ID,
    HWA_LEARNING_RATES,
    LEVEL_SPACING_RAW_X,
    CorruptSourceHwaCrossArrayPulseAdamTrainSpec,
    parse_corrupt_source_hwa_cross_array_pulse_adam_config,
)
from experiments.schema import ConfigError, RunMode


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "examples/mnist_relu_drn/"
    "ibm_om_corrupt_source_hwa_cross_array_pulse_adam/standard_ladder.json"
)
RUNTIME_MODULE = (
    "experiments.mnist_relu_drn."
    "ibm_om_corrupt_source_hwa_cross_array_pulse_adam_runtime"
)


def _payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "source_assignment_seed": 93001,
        "source_train_endpoint_seeds": [93201, 93202, 93203, 93204],
        "source_selection_endpoint_seeds": [93221, 93222, 93223, 93224],
        "target_assignment_seed": 93002,
        "target_endpoint_seeds": [93301, 93302, 93303, 93304, 93305],
        "p0_endpoint_seed": 93301,
    }


def test_config_resolves_the_matched_standard_ladder() -> None:
    definition, spec = resolve_experiment_config(CONFIG_PATH, RunMode.TRAIN)
    assert definition is EXPERIMENT_REGISTRY[EXPERIMENT_ID]
    assert isinstance(spec, CorruptSourceHwaCrossArrayPulseAdamTrainSpec)
    protocol = spec.protocol
    assert protocol.source.assignment_seed == 93001
    assert protocol.source.train_endpoint_seeds == (93201, 93202, 93203, 93204)
    assert protocol.source.selection_endpoint_seeds == (93221, 93222, 93223, 93224)
    assert protocol.training.epochs == 3
    assert protocol.training.learning_rates == HWA_LEARNING_RATES
    assert protocol.training.endpoint_objective == "mean2"
    assert protocol.training.ideal_gradient_weight == pytest.approx(0.25)
    assert protocol.training.persistent_gradient_weights == pytest.approx((0.375, 0.375))
    assert protocol.mapping.level_spacing_raw_x == pytest.approx(LEVEL_SPACING_RAW_X)
    assert protocol.mapping.conductance_formula == "G=raw_a_winsorized+1=2*x"
    assert protocol.mapping.negative_conductance_allowed is False
    assert protocol.target.assignment_seed == 93002
    assert protocol.target.endpoint_seeds == (93301, 93302, 93303, 93304, 93305)
    assert protocol.target.fine_tune_endpoint_seed == 93301
    assert protocol.deployment.target_clipping is False
    assert protocol.deployment.persistent_endpoint_applied_to_drn is True
    assert protocol.deployment.apparent_endpoint_applied_to_drn is False
    assert protocol.recovery.optimizer == "digital_Adam"
    assert protocol.recovery.pulse_selection_seed == 93401
    assert protocol.recovery.verify_reads_during_updates == 0
    assert spec.student.model.conductance_min == 0.0
    assert spec.student.model.conductance_max == 2.0


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("source_assignment_seed", 93002),
        ("source_train_endpoint_seeds", [93201]),
        ("source_selection_endpoint_seeds", [93201, 93202, 93203, 93204]),
        ("target_assignment_seed", 93001),
        ("target_endpoint_seeds", [93301]),
        ("p0_endpoint_seed", 93302),
    ),
)
def test_config_rejects_protocol_drift(field: str, value: object) -> None:
    payload = _payload()
    payload[field] = value
    with pytest.raises(ConfigError):
        parse_corrupt_source_hwa_cross_array_pulse_adam_config(payload)


def test_default_cli_dispatch_is_lazy_and_additive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = ModuleType(RUNTIME_MODULE)
    seen: list[TrainRequest] = []

    def run_train(request: TrainRequest) -> int:
        seen.append(request)
        return 59

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
    assert result == 59
    assert len(seen) == 1
    assert seen[0].definition is EXPERIMENT_REGISTRY[EXPERIMENT_ID]
    assert seen[0].weights is None
    assert seen[0].teacher_weights == teacher
