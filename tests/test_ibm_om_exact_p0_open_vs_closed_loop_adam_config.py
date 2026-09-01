from __future__ import annotations

import json
from pathlib import Path
import sys
from types import ModuleType

import pytest

from ebl.cli import TrainRequest, main
from experiments.definitions import EXPERIMENT_REGISTRY, resolve_experiment_config
from experiments.mnist_relu_drn.ibm_om_exact_p0_open_vs_closed_loop_adam_config import (
    EXPERIMENT_ID,
    EXPECTED_DEVICE_MODEL_SHA256,
    EXPECTED_P0_SHA256,
    EXPECTED_PUBLISHED_POPULATION_FINGERPRINT,
    EXPECTED_PUBLISHED_POPULATION_SHA256,
    RECOVERY_ARM_IDS,
    ExactP0OpenVsClosedLoopAdamTrainSpec,
    parse_exact_p0_open_vs_closed_loop_adam_config,
)
from experiments.schema import ConfigError, RunMode


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "examples/mnist_relu_drn/ibm_om_exact_p0_open_vs_closed_loop_adam/paired_recovery.json"
)
RUNTIME_MODULE = (
    "experiments.mnist_relu_drn.ibm_om_exact_p0_open_vs_closed_loop_adam_runtime"
)


def _payload() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_config_resolves_exact_matched_comparator() -> None:
    definition, spec = resolve_experiment_config(CONFIG_PATH, RunMode.TRAIN)
    assert definition is EXPERIMENT_REGISTRY[EXPERIMENT_ID]
    assert isinstance(spec, ExactP0OpenVsClosedLoopAdamTrainSpec)
    protocol = spec.protocol
    assert protocol.p0.assignment_seed == 94004
    assert protocol.p0.endpoint_seed == 94301
    assert protocol.p0.sha256 == EXPECTED_P0_SHA256
    assert protocol.p0.recovery_data_order_seed == 94402
    assert protocol.device_model.receipt_sha256 == EXPECTED_DEVICE_MODEL_SHA256
    assert (
        protocol.device_model.published_population_sha256
        == EXPECTED_PUBLISHED_POPULATION_SHA256
    )
    assert (
        protocol.device_model.published_population_fingerprint
        == EXPECTED_PUBLISHED_POPULATION_FINGERPRINT
    )
    recovery = protocol.recovery
    assert tuple(arm.arm_id for arm in recovery.arms) == RECOVERY_ARM_IDS
    assert recovery.objective == "teacher_output_kl_only"
    assert recovery.updated_layer_indices == (0, 1)
    assert recovery.epochs == 3
    assert recovery.learning_rate_raw_x == pytest.approx(3e-5)
    assert recovery.pulse_cap == 64
    assert recovery.arms[1].maximum_pulses_per_cell_per_minibatch == 1
    assert recovery.arms[1].desired_target_projection == (
        "clamp_to_[0,1]_after_each_Adam_increment"
    )
    assert recovery.test_policy.startswith("sealed_until_both")
    effective_history = (
        protocol.historical_open_loop_parity
        .cumulative_effective_state_change_events_by_epoch
    )
    assert effective_history == (
        46060,
        93284,
        141515,
    )
    assert spec.student.runtime.device == "cuda"


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("target_assignment_seed", 94003),
        ("p0_endpoint_seed", 94302),
        ("p0_sha256", "0" * 64),
        ("device_model_sha256", "1" * 64),
        ("published_population_sha256", "2" * 64),
        ("published_population_fingerprint", "3" * 64),
        ("recovery_data_order_seed", 94403),
        ("recovery_pulse_selection_seed", 94402),
        ("arms", list(reversed(RECOVERY_ARM_IDS))),
    ),
)
def test_config_rejects_protocol_drift(field: str, replacement: object) -> None:
    payload = _payload()
    payload[field] = replacement
    with pytest.raises(ConfigError):
        parse_exact_p0_open_vs_closed_loop_adam_config(payload)


def test_default_cli_dispatches_all_three_explicit_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = ModuleType(RUNTIME_MODULE)
    seen: list[TrainRequest] = []

    def run_train(request: TrainRequest) -> int:
        seen.append(request)
        return 67

    runtime.run_train = run_train  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, RUNTIME_MODULE, runtime)
    teacher = ROOT / "data/mnist_relu_teacher_fixed_init_20260816.pt"
    p0 = tmp_path / "exact_p0.pt"
    model = tmp_path / "paired.json"
    result = main(
        [
            "train",
            "--config",
            str(CONFIG_PATH),
            "--output-dir",
            str(tmp_path / "run"),
            "--teacher-weights",
            str(teacher),
            "--weights",
            str(p0),
            "--device-model",
            str(model),
        ]
    )
    assert result == 67
    assert len(seen) == 1
    assert seen[0].weights == p0
    assert seen[0].device_model == model
    assert seen[0].teacher_weights == teacher
