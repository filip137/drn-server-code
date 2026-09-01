from __future__ import annotations

import json
from pathlib import Path
import sys
from types import ModuleType

import pytest

from ebl.cli import TrainRequest, main
from experiments.definitions import EXPERIMENT_REGISTRY, resolve_experiment_config
from experiments.mnist_relu_drn.ibm_om_exact_p0_hybrid_fraction_pv_adam import ARM_IDS
from experiments.mnist_relu_drn.ibm_om_exact_p0_hybrid_fraction_pv_adam_config import (
    EXPERIMENT_ID,
    ExactP0HybridFractionPvAdamTrainSpec,
    parse_exact_p0_hybrid_fraction_pv_adam_config,
)
from experiments.schema import ConfigError, RunMode


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "examples/mnist_relu_drn/ibm_om_exact_p0_hybrid_fraction_pv_adam/fraction_sweep.json"
RUNTIME_MODULE = (
    "experiments.mnist_relu_drn.ibm_om_exact_p0_hybrid_fraction_pv_adam_runtime"
)


def _payload() -> dict[str, object]:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_config_resolves_exact_nested_fraction_and_sealed_test_contract() -> None:
    definition, spec = resolve_experiment_config(CONFIG, RunMode.TRAIN)
    assert definition is EXPERIMENT_REGISTRY[EXPERIMENT_ID]
    assert isinstance(spec, ExactP0HybridFractionPvAdamTrainSpec)
    partial = spec.protocol.partial
    assert tuple(arm.arm_id for arm in partial.arms) == ARM_IDS
    assert partial.top_w1_candidate_counts == (125, 250, 500)
    assert partial.random_selector_seed == 94_501
    assert partial.expected_random_w1_index_sha256 == (
        "85e1f3f7b1046d3acc8065b8460354950eda03973173159782a93f4b45b77887"
    )
    assert partial.minimum_validation_accuracy_gain == pytest.approx(0.01)
    assert partial.maximum_stage1_full_issued_pulse_fraction == pytest.approx(0.25)
    assert partial.require_lower_validation_kl
    assert not partial.stage1_reference.rerun_full
    assert "random500_is_validation_only" in partial.test_policy
    assert partial.physical_conductance_formula == "nonnegative_G=a+1=2*x"
    assert [arm.terminal_test_eligible for arm in partial.arms] == [
        True,
        True,
        True,
        True,
        False,
    ]


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("p0_sha256", "0" * 64),
        ("recovery_epochs", 2),
        ("top_w1_candidate_counts", [250, 500]),
        ("random_selector_seed", 94_502),
        ("expected_random_w1_index_sha256", "1" * 64),
        ("minimum_validation_accuracy_gain", 0.02),
        ("require_lower_validation_kl", False),
        ("maximum_stage1_full_issued_pulse_fraction", 0.5),
        ("stage1_summary_sha256", "2" * 64),
        ("arms", list(reversed(ARM_IDS))),
    ),
)
def test_config_rejects_protocol_drift(field: str, replacement: object) -> None:
    payload = _payload()
    payload[field] = replacement
    with pytest.raises(ConfigError):
        parse_exact_p0_hybrid_fraction_pv_adam_config(payload)


def test_cli_dispatches_exact_p0_device_and_teacher_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = ModuleType(RUNTIME_MODULE)
    seen: list[TrainRequest] = []

    def run_train(request: TrainRequest) -> int:
        seen.append(request)
        return 79

    runtime.run_train = run_train  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, RUNTIME_MODULE, runtime)
    teacher = ROOT / "data/mnist_relu_teacher_fixed_init_20260816.pt"
    p0 = tmp_path / "exact_p0.pt"
    model = tmp_path / "paired.json"
    result = main(
        [
            "train",
            "--config",
            str(CONFIG),
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
    assert result == 79
    assert len(seen) == 1
    assert seen[0].teacher_weights == teacher
    assert seen[0].weights == p0
    assert seen[0].device_model == model
