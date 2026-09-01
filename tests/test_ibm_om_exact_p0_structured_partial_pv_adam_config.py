from __future__ import annotations

import json
from pathlib import Path
import sys
from types import ModuleType

import pytest

from ebl.cli import TrainRequest, main
from experiments.definitions import EXPERIMENT_REGISTRY, resolve_experiment_config
from experiments.mnist_relu_drn.ibm_om_exact_p0_open_vs_closed_loop_adam_config import (
    EXPECTED_DEVICE_MODEL_SHA256,
    EXPECTED_P0_SHA256,
    EXPECTED_TEACHER_WEIGHTS_SHA256,
    ExactP0OpenVsClosedLoopAdamProtocol,
)
from experiments.mnist_relu_drn.ibm_om_exact_p0_structured_partial_pv_adam_config import (
    ARM_IDS,
    EXPERIMENT_ID,
    RANKING_AGGREGATION,
    RANKING_COHORT,
    RANKING_TIE_BREAK,
    ExactP0StructuredPartialPvAdamTrainSpec,
    parse_exact_p0_structured_partial_pv_adam_config,
)
from experiments.schema import ConfigError, RunMode


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "examples/mnist_relu_drn/ibm_om_exact_p0_structured_partial_pv_adam/partial_sweep.json"
)
RUNTIME_MODULE = (
    "experiments.mnist_relu_drn.ibm_om_exact_p0_structured_partial_pv_adam_runtime"
)


def _payload() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_config_resolves_exact_p0_structured_partial_protocol() -> None:
    definition, spec = resolve_experiment_config(CONFIG_PATH, RunMode.TRAIN)
    assert definition is EXPERIMENT_REGISTRY[EXPERIMENT_ID]
    assert isinstance(spec, ExactP0StructuredPartialPvAdamTrainSpec)
    assert spec.experiment_id == EXPERIMENT_ID
    assert isinstance(spec.protocol.base, ExactP0OpenVsClosedLoopAdamProtocol)

    base = spec.protocol.base
    assert base.expected_teacher_weights_sha256 == EXPECTED_TEACHER_WEIGHTS_SHA256
    assert base.p0.assignment_seed == 94004
    assert base.p0.endpoint_seed == 94301
    assert base.p0.sha256 == EXPECTED_P0_SHA256
    assert base.device_model.receipt_sha256 == EXPECTED_DEVICE_MODEL_SHA256
    assert base.recovery.epochs == 3
    assert base.recovery.learning_rate_raw_x == pytest.approx(3.0e-5)
    assert base.recovery.pulse_cap == 64

    partial = spec.protocol.partial
    assert partial.controller == base.recovery.arms[1].arm_id
    assert partial.epochs == base.recovery.epochs == spec.student.settings.num_epochs
    assert partial.matched_logical_count == 500
    assert tuple(arm.arm_id for arm in partial.arms) == ARM_IDS
    assert [arm.expected_logical_quads for arm in partial.arms] == [
        39_700,
        39_200,
        500,
        500,
        500,
    ]
    assert [arm.matched_500_quad_address_arm for arm in partial.arms] == [
        False,
        False,
        True,
        True,
        True,
    ]
    assert partial.mask_application == ("Adam_target_updates", "pulse_eligibility")
    assert partial.frozen_cell_policy == (
        "persistent_full_G_forward_with_zero_update_pulses"
    )

    ranking = partial.ranking
    assert ranking.cohort == RANKING_COHORT
    assert (ranking.examples, ranking.batch_size, ranking.minibatches) == (1024, 128, 8)
    assert ranking.objective == "teacher_output_kl_only"
    assert ranking.aggregation == RANKING_AGGREGATION
    assert ranking.tie_break == RANKING_TIE_BREAK
    assert ranking.global_flat_order == "W1_then_W2_row_major_logical_quad_order"
    assert not any(
        (
            ranking.uses_new_seed,
            ranking.uses_device_bounds,
            ranking.uses_corrupt_mask,
            ranking.uses_validation,
            ranking.uses_test,
        )
    )
    assert spec.student.runtime.device == "cuda"
    assert spec.student.mapping.calibration_examples == 1024
    assert spec.student.mapping.calibration_batch_size == 128


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("teacher_weights_sha256", "0" * 64),
        ("target_assignment_seed", 94003),
        ("p0_endpoint_seed", 94302),
        ("p0_sha256", "1" * 64),
        ("device_model_sha256", "2" * 64),
        ("recovery_epochs", 2),
        ("recovery_learning_rate_raw_x", 4.0e-5),
        ("nominal_delta_x", 0.05),
        ("ranking_cohort", "new loader"),
        ("ranking_aggregation", "per-example gradients"),
        ("ranking_tie_break", "unstable"),
        ("matched_logical_count", 499),
        ("arms", list(reversed(ARM_IDS))),
    ),
)
def test_config_rejects_protocol_drift(field: str, replacement: object) -> None:
    payload = _payload()
    payload[field] = replacement
    with pytest.raises(ConfigError):
        parse_exact_p0_structured_partial_pv_adam_config(payload)


def test_default_cli_dispatches_all_three_explicit_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = ModuleType(RUNTIME_MODULE)
    seen: list[TrainRequest] = []

    def run_train(request: TrainRequest) -> int:
        seen.append(request)
        return 73

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
    assert result == 73
    assert len(seen) == 1
    assert seen[0].weights == p0
    assert seen[0].device_model == model
    assert seen[0].teacher_weights == teacher
