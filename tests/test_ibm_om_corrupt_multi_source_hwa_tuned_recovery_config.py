from __future__ import annotations

from pathlib import Path
import sys
from types import ModuleType

import pytest

from ebl.cli import TrainRequest, main
from experiments.definitions import EXPERIMENT_REGISTRY, resolve_experiment_config
from experiments.mnist_relu_drn.ibm_om_corrupt_multi_source_hwa_tuned_recovery_config import (
    ARM_SIMPLICITY_ORDER,
    BASE_HWA_LEARNING_RATES,
    EXPERIMENT_ID,
    HWA_ARM_IDS,
    HWA_EPOCHS,
    CorruptMultiSourceHwaTunedRecoveryTrainSpec,
    parse_corrupt_multi_source_hwa_tuned_recovery_config,
)
from experiments.schema import ConfigError, RunMode


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "examples/mnist_relu_drn/"
    "ibm_om_corrupt_multi_source_hwa_tuned_recovery/standard_ladder.json"
)
RUNTIME_MODULE = (
    "experiments.mnist_relu_drn."
    "ibm_om_corrupt_multi_source_hwa_tuned_recovery_runtime"
)


def _payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "training_assignment_seeds": [94001, 94002],
        "training_endpoint_seeds": [
            [94101, 94102, 94103, 94104],
            [94111, 94112, 94113, 94114],
        ],
        "development_assignment_seed": 94003,
        "development_endpoint_seeds": [94201, 94202, 94203, 94204],
        "target_assignment_seed": 94004,
        "target_endpoint_seeds": [94301, 94302, 94303, 94304, 94305],
        "p0_endpoint_seed": 94301,
        "recovery_pulse_selection_seed": 94401,
        "recovery_data_order_seed": 94402,
        "arms": list(HWA_ARM_IDS),
    }


def test_config_resolves_frozen_multi_source_tuned_ladder() -> None:
    definition, spec = resolve_experiment_config(CONFIG_PATH, RunMode.TRAIN)
    assert definition is EXPERIMENT_REGISTRY[EXPERIMENT_ID]
    assert definition.supported_modes == (RunMode.TRAIN,)
    assert isinstance(spec, CorruptMultiSourceHwaTunedRecoveryTrainSpec)

    protocol = spec.protocol
    assert protocol.source.training_assignment_seeds == (94001, 94002)
    assert protocol.source.training_endpoint_seeds_by_assignment == (
        (94001, (94101, 94102, 94103, 94104)),
        (94002, (94111, 94112, 94113, 94114)),
    )
    assert protocol.source.training_endpoint_seeds == (
        (94101, 94102, 94103, 94104),
        (94111, 94112, 94113, 94114),
    )
    assert protocol.source.development_assignment_seed == 94003
    assert protocol.source.development_endpoint_seeds == (94201, 94202, 94203, 94204)

    training = protocol.training
    assert training.epochs == HWA_EPOCHS == 6
    assert training.base_learning_rates == BASE_HWA_LEARNING_RATES
    assert tuple(arm.arm_id for arm in training.arms) == HWA_ARM_IDS
    assert training.arms[0].training_assignment_seeds == (94001,)
    assert training.arms[2].training_assignment_seeds == (94001, 94002)
    assert training.arms[0].w1_learning_rate == pytest.approx(
        BASE_HWA_LEARNING_RATES[0]
    )
    assert training.arms[3].w1_learning_rate == pytest.approx(
        BASE_HWA_LEARNING_RATES[0]
    )
    assert training.arms[0].w2_learning_rate == pytest.approx(
        BASE_HWA_LEARNING_RATES[1]
    )
    assert training.arms[1].w2_learning_rate == pytest.approx(
        30.0 * BASE_HWA_LEARNING_RATES[1]
    )
    assert tuple(arm.w2_learning_rate_factor for arm in training.arms) == (
        1,
        30,
        1,
        30,
    )
    assert training.assignment_schedule == (
        "global_minibatch_ordinal_modulo_source_count_start_94001"
    )
    assert training.persistent_sample_schedule == (
        "cyclic_adjacent_pair_by_assignment_visit_ordinal"
    )
    assert training.endpoint_objective == "mean2"
    assert training.ideal_gradient_weight == pytest.approx(0.25)
    assert training.persistent_gradient_weights == pytest.approx((0.375, 0.375))

    selection = protocol.selection
    assert selection.candidate_unit == "joint_arm_and_trained_epoch"
    assert selection.epoch_zero_eligible is False
    assert selection.metric_order == (
        "persistent_mean_accuracy_descending",
        "persistent_minimum_accuracy_descending",
        "persistent_mean_kl_teacher_student_ascending",
        "epoch_ascending",
        "arm_simplicity_order_ascending",
    )
    assert selection.arm_simplicity_order == ARM_SIMPLICITY_ORDER
    assert selection.target_progression == (
        "only_joint_winner_plus_common_epoch0_control_to_sealed_target"
    )

    assert protocol.target.assignment_seed == 94004
    assert protocol.target.endpoint_seeds == (94301, 94302, 94303, 94304, 94305)
    assert protocol.target.fine_tune_endpoint_seed == 94301
    assert protocol.deployment.maximum_random_draws == 385
    assert protocol.deployment.target_clipping is False

    recovery = protocol.recovery
    assert recovery.objective == "teacher_output_kl_only"
    assert recovery.output_kl_only is True
    assert recovery.updated_layer_indices == (0, 1)
    assert recovery.frozen_layer_indices == ()
    assert recovery.start_state == "selected_hwa_target_seed_94301_P0"
    assert recovery.epochs == 3
    assert recovery.learning_rate_raw_x == pytest.approx(3.0e-5)
    assert recovery.pulse_cap == 64
    assert recovery.maximum_random_draws == 385
    assert recovery.pulse_selection_seed == 94401
    assert recovery.data_order_seed == 94402
    assert recovery.verify_reads_during_updates == 0
    assert recovery.conceptual_column_phases_per_minibatch == (100, 20)
    assert recovery.validation_used_for_selection is True
    assert recovery.checkpoint_policy == (
        "target_validation_accuracy_then_KL_then_earlier_epoch_"
        "among_epochs_1_to_3_before_test"
    )
    assert protocol.mapping.fixed_logit_gain == pytest.approx(14.12537544622754)
    assert spec.student.runtime.device == "cuda"


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("training_assignment_seeds", [94001]),
        ("training_endpoint_seeds", [[94101, 94102, 94103, 94104]]),
        ("development_assignment_seed", 94004),
        ("target_assignment_seed", 94003),
        ("target_endpoint_seeds", [94301]),
        ("p0_endpoint_seed", 94302),
        ("recovery_pulse_selection_seed", 94402),
        ("recovery_data_order_seed", 94403),
        ("arms", list(reversed(HWA_ARM_IDS))),
    ),
)
def test_config_rejects_protocol_drift(field: str, value: object) -> None:
    payload = _payload()
    payload[field] = value
    with pytest.raises(ConfigError):
        parse_corrupt_multi_source_hwa_tuned_recovery_config(payload)


def test_default_cli_dispatch_is_lazy_and_additive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = ModuleType(RUNTIME_MODULE)
    seen: list[TrainRequest] = []

    def run_train(request: TrainRequest) -> int:
        seen.append(request)
        return 61

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
    assert result == 61
    assert len(seen) == 1
    request = seen[0]
    assert request.definition is EXPERIMENT_REGISTRY[EXPERIMENT_ID]
    assert isinstance(request.spec, CorruptMultiSourceHwaTunedRecoveryTrainSpec)
    assert request.config_path == CONFIG_PATH
    assert request.output_dir == tmp_path / "run"
    assert request.weights is None
    assert request.teacher_weights == teacher
