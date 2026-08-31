"""Strict config for fresh-array Winsorized transfer and open-loop Adam."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from experiments.mnist_relu.config import _keys, _object
from experiments.mnist_relu_drn.config import (
    EXPERIMENT_ID as STUDENT_EXPERIMENT_ID,
    StudentConfig,
    StudentTrainSpec,
    parse_student_config,
    resolve_student_spec,
)
from experiments.schema import RunMode, config_error


EXPERIMENT_ID = "mnist_ibm_om_winsorized_cross_array_open_loop_adam.v1"
SCHEMA_VERSION = 1
SOURCE_ASSIGNMENT_SEED = 87003
TARGET_ASSIGNMENT_SEED = 87004
TARGET_ENDPOINT_SEEDS = (89401, 89402, 89403, 89404, 89405)
FINE_TUNE_ENDPOINT_SEED = 89401
PULSE_SELECTION_SEED = 90404
EXPECTED_TEACHER_WEIGHTS_PATH = "data/mnist_relu_teacher_fixed_init_20260816.pt"
EXPECTED_TEACHER_WEIGHTS_SHA256 = (
    "9a961a77628e365b54fdf59304ec7fddf46f1f4834c4c03cecea9c204f837f52"
)
SOURCE_RECOVERY_RESULT_ID = (
    "mnist-ibm-om-winsorized-onchip-adam-exploratory-20260829-v1"
)
SOURCE_RECOVERY_RUN_RELATIVE_PATH = (
    "runs/alpha_000_spacing_1delta/"
    "20260828T224530.737807Z-1ae7917a-10efc98a"
)
SOURCE_RECOVERY_CHECKPOINT_SHA256 = (
    "13c7286725f537e2f296fcde24f9b492295a1650574373a2425cb66585f69c96"
)
SOURCE_RECOVERY_SUMMARY_SHA256 = (
    "95a2c611473fa9ea9c96690557eab64d3325e75715d7ae413b24e649b559a792"
)
SOURCE_RECOVERY_TEST_CORRECT = 9414
SOURCE_RECOVERY_TEST_EXAMPLES = 10_000
NOMINAL_DELTA_X = 0.04745
MAXIMUM_PROGRAM_PULSES = 128
RECOVERY_PULSE_CAP = 64
MAXIMUM_RANDOM_DRAWS = 1 + 2 * (MAXIMUM_PROGRAM_PULSES + RECOVERY_PULSE_CAP)
LEARNING_RATE_RAW_X = 3.0e-5
FIXED_LOGIT_GAIN = 14.12537544622754


@dataclass(frozen=True)
class CrossArraySourceContract:
    assignment_seed: int
    result_id: str
    run_relative_path: str
    checkpoint_sha256: str
    summary_sha256: str
    selected_technique: str
    selected_learning_rate_raw_x: float
    test_correct: int
    test_examples: int
    transfer_object: str


@dataclass(frozen=True)
class CrossArrayTargetContract:
    assignment_seed: int
    role: str
    endpoint_seeds: tuple[int, ...]
    fine_tune_endpoint_seed: int
    identity_policy: str
    winsorization: str
    reset_commissioning_samples: int
    baseline_policy: str


@dataclass(frozen=True)
class CrossArrayDeploymentContract:
    requested_target_policy: str
    target_clipping: bool
    unsupported_target_policy: str
    literal_full_g_diagnostic: bool
    controller: str
    maximum_program_pulses: int
    verify_tolerance_raw_x: float
    maximum_random_draws: int
    inference_read_noise: bool
    apparent_endpoint_applied_to_drn: bool


@dataclass(frozen=True)
class CrossArrayRecoveryContract:
    optimizer: str
    learning_rate_raw_x: float
    beta1: float
    beta2: float
    epsilon: float
    epochs: int
    maximum_batches: int | None
    pulse_cap: int
    pulse_selection_seed: int
    update_surface: str
    conceptual_column_phases_per_minibatch: tuple[int, int]
    vectorization_equivalence: str
    verify_reads_during_updates: int
    validation_role: str
    test_role: str


@dataclass(frozen=True)
class CrossArrayExecutionContract:
    evidence_tier: str
    claim_label: str
    evidence_class: str
    state_authority: str
    device: str


@dataclass(frozen=True)
class WinsorizedCrossArrayOpenLoopAdamProtocol:
    expected_teacher_weights_path: str
    expected_teacher_weights_sha256: str
    source: CrossArraySourceContract
    target: CrossArrayTargetContract
    deployment: CrossArrayDeploymentContract
    recovery: CrossArrayRecoveryContract
    execution: CrossArrayExecutionContract


@dataclass(frozen=True)
class WinsorizedCrossArrayOpenLoopAdamConfig:
    schema_version: int
    experiment_id: str
    protocol: WinsorizedCrossArrayOpenLoopAdamProtocol
    student: StudentConfig


@dataclass(frozen=True)
class WinsorizedCrossArrayOpenLoopAdamTrainSpec:
    experiment_id: str
    protocol: WinsorizedCrossArrayOpenLoopAdamProtocol
    student: StudentTrainSpec


def _student_payload() -> Mapping[str, Any]:
    return {
        "schema_version": 1,
        "experiment_id": STUDENT_EXPERIMENT_ID,
        "runtime": {
            "seed": 42,
            "data_seed": 42,
            "device": "cuda",
            "dtype": "float32",
        },
        "data": {
            "batch_size": 16,
            "validation_points": 5000,
            "num_points": None,
            "shuffle": True,
        },
        "teacher": {
            "type": "bias_free_relu",
            "initialization": "signed_weight_mapping",
        },
        "model": {
            "dims": [1568, 100, 20],
            "input_gain": 100.0,
            "conductance_min": 0.0,
            "conductance_max": 2.0,
            "voltage_amp": 4.0,
            "current_amp": 0.25,
            "encoding": "single",
            "include_biases": False,
            "non_linearity": {
                "type": "perfect_diode",
                "quadratic_diode_param": {},
                "exponential_diode_param": {},
                "hard_sigmoid_param": {},
            },
        },
        "solver": {
            "inference_iterations": 4,
            "training_iterations": 4,
            "mode": "asynchronous",
            "overrelaxation_factor": 1.1,
        },
        "mapping": {
            "scale_fractions": [1.0],
            "scale_fraction_pairs": [[1.0, 1.0]],
            "range_placement": "lower",
            "calibration_examples": 1024,
            "calibration_batch_size": 128,
            "logit_gain_min": 0.001,
            "logit_gain_max": 1000.0,
            "logit_gain_steps": 121,
        },
        "modes": {
            "train": {
                "num_epochs": 1,
                "learning_rates": [0.0, 0.0],
                "temperature": 1.0,
                "log_every": 1,
                "max_batches": None,
                "max_validation_batches": None,
                "minimum_relative_kl_improvement": 0.0,
                "weight_modifier": {"type": "none", "parameters": {}},
                "update_backend": {"type": "ideal", "parameters": {}},
                "selection_evaluation": "clean",
                "selection_metric": "student_accuracy",
                "selection_noise_repeats": 1,
                "selection_weight_modifier": {"type": "none", "parameters": {}},
            }
        },
    }


def parse_winsorized_cross_array_open_loop_adam_config(
    payload: Mapping[str, Any],
) -> WinsorizedCrossArrayOpenLoopAdamConfig:
    raw = _object(payload, "config")
    _keys(
        raw,
        "config",
        {
            "schema_version",
            "experiment_id",
            "source_assignment_seed",
            "target_assignment_seed",
            "target_endpoint_seeds",
            "fine_tune_endpoint_seed",
        },
    )
    if raw["schema_version"] != SCHEMA_VERSION:
        raise config_error("config.schema_version", "to equal 1", raw["schema_version"])
    if raw["experiment_id"] != EXPERIMENT_ID:
        raise config_error(
            "config.experiment_id", f"to equal {EXPERIMENT_ID!r}", raw["experiment_id"]
        )
    expected = {
        "source_assignment_seed": SOURCE_ASSIGNMENT_SEED,
        "target_assignment_seed": TARGET_ASSIGNMENT_SEED,
        "target_endpoint_seeds": list(TARGET_ENDPOINT_SEEDS),
        "fine_tune_endpoint_seed": FINE_TUNE_ENDPOINT_SEED,
    }
    for name, value in expected.items():
        if raw[name] != value:
            raise config_error(f"config.{name}", f"to equal {value!r}", raw[name])

    protocol = WinsorizedCrossArrayOpenLoopAdamProtocol(
        expected_teacher_weights_path=EXPECTED_TEACHER_WEIGHTS_PATH,
        expected_teacher_weights_sha256=EXPECTED_TEACHER_WEIGHTS_SHA256,
        source=CrossArraySourceContract(
            assignment_seed=SOURCE_ASSIGNMENT_SEED,
            result_id=SOURCE_RECOVERY_RESULT_ID,
            run_relative_path=SOURCE_RECOVERY_RUN_RELATIVE_PATH,
            checkpoint_sha256=SOURCE_RECOVERY_CHECKPOINT_SHA256,
            summary_sha256=SOURCE_RECOVERY_SUMMARY_SHA256,
            selected_technique="direct_physical_rail",
            selected_learning_rate_raw_x=LEARNING_RATE_RAW_X,
            test_correct=SOURCE_RECOVERY_TEST_CORRECT,
            test_examples=SOURCE_RECOVERY_TEST_EXAMPLES,
            transfer_object="signed_full_G_quad_contrast_C_over_2",
        ),
        target=CrossArrayTargetContract(
            assignment_seed=TARGET_ASSIGNMENT_SEED,
            role="fresh_untouched_assignment_excluded_from_QAT_and_source_recovery",
            endpoint_seeds=TARGET_ENDPOINT_SEEDS,
            fine_tune_endpoint_seed=FINE_TUNE_ENDPOINT_SEED,
            identity_policy="counterfactual_repaired_joint_quad_assignment",
            winsorization="sample_then_winsorize_each_raw_a_bound_to_[-1,1]",
            reset_commissioning_samples=8,
            baseline_policy="alpha_zero_shared_destination_RESET_max",
        ),
        deployment=CrossArrayDeploymentContract(
            requested_target_policy=(
                "target_own_shared_destination_baseline_plus_unmodified_source_"
                "signed_contrast_on_sign_selected_rails"
            ),
            target_clipping=False,
            unsupported_target_policy=(
                "retain_requested_target_and_allow_PV_saturation_or_budget_exhaustion"
            ),
            literal_full_g_diagnostic=True,
            controller="one_pulse_program_and_verify",
            maximum_program_pulses=MAXIMUM_PROGRAM_PULSES,
            verify_tolerance_raw_x=0.5 * NOMINAL_DELTA_X,
            maximum_random_draws=MAXIMUM_RANDOM_DRAWS,
            inference_read_noise=False,
            apparent_endpoint_applied_to_drn=False,
        ),
        recovery=CrossArrayRecoveryContract(
            optimizer="digital_Adam",
            learning_rate_raw_x=LEARNING_RATE_RAW_X,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
            epochs=1,
            maximum_batches=None,
            pulse_cap=RECOVERY_PULSE_CAP,
            pulse_selection_seed=PULSE_SELECTION_SEED,
            update_surface=(
                "column_serial_open_loop_stochastic_coincidence_emulator"
            ),
            conceptual_column_phases_per_minibatch=(100, 20),
            vectorization_equivalence=(
                "all_column_phases_touch_disjoint_cells_so_one_vectorized_plant_"
                "pulse_is_state_and_RNG_equivalent"
            ),
            verify_reads_during_updates=0,
            validation_role="diagnostic_only_no_selection_or_retuning",
            test_role="opened_before_and_after_fixed_recovery",
        ),
        execution=CrossArrayExecutionContract(
            evidence_tier="exploratory_noncanonical",
            claim_label=(
                "digital-Adam, column-serial open-loop stochastic-coincidence emulator"
            ),
            evidence_class="model_based_winsorized_IBM_OM_control",
            state_authority="persistent_raw_active_plant_only_after_deployment",
            device="cuda",
        ),
    )
    return WinsorizedCrossArrayOpenLoopAdamConfig(
        schema_version=SCHEMA_VERSION,
        experiment_id=EXPERIMENT_ID,
        protocol=protocol,
        student=parse_student_config(_student_payload()),
    )


def resolve_winsorized_cross_array_open_loop_adam_spec(
    document: WinsorizedCrossArrayOpenLoopAdamConfig, mode: RunMode
) -> WinsorizedCrossArrayOpenLoopAdamTrainSpec:
    if mode is not RunMode.TRAIN:
        raise config_error(
            "the requested run mode",
            "to be 'train' for this train-only experiment",
            mode.value,
        )
    student = resolve_student_spec(document.student, mode)
    if not isinstance(student, StudentTrainSpec):  # pragma: no cover
        raise RuntimeError("Expected the parent parser to resolve training.")
    return WinsorizedCrossArrayOpenLoopAdamTrainSpec(
        experiment_id=EXPERIMENT_ID,
        protocol=document.protocol,
        student=student,
    )


__all__ = [
    "EXPERIMENT_ID",
    "FINE_TUNE_ENDPOINT_SEED",
    "LEARNING_RATE_RAW_X",
    "MAXIMUM_RANDOM_DRAWS",
    "NOMINAL_DELTA_X",
    "SCHEMA_VERSION",
    "SOURCE_ASSIGNMENT_SEED",
    "TARGET_ASSIGNMENT_SEED",
    "TARGET_ENDPOINT_SEEDS",
    "WinsorizedCrossArrayOpenLoopAdamConfig",
    "WinsorizedCrossArrayOpenLoopAdamProtocol",
    "WinsorizedCrossArrayOpenLoopAdamTrainSpec",
    "parse_winsorized_cross_array_open_loop_adam_config",
    "resolve_winsorized_cross_array_open_loop_adam_spec",
]
