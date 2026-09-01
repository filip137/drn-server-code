"""Strict config for exact-P0 open- versus closed-loop Adam recovery.

This exploratory successor changes only the pulse controller used to realize
one shared digital Adam command stream.  Both arms start from the same saved
target-94004/endpoint-94301 physical continuation and MNIST loader state.
"""

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


EXPERIMENT_ID = "mnist_ibm_om_exact_p0_open_vs_closed_loop_adam.v1"
SCHEMA_VERSION = 1

EXPECTED_TEACHER_WEIGHTS_PATH = "data/mnist_relu_teacher_fixed_init_20260816.pt"
EXPECTED_TEACHER_WEIGHTS_SHA256 = (
    "9a961a77628e365b54fdf59304ec7fddf46f1f4834c4c03cecea9c204f837f52"
)
EXPECTED_P0_SHA256 = (
    "299d4800f89f379bd85cb25b824c89ef9fe92313fd82d70e27e7ee3f320bbb04"
)
EXPECTED_DEVICE_MODEL_SHA256 = (
    "2107dd670e769361e325661d77f7dec39e7f331db9cc4a3b283a945bd51d4321"
)
EXPECTED_PUBLISHED_POPULATION_SHA256 = (
    "91844d19fc3a1d3043969fb0d2fa793404d71f9a20ed03977aaa63e8a67c86b5"
)
EXPECTED_PUBLISHED_POPULATION_FINGERPRINT = (
    "487e9ec1fec97f4f43ce102198fc05afe994760200c675d5595bcd94e1f54039"
)
EXPECTED_SOURCE_HWA_CHECKPOINT_SHA256 = (
    "d976b28789b1799cc6c7b1c0d66aadf4f8d635a5dce56ede5d6f1d76c1e41b72"
)

TARGET_ASSIGNMENT_SEED = 94004
P0_ENDPOINT_SEED = 94301
RECOVERY_PULSE_SELECTION_SEED = 94401
RECOVERY_DATA_ORDER_SEED = 94402
RECOVERY_EPOCHS = 3
RECOVERY_LEARNING_RATE_RAW_X = 3.0e-5
NOMINAL_DELTA_X = 0.04745
VERIFY_TOLERANCE_RAW_X = NOMINAL_DELTA_X / 2.0
RECOVERY_PULSE_CAP = 64
MAXIMUM_RANDOM_DRAWS = 385
FIXED_LOGIT_GAIN = 14.12537544622754
RECOVERY_ARM_IDS = (
    "open_loop_adam",
    "incremental_one_pulse_closed_loop_target_tracking",
)


@dataclass(frozen=True)
class ExactP0Contract:
    assignment_seed: int
    endpoint_seed: int
    sha256: str
    source_hwa_checkpoint_sha256: str
    state_coordinate: str
    recovery_data_order_seed: int
    maximum_random_draws: int


@dataclass(frozen=True)
class DeviceModelContract:
    assignment_seed: int
    receipt_sha256: str
    published_population_sha256: str
    published_population_fingerprint: str
    corruption_policy: str
    conductance_formula: str


@dataclass(frozen=True)
class RecoveryArmContract:
    arm_id: str
    controller: str
    pulse_selection_seed: int | None
    verify_tolerance_raw_x: float | None
    maximum_pulses_per_cell_per_minibatch: int
    desired_target_initialization: str | None
    desired_target_projection: str | None
    verify_reads_during_updates: str


@dataclass(frozen=True)
class RecoveryContract:
    objective: str
    optimizer: str
    learning_rate_raw_x: float
    beta1: float
    beta2: float
    epsilon: float
    epochs: int
    maximum_batches: int | None
    updated_layer_indices: tuple[int, int]
    pulse_cap: int
    nominal_delta_x: float
    maximum_random_draws: int
    data_order_seed: int
    fixed_logit_gain: float
    arms: tuple[RecoveryArmContract, RecoveryArmContract]
    epoch_selection_metrics: tuple[str, str, str]
    test_policy: str
    state_authority: str


@dataclass(frozen=True)
class HistoricalOpenLoopParityContract:
    initial_train_generator_state_sha256: str
    p0_validation_correct: int
    p0_validation_kl: float
    p0_validation_prediction_sha256: str
    validation_correct_by_epoch: tuple[int, int, int]
    validation_kl_by_epoch: tuple[float, float, float]
    validation_prediction_sha256_by_epoch: tuple[str, str, str]
    cumulative_issued_commands_by_epoch: tuple[int, int, int]
    cumulative_effective_state_change_events_by_epoch: tuple[int, int, int]
    train_generator_state_sha256_by_epoch: tuple[str, str, str]
    selected_epoch: int
    p0_test_correct: int
    p0_test_prediction_sha256: str
    selected_test_correct: int
    selected_test_kl: float
    selected_test_prediction_sha256: str


@dataclass(frozen=True)
class ExactP0OpenVsClosedLoopAdamProtocol:
    expected_teacher_weights_path: str
    expected_teacher_weights_sha256: str
    p0: ExactP0Contract
    device_model: DeviceModelContract
    recovery: RecoveryContract
    historical_open_loop_parity: HistoricalOpenLoopParityContract
    evidence_tier: str


@dataclass(frozen=True)
class ExactP0OpenVsClosedLoopAdamConfig:
    schema_version: int
    experiment_id: str
    protocol: ExactP0OpenVsClosedLoopAdamProtocol
    student: StudentConfig


@dataclass(frozen=True)
class ExactP0OpenVsClosedLoopAdamTrainSpec:
    experiment_id: str
    protocol: ExactP0OpenVsClosedLoopAdamProtocol
    student: StudentTrainSpec


def _student_payload() -> Mapping[str, Any]:
    return {
        "schema_version": 1,
        "experiment_id": STUDENT_EXPERIMENT_ID,
        "runtime": {"seed": 42, "data_seed": 42, "device": "cuda", "dtype": "float32"},
        "data": {
            "batch_size": 16,
            "validation_points": 5000,
            "num_points": None,
            "shuffle": True,
        },
        "teacher": {"type": "bias_free_relu", "initialization": "signed_weight_mapping"},
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
                "num_epochs": RECOVERY_EPOCHS,
                "learning_rates": [RECOVERY_LEARNING_RATE_RAW_X] * 2,
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


def _protocol() -> ExactP0OpenVsClosedLoopAdamProtocol:
    arms = (
        RecoveryArmContract(
            arm_id="open_loop_adam",
            controller="column_serial_open_loop_stochastic_coincidence_emulator",
            pulse_selection_seed=RECOVERY_PULSE_SELECTION_SEED,
            verify_tolerance_raw_x=None,
            maximum_pulses_per_cell_per_minibatch=1,
            desired_target_initialization=None,
            desired_target_projection=None,
            verify_reads_during_updates="zero",
        ),
        RecoveryArmContract(
            arm_id="incremental_one_pulse_closed_loop_target_tracking",
            controller="capability_limited_apparent_feedback_one_pulse_program_verify",
            pulse_selection_seed=None,
            verify_tolerance_raw_x=VERIFY_TOLERANCE_RAW_X,
            maximum_pulses_per_cell_per_minibatch=1,
            desired_target_initialization="clamp((P0_apparent_raw_a+1)/2,0,1)",
            desired_target_projection="clamp_to_[0,1]_after_each_Adam_increment",
            verify_reads_during_updates=(
                "zero_pre_pulse_reads_then_one_full_port_API_call_after_each_issued_pulse_round"
            ),
        ),
    )
    return ExactP0OpenVsClosedLoopAdamProtocol(
        expected_teacher_weights_path=EXPECTED_TEACHER_WEIGHTS_PATH,
        expected_teacher_weights_sha256=EXPECTED_TEACHER_WEIGHTS_SHA256,
        p0=ExactP0Contract(
            assignment_seed=TARGET_ASSIGNMENT_SEED,
            endpoint_seed=P0_ENDPOINT_SEED,
            sha256=EXPECTED_P0_SHA256,
            source_hwa_checkpoint_sha256=EXPECTED_SOURCE_HWA_CHECKPOINT_SHA256,
            state_coordinate="native_raw_active_a",
            recovery_data_order_seed=RECOVERY_DATA_ORDER_SEED,
            maximum_random_draws=MAXIMUM_RANDOM_DRAWS,
        ),
        device_model=DeviceModelContract(
            assignment_seed=TARGET_ASSIGNMENT_SEED,
            receipt_sha256=EXPECTED_DEVICE_MODEL_SHA256,
            published_population_sha256=EXPECTED_PUBLISHED_POPULATION_SHA256,
            published_population_fingerprint=EXPECTED_PUBLISHED_POPULATION_FINGERPRINT,
            corruption_policy="published",
            conductance_formula="G=raw_a_winsorized+1=2*x",
        ),
        recovery=RecoveryContract(
            objective="teacher_output_kl_only",
            optimizer="digital_Adam_shared_equations",
            learning_rate_raw_x=RECOVERY_LEARNING_RATE_RAW_X,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
            epochs=RECOVERY_EPOCHS,
            maximum_batches=None,
            updated_layer_indices=(0, 1),
            pulse_cap=RECOVERY_PULSE_CAP,
            nominal_delta_x=NOMINAL_DELTA_X,
            maximum_random_draws=MAXIMUM_RANDOM_DRAWS,
            data_order_seed=RECOVERY_DATA_ORDER_SEED,
            fixed_logit_gain=FIXED_LOGIT_GAIN,
            arms=arms,
            epoch_selection_metrics=(
                "validation_accuracy_descending",
                "validation_KL_ascending",
                "epoch_ascending",
            ),
            test_policy="sealed_until_both_arm_epoch_selections_freeze_then_terminal_once",
            state_authority="persistent_raw_active_plant_G_equals_a_plus_1",
        ),
        historical_open_loop_parity=HistoricalOpenLoopParityContract(
            initial_train_generator_state_sha256=(
                "3abd672d42069bb67d762e8d1ff67ee34948e7597c4bcdbddae5d9b7a8c57cf4"
            ),
            p0_validation_correct=2235,
            p0_validation_kl=3.3445895696640013,
            p0_validation_prediction_sha256=(
                "e7069b2a11254ae3804b3389aeebf15c52d9d8a7d0725e6d41b29fc3a6c3886c"
            ),
            validation_correct_by_epoch=(4530, 4580, 4610),
            validation_kl_by_epoch=(
                0.2361830812484026,
                0.1999324650734663,
                0.18600318866223098,
            ),
            validation_prediction_sha256_by_epoch=(
                "e7a94686afeccb053b11392ca6f156e1539f9c4c746001e2fb82e2aa4b18a582",
                "3fa0a0277379f813a93b7fa9f75981bcadaab963aa64fe12e97ad9ca48724b8b",
                "a60d3b44e84958e265ea33e778172441a89d74afd5bc720b90fcf2a454d939f5",
            ),
            cumulative_issued_commands_by_epoch=(56068, 113156, 170931),
            cumulative_effective_state_change_events_by_epoch=(
                46060,
                93284,
                141515,
            ),
            train_generator_state_sha256_by_epoch=(
                "56ad35bc1bb14e7274309735fc523734d65ce630057cb14ee2b535c92edb2c2f",
                "6fd2daf706ce9d027c809a8d47ed8c04e5f19467506022049f13231a85c2c037",
                "9f12c6bbeda63372fecc36702b8c5a5795a9510bc10f38327726aba24b0bd8a2",
            ),
            selected_epoch=3,
            p0_test_correct=4445,
            p0_test_prediction_sha256=(
                "8f3252b91bb1fa56f8454d704b9b547edf9ba8c8ffa772ddf7528d3bd5ae4f28"
            ),
            selected_test_correct=9278,
            selected_test_kl=0.16868098396100104,
            selected_test_prediction_sha256=(
                "1ba31cd2115d607acf06d0d4abcf552d2a554ae16090cb5cb2cc966633ab7529"
            ),
        ),
        evidence_tier="exploratory_noncanonical",
    )


def parse_exact_p0_open_vs_closed_loop_adam_config(
    payload: Mapping[str, Any],
) -> ExactP0OpenVsClosedLoopAdamConfig:
    raw = _object(payload, "config")
    expected = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "target_assignment_seed": TARGET_ASSIGNMENT_SEED,
        "p0_endpoint_seed": P0_ENDPOINT_SEED,
        "p0_sha256": EXPECTED_P0_SHA256,
        "device_model_sha256": EXPECTED_DEVICE_MODEL_SHA256,
        "published_population_sha256": EXPECTED_PUBLISHED_POPULATION_SHA256,
        "published_population_fingerprint": EXPECTED_PUBLISHED_POPULATION_FINGERPRINT,
        "recovery_data_order_seed": RECOVERY_DATA_ORDER_SEED,
        "recovery_pulse_selection_seed": RECOVERY_PULSE_SELECTION_SEED,
        "arms": list(RECOVERY_ARM_IDS),
    }
    _keys(raw, "config", set(expected))
    for name, value in expected.items():
        if type(raw[name]) is not type(value) or raw[name] != value:
            raise config_error(f"config.{name}", f"to equal {value!r}", raw[name])
    return ExactP0OpenVsClosedLoopAdamConfig(
        schema_version=SCHEMA_VERSION,
        experiment_id=EXPERIMENT_ID,
        protocol=_protocol(),
        student=parse_student_config(_student_payload()),
    )


def resolve_exact_p0_open_vs_closed_loop_adam_spec(
    document: ExactP0OpenVsClosedLoopAdamConfig,
    mode: RunMode,
) -> ExactP0OpenVsClosedLoopAdamTrainSpec:
    if mode is not RunMode.TRAIN:
        raise config_error("mode", "to equal 'train'", mode.value)
    student = resolve_student_spec(document.student, mode)
    if not isinstance(student, StudentTrainSpec):  # pragma: no cover
        raise RuntimeError("Expected parent student training spec.")
    if student.runtime.device != "cuda":  # pragma: no cover
        raise config_error("student.runtime.device", "to equal 'cuda'", student.runtime.device)
    return ExactP0OpenVsClosedLoopAdamTrainSpec(
        experiment_id=EXPERIMENT_ID,
        protocol=document.protocol,
        student=student,
    )


__all__ = [
    "EXPERIMENT_ID",
    "EXPECTED_DEVICE_MODEL_SHA256",
    "EXPECTED_P0_SHA256",
    "EXPECTED_PUBLISHED_POPULATION_FINGERPRINT",
    "EXPECTED_PUBLISHED_POPULATION_SHA256",
    "EXPECTED_TEACHER_WEIGHTS_SHA256",
    "ExactP0OpenVsClosedLoopAdamConfig",
    "ExactP0OpenVsClosedLoopAdamProtocol",
    "ExactP0OpenVsClosedLoopAdamTrainSpec",
    "FIXED_LOGIT_GAIN",
    "MAXIMUM_RANDOM_DRAWS",
    "NOMINAL_DELTA_X",
    "P0_ENDPOINT_SEED",
    "RECOVERY_ARM_IDS",
    "RECOVERY_DATA_ORDER_SEED",
    "RECOVERY_EPOCHS",
    "RECOVERY_LEARNING_RATE_RAW_X",
    "RECOVERY_PULSE_CAP",
    "RECOVERY_PULSE_SELECTION_SEED",
    "TARGET_ASSIGNMENT_SEED",
    "VERIFY_TOLERANCE_RAW_X",
    "parse_exact_p0_open_vs_closed_loop_adam_config",
    "resolve_exact_p0_open_vs_closed_loop_adam_spec",
]
