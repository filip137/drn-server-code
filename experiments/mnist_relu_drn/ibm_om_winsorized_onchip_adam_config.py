"""Strict config for the target-87003 persistent pulse-Adam screen."""

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


EXPERIMENT_ID = "mnist_ibm_om_winsorized_onchip_adam.v1"
SCHEMA_VERSION = 1
SPACING_DELTA_X_MULTIPLIERS = (1, 2, 4)
TARGET_ASSIGNMENT_SEED = 87003
P0_ENDPOINT_SEED = 89301
EXPECTED_TEACHER_WEIGHTS_PATH = "data/mnist_relu_teacher_fixed_init_20260816.pt"
EXPECTED_TEACHER_WEIGHTS_SHA256 = (
    "9a961a77628e365b54fdf59304ec7fddf46f1f4834c4c03cecea9c204f837f52"
)
SOURCE_EXPLORATORY_RESULT_ID = (
    "mnist-ibm-om-baseline-spacing-pv-winsorized-nominal-"
    "exploratory-20260828-v2"
)
TARGET_HARDWARE_INSTANCE_ID = (
    "3be072fb1dcad07d54176b8e0fd30e8f376306853f1955729f0a00254c5475d9"
)
TARGET_POPULATION_FINGERPRINT = (
    "60f932c0bea38c43dc71f934627542b647eabebedd9163a1f3612c954cfc94cc"
)
TARGET_POPULATION_SHA256 = (
    "d864ee09a87c73d78082cefc5e4c00c82a482cbd5426db675a5592b2e83b90e2"
)
TARGET_COMMISSIONING_SHA256 = (
    "0b0793ffc7d57391570349324a1573197001f3b542eab6509e0b61c0abe7c9c7"
)
TARGET_JOINT_ASSIGNMENT_SHA256 = (
    "278088be9193fc1f93e4a08cc4476a9a2540a360024540aa1936d402e1930c99"
)
TARGET_MAPPING_SHA256_BY_SPACING = {
    1: "c0feb5ec9c515557b9a6f89515cd5ce0dfef936ce9df2db471e96f922e2536de",
    2: "f128e452d59464d1c946e609be89f9220b0bc6a1565b8599c84ef39a4add7f75",
    4: "af8159c8f6f6f3f5793457d15cf81337a2db6ec927dfc60efe0e9fd3921f2de9",
}
FIXED_LOGIT_GAIN = 14.12537544622754
NOMINAL_DELTA_X = 0.04745
MAXIMUM_PROGRAM_PULSES = 128
RECOVERY_PULSE_CAP = 64
MAXIMUM_RANDOM_DRAWS = 1 + 2 * (MAXIMUM_PROGRAM_PULSES + RECOVERY_PULSE_CAP)
LEARNING_RATES_RAW_X = (3.0e-5, 1.0e-4, 3.0e-4)
PULSE_SELECTION_SEED = 90403
SCREEN_MAX_BATCHES = None


@dataclass(frozen=True)
class OnchipAdamArtifactContract:
    source_exploratory_result_id: str
    target_hardware_instance_id: str
    target_population_fingerprint: str
    target_population_sha256: str
    target_commissioning_sha256: str
    target_joint_assignment_sha256: str
    target_mapping_sha256: str
    qat_training_assignment_seeds: tuple[int, int]
    qat_training_population_fingerprints: tuple[str, str]
    qat_development_assignment_seed: int
    qat_development_population_fingerprint: str
    qat_assignment_cycle: str
    qat_expected_global_minibatch_ordinal: int


@dataclass(frozen=True)
class OnchipAdamP0Contract:
    assignment_seed: int
    endpoint_seed: int
    maximum_program_pulses: int
    verify_tolerance_raw_x: float
    recovery_pulse_cap: int
    maximum_random_draws: int
    no_remap_after_p0: bool
    apparent_endpoint_applied_to_drn: bool


@dataclass(frozen=True)
class OnchipAdamRecoveryContract:
    techniques: tuple[str, str]
    learning_rates_raw_x: tuple[float, float, float]
    beta1: float
    beta2: float
    epsilon: float
    pulse_translation: str
    pulse_selection_seed: int
    epochs: int
    maximum_batches: int | None
    selection_split: str
    selection_rule: str


@dataclass(frozen=True)
class OnchipAdamExecutionContract:
    evidence_tier: str
    claim_label: str
    state_authority: str
    device: str
    inference_state: str
    controller_state: str


@dataclass(frozen=True)
class WinsorizedOnchipAdamProtocol:
    spacing_delta_x_multiplier: int
    fixed_logit_gain: float
    expected_teacher_weights_path: str
    expected_teacher_weights_sha256: str
    artifacts: OnchipAdamArtifactContract
    p0: OnchipAdamP0Contract
    recovery: OnchipAdamRecoveryContract
    execution: OnchipAdamExecutionContract


@dataclass(frozen=True)
class WinsorizedOnchipAdamConfig:
    schema_version: int
    experiment_id: str
    protocol: WinsorizedOnchipAdamProtocol
    student: StudentConfig


@dataclass(frozen=True)
class WinsorizedOnchipAdamTrainSpec:
    experiment_id: str
    protocol: WinsorizedOnchipAdamProtocol
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
                "max_batches": SCREEN_MAX_BATCHES,
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


def parse_winsorized_onchip_adam_config(
    payload: Mapping[str, Any],
) -> WinsorizedOnchipAdamConfig:
    raw = _object(payload, "config")
    _keys(
        raw,
        "config",
        {
            "schema_version",
            "experiment_id",
            "spacing_delta_x_multiplier",
            "target_assignment_seed",
            "p0_endpoint_seed",
        },
    )
    if raw["schema_version"] != SCHEMA_VERSION:
        raise config_error("config.schema_version", "to equal 1", raw["schema_version"])
    if raw["experiment_id"] != EXPERIMENT_ID:
        raise config_error(
            "config.experiment_id", f"to equal {EXPERIMENT_ID!r}", raw["experiment_id"]
        )
    spacing = raw["spacing_delta_x_multiplier"]
    if type(spacing) is not int or spacing not in SPACING_DELTA_X_MULTIPLIERS:
        raise config_error(
            "config.spacing_delta_x_multiplier",
            f"to be one of {SPACING_DELTA_X_MULTIPLIERS!r}",
            spacing,
        )
    if raw["target_assignment_seed"] != TARGET_ASSIGNMENT_SEED:
        raise config_error(
            "config.target_assignment_seed", "to equal 87003", raw["target_assignment_seed"]
        )
    if raw["p0_endpoint_seed"] != P0_ENDPOINT_SEED:
        raise config_error(
            "config.p0_endpoint_seed", "to equal 89301", raw["p0_endpoint_seed"]
        )
    student = parse_student_config(_student_payload())
    protocol = WinsorizedOnchipAdamProtocol(
        spacing_delta_x_multiplier=spacing,
        fixed_logit_gain=FIXED_LOGIT_GAIN,
        expected_teacher_weights_path=EXPECTED_TEACHER_WEIGHTS_PATH,
        expected_teacher_weights_sha256=EXPECTED_TEACHER_WEIGHTS_SHA256,
        artifacts=OnchipAdamArtifactContract(
            source_exploratory_result_id=SOURCE_EXPLORATORY_RESULT_ID,
            target_hardware_instance_id=TARGET_HARDWARE_INSTANCE_ID,
            target_population_fingerprint=TARGET_POPULATION_FINGERPRINT,
            target_population_sha256=TARGET_POPULATION_SHA256,
            target_commissioning_sha256=TARGET_COMMISSIONING_SHA256,
            target_joint_assignment_sha256=TARGET_JOINT_ASSIGNMENT_SHA256,
            target_mapping_sha256=TARGET_MAPPING_SHA256_BY_SPACING[spacing],
            qat_training_assignment_seeds=(86001, 87001),
            qat_training_population_fingerprints=(
                "7c5ef35d1d6ff77341be162c44372f0e81ada7dbf070567905fdb05b331e06a4",
                "36fbad76021c7d93bd0a888981b6c062eea938f3839a2d50b1112b181eedcb7d",
            ),
            qat_development_assignment_seed=87002,
            qat_development_population_fingerprint=(
                "59633236c00b2e7d0e169d94802e2828353bc41357e578f1d38f0c22011eef68"
            ),
            qat_assignment_cycle="global_minibatch_ordinal_modulo_two_start_86001",
            qat_expected_global_minibatch_ordinal=34380,
        ),
        p0=OnchipAdamP0Contract(
            assignment_seed=TARGET_ASSIGNMENT_SEED,
            endpoint_seed=P0_ENDPOINT_SEED,
            maximum_program_pulses=MAXIMUM_PROGRAM_PULSES,
            verify_tolerance_raw_x=0.5 * NOMINAL_DELTA_X,
            recovery_pulse_cap=RECOVERY_PULSE_CAP,
            maximum_random_draws=MAXIMUM_RANDOM_DRAWS,
            no_remap_after_p0=True,
            apparent_endpoint_applied_to_drn=False,
        ),
        recovery=OnchipAdamRecoveryContract(
            techniques=("direct_physical_rail", "contrast_constrained"),
            learning_rates_raw_x=LEARNING_RATES_RAW_X,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
            pulse_translation=(
                "one_bernoulli_pulse_per_cell_per_minibatch_with_"
                "p=min(abs(delta_x)/nominal_delta_x,1)"
            ),
            pulse_selection_seed=PULSE_SELECTION_SEED,
            epochs=1,
            maximum_batches=SCREEN_MAX_BATCHES,
            selection_split="validation_5000_only",
            selection_rule="maximum_student_correct_then_minimum_teacher_KL",
        ),
        execution=OnchipAdamExecutionContract(
            evidence_tier="exploratory_noncanonical",
            claim_label="hardware-in-loop pulse-mediated Adam",
            state_authority="persistent_raw_active_plant_only",
            device="cuda",
            inference_state="persistent_full_G_equals_2x",
            controller_state="digital_Adam_moments_and_pulse_selection_RNG_only",
        ),
    )
    return WinsorizedOnchipAdamConfig(
        schema_version=SCHEMA_VERSION,
        experiment_id=EXPERIMENT_ID,
        protocol=protocol,
        student=student,
    )


def resolve_winsorized_onchip_adam_spec(
    document: WinsorizedOnchipAdamConfig, mode: RunMode
) -> WinsorizedOnchipAdamTrainSpec:
    if mode is not RunMode.TRAIN:
        raise config_error(
            "the requested run mode",
            "to be 'train' for this train-only experiment",
            mode.value,
        )
    student = resolve_student_spec(document.student, mode)
    if not isinstance(student, StudentTrainSpec):  # pragma: no cover
        raise RuntimeError("Expected the parent parser to resolve training.")
    return WinsorizedOnchipAdamTrainSpec(
        experiment_id=EXPERIMENT_ID,
        protocol=document.protocol,
        student=student,
    )


__all__ = [
    "EXPERIMENT_ID",
    "LEARNING_RATES_RAW_X",
    "MAXIMUM_RANDOM_DRAWS",
    "NOMINAL_DELTA_X",
    "P0_ENDPOINT_SEED",
    "SCHEMA_VERSION",
    "SPACING_DELTA_X_MULTIPLIERS",
    "TARGET_ASSIGNMENT_SEED",
    "WinsorizedOnchipAdamConfig",
    "WinsorizedOnchipAdamProtocol",
    "WinsorizedOnchipAdamTrainSpec",
    "parse_winsorized_onchip_adam_config",
    "resolve_winsorized_onchip_adam_spec",
]
