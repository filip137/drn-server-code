"""Strict config for the corrupt-source IBM OM deployment ladder."""

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


EXPERIMENT_ID = "mnist_ibm_om_corrupt_source_hwa_cross_array_pulse_adam.v1"
SCHEMA_VERSION = 1
EXPECTED_TEACHER_WEIGHTS_PATH = "data/mnist_relu_teacher_fixed_init_20260816.pt"
EXPECTED_TEACHER_WEIGHTS_SHA256 = (
    "9a961a77628e365b54fdf59304ec7fddf46f1f4834c4c03cecea9c204f837f52"
)

SOURCE_ASSIGNMENT_SEED = 93001
SOURCE_TRAIN_ENDPOINT_SEEDS = (93201, 93202, 93203, 93204)
SOURCE_SELECTION_ENDPOINT_SEEDS = (93221, 93222, 93223, 93224)
TARGET_ASSIGNMENT_SEED = 93002
TARGET_ENDPOINT_SEEDS = (93301, 93302, 93303, 93304, 93305)
P0_ENDPOINT_SEED = 93301
PULSE_SELECTION_SEED = 93401

PUBLISHED_CORRUPT_PROBABILITY = 0.1348
NOMINAL_DW_MIN_RAW_A = 0.0949
LEVEL_SPACING_RAW_X = NOMINAL_DW_MIN_RAW_A / 2.0
VERIFY_TOLERANCE_RAW_X = LEVEL_SPACING_RAW_X / 2.0
MAXIMUM_PROGRAM_PULSES = 128
RECOVERY_PULSE_CAP = 64
MAXIMUM_RANDOM_DRAWS = 1 + 2 * (MAXIMUM_PROGRAM_PULSES + RECOVERY_PULSE_CAP)
HWA_EPOCHS = 3
HWA_LEARNING_RATES = (0.004411914893617021, 0.000011974808510638298)
RECOVERY_LEARNING_RATE_RAW_X = 3.0e-5
FIXED_LOGIT_GAIN = 14.12537544622754


@dataclass(frozen=True)
class SourceArrayContract:
    assignment_seed: int
    train_endpoint_seeds: tuple[int, int, int, int]
    selection_endpoint_seeds: tuple[int, int, int, int]
    published_corrupt_probability: float
    physical_population_policy: str
    mapping_population_policy: str
    paired_identity_policy: str


@dataclass(frozen=True)
class MappingContract:
    native_coordinate: str
    winsorization: str
    level_spacing_raw_x: float
    spacing_delta_x_multiplier: int
    baseline_policy: str
    conductance_formula: str
    negative_conductance_allowed: bool
    fixed_logit_gain: float


@dataclass(frozen=True)
class HwaTrainingContract:
    epochs: int
    optimizer: str
    learning_rates: tuple[float, float]
    momentum: float
    weight_decay: float
    endpoint_objective: str
    ideal_gradient_weight: float
    persistent_gradient_weights: tuple[float, float]
    persistent_sample_schedule: str
    maximum_program_pulses: int
    verify_tolerance_raw_x: float
    checkpoint_selection: str


@dataclass(frozen=True)
class TargetArrayContract:
    assignment_seed: int
    endpoint_seeds: tuple[int, int, int, int, int]
    fine_tune_endpoint_seed: int
    role: str
    physical_population_policy: str
    mapping_population_policy: str


@dataclass(frozen=True)
class DeploymentContract:
    controller: str
    maximum_program_pulses: int
    verify_tolerance_raw_x: float
    maximum_random_draws: int
    target_mapping_policy: str
    target_clipping: bool
    persistent_endpoint_applied_to_drn: bool
    apparent_endpoint_applied_to_drn: bool
    inference_read_noise: bool


@dataclass(frozen=True)
class RecoveryContract:
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
    state_authority: str


@dataclass(frozen=True)
class ExecutionContract:
    evidence_tier: str
    evidence_class: str
    device: str
    source_initialization: str
    source_target_separation: str
    external_aihwkit_environment_variable: str


@dataclass(frozen=True)
class CorruptSourceHwaCrossArrayPulseAdamProtocol:
    expected_teacher_weights_path: str
    expected_teacher_weights_sha256: str
    source: SourceArrayContract
    mapping: MappingContract
    training: HwaTrainingContract
    target: TargetArrayContract
    deployment: DeploymentContract
    recovery: RecoveryContract
    execution: ExecutionContract


@dataclass(frozen=True)
class CorruptSourceHwaCrossArrayPulseAdamConfig:
    schema_version: int
    experiment_id: str
    protocol: CorruptSourceHwaCrossArrayPulseAdamProtocol
    student: StudentConfig


@dataclass(frozen=True)
class CorruptSourceHwaCrossArrayPulseAdamTrainSpec:
    experiment_id: str
    protocol: CorruptSourceHwaCrossArrayPulseAdamProtocol
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
                "num_epochs": HWA_EPOCHS,
                "learning_rates": list(HWA_LEARNING_RATES),
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


def _protocol() -> CorruptSourceHwaCrossArrayPulseAdamProtocol:
    return CorruptSourceHwaCrossArrayPulseAdamProtocol(
        expected_teacher_weights_path=EXPECTED_TEACHER_WEIGHTS_PATH,
        expected_teacher_weights_sha256=EXPECTED_TEACHER_WEIGHTS_SHA256,
        source=SourceArrayContract(
            assignment_seed=SOURCE_ASSIGNMENT_SEED,
            train_endpoint_seeds=SOURCE_TRAIN_ENDPOINT_SEEDS,
            selection_endpoint_seeds=SOURCE_SELECTION_ENDPOINT_SEEDS,
            published_corrupt_probability=PUBLISHED_CORRUPT_PROBABILITY,
            physical_population_policy="published_native_corrupt_devices_retained",
            mapping_population_policy="paired_counterfactual_repaired_template_only",
            paired_identity_policy="same_base_draw_repair_only_published_corrupt_coordinates",
        ),
        mapping=MappingContract(
            native_coordinate="AIHWKit_raw_a_winsorized_to_[-1,1]",
            winsorization=(
                "healthy_bounds_intersect_nominal_support_corrupt_singletons_unchanged"
            ),
            level_spacing_raw_x=LEVEL_SPACING_RAW_X,
            spacing_delta_x_multiplier=1,
            baseline_policy="alpha_zero_shared_destination_repaired_RESET_max",
            conductance_formula="G=raw_a_winsorized+1=2*x",
            negative_conductance_allowed=False,
            fixed_logit_gain=FIXED_LOGIT_GAIN,
        ),
        training=HwaTrainingContract(
            epochs=HWA_EPOCHS,
            optimizer="sgd",
            learning_rates=HWA_LEARNING_RATES,
            momentum=0.0,
            weight_decay=0.0,
            endpoint_objective="mean2",
            ideal_gradient_weight=0.25,
            persistent_gradient_weights=(0.375, 0.375),
            persistent_sample_schedule="cyclic_adjacent_pair_by_global_minibatch",
            maximum_program_pulses=MAXIMUM_PROGRAM_PULSES,
            verify_tolerance_raw_x=VERIFY_TOLERANCE_RAW_X,
            checkpoint_selection=(
                "source_selection_endpoint_mean_accuracy_then_minimum_then_KL_then_earlier_epoch"
            ),
        ),
        target=TargetArrayContract(
            assignment_seed=TARGET_ASSIGNMENT_SEED,
            endpoint_seeds=TARGET_ENDPOINT_SEEDS,
            fine_tune_endpoint_seed=P0_ENDPOINT_SEED,
            role="fresh_untouched_until_source_HWA_checkpoint_frozen",
            physical_population_policy="published_native_corrupt_devices_retained",
            mapping_population_policy="target_paired_counterfactual_repaired_template_only",
        ),
        deployment=DeploymentContract(
            controller="one_pulse_program_and_verify",
            maximum_program_pulses=MAXIMUM_PROGRAM_PULSES,
            verify_tolerance_raw_x=VERIFY_TOLERANCE_RAW_X,
            maximum_random_draws=MAXIMUM_RANDOM_DRAWS,
            target_mapping_policy=(
                "logical_HWA_master_remapped_through_target_repaired_template_then_"
                "programmed_on_target_published_plant"
            ),
            target_clipping=False,
            persistent_endpoint_applied_to_drn=True,
            apparent_endpoint_applied_to_drn=False,
            inference_read_noise=False,
        ),
        recovery=RecoveryContract(
            optimizer="digital_Adam",
            learning_rate_raw_x=RECOVERY_LEARNING_RATE_RAW_X,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
            epochs=1,
            maximum_batches=None,
            pulse_cap=RECOVERY_PULSE_CAP,
            pulse_selection_seed=PULSE_SELECTION_SEED,
            update_surface="column_serial_open_loop_stochastic_coincidence_emulator",
            conceptual_column_phases_per_minibatch=(100, 20),
            vectorization_equivalence=(
                "disjoint_columns_vectorized_with_independent_per_cell_plant_RNG"
            ),
            verify_reads_during_updates=0,
            state_authority="persistent_target_raw_active_plant_only_after_P0",
        ),
        execution=ExecutionContract(
            evidence_tier="exploratory_noncanonical",
            evidence_class="model_based_aihwkit_1.1.0_IBM_OM_fitted_preset",
            device="cuda",
            source_initialization="frozen_ReLU_weights_to_normalized_DRN_logical_master",
            source_target_separation="assignment_93002_never_used_for_HWA_or_selection",
            external_aihwkit_environment_variable="EBL_AIHWKIT_PYTHON",
        ),
    )


def parse_corrupt_source_hwa_cross_array_pulse_adam_config(
    payload: Mapping[str, Any],
) -> CorruptSourceHwaCrossArrayPulseAdamConfig:
    raw = _object(payload, "config")
    expected = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "source_assignment_seed": SOURCE_ASSIGNMENT_SEED,
        "source_train_endpoint_seeds": list(SOURCE_TRAIN_ENDPOINT_SEEDS),
        "source_selection_endpoint_seeds": list(SOURCE_SELECTION_ENDPOINT_SEEDS),
        "target_assignment_seed": TARGET_ASSIGNMENT_SEED,
        "target_endpoint_seeds": list(TARGET_ENDPOINT_SEEDS),
        "p0_endpoint_seed": P0_ENDPOINT_SEED,
    }
    _keys(raw, "config", set(expected))
    for name, value in expected.items():
        if type(raw[name]) is not type(value) or raw[name] != value:
            raise config_error(f"config.{name}", f"to equal {value!r}", raw[name])
    return CorruptSourceHwaCrossArrayPulseAdamConfig(
        schema_version=SCHEMA_VERSION,
        experiment_id=EXPERIMENT_ID,
        protocol=_protocol(),
        student=parse_student_config(_student_payload()),
    )


def resolve_corrupt_source_hwa_cross_array_pulse_adam_spec(
    document: CorruptSourceHwaCrossArrayPulseAdamConfig,
    mode: RunMode,
) -> CorruptSourceHwaCrossArrayPulseAdamTrainSpec:
    if mode is not RunMode.TRAIN:
        raise config_error("mode", "to equal 'train'", mode.value)
    student = resolve_student_spec(document.student, mode)
    if not isinstance(student, StudentTrainSpec):  # pragma: no cover
        raise RuntimeError("Expected parent student training spec.")
    if student.runtime.device != "cuda":  # pragma: no cover
        raise config_error("student.runtime.device", "to equal 'cuda'", student.runtime.device)
    return CorruptSourceHwaCrossArrayPulseAdamTrainSpec(
        experiment_id=EXPERIMENT_ID,
        protocol=document.protocol,
        student=student,
    )


__all__ = [
    "EXPERIMENT_ID",
    "EXPECTED_TEACHER_WEIGHTS_PATH",
    "EXPECTED_TEACHER_WEIGHTS_SHA256",
    "HWA_EPOCHS",
    "HWA_LEARNING_RATES",
    "LEVEL_SPACING_RAW_X",
    "MAXIMUM_PROGRAM_PULSES",
    "MAXIMUM_RANDOM_DRAWS",
    "P0_ENDPOINT_SEED",
    "PULSE_SELECTION_SEED",
    "RECOVERY_PULSE_CAP",
    "SCHEMA_VERSION",
    "SOURCE_ASSIGNMENT_SEED",
    "SOURCE_SELECTION_ENDPOINT_SEEDS",
    "SOURCE_TRAIN_ENDPOINT_SEEDS",
    "TARGET_ASSIGNMENT_SEED",
    "TARGET_ENDPOINT_SEEDS",
    "VERIFY_TOLERANCE_RAW_X",
    "CorruptSourceHwaCrossArrayPulseAdamConfig",
    "CorruptSourceHwaCrossArrayPulseAdamProtocol",
    "CorruptSourceHwaCrossArrayPulseAdamTrainSpec",
    "parse_corrupt_source_hwa_cross_array_pulse_adam_config",
    "resolve_corrupt_source_hwa_cross_array_pulse_adam_spec",
]
