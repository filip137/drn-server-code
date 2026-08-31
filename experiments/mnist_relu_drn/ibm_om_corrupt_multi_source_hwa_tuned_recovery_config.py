"""Strict config for multi-source corrupt-array HWA and tuned recovery.

The four HWA arms change only source-assignment exposure and the output-layer
learning-rate multiplier.  A joint arm/epoch winner is selected on a disjoint
development array before the sealed target array is sampled.
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


EXPERIMENT_ID = "mnist_ibm_om_corrupt_multi_source_hwa_tuned_recovery.v1"
SCHEMA_VERSION = 1
EXPECTED_TEACHER_WEIGHTS_PATH = "data/mnist_relu_teacher_fixed_init_20260816.pt"
EXPECTED_TEACHER_WEIGHTS_SHA256 = (
    "9a961a77628e365b54fdf59304ec7fddf46f1f4834c4c03cecea9c204f837f52"
)

TRAIN_ASSIGNMENT_SEEDS = (94001, 94002)
TRAIN_ENDPOINT_SEEDS = (
    (94101, 94102, 94103, 94104),
    (94111, 94112, 94113, 94114),
)
TRAIN_ENDPOINT_SEEDS_BY_ASSIGNMENT = tuple(
    zip(TRAIN_ASSIGNMENT_SEEDS, TRAIN_ENDPOINT_SEEDS, strict=True)
)
DEVELOPMENT_ASSIGNMENT_SEED = 94003
DEVELOPMENT_ENDPOINT_SEEDS = (94201, 94202, 94203, 94204)
TARGET_ASSIGNMENT_SEED = 94004
TARGET_ENDPOINT_SEEDS = (94301, 94302, 94303, 94304, 94305)
P0_ENDPOINT_SEED = 94301
RECOVERY_PULSE_SELECTION_SEED = 94401
RECOVERY_DATA_ORDER_SEED = 94402

PUBLISHED_CORRUPT_PROBABILITY = 0.1348
NOMINAL_DW_MIN_RAW_A = 0.0949
LEVEL_SPACING_RAW_X = NOMINAL_DW_MIN_RAW_A / 2.0
VERIFY_TOLERANCE_RAW_X = LEVEL_SPACING_RAW_X / 2.0
MAXIMUM_PROGRAM_PULSES = 128
RECOVERY_PULSE_CAP = 64
MAXIMUM_RANDOM_DRAWS = 1 + 2 * (MAXIMUM_PROGRAM_PULSES + RECOVERY_PULSE_CAP)
HWA_EPOCHS = 6
HWA_W1_LEARNING_RATE = 0.004411914893617021
HWA_BASE_W2_LEARNING_RATE = 0.000011974808510638298
BASE_HWA_LEARNING_RATES = (HWA_W1_LEARNING_RATE, HWA_BASE_W2_LEARNING_RATE)
HWA_W2_MULTIPLIERS = (1.0, 30.0)
TARGET_RECOVERY_EPOCHS = 3
RECOVERY_LEARNING_RATE_RAW_X = 3.0e-5
FIXED_LOGIT_GAIN = 14.12537544622754

HWA_ARM_IDS = (
    "single_w2_x1",
    "single_w2_x30",
    "multi_w2_x1",
    "multi_w2_x30",
)
ARM_SIMPLICITY_ORDER = HWA_ARM_IDS
TRAINING_ASSIGNMENT_SCHEDULE = (
    "global_minibatch_ordinal_modulo_source_count_start_94001"
)
PERSISTENT_ENDPOINT_SCHEDULE = "cyclic_adjacent_pair_by_assignment_visit_ordinal"


@dataclass(frozen=True)
class SourceArraysContract:
    training_assignment_seeds: tuple[int, int]
    training_endpoint_seeds: tuple[
        tuple[int, int, int, int], tuple[int, int, int, int]
    ]
    training_endpoint_seeds_by_assignment: tuple[
        tuple[int, tuple[int, int, int, int]],
        tuple[int, tuple[int, int, int, int]],
    ]
    development_assignment_seed: int
    development_endpoint_seeds: tuple[int, int, int, int]
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
class HwaArmContract:
    arm_id: str
    source_mode: str
    training_assignment_seeds: tuple[int, ...]
    assignment_schedule: str
    w1_learning_rate: float
    base_w2_learning_rate: float
    w2_learning_rate_factor: int
    w2_learning_rate_multiplier: float
    w2_learning_rate: float


@dataclass(frozen=True)
class HwaTrainingContract:
    epochs: int
    arms: tuple[HwaArmContract, ...]
    optimizer: str
    base_learning_rates: tuple[float, float]
    w1_learning_rate: float
    base_w2_learning_rate: float
    momentum: float
    weight_decay: float
    endpoint_objective: str
    ideal_gradient_weight: float
    persistent_gradient_weights: tuple[float, float]
    persistent_sample_schedule: str
    assignment_schedule: str
    maximum_program_pulses: int
    verify_tolerance_raw_x: float


@dataclass(frozen=True)
class HwaSelectionContract:
    candidate_unit: str
    development_assignment_seed: int
    development_endpoint_seeds: tuple[int, int, int, int]
    epoch_zero_eligible: bool
    metric_order: tuple[str, ...]
    arm_simplicity_order: tuple[str, ...]
    target_progression: str


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
    objective: str
    output_kl_only: bool
    updated_layer_indices: tuple[int, ...]
    frozen_layer_indices: tuple[int, ...]
    start_state: str
    learning_rate_raw_x: float
    beta1: float
    beta2: float
    epsilon: float
    epochs: int
    maximum_batches: int | None
    pulse_cap: int
    maximum_random_draws: int
    pulse_selection_seed: int
    data_order_seed: int
    update_surface: str
    conceptual_column_phases_per_minibatch: tuple[int, int]
    vectorization_equivalence: str
    verify_reads_during_updates: int
    validation_used_for_selection: bool
    checkpoint_policy: str
    state_authority: str


@dataclass(frozen=True)
class ExecutionContract:
    evidence_tier: str
    evidence_class: str
    device: str
    source_initialization: str
    source_development_target_separation: str
    external_aihwkit_environment_variable: str


@dataclass(frozen=True)
class CorruptMultiSourceHwaTunedRecoveryProtocol:
    expected_teacher_weights_path: str
    expected_teacher_weights_sha256: str
    source: SourceArraysContract
    mapping: MappingContract
    training: HwaTrainingContract
    selection: HwaSelectionContract
    target: TargetArrayContract
    deployment: DeploymentContract
    recovery: RecoveryContract
    execution: ExecutionContract


@dataclass(frozen=True)
class CorruptMultiSourceHwaTunedRecoveryConfig:
    schema_version: int
    experiment_id: str
    protocol: CorruptMultiSourceHwaTunedRecoveryProtocol
    student: StudentConfig


@dataclass(frozen=True)
class CorruptMultiSourceHwaTunedRecoveryTrainSpec:
    experiment_id: str
    protocol: CorruptMultiSourceHwaTunedRecoveryProtocol
    student: StudentTrainSpec


def _arm(
    arm_id: str,
    *,
    source_mode: str,
    assignment_seeds: tuple[int, ...],
    w2_multiplier: float,
) -> HwaArmContract:
    return HwaArmContract(
        arm_id=arm_id,
        source_mode=source_mode,
        training_assignment_seeds=assignment_seeds,
        assignment_schedule=TRAINING_ASSIGNMENT_SCHEDULE,
        w1_learning_rate=HWA_W1_LEARNING_RATE,
        base_w2_learning_rate=HWA_BASE_W2_LEARNING_RATE,
        w2_learning_rate_factor=int(w2_multiplier),
        w2_learning_rate_multiplier=w2_multiplier,
        w2_learning_rate=HWA_BASE_W2_LEARNING_RATE * w2_multiplier,
    )


HWA_ARMS = (
    _arm(
        "single_w2_x1",
        source_mode="single_source_94001",
        assignment_seeds=(94001,),
        w2_multiplier=1.0,
    ),
    _arm(
        "single_w2_x30",
        source_mode="single_source_94001",
        assignment_seeds=(94001,),
        w2_multiplier=30.0,
    ),
    _arm(
        "multi_w2_x1",
        source_mode="alternating_sources_94001_94002",
        assignment_seeds=TRAIN_ASSIGNMENT_SEEDS,
        w2_multiplier=1.0,
    ),
    _arm(
        "multi_w2_x30",
        source_mode="alternating_sources_94001_94002",
        assignment_seeds=TRAIN_ASSIGNMENT_SEEDS,
        w2_multiplier=30.0,
    ),
)


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
                "learning_rates": list(BASE_HWA_LEARNING_RATES),
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


def _protocol() -> CorruptMultiSourceHwaTunedRecoveryProtocol:
    return CorruptMultiSourceHwaTunedRecoveryProtocol(
        expected_teacher_weights_path=EXPECTED_TEACHER_WEIGHTS_PATH,
        expected_teacher_weights_sha256=EXPECTED_TEACHER_WEIGHTS_SHA256,
        source=SourceArraysContract(
            training_assignment_seeds=TRAIN_ASSIGNMENT_SEEDS,
            training_endpoint_seeds=TRAIN_ENDPOINT_SEEDS,
            training_endpoint_seeds_by_assignment=(
                (94001, TRAIN_ENDPOINT_SEEDS[0]),
                (94002, TRAIN_ENDPOINT_SEEDS[1]),
            ),
            development_assignment_seed=DEVELOPMENT_ASSIGNMENT_SEED,
            development_endpoint_seeds=DEVELOPMENT_ENDPOINT_SEEDS,
            published_corrupt_probability=PUBLISHED_CORRUPT_PROBABILITY,
            physical_population_policy="published_native_corrupt_devices_retained",
            mapping_population_policy="paired_counterfactual_repaired_template_only",
            paired_identity_policy=(
                "same_base_draw_repair_only_published_corrupt_coordinates"
            ),
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
            arms=HWA_ARMS,
            optimizer="sgd",
            base_learning_rates=BASE_HWA_LEARNING_RATES,
            w1_learning_rate=HWA_W1_LEARNING_RATE,
            base_w2_learning_rate=HWA_BASE_W2_LEARNING_RATE,
            momentum=0.0,
            weight_decay=0.0,
            endpoint_objective="mean2",
            ideal_gradient_weight=0.25,
            persistent_gradient_weights=(0.375, 0.375),
            persistent_sample_schedule=PERSISTENT_ENDPOINT_SCHEDULE,
            assignment_schedule=TRAINING_ASSIGNMENT_SCHEDULE,
            maximum_program_pulses=MAXIMUM_PROGRAM_PULSES,
            verify_tolerance_raw_x=VERIFY_TOLERANCE_RAW_X,
        ),
        selection=HwaSelectionContract(
            candidate_unit="joint_arm_and_trained_epoch",
            development_assignment_seed=DEVELOPMENT_ASSIGNMENT_SEED,
            development_endpoint_seeds=DEVELOPMENT_ENDPOINT_SEEDS,
            epoch_zero_eligible=False,
            metric_order=(
                "persistent_mean_accuracy_descending",
                "persistent_minimum_accuracy_descending",
                "persistent_mean_kl_teacher_student_ascending",
                "epoch_ascending",
                "arm_simplicity_order_ascending",
            ),
            arm_simplicity_order=ARM_SIMPLICITY_ORDER,
            target_progression=(
                "only_joint_winner_plus_common_epoch0_control_to_sealed_target"
            ),
        ),
        target=TargetArrayContract(
            assignment_seed=TARGET_ASSIGNMENT_SEED,
            endpoint_seeds=TARGET_ENDPOINT_SEEDS,
            fine_tune_endpoint_seed=P0_ENDPOINT_SEED,
            role="sealed_untouched_until_joint_development_winner_frozen",
            physical_population_policy="published_native_corrupt_devices_retained",
            mapping_population_policy=(
                "target_paired_counterfactual_repaired_template_only"
            ),
        ),
        deployment=DeploymentContract(
            controller="one_pulse_program_and_verify",
            maximum_program_pulses=MAXIMUM_PROGRAM_PULSES,
            verify_tolerance_raw_x=VERIFY_TOLERANCE_RAW_X,
            maximum_random_draws=MAXIMUM_RANDOM_DRAWS,
            target_mapping_policy=(
                "logical_master_remapped_through_target_repaired_template_then_"
                "programmed_on_target_published_plant"
            ),
            target_clipping=False,
            persistent_endpoint_applied_to_drn=True,
            apparent_endpoint_applied_to_drn=False,
            inference_read_noise=False,
        ),
        recovery=RecoveryContract(
            optimizer="digital_Adam",
            objective="teacher_output_kl_only",
            output_kl_only=True,
            updated_layer_indices=(0, 1),
            frozen_layer_indices=(),
            start_state="selected_hwa_target_seed_94301_P0",
            learning_rate_raw_x=RECOVERY_LEARNING_RATE_RAW_X,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
            epochs=TARGET_RECOVERY_EPOCHS,
            maximum_batches=None,
            pulse_cap=RECOVERY_PULSE_CAP,
            maximum_random_draws=MAXIMUM_RANDOM_DRAWS,
            pulse_selection_seed=RECOVERY_PULSE_SELECTION_SEED,
            data_order_seed=RECOVERY_DATA_ORDER_SEED,
            update_surface="column_serial_open_loop_stochastic_coincidence_emulator",
            conceptual_column_phases_per_minibatch=(100, 20),
            vectorization_equivalence=(
                "disjoint_columns_vectorized_with_independent_per_cell_plant_RNG"
            ),
            verify_reads_during_updates=0,
            validation_used_for_selection=True,
            checkpoint_policy=(
                "target_validation_accuracy_then_KL_then_earlier_epoch_"
                "among_epochs_1_to_3_before_test"
            ),
            state_authority="persistent_target_raw_active_plant_only_after_P0",
        ),
        execution=ExecutionContract(
            evidence_tier="exploratory_noncanonical",
            evidence_class="model_based_aihwkit_1.1.0_IBM_OM_fitted_preset",
            device="cuda",
            source_initialization=(
                "frozen_ReLU_weights_to_normalized_DRN_logical_master"
            ),
            source_development_target_separation=(
                "train_94001_94002_then_select_94003_then_open_sealed_94004"
            ),
            external_aihwkit_environment_variable="EBL_AIHWKIT_PYTHON",
        ),
    )


def parse_corrupt_multi_source_hwa_tuned_recovery_config(
    payload: Mapping[str, Any],
) -> CorruptMultiSourceHwaTunedRecoveryConfig:
    raw = _object(payload, "config")
    expected = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "training_assignment_seeds": list(TRAIN_ASSIGNMENT_SEEDS),
        "training_endpoint_seeds": [list(value) for value in TRAIN_ENDPOINT_SEEDS],
        "development_assignment_seed": DEVELOPMENT_ASSIGNMENT_SEED,
        "development_endpoint_seeds": list(DEVELOPMENT_ENDPOINT_SEEDS),
        "target_assignment_seed": TARGET_ASSIGNMENT_SEED,
        "target_endpoint_seeds": list(TARGET_ENDPOINT_SEEDS),
        "p0_endpoint_seed": P0_ENDPOINT_SEED,
        "recovery_pulse_selection_seed": RECOVERY_PULSE_SELECTION_SEED,
        "recovery_data_order_seed": RECOVERY_DATA_ORDER_SEED,
        "arms": list(HWA_ARM_IDS),
    }
    _keys(raw, "config", set(expected))
    for name, value in expected.items():
        if type(raw[name]) is not type(value) or raw[name] != value:
            raise config_error(f"config.{name}", f"to equal {value!r}", raw[name])
    return CorruptMultiSourceHwaTunedRecoveryConfig(
        schema_version=SCHEMA_VERSION,
        experiment_id=EXPERIMENT_ID,
        protocol=_protocol(),
        student=parse_student_config(_student_payload()),
    )


def resolve_corrupt_multi_source_hwa_tuned_recovery_spec(
    document: CorruptMultiSourceHwaTunedRecoveryConfig,
    mode: RunMode,
) -> CorruptMultiSourceHwaTunedRecoveryTrainSpec:
    if mode is not RunMode.TRAIN:
        raise config_error("mode", "to equal 'train'", mode.value)
    student = resolve_student_spec(document.student, mode)
    if not isinstance(student, StudentTrainSpec):  # pragma: no cover
        raise RuntimeError("Expected parent student training spec.")
    if student.runtime.device != "cuda":  # pragma: no cover
        raise config_error("student.runtime.device", "to equal 'cuda'", student.runtime.device)
    return CorruptMultiSourceHwaTunedRecoveryTrainSpec(
        experiment_id=EXPERIMENT_ID,
        protocol=document.protocol,
        student=student,
    )


__all__ = [
    "ARM_SIMPLICITY_ORDER",
    "BASE_HWA_LEARNING_RATES",
    "DEVELOPMENT_ASSIGNMENT_SEED",
    "DEVELOPMENT_ENDPOINT_SEEDS",
    "EXPERIMENT_ID",
    "EXPECTED_TEACHER_WEIGHTS_PATH",
    "EXPECTED_TEACHER_WEIGHTS_SHA256",
    "FIXED_LOGIT_GAIN",
    "HWA_ARMS",
    "HWA_ARM_IDS",
    "HWA_BASE_W2_LEARNING_RATE",
    "HWA_EPOCHS",
    "HWA_W1_LEARNING_RATE",
    "HWA_W2_MULTIPLIERS",
    "LEVEL_SPACING_RAW_X",
    "MAXIMUM_PROGRAM_PULSES",
    "MAXIMUM_RANDOM_DRAWS",
    "P0_ENDPOINT_SEED",
    "PERSISTENT_ENDPOINT_SCHEDULE",
    "RECOVERY_LEARNING_RATE_RAW_X",
    "RECOVERY_DATA_ORDER_SEED",
    "RECOVERY_PULSE_CAP",
    "RECOVERY_PULSE_SELECTION_SEED",
    "SCHEMA_VERSION",
    "TARGET_ASSIGNMENT_SEED",
    "TARGET_ENDPOINT_SEEDS",
    "TARGET_RECOVERY_EPOCHS",
    "TRAIN_ASSIGNMENT_SEEDS",
    "TRAIN_ENDPOINT_SEEDS",
    "TRAIN_ENDPOINT_SEEDS_BY_ASSIGNMENT",
    "TRAINING_ASSIGNMENT_SCHEDULE",
    "VERIFY_TOLERANCE_RAW_X",
    "CorruptMultiSourceHwaTunedRecoveryConfig",
    "CorruptMultiSourceHwaTunedRecoveryProtocol",
    "CorruptMultiSourceHwaTunedRecoveryTrainSpec",
    "HwaArmContract",
    "parse_corrupt_multi_source_hwa_tuned_recovery_config",
    "resolve_corrupt_multi_source_hwa_tuned_recovery_spec",
]
