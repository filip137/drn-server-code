"""Strict config for STAR-inspired hidden-state BPTT fault recovery.

This exploratory family deliberately isolates permanent AIHWKit-compatible
corrupt-device faults from program-and-verify error.  It is a BPTT objective
study, not an implementation of the paper's local centered-EP STAR rule.
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


EXPERIMENT_ID = "mnist_ibm_om_star_inspired_hidden_kd_fault_recovery.v1"
SCHEMA_VERSION = 1
TARGET_ASSIGNMENT_SEED = 87004
DEVELOPMENT_FAULT_SEED = 91501
EVALUATION_FAULT_SEEDS = (91502, 91503, 91504, 91505)
FAULT_PROBABILITY = 0.1348
CORRUPT_DEVICES_RANGE_RAW_A = 0.01
EXPECTED_TEACHER_WEIGHTS_PATH = "data/mnist_relu_teacher_fixed_init_20260816.pt"
EXPECTED_TEACHER_WEIGHTS_SHA256 = (
    "9a961a77628e365b54fdf59304ec7fddf46f1f4834c4c03cecea9c204f837f52"
)
SOURCE_RESULT_ID = (
    "mnist-ibm-om-winsorized-cross-array-open-loop-adam-"
    "exploratory-20260829-v1"
)
SOURCE_RUN_RELATIVE_PATH = (
    "runs/fresh_87004/20260829T121606.793011Z-8adf8460-a5c62c4d"
)
SOURCE_SUMMARY_SHA256 = (
    "4771e2ac2ffd57baed6d6088e335023ab9778883a0d9e5d965a85d25d2c3b323"
)
POPULATION_SHA256 = (
    "d30274ad576bb24cd2c2d17d01bd6c3cd1bb99a908ee3990f8c1d6e12865e741"
)
POPULATION_RECEIPT_SHA256 = (
    "98ede52f7c8a8cda09dd9346d2ca9e3195fd024492f99295f1765aa9d562b76c"
)
POPULATION_FINGERPRINT = (
    "214a4c48c81d4cd1edecef356aae7fb564ab60541cdb4de34caf246dfd0026b8"
)
SOURCE_HARDWARE_INSTANCE_ID = (
    "fabbba85777d05cd41f6e08caccd597bc9a32fd0510938a86f3b3c9f51ec5fdb"
)
MAPPING_SHA256 = (
    "58765a20190c030d15fe8a81daf420305ead4a95786c4fabb8eb46a2a7b84e2f"
)
MAPPING_RECEIPT_SHA256 = (
    "5219245f090100f1a91587a33ea786f2710c48ad042f87e2fb31eb2b53256615"
)
MAPPING_TENSOR_SHA256 = (
    "f569845cedd5c5004e7aa7c528276eea527a4abc193d8db981da5c97acb99e84",
    "935ed06b05ae1f4363a8bf95096e69f4c5f09133110214b71eea149dc6ebde33",
)
HEALTHY_VALIDATION_CORRECT = 4776
HEALTHY_VALIDATION_PREDICTION_SHA256 = (
    "8519adc9cbb75c7c7293972c00f466de9887957e3470976ae89f2a2464e4aa15"
)
HEALTHY_TEST_CORRECT = 9599
HEALTHY_TEST_PREDICTION_SHA256 = (
    "98ffa9a3142d0aa46db381111d6d00316bcc9784d62c095adb94aa31fc46644c"
)
FIXED_LOGIT_GAIN = 14.12537544622754
NOMINAL_DELTA_X = 0.04745
ARMS = (
    "output_kl_only",
    "relu_sample_hidden_kl",
    "healthy_class_hidden_kl",
    "healthy_class_hidden_mse",
)
LAMBDA_MULTIPLIERS = (0.3, 1.0, 3.0)
OUTPUT_ONLY_DIAGNOSTIC_LRS = (1.5e-5, 3.0e-5, 6.0e-5)


@dataclass(frozen=True)
class StarArtifactContract:
    result_id: str
    run_relative_path: str
    source_summary_sha256: str
    population_sha256: str
    population_receipt_sha256: str
    population_fingerprint: str
    source_hardware_instance_id: str
    mapping_sha256: str
    mapping_receipt_sha256: str
    mapping_tensor_sha256: tuple[str, str]
    teacher_path: str
    teacher_sha256: str
    healthy_validation_correct: int
    healthy_validation_prediction_sha256: str
    healthy_test_correct: int
    healthy_test_prediction_sha256: str


@dataclass(frozen=True)
class StarFaultContract:
    policy: str
    probability: float
    corrupt_devices_range_raw_a: float
    development_seed: int
    evaluation_seeds: tuple[int, ...]
    independent_per_crosspoint: bool
    stuck_value_distribution: str
    pulse_directions_disabled: bool
    mask_exposed_to_learner: bool


@dataclass(frozen=True)
class StarObjectiveContract:
    arms: tuple[str, ...]
    routing: str
    relu_hidden_temperature_policy: str
    class_hidden_temperature: float
    class_prototype_source: str
    class_prototype_examples: int
    class_prototype_accumulation_dtype: str
    lambda_calibration_examples: int
    lambda_multipliers: tuple[float, ...]
    selection_rule: str


@dataclass(frozen=True)
class StarTrainingContract:
    optimizer: str
    learning_rate_raw_x: float
    output_only_diagnostic_learning_rates_raw_x: tuple[float, ...]
    beta1: float
    beta2: float
    epsilon: float
    batch_size: int
    epochs: int
    pulse_cap: int
    nominal_delta_x: float
    continuous_screen_examples: int
    pulse_count_mismatch_fraction: float
    continuous_rung: bool
    open_loop_pulse_rung: bool


@dataclass(frozen=True)
class StarExecutionContract:
    evidence_tier: str
    claim_label: str
    device: str
    fixed_logit_gain: float
    raw_device_coordinate: str
    circuit_conductance_coordinate: str
    negative_conductance_allowed: bool
    program_verify: bool
    verify_reads: int
    inference_read_noise: bool
    cycle_to_cycle_noise: bool
    apparent_write_noise: bool


@dataclass(frozen=True)
class StarInspiredHiddenKdFaultRecoveryProtocol:
    target_assignment_seed: int
    artifacts: StarArtifactContract
    faults: StarFaultContract
    objectives: StarObjectiveContract
    training: StarTrainingContract
    execution: StarExecutionContract


@dataclass(frozen=True)
class StarInspiredHiddenKdFaultRecoveryConfig:
    schema_version: int
    experiment_id: str
    protocol: StarInspiredHiddenKdFaultRecoveryProtocol
    student: StudentConfig


@dataclass(frozen=True)
class StarInspiredHiddenKdFaultRecoveryTrainSpec:
    experiment_id: str
    protocol: StarInspiredHiddenKdFaultRecoveryProtocol
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


def _protocol() -> StarInspiredHiddenKdFaultRecoveryProtocol:
    return StarInspiredHiddenKdFaultRecoveryProtocol(
        target_assignment_seed=TARGET_ASSIGNMENT_SEED,
        artifacts=StarArtifactContract(
            result_id=SOURCE_RESULT_ID,
            run_relative_path=SOURCE_RUN_RELATIVE_PATH,
            source_summary_sha256=SOURCE_SUMMARY_SHA256,
            population_sha256=POPULATION_SHA256,
            population_receipt_sha256=POPULATION_RECEIPT_SHA256,
            population_fingerprint=POPULATION_FINGERPRINT,
            source_hardware_instance_id=SOURCE_HARDWARE_INSTANCE_ID,
            mapping_sha256=MAPPING_SHA256,
            mapping_receipt_sha256=MAPPING_RECEIPT_SHA256,
            mapping_tensor_sha256=MAPPING_TENSOR_SHA256,
            teacher_path=EXPECTED_TEACHER_WEIGHTS_PATH,
            teacher_sha256=EXPECTED_TEACHER_WEIGHTS_SHA256,
            healthy_validation_correct=HEALTHY_VALIDATION_CORRECT,
            healthy_validation_prediction_sha256=(
                HEALTHY_VALIDATION_PREDICTION_SHA256
            ),
            healthy_test_correct=HEALTHY_TEST_CORRECT,
            healthy_test_prediction_sha256=HEALTHY_TEST_PREDICTION_SHA256,
        ),
        faults=StarFaultContract(
            policy="aihwkit_corrupt_device_compatible",
            probability=FAULT_PROBABILITY,
            corrupt_devices_range_raw_a=CORRUPT_DEVICES_RANGE_RAW_A,
            development_seed=DEVELOPMENT_FAULT_SEED,
            evaluation_seeds=EVALUATION_FAULT_SEEDS,
            independent_per_crosspoint=True,
            stuck_value_distribution="uniform_support_intersection",
            pulse_directions_disabled=True,
            mask_exposed_to_learner=False,
        ),
        objectives=StarObjectiveContract(
            arms=ARMS,
            routing="full_summed_loss_bptt_through_both_physical_layers",
            relu_hidden_temperature_policy="healthy_teacher_hidden_rms",
            class_hidden_temperature=1.0,
            class_prototype_source="healthy_drn_train_split_class_mean_raw_state",
            class_prototype_examples=55_000,
            class_prototype_accumulation_dtype="float64",
            lambda_calibration_examples=1024,
            lambda_multipliers=LAMBDA_MULTIPLIERS,
            selection_rule=(
                "validation_correct_then_output_kl_then_multiplier_distance_to_1_"
                "then_smaller_multiplier"
            ),
        ),
        training=StarTrainingContract(
            optimizer="adam",
            learning_rate_raw_x=3.0e-5,
            output_only_diagnostic_learning_rates_raw_x=OUTPUT_ONLY_DIAGNOSTIC_LRS,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
            batch_size=16,
            epochs=1,
            pulse_cap=64,
            nominal_delta_x=NOMINAL_DELTA_X,
            continuous_screen_examples=1024,
            pulse_count_mismatch_fraction=0.10,
            continuous_rung=True,
            open_loop_pulse_rung=True,
        ),
        execution=StarExecutionContract(
            evidence_tier="exploratory_noncanonical",
            claim_label="STAR-inspired hidden-state BPTT fault recovery",
            device="cuda",
            fixed_logit_gain=FIXED_LOGIT_GAIN,
            raw_device_coordinate="a_in_minus1_plus1",
            circuit_conductance_coordinate="G_equals_a_plus_1_equals_2x",
            negative_conductance_allowed=False,
            program_verify=False,
            verify_reads=0,
            inference_read_noise=False,
            cycle_to_cycle_noise=False,
            apparent_write_noise=False,
        ),
    )


def parse_star_inspired_hidden_kd_fault_recovery_config(
    payload: Mapping[str, Any],
) -> StarInspiredHiddenKdFaultRecoveryConfig:
    raw = _object(payload, "config")
    _keys(
        raw,
        "config",
        {
            "schema_version",
            "experiment_id",
            "target_assignment_seed",
            "development_fault_seed",
            "evaluation_fault_seeds",
        },
    )
    expected = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "target_assignment_seed": TARGET_ASSIGNMENT_SEED,
        "development_fault_seed": DEVELOPMENT_FAULT_SEED,
        "evaluation_fault_seeds": list(EVALUATION_FAULT_SEEDS),
    }
    for key, value in expected.items():
        if raw.get(key) != value:
            raise config_error(f"config.{key}", f"to equal {value!r}", raw.get(key))
    return StarInspiredHiddenKdFaultRecoveryConfig(
        schema_version=SCHEMA_VERSION,
        experiment_id=EXPERIMENT_ID,
        protocol=_protocol(),
        student=parse_student_config(_student_payload()),
    )


def resolve_star_inspired_hidden_kd_fault_recovery_spec(
    config: StarInspiredHiddenKdFaultRecoveryConfig,
    mode: RunMode,
) -> StarInspiredHiddenKdFaultRecoveryTrainSpec:
    if mode != RunMode.TRAIN:
        raise config_error("mode", "to equal 'train'", mode.value)
    student = resolve_student_spec(config.student, RunMode.TRAIN)
    if student.runtime.device != "cuda":
        raise config_error("student.runtime.device", "to equal 'cuda'", student.runtime.device)
    return StarInspiredHiddenKdFaultRecoveryTrainSpec(
        experiment_id=config.experiment_id,
        protocol=config.protocol,
        student=student,
    )


__all__ = [
    "EXPERIMENT_ID",
    "SCHEMA_VERSION",
    "StarInspiredHiddenKdFaultRecoveryConfig",
    "StarInspiredHiddenKdFaultRecoveryTrainSpec",
    "parse_star_inspired_hidden_kd_fault_recovery_config",
    "resolve_star_inspired_hidden_kd_fault_recovery_spec",
]
