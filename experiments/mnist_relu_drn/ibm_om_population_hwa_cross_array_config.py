"""Strict config for positive-G population HWA followed by A--D writes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from experiments.mnist_relu.config import _keys, _object
from experiments.mnist_relu_drn.config import (
    StudentConfig,
    StudentTrainSpec,
    parse_student_config,
    resolve_student_spec,
)
from experiments.schema import RunMode, config_error


EXPERIMENT_ID = "mnist_ibm_om_population_hwa_cross_array.v1"
SCHEMA_VERSION = 1
EXPECTED_TEACHER_WEIGHTS_PATH = "data/mnist_relu_teacher_fixed_init_20260816.pt"
EXPECTED_TEACHER_WEIGHTS_SHA256 = (
    "9a961a77628e365b54fdf59304ec7fddf46f1f4834c4c03cecea9c204f837f52"
)
HWA_CONDITION_KEY = "adaptive__lower_to_target__tau_step_0.5"
HWA_ESTIMATOR_KEY = "lower_to_target__tau_step_0.5"
HWA_SEED = 88042
HWA_EPOCHS = 10
# Keep the most recent matched DRN HWA layer rates fixed so that the new
# intervention is the population programming-error policy rather than an
# optimizer retune.
HWA_LEARNING_RATES = (0.004411914893617021, 0.000011974808510638298)
FIXED_LOGIT_GAIN = 14.12537544622754
NOMINAL_DELTA_X = 0.04745
MAXIMUM_PROGRAM_PULSES = 128
MAXIMUM_RANDOM_DRAWS = 1 + 2 * MAXIMUM_PROGRAM_PULSES
VERIFY_TOLERANCE_RAW_X = NOMINAL_DELTA_X / 2.0
ARRAY_ASSIGNMENT_SEEDS = {
    "A": 87004,
    "B": 87005,
    "C": 87006,
    "D": 87007,
}
ARRAY_ENDPOINT_SEEDS = {
    "A": (89402, 89403, 89404, 89405),
    "B": (89502, 89503, 89504, 89505),
    "C": (89602, 89603, 89604, 89605),
    "D": (89702, 89703, 89704, 89705),
}
PROFILES = ("exploratory_full", "smoke")


@dataclass(frozen=True)
class PopulationHwaArm:
    arm_id: str
    quantization_spacing_raw_x: float | None
    role: str


@dataclass(frozen=True)
class ArrayDeploymentContract:
    label: str
    assignment_seed: int
    endpoint_seeds: tuple[int, ...]
    opening_phase: str


@dataclass(frozen=True)
class PopulationHwaCrossArrayProtocol:
    profile: str
    evidence_tier: str
    expected_teacher_weights_path: str
    expected_teacher_weights_sha256: str
    initialization: Mapping[str, object]
    hwa: Mapping[str, object]
    hwa_arms: tuple[PopulationHwaArm, ...]
    arrays: tuple[ArrayDeploymentContract, ...]
    deployment: Mapping[str, object]
    execution: Mapping[str, object]


@dataclass(frozen=True)
class PopulationHwaCrossArrayConfig:
    schema_version: int
    experiment_id: str
    protocol: PopulationHwaCrossArrayProtocol
    student: StudentConfig


@dataclass(frozen=True)
class PopulationHwaCrossArrayTrainSpec:
    experiment_id: str
    protocol: PopulationHwaCrossArrayProtocol
    student: StudentTrainSpec


def _student_payload(*, smoke: bool) -> dict[str, object]:
    return {
        "schema_version": 1,
        "experiment_id": "mnist_relu_drn_kd.v1",
        "runtime": {
            "seed": 42,
            "data_seed": 42,
            "device": "cuda",
            "dtype": "float32",
        },
        "data": {
            "batch_size": 16,
            "validation_points": 64 if smoke else 5000,
            "num_points": 64 if smoke else None,
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
            "calibration_examples": 64 if smoke else 1024,
            "calibration_batch_size": 16 if smoke else 128,
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
                "max_batches": 1 if smoke else None,
                "max_validation_batches": 1 if smoke else None,
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


def _protocol(profile: str) -> PopulationHwaCrossArrayProtocol:
    smoke = profile == "smoke"
    arrays = tuple(
        ArrayDeploymentContract(
            label=label,
            assignment_seed=ARRAY_ASSIGNMENT_SEEDS[label],
            endpoint_seeds=(
                ARRAY_ENDPOINT_SEEDS[label][:1]
                if smoke
                else ARRAY_ENDPOINT_SEEDS[label]
            ),
            opening_phase=(
                "before_hwa_raw_relu_baseline_only"
                if label == "A"
                else "after_both_hwa_checkpoints_are_frozen"
            ),
        )
        for label in ("A", "B", "C", "D")
    )
    return PopulationHwaCrossArrayProtocol(
        profile=profile,
        evidence_tier="exploratory_noncanonical",
        expected_teacher_weights_path=EXPECTED_TEACHER_WEIGHTS_PATH,
        expected_teacher_weights_sha256=EXPECTED_TEACHER_WEIGHTS_SHA256,
        initialization={
            "logical_master": "raw_ReLU_weight_divided_by_layer_absmax_fp32",
            "logical_bounds": [-1.0, 1.0],
            "global_physical_mapping": (
                "G=2*(relu(u),relu(-u),relu(-u),relu(u))"
            ),
            "global_physical_bounds": [0.0, 2.0],
            "array_A_role": (
                "pre_HWA_deployment_baseline_only_not_HWA_state_or_calibration"
            ),
        },
        hwa={
            "epochs": HWA_EPOCHS,
            "optimizer": "Adam",
            "learning_rates": list(HWA_LEARNING_RATES),
            "learning_rate_source": (
                "frozen_matched_DRN_HWA_rates_from_the_preceding_cross_array_ladder"
            ),
            "betas": [0.9, 0.999],
            "epsilon": 1.0e-8,
            "objective": "teacher_kl",
            "condition_key": HWA_CONDITION_KEY,
            "estimator_key": HWA_ESTIMATOR_KEY,
            "programming_error_seed": HWA_SEED,
            "strength_schedule": "linear_one_indexed_epoch_ramp",
            "initial_strength": 0.0,
            "final_strength": 1.0,
            "ramp_epochs": HWA_EPOCHS,
            "sample_scope": "one_fresh_draw_per_physical_cell_per_minibatch",
            "same_draw_forward_backward": True,
            "fixed_array_identity": False,
            "fixed_per_cell_bounds": False,
            "fixed_reference": False,
            "corrupt_mask": False,
            "persistent_training_state": False,
            "checkpoint_policy": "fixed_final_epoch_10_no_selection",
        },
        hwa_arms=(
            PopulationHwaArm(
                arm_id="population_hwa_continuous",
                quantization_spacing_raw_x=None,
                role="programming_error_HWA_without_quantization",
            ),
            PopulationHwaArm(
                arm_id="population_hwa_one_delta_qat",
                quantization_spacing_raw_x=NOMINAL_DELTA_X,
                role="separate_global_one_delta_quantization_control",
            ),
        ),
        arrays=arrays,
        deployment={
            "mapping": "array_owned_shared_destination_alpha_0_one_delta",
            "mapping_population": "paired_counterfactual_repaired",
            "physical_populations": ["counterfactual_repaired", "published_corrupt"],
            "controller": "adaptive_apparent_verify",
            "maximum_program_pulses": MAXIMUM_PROGRAM_PULSES,
            "maximum_random_draws": MAXIMUM_RANDOM_DRAWS,
            "verify_tolerance_raw_x": VERIFY_TOLERANCE_RAW_X,
            "inference_state": "persistent_full_G_equals_2x",
            "apparent_endpoint_applied_to_drn": False,
            "target_clipping": False,
            "inference_read_noise": False,
            "level_spacing_raw_x": NOMINAL_DELTA_X,
            "fixed_logit_gain": FIXED_LOGIT_GAIN,
        },
        execution={
            "profile": "direct_local_cuda",
            "maximum_training_batches_per_epoch": 1 if smoke else None,
            "evaluation_sample_limit": 64 if smoke else None,
            "array_opening_order": [
                "write_raw_ReLU_on_A",
                "population_HWA_without_array_identity",
                "freeze_HWA_checkpoints",
                "sample_and_write_B_C_D",
            ],
        },
    )


def parse_population_hwa_cross_array_config(
    payload: Mapping[str, object],
) -> PopulationHwaCrossArrayConfig:
    raw = _object(payload, "config")
    _keys(raw, "config", {"schema_version", "experiment_id", "profile"})
    if raw["schema_version"] != SCHEMA_VERSION:
        raise config_error("config.schema_version", "to equal 1", raw["schema_version"])
    if raw["experiment_id"] != EXPERIMENT_ID:
        raise config_error(
            "config.experiment_id", f"to equal {EXPERIMENT_ID!r}", raw["experiment_id"]
        )
    profile = raw["profile"]
    if profile not in PROFILES:
        raise config_error("config.profile", f"to be one of {PROFILES!r}", profile)
    assert isinstance(profile, str)
    return PopulationHwaCrossArrayConfig(
        schema_version=SCHEMA_VERSION,
        experiment_id=EXPERIMENT_ID,
        protocol=_protocol(profile),
        student=parse_student_config(_student_payload(smoke=profile == "smoke")),
    )


def resolve_population_hwa_cross_array_spec(
    document: PopulationHwaCrossArrayConfig,
    mode: RunMode,
) -> PopulationHwaCrossArrayTrainSpec:
    if mode is not RunMode.TRAIN:
        raise config_error("mode", "to equal 'train'", mode.value)
    student = resolve_student_spec(document.student, mode)
    if not isinstance(student, StudentTrainSpec):
        raise RuntimeError("Population-HWA student did not resolve to train mode.")
    return PopulationHwaCrossArrayTrainSpec(
        experiment_id=document.experiment_id,
        protocol=document.protocol,
        student=student,
    )


__all__ = [
    "ARRAY_ASSIGNMENT_SEEDS",
    "ARRAY_ENDPOINT_SEEDS",
    "EXPERIMENT_ID",
    "HWA_CONDITION_KEY",
    "HWA_ESTIMATOR_KEY",
    "PopulationHwaCrossArrayConfig",
    "PopulationHwaCrossArrayProtocol",
    "PopulationHwaCrossArrayTrainSpec",
    "SCHEMA_VERSION",
    "parse_population_hwa_cross_array_config",
    "resolve_population_hwa_cross_array_spec",
]
