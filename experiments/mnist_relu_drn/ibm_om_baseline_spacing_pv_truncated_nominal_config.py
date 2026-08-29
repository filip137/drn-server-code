"""Strict config for the exploratory nominal-bound-winsorized IBM OM screen."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from experiments.mnist_relu.config import _keys, _object
from experiments.mnist_relu_drn.config import (
    EXPERIMENT_ID as STUDENT_EXPERIMENT_ID,
    StudentConfig,
    StudentValidateSpec,
    parse_student_config,
    resolve_student_spec,
)
from experiments.mnist_relu_drn.ibm_om_baseline_spacing_pv_no_clip_config import (
    AssignmentContract,
    CalibrationContract,
    CommissioningContract,
    ExecutionContract,
    ExclusionContract,
    JointAssignmentRepairContract,
    MetricContract,
    OmDeviceContract,
    PhysicalInvariantContract,
    PredecessorContract,
    ProgramVerifyContract,
    SourceContract,
    _same_json_value,
)
from experiments.schema import RunMode, config_error


EXPERIMENT_ID = "mnist_ibm_om_baseline_spacing_pv_truncated_nominal.v1"
SCHEMA_VERSION = 1
EXPECTED_WEIGHTS_SHA256 = (
    "9a961a77628e365b54fdf59304ec7fddf46f1f4834c4c03cecea9c204f837f52"
)
DEVELOPMENT_ASSIGNMENT_SEED = 86001
HELDOUT_ASSIGNMENT_SEEDS = (87001, 87002, 87003)
ENDPOINT_SEEDS_BY_ASSIGNMENT = {
    87001: (89101, 89102, 89103, 89104, 89105),
    87002: (89201, 89202, 89203, 89204, 89205),
    87003: (89301, 89302, 89303, 89304, 89305),
}
BASELINE_POSITION_FRACTIONS = (0.0, 0.25, 0.5)
SPACING_DELTA_X_MULTIPLIERS = (1, 2, 4)
SCALE_FRACTIONS = (0.125, 0.25, 0.5, 1.0)
RESET_READ_SAMPLES = 8
MAXIMUM_DONOR_CANDIDATES_PER_QUAD = 128
MAXIMUM_PROGRAM_PULSES = 128
P_AND_V_REPEAT_COUNT = 5
VERIFY_TOLERANCE_DELTA_X_RATIO = 0.5


@dataclass(frozen=True)
class TruncationContract:
    policy: str
    operation_order: str
    raw_a_minimum: float
    raw_a_maximum: float
    raw_x_minimum: float
    raw_x_maximum: float
    conductance_formula: str
    conductance_minimum: float
    conductance_maximum: float
    empty_intersection: str
    endpoint_policy: str


@dataclass(frozen=True)
class BaselineSpacingPvTruncatedNominalProtocol:
    metrics: MetricContract
    execution: ExecutionContract
    device: OmDeviceContract
    source: SourceContract
    predecessor: PredecessorContract
    calibration: CalibrationContract
    assignments: AssignmentContract
    truncation: TruncationContract
    baseline_policy: str
    baseline_position_fraction: float
    group_lower_definition: str
    group_upper_definition: str
    baseline_definition: str
    spacing_delta_x_multiplier: int
    delta_x_definition: str
    level_index_policy: str
    level_capacity_policy: str
    logical_rounding: str
    commissioning_read_samples: int
    commissioning: CommissioningContract
    joint_assignment_repair: JointAssignmentRepairContract
    program_verify: ProgramVerifyContract
    physical_invariants: PhysicalInvariantContract
    exclusions: ExclusionContract


@dataclass(frozen=True)
class BaselineSpacingPvTruncatedNominalConfig:
    schema_version: int
    experiment_id: str
    protocol: BaselineSpacingPvTruncatedNominalProtocol
    student: StudentConfig


@dataclass(frozen=True)
class BaselineSpacingPvTruncatedNominalValidateSpec:
    experiment_id: str
    protocol: BaselineSpacingPvTruncatedNominalProtocol
    student: StudentValidateSpec


_METRICS = {
    "ideal_quantized": "ibm_om.ideal_truncated_nominal_uniform_spacing_init.v1",
    "pv_persistent": "ibm_om.pv_persistent_truncated_nominal_uniform_spacing_init.v1",
    "continuous_diagnostic": "ibm_om.ideal_truncated_nominal_continuous_init.v1",
}
_EXECUTION = {
    "profile": "exploratory_cuda",
    "evidence_tier": "exploratory_noncanonical",
    "workflow": "direct_cuda_no_study_launcher_or_smoke",
}
_DEVICE = {
    "evidence_class": "model_based_aihwkit_preset_bound_winsorization_control",
    "preset": "reram_array_om",
    "required_aihwkit_version": "1.1.0",
    "corruption_policy": "counterfactual_repaired",
    "native_coordinate": "raw_a_with_hard_support_truncated_to_[-1,1]",
}
_SOURCE = {
    "weights_role": "frozen_relu_source_and_teacher",
    "expected_weights_sha256": EXPECTED_WEIGHTS_SHA256,
}
_PREDECESSOR = {
    "experiment_id": "mnist_ibm_om_baseline_spacing_pv_no_clip.v1",
    "mapping_status": "global_extreme_affine_shift_diagnostic",
    "assignment_reconstruction": "same_frozen_joint_identity_then_winsorize_bounds",
    "development_hardware_instance_id": (
        "feddd5a62ae3ace700dc46f155f54236c58fa769c0d4d6603613ed857b7a0150"
    ),
    "heldout_hardware_instance_ids": [
        "58fb1079e7bd60f53ba93ba61a3dca887352846b9f5a5768e9cd7bb72aed8492",
        "215d46a5ec563abdf2c9450625dea0d6620280dec8a18a413e6199adc72efb50",
        "3be072fb1dcad07d54176b8e0fd30e8f376306853f1955729f0a00254c5475d9",
    ],
}
_CALIBRATION = {
    "development_assignment_seed": DEVELOPMENT_ASSIGNMENT_SEED,
    "scale_fractions": list(SCALE_FRACTIONS),
    "selection_domain": "continuous_truncated_nominal_development_assignment_86001",
    "selection_metric": (
        "continuous_accuracy_then_calibrated_kl_then_lexicographic_scale_pair"
    ),
    "logit_gain": "positive_kl_fit_on_selected_continuous_mapping",
    "sharing": "one_per_alpha_frozen_across_spacing_heldout_and_endpoint_repeats",
    "heldout_application": (
        "freeze_alpha_specific_scale_pair_and_gain_before_quantization_or_pv"
    ),
}
_TRUNCATION = {
    "policy": "nominal_bound_winsorization_no_rejection_or_resampling",
    "operation_order": "sample_frozen_identity_then_winsorize_bounds_then_commission_then_pv",
    "raw_a_minimum": -1.0,
    "raw_a_maximum": 1.0,
    "raw_x_minimum": 0.0,
    "raw_x_maximum": 1.0,
    "conductance_formula": "G=a_truncated+1=2*x",
    "conductance_minimum": 0.0,
    "conductance_maximum": 2.0,
    "empty_intersection": "invalidate_identity",
    "endpoint_policy": "persistent_endpoint_direct_no_post_handoff_clipping",
}
_COMMISSIONING = {
    "samples_per_cell": RESET_READ_SAMPLES,
    "sample_sequence": (
        "after_bound_winsorization_initialize_at_winsorized_lower_bound_then_"
        "one_reset_pulse_and_apparent_read_per_sample"
    ),
    "read_coordinate": "raw_active_a_then_map_to_x=(a+1)/2",
    "baseline_estimator": "per_cell_arithmetic_mean",
    "bound_policy": "bound_mean_to_own_winsorized_support",
    "noise_policy": "preset_cycle_to_cycle_and_apparent_write_noise_enabled",
    "freeze_policy": (
        "deterministic_per_winsorized_identity_reused_across_alpha_spacing_and_repeats"
    ),
}
_JOINT_REPAIR = {
    "scope": "reuse_complete_quad_joint_identity_from_predecessor_before_winsorization",
    "donor_seed_derivation": (
        "derive_seed(assignment_seed,mnist_ibm_om_baseline_selection_joint_donor_v1)"
    ),
    "destination_traversal": "canonical_layer_then_logical_quad_order",
    "donor_traversal": "same_layer_unique_donor_quads_in_canonical_order",
    "eligibility": "reuse_frozen_predecessor_joint_identity_assignment_exactly",
    "identity_copy": "copy_complete_four_cell_quad_then_winsorize_each_cell_bounds",
    "maximum_candidates_per_quad": MAXIMUM_DONOR_CANDIDATES_PER_QUAD,
    "exhaustion": "invalidate_assignment",
}
_PROGRAM_VERIFY = {
    "controller": "one_pulse",
    "start_protocol": "conditioned_winsorized_lower_bound_to_target",
    "tolerance_delta_x_ratio": VERIFY_TOLERANCE_DELTA_X_RATIO,
    "maximum_program_pulses": MAXIMUM_PROGRAM_PULSES,
    "verify_coordinate": "x=(a_truncated+1)/2",
    "persistent_coordinate": "x=(a_truncated+1)/2",
    "cycle_to_cycle_noise": True,
    "apparent_verify_noise": True,
    "inference_read_noise": False,
    "endpoint_for_accuracy": "persistent",
    "circuit_handoff": "G=2*x_persistent_without_projection",
    "apparent_endpoint_role": "controller_and_diagnostic_only_never_applied_to_drn",
    "repeat_count": P_AND_V_REPEAT_COUNT,
}
_PHYSICAL_INVARIANTS = {
    "conductance_representation": "single_device_G=a_truncated+1=B+n*(2*h_x)",
    "destination_column_groups": [["G++", "G-+"], ["G+-", "G--"]],
    "exact_zero_requirement": "C(B+,B-,B+,B-)=0",
    "transfer": "full_signed_four_conductance_contrast_no_reference_subtraction",
    "loading": "full_conductance_sum",
    "support": "sampled_bounds_winsorized_to_raw_a_[-1,1]_before_pulsing",
    "out_of_bounds": "invalidate_no_endpoint_projection_or_clipping",
}
_EXCLUSIONS = {
    "optimizer_updates": 0,
    "hardware_aware_training": False,
    "inference_read_noise": 0.0,
    "retention_drift": 0.0,
    "logical_rewrites": 0,
    "post_deployment_recovery": False,
}

_STUDENT_RUNTIME = {"seed": 42, "data_seed": 42, "device": "cuda", "dtype": "float32"}
_STUDENT_DATA = {
    "batch_size": 16,
    "validation_points": 5000,
    "num_points": None,
    "shuffle": True,
}
_STUDENT_MODEL = {
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
}
_STUDENT_SOLVER = {
    "inference_iterations": 4,
    "training_iterations": 4,
    "mode": "asynchronous",
    "overrelaxation_factor": 1.1,
}
_STUDENT_MAPPING = {
    "scale_fractions": list(SCALE_FRACTIONS),
    "scale_fraction_pairs": None,
    "range_placement": "lower",
    "calibration_examples": 1024,
    "calibration_batch_size": 128,
    "logit_gain_min": 0.001,
    "logit_gain_max": 1000.0,
    "logit_gain_steps": 121,
}
_STUDENT_TEACHER = {"type": "bias_free_relu", "initialization": "signed_weight_mapping"}


def _require_exact(raw: Mapping[str, Any], expected: Mapping[str, Any], *, path: str) -> None:
    _keys(raw, path, set(expected))
    for name, expected_value in expected.items():
        if not _same_json_value(raw[name], expected_value):
            raise config_error(f"{path}.{name}", f"to equal {expected_value!r}", raw[name])


def _parse_protocol(value: Any) -> BaselineSpacingPvTruncatedNominalProtocol:
    path = "config.baseline_spacing_pv_truncated_nominal"
    raw = _object(value, path)
    fields = {
        "metrics", "execution", "device", "source", "predecessor", "calibration",
        "assignments", "truncation", "baseline_policy", "baseline_position_fraction",
        "group_lower_definition", "group_upper_definition", "baseline_definition",
        "spacing_delta_x_multiplier", "delta_x_definition", "level_index_policy",
        "level_capacity_policy", "logical_rounding", "commissioning_read_samples",
        "commissioning", "joint_assignment_repair", "program_verify",
        "physical_invariants", "exclusions",
    }
    _keys(raw, path, fields)
    for name, expected in (
        ("metrics", _METRICS), ("execution", _EXECUTION), ("device", _DEVICE),
        ("source", _SOURCE), ("predecessor", _PREDECESSOR),
        ("calibration", _CALIBRATION), ("truncation", _TRUNCATION),
        ("commissioning", _COMMISSIONING), ("joint_assignment_repair", _JOINT_REPAIR),
        ("program_verify", _PROGRAM_VERIFY), ("physical_invariants", _PHYSICAL_INVARIANTS),
        ("exclusions", _EXCLUSIONS),
    ):
        _require_exact(_object(raw[name], f"{path}.{name}"), expected, path=f"{path}.{name}")

    assignment_raw = _object(raw["assignments"], f"{path}.assignments")
    _keys(assignment_raw, f"{path}.assignments", {"allowed_heldout_seeds", "heldout_seed", "endpoint_seeds"})
    heldout = assignment_raw["heldout_seed"]
    if type(heldout) is not int or heldout not in HELDOUT_ASSIGNMENT_SEEDS:
        raise config_error(f"{path}.assignments.heldout_seed", f"to be one of {HELDOUT_ASSIGNMENT_SEEDS!r}", heldout)
    expected_assignment = {
        "allowed_heldout_seeds": list(HELDOUT_ASSIGNMENT_SEEDS),
        "heldout_seed": heldout,
        "endpoint_seeds": list(ENDPOINT_SEEDS_BY_ASSIGNMENT[heldout]),
    }
    _require_exact(assignment_raw, expected_assignment, path=f"{path}.assignments")

    exact_scalars = {
        "baseline_policy": "shared_destination_columns",
        "group_lower_definition": "L_j=max_bounded_truncated_RESET_mean_in_x_destination_column",
        "group_upper_definition": "U_j=min_truncated_sampled_x_upper_bound_in_destination_column",
        "baseline_definition": "B_j=L_j+alpha*(U_j-L_j)",
        "delta_x_definition": "nominal_dw_min/2_in_x",
        "level_index_policy": "nonnegative_sign_selected_integer_levels",
        "level_capacity_policy": "whole_truncated_support_levels_no_short_terminal_interval",
        "logical_rounding": "nearest_nonnegative_integer_half_away_from_zero",
        "commissioning_read_samples": RESET_READ_SAMPLES,
    }
    for name, expected in exact_scalars.items():
        if not _same_json_value(raw[name], expected):
            raise config_error(f"{path}.{name}", f"to equal {expected!r}", raw[name])
    alpha = raw["baseline_position_fraction"]
    if type(alpha) is not float or alpha not in BASELINE_POSITION_FRACTIONS:
        raise config_error(f"{path}.baseline_position_fraction", f"to be one of {BASELINE_POSITION_FRACTIONS!r} as a JSON float", alpha)
    spacing = raw["spacing_delta_x_multiplier"]
    if type(spacing) is not int or spacing not in SPACING_DELTA_X_MULTIPLIERS:
        raise config_error(f"{path}.spacing_delta_x_multiplier", f"to be one of {SPACING_DELTA_X_MULTIPLIERS!r}", spacing)

    return BaselineSpacingPvTruncatedNominalProtocol(
        metrics=MetricContract(**_METRICS),
        execution=ExecutionContract(**_EXECUTION),
        device=OmDeviceContract(**_DEVICE),
        source=SourceContract(**_SOURCE),
        predecessor=PredecessorContract(**{**_PREDECESSOR, "heldout_hardware_instance_ids": tuple(_PREDECESSOR["heldout_hardware_instance_ids"])}),
        calibration=CalibrationContract(**{**_CALIBRATION, "scale_fractions": SCALE_FRACTIONS}),
        assignments=AssignmentContract(allowed_heldout_seeds=HELDOUT_ASSIGNMENT_SEEDS, heldout_seed=heldout, endpoint_seeds=ENDPOINT_SEEDS_BY_ASSIGNMENT[heldout]),
        truncation=TruncationContract(**_TRUNCATION),
        baseline_policy=raw["baseline_policy"], baseline_position_fraction=alpha,
        group_lower_definition=raw["group_lower_definition"], group_upper_definition=raw["group_upper_definition"], baseline_definition=raw["baseline_definition"],
        spacing_delta_x_multiplier=spacing, delta_x_definition=raw["delta_x_definition"],
        level_index_policy=raw["level_index_policy"], level_capacity_policy=raw["level_capacity_policy"], logical_rounding=raw["logical_rounding"],
        commissioning_read_samples=RESET_READ_SAMPLES,
        commissioning=CommissioningContract(**_COMMISSIONING),
        joint_assignment_repair=JointAssignmentRepairContract(**_JOINT_REPAIR),
        program_verify=ProgramVerifyContract(**_PROGRAM_VERIFY),
        physical_invariants=PhysicalInvariantContract(**{**_PHYSICAL_INVARIANTS, "destination_column_groups": tuple(tuple(group) for group in _PHYSICAL_INVARIANTS["destination_column_groups"])}),
        exclusions=ExclusionContract(**_EXCLUSIONS),
    )


def _validate_student_contract(student: StudentConfig, raw: Mapping[str, Any]) -> None:
    if set(student.modes) != {RunMode.VALIDATE.value}:
        raise config_error("config.modes", "to define exactly the validate mode", tuple(student.modes))
    for name, expected in (
        ("runtime", _STUDENT_RUNTIME), ("data", _STUDENT_DATA),
        ("model", _STUDENT_MODEL), ("solver", _STUDENT_SOLVER),
        ("mapping", _STUDENT_MAPPING), ("teacher", _STUDENT_TEACHER),
    ):
        _require_exact(_object(raw[name], f"config.{name}"), expected, path=f"config.{name}")
    modes = _object(raw["modes"], "config.modes")
    _keys(modes, "config.modes", {"validate"})
    _require_exact(_object(modes["validate"], "config.modes.validate"), {
        "split": "test", "sample_limit": None, "noise_repeats": 1,
        "weight_modifier": {"type": "none", "parameters": {}},
    }, path="config.modes.validate")


def parse_baseline_spacing_pv_truncated_nominal_config(
    payload: Mapping[str, Any],
) -> BaselineSpacingPvTruncatedNominalConfig:
    raw = _object(payload, "config")
    student_keys = {"schema_version", "experiment_id", "runtime", "data", "teacher", "model", "solver", "mapping", "modes"}
    _keys(raw, "config", student_keys | {"baseline_spacing_pv_truncated_nominal"})
    if raw["schema_version"] != SCHEMA_VERSION:
        raise config_error("config.schema_version", "to equal 1", raw["schema_version"])
    if raw["experiment_id"] != EXPERIMENT_ID:
        raise config_error("config.experiment_id", f"to equal {EXPERIMENT_ID!r}", raw["experiment_id"])
    protocol = _parse_protocol(raw["baseline_spacing_pv_truncated_nominal"])
    student_payload = {key: raw[key] for key in student_keys}
    student_payload["experiment_id"] = STUDENT_EXPERIMENT_ID
    student = parse_student_config(student_payload)
    _validate_student_contract(student, raw)
    return BaselineSpacingPvTruncatedNominalConfig(SCHEMA_VERSION, EXPERIMENT_ID, protocol, student)


def resolve_baseline_spacing_pv_truncated_nominal_spec(
    document: BaselineSpacingPvTruncatedNominalConfig, mode: RunMode,
) -> BaselineSpacingPvTruncatedNominalValidateSpec:
    if mode is not RunMode.VALIDATE:
        raise config_error("the requested run mode", "to be 'validate' for this validate-only experiment", mode.value)
    student = resolve_student_spec(document.student, mode)
    if not isinstance(student, StudentValidateSpec):  # pragma: no cover
        raise RuntimeError("Expected the parent parser to resolve validation.")
    return BaselineSpacingPvTruncatedNominalValidateSpec(EXPERIMENT_ID, document.protocol, student)


__all__ = [
    "BASELINE_POSITION_FRACTIONS", "BaselineSpacingPvTruncatedNominalConfig",
    "BaselineSpacingPvTruncatedNominalProtocol", "BaselineSpacingPvTruncatedNominalValidateSpec",
    "DEVELOPMENT_ASSIGNMENT_SEED", "ENDPOINT_SEEDS_BY_ASSIGNMENT", "EXPERIMENT_ID",
    "HELDOUT_ASSIGNMENT_SEEDS", "SCHEMA_VERSION", "SPACING_DELTA_X_MULTIPLIERS",
    "TruncationContract", "parse_baseline_spacing_pv_truncated_nominal_config",
    "resolve_baseline_spacing_pv_truncated_nominal_spec",
]
