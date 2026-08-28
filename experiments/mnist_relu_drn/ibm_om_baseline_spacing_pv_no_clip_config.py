"""Strict config for the exploratory IBM OM no-clipping successor screen.

The predecessor baseline/spacing experiment intersected native IBM OM states
with a public ``x in [0, 1]`` interval and projected P&V endpoints at circuit
handoff.  This successor instead freezes one study-wide affine translation of
the complete sampled raw-``x`` support.  It deliberately provides only a
configuration and registry surface; numerical execution requires a dedicated
runtime that implements these semantics without falling back to the clipped
predecessor.
"""

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
from experiments.schema import RunMode, config_error


EXPERIMENT_ID = "mnist_ibm_om_baseline_spacing_pv_no_clip.v1"
SCHEMA_VERSION = 1

IDEAL_QUANTIZED_METRIC_DEFINITION = (
    "ibm_om.ideal_raw_support_affine_uniform_spacing_init.v1"
)
PV_PERSISTENT_METRIC_DEFINITION = (
    "ibm_om.pv_persistent_raw_support_affine_uniform_spacing_init.v1"
)
CONTINUOUS_DIAGNOSTIC_METRIC_DEFINITION = (
    "ibm_om.ideal_raw_support_affine_continuous_init.v1"
)

PREDECESSOR_EXPERIMENT_ID = "mnist_ibm_om_baseline_spacing_pv.v1"
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
P_AND_V_REPEAT_COUNT = 5
MAXIMUM_PROGRAM_PULSES = 128
VERIFY_TOLERANCE_DELTA_X_RATIO = 0.5

# Literal finite-cohort calibration audited from the already frozen repaired
# development and held-out populations.  Keep the floor and strict-positive
# margin separate so the provenance of the affine origin remains explicit.
EXACT_SUPPORT_FLOOR_X = -1.3759238719940186
STRICT_POSITIVE_MARGIN_X = 1e-6
AFFINE_SUPPORT_ORIGIN_X = -1.3759248719940185
AFFINE_SLOPE_G_PER_X = 0.00011
EXACT_SUPPORT_CEILING_X = 1.8474750518798828
MAPPED_MINIMUM_G = 1.1e-10
MAPPED_CEILING_G = 0.0003545739916261292


@dataclass(frozen=True)
class MetricContract:
    ideal_quantized: str
    pv_persistent: str
    continuous_diagnostic: str


@dataclass(frozen=True)
class ExecutionContract:
    profile: str
    evidence_tier: str
    workflow: str


@dataclass(frozen=True)
class OmDeviceContract:
    evidence_class: str
    preset: str
    required_aihwkit_version: str
    corruption_policy: str
    native_coordinate: str


@dataclass(frozen=True)
class SourceContract:
    weights_role: str
    expected_weights_sha256: str


@dataclass(frozen=True)
class PredecessorContract:
    experiment_id: str
    mapping_status: str
    assignment_reconstruction: str
    development_hardware_instance_id: str
    heldout_hardware_instance_ids: tuple[str, ...]


@dataclass(frozen=True)
class CalibrationContract:
    development_assignment_seed: int
    scale_fractions: tuple[float, ...]
    selection_domain: str
    selection_metric: str
    logit_gain: str
    sharing: str
    heldout_application: str


@dataclass(frozen=True)
class AssignmentContract:
    allowed_heldout_seeds: tuple[int, ...]
    heldout_seed: int
    endpoint_seeds: tuple[int, ...]


@dataclass(frozen=True)
class AffineEmbeddingContract:
    policy: str
    native_coordinate: str
    exact_support_floor_x: float
    strict_positive_margin_x: float
    support_origin_x: float
    slope_g_per_x: float
    exact_support_ceiling_x: float
    mapped_minimum_g: float
    mapped_ceiling_g: float
    floor_scope: str
    scale_policy: str
    endpoint_policy: str


@dataclass(frozen=True)
class CommissioningContract:
    samples_per_cell: int
    sample_sequence: str
    read_coordinate: str
    baseline_estimator: str
    bound_policy: str
    noise_policy: str
    freeze_policy: str


@dataclass(frozen=True)
class JointAssignmentRepairContract:
    scope: str
    donor_seed_derivation: str
    destination_traversal: str
    donor_traversal: str
    eligibility: str
    identity_copy: str
    maximum_candidates_per_quad: int
    exhaustion: str


@dataclass(frozen=True)
class ProgramVerifyContract:
    controller: str
    start_protocol: str
    tolerance_delta_x_ratio: float
    maximum_program_pulses: int
    verify_coordinate: str
    persistent_coordinate: str
    cycle_to_cycle_noise: bool
    apparent_verify_noise: bool
    inference_read_noise: bool
    endpoint_for_accuracy: str
    circuit_handoff: str
    apparent_endpoint_role: str
    repeat_count: int


@dataclass(frozen=True)
class PhysicalInvariantContract:
    conductance_representation: str
    destination_column_groups: tuple[tuple[str, str], ...]
    exact_zero_requirement: str
    transfer: str
    loading: str
    support: str
    out_of_bounds: str


@dataclass(frozen=True)
class ExclusionContract:
    optimizer_updates: int
    hardware_aware_training: bool
    inference_read_noise: float
    retention_drift: float
    logical_rewrites: int
    post_deployment_recovery: bool


@dataclass(frozen=True)
class BaselineSpacingPvNoClipProtocol:
    metrics: MetricContract
    execution: ExecutionContract
    device: OmDeviceContract
    source: SourceContract
    predecessor: PredecessorContract
    calibration: CalibrationContract
    assignments: AssignmentContract
    affine_embedding: AffineEmbeddingContract
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
class BaselineSpacingPvNoClipConfig:
    schema_version: int
    experiment_id: str
    protocol: BaselineSpacingPvNoClipProtocol
    student: StudentConfig


@dataclass(frozen=True)
class BaselineSpacingPvNoClipValidateSpec:
    experiment_id: str
    protocol: BaselineSpacingPvNoClipProtocol
    student: StudentValidateSpec


_METRICS = {
    "ideal_quantized": IDEAL_QUANTIZED_METRIC_DEFINITION,
    "pv_persistent": PV_PERSISTENT_METRIC_DEFINITION,
    "continuous_diagnostic": CONTINUOUS_DIAGNOSTIC_METRIC_DEFINITION,
}
_EXECUTION = {
    "profile": "production",
    "evidence_tier": "exploratory_noncanonical",
    "workflow": "direct_cuda_no_study_launcher_or_smoke",
}
_DEVICE = {
    "evidence_class": "model_based_aihwkit_preset",
    "preset": "reram_array_om",
    "required_aihwkit_version": "1.1.0",
    "corruption_policy": "counterfactual_repaired",
    "native_coordinate": "raw_x=(a+1)/2",
}
_SOURCE = {
    "weights_role": "frozen_relu_source_and_teacher",
    "expected_weights_sha256": EXPECTED_WEIGHTS_SHA256,
}
_PREDECESSOR = {
    "experiment_id": PREDECESSOR_EXPERIMENT_ID,
    "mapping_status": "rejected_public_0_1_clipped_diagnostic",
    "assignment_reconstruction": (
        "same_identity_commissioning_and_joint_repair_semantics"
    ),
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
    "selection_domain": "continuous_raw_support_development_assignment_86001",
    "selection_metric": (
        "continuous_accuracy_then_calibrated_kl_then_lexicographic_scale_pair"
    ),
    "logit_gain": "positive_kl_fit_on_selected_continuous_mapping",
    "sharing": (
        "one_per_alpha_frozen_across_spacing_heldout_and_endpoint_repeats"
    ),
    "heldout_application": (
        "freeze_alpha_specific_scale_pair_and_gain_before_quantization_or_pv"
    ),
}
_AFFINE_EMBEDDING = {
    "policy": "study_wide_pure_translation_no_rescaling",
    "native_coordinate": "raw_x=(a+1)/2",
    "exact_support_floor_x": EXACT_SUPPORT_FLOOR_X,
    "strict_positive_margin_x": STRICT_POSITIVE_MARGIN_X,
    "support_origin_x": AFFINE_SUPPORT_ORIGIN_X,
    "slope_g_per_x": AFFINE_SLOPE_G_PER_X,
    "exact_support_ceiling_x": EXACT_SUPPORT_CEILING_X,
    "mapped_minimum_g": MAPPED_MINIMUM_G,
    "mapped_ceiling_g": MAPPED_CEILING_G,
    "floor_scope": (
        "minimum_exact_persistent_lower_support_across_frozen_repaired_"
        "assignments_86001_87001_87002_87003"
    ),
    "scale_policy": "retain_predecessor_slope_without_upper_range_rescaling",
    "endpoint_policy": (
        "apply_same_affine_map_to_targets_and_persistent_endpoints_no_projection"
    ),
}
_COMMISSIONING = {
    "samples_per_cell": RESET_READ_SAMPLES,
    "sample_sequence": (
        "initialize_at_sampled_lower_bound_then_one_reset_pulse_and_"
        "apparent_read_per_sample"
    ),
    "read_coordinate": "raw_active_a_then_map_to_raw_x=(a+1)/2",
    "baseline_estimator": "per_cell_arithmetic_mean",
    "bound_policy": (
        "bound_mean_to_exact_sampled_raw_x_min_max_without_public_0_1_intersection"
    ),
    "noise_policy": "preset_cycle_to_cycle_and_apparent_write_noise_enabled",
    "freeze_policy": (
        "reuse_identity_and_commissioning_receipt_across_alpha_spacing_"
        "and_endpoint_repeats"
    ),
}
_JOINT_REPAIR = {
    "scope": "same_complete_quad_joint_repair_as_clipped_predecessor",
    "donor_seed_derivation": (
        "derive_seed(assignment_seed,mnist_ibm_om_baseline_selection_joint_donor_v1)"
    ),
    "destination_traversal": "canonical_layer_then_logical_quad_order",
    "donor_traversal": "same_layer_unique_donor_quads_in_canonical_order",
    "eligibility": "reuse_frozen_predecessor_joint_identity_assignment_exactly",
    "identity_copy": "copy_complete_four_cell_quad_and_commissioned_observations",
    "maximum_candidates_per_quad": MAXIMUM_DONOR_CANDIDATES_PER_QUAD,
    "exhaustion": "invalidate_assignment",
}
_PROGRAM_VERIFY = {
    "controller": "one_pulse",
    "start_protocol": "conditioned_lower_bound_to_target",
    "tolerance_delta_x_ratio": VERIFY_TOLERANCE_DELTA_X_RATIO,
    "maximum_program_pulses": MAXIMUM_PROGRAM_PULSES,
    "verify_coordinate": "raw_x=(a+1)/2",
    "persistent_coordinate": "raw_x=(a+1)/2",
    "cycle_to_cycle_noise": True,
    "apparent_verify_noise": True,
    "inference_read_noise": False,
    "endpoint_for_accuracy": "persistent",
    "circuit_handoff": "affine_map_persistent_raw_x_without_projection",
    "apparent_endpoint_role": (
        "controller_and_diagnostic_only_never_applied_to_drn"
    ),
    "repeat_count": P_AND_V_REPEAT_COUNT,
}
_PHYSICAL_INVARIANTS = {
    "conductance_representation": (
        "G=slope_g_per_x*(raw_x-support_origin_x)=B+n*(slope_g_per_x*h)"
    ),
    "destination_column_groups": [["G++", "G-+"], ["G+-", "G--"]],
    "exact_zero_requirement": "C(B+,B-,B+,B-)=0",
    "transfer": "full_signed_four_conductance_contrast",
    "loading": "full_conductance_sum",
    "support": "literal_sampled_native_min_bound_to_max_bound_no_public_intersection",
    "out_of_bounds": (
        "invalidate_native_support_or_affine_conductance_violation_no_projection"
    ),
}
_EXCLUSIONS = {
    "optimizer_updates": 0,
    "hardware_aware_training": False,
    "inference_read_noise": 0.0,
    "retention_drift": 0.0,
    "logical_rewrites": 0,
    "post_deployment_recovery": False,
}

_STUDENT_RUNTIME = {
    "seed": 42,
    "data_seed": 42,
    "device": "cuda",
    "dtype": "float32",
}
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
    "conductance_max": MAPPED_CEILING_G,
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
_STUDENT_TEACHER = {
    "type": "bias_free_relu",
    "initialization": "signed_weight_mapping",
}


def _same_json_value(provided: Any, expected: Any) -> bool:
    """Compare frozen JSON values without bool/number coercion."""

    if type(provided) is not type(expected):
        return False
    if isinstance(expected, list):
        return len(provided) == len(expected) and all(
            _same_json_value(left, right)
            for left, right in zip(provided, expected)
        )
    if isinstance(expected, dict):
        return set(provided) == set(expected) and all(
            _same_json_value(provided[key], value)
            for key, value in expected.items()
        )
    return bool(provided == expected)


def _require_exact(
    raw: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    path: str,
) -> None:
    _keys(raw, path, set(expected))
    for name, expected_value in expected.items():
        provided = raw[name]
        if not _same_json_value(provided, expected_value):
            raise config_error(
                f"{path}.{name}", f"to equal {expected_value!r}", provided
            )


def _parse_assignments(value: Any, *, path: str) -> AssignmentContract:
    raw = _object(value, path)
    _keys(raw, path, {"allowed_heldout_seeds", "heldout_seed", "endpoint_seeds"})
    allowed = list(HELDOUT_ASSIGNMENT_SEEDS)
    if not _same_json_value(raw["allowed_heldout_seeds"], allowed):
        raise config_error(
            f"{path}.allowed_heldout_seeds", f"to equal {allowed!r}",
            raw["allowed_heldout_seeds"],
        )
    heldout = raw["heldout_seed"]
    if type(heldout) is not int or heldout not in HELDOUT_ASSIGNMENT_SEEDS:
        raise config_error(
            f"{path}.heldout_seed",
            f"to be one of {HELDOUT_ASSIGNMENT_SEEDS!r}", heldout,
        )
    endpoints = list(ENDPOINT_SEEDS_BY_ASSIGNMENT[heldout])
    if not _same_json_value(raw["endpoint_seeds"], endpoints):
        raise config_error(
            f"{path}.endpoint_seeds",
            f"to equal {endpoints!r} for heldout seed {heldout}",
            raw["endpoint_seeds"],
        )
    return AssignmentContract(
        allowed_heldout_seeds=HELDOUT_ASSIGNMENT_SEEDS,
        heldout_seed=heldout,
        endpoint_seeds=ENDPOINT_SEEDS_BY_ASSIGNMENT[heldout],
    )


def _parse_protocol(value: Any) -> BaselineSpacingPvNoClipProtocol:
    path = "config.baseline_spacing_pv_no_clip"
    raw = _object(value, path)
    _keys(
        raw,
        path,
        {
            "metrics", "execution", "device", "source", "predecessor",
            "calibration", "assignments", "affine_embedding",
            "baseline_policy", "baseline_position_fraction",
            "group_lower_definition", "group_upper_definition",
            "baseline_definition", "spacing_delta_x_multiplier",
            "delta_x_definition", "level_index_policy",
            "level_capacity_policy", "logical_rounding",
            "commissioning_read_samples", "commissioning",
            "joint_assignment_repair", "program_verify",
            "physical_invariants", "exclusions",
        },
    )
    exact_sections = (
        ("metrics", _METRICS),
        ("execution", _EXECUTION),
        ("device", _DEVICE),
        ("source", _SOURCE),
        ("predecessor", _PREDECESSOR),
        ("calibration", _CALIBRATION),
        ("affine_embedding", _AFFINE_EMBEDDING),
        ("commissioning", _COMMISSIONING),
        ("joint_assignment_repair", _JOINT_REPAIR),
        ("program_verify", _PROGRAM_VERIFY),
        ("physical_invariants", _PHYSICAL_INVARIANTS),
        ("exclusions", _EXCLUSIONS),
    )
    for name, expected in exact_sections:
        _require_exact(_object(raw[name], f"{path}.{name}"), expected, path=f"{path}.{name}")

    assignments = _parse_assignments(raw["assignments"], path=f"{path}.assignments")
    expected_scalars = {
        "baseline_policy": "shared_destination_columns",
        "group_lower_definition": (
            "L_j=max_bounded_commissioned_reset_mean_in_raw_x_destination_column"
        ),
        "group_upper_definition": (
            "U_j=min_exact_sampled_raw_x_upper_bound_in_destination_column"
        ),
        "baseline_definition": "B_j=L_j+alpha*(U_j-L_j)",
        "delta_x_definition": "nominal_dw_min/2_in_raw_x",
        "level_index_policy": "nonnegative_sign_selected_integer_levels",
        "level_capacity_policy": "whole_native_support_levels_no_short_terminal_interval",
        "logical_rounding": "nearest_nonnegative_integer_half_away_from_zero",
        "commissioning_read_samples": RESET_READ_SAMPLES,
    }
    for name, expected in expected_scalars.items():
        if not _same_json_value(raw[name], expected):
            raise config_error(f"{path}.{name}", f"to equal {expected!r}", raw[name])

    alpha = raw["baseline_position_fraction"]
    if type(alpha) is not float or alpha not in BASELINE_POSITION_FRACTIONS:
        raise config_error(
            f"{path}.baseline_position_fraction",
            f"to be one of {BASELINE_POSITION_FRACTIONS!r} as a JSON float",
            alpha,
        )
    spacing = raw["spacing_delta_x_multiplier"]
    if type(spacing) is not int or spacing not in SPACING_DELTA_X_MULTIPLIERS:
        raise config_error(
            f"{path}.spacing_delta_x_multiplier",
            f"to be one of {SPACING_DELTA_X_MULTIPLIERS!r}", spacing,
        )

    return BaselineSpacingPvNoClipProtocol(
        metrics=MetricContract(**_METRICS),
        execution=ExecutionContract(**_EXECUTION),
        device=OmDeviceContract(**_DEVICE),
        source=SourceContract(**_SOURCE),
        predecessor=PredecessorContract(
            **{
                **_PREDECESSOR,
                "heldout_hardware_instance_ids": tuple(
                    _PREDECESSOR["heldout_hardware_instance_ids"]
                ),
            }
        ),
        calibration=CalibrationContract(
            **{**_CALIBRATION, "scale_fractions": SCALE_FRACTIONS}
        ),
        assignments=assignments,
        affine_embedding=AffineEmbeddingContract(**_AFFINE_EMBEDDING),
        baseline_policy=raw["baseline_policy"],
        baseline_position_fraction=alpha,
        group_lower_definition=raw["group_lower_definition"],
        group_upper_definition=raw["group_upper_definition"],
        baseline_definition=raw["baseline_definition"],
        spacing_delta_x_multiplier=spacing,
        delta_x_definition=raw["delta_x_definition"],
        level_index_policy=raw["level_index_policy"],
        level_capacity_policy=raw["level_capacity_policy"],
        logical_rounding=raw["logical_rounding"],
        commissioning_read_samples=RESET_READ_SAMPLES,
        commissioning=CommissioningContract(**_COMMISSIONING),
        joint_assignment_repair=JointAssignmentRepairContract(**_JOINT_REPAIR),
        program_verify=ProgramVerifyContract(**_PROGRAM_VERIFY),
        physical_invariants=PhysicalInvariantContract(
            **{
                **_PHYSICAL_INVARIANTS,
                "destination_column_groups": tuple(
                    tuple(group)
                    for group in _PHYSICAL_INVARIANTS["destination_column_groups"]
                ),
            }
        ),
        exclusions=ExclusionContract(**_EXCLUSIONS),
    )


def _validate_student_contract(student: StudentConfig, raw: Mapping[str, Any]) -> None:
    if set(student.modes) != {RunMode.VALIDATE.value}:
        raise config_error(
            "config.modes", "to define exactly the validate mode", tuple(student.modes)
        )
    for name, expected in (
        ("runtime", _STUDENT_RUNTIME),
        ("data", _STUDENT_DATA),
        ("model", _STUDENT_MODEL),
        ("solver", _STUDENT_SOLVER),
        ("mapping", _STUDENT_MAPPING),
        ("teacher", _STUDENT_TEACHER),
    ):
        _require_exact(_object(raw[name], f"config.{name}"), expected, path=f"config.{name}")
    modes = _object(raw["modes"], "config.modes")
    _keys(modes, "config.modes", {"validate"})
    _require_exact(
        _object(modes["validate"], "config.modes.validate"),
        {
            "split": "test",
            "sample_limit": None,
            "noise_repeats": 1,
            "weight_modifier": {"type": "none", "parameters": {}},
        },
        path="config.modes.validate",
    )


def parse_baseline_spacing_pv_no_clip_config(
    payload: Mapping[str, Any],
) -> BaselineSpacingPvNoClipConfig:
    """Parse one immutable no-clipping alpha/spacing/assignment config."""

    raw = _object(payload, "config")
    student_keys = {
        "schema_version", "experiment_id", "runtime", "data", "teacher",
        "model", "solver", "mapping", "modes",
    }
    _keys(raw, "config", student_keys | {"baseline_spacing_pv_no_clip"})
    if raw["schema_version"] != SCHEMA_VERSION:
        raise config_error("config.schema_version", "to equal 1", raw["schema_version"])
    if raw["experiment_id"] != EXPERIMENT_ID:
        raise config_error(
            "config.experiment_id", f"to equal {EXPERIMENT_ID!r}", raw["experiment_id"]
        )
    modes = _object(raw["modes"], "config.modes")
    if set(modes) != {RunMode.VALIDATE.value}:
        raise config_error(
            "config.modes", "to define exactly the validate mode", tuple(modes)
        )

    protocol = _parse_protocol(raw["baseline_spacing_pv_no_clip"])
    student_payload = {key: raw[key] for key in student_keys}
    student_payload["experiment_id"] = STUDENT_EXPERIMENT_ID
    student = parse_student_config(student_payload)
    _validate_student_contract(student, raw)
    return BaselineSpacingPvNoClipConfig(
        schema_version=SCHEMA_VERSION,
        experiment_id=EXPERIMENT_ID,
        protocol=protocol,
        student=student,
    )


def resolve_baseline_spacing_pv_no_clip_spec(
    document: BaselineSpacingPvNoClipConfig,
    mode: RunMode,
) -> BaselineSpacingPvNoClipValidateSpec:
    """Resolve validation only and retain the nested parent student spec."""

    if mode is not RunMode.VALIDATE:
        raise config_error(
            "the requested run mode",
            "to be 'validate' for this validate-only experiment",
            mode.value,
        )
    student = resolve_student_spec(document.student, mode)
    if not isinstance(student, StudentValidateSpec):  # pragma: no cover
        raise RuntimeError("Expected the parent parser to resolve validation.")
    return BaselineSpacingPvNoClipValidateSpec(
        experiment_id=EXPERIMENT_ID,
        protocol=document.protocol,
        student=student,
    )


__all__ = [
    "AFFINE_SLOPE_G_PER_X",
    "AFFINE_SUPPORT_ORIGIN_X",
    "BASELINE_POSITION_FRACTIONS",
    "BaselineSpacingPvNoClipConfig",
    "BaselineSpacingPvNoClipProtocol",
    "BaselineSpacingPvNoClipValidateSpec",
    "CONTINUOUS_DIAGNOSTIC_METRIC_DEFINITION",
    "DEVELOPMENT_ASSIGNMENT_SEED",
    "ENDPOINT_SEEDS_BY_ASSIGNMENT",
    "EXACT_SUPPORT_CEILING_X",
    "EXACT_SUPPORT_FLOOR_X",
    "EXPECTED_WEIGHTS_SHA256",
    "EXPERIMENT_ID",
    "HELDOUT_ASSIGNMENT_SEEDS",
    "IDEAL_QUANTIZED_METRIC_DEFINITION",
    "MAPPED_CEILING_G",
    "MAPPED_MINIMUM_G",
    "MAXIMUM_DONOR_CANDIDATES_PER_QUAD",
    "MAXIMUM_PROGRAM_PULSES",
    "P_AND_V_REPEAT_COUNT",
    "PV_PERSISTENT_METRIC_DEFINITION",
    "RESET_READ_SAMPLES",
    "SCALE_FRACTIONS",
    "SCHEMA_VERSION",
    "SPACING_DELTA_X_MULTIPLIERS",
    "STRICT_POSITIVE_MARGIN_X",
    "VERIFY_TOLERANCE_DELTA_X_RATIO",
    "parse_baseline_spacing_pv_no_clip_config",
    "resolve_baseline_spacing_pv_no_clip_spec",
]
