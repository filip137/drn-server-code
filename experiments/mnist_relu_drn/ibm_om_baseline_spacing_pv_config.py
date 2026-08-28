"""Strict validate-only contract for the IBM OM baseline/spacing P&V study.

One config selects one baseline position, one uniform spacing, and one held-out
OM assignment.  The dedicated runtime evaluates the exact ideal-quantized
target and five pulse-resolved persistent P&V repeats from the same mapping.
Calibration is selected continuously once per baseline position and is frozen
across spacing, held-out assignment, and endpoint repeat.
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


EXPERIMENT_ID = "mnist_ibm_om_baseline_spacing_pv.v1"
SCHEMA_VERSION = 1

IDEAL_QUANTIZED_METRIC_DEFINITION = (
    "ibm_om.ideal_bounded_uniform_spacing_init.v1"
)
PV_PERSISTENT_METRIC_DEFINITION = (
    "ibm_om.pv_persistent_uniform_spacing_init.v1"
)
CONTINUOUS_DIAGNOSTIC_METRIC_DEFINITION = (
    "ibm_om.ideal_bounded_continuous_init.v1"
)

STUDY_ID = (
    "mnist-ibm-om-shared-destination-baseline-spacing-pv-20260828-v1"
)
REFERENCE_STUDY_ID = "mnist-ibm-om-four-device-baseline-selection-20260828-v1"
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
EXECUTION_PROFILES = ("production", "smoke")


@dataclass(frozen=True)
class MetricContract:
    ideal_quantized: str
    pv_persistent: str
    continuous_diagnostic: str


@dataclass(frozen=True)
class ExecutionContract:
    profile: str


@dataclass(frozen=True)
class OmDeviceContract:
    evidence_class: str
    preset: str
    required_aihwkit_version: str
    corruption_policy: str
    conductance_coordinate: str


@dataclass(frozen=True)
class SourceContract:
    weights_role: str
    expected_weights_sha256: str


@dataclass(frozen=True)
class HistoricalParityContract:
    development_hardware_instance_id: str
    heldout_assignment_seeds: tuple[int, ...]
    heldout_hardware_instance_ids: tuple[str, ...]
    ideal_correct: tuple[int, ...]
    ideal_prediction_sha256: tuple[str, ...]
    selected_scale_fractions: tuple[float, ...]
    fixed_logit_gain: float


@dataclass(frozen=True)
class ReferenceStudyContract:
    study_id: str
    selected_policy: str
    assignment_reconstruction: str
    alpha0_spacing4_parity: HistoricalParityContract


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
    public_conductance_handoff: str
    repeat_count: int


@dataclass(frozen=True)
class PhysicalInvariantContract:
    conductance_representation: str
    destination_column_groups: tuple[tuple[str, str], ...]
    exact_zero_requirement: str
    transfer: str
    loading: str
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
class BaselineSpacingPvProtocol:
    metrics: MetricContract
    execution: ExecutionContract
    device: OmDeviceContract
    source: SourceContract
    reference_study: ReferenceStudyContract
    calibration: CalibrationContract
    assignments: AssignmentContract
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
class BaselineSpacingPvConfig:
    schema_version: int
    experiment_id: str
    protocol: BaselineSpacingPvProtocol
    student: StudentConfig


@dataclass(frozen=True)
class BaselineSpacingPvValidateSpec:
    experiment_id: str
    protocol: BaselineSpacingPvProtocol
    student: StudentValidateSpec


_METRICS = {
    "ideal_quantized": IDEAL_QUANTIZED_METRIC_DEFINITION,
    "pv_persistent": PV_PERSISTENT_METRIC_DEFINITION,
    "continuous_diagnostic": CONTINUOUS_DIAGNOSTIC_METRIC_DEFINITION,
}

_DEVICE = {
    "evidence_class": "model_based_aihwkit_preset",
    "preset": "reram_array_om",
    "required_aihwkit_version": "1.1.0",
    "corruption_policy": "counterfactual_repaired",
    "conductance_coordinate": "x=(a+1)/2",
}

_SOURCE = {
    "weights_role": "frozen_relu_source_and_teacher",
    "expected_weights_sha256": EXPECTED_WEIGHTS_SHA256,
}

_HISTORICAL_PARITY = {
    "development_hardware_instance_id": (
        "feddd5a62ae3ace700dc46f155f54236c58fa769c0d4d6603613ed857b7a0150"
    ),
    "heldout_assignment_seeds": list(HELDOUT_ASSIGNMENT_SEEDS),
    "heldout_hardware_instance_ids": [
        "58fb1079e7bd60f53ba93ba61a3dca887352846b9f5a5768e9cd7bb72aed8492",
        "215d46a5ec563abdf2c9450625dea0d6620280dec8a18a413e6199adc72efb50",
        "3be072fb1dcad07d54176b8e0fd30e8f376306853f1955729f0a00254c5475d9",
    ],
    "ideal_correct": [9098, 8580, 8889],
    "ideal_prediction_sha256": [
        "32714304647fba730bfd769b8d5b43cf5d6b3dad7cbf049da595a7f3f2c0a4d6",
        "3be15ece3ad194e70c846d4207ef3b097162b72ae39764c0f405a759dfe0789f",
        "43ddf6df2b0c685a185ed2d8fe946d848259e32d5df3c8c02286b1a92512980f",
    ],
    "selected_scale_fractions": [1.0, 1.0],
    "fixed_logit_gain": 14.12537544622754,
}

_REFERENCE_STUDY = {
    "study_id": REFERENCE_STUDY_ID,
    "selected_policy": "shared_destination_columns_reset_max",
    "assignment_reconstruction": (
        "same_identity_commissioning_and_joint_repair_semantics"
    ),
    "alpha0_spacing4_parity": _HISTORICAL_PARITY,
}

_CALIBRATION = {
    "development_assignment_seed": DEVELOPMENT_ASSIGNMENT_SEED,
    "scale_fractions": list(SCALE_FRACTIONS),
    "selection_domain": "continuous_bounded_development_assignment_86001",
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

_COMMISSIONING = {
    "samples_per_cell": RESET_READ_SAMPLES,
    "sample_sequence": (
        "initialize_at_sampled_lower_bound_then_one_reset_pulse_and_"
        "apparent_read_per_sample"
    ),
    "read_coordinate": "raw_active_a_then_map_to_x=(a+1)/2",
    "baseline_estimator": "per_cell_arithmetic_mean",
    "bound_policy": "clip_mapped_mean_to_sampled_active_conductance_bounds",
    "noise_policy": "preset_cycle_to_cycle_and_apparent_write_noise_enabled",
    "freeze_policy": (
        "reuse_identity_and_commissioning_receipt_across_alpha_spacing_"
        "and_endpoint_repeats"
    ),
}

_JOINT_REPAIR = {
    "scope": "same_complete_quad_joint_repair_as_reference_baseline_study",
    "donor_seed_derivation": (
        "derive_seed(assignment_seed,mnist_ibm_om_baseline_selection_joint_donor_v1)"
    ),
    "destination_traversal": "canonical_layer_then_logical_quad_order",
    "donor_traversal": "same_layer_unique_donor_quads_in_canonical_order",
    "eligibility": (
        "reference_study_four_policy_feasibility_before_alpha_spacing_or_pv"
    ),
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
    "public_conductance_handoff": (
        "hard_clip_raw_x_to_public_0_1_at_circuit_handoff_with_saved_masks"
    ),
    "repeat_count": P_AND_V_REPEAT_COUNT,
}

_PHYSICAL_INVARIANTS = {
    "conductance_representation": "G=B+n*h",
    "destination_column_groups": [["G++", "G-+"], ["G+-", "G--"]],
    "exact_zero_requirement": "C(B+,B-,B+,B-)=0",
    "transfer": "full_signed_four_conductance_contrast",
    "loading": "full_conductance_sum",
    "out_of_bounds": (
        "invalidate_ideal_targets_or_support_errors;retain_raw_pv_endpoint_and_"
        "project_only_at_declared_public_circuit_handoff"
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
    "conductance_max": 0.00011,
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
                f"{path}.{name}",
                f"to equal {expected_value!r}",
                provided,
            )


def _parse_protocol(value: Any) -> BaselineSpacingPvProtocol:
    path = "config.baseline_spacing_pv"
    raw = _object(value, path)
    _keys(
        raw,
        path,
        {
            "metrics",
            "execution",
            "device",
            "source",
            "reference_study",
            "calibration",
            "assignments",
            "baseline_policy",
            "baseline_position_fraction",
            "group_lower_definition",
            "group_upper_definition",
            "baseline_definition",
            "spacing_delta_x_multiplier",
            "delta_x_definition",
            "level_index_policy",
            "level_capacity_policy",
            "logical_rounding",
            "commissioning_read_samples",
            "commissioning",
            "joint_assignment_repair",
            "program_verify",
            "physical_invariants",
            "exclusions",
        },
    )

    metrics_raw = _object(raw["metrics"], f"{path}.metrics")
    _require_exact(metrics_raw, _METRICS, path=f"{path}.metrics")

    execution_path = f"{path}.execution"
    execution_raw = _object(raw["execution"], execution_path)
    _keys(execution_raw, execution_path, {"profile"})
    profile = execution_raw["profile"]
    if profile not in EXECUTION_PROFILES:
        raise config_error(
            f"{execution_path}.profile",
            f"to be one of {EXECUTION_PROFILES!r}",
            profile,
        )

    device_raw = _object(raw["device"], f"{path}.device")
    _require_exact(device_raw, _DEVICE, path=f"{path}.device")
    source_raw = _object(raw["source"], f"{path}.source")
    _require_exact(source_raw, _SOURCE, path=f"{path}.source")
    reference_raw = _object(raw["reference_study"], f"{path}.reference_study")
    _require_exact(reference_raw, _REFERENCE_STUDY, path=f"{path}.reference_study")
    calibration_raw = _object(raw["calibration"], f"{path}.calibration")
    _require_exact(calibration_raw, _CALIBRATION, path=f"{path}.calibration")

    assignments_path = f"{path}.assignments"
    assignments_raw = _object(raw["assignments"], assignments_path)
    _keys(
        assignments_raw,
        assignments_path,
        {"allowed_heldout_seeds", "heldout_seed", "endpoint_seeds"},
    )
    expected_allowed = list(HELDOUT_ASSIGNMENT_SEEDS)
    if not _same_json_value(
        assignments_raw["allowed_heldout_seeds"], expected_allowed
    ):
        raise config_error(
            f"{assignments_path}.allowed_heldout_seeds",
            f"to equal {expected_allowed!r}",
            assignments_raw["allowed_heldout_seeds"],
        )
    heldout_seed = assignments_raw["heldout_seed"]
    if type(heldout_seed) is not int or heldout_seed not in HELDOUT_ASSIGNMENT_SEEDS:
        raise config_error(
            f"{assignments_path}.heldout_seed",
            f"to be one of {HELDOUT_ASSIGNMENT_SEEDS!r}",
            heldout_seed,
        )
    expected_endpoint_seeds = list(ENDPOINT_SEEDS_BY_ASSIGNMENT[heldout_seed])
    if not _same_json_value(
        assignments_raw["endpoint_seeds"], expected_endpoint_seeds
    ):
        raise config_error(
            f"{assignments_path}.endpoint_seeds",
            f"to equal {expected_endpoint_seeds!r} for heldout seed {heldout_seed}",
            assignments_raw["endpoint_seeds"],
        )

    if raw["baseline_policy"] != "shared_destination_columns":
        raise config_error(
            f"{path}.baseline_policy",
            "to equal 'shared_destination_columns'",
            raw["baseline_policy"],
        )
    alpha = raw["baseline_position_fraction"]
    if type(alpha) is not float or alpha not in BASELINE_POSITION_FRACTIONS:
        raise config_error(
            f"{path}.baseline_position_fraction",
            f"to be one of {BASELINE_POSITION_FRACTIONS!r} as a JSON float",
            alpha,
        )
    expected_scalars = {
        "group_lower_definition": (
            "L_j=max_bounded_commissioned_reset_mean_in_destination_column"
        ),
        "group_upper_definition": (
            "U_j=min_sampled_upper_bound_in_destination_column"
        ),
        "baseline_definition": "B_j=L_j+alpha*(U_j-L_j)",
        "delta_x_definition": "nominal_dw_min/2",
        "level_index_policy": "nonnegative_sign_selected_integer_levels",
        "level_capacity_policy": "whole_in_bound_levels_no_short_terminal_interval",
        "logical_rounding": "nearest_nonnegative_integer_half_away_from_zero",
        "commissioning_read_samples": RESET_READ_SAMPLES,
    }
    for name, expected in expected_scalars.items():
        if not _same_json_value(raw[name], expected):
            raise config_error(
                f"{path}.{name}", f"to equal {expected!r}", raw[name]
            )

    spacing = raw["spacing_delta_x_multiplier"]
    if type(spacing) is not int or spacing not in SPACING_DELTA_X_MULTIPLIERS:
        raise config_error(
            f"{path}.spacing_delta_x_multiplier",
            f"to be one of {SPACING_DELTA_X_MULTIPLIERS!r}",
            spacing,
        )

    commissioning_raw = _object(raw["commissioning"], f"{path}.commissioning")
    _require_exact(
        commissioning_raw,
        _COMMISSIONING,
        path=f"{path}.commissioning",
    )
    repair_raw = _object(
        raw["joint_assignment_repair"], f"{path}.joint_assignment_repair"
    )
    _require_exact(
        repair_raw,
        _JOINT_REPAIR,
        path=f"{path}.joint_assignment_repair",
    )
    pv_raw = _object(raw["program_verify"], f"{path}.program_verify")
    _require_exact(pv_raw, _PROGRAM_VERIFY, path=f"{path}.program_verify")
    invariants_raw = _object(
        raw["physical_invariants"], f"{path}.physical_invariants"
    )
    _require_exact(
        invariants_raw,
        _PHYSICAL_INVARIANTS,
        path=f"{path}.physical_invariants",
    )
    exclusions_raw = _object(raw["exclusions"], f"{path}.exclusions")
    _require_exact(exclusions_raw, _EXCLUSIONS, path=f"{path}.exclusions")

    parity = HistoricalParityContract(
        development_hardware_instance_id=_HISTORICAL_PARITY[
            "development_hardware_instance_id"
        ],
        heldout_assignment_seeds=HELDOUT_ASSIGNMENT_SEEDS,
        heldout_hardware_instance_ids=tuple(
            _HISTORICAL_PARITY["heldout_hardware_instance_ids"]
        ),
        ideal_correct=tuple(_HISTORICAL_PARITY["ideal_correct"]),
        ideal_prediction_sha256=tuple(
            _HISTORICAL_PARITY["ideal_prediction_sha256"]
        ),
        selected_scale_fractions=tuple(
            _HISTORICAL_PARITY["selected_scale_fractions"]
        ),
        fixed_logit_gain=float(_HISTORICAL_PARITY["fixed_logit_gain"]),
    )
    return BaselineSpacingPvProtocol(
        metrics=MetricContract(**_METRICS),
        execution=ExecutionContract(profile=str(profile)),
        device=OmDeviceContract(**_DEVICE),
        source=SourceContract(**_SOURCE),
        reference_study=ReferenceStudyContract(
            study_id=REFERENCE_STUDY_ID,
            selected_policy=_REFERENCE_STUDY["selected_policy"],
            assignment_reconstruction=_REFERENCE_STUDY[
                "assignment_reconstruction"
            ],
            alpha0_spacing4_parity=parity,
        ),
        calibration=CalibrationContract(
            development_assignment_seed=DEVELOPMENT_ASSIGNMENT_SEED,
            scale_fractions=SCALE_FRACTIONS,
            selection_domain=_CALIBRATION["selection_domain"],
            selection_metric=_CALIBRATION["selection_metric"],
            logit_gain=_CALIBRATION["logit_gain"],
            sharing=_CALIBRATION["sharing"],
            heldout_application=_CALIBRATION["heldout_application"],
        ),
        assignments=AssignmentContract(
            allowed_heldout_seeds=HELDOUT_ASSIGNMENT_SEEDS,
            heldout_seed=int(heldout_seed),
            endpoint_seeds=ENDPOINT_SEEDS_BY_ASSIGNMENT[int(heldout_seed)],
        ),
        baseline_policy="shared_destination_columns",
        baseline_position_fraction=float(alpha),
        group_lower_definition=str(raw["group_lower_definition"]),
        group_upper_definition=str(raw["group_upper_definition"]),
        baseline_definition=str(raw["baseline_definition"]),
        spacing_delta_x_multiplier=int(spacing),
        delta_x_definition=str(raw["delta_x_definition"]),
        level_index_policy=str(raw["level_index_policy"]),
        level_capacity_policy=str(raw["level_capacity_policy"]),
        logical_rounding=str(raw["logical_rounding"]),
        commissioning_read_samples=RESET_READ_SAMPLES,
        commissioning=CommissioningContract(**_COMMISSIONING),
        joint_assignment_repair=JointAssignmentRepairContract(**_JOINT_REPAIR),
        program_verify=ProgramVerifyContract(**_PROGRAM_VERIFY),
        physical_invariants=PhysicalInvariantContract(
            **{
                **_PHYSICAL_INVARIANTS,
                "destination_column_groups": tuple(
                    tuple(group)
                    for group in _PHYSICAL_INVARIANTS[
                        "destination_column_groups"
                    ]
                ),
            }
        ),
        exclusions=ExclusionContract(**_EXCLUSIONS),
    )


def _validate_student_contract(
    student: StudentConfig,
    raw: Mapping[str, Any],
    *,
    profile: str,
) -> None:
    if set(student.modes) != {RunMode.VALIDATE.value}:
        raise config_error(
            "config.modes",
            "to define exactly the validate mode",
            tuple(student.modes),
        )
    _require_exact(
        _object(raw["runtime"], "config.runtime"),
        _STUDENT_RUNTIME,
        path="config.runtime",
    )
    _require_exact(
        _object(raw["data"], "config.data"),
        _STUDENT_DATA,
        path="config.data",
    )
    _require_exact(
        _object(raw["model"], "config.model"),
        _STUDENT_MODEL,
        path="config.model",
    )
    _require_exact(
        _object(raw["solver"], "config.solver"),
        _STUDENT_SOLVER,
        path="config.solver",
    )
    _require_exact(
        _object(raw["mapping"], "config.mapping"),
        _STUDENT_MAPPING,
        path="config.mapping",
    )
    _require_exact(
        _object(raw["teacher"], "config.teacher"),
        _STUDENT_TEACHER,
        path="config.teacher",
    )
    modes_raw = _object(raw["modes"], "config.modes")
    _keys(modes_raw, "config.modes", {"validate"})
    expected_validate = {
        "split": "test",
        "sample_limit": None if profile == "production" else 32,
        "noise_repeats": 1,
        "weight_modifier": {"type": "none", "parameters": {}},
    }
    _require_exact(
        _object(modes_raw["validate"], "config.modes.validate"),
        expected_validate,
        path="config.modes.validate",
    )


def parse_baseline_spacing_pv_config(
    payload: Mapping[str, Any],
) -> BaselineSpacingPvConfig:
    """Parse one immutable alpha/spacing/assignment validation config."""

    raw = _object(payload, "config")
    student_keys = {
        "schema_version",
        "experiment_id",
        "runtime",
        "data",
        "teacher",
        "model",
        "solver",
        "mapping",
        "modes",
    }
    _keys(raw, "config", student_keys | {"baseline_spacing_pv"})
    if raw["schema_version"] != SCHEMA_VERSION:
        raise config_error(
            "config.schema_version", "to equal 1", raw["schema_version"]
        )
    if raw["experiment_id"] != EXPERIMENT_ID:
        raise config_error(
            "config.experiment_id",
            f"to equal {EXPERIMENT_ID!r}",
            raw["experiment_id"],
        )
    modes_raw = _object(raw["modes"], "config.modes")
    if set(modes_raw) != {RunMode.VALIDATE.value}:
        raise config_error(
            "config.modes",
            "to define exactly the validate mode",
            tuple(modes_raw),
        )

    protocol = _parse_protocol(raw["baseline_spacing_pv"])
    student_payload = {key: raw[key] for key in student_keys}
    student_payload["experiment_id"] = STUDENT_EXPERIMENT_ID
    student = parse_student_config(student_payload)
    _validate_student_contract(student, raw, profile=protocol.execution.profile)
    return BaselineSpacingPvConfig(
        schema_version=SCHEMA_VERSION,
        experiment_id=EXPERIMENT_ID,
        protocol=protocol,
        student=student,
    )


def resolve_baseline_spacing_pv_spec(
    document: BaselineSpacingPvConfig,
    mode: RunMode,
) -> BaselineSpacingPvValidateSpec:
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
    return BaselineSpacingPvValidateSpec(
        experiment_id=EXPERIMENT_ID,
        protocol=document.protocol,
        student=student,
    )


__all__ = [
    "BASELINE_POSITION_FRACTIONS",
    "BaselineSpacingPvConfig",
    "BaselineSpacingPvProtocol",
    "BaselineSpacingPvValidateSpec",
    "CONTINUOUS_DIAGNOSTIC_METRIC_DEFINITION",
    "DEVELOPMENT_ASSIGNMENT_SEED",
    "ENDPOINT_SEEDS_BY_ASSIGNMENT",
    "EXECUTION_PROFILES",
    "EXPECTED_WEIGHTS_SHA256",
    "EXPERIMENT_ID",
    "HELDOUT_ASSIGNMENT_SEEDS",
    "IDEAL_QUANTIZED_METRIC_DEFINITION",
    "MAXIMUM_DONOR_CANDIDATES_PER_QUAD",
    "MAXIMUM_PROGRAM_PULSES",
    "P_AND_V_REPEAT_COUNT",
    "PV_PERSISTENT_METRIC_DEFINITION",
    "REFERENCE_STUDY_ID",
    "RESET_READ_SAMPLES",
    "SCALE_FRACTIONS",
    "SCHEMA_VERSION",
    "SPACING_DELTA_X_MULTIPLIERS",
    "STUDY_ID",
    "VERIFY_TOLERANCE_DELTA_X_RATIO",
    "parse_baseline_spacing_pv_config",
    "resolve_baseline_spacing_pv_spec",
]
