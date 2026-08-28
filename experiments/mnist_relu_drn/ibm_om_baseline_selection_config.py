"""Strict validate-only contract for four-device IBM OM baseline selection.

This experiment wraps the ordinary MNIST ReLU-to-DRN validation surface but
freezes a narrower ideal-initialization study.  One config selects one
baseline policy and one held-out OM assignment.  The source supplied through
``--weights`` is the frozen bias-free ReLU checkpoint itself; no trained DRN,
device model, programming controller, or optimizer state participates.
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


EXPERIMENT_ID = "mnist_ibm_om_baseline_selection.v1"
SCHEMA_VERSION = 1

CONTINUOUS_METRIC_DEFINITION = "ibm_om.ideal_bounded_continuous_init.v1"
STANDARD4DELTA_METRIC_DEFINITION = (
    "ibm_om.ideal_bounded_standard4delta_init.v1"
)
EXPECTED_WEIGHTS_SHA256 = (
    "9a961a77628e365b54fdf59304ec7fddf46f1f4834c4c03cecea9c204f837f52"
)
DEVELOPMENT_ASSIGNMENT_SEED = 86001
HELDOUT_ASSIGNMENT_SEEDS = (87001, 87002, 87003)
BASELINE_POLICIES = (
    "independent_cell_reset_mean",
    "shared_quad_reset_max",
    "shared_destination_columns_reset_max",
    "reference_enforced_destination_columns",
)
EXECUTION_PROFILES = ("production", "smoke")
RESET_READ_SAMPLES = 8
SCALE_FRACTIONS = (0.125, 0.25, 0.5, 1.0)
SPACING_DELTA_MULTIPLES = 4
MAXIMUM_DONOR_CANDIDATES_PER_QUAD = 128


@dataclass(frozen=True)
class MetricContract:
    primary: str
    diagnostic: str


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
class AssignmentContract:
    development_seed: int
    allowed_heldout_seeds: tuple[int, ...]
    heldout_seed: int


@dataclass(frozen=True)
class BaselinePolicyContract:
    policies: tuple[str, ...]
    selected_policy: str
    conductance_representation: str
    independent_cell_reset_mean: str
    shared_quad_reset_max: str
    shared_destination_columns_reset_max: str
    destination_column_groups: tuple[tuple[str, str], ...]
    reference_enforced_destination_columns: str
    reference_role_assignment: str
    exact_zero_requirement: str


@dataclass(frozen=True)
class ResetCommissioningContract:
    candidates: str
    samples_per_cell: int
    sample_sequence: str
    read_coordinate: str
    baseline_estimator: str
    cross_cell_pooling: str
    standard_error_guard: float
    bound_policy: str
    commissioning_noise: str
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
class MappingContract:
    scale_fractions: tuple[float, ...]
    selection_domain: str
    selection_metric: str
    logit_gain: str
    heldout_application: str
    logical_normalization: str
    logical_sign: str
    active_offsets: str
    active_headroom: str
    continuous_target: str
    delta_definition: str
    spacing_delta_multiples: int
    logical_rounding: str
    standard4delta_refit: bool
    physical_conductance: str
    transfer: str
    loading: str


@dataclass(frozen=True)
class ExclusionContract:
    optimizer_updates: int
    program_and_verify: bool
    hardware_aware_training: bool
    target_write_noise: float
    inference_read_noise: float
    retention_drift: float
    post_deployment_recovery: bool


@dataclass(frozen=True)
class BaselineSelectionProtocol:
    metrics: MetricContract
    execution: ExecutionContract
    device: OmDeviceContract
    source: SourceContract
    assignments: AssignmentContract
    baseline: BaselinePolicyContract
    reset_commissioning: ResetCommissioningContract
    joint_assignment_repair: JointAssignmentRepairContract
    mapping: MappingContract
    exclusions: ExclusionContract


@dataclass(frozen=True)
class BaselineSelectionConfig:
    schema_version: int
    experiment_id: str
    protocol: BaselineSelectionProtocol
    student: StudentConfig


@dataclass(frozen=True)
class BaselineSelectionValidateSpec:
    experiment_id: str
    protocol: BaselineSelectionProtocol
    student: StudentValidateSpec


_METRICS = {
    "primary": CONTINUOUS_METRIC_DEFINITION,
    "diagnostic": STANDARD4DELTA_METRIC_DEFINITION,
}

_DEVICE = {
    "evidence_class": "model_based_aihwkit_preset",
    "preset": "reram_array_om",
    "required_aihwkit_version": "1.1.0",
    "corruption_policy": "counterfactual_repaired",
    "conductance_coordinate": "x=clip((a+1)/2,0,1)",
}

_SOURCE = {
    "weights_role": "frozen_relu_source_and_teacher",
    "expected_weights_sha256": EXPECTED_WEIGHTS_SHA256,
}

_BASELINE = {
    "conductance_representation": "G=B+d",
    "independent_cell_reset_mean": "one_bounded_commissioned_mean_per_cell",
    "shared_quad_reset_max": "maximum_of_four_bounded_commissioned_means",
    "shared_destination_columns_reset_max": (
        "maximum_bounded_commissioned_mean_within_each_destination_column"
    ),
    "destination_column_groups": [["G++", "G-+"], ["G+-", "G--"]],
    "reference_enforced_destination_columns": (
        "active_cell_baseline_equals_partner_intrinsic_reference_in_same_destination_column"
    ),
    "reference_role_assignment": (
        "positive_diagonal_active_negative_off_diagonal_active"
    ),
    "exact_zero_requirement": (
        "shared_quad_destination_column_and_reference_policies_have_C0_equal_zero"
    ),
}

_RESET_COMMISSIONING = {
    "candidates": "primary_and_donor_populations_before_joint_assignment",
    "samples_per_cell": RESET_READ_SAMPLES,
    "sample_sequence": (
        "initialize_at_sampled_lower_bound_then_one_reset_pulse_and_apparent_read_per_sample"
    ),
    "read_coordinate": "raw_active_a_before_conductance_mapping",
    "baseline_estimator": "per_cell_arithmetic_mean",
    "cross_cell_pooling": "none_before_policy_baseline_selection",
    "standard_error_guard": 0.0,
    "bound_policy": "clip_mapped_mean_to_sampled_active_conductance_bounds",
    "commissioning_noise": "preset_cycle_to_cycle_and_apparent_write_noise_enabled",
    "freeze_policy": (
        "splice_identity_and_its_commissioned_observations_then_reuse_for_every_scale_candidate"
    ),
}

_JOINT_REPAIR = {
    "scope": "one_shared_assignment_for_all_four_baseline_policies",
    "donor_seed_derivation": (
        "derive_seed(assignment_seed,mnist_ibm_om_baseline_selection_joint_donor_v1)"
    ),
    "destination_traversal": "canonical_layer_then_logical_quad_order",
    "donor_traversal": "same_layer_unique_donor_quads_in_canonical_order",
    "eligibility": (
        "feasible_for_all_four_policies_after_source_sign_roles_and_reset_commissioning"
    ),
    "identity_copy": "copy_complete_four_cell_quad_and_commissioned_observations",
    "maximum_candidates_per_quad": MAXIMUM_DONOR_CANDIDATES_PER_QUAD,
    "exhaustion": "invalidate_assignment",
}

_MAPPING = {
    "scale_fractions": list(SCALE_FRACTIONS),
    "selection_domain": "development_assignment_86001_calibration_subset",
    "selection_metric": (
        "continuous_accuracy_then_calibrated_kl_then_lexicographic_scale_pair"
    ),
    "logit_gain": "positive_kl_fit_on_selected_continuous_mapping",
    "heldout_application": "freeze_selected_scale_pair_and_gain_per_policy",
    "logical_normalization": "U=W/layer_absmax",
    "logical_sign": "dual_rail_role_placement",
    "active_offsets": "nonnegative_two_sign_selected_active_cells",
    "active_headroom": "minimum_across_two_sign_selected_active_cells",
    "continuous_target": "fraction_times_active_headroom_times_abs_U",
    "delta_definition": "nominal_dw_min_over_two_in_unit_x_coordinate",
    "spacing_delta_multiples": SPACING_DELTA_MULTIPLES,
    "logical_rounding": "nearest_nonnegative_integer_half_away_from_zero",
    "standard4delta_refit": False,
    "physical_conductance": "G=B+d",
    "transfer": "full_signed_four_conductance_contrast",
    "loading": "full_conductance_sum",
}

_EXCLUSIONS = {
    "optimizer_updates": 0,
    "program_and_verify": False,
    "hardware_aware_training": False,
    "target_write_noise": 0.0,
    "inference_read_noise": 0.0,
    "retention_drift": 0.0,
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


def _parse_protocol(value: Any) -> BaselineSelectionProtocol:
    path = "config.baseline_selection"
    raw = _object(value, path)
    _keys(
        raw,
        path,
        {
            "metrics",
            "execution",
            "device",
            "source",
            "assignments",
            "baseline",
            "reset_commissioning",
            "joint_assignment_repair",
            "mapping",
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

    assignments_path = f"{path}.assignments"
    assignments_raw = _object(raw["assignments"], assignments_path)
    _keys(
        assignments_raw,
        assignments_path,
        {"development_seed", "allowed_heldout_seeds", "heldout_seed"},
    )
    expected_assignments = {
        "development_seed": DEVELOPMENT_ASSIGNMENT_SEED,
        "allowed_heldout_seeds": list(HELDOUT_ASSIGNMENT_SEEDS),
    }
    for name, expected in expected_assignments.items():
        if not _same_json_value(assignments_raw[name], expected):
            raise config_error(
                f"{assignments_path}.{name}", f"to equal {expected!r}", assignments_raw[name]
            )
    heldout_seed = assignments_raw["heldout_seed"]
    if type(heldout_seed) is not int or heldout_seed not in HELDOUT_ASSIGNMENT_SEEDS:
        raise config_error(
            f"{assignments_path}.heldout_seed",
            f"to be one of {HELDOUT_ASSIGNMENT_SEEDS!r}",
            heldout_seed,
        )

    baseline_path = f"{path}.baseline"
    baseline_raw = _object(raw["baseline"], baseline_path)
    _keys(
        baseline_raw,
        baseline_path,
        {"policies", "selected_policy"} | set(_BASELINE),
    )
    if not _same_json_value(baseline_raw["policies"], list(BASELINE_POLICIES)):
        raise config_error(
            f"{baseline_path}.policies",
            f"to equal {list(BASELINE_POLICIES)!r}",
            baseline_raw["policies"],
        )
    selected_policy = baseline_raw["selected_policy"]
    if selected_policy not in BASELINE_POLICIES:
        raise config_error(
            f"{baseline_path}.selected_policy",
            f"to be one of {BASELINE_POLICIES!r}",
            selected_policy,
        )
    for name, expected in _BASELINE.items():
        if not _same_json_value(baseline_raw[name], expected):
            raise config_error(
                f"{baseline_path}.{name}", f"to equal {expected!r}", baseline_raw[name]
            )

    reset_raw = _object(raw["reset_commissioning"], f"{path}.reset_commissioning")
    _require_exact(
        reset_raw,
        _RESET_COMMISSIONING,
        path=f"{path}.reset_commissioning",
    )
    repair_raw = _object(
        raw["joint_assignment_repair"], f"{path}.joint_assignment_repair"
    )
    _require_exact(
        repair_raw,
        _JOINT_REPAIR,
        path=f"{path}.joint_assignment_repair",
    )
    mapping_raw = _object(raw["mapping"], f"{path}.mapping")
    _require_exact(mapping_raw, _MAPPING, path=f"{path}.mapping")
    exclusions_raw = _object(raw["exclusions"], f"{path}.exclusions")
    _require_exact(exclusions_raw, _EXCLUSIONS, path=f"{path}.exclusions")

    return BaselineSelectionProtocol(
        metrics=MetricContract(**_METRICS),
        execution=ExecutionContract(profile=str(profile)),
        device=OmDeviceContract(**_DEVICE),
        source=SourceContract(**_SOURCE),
        assignments=AssignmentContract(
            development_seed=DEVELOPMENT_ASSIGNMENT_SEED,
            allowed_heldout_seeds=HELDOUT_ASSIGNMENT_SEEDS,
            heldout_seed=int(heldout_seed),
        ),
        baseline=BaselinePolicyContract(
            policies=BASELINE_POLICIES,
            selected_policy=str(selected_policy),
            conductance_representation=_BASELINE["conductance_representation"],
            independent_cell_reset_mean=_BASELINE[
                "independent_cell_reset_mean"
            ],
            shared_quad_reset_max=_BASELINE["shared_quad_reset_max"],
            shared_destination_columns_reset_max=_BASELINE[
                "shared_destination_columns_reset_max"
            ],
            destination_column_groups=tuple(
                tuple(group) for group in _BASELINE["destination_column_groups"]
            ),
            reference_enforced_destination_columns=_BASELINE[
                "reference_enforced_destination_columns"
            ],
            reference_role_assignment=_BASELINE["reference_role_assignment"],
            exact_zero_requirement=_BASELINE["exact_zero_requirement"],
        ),
        reset_commissioning=ResetCommissioningContract(**_RESET_COMMISSIONING),
        joint_assignment_repair=JointAssignmentRepairContract(**_JOINT_REPAIR),
        mapping=MappingContract(
            **{
                **_MAPPING,
                "scale_fractions": SCALE_FRACTIONS,
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
    expected_runtime = {
        **_STUDENT_RUNTIME,
        "device": "cuda",
    }
    _require_exact(
        _object(raw["runtime"], "config.runtime"),
        expected_runtime,
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


def parse_baseline_selection_config(
    payload: Mapping[str, Any],
) -> BaselineSelectionConfig:
    """Parse one immutable policy/assignment validation config."""

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
    _keys(raw, "config", student_keys | {"baseline_selection"})
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

    protocol = _parse_protocol(raw["baseline_selection"])
    student_payload = {key: raw[key] for key in student_keys}
    student_payload["experiment_id"] = STUDENT_EXPERIMENT_ID
    student = parse_student_config(student_payload)
    _validate_student_contract(
        student,
        raw,
        profile=protocol.execution.profile,
    )
    return BaselineSelectionConfig(
        schema_version=SCHEMA_VERSION,
        experiment_id=EXPERIMENT_ID,
        protocol=protocol,
        student=student,
    )


def resolve_baseline_selection_spec(
    document: BaselineSelectionConfig,
    mode: RunMode,
) -> BaselineSelectionValidateSpec:
    """Resolve only validation and retain the nested parent student spec."""

    if mode is not RunMode.VALIDATE:
        raise config_error(
            "the requested run mode",
            "to be 'validate' for this validate-only experiment",
            mode.value,
        )
    student = resolve_student_spec(document.student, mode)
    if not isinstance(student, StudentValidateSpec):  # pragma: no cover
        raise RuntimeError("Expected the parent parser to resolve validation.")
    return BaselineSelectionValidateSpec(
        experiment_id=EXPERIMENT_ID,
        protocol=document.protocol,
        student=student,
    )


__all__ = [
    "BASELINE_POLICIES",
    "BaselineSelectionConfig",
    "BaselineSelectionProtocol",
    "BaselineSelectionValidateSpec",
    "CONTINUOUS_METRIC_DEFINITION",
    "DEVELOPMENT_ASSIGNMENT_SEED",
    "EXECUTION_PROFILES",
    "EXPECTED_WEIGHTS_SHA256",
    "EXPERIMENT_ID",
    "HELDOUT_ASSIGNMENT_SEEDS",
    "MAXIMUM_DONOR_CANDIDATES_PER_QUAD",
    "RESET_READ_SAMPLES",
    "SCALE_FRACTIONS",
    "SCHEMA_VERSION",
    "SPACING_DELTA_MULTIPLES",
    "STANDARD4DELTA_METRIC_DEFINITION",
    "parse_baseline_selection_config",
    "resolve_baseline_selection_spec",
]
