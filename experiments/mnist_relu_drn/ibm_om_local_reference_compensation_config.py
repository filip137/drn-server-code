"""Strict validate-only contract for local IBM OM reference compensation.

The experiment wraps ``mnist_relu_drn_kd.v1`` while freezing the scientific
intervention: sampled device identities stay at their original quad addresses
and only the stored zero baseline is changed.  The control clips the mapped
intrinsic symmetry point to the active interval.  The treatment computes the
nearest in-bounds baseline whose four-rail signed contrast is exactly zero.
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


EXPERIMENT_ID = "ibm_om_local_reference_compensation.v1"
SCHEMA_VERSION = 1
METRIC_DEFINITION = "ibm_om.local_reference_compensated_continuous_init.v1"

EXPECTED_WEIGHTS_SHA256 = (
    "a99995a3e5b321a56bb7e00e29840c3b76f3fee95b8d8c80fdf0fa16e93b3563"
)
EXPECTED_TEACHER_SHA256 = (
    "9a961a77628e365b54fdf59304ec7fddf46f1f4834c4c03cecea9c204f837f52"
)

DEVELOPMENT_ASSIGNMENT_SEED = 86001
HELDOUT_ASSIGNMENT_SEEDS = (87001, 87002, 87003)
BASELINE_POLICIES = (
    "local_nearest_symmetry",
    "local_min_l2_exact_zero",
)


@dataclass(frozen=True)
class OmDeviceContract:
    evidence_class: str
    preset: str
    required_aihwkit_version: str
    corruption_policy: str


@dataclass(frozen=True)
class SourceContract:
    expected_weights_sha256: str
    expected_teacher_sha256: str


@dataclass(frozen=True)
class AssignmentContract:
    development_seed: int
    heldout_seeds: tuple[int, ...]


@dataclass(frozen=True)
class IdentityBindingContract:
    policy: str
    scope: str
    reassignment: str
    identity_order: str


@dataclass(frozen=True)
class NearestSymmetryAlgorithm:
    mapped_reference: str
    objective: str
    constraints: str


@dataclass(frozen=True)
class ExactZeroAlgorithm:
    scope: str
    sign_order: tuple[str, ...]
    objective: str
    constraints: tuple[str, ...]
    solver: str
    intrinsic_reference_mutation: bool


@dataclass(frozen=True)
class BaselineContract:
    policies: tuple[str, ...]
    local_nearest_symmetry: NearestSymmetryAlgorithm
    local_min_l2_exact_zero: ExactZeroAlgorithm


@dataclass(frozen=True)
class ContinuousMappingContract:
    coordinate: str
    active_offsets: str
    matched_headroom: str
    identical_offsets_across_policies: bool
    logical_sign: str
    physical_conductance: str
    transfer: str
    loading: str
    quantization: str
    four_delta_spacing: None
    pulse_cap: None
    scale_fraction_source: str
    scale_fractions: tuple[float, ...]


@dataclass(frozen=True)
class CalibrationContract:
    development_population: str
    scale_selection: str
    logit_gain: str
    primary_control: str
    primary_treatment: str
    primary_comparison: str
    secondary_matrix: str
    secondary_comparison: str


@dataclass(frozen=True)
class ExclusionContract:
    optimizer_updates: int
    program_and_verify: bool
    hardware_aware_training: bool
    write_noise: float
    read_noise: float
    retention_drift: float


@dataclass(frozen=True)
class LocalCompensationProtocol:
    metric_definition: str
    device: OmDeviceContract
    source: SourceContract
    assignments: AssignmentContract
    identity_binding: IdentityBindingContract
    baselines: BaselineContract
    continuous_mapping: ContinuousMappingContract
    calibration: CalibrationContract
    exclusions: ExclusionContract


@dataclass(frozen=True)
class LocalCompensationConfig:
    schema_version: int
    experiment_id: str
    protocol: LocalCompensationProtocol
    student: StudentConfig


@dataclass(frozen=True)
class LocalCompensationValidateSpec:
    experiment_id: str
    protocol: LocalCompensationProtocol
    student: StudentValidateSpec


_DEVICE = {
    "evidence_class": "model_based_aihwkit_preset",
    "preset": "reram_array_om",
    "required_aihwkit_version": "1.1.0",
    "corruption_policy": "counterfactual_repaired",
}

_IDENTITY_BINDING = {
    "policy": "sampled_order",
    "scope": "fixed_existing_four_cell_quads",
    "reassignment": "none",
    "identity_order": "preserve_sampled_flat_order",
}

_NEAREST_SYMMETRY = {
    "mapped_reference": "rho=clip((r+1)/2,0,1)",
    "objective": "B_i=clip(rho_i,[l_i,u_i])",
    "constraints": "l_i<=B_i<=u_i",
}

_EXACT_ZERO = {
    "scope": "each_existing_four_cell_quad",
    "sign_order": ["++", "+-", "-+", "--"],
    "objective": "argmin_B 0.5*sum_i((B_i-rho_i)^2)",
    "constraints": [
        "l_i<=B_i<=u_i",
        "B_++-B_+--B_-++B_--=0",
    ],
    "solver": "monotone_clipped_lagrange_bisection_v1",
    "intrinsic_reference_mutation": False,
}

_CONTINUOUS_MAPPING = {
    "coordinate": "rho=clip((r+1)/2,0,1)",
    "active_offsets": "positive_only",
    "matched_headroom": (
        "minimum_positive_headroom_across_both_baseline_policies_and_four_cells"
    ),
    "identical_offsets_across_policies": True,
    "logical_sign": "dual_rail_role_placement",
    "physical_conductance": "G=B+d",
    "transfer": "full_signed_conductance_contrast",
    "loading": "full_conductance_sum",
    "quantization": "none",
    "four_delta_spacing": None,
    "pulse_cap": None,
}

_CALIBRATION = {
    "development_population": "assignment_86001_only",
    "scale_selection": (
        "maximize_accuracy_then_minimize_calibrated_kl_then_lexicographic_pair"
    ),
    "logit_gain": "positive_logit_gain_grid",
    "primary_control": "local_nearest_symmetry@cal_local_nearest_symmetry",
    "primary_treatment": (
        "local_min_l2_exact_zero@cal_local_nearest_symmetry"
    ),
    "primary_comparison": "primary_treatment-minus-primary_control",
    "secondary_matrix": "all_two_by_two_policy_by_calibration_policy",
    "secondary_comparison": (
        "local_min_l2_exact_zero@cal_local_min_l2_exact_zero-minus-"
        "local_nearest_symmetry@cal_local_nearest_symmetry"
    ),
}

_EXCLUSIONS = {
    "optimizer_updates": 0,
    "program_and_verify": False,
    "hardware_aware_training": False,
    "write_noise": 0.0,
    "read_noise": 0.0,
    "retention_drift": 0.0,
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


def _parse_protocol(
    value: Any,
    *,
    student: StudentConfig,
) -> LocalCompensationProtocol:
    path = "config.local_reference_compensation"
    raw = _object(value, path)
    _keys(
        raw,
        path,
        {
            "metric_definition",
            "device",
            "source",
            "assignments",
            "identity_binding",
            "baselines",
            "continuous_mapping",
            "calibration",
            "exclusions",
        },
    )
    if raw["metric_definition"] != METRIC_DEFINITION:
        raise config_error(
            f"{path}.metric_definition",
            f"to equal {METRIC_DEFINITION!r}",
            raw["metric_definition"],
        )

    device_raw = _object(raw["device"], f"{path}.device")
    _require_exact(device_raw, _DEVICE, path=f"{path}.device")

    source_path = f"{path}.source"
    source_raw = _object(raw["source"], source_path)
    expected_source = {
        "expected_weights_sha256": EXPECTED_WEIGHTS_SHA256,
        "expected_teacher_sha256": EXPECTED_TEACHER_SHA256,
    }
    _require_exact(source_raw, expected_source, path=source_path)

    assignments_path = f"{path}.assignments"
    assignments_raw = _object(raw["assignments"], assignments_path)
    expected_assignments = {
        "development_seed": DEVELOPMENT_ASSIGNMENT_SEED,
        "heldout_seeds": list(HELDOUT_ASSIGNMENT_SEEDS),
    }
    _require_exact(assignments_raw, expected_assignments, path=assignments_path)

    identity_path = f"{path}.identity_binding"
    identity_raw = _object(raw["identity_binding"], identity_path)
    _require_exact(identity_raw, _IDENTITY_BINDING, path=identity_path)

    baselines_path = f"{path}.baselines"
    baselines_raw = _object(raw["baselines"], baselines_path)
    _keys(
        baselines_raw,
        baselines_path,
        {
            "policies",
            "local_nearest_symmetry",
            "local_min_l2_exact_zero",
        },
    )
    if baselines_raw["policies"] != list(BASELINE_POLICIES):
        raise config_error(
            f"{baselines_path}.policies",
            f"to equal {list(BASELINE_POLICIES)!r}",
            baselines_raw["policies"],
        )
    nearest_path = f"{baselines_path}.local_nearest_symmetry"
    nearest_raw = _object(baselines_raw["local_nearest_symmetry"], nearest_path)
    _require_exact(nearest_raw, _NEAREST_SYMMETRY, path=nearest_path)
    exact_path = f"{baselines_path}.local_min_l2_exact_zero"
    exact_raw = _object(baselines_raw["local_min_l2_exact_zero"], exact_path)
    _require_exact(exact_raw, _EXACT_ZERO, path=exact_path)

    mapping_path = f"{path}.continuous_mapping"
    mapping_raw = _object(raw["continuous_mapping"], mapping_path)
    _keys(
        mapping_raw,
        mapping_path,
        set(_CONTINUOUS_MAPPING) | {"scale_fraction_source", "scale_fractions"},
    )
    for name, expected_value in _CONTINUOUS_MAPPING.items():
        if not _same_json_value(mapping_raw[name], expected_value):
            raise config_error(
                f"{mapping_path}.{name}",
                f"to equal {expected_value!r}",
                mapping_raw[name],
            )
    scale_source = mapping_raw["scale_fraction_source"]
    supplied_fractions = mapping_raw["scale_fractions"]
    inherited_fractions = student.mapping.scale_fractions
    if scale_source == "inherit_student_mapping":
        if supplied_fractions is not None:
            raise config_error(
                f"{mapping_path}.scale_fractions",
                "to be null when scale_fraction_source is "
                "'inherit_student_mapping'",
                supplied_fractions,
            )
        resolved_fractions = inherited_fractions
    elif scale_source == "explicit_frozen":
        if not isinstance(supplied_fractions, (list, tuple)):
            raise config_error(
                f"{mapping_path}.scale_fractions",
                "to equal the student mapping scale fractions",
                supplied_fractions,
            )
        resolved_fractions = tuple(supplied_fractions)
        if resolved_fractions != inherited_fractions:
            raise config_error(
                f"{mapping_path}.scale_fractions",
                f"to equal {list(inherited_fractions)!r}",
                supplied_fractions,
            )
    else:
        raise config_error(
            f"{mapping_path}.scale_fraction_source",
            "to be 'inherit_student_mapping' or 'explicit_frozen'",
            scale_source,
        )

    calibration_path = f"{path}.calibration"
    calibration_raw = _object(raw["calibration"], calibration_path)
    _require_exact(calibration_raw, _CALIBRATION, path=calibration_path)

    exclusions_path = f"{path}.exclusions"
    exclusions_raw = _object(raw["exclusions"], exclusions_path)
    _require_exact(exclusions_raw, _EXCLUSIONS, path=exclusions_path)

    return LocalCompensationProtocol(
        metric_definition=METRIC_DEFINITION,
        device=OmDeviceContract(**_DEVICE),
        source=SourceContract(**expected_source),
        assignments=AssignmentContract(
            development_seed=DEVELOPMENT_ASSIGNMENT_SEED,
            heldout_seeds=HELDOUT_ASSIGNMENT_SEEDS,
        ),
        identity_binding=IdentityBindingContract(**_IDENTITY_BINDING),
        baselines=BaselineContract(
            policies=BASELINE_POLICIES,
            local_nearest_symmetry=NearestSymmetryAlgorithm(**_NEAREST_SYMMETRY),
            local_min_l2_exact_zero=ExactZeroAlgorithm(
                **{
                    **_EXACT_ZERO,
                    "sign_order": tuple(_EXACT_ZERO["sign_order"]),
                    "constraints": tuple(_EXACT_ZERO["constraints"]),
                }
            ),
        ),
        continuous_mapping=ContinuousMappingContract(
            **_CONTINUOUS_MAPPING,
            scale_fraction_source=scale_source,
            scale_fractions=tuple(float(value) for value in resolved_fractions),
        ),
        calibration=CalibrationContract(**_CALIBRATION),
        exclusions=ExclusionContract(**_EXCLUSIONS),
    )


def _validate_student_contract(student: StudentConfig, raw: Mapping[str, Any]) -> None:
    if set(student.modes) != {RunMode.VALIDATE.value}:
        raise config_error(
            "config.modes",
            "to define exactly the validate mode",
            tuple(student.modes),
        )
    validate_raw = _object(raw["modes"]["validate"], "config.modes.validate")
    _keys(
        validate_raw,
        "config.modes.validate",
        {"split", "sample_limit", "weight_modifier", "noise_repeats"},
    )
    if validate_raw["split"] != "test":
        raise config_error(
            "config.modes.validate.split", "to equal 'test'", validate_raw["split"]
        )
    if validate_raw["sample_limit"] is not None:
        raise config_error(
            "config.modes.validate.sample_limit",
            "to be null",
            validate_raw["sample_limit"],
        )
    if validate_raw["noise_repeats"] != 1:
        raise config_error(
            "config.modes.validate.noise_repeats",
            "to equal 1",
            validate_raw["noise_repeats"],
        )
    modifier_raw = _object(
        validate_raw["weight_modifier"],
        "config.modes.validate.weight_modifier",
    )
    _require_exact(
        modifier_raw,
        {"type": "none", "parameters": {}},
        path="config.modes.validate.weight_modifier",
    )
    if student.model.encoding != "single":
        raise config_error(
            "config.model.encoding",
            "to equal 'single' for four devices per logical weight",
            student.model.encoding,
        )
    if student.model.include_biases:
        raise config_error(
            "config.model.include_biases", "to be false", student.model.include_biases
        )
    if student.model.non_linearity != "perfect_diode":
        raise config_error(
            "config.model.non_linearity.type",
            "to equal 'perfect_diode'",
            student.model.non_linearity,
        )
    if student.teacher.type != "bias_free_relu":
        raise config_error(
            "config.teacher.type", "to equal 'bias_free_relu'", student.teacher.type
        )
    if student.mapping.scale_fraction_pairs is not None:
        raise config_error(
            "config.mapping.scale_fraction_pairs",
            "to be null so the scale grid is inherited",
            student.mapping.scale_fraction_pairs,
        )
    if student.mapping.range_placement != "lower":
        raise config_error(
            "config.mapping.range_placement",
            "to equal 'lower'",
            student.mapping.range_placement,
        )


def parse_local_reference_compensation_config(
    payload: Mapping[str, Any],
) -> LocalCompensationConfig:
    """Parse the strict local-reference compensation wrapper."""

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
    _keys(raw, "config", student_keys | {"local_reference_compensation"})
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

    student_payload = {key: raw[key] for key in student_keys}
    student_payload["experiment_id"] = STUDENT_EXPERIMENT_ID
    student = parse_student_config(student_payload)
    _validate_student_contract(student, raw)
    protocol = _parse_protocol(raw["local_reference_compensation"], student=student)
    return LocalCompensationConfig(
        schema_version=SCHEMA_VERSION,
        experiment_id=EXPERIMENT_ID,
        protocol=protocol,
        student=student,
    )


def resolve_local_reference_compensation_spec(
    document: LocalCompensationConfig,
    mode: RunMode,
) -> LocalCompensationValidateSpec:
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
    return LocalCompensationValidateSpec(
        experiment_id=EXPERIMENT_ID,
        protocol=document.protocol,
        student=student,
    )


__all__ = [
    "BASELINE_POLICIES",
    "DEVELOPMENT_ASSIGNMENT_SEED",
    "EXPECTED_TEACHER_SHA256",
    "EXPECTED_WEIGHTS_SHA256",
    "EXPERIMENT_ID",
    "HELDOUT_ASSIGNMENT_SEEDS",
    "LocalCompensationConfig",
    "LocalCompensationProtocol",
    "LocalCompensationValidateSpec",
    "METRIC_DEFINITION",
    "SCHEMA_VERSION",
    "parse_local_reference_compensation_config",
    "resolve_local_reference_compensation_spec",
]
