"""Strict validate-only contract for the IBM OM four-reference balance screen.

The experiment is a protocol wrapper around ``mnist_relu_drn_kd.v1``.  It
freezes the scientific contract and exposes the already parsed student
validation specification to the dedicated native validation runtime.
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


EXPERIMENT_ID = "ibm_om_four_reference_balance.v1"
SCHEMA_VERSION = 1
METRIC_DEFINITION = "ibm_om.reference_balanced_continuous_init.v1"

EXPECTED_WEIGHTS_SHA256 = (
    "a99995a3e5b321a56bb7e00e29840c3b76f3fee95b8d8c80fdf0fa16e93b3563"
)
EXPECTED_TEACHER_SHA256 = (
    "9a961a77628e365b54fdf59304ec7fddf46f1f4834c4c03cecea9c204f837f52"
)

DEVELOPMENT_ASSIGNMENT_SEED = 86001
HELDOUT_ASSIGNMENT_SEEDS = (87001, 87002, 87003)
BINDING_POLICIES = ("random_binding", "reference_balanced_binding_v1")


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
class ReferenceBalancedAlgorithm:
    scope: str
    sort_key: str
    quartet_grouping: str
    partition_candidates: str
    partition_objective: str
    tie_break: str
    quad_placement: str
    quad_shuffle_seed: str


@dataclass(frozen=True)
class BindingContract:
    policies: tuple[str, ...]
    reference_balanced_binding_v1: ReferenceBalancedAlgorithm


@dataclass(frozen=True)
class ContinuousMappingContract:
    coordinate: str
    active_offsets: str
    active_start: str
    quad_headroom: str
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
class ExclusionContract:
    optimizer_updates: int
    program_and_verify: bool
    hardware_aware_training: bool
    write_noise: float
    read_noise: float
    retention_drift: float


@dataclass(frozen=True)
class BalanceProtocol:
    metric_definition: str
    device: OmDeviceContract
    source: SourceContract
    assignments: AssignmentContract
    binding: BindingContract
    continuous_mapping: ContinuousMappingContract
    exclusions: ExclusionContract


@dataclass(frozen=True)
class BalanceConfig:
    schema_version: int
    experiment_id: str
    protocol: BalanceProtocol
    student: StudentConfig


@dataclass(frozen=True)
class BalanceValidateSpec:
    """Resolved balance contract with its immutable student validate spec."""

    experiment_id: str
    protocol: BalanceProtocol
    student: StudentValidateSpec


_DEVICE = {
    "evidence_class": "model_based_aihwkit_preset",
    "preset": "reram_array_om",
    "required_aihwkit_version": "1.1.0",
    "corruption_policy": "counterfactual_repaired",
}

_ALGORITHM = {
    "scope": "global_within_layer",
    "sort_key": "mapped_reference_then_original_flat_index",
    "quartet_grouping": "sorted_consecutive_quartets",
    "partition_candidates": "three_unique_two_vs_two_sign_partitions",
    "partition_objective": "minimum_absolute_signed_reference_sum_mismatch",
    "tie_break": "lexicographic_original_flat_indices",
    "quad_placement": "deterministic_quad_shuffle",
    "quad_shuffle_seed": (
        "derive_seed(assignment_seed,layer_key,reference_balanced_binding_v1)"
    ),
}

_CONTINUOUS_MAPPING = {
    "coordinate": "clip((a+1)/2,0,1)",
    "active_offsets": "positive_only",
    "active_start": "clip_reference_to_active_bounds",
    "quad_headroom": "minimum_positive_headroom_across_four_cells",
    "logical_sign": "dual_rail_role_placement",
    "physical_conductance": "G=baseline+offset",
    "transfer": "full_signed_conductance_contrast",
    "loading": "full_conductance_sum",
    "quantization": "none",
    "four_delta_spacing": None,
    "pulse_cap": None,
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
) -> BalanceProtocol:
    path = "config.reference_balance"
    raw = _object(value, path)
    _keys(
        raw,
        path,
        {
            "metric_definition",
            "device",
            "source",
            "assignments",
            "binding",
            "continuous_mapping",
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
    _require_exact(
        assignments_raw,
        expected_assignments,
        path=assignments_path,
    )

    binding_path = f"{path}.binding"
    binding_raw = _object(raw["binding"], binding_path)
    _keys(
        binding_raw,
        binding_path,
        {"policies", "reference_balanced_binding_v1"},
    )
    if binding_raw["policies"] != list(BINDING_POLICIES):
        raise config_error(
            f"{binding_path}.policies",
            f"to equal {list(BINDING_POLICIES)!r}",
            binding_raw["policies"],
        )
    algorithm_path = f"{binding_path}.reference_balanced_binding_v1"
    algorithm_raw = _object(
        binding_raw["reference_balanced_binding_v1"], algorithm_path
    )
    _require_exact(algorithm_raw, _ALGORITHM, path=algorithm_path)

    mapping_path = f"{path}.continuous_mapping"
    mapping_raw = _object(raw["continuous_mapping"], mapping_path)
    _keys(
        mapping_raw,
        mapping_path,
        set(_CONTINUOUS_MAPPING) | {"scale_fraction_source", "scale_fractions"},
    )
    for name, expected_value in _CONTINUOUS_MAPPING.items():
        if mapping_raw[name] != expected_value:
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

    exclusions_raw = _object(raw["exclusions"], f"{path}.exclusions")
    _require_exact(exclusions_raw, _EXCLUSIONS, path=f"{path}.exclusions")

    return BalanceProtocol(
        metric_definition=METRIC_DEFINITION,
        device=OmDeviceContract(**_DEVICE),
        source=SourceContract(**expected_source),
        assignments=AssignmentContract(
            development_seed=DEVELOPMENT_ASSIGNMENT_SEED,
            heldout_seeds=HELDOUT_ASSIGNMENT_SEEDS,
        ),
        binding=BindingContract(
            policies=BINDING_POLICIES,
            reference_balanced_binding_v1=ReferenceBalancedAlgorithm(
                **_ALGORITHM
            ),
        ),
        continuous_mapping=ContinuousMappingContract(
            **_CONTINUOUS_MAPPING,
            scale_fraction_source=scale_source,
            scale_fractions=tuple(float(value) for value in resolved_fractions),
        ),
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
            "to be null so the declared scale-fraction grid is inherited",
            student.mapping.scale_fraction_pairs,
        )
    if student.mapping.range_placement != "lower":
        raise config_error(
            "config.mapping.range_placement",
            "to equal 'lower'",
            student.mapping.range_placement,
        )


def parse_balance_config(payload: Mapping[str, Any]) -> BalanceConfig:
    """Parse the strict balance wrapper used by the registered runtime."""

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
    _keys(raw, "config", student_keys | {"reference_balance"})
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
    protocol = _parse_protocol(raw["reference_balance"], student=student)
    return BalanceConfig(
        schema_version=SCHEMA_VERSION,
        experiment_id=EXPERIMENT_ID,
        protocol=protocol,
        student=student,
    )


def resolve_balance_spec(
    document: BalanceConfig,
    mode: RunMode,
) -> BalanceValidateSpec:
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
    return BalanceValidateSpec(
        experiment_id=EXPERIMENT_ID,
        protocol=document.protocol,
        student=student,
    )


# Explicit long-form aliases keep imports descriptive at the registry boundary.
parse_four_reference_balance_config = parse_balance_config
resolve_four_reference_balance_spec = resolve_balance_spec


__all__ = [
    "BINDING_POLICIES",
    "BalanceConfig",
    "BalanceProtocol",
    "BalanceValidateSpec",
    "DEVELOPMENT_ASSIGNMENT_SEED",
    "EXPECTED_TEACHER_SHA256",
    "EXPECTED_WEIGHTS_SHA256",
    "EXPERIMENT_ID",
    "HELDOUT_ASSIGNMENT_SEEDS",
    "METRIC_DEFINITION",
    "SCHEMA_VERSION",
    "parse_balance_config",
    "parse_four_reference_balance_config",
    "resolve_balance_spec",
    "resolve_four_reference_balance_spec",
]
