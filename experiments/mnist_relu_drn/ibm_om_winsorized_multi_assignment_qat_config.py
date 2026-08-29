"""Strict exploratory config for multi-assignment Winsorized IBM OM QAT."""

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
from experiments.mnist_relu_drn.ibm_om_winsorized_qat_config import (
    COMMON_SOURCE_WEIGHT_SHA256,
    EXPECTED_TEACHER_WEIGHTS_PATH,
    EXPECTED_TEACHER_WEIGHTS_SHA256,
    FIXED_LOGIT_GAIN,
    FIXED_SCALE_FRACTIONS,
    LEARNING_RATES,
    SPACING_DELTA_X_MULTIPLIERS,
    _COMMISSIONING,
    _DEVICE,
    _JOINT_ASSIGNMENT_REPAIR,
    _MAPPING_COMMON,
    _SOURCE,
    _STUDENT_DATA,
    _STUDENT_MAPPING,
    _STUDENT_MODEL,
    _STUDENT_RUNTIME,
    _STUDENT_SOLVER,
    _STUDENT_TEACHER,
    _STUDENT_TRAIN,
    _require_exact,
    _same_json_value,
    _validate_student_contract,
)
from experiments.schema import RunMode, config_error


EXPERIMENT_ID = "mnist_ibm_om_winsorized_multi_assignment_qat.v1"
SCHEMA_VERSION = 1
NUM_EPOCHS = 10
TRAINING_ASSIGNMENT_SEEDS = (86001, 87001)
DEVELOPMENT_ASSIGNMENT_SEED = 87002
FINAL_ASSIGNMENT_SEED = 87003
FINAL_ENDPOINT_SEEDS = (89301, 89302, 89303, 89304, 89305)
SOURCE_EXPLORATORY_RESULT_ID = (
    "mnist-ibm-om-baseline-spacing-pv-winsorized-nominal-"
    "exploratory-20260828-v2"
)


_RUNS_BY_SPACING = {
    1: {
        87001: (
            "alpha_000_spacing_1delta-heldout-87001/"
            "20260828T174024.922193Z-1c112232-8402a059",
            "c323e9106cd896a7123d05b52a28e39ad68ea4391c84cb5cf9b732015498517a",
        ),
        87002: (
            "alpha_000_spacing_1delta-heldout-87002/"
            "20260828T174024.850453Z-d8ded840-6db4021f",
            "9011d7027bd8ba68cfb466c28f5760550a98062ca36f98acc688399ce785e7bb",
        ),
        87003: (
            "alpha_000_spacing_1delta-heldout-87003/"
            "20260828T174024.842948Z-f7fe0d0b-6832dbfd",
            "9337993dd567c4c53d929335c9509609d46f3ef13486da8fc08133afced9d812",
        ),
    },
    2: {
        87001: (
            "alpha_000_spacing_2delta-heldout-87001/"
            "20260828T174024.868436Z-fadc637f-c7f13e9f",
            "18021d36e22296f4094c771d9109e3ca3bea2282451e381c3212271e12650a7f",
        ),
        87002: (
            "alpha_000_spacing_2delta-heldout-87002/"
            "20260828T174144.368371Z-a58dd6f7-9ef1d401",
            "56c2efb496897922a762dd88f9f22aae77c0051e14fc8680c312b5484cea936b",
        ),
        87003: (
            "alpha_000_spacing_2delta-heldout-87003/"
            "20260828T174144.748876Z-bd05ba64-47d49272",
            "e3aed0cb33287bd3b1dbe7c108fbc52ee2e7650e0870ecabbe32fa366a991895",
        ),
    },
    4: {
        87001: (
            "alpha_000_spacing_4delta-heldout-87001/"
            "20260828T174149.943143Z-51dd5b9e-0ca48b1e",
            "c60ee51a55e4ce605fcaec835fef46642f947b806c4b25fd712f6fd6db7b8333",
        ),
        87002: (
            "alpha_000_spacing_4delta-heldout-87002/"
            "20260828T174150.045625Z-3223d6b6-cdc9535e",
            "310c3a3538db88654b696127f606e9510a1dc8f9fc57ec038356bc558306adaf",
        ),
        87003: (
            "alpha_000_spacing_4delta-heldout-87003/"
            "20260828T174304.918611Z-0e5d3d69-e7a4f302",
            "2224200c6f8bd998de93adf4a14465dfa9d02ce88518e28d1658e43de8dc96f6",
        ),
    },
}


@dataclass(frozen=True)
class PredecessorBundleContract:
    assignment_seed: int
    role: str
    component: str
    run_relative_path: str
    result_sha256: str


@dataclass(frozen=True)
class MultiAssignmentContract:
    training_assignment_seeds: tuple[int, int]
    development_assignment_seed: int
    final_assignment_seed: int
    training_cycle: str
    final_endpoint_seeds: tuple[int, ...]


@dataclass(frozen=True)
class MultiAssignmentMappingContract:
    baseline_policy: str
    baseline_position_fraction: float
    baseline_definition: str
    destination_column_groups: tuple[tuple[str, str], tuple[str, str]]
    spacing_delta_x_multiplier: int
    delta_x_definition: str
    conductance_formula: str
    fixed_scale_fractions: tuple[float, float]
    fixed_logit_gain: float
    logical_rounding: str
    level_capacity_policy: str
    reference_policy: str
    circuit_accounting: str


@dataclass(frozen=True)
class MultiAssignmentTrainingContract:
    epochs: int
    optimizer: str
    learning_rates: tuple[float, float]
    momentum: float
    weight_decay: float
    quantizer: str
    assignment_cycle: str
    program_verify: bool
    read_noise: bool


@dataclass(frozen=True)
class MultiAssignmentEvaluationContract:
    development_role: str
    development_selection_metric: str
    fixed_headline_epoch: int
    final_role: str
    final_program_verify: bool
    inference_read_noise: bool


@dataclass(frozen=True)
class WinsorizedMultiAssignmentQatProtocol:
    source: Mapping[str, Any]
    assignments: MultiAssignmentContract
    predecessors: tuple[PredecessorBundleContract, ...]
    device: Mapping[str, Any]
    commissioning: Mapping[str, Any]
    joint_assignment_repair: Mapping[str, Any]
    mapping: MultiAssignmentMappingContract
    initialization: Mapping[str, Any]
    training: MultiAssignmentTrainingContract
    evaluation: MultiAssignmentEvaluationContract
    execution: Mapping[str, Any]


@dataclass(frozen=True)
class WinsorizedMultiAssignmentQatConfig:
    schema_version: int
    experiment_id: str
    protocol: WinsorizedMultiAssignmentQatProtocol
    student: StudentConfig


@dataclass(frozen=True)
class WinsorizedMultiAssignmentQatTrainSpec:
    experiment_id: str
    protocol: WinsorizedMultiAssignmentQatProtocol
    student: StudentTrainSpec


_ASSIGNMENTS = {
    "training_assignment_seeds": list(TRAINING_ASSIGNMENT_SEEDS),
    "development_assignment_seed": DEVELOPMENT_ASSIGNMENT_SEED,
    "final_assignment_seed": FINAL_ASSIGNMENT_SEED,
    "training_cycle": "global_minibatch_ordinal_modulo_two_start_86001",
    "final_endpoint_seeds": list(FINAL_ENDPOINT_SEEDS),
}
_INITIALIZATION = {
    "source_weight_sha256_by_layer": list(COMMON_SOURCE_WEIGHT_SHA256),
    "logical_master": "teacher_weight_divided_by_layer_absmax_fp32",
    "epoch0_parity": (
        "exact_reproduction_of_each_assignment_predecessor_quantized_conductance"
    ),
    "clean_shadow_policy": "one_common_normalized_fp32_master_across_assignments",
}
_TRAINING = {
    "epochs": NUM_EPOCHS,
    "optimizer": "sgd",
    "learning_rates": list(LEARNING_RATES),
    "momentum": 0.0,
    "weight_decay": 0.0,
    "quantizer": "deterministic_uniform_device_codebook_with_identity_ste",
    "assignment_cycle": "global_minibatch_ordinal_modulo_two_start_86001",
    "program_verify": False,
    "read_noise": False,
}
_EVALUATION = {
    "development_role": "selection_only_disjoint_assignment_87002",
    "development_selection_metric": (
        "student_correct_then_lower_kl_then_earlier_epoch"
    ),
    "fixed_headline_epoch": 10,
    "final_role": "heldout_from_qat_and_selection_assignment_87003",
    "final_program_verify": True,
    "inference_read_noise": False,
}
_EXECUTION = {
    "evidence_tier": "exploratory_noncanonical",
    "profile": "direct_local_cuda",
    "training_forward": "deterministic_ideal_quantized_winsorized_mapping",
    "straight_through_estimator": "identity_gradient_through_quantizer",
    "training_program_verify": False,
    "inference_read_noise": False,
    "retention_drift": False,
    "checkpoint_policy": (
        "headline_exact_epoch_10_best_87002_development_is_diagnostic_only"
    ),
}


def _expected_predecessors(spacing: int) -> list[dict[str, Any]]:
    runs = _RUNS_BY_SPACING[spacing]
    path_87001, result_87001 = runs[87001]
    path_87002, result_87002 = runs[87002]
    path_87003, result_87003 = runs[87003]
    return [
        {
            "assignment_seed": 86001,
            "role": "training_cycle_0",
            "component": "development",
            "run_relative_path": path_87001,
            "result_sha256": result_87001,
        },
        {
            "assignment_seed": 87001,
            "role": "training_cycle_1",
            "component": "heldout",
            "run_relative_path": path_87001,
            "result_sha256": result_87001,
        },
        {
            "assignment_seed": 87002,
            "role": "development_selection",
            "component": "heldout",
            "run_relative_path": path_87002,
            "result_sha256": result_87002,
        },
        {
            "assignment_seed": 87003,
            "role": "final_heldout",
            "component": "heldout",
            "run_relative_path": path_87003,
            "result_sha256": result_87003,
        },
    ]


def _parse_protocol(value: Any) -> WinsorizedMultiAssignmentQatProtocol:
    path = "config.winsorized_multi_assignment_qat"
    raw = _object(value, path)
    fields = {
        "source",
        "assignments",
        "predecessors",
        "device",
        "commissioning",
        "joint_assignment_repair",
        "mapping",
        "initialization",
        "training",
        "evaluation",
        "execution",
    }
    _keys(raw, path, fields)

    mapping = _object(raw["mapping"], f"{path}.mapping")
    spacing = mapping.get("spacing_delta_x_multiplier")
    if type(spacing) is not int or spacing not in SPACING_DELTA_X_MULTIPLIERS:
        raise config_error(
            f"{path}.mapping.spacing_delta_x_multiplier",
            f"to be one of {SPACING_DELTA_X_MULTIPLIERS!r}",
            spacing,
        )
    expected_mapping = {
        **_MAPPING_COMMON,
        "spacing_delta_x_multiplier": spacing,
    }
    expected_predecessors = _expected_predecessors(spacing)
    exact_sections = {
        "source": _SOURCE,
        "assignments": _ASSIGNMENTS,
        "predecessors": expected_predecessors,
        "device": _DEVICE,
        "commissioning": _COMMISSIONING,
        "joint_assignment_repair": _JOINT_ASSIGNMENT_REPAIR,
        "mapping": expected_mapping,
        "initialization": _INITIALIZATION,
        "training": _TRAINING,
        "evaluation": _EVALUATION,
        "execution": _EXECUTION,
    }
    for name, expected in exact_sections.items():
        if name == "predecessors":
            if not _same_json_value(raw[name], expected):
                raise config_error(
                    f"{path}.{name}", "to equal the frozen predecessor list", raw[name]
                )
        else:
            _require_exact(
                _object(raw[name], f"{path}.{name}"),
                expected,
                path=f"{path}.{name}",
            )

    return WinsorizedMultiAssignmentQatProtocol(
        source=dict(_SOURCE),
        assignments=MultiAssignmentContract(
            training_assignment_seeds=TRAINING_ASSIGNMENT_SEEDS,
            development_assignment_seed=DEVELOPMENT_ASSIGNMENT_SEED,
            final_assignment_seed=FINAL_ASSIGNMENT_SEED,
            training_cycle=_ASSIGNMENTS["training_cycle"],
            final_endpoint_seeds=FINAL_ENDPOINT_SEEDS,
        ),
        predecessors=tuple(
            PredecessorBundleContract(**item) for item in expected_predecessors
        ),
        device=dict(_DEVICE),
        commissioning=dict(_COMMISSIONING),
        joint_assignment_repair=dict(_JOINT_ASSIGNMENT_REPAIR),
        mapping=MultiAssignmentMappingContract(
            **{
                **_MAPPING_COMMON,
                "destination_column_groups": tuple(
                    tuple(group)
                    for group in _MAPPING_COMMON["destination_column_groups"]
                ),
                "fixed_scale_fractions": FIXED_SCALE_FRACTIONS,
                "spacing_delta_x_multiplier": spacing,
            }
        ),
        initialization=dict(_INITIALIZATION),
        training=MultiAssignmentTrainingContract(
            epochs=NUM_EPOCHS,
            optimizer="sgd",
            learning_rates=LEARNING_RATES,
            momentum=0.0,
            weight_decay=0.0,
            quantizer=_TRAINING["quantizer"],
            assignment_cycle=_TRAINING["assignment_cycle"],
            program_verify=False,
            read_noise=False,
        ),
        evaluation=MultiAssignmentEvaluationContract(
            development_role=_EVALUATION["development_role"],
            development_selection_metric=(
                _EVALUATION["development_selection_metric"]
            ),
            fixed_headline_epoch=10,
            final_role=_EVALUATION["final_role"],
            final_program_verify=True,
            inference_read_noise=False,
        ),
        execution=dict(_EXECUTION),
    )


def parse_winsorized_multi_assignment_qat_config(
    payload: Mapping[str, Any],
) -> WinsorizedMultiAssignmentQatConfig:
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
    _keys(raw, "config", student_keys | {"winsorized_multi_assignment_qat"})
    if raw["schema_version"] != SCHEMA_VERSION:
        raise config_error(
            "config.schema_version", "to equal 1", raw["schema_version"]
        )
    if raw["experiment_id"] != EXPERIMENT_ID:
        raise config_error(
            "config.experiment_id", f"to equal {EXPERIMENT_ID!r}", raw["experiment_id"]
        )
    protocol = _parse_protocol(raw["winsorized_multi_assignment_qat"])
    student_payload = {key: raw[key] for key in student_keys}
    student_payload["experiment_id"] = STUDENT_EXPERIMENT_ID
    student = parse_student_config(student_payload)
    _validate_student_contract(student, raw)
    return WinsorizedMultiAssignmentQatConfig(
        schema_version=SCHEMA_VERSION,
        experiment_id=EXPERIMENT_ID,
        protocol=protocol,
        student=student,
    )


def resolve_winsorized_multi_assignment_qat_spec(
    document: WinsorizedMultiAssignmentQatConfig,
    mode: RunMode,
) -> WinsorizedMultiAssignmentQatTrainSpec:
    if mode is not RunMode.TRAIN:
        raise config_error(
            "the requested run mode",
            "to be 'train' for this train-only experiment",
            mode.value,
        )
    student = resolve_student_spec(document.student, mode)
    if not isinstance(student, StudentTrainSpec):  # pragma: no cover
        raise RuntimeError("Expected the parent parser to resolve training.")
    return WinsorizedMultiAssignmentQatTrainSpec(
        experiment_id=EXPERIMENT_ID,
        protocol=document.protocol,
        student=student,
    )


__all__ = [
    "DEVELOPMENT_ASSIGNMENT_SEED",
    "EXPERIMENT_ID",
    "EXPECTED_TEACHER_WEIGHTS_PATH",
    "EXPECTED_TEACHER_WEIGHTS_SHA256",
    "FINAL_ASSIGNMENT_SEED",
    "FINAL_ENDPOINT_SEEDS",
    "NUM_EPOCHS",
    "SCHEMA_VERSION",
    "SPACING_DELTA_X_MULTIPLIERS",
    "TRAINING_ASSIGNMENT_SEEDS",
    "PredecessorBundleContract",
    "WinsorizedMultiAssignmentQatConfig",
    "WinsorizedMultiAssignmentQatProtocol",
    "WinsorizedMultiAssignmentQatTrainSpec",
    "parse_winsorized_multi_assignment_qat_config",
    "resolve_winsorized_multi_assignment_qat_spec",
]
