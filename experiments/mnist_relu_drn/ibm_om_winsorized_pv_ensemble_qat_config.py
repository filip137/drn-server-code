"""Strict exploratory config for Winsorized IBM OM P&V-ensemble QAT."""

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
from experiments.mnist_relu_drn.ibm_om_winsorized_multi_assignment_qat_config import (
    PredecessorBundleContract,
    SOURCE_EXPLORATORY_RESULT_ID,
    _expected_predecessors,
)
from experiments.mnist_relu_drn.ibm_om_winsorized_qat_config import (
    COMMON_SOURCE_WEIGHT_SHA256,
    EXPECTED_TEACHER_WEIGHTS_PATH,
    EXPECTED_TEACHER_WEIGHTS_SHA256,
    FIXED_LOGIT_GAIN,
    FIXED_SCALE_FRACTIONS,
    LEARNING_RATES,
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
    _require_exact,
    _same_json_value,
)
from experiments.schema import RunMode, config_error


EXPERIMENT_ID = "mnist_ibm_om_winsorized_pv_ensemble_qat.v1"
SCHEMA_VERSION = 1
NUM_EPOCHS = 3
SPACING_DELTA_X_MULTIPLIER = 1
TRAINING_ASSIGNMENT_SEEDS = (86001, 87001)
DEVELOPMENT_ASSIGNMENT_SEED = 87002
FINAL_ASSIGNMENT_SEED = 87003
TRAINING_ENDPOINT_SEEDS = (89501, 89502, 89503, 89504)
DEVELOPMENT_ENDPOINT_SEEDS = (89521, 89522, 89523, 89524)
FINAL_ENDPOINT_SEEDS = (89301, 89302, 89303, 89304, 89305)
ARMS = ("deterministic", "mean2", "tail4")


_ARM_TRAINING = {
    "deterministic": {
        "persistent_samples_per_minibatch": 0,
        "ideal_loss_weight": 1.0,
        "mean_pv_loss_weight": 0.0,
        "tail_pv_loss_weight": 0.0,
        "persistent_endpoint_source": "none_deterministic_codebook_control",
        "training_endpoint_bank_size": 0,
    },
    "mean2": {
        "persistent_samples_per_minibatch": 2,
        "ideal_loss_weight": 0.25,
        "mean_pv_loss_weight": 0.75,
        "tail_pv_loss_weight": 0.0,
        "persistent_endpoint_source": (
            "exact_precomputed_one_pulse_per_cell_per_code_persistent_x"
        ),
        "training_endpoint_bank_size": 4,
    },
    "tail4": {
        "persistent_samples_per_minibatch": 4,
        "ideal_loss_weight": 0.25,
        "mean_pv_loss_weight": 0.50,
        "tail_pv_loss_weight": 0.25,
        "persistent_endpoint_source": (
            "exact_precomputed_one_pulse_per_cell_per_code_persistent_x"
        ),
        "training_endpoint_bank_size": 4,
    },
}


@dataclass(frozen=True)
class PvEnsembleAssignmentContract:
    training_assignment_seeds: tuple[int, int]
    development_assignment_seed: int
    final_assignment_seed: int
    training_endpoint_seeds: tuple[int, ...]
    development_endpoint_seeds: tuple[int, ...]
    final_endpoint_seeds: tuple[int, ...]
    assignment_cycle: str
    mean2_endpoint_schedule: str


@dataclass(frozen=True)
class PvEnsembleMappingContract:
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
class PvEnsembleTrainingContract:
    epochs: int
    optimizer: str
    learning_rates: tuple[float, float]
    momentum: float
    weight_decay: float
    quantizer: str
    assignment_cycle: str
    arm: str
    persistent_samples_per_minibatch: int
    ideal_loss_weight: float
    mean_pv_loss_weight: float
    tail_pv_loss_weight: float
    persistent_endpoint_source: str
    training_endpoint_bank_size: int
    endpoint_table_target_policy: str
    endpoint_table_controller: str
    endpoint_table_initialization: str
    apparent_verify_applied_to_drn: bool
    post_program_projection: str
    read_noise: bool


@dataclass(frozen=True)
class PvEnsembleEvaluationContract:
    development_role: str
    development_endpoint_bank_size: int
    development_selection_metric: str
    fixed_headline_epoch: int
    final_role: str
    final_program_verify: bool
    inference_read_noise: bool


@dataclass(frozen=True)
class WinsorizedPvEnsembleQatProtocol:
    source: Mapping[str, Any]
    assignments: PvEnsembleAssignmentContract
    predecessors: tuple[PredecessorBundleContract, ...]
    device: Mapping[str, Any]
    commissioning: Mapping[str, Any]
    joint_assignment_repair: Mapping[str, Any]
    mapping: PvEnsembleMappingContract
    initialization: Mapping[str, Any]
    training: PvEnsembleTrainingContract
    evaluation: PvEnsembleEvaluationContract
    execution: Mapping[str, Any]


@dataclass(frozen=True)
class WinsorizedPvEnsembleQatConfig:
    schema_version: int
    experiment_id: str
    protocol: WinsorizedPvEnsembleQatProtocol
    student: StudentConfig


@dataclass(frozen=True)
class WinsorizedPvEnsembleQatTrainSpec:
    experiment_id: str
    protocol: WinsorizedPvEnsembleQatProtocol
    student: StudentTrainSpec


_ASSIGNMENTS = {
    "training_assignment_seeds": list(TRAINING_ASSIGNMENT_SEEDS),
    "development_assignment_seed": DEVELOPMENT_ASSIGNMENT_SEED,
    "final_assignment_seed": FINAL_ASSIGNMENT_SEED,
    "training_endpoint_seeds": list(TRAINING_ENDPOINT_SEEDS),
    "development_endpoint_seeds": list(DEVELOPMENT_ENDPOINT_SEEDS),
    "final_endpoint_seeds": list(FINAL_ENDPOINT_SEEDS),
    "assignment_cycle": "global_minibatch_ordinal_modulo_two_start_86001",
    "mean2_endpoint_schedule": (
        "global_minibatch_ordinal_cyclic_adjacent_pair_over_four_seed_bank"
    ),
}
_INITIALIZATION = {
    "source_weight_sha256_by_layer": list(COMMON_SOURCE_WEIGHT_SHA256),
    "logical_master": "teacher_weight_divided_by_layer_absmax_fp32",
    "epoch0_parity": (
        "exact_reproduction_of_each_assignment_predecessor_quantized_conductance"
    ),
    "clean_shadow_policy": "one_common_normalized_fp32_master_across_assignments",
}
_TRAINING_COMMON = {
    "epochs": NUM_EPOCHS,
    "optimizer": "sgd",
    "learning_rates": list(LEARNING_RATES),
    "momentum": 0.0,
    "weight_decay": 0.0,
    "quantizer": "uniform_device_codebook_with_identity_ste",
    "assignment_cycle": _ASSIGNMENTS["assignment_cycle"],
    "endpoint_table_target_policy": (
        "B_plus_n_times_h_valid_entries_unsupported_pairs_NaN_never_lookup"
    ),
    "endpoint_table_controller": "one_pulse_maximum_128_tolerance_half_delta_x",
    "endpoint_table_initialization": "sampled_reset_lower_before_each_target",
    "apparent_verify_applied_to_drn": False,
    "post_program_projection": "none",
    "read_noise": False,
}
_EVALUATION = {
    "development_role": "selection_only_disjoint_assignment_87002",
    "development_endpoint_bank_size": len(DEVELOPMENT_ENDPOINT_SEEDS),
    "development_selection_metric": (
        "persistent_mean_correct_then_min_correct_then_lower_mean_kl_then_earlier_epoch"
    ),
    "fixed_headline_epoch": NUM_EPOCHS,
    "final_role": "heldout_from_qat_and_selection_assignment_87003",
    "final_program_verify": True,
    "inference_read_noise": False,
}
_EXECUTION = {
    "evidence_tier": "exploratory_noncanonical",
    "profile": "direct_local_cuda",
    "straight_through_estimator": (
        "persistent_full_G_circuit_gradient_through_ideal_code_target_envelope"
    ),
    "persistent_state_coordinate": "raw_x_then_full_G_equals_2x",
    "apparent_verify_role": "controller_and_diagnostics_only",
    "retention_drift": False,
    "checkpoint_policy": (
        "headline_exact_epoch_3_best_87002_development_is_diagnostic_only"
    ),
}
_STUDENT_TRAIN_3 = {
    "num_epochs": NUM_EPOCHS,
    "learning_rates": list(LEARNING_RATES),
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


def _expected_training(arm: str) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError(f"Unsupported P&V ensemble arm: {arm!r}.")
    return {**_TRAINING_COMMON, "arm": arm, **_ARM_TRAINING[arm]}


def _parse_protocol(value: Any) -> WinsorizedPvEnsembleQatProtocol:
    path = "config.winsorized_pv_ensemble_qat"
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
    training = _object(raw["training"], f"{path}.training")
    arm = training.get("arm")
    if type(arm) is not str or arm not in ARMS:
        raise config_error(f"{path}.training.arm", f"to be one of {ARMS!r}", arm)

    expected_mapping = {
        **_MAPPING_COMMON,
        "spacing_delta_x_multiplier": SPACING_DELTA_X_MULTIPLIER,
    }
    expected_predecessors = _expected_predecessors(SPACING_DELTA_X_MULTIPLIER)
    exact_sections = {
        "source": _SOURCE,
        "assignments": _ASSIGNMENTS,
        "predecessors": expected_predecessors,
        "device": _DEVICE,
        "commissioning": _COMMISSIONING,
        "joint_assignment_repair": _JOINT_ASSIGNMENT_REPAIR,
        "mapping": expected_mapping,
        "initialization": _INITIALIZATION,
        "training": _expected_training(arm),
        "evaluation": _EVALUATION,
        "execution": _EXECUTION,
    }
    for name, expected in exact_sections.items():
        if name == "predecessors":
            if not _same_json_value(raw[name], expected):
                raise config_error(
                    f"{path}.{name}", "to equal the frozen h=delta predecessor list", raw[name]
                )
        else:
            _require_exact(
                _object(raw[name], f"{path}.{name}"),
                expected,
                path=f"{path}.{name}",
            )

    assignment = PvEnsembleAssignmentContract(
        training_assignment_seeds=TRAINING_ASSIGNMENT_SEEDS,
        development_assignment_seed=DEVELOPMENT_ASSIGNMENT_SEED,
        final_assignment_seed=FINAL_ASSIGNMENT_SEED,
        training_endpoint_seeds=TRAINING_ENDPOINT_SEEDS,
        development_endpoint_seeds=DEVELOPMENT_ENDPOINT_SEEDS,
        final_endpoint_seeds=FINAL_ENDPOINT_SEEDS,
        assignment_cycle=_ASSIGNMENTS["assignment_cycle"],
        mean2_endpoint_schedule=_ASSIGNMENTS["mean2_endpoint_schedule"],
    )
    if not (
        set(assignment.training_assignment_seeds).isdisjoint(
            {assignment.development_assignment_seed, assignment.final_assignment_seed}
        )
        and assignment.development_assignment_seed != assignment.final_assignment_seed
        and set(assignment.training_endpoint_seeds).isdisjoint(
            set(assignment.development_endpoint_seeds)
            | set(assignment.final_endpoint_seeds)
        )
        and set(assignment.development_endpoint_seeds).isdisjoint(
            assignment.final_endpoint_seeds
        )
    ):
        raise RuntimeError("Expected disjoint P&V ensemble assignment and endpoint roles.")
    selected_training = _expected_training(arm)
    return WinsorizedPvEnsembleQatProtocol(
        source=dict(_SOURCE),
        assignments=assignment,
        predecessors=tuple(
            PredecessorBundleContract(**item) for item in expected_predecessors
        ),
        device=dict(_DEVICE),
        commissioning=dict(_COMMISSIONING),
        joint_assignment_repair=dict(_JOINT_ASSIGNMENT_REPAIR),
        mapping=PvEnsembleMappingContract(
            **{
                **_MAPPING_COMMON,
                "destination_column_groups": tuple(
                    tuple(group) for group in _MAPPING_COMMON["destination_column_groups"]
                ),
                "fixed_scale_fractions": FIXED_SCALE_FRACTIONS,
                "spacing_delta_x_multiplier": SPACING_DELTA_X_MULTIPLIER,
            }
        ),
        initialization=dict(_INITIALIZATION),
        training=PvEnsembleTrainingContract(
            epochs=NUM_EPOCHS,
            optimizer="sgd",
            learning_rates=LEARNING_RATES,
            momentum=0.0,
            weight_decay=0.0,
            quantizer=selected_training["quantizer"],
            assignment_cycle=selected_training["assignment_cycle"],
            arm=arm,
            persistent_samples_per_minibatch=selected_training[
                "persistent_samples_per_minibatch"
            ],
            ideal_loss_weight=selected_training["ideal_loss_weight"],
            mean_pv_loss_weight=selected_training["mean_pv_loss_weight"],
            tail_pv_loss_weight=selected_training["tail_pv_loss_weight"],
            persistent_endpoint_source=selected_training[
                "persistent_endpoint_source"
            ],
            training_endpoint_bank_size=selected_training[
                "training_endpoint_bank_size"
            ],
            endpoint_table_target_policy=selected_training[
                "endpoint_table_target_policy"
            ],
            endpoint_table_controller=selected_training[
                "endpoint_table_controller"
            ],
            endpoint_table_initialization=selected_training[
                "endpoint_table_initialization"
            ],
            apparent_verify_applied_to_drn=False,
            post_program_projection="none",
            read_noise=False,
        ),
        evaluation=PvEnsembleEvaluationContract(**_EVALUATION),
        execution=dict(_EXECUTION),
    )


def _validate_student_contract(student: StudentConfig, raw: Mapping[str, Any]) -> None:
    if set(student.modes) != {RunMode.TRAIN.value}:
        raise config_error("config.modes", "to define exactly the train mode", tuple(student.modes))
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
    _keys(modes, "config.modes", {"train"})
    _require_exact(
        _object(modes["train"], "config.modes.train"),
        _STUDENT_TRAIN_3,
        path="config.modes.train",
    )


def parse_winsorized_pv_ensemble_qat_config(
    payload: Mapping[str, Any],
) -> WinsorizedPvEnsembleQatConfig:
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
    _keys(raw, "config", student_keys | {"winsorized_pv_ensemble_qat"})
    if raw["schema_version"] != SCHEMA_VERSION:
        raise config_error("config.schema_version", "to equal 1", raw["schema_version"])
    if raw["experiment_id"] != EXPERIMENT_ID:
        raise config_error(
            "config.experiment_id", f"to equal {EXPERIMENT_ID!r}", raw["experiment_id"]
        )
    protocol = _parse_protocol(raw["winsorized_pv_ensemble_qat"])
    student_payload = {key: raw[key] for key in student_keys}
    student_payload["experiment_id"] = STUDENT_EXPERIMENT_ID
    student = parse_student_config(student_payload)
    _validate_student_contract(student, raw)
    return WinsorizedPvEnsembleQatConfig(
        schema_version=SCHEMA_VERSION,
        experiment_id=EXPERIMENT_ID,
        protocol=protocol,
        student=student,
    )


def resolve_winsorized_pv_ensemble_qat_spec(
    document: WinsorizedPvEnsembleQatConfig,
    mode: RunMode,
) -> WinsorizedPvEnsembleQatTrainSpec:
    if mode is not RunMode.TRAIN:
        raise config_error(
            "the requested run mode",
            "to be 'train' for this train-only experiment",
            mode.value,
        )
    student = resolve_student_spec(document.student, mode)
    if not isinstance(student, StudentTrainSpec):  # pragma: no cover
        raise RuntimeError("Expected the parent parser to resolve training.")
    return WinsorizedPvEnsembleQatTrainSpec(
        experiment_id=EXPERIMENT_ID,
        protocol=document.protocol,
        student=student,
    )


__all__ = [
    "ARMS",
    "DEVELOPMENT_ASSIGNMENT_SEED",
    "DEVELOPMENT_ENDPOINT_SEEDS",
    "EXPECTED_TEACHER_WEIGHTS_PATH",
    "EXPECTED_TEACHER_WEIGHTS_SHA256",
    "EXPERIMENT_ID",
    "FINAL_ASSIGNMENT_SEED",
    "FINAL_ENDPOINT_SEEDS",
    "LEARNING_RATES",
    "NUM_EPOCHS",
    "SCHEMA_VERSION",
    "SOURCE_EXPLORATORY_RESULT_ID",
    "SPACING_DELTA_X_MULTIPLIER",
    "TRAINING_ASSIGNMENT_SEEDS",
    "TRAINING_ENDPOINT_SEEDS",
    "WinsorizedPvEnsembleQatConfig",
    "WinsorizedPvEnsembleQatProtocol",
    "WinsorizedPvEnsembleQatTrainSpec",
    "parse_winsorized_pv_ensemble_qat_config",
    "resolve_winsorized_pv_ensemble_qat_spec",
]
