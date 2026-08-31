"""Strict config for the isolated IBM-OM TT output-transfer follow-up.

The completed endpoint-optimizer comparison left the output-layer transfer
effectively inactive.  This successor freezes each TT variant's selected W1
transfer lambda and changes only its W2 lambda.  Endpoint 89401 remains the
validation-only development realization; 89402--89405 remain evaluation
realizations of the same target-87004 population, not independent arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from experiments.mnist_relu.config import _keys, _object
from experiments.mnist_relu_drn.config import (
    StudentConfig,
    StudentTrainSpec,
    resolve_student_spec,
)
from experiments.mnist_relu_drn.ibm_om_winsorized_endpoint_optimizer_recovery_config import (
    DEVELOPMENT_ENDPOINT_SEED,
    EVALUATION_ENDPOINT_SEEDS,
    TARGET_ASSIGNMENT_SEED,
    EndpointOptimizerRecoveryProtocol,
    parse_endpoint_optimizer_recovery_config,
)
from experiments.schema import RunMode, config_error


EXPERIMENT_ID = "mnist_ibm_om_winsorized_tt_output_transfer_followup.v1"
SCHEMA_VERSION = 1
SOURCE_RESULT_ID = (
    "mnist-ibm-om-winsorized-endpoint-optimizer-recovery-"
    "exploratory-20260829-v1"
)
SOURCE_RUN_RELATIVE_PATH = "20260829T142952.088023Z-0ae69031-ec21506d"
SOURCE_SUMMARY_RELATIVE_PATH = "artifacts/scientific_summary.json"
SOURCE_SUMMARY_SHA256 = (
    "abd79cec6601f3ddedccc43bee78d0ac7e700b1903120317b1262364018d22f4"
)

TT_VARIANTS = ("tt_v1", "tt_v2")
W2_MULTIPLIER_GRID = (10.0, 30.0, 100.0)
DEVELOPMENT_MAXIMUM_BATCHES = 512
FROZEN_SELECTED_LAMBDAS = {
    "tt_v1": (0.04411914893617021, 0.00011974808510638298),
    "tt_v2": (0.13235744680851064, 0.00035924425531914895),
}


@dataclass(frozen=True)
class SourceOptimizerRecoveryContract:
    result_id: str
    run_relative_path: str
    summary_relative_path: str
    summary_sha256: str
    selected_effective_lambdas: tuple[tuple[str, tuple[float, float]], ...]


@dataclass(frozen=True)
class OutputTransferDevelopmentContract:
    endpoint_seed: int
    role: str
    maximum_batches: int
    w2_multiplier_grid: tuple[float, ...]
    selection_metric: str
    tie_breakers: tuple[str, ...]
    test_opened: bool


@dataclass(frozen=True)
class OutputTransferEvaluationContract:
    endpoint_seeds: tuple[int, ...]
    endpoint_semantics: str
    independent_population_count: int
    optimizers: tuple[str, ...]
    epochs: int
    test_role: str
    selection_after_freeze: bool


@dataclass(frozen=True)
class TtOutputTransferFollowupProtocol:
    target_assignment_seed: int
    base_recovery: EndpointOptimizerRecoveryProtocol
    source_optimizer_recovery: SourceOptimizerRecoveryContract
    development: OutputTransferDevelopmentContract
    evaluation: OutputTransferEvaluationContract
    evidence_tier: str
    full_conductance_forward: str
    verify_reads_during_training: int


@dataclass(frozen=True)
class TtOutputTransferFollowupConfig:
    schema_version: int
    experiment_id: str
    protocol: TtOutputTransferFollowupProtocol
    student: StudentConfig


@dataclass(frozen=True)
class TtOutputTransferFollowupTrainSpec:
    experiment_id: str
    protocol: TtOutputTransferFollowupProtocol
    student: StudentTrainSpec


def _base_document() -> Any:
    return parse_endpoint_optimizer_recovery_config(
        {
            "schema_version": 1,
            "experiment_id": (
                "mnist_ibm_om_winsorized_endpoint_optimizer_recovery.v1"
            ),
            "target_assignment_seed": TARGET_ASSIGNMENT_SEED,
            "development_endpoint_seed": DEVELOPMENT_ENDPOINT_SEED,
            "evaluation_endpoint_seeds": list(EVALUATION_ENDPOINT_SEEDS),
        }
    )


def parse_tt_output_transfer_followup_config(
    payload: Mapping[str, Any],
) -> TtOutputTransferFollowupConfig:
    raw = _object(payload, "config")
    _keys(
        raw,
        "config",
        {
            "schema_version",
            "experiment_id",
            "target_assignment_seed",
            "development_endpoint_seed",
            "evaluation_endpoint_seeds",
            "w2_multiplier_grid",
        },
    )
    expected = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "target_assignment_seed": TARGET_ASSIGNMENT_SEED,
        "development_endpoint_seed": DEVELOPMENT_ENDPOINT_SEED,
        "evaluation_endpoint_seeds": list(EVALUATION_ENDPOINT_SEEDS),
        "w2_multiplier_grid": list(W2_MULTIPLIER_GRID),
    }
    for name, value in expected.items():
        if raw[name] != value:
            raise config_error(f"config.{name}", f"to equal {value!r}", raw[name])

    base = _base_document()
    protocol = TtOutputTransferFollowupProtocol(
        target_assignment_seed=TARGET_ASSIGNMENT_SEED,
        base_recovery=base.protocol,
        source_optimizer_recovery=SourceOptimizerRecoveryContract(
            result_id=SOURCE_RESULT_ID,
            run_relative_path=SOURCE_RUN_RELATIVE_PATH,
            summary_relative_path=SOURCE_SUMMARY_RELATIVE_PATH,
            summary_sha256=SOURCE_SUMMARY_SHA256,
            selected_effective_lambdas=tuple(
                (name, FROZEN_SELECTED_LAMBDAS[name]) for name in TT_VARIANTS
            ),
        ),
        development=OutputTransferDevelopmentContract(
            endpoint_seed=DEVELOPMENT_ENDPOINT_SEED,
            role=(
                "validation_only_W2_transfer_multiplier_screen_with_W1_frozen"
            ),
            maximum_batches=DEVELOPMENT_MAXIMUM_BATCHES,
            w2_multiplier_grid=W2_MULTIPLIER_GRID,
            selection_metric="maximum_validation_student_correct",
            tie_breakers=(
                "lower_validation_teacher_student_KL",
                "fewer_slow_C_pulses",
                "smaller_W2_multiplier",
            ),
            test_opened=False,
        ),
        evaluation=OutputTransferEvaluationContract(
            endpoint_seeds=EVALUATION_ENDPOINT_SEEDS,
            endpoint_semantics=(
                "four_stochastic_PV_endpoint_realizations_on_one_target_87004_population"
            ),
            independent_population_count=1,
            optimizers=TT_VARIANTS,
            epochs=1,
            test_role="before_and_after_fixed_one_epoch_TT_arm_no_selection",
            selection_after_freeze=False,
        ),
        evidence_tier="exploratory_noncanonical",
        full_conductance_forward=(
            "G=native_raw_active_a+1_without_remap_or_clipping"
        ),
        verify_reads_during_training=0,
    )
    return TtOutputTransferFollowupConfig(
        schema_version=SCHEMA_VERSION,
        experiment_id=EXPERIMENT_ID,
        protocol=protocol,
        student=base.student,
    )


def resolve_tt_output_transfer_followup_spec(
    document: TtOutputTransferFollowupConfig, mode: RunMode
) -> TtOutputTransferFollowupTrainSpec:
    if mode is not RunMode.TRAIN:
        raise config_error(
            "the requested run mode",
            "to be 'train' for this train-only experiment",
            mode.value,
        )
    student = resolve_student_spec(document.student, mode)
    if not isinstance(student, StudentTrainSpec):  # pragma: no cover
        raise RuntimeError("Expected the parent parser to resolve training.")
    return TtOutputTransferFollowupTrainSpec(
        experiment_id=EXPERIMENT_ID,
        protocol=document.protocol,
        student=student,
    )


__all__ = [
    "EXPERIMENT_ID",
    "FROZEN_SELECTED_LAMBDAS",
    "SCHEMA_VERSION",
    "TT_VARIANTS",
    "TtOutputTransferFollowupTrainSpec",
    "W2_MULTIPLIER_GRID",
    "parse_tt_output_transfer_followup_config",
    "resolve_tt_output_transfer_followup_spec",
]
