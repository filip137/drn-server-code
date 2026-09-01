"""Strict config for structured partial closed-loop recovery from exact P0.

The experiment inherits the immutable target-94004/endpoint-94301 deployment,
loader, optimizer, and controller settings from the exact open/closed
comparator.  Its only intervention is a frozen logical-quad update mask.  The
top-k masks are ranked once on the existing train-only mapping calibration
loader and never use validation, test, device bounds, or the corrupt mask.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from experiments.mnist_relu.config import _keys, _object
from experiments.mnist_relu_drn.config import (
    StudentConfig,
    StudentTrainSpec,
    parse_student_config,
    resolve_student_spec,
)
from experiments.mnist_relu_drn.ibm_om_exact_p0_open_vs_closed_loop_adam_config import (
    EXPECTED_DEVICE_MODEL_SHA256,
    EXPECTED_P0_SHA256,
    EXPECTED_PUBLISHED_POPULATION_FINGERPRINT,
    EXPECTED_PUBLISHED_POPULATION_SHA256,
    EXPECTED_SOURCE_HWA_CHECKPOINT_SHA256,
    EXPECTED_TEACHER_WEIGHTS_PATH,
    EXPECTED_TEACHER_WEIGHTS_SHA256,
    FIXED_LOGIT_GAIN,
    MAXIMUM_RANDOM_DRAWS,
    NOMINAL_DELTA_X,
    P0_ENDPOINT_SEED,
    RECOVERY_DATA_ORDER_SEED,
    RECOVERY_EPOCHS,
    RECOVERY_LEARNING_RATE_RAW_X,
    RECOVERY_PULSE_CAP,
    TARGET_ASSIGNMENT_SEED,
    VERIFY_TOLERANCE_RAW_X,
    ExactP0OpenVsClosedLoopAdamProtocol,
    _protocol as _base_protocol,
    _student_payload as _base_student_payload,
)
from experiments.schema import RunMode, config_error


EXPERIMENT_ID = "mnist_ibm_om_exact_p0_structured_partial_pv_adam.v1"
SCHEMA_VERSION = 1

ARM_IDS = ("full", "w1_only", "w2_only", "w1_top500", "global_top500")
MATCHED_LOGICAL_COUNT = 500
RANKING_EXAMPLES = 1024
RANKING_BATCH_SIZE = 128
RANKING_MINIBATCHES = 8
RANKING_COHORT = (
    "existing 1024-example train-only mapping calibration loader; no new seed"
)
RANKING_AGGREGATION = (
    "mean across the 8 deterministic 128-example calibration minibatches of "
    "the mean absolute dL/dx=2*dL/dG across each canonical quad's 4 physical cells"
)
RANKING_TIE_BREAK = "stable flat-index tie-break"
CANONICAL_LAYOUTS = ("halves", "paired")


@dataclass(frozen=True)
class StructuredPartialRankingContract:
    cohort: str
    examples: int
    batch_size: int
    minibatches: int
    objective: str
    aggregation: str
    canonical_layouts: tuple[str, str]
    global_flat_order: str
    tie_break: str
    uses_new_seed: bool
    uses_device_bounds: bool
    uses_corrupt_mask: bool
    uses_validation: bool
    uses_test: bool


@dataclass(frozen=True)
class StructuredPartialArmContract:
    arm_id: str
    mask_policy: str
    expected_logical_quads: int
    expected_physical_cells: int
    matched_500_quad_address_arm: bool


@dataclass(frozen=True)
class StructuredPartialContract:
    controller: str
    epochs: int
    matched_logical_count: int
    mask_unit: str
    mask_application: tuple[str, str]
    frozen_cell_policy: str
    ranking: StructuredPartialRankingContract
    arms: tuple[
        StructuredPartialArmContract,
        StructuredPartialArmContract,
        StructuredPartialArmContract,
        StructuredPartialArmContract,
        StructuredPartialArmContract,
    ]
    epoch_selection_metrics: tuple[str, str, str]
    test_policy: str


@dataclass(frozen=True)
class ExactP0StructuredPartialPvAdamProtocol:
    base: ExactP0OpenVsClosedLoopAdamProtocol
    partial: StructuredPartialContract


@dataclass(frozen=True)
class ExactP0StructuredPartialPvAdamConfig:
    schema_version: int
    experiment_id: str
    protocol: ExactP0StructuredPartialPvAdamProtocol
    student: StudentConfig


@dataclass(frozen=True)
class ExactP0StructuredPartialPvAdamTrainSpec:
    experiment_id: str
    protocol: ExactP0StructuredPartialPvAdamProtocol
    student: StudentTrainSpec


def _partial_contract() -> StructuredPartialContract:
    ranking = StructuredPartialRankingContract(
        cohort=RANKING_COHORT,
        examples=RANKING_EXAMPLES,
        batch_size=RANKING_BATCH_SIZE,
        minibatches=RANKING_MINIBATCHES,
        objective="teacher_output_kl_only",
        aggregation=RANKING_AGGREGATION,
        canonical_layouts=CANONICAL_LAYOUTS,
        global_flat_order="W1_then_W2_row_major_logical_quad_order",
        tie_break=RANKING_TIE_BREAK,
        uses_new_seed=False,
        uses_device_bounds=False,
        uses_corrupt_mask=False,
        uses_validation=False,
        uses_test=False,
    )
    arms = (
        StructuredPartialArmContract(
            arm_id="full",
            mask_policy="all_W1_and_W2_logical_quads",
            expected_logical_quads=39_700,
            expected_physical_cells=158_800,
            matched_500_quad_address_arm=False,
        ),
        StructuredPartialArmContract(
            arm_id="w1_only",
            mask_policy="all_W1_logical_quads_and_no_W2_quads",
            expected_logical_quads=39_200,
            expected_physical_cells=156_800,
            matched_500_quad_address_arm=False,
        ),
        StructuredPartialArmContract(
            arm_id="w2_only",
            mask_policy="all_W2_logical_quads_and_no_W1_quads",
            expected_logical_quads=500,
            expected_physical_cells=2_000,
            matched_500_quad_address_arm=True,
        ),
        StructuredPartialArmContract(
            arm_id="w1_top500",
            mask_policy="top_500_W1_logical_quads_by_frozen_training_ranking",
            expected_logical_quads=500,
            expected_physical_cells=2_000,
            matched_500_quad_address_arm=True,
        ),
        StructuredPartialArmContract(
            arm_id="global_top500",
            mask_policy="top_500_global_logical_quads_by_frozen_training_ranking",
            expected_logical_quads=500,
            expected_physical_cells=2_000,
            matched_500_quad_address_arm=True,
        ),
    )
    return StructuredPartialContract(
        controller="incremental_one_pulse_closed_loop_target_tracking",
        epochs=RECOVERY_EPOCHS,
        matched_logical_count=MATCHED_LOGICAL_COUNT,
        mask_unit="canonical_four_physical_cell_logical_quad",
        mask_application=("Adam_target_updates", "pulse_eligibility"),
        frozen_cell_policy="persistent_full_G_forward_with_zero_update_pulses",
        ranking=ranking,
        arms=arms,
        epoch_selection_metrics=(
            "validation_accuracy_descending",
            "validation_KL_ascending",
            "epoch_ascending",
        ),
        test_policy="sealed_until_all_five_arm_epoch_selections_freeze_then_terminal_once",
    )


def _protocol() -> ExactP0StructuredPartialPvAdamProtocol:
    return ExactP0StructuredPartialPvAdamProtocol(
        base=_base_protocol(),
        partial=_partial_contract(),
    )


def parse_exact_p0_structured_partial_pv_adam_config(
    payload: Mapping[str, Any],
) -> ExactP0StructuredPartialPvAdamConfig:
    raw = _object(payload, "config")
    expected = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "teacher_weights_path": EXPECTED_TEACHER_WEIGHTS_PATH,
        "teacher_weights_sha256": EXPECTED_TEACHER_WEIGHTS_SHA256,
        "target_assignment_seed": TARGET_ASSIGNMENT_SEED,
        "p0_endpoint_seed": P0_ENDPOINT_SEED,
        "p0_sha256": EXPECTED_P0_SHA256,
        "source_hwa_checkpoint_sha256": EXPECTED_SOURCE_HWA_CHECKPOINT_SHA256,
        "device_model_sha256": EXPECTED_DEVICE_MODEL_SHA256,
        "published_population_sha256": EXPECTED_PUBLISHED_POPULATION_SHA256,
        "published_population_fingerprint": EXPECTED_PUBLISHED_POPULATION_FINGERPRINT,
        "recovery_data_order_seed": RECOVERY_DATA_ORDER_SEED,
        "recovery_epochs": RECOVERY_EPOCHS,
        "recovery_learning_rate_raw_x": RECOVERY_LEARNING_RATE_RAW_X,
        "nominal_delta_x": NOMINAL_DELTA_X,
        "verify_tolerance_raw_x": VERIFY_TOLERANCE_RAW_X,
        "recovery_pulse_cap": RECOVERY_PULSE_CAP,
        "maximum_random_draws": MAXIMUM_RANDOM_DRAWS,
        "fixed_logit_gain": FIXED_LOGIT_GAIN,
        "ranking_cohort": RANKING_COHORT,
        "ranking_aggregation": RANKING_AGGREGATION,
        "ranking_tie_break": RANKING_TIE_BREAK,
        "matched_logical_count": MATCHED_LOGICAL_COUNT,
        "arms": list(ARM_IDS),
    }
    _keys(raw, "config", set(expected))
    for name, value in expected.items():
        if type(raw[name]) is not type(value) or raw[name] != value:
            raise config_error(f"config.{name}", f"to equal {value!r}", raw[name])
    return ExactP0StructuredPartialPvAdamConfig(
        schema_version=SCHEMA_VERSION,
        experiment_id=EXPERIMENT_ID,
        protocol=_protocol(),
        student=parse_student_config(_base_student_payload()),
    )


def resolve_exact_p0_structured_partial_pv_adam_spec(
    document: ExactP0StructuredPartialPvAdamConfig,
    mode: RunMode,
) -> ExactP0StructuredPartialPvAdamTrainSpec:
    if mode is not RunMode.TRAIN:
        raise config_error("mode", "to equal 'train'", mode.value)
    student = resolve_student_spec(document.student, mode)
    if not isinstance(student, StudentTrainSpec):  # pragma: no cover
        raise RuntimeError("Expected parent student training spec.")
    if student.runtime.device != "cuda":  # pragma: no cover
        raise config_error("student.runtime.device", "to equal 'cuda'", student.runtime.device)
    if (
        student.mapping.calibration_examples != RANKING_EXAMPLES
        or student.mapping.calibration_batch_size != RANKING_BATCH_SIZE
        or student.settings.num_epochs != RECOVERY_EPOCHS
    ):  # pragma: no cover - parent payload is immutable and covered upstream
        raise RuntimeError("Structured-partial parent student settings drifted.")
    return ExactP0StructuredPartialPvAdamTrainSpec(
        experiment_id=EXPERIMENT_ID,
        protocol=document.protocol,
        student=student,
    )


__all__ = [
    "ARM_IDS",
    "CANONICAL_LAYOUTS",
    "EXPERIMENT_ID",
    "ExactP0StructuredPartialPvAdamConfig",
    "ExactP0StructuredPartialPvAdamProtocol",
    "ExactP0StructuredPartialPvAdamTrainSpec",
    "MATCHED_LOGICAL_COUNT",
    "RANKING_AGGREGATION",
    "RANKING_BATCH_SIZE",
    "RANKING_COHORT",
    "RANKING_EXAMPLES",
    "RANKING_MINIBATCHES",
    "RANKING_TIE_BREAK",
    "SCHEMA_VERSION",
    "StructuredPartialArmContract",
    "StructuredPartialContract",
    "StructuredPartialRankingContract",
    "parse_exact_p0_structured_partial_pv_adam_config",
    "resolve_exact_p0_structured_partial_pv_adam_spec",
]
