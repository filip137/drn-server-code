"""Strict contract for the nested W2-plus-W1 exact-P0 P&V screen.

This exploratory successor changes only the static quad-coherent update mask.
It reuses the exact stage-1 P0, loader, objective, Adam settings, pulse plant,
and apparent-feedback controller.  Stage-1 W2-only must replay exactly, while
the stage-1 full arm is a hashed historical reference and is not rerun.
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
from experiments.mnist_relu_drn.ibm_om_exact_p0_structured_partial_pv_adam_config import (
    CANONICAL_LAYOUTS,
    RANKING_AGGREGATION,
    RANKING_BATCH_SIZE,
    RANKING_COHORT,
    RANKING_EXAMPLES,
    RANKING_MINIBATCHES,
    RANKING_TIE_BREAK,
    StructuredPartialRankingContract,
)
from experiments.schema import RunMode, config_error


EXPERIMENT_ID = "mnist_ibm_om_exact_p0_hybrid_fraction_pv_adam.v1"
SCHEMA_VERSION = 1

ARM_IDS = (
    "w2_only",
    "w2_plus_w1_top125",
    "w2_plus_w1_top250",
    "w2_plus_w1_top500",
    "w2_plus_w1_random500",
)
TOP_W1_COUNTS = (125, 250, 500)
RANDOM_SELECTOR_SEED = 94_501
RANDOM_SELECTOR_ALGORITHM = (
    "sha256(seed_u64_be||flat_W1_index_u64_be), lexicographic_digest_then_index"
)

STAGE1_SUMMARY_PATH = (
    "results/mnist-ibm-om-exact-p0-structured-partial-pv-adam-"
    "exploratory-20260901-v1/20260901T110733.146890Z-23450226-0401071f/"
    "artifacts/scientific_summary.json"
)
STAGE1_SUMMARY_SHA256 = (
    "07a2b77717e25ee720775b281d7f96227ec6a005fcea19c5a0ba9917bc84cff9"
)
STAGE1_EXPERIMENT_ID = "mnist_ibm_om_exact_p0_structured_partial_pv_adam.v1"
EXPECTED_RANDOM_W1_INDEX_SHA256 = (
    "85e1f3f7b1046d3acc8065b8460354950eda03973173159782a93f4b45b77887"
)
MINIMUM_VALIDATION_ACCURACY_GAIN = 0.01
MAXIMUM_STAGE1_FULL_ISSUED_PULSE_FRACTION = 0.25


@dataclass(frozen=True)
class Stage1PartialReferenceContract:
    summary_path: str
    summary_sha256: str
    experiment_id: str
    w2_parity_arm_id: str
    full_reference_arm_id: str
    rerun_full: bool


@dataclass(frozen=True)
class HybridFractionArmContract:
    arm_id: str
    w1_policy: str
    expected_w1_logical_quads: int
    expected_w2_logical_quads: int
    expected_physical_cells: int
    validation_candidate: bool
    terminal_test_eligible: bool


@dataclass(frozen=True)
class HybridFractionContract:
    controller: str
    epochs: int
    mask_unit: str
    mask_application: tuple[str, str]
    frozen_cell_policy: str
    physical_conductance_formula: str
    ranking: StructuredPartialRankingContract
    random_selector_seed: int
    random_selector_algorithm: str
    expected_random_w1_index_sha256: str
    arms: tuple[
        HybridFractionArmContract,
        HybridFractionArmContract,
        HybridFractionArmContract,
        HybridFractionArmContract,
        HybridFractionArmContract,
    ]
    top_w1_candidate_counts: tuple[int, int, int]
    minimum_validation_accuracy_gain: float
    require_lower_validation_kl: bool
    maximum_stage1_full_issued_pulse_fraction: float
    candidate_selection_order: str
    epoch_selection_metrics: tuple[str, str, str]
    test_policy: str
    stage1_reference: Stage1PartialReferenceContract


@dataclass(frozen=True)
class ExactP0HybridFractionPvAdamProtocol:
    base: ExactP0OpenVsClosedLoopAdamProtocol
    partial: HybridFractionContract


@dataclass(frozen=True)
class ExactP0HybridFractionPvAdamConfig:
    schema_version: int
    experiment_id: str
    protocol: ExactP0HybridFractionPvAdamProtocol
    student: StudentConfig


@dataclass(frozen=True)
class ExactP0HybridFractionPvAdamTrainSpec:
    experiment_id: str
    protocol: ExactP0HybridFractionPvAdamProtocol
    student: StudentTrainSpec


def _ranking_contract() -> StructuredPartialRankingContract:
    return StructuredPartialRankingContract(
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


def _partial_contract() -> HybridFractionContract:
    arm_w1_counts = (0, 125, 250, 500, 500)
    policies = (
        "none",
        "nested_frozen_P0_gradient_rank_prefix",
        "nested_frozen_P0_gradient_rank_prefix",
        "nested_frozen_P0_gradient_rank_prefix",
        "deterministic_random_flat_W1_addresses",
    )
    arms = tuple(
        HybridFractionArmContract(
            arm_id=arm_id,
            w1_policy=policy,
            expected_w1_logical_quads=w1_count,
            expected_w2_logical_quads=500,
            expected_physical_cells=4 * (w1_count + 500),
            validation_candidate=arm_id.startswith("w2_plus_w1_top"),
            terminal_test_eligible=(
                arm_id == "w2_only" or arm_id.startswith("w2_plus_w1_top")
            ),
        )
        for arm_id, policy, w1_count in zip(
            ARM_IDS, policies, arm_w1_counts, strict=True
        )
    )
    return HybridFractionContract(
        controller="incremental_one_pulse_closed_loop_target_tracking",
        epochs=RECOVERY_EPOCHS,
        mask_unit="canonical_four_physical_cell_logical_quad",
        mask_application=("Adam_target_updates", "pulse_eligibility"),
        frozen_cell_policy="persistent_full_G_forward_with_zero_update_pulses",
        physical_conductance_formula="nonnegative_G=a+1=2*x",
        ranking=_ranking_contract(),
        random_selector_seed=RANDOM_SELECTOR_SEED,
        random_selector_algorithm=RANDOM_SELECTOR_ALGORITHM,
        expected_random_w1_index_sha256=EXPECTED_RANDOM_W1_INDEX_SHA256,
        arms=arms,  # type: ignore[arg-type]
        top_w1_candidate_counts=TOP_W1_COUNTS,
        minimum_validation_accuracy_gain=MINIMUM_VALIDATION_ACCURACY_GAIN,
        require_lower_validation_kl=True,
        maximum_stage1_full_issued_pulse_fraction=(
            MAXIMUM_STAGE1_FULL_ISSUED_PULSE_FRACTION
        ),
        candidate_selection_order="smallest_W1_topK_first",
        epoch_selection_metrics=(
            "validation_accuracy_descending",
            "validation_KL_ascending",
            "epoch_ascending",
        ),
        test_policy=(
            "seal_test_until_all_five_validation_epoch_selections_freeze; "
            "then_open_only_w2_parity_and_smallest_qualifying_topK_union; "
            "random500_is_validation_only"
        ),
        stage1_reference=Stage1PartialReferenceContract(
            summary_path=STAGE1_SUMMARY_PATH,
            summary_sha256=STAGE1_SUMMARY_SHA256,
            experiment_id=STAGE1_EXPERIMENT_ID,
            w2_parity_arm_id="w2_only",
            full_reference_arm_id="full",
            rerun_full=False,
        ),
    )


def _protocol() -> ExactP0HybridFractionPvAdamProtocol:
    return ExactP0HybridFractionPvAdamProtocol(
        base=_base_protocol(), partial=_partial_contract()
    )


def parse_exact_p0_hybrid_fraction_pv_adam_config(
    payload: Mapping[str, Any],
) -> ExactP0HybridFractionPvAdamConfig:
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
        "top_w1_candidate_counts": list(TOP_W1_COUNTS),
        "random_w1_count": 500,
        "random_selector_seed": RANDOM_SELECTOR_SEED,
        "random_selector_algorithm": RANDOM_SELECTOR_ALGORITHM,
        "expected_random_w1_index_sha256": EXPECTED_RANDOM_W1_INDEX_SHA256,
        "minimum_validation_accuracy_gain": MINIMUM_VALIDATION_ACCURACY_GAIN,
        "require_lower_validation_kl": True,
        "maximum_stage1_full_issued_pulse_fraction": (
            MAXIMUM_STAGE1_FULL_ISSUED_PULSE_FRACTION
        ),
        "stage1_summary_path": STAGE1_SUMMARY_PATH,
        "stage1_summary_sha256": STAGE1_SUMMARY_SHA256,
        "arms": list(ARM_IDS),
    }
    _keys(raw, "config", set(expected))
    for name, value in expected.items():
        if type(raw[name]) is not type(value) or raw[name] != value:
            raise config_error(f"config.{name}", f"to equal {value!r}", raw[name])
    return ExactP0HybridFractionPvAdamConfig(
        schema_version=SCHEMA_VERSION,
        experiment_id=EXPERIMENT_ID,
        protocol=_protocol(),
        student=parse_student_config(_base_student_payload()),
    )


def resolve_exact_p0_hybrid_fraction_pv_adam_spec(
    document: ExactP0HybridFractionPvAdamConfig,
    mode: RunMode,
) -> ExactP0HybridFractionPvAdamTrainSpec:
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
    ):  # pragma: no cover
        raise RuntimeError("Hybrid-fraction parent student settings drifted.")
    return ExactP0HybridFractionPvAdamTrainSpec(
        experiment_id=EXPERIMENT_ID,
        protocol=document.protocol,
        student=student,
    )


__all__ = [
    "ARM_IDS",
    "EXPECTED_RANDOM_W1_INDEX_SHA256",
    "EXPERIMENT_ID",
    "ExactP0HybridFractionPvAdamConfig",
    "ExactP0HybridFractionPvAdamProtocol",
    "ExactP0HybridFractionPvAdamTrainSpec",
    "HybridFractionArmContract",
    "HybridFractionContract",
    "MAXIMUM_STAGE1_FULL_ISSUED_PULSE_FRACTION",
    "MINIMUM_VALIDATION_ACCURACY_GAIN",
    "SCHEMA_VERSION",
    "STAGE1_SUMMARY_PATH",
    "STAGE1_SUMMARY_SHA256",
    "Stage1PartialReferenceContract",
    "parse_exact_p0_hybrid_fraction_pv_adam_config",
    "resolve_exact_p0_hybrid_fraction_pv_adam_spec",
]
