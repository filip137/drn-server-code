"""Strict contract for the stage-3 exact-P0 hybrid P&V extension.

The extension changes only the static quad-coherent W1 update mask.  It reuses
the exact P0, frozen stage-1 ranking, loader, objective, Adam equations, pulse
plant, and apparent-feedback controller.  The stage-2 W2-plus-W1-top500 arm
must replay exactly, while the stage-1 full arm remains a hashed reference.
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


EXPERIMENT_ID = "mnist_ibm_om_exact_p0_hybrid_extension_pv_adam.v1"
SCHEMA_VERSION = 1

ANCHOR_ARM_ID = "w2_plus_w1_top500"
LARGER_TOP_W1_COUNTS = (1_000, 2_000, 4_000)
RANDOM_SELECTOR_SEED = 94_501
RANDOM_SELECTOR_ALGORITHM = (
    "sha256(seed_u64_be||flat_W1_index_u64_be), lexicographic_digest_then_index"
)
ARM_IDS = (
    ANCHOR_ARM_ID,
    "w2_plus_w1_top1000",
    "w2_plus_w1_top2000",
    "w2_plus_w1_top4000",
    "w2_plus_w1_random2000",
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
EXPECTED_STAGE1_FULL_REFERENCE_SHA256 = (
    "7d7b980c17da6bd5c8063ef53e6dad7e2ad87049c23a582b187b07a37a742155"
)

STAGE2_SUMMARY_PATH = (
    "results/mnist-ibm-om-exact-p0-hybrid-fraction-pv-adam-"
    "exploratory-20260901-v1/20260901T114836.063168Z-93bc30a9-0f3f22c2/"
    "artifacts/scientific_summary.json"
)
STAGE2_SUMMARY_SHA256 = (
    "f633568529acd09fb2e2b9c42b6d7c71ebc7d01ffdf8f7aeb33016a3db27a8c2"
)
STAGE2_EXPERIMENT_ID = "mnist_ibm_om_exact_p0_hybrid_fraction_pv_adam.v1"
EXPECTED_STAGE2_ANCHOR_EPOCH_TRAJECTORY_SHA256 = (
    "6b7989a038c09cb14f0e64a6fc490a9944cd7a6d8387b40df0610b7c05382bf9",
    "58b4c0d8f20cb89e51506b805a7c729c872a3692ab21f401df5542b4fb2abad9",
    "3ea9ff7d6cf72d44d095b0c572077fcf917637db2865f7fa1eb841af8a90e46f",
)
EXPECTED_STAGE2_ANCHOR_TEST_SHA256 = (
    "5b140e0b74b5dca4308c8912536b4f031fc4a498497385ef0ee03926bb90f422"
)
EXPECTED_STAGE2_ANCHOR_MASK_SHA256 = (
    "3bdf64e2c8546cb2fc35d60419e92700a941fb374764bc2a66e96b92fda33a2c"
)
EXPECTED_RANDOM_W1_INDEX_SHA256 = (
    "9048ca1fdae7e85ee6941a4d2553892e57ef1e6eea8d9794a7f66a5c3076d792"
)
MINIMUM_VALIDATION_ACCURACY_GAIN = 0.01
MAXIMUM_STAGE1_FULL_ISSUED_PULSE_FRACTION = 0.25


@dataclass(frozen=True)
class FrozenSummaryReferenceContract:
    summary_path: str
    summary_sha256: str
    experiment_id: str


@dataclass(frozen=True)
class Stage2AnchorReferenceContract:
    summary: FrozenSummaryReferenceContract
    anchor_arm_id: str
    expected_epoch_trajectory_sha256: tuple[str, str, str]
    expected_selected_test_sha256: str
    expected_flat_physical_mask_sha256: str
    rerun_anchor: bool


@dataclass(frozen=True)
class HybridExtensionArmContract:
    arm_id: str
    w1_policy: str
    expected_w1_logical_quads: int
    expected_w2_logical_quads: int
    expected_physical_cells: int
    validation_candidate: bool
    terminal_test_eligible: bool


@dataclass(frozen=True)
class HybridExtensionContract:
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
        HybridExtensionArmContract,
        HybridExtensionArmContract,
        HybridExtensionArmContract,
        HybridExtensionArmContract,
        HybridExtensionArmContract,
    ]
    larger_top_w1_candidate_counts: tuple[int, int, int]
    minimum_validation_accuracy_gain: float
    require_lower_validation_kl: bool
    maximum_stage1_full_issued_pulse_fraction: float
    candidate_selection_order: str
    epoch_selection_metrics: tuple[str, str, str]
    test_policy: str
    stage1_reference: FrozenSummaryReferenceContract
    stage1_full_reference_arm_id: str
    expected_stage1_full_reference_sha256: str
    rerun_stage1_full: bool
    stage2_reference: Stage2AnchorReferenceContract


@dataclass(frozen=True)
class ExactP0HybridExtensionPvAdamProtocol:
    base: ExactP0OpenVsClosedLoopAdamProtocol
    partial: HybridExtensionContract


@dataclass(frozen=True)
class ExactP0HybridExtensionPvAdamConfig:
    schema_version: int
    experiment_id: str
    protocol: ExactP0HybridExtensionPvAdamProtocol
    student: StudentConfig


@dataclass(frozen=True)
class ExactP0HybridExtensionPvAdamTrainSpec:
    experiment_id: str
    protocol: ExactP0HybridExtensionPvAdamProtocol
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


def _partial_contract() -> HybridExtensionContract:
    arm_w1_counts = (500, 1_000, 2_000, 4_000, 2_000)
    policies = (
        "nested_frozen_P0_gradient_rank_prefix",
        "nested_frozen_P0_gradient_rank_prefix",
        "nested_frozen_P0_gradient_rank_prefix",
        "nested_frozen_P0_gradient_rank_prefix",
        "deterministic_random_flat_W1_addresses",
    )
    arms = tuple(
        HybridExtensionArmContract(
            arm_id=arm_id,
            w1_policy=policy,
            expected_w1_logical_quads=w1_count,
            expected_w2_logical_quads=500,
            expected_physical_cells=4 * (w1_count + 500),
            validation_candidate=arm_id.startswith("w2_plus_w1_top")
            and arm_id != ANCHOR_ARM_ID,
            terminal_test_eligible=arm_id != "w2_plus_w1_random2000",
        )
        for arm_id, policy, w1_count in zip(
            ARM_IDS, policies, arm_w1_counts, strict=True
        )
    )
    stage1_reference = FrozenSummaryReferenceContract(
        summary_path=STAGE1_SUMMARY_PATH,
        summary_sha256=STAGE1_SUMMARY_SHA256,
        experiment_id=STAGE1_EXPERIMENT_ID,
    )
    stage2_reference = Stage2AnchorReferenceContract(
        summary=FrozenSummaryReferenceContract(
            summary_path=STAGE2_SUMMARY_PATH,
            summary_sha256=STAGE2_SUMMARY_SHA256,
            experiment_id=STAGE2_EXPERIMENT_ID,
        ),
        anchor_arm_id=ANCHOR_ARM_ID,
        expected_epoch_trajectory_sha256=(
            EXPECTED_STAGE2_ANCHOR_EPOCH_TRAJECTORY_SHA256
        ),
        expected_selected_test_sha256=EXPECTED_STAGE2_ANCHOR_TEST_SHA256,
        expected_flat_physical_mask_sha256=EXPECTED_STAGE2_ANCHOR_MASK_SHA256,
        rerun_anchor=True,
    )
    return HybridExtensionContract(
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
        larger_top_w1_candidate_counts=LARGER_TOP_W1_COUNTS,
        minimum_validation_accuracy_gain=MINIMUM_VALIDATION_ACCURACY_GAIN,
        require_lower_validation_kl=True,
        maximum_stage1_full_issued_pulse_fraction=(
            MAXIMUM_STAGE1_FULL_ISSUED_PULSE_FRACTION
        ),
        candidate_selection_order="smallest_larger_W1_topK_first",
        epoch_selection_metrics=(
            "validation_accuracy_descending",
            "validation_KL_ascending",
            "epoch_ascending",
        ),
        test_policy=(
            "seal_test_until_all_five_validation_epoch_selections_freeze; "
            "then_open_only_stage2_top500_parity_anchor_and_smallest_qualifying_"
            "larger_topK_union; random2000_is_validation_only"
        ),
        stage1_reference=stage1_reference,
        stage1_full_reference_arm_id="full",
        expected_stage1_full_reference_sha256=(
            EXPECTED_STAGE1_FULL_REFERENCE_SHA256
        ),
        rerun_stage1_full=False,
        stage2_reference=stage2_reference,
    )


def _protocol() -> ExactP0HybridExtensionPvAdamProtocol:
    return ExactP0HybridExtensionPvAdamProtocol(
        base=_base_protocol(), partial=_partial_contract()
    )


def parse_exact_p0_hybrid_extension_pv_adam_config(
    payload: Mapping[str, Any],
) -> ExactP0HybridExtensionPvAdamConfig:
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
        "anchor_w1_count": 500,
        "larger_top_w1_candidate_counts": list(LARGER_TOP_W1_COUNTS),
        "random_w1_count": 2_000,
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
        "expected_stage1_full_reference_sha256": (
            EXPECTED_STAGE1_FULL_REFERENCE_SHA256
        ),
        "stage2_summary_path": STAGE2_SUMMARY_PATH,
        "stage2_summary_sha256": STAGE2_SUMMARY_SHA256,
        "expected_stage2_anchor_epoch_trajectory_sha256": list(
            EXPECTED_STAGE2_ANCHOR_EPOCH_TRAJECTORY_SHA256
        ),
        "expected_stage2_anchor_test_sha256": (
            EXPECTED_STAGE2_ANCHOR_TEST_SHA256
        ),
        "expected_stage2_anchor_mask_sha256": (
            EXPECTED_STAGE2_ANCHOR_MASK_SHA256
        ),
        "arms": list(ARM_IDS),
    }
    _keys(raw, "config", set(expected))
    for name, value in expected.items():
        if type(raw[name]) is not type(value) or raw[name] != value:
            raise config_error(f"config.{name}", f"to equal {value!r}", raw[name])
    return ExactP0HybridExtensionPvAdamConfig(
        schema_version=SCHEMA_VERSION,
        experiment_id=EXPERIMENT_ID,
        protocol=_protocol(),
        student=parse_student_config(_base_student_payload()),
    )


def resolve_exact_p0_hybrid_extension_pv_adam_spec(
    document: ExactP0HybridExtensionPvAdamConfig,
    mode: RunMode,
) -> ExactP0HybridExtensionPvAdamTrainSpec:
    if mode is not RunMode.TRAIN:
        raise config_error("mode", "to equal 'train'", mode.value)
    student = resolve_student_spec(document.student, mode)
    if not isinstance(student, StudentTrainSpec):  # pragma: no cover
        raise RuntimeError("Expected parent student training spec.")
    if student.runtime.device != "cuda":  # pragma: no cover
        raise config_error(
            "student.runtime.device", "to equal 'cuda'", student.runtime.device
        )
    if (
        student.mapping.calibration_examples != RANKING_EXAMPLES
        or student.mapping.calibration_batch_size != RANKING_BATCH_SIZE
        or student.settings.num_epochs != RECOVERY_EPOCHS
    ):  # pragma: no cover
        raise RuntimeError("Hybrid-extension parent student settings drifted.")
    return ExactP0HybridExtensionPvAdamTrainSpec(
        experiment_id=EXPERIMENT_ID,
        protocol=document.protocol,
        student=student,
    )


__all__ = [
    "ARM_IDS",
    "EXPECTED_RANDOM_W1_INDEX_SHA256",
    "EXPECTED_STAGE1_FULL_REFERENCE_SHA256",
    "EXPECTED_STAGE2_ANCHOR_EPOCH_TRAJECTORY_SHA256",
    "EXPECTED_STAGE2_ANCHOR_MASK_SHA256",
    "EXPECTED_STAGE2_ANCHOR_TEST_SHA256",
    "EXPERIMENT_ID",
    "ExactP0HybridExtensionPvAdamConfig",
    "ExactP0HybridExtensionPvAdamProtocol",
    "ExactP0HybridExtensionPvAdamTrainSpec",
    "FrozenSummaryReferenceContract",
    "HybridExtensionArmContract",
    "HybridExtensionContract",
    "MAXIMUM_STAGE1_FULL_ISSUED_PULSE_FRACTION",
    "MINIMUM_VALIDATION_ACCURACY_GAIN",
    "SCHEMA_VERSION",
    "STAGE1_SUMMARY_PATH",
    "STAGE1_SUMMARY_SHA256",
    "STAGE2_SUMMARY_PATH",
    "STAGE2_SUMMARY_SHA256",
    "Stage2AnchorReferenceContract",
    "parse_exact_p0_hybrid_extension_pv_adam_config",
    "resolve_exact_p0_hybrid_extension_pv_adam_spec",
]
