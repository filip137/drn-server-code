"""CUDA runtime for the nested W2-plus-W1 exact-P0 P&V screen."""

from __future__ import annotations

import copy
import gc
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping, TYPE_CHECKING

import torch

from experiments.artifacts import RunStore, atomic_write_json, sha256_file
from experiments.mnist_relu_drn.components import build_student_stack
from experiments.mnist_relu_drn.ibm_om_baseline_selection_runtime import (
    _artifact_records,
    _input,
)
from experiments.mnist_relu_drn.ibm_om_exact_p0_hybrid_fraction_pv_adam import (
    ARM_IDS,
    RANDOM_SELECTOR_ALGORITHM,
    RANDOM_SELECTOR_SEED,
    TOP_W1_COUNTS,
    HybridFractionUpdateMask,
    build_hybrid_fraction_update_masks,
)
from experiments.mnist_relu_drn.ibm_om_exact_p0_open_vs_closed_loop_adam_runtime import (
    EVIDENCE_TIER,
    _evaluate_payload_test,
    _load_device_model,
    _load_exact_p0,
    _run_arm,
    _strict_protocol as _strict_base_protocol,
)
from experiments.mnist_relu_drn.ibm_om_exact_p0_structured_partial_pv_adam import (
    LAYOUTS,
)
from experiments.mnist_relu_drn.ibm_om_exact_p0_structured_partial_pv_adam_runtime import (
    CLOSED_WRITER_ARM_ID,
    _rank_partial_masks,
    _selected_epoch_row,
)
from experiments.mnist_relu_drn.runtime import _load_teacher
from experiments.mnist_shared import build_mnist_loaders
from experiments.schema import to_plain_data
from training.checkpoint import atomic_torch_save
from training.ibm_reram_raw_active_program_verify import (
    array_population_as_pulse_population,
)


if TYPE_CHECKING:
    from ebl.cli import TrainRequest


_ROOT = Path(__file__).resolve().parents[2]
SUMMARY_SCHEMA = "ebl.mnist_relu_drn.ibm_om_exact_p0_hybrid_fraction_pv_adam"
SUMMARY_SCHEMA_VERSION = 1
STAGE1_SUMMARY_SCHEMA = (
    "ebl.mnist_relu_drn.ibm_om_exact_p0_structured_partial_pv_adam"
)


def _strict_protocol(protocol: Any) -> None:
    _strict_base_protocol(protocol.base)
    partial = protocol.partial
    if (
        tuple(arm.arm_id for arm in partial.arms) != ARM_IDS
        or partial.controller != CLOSED_WRITER_ARM_ID
        or partial.epochs != 3
        or partial.mask_unit != "canonical_four_physical_cell_logical_quad"
        or partial.mask_application != ("Adam_target_updates", "pulse_eligibility")
        or partial.physical_conductance_formula != "nonnegative_G=a+1=2*x"
        or partial.ranking.examples != 1024
        or partial.ranking.batch_size != 128
        or partial.ranking.minibatches != 8
        or partial.ranking.canonical_layouts != LAYOUTS
        or partial.ranking.uses_new_seed
        or partial.ranking.uses_device_bounds
        or partial.ranking.uses_corrupt_mask
        or partial.ranking.uses_validation
        or partial.ranking.uses_test
        or partial.random_selector_algorithm != RANDOM_SELECTOR_ALGORITHM
        or partial.random_selector_seed != RANDOM_SELECTOR_SEED
        or partial.top_w1_candidate_counts != TOP_W1_COUNTS
        or partial.minimum_validation_accuracy_gain != 0.01
        or not partial.require_lower_validation_kl
        or partial.maximum_stage1_full_issued_pulse_fraction != 0.25
        or partial.stage1_reference.rerun_full
    ):
        raise ValueError("Resolved hybrid-fraction recovery protocol drifted.")
    expected = {
        "w2_only": (0, 500, 2_000, False, True),
        "w2_plus_w1_top125": (125, 500, 2_500, True, True),
        "w2_plus_w1_top250": (250, 500, 3_000, True, True),
        "w2_plus_w1_top500": (500, 500, 4_000, True, True),
        "w2_plus_w1_random500": (500, 500, 4_000, False, False),
    }
    for arm in partial.arms:
        observed = (
            arm.expected_w1_logical_quads,
            arm.expected_w2_logical_quads,
            arm.expected_physical_cells,
            arm.validation_candidate,
            arm.terminal_test_eligible,
        )
        if observed != expected[arm.arm_id]:
            raise ValueError("Hybrid-fraction arm contract drifted.")


def _load_stage1_reference(protocol: Any) -> tuple[Path, Mapping[str, Any]]:
    contract = protocol.partial.stage1_reference
    path = (_ROOT / contract.summary_path).resolve()
    try:
        path.relative_to(_ROOT)
    except ValueError as error:  # pragma: no cover - fixed config is in-tree
        raise ValueError("Stage-1 reference escaped the repository root.") from error
    if not path.is_file() or sha256_file(path) != contract.summary_sha256:
        raise ValueError("Frozen stage-1 scientific-summary SHA-256 mismatch.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema") != STAGE1_SUMMARY_SCHEMA
        or payload.get("schema_version") != 1
        or payload.get("status") != "completed"
        or payload.get("evidence_tier") != EVIDENCE_TIER
        or payload.get("exact_p0", {}).get("sha256") != protocol.base.p0.sha256
        or payload.get("protocol", {}).get("arms")
        != ["full", "w1_only", "w2_only", "w1_top500", "global_top500"]
        or not payload.get("matched_gates", {}).get(
            "test_sealed_until_all_epoch_selections_frozen"
        )
        or not payload.get("matched_gates", {}).get(
            "full_mask_historical_parity_exact"
        )
    ):
        raise ValueError("Frozen stage-1 reference contract is invalid.")
    return path, payload


def _json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _epoch_trajectory_sha256(row: Mapping[str, Any]) -> str:
    canonical = copy.deepcopy(dict(row))
    canonical.pop("checkpoint", None)
    canonical.pop("checkpoint_sha256", None)
    return _json_sha256(canonical)


def _assert_w2_historical_epoch_parity(
    report: Mapping[str, Any], *, stage1: Mapping[str, Any]
) -> Mapping[str, Any]:
    expected = stage1["arms"]["w2_only"]
    rows = report["epoch_reports"]
    expected_rows = expected["epoch_reports"]
    if (
        len(rows) != 3
        or len(expected_rows) != 3
        or int(report["selected_epoch"]) != int(expected["selected_epoch"])
    ):
        raise RuntimeError("W2-only historical epoch coverage changed.")
    observed_hashes = [_epoch_trajectory_sha256(row) for row in rows]
    expected_hashes = [_epoch_trajectory_sha256(row) for row in expected_rows]
    if observed_hashes != expected_hashes:
        raise RuntimeError("W2-only trajectory hash no longer matches stage 1.")
    return {
        "stage1_epoch_trajectory_sha256": expected_hashes,
        "observed_epoch_trajectory_sha256": observed_hashes,
        "selected_epoch_exact": True,
    }


def _assert_w2_historical_test_parity(
    test: Mapping[str, Any], *, stage1: Mapping[str, Any]
) -> Mapping[str, Any]:
    expected = stage1["arms"]["w2_only"]["selected_test"]
    expected_hash = _json_sha256(expected)
    observed_hash = _json_sha256(test)
    if observed_hash != expected_hash:
        raise RuntimeError("W2-only terminal test hash no longer matches stage 1.")
    return {
        "stage1_selected_test_sha256": expected_hash,
        "observed_selected_test_sha256": observed_hash,
        "exact": True,
    }


def _pairwise_overlap(
    masks: Mapping[str, HybridFractionUpdateMask],
) -> Mapping[str, Mapping[str, Any]]:
    overlap: dict[str, Mapping[str, Any]] = {}
    for left_index, left_id in enumerate(ARM_IDS):
        left = torch.cat(
            tuple(value.reshape(-1) for value in masks[left_id].logical_masks)
        )
        for right_id in ARM_IDS[left_index + 1 :]:
            right = torch.cat(
                tuple(value.reshape(-1) for value in masks[right_id].logical_masks)
            )
            intersection = int((left & right).sum().item())
            union = int((left | right).sum().item())
            overlap[f"{left_id}__{right_id}"] = {
                "logical_intersection": intersection,
                "logical_union": union,
                "logical_jaccard": intersection / union,
                "identical": bool(torch.equal(left, right)),
            }
    return overlap


def _save_mask_artifacts(
    *,
    store: RunStore,
    p0_sha256: str,
    masks: Mapping[str, HybridFractionUpdateMask],
    ranking_payload: Mapping[str, Any],
    provenance: Mapping[str, Any],
    stage1: Mapping[str, Any],
    protocol: Any,
) -> tuple[list[Mapping[str, Any]], Mapping[str, Any]]:
    stage1_ranking = stage1["mask_artifact"]["ranking"]
    ranking = ranking_payload["ranking"]
    for key in (
        "cohort_sha256",
        "input_batches_sha256",
        "label_batches_sha256",
        "score_sha256_by_layer",
    ):
        if ranking[key] != stage1_ranking[key]:
            raise RuntimeError("P0 gradient ranking no longer matches stage 1.")
    w2_report = masks["w2_only"].report()
    if (
        w2_report["flat_physical_mask_sha256"]
        != stage1["mask_artifact"]["arms"]["w2_only"][
            "flat_physical_mask_sha256"
        ]
    ):
        raise RuntimeError("W2-only mask no longer matches stage 1.")
    random_report = masks["w2_plus_w1_random500"].report()
    if (
        random_report["selected_w1_flat_indices_sha256"]
        != protocol.partial.expected_random_w1_index_sha256
    ):
        raise RuntimeError("Deterministic random-W1 selector hash changed.")
    overlap = _pairwise_overlap(masks)
    payload = {
        "schema": "ebl.mnist_relu_drn.ibm_om_hybrid_fraction_update_masks",
        "schema_version": 1,
        "evidence_tier": EVIDENCE_TIER,
        "source_p0_sha256": p0_sha256,
        "provenance": dict(provenance),
        "ranking": ranking,
        "quad_scores": [
            value.detach().cpu() for value in ranking_payload["scores"]
        ],
        "selector_contract": {
            "top_W1_counts": list(TOP_W1_COUNTS),
            "top_prefixes_nested": True,
            "random_selector_seed": protocol.partial.random_selector_seed,
            "random_selector_algorithm": protocol.partial.random_selector_algorithm,
            "random_selected_W1_indices_sha256": random_report[
                "selected_w1_flat_indices_sha256"
            ],
            "uses_device_bounds": False,
            "uses_corrupt_mask": False,
            "uses_P0_full_nonnegative_conductances_for_gradient_forward": True,
            "uses_hidden_plant_parameters_or_rng_state": False,
            "uses_validation": False,
            "uses_test": False,
        },
        "arms": {
            arm_id: {
                "logical_masks": [
                    value.detach().cpu() for value in mask.logical_masks
                ],
                "physical_masks": [
                    value.detach().cpu() for value in mask.physical_masks
                ],
                "report": mask.report(),
            }
            for arm_id, mask in masks.items()
        },
        "pairwise_logical_mask_overlap": overlap,
    }
    path = atomic_torch_save(
        payload, store.run_dir / "artifacts/hybrid_fraction_update_masks.pt"
    )
    receipt = {
        "schema": payload["schema"] + ".receipt",
        "schema_version": 1,
        "path": str(path),
        "sha256": sha256_file(path),
        "source_p0_sha256": p0_sha256,
        "provenance": dict(provenance),
        "ranking": ranking,
        "selector_contract": payload["selector_contract"],
        "arms": {arm_id: mask.report() for arm_id, mask in masks.items()},
        "pairwise_logical_mask_overlap": overlap,
        "stage1_ranking_hash_parity": True,
        "stage1_w2_mask_hash_parity": True,
    }
    receipt_path = store.run_dir / "artifacts/hybrid_fraction_update_masks.receipt.json"
    atomic_write_json(receipt_path, receipt)
    return (
        [
            {"path": str(path), "kind": "hybrid_fraction_update_masks"},
            {
                "path": str(receipt_path),
                "kind": "hybrid_fraction_update_masks_receipt",
            },
        ],
        receipt,
    )


def _select_smallest_qualifying_union(
    arm_reports: Mapping[str, Mapping[str, Any]],
    *,
    stage1: Mapping[str, Any],
    protocol: Any,
) -> tuple[str | None, Mapping[str, Mapping[str, Any]]]:
    w2_selected = _selected_epoch_row(arm_reports["w2_only"])
    w2_accuracy = float(w2_selected["validation"]["student_accuracy"])
    w2_correct = int(w2_selected["validation"]["student_correct"])
    validation_examples = int(w2_selected["validation"]["examples"])
    minimum_correct_gain = math.ceil(
        protocol.partial.minimum_validation_accuracy_gain * validation_examples
    )
    w2_kl = float(w2_selected["validation"]["kl_teacher_student"])
    stage1_full_pulses = int(
        stage1["comparison"]["by_arm"]["full"]["selected_checkpoint_cost"][
            "issued_commands"
        ]
    )
    gates: dict[str, Mapping[str, Any]] = {}
    winner: str | None = None
    for count in protocol.partial.top_w1_candidate_counts:
        arm_id = f"w2_plus_w1_top{count}"
        selected = _selected_epoch_row(arm_reports[arm_id])
        accuracy = float(selected["validation"]["student_accuracy"])
        correct = int(selected["validation"]["student_correct"])
        examples = int(selected["validation"]["examples"])
        if examples != validation_examples:
            raise RuntimeError("Hybrid-fraction validation denominators changed.")
        kl_value = float(selected["validation"]["kl_teacher_student"])
        pulses = int(selected["issued_commands_cumulative"])
        accuracy_gain = accuracy - w2_accuracy
        correct_gain = correct - w2_correct
        accuracy_gate = correct_gain >= minimum_correct_gain
        kl_gate = kl_value < w2_kl
        pulse_fraction = pulses / stage1_full_pulses
        pulse_gate = (
            pulse_fraction
            <= protocol.partial.maximum_stage1_full_issued_pulse_fraction
        )
        qualifies = accuracy_gate and kl_gate and pulse_gate
        gates[arm_id] = {
            "selected_epoch": int(arm_reports[arm_id]["selected_epoch"]),
            "validation_accuracy": accuracy,
            "validation_accuracy_gain_over_exact_w2": accuracy_gain,
            "validation_correct_gain_over_exact_w2": correct_gain,
            "minimum_required_validation_correct_gain": minimum_correct_gain,
            "minimum_required_validation_accuracy_gain": (
                protocol.partial.minimum_validation_accuracy_gain
            ),
            "validation_accuracy_gate": accuracy_gate,
            "validation_KL": kl_value,
            "exact_w2_validation_KL": w2_kl,
            "lower_validation_KL_gate": kl_gate,
            "selected_issued_commands": pulses,
            "stage1_full_selected_issued_commands": stage1_full_pulses,
            "issued_command_fraction_of_stage1_full": pulse_fraction,
            "maximum_allowed_stage1_full_fraction": (
                protocol.partial.maximum_stage1_full_issued_pulse_fraction
            ),
            "pulse_gate": pulse_gate,
            "qualifies": qualifies,
        }
        if winner is None and qualifies:
            winner = arm_id
    return winner, gates


def run_train(request: "TrainRequest") -> int:
    from experiments.mnist_relu_drn.ibm_om_exact_p0_hybrid_fraction_pv_adam_config import (
        ExactP0HybridFractionPvAdamTrainSpec,
    )

    spec = request.spec
    if not isinstance(spec, ExactP0HybridFractionPvAdamTrainSpec):
        raise TypeError("Expected the dedicated hybrid-fraction recovery train spec.")
    if request.teacher_weights is None or request.weights is None or request.device_model is None:
        raise ValueError("Expected explicit --teacher-weights, --weights exact P0, and --device-model.")
    for name in ("base_weights", "resume", "device_data"):
        if getattr(request, name) is not None:
            raise ValueError(f"Hybrid-fraction recovery does not accept {name}.")
    protocol = spec.protocol
    _strict_protocol(protocol)
    base = protocol.base
    teacher_path = request.teacher_weights.expanduser().resolve()
    p0_path = request.weights.expanduser().resolve()
    device_model_path = request.device_model.expanduser().resolve()
    if not teacher_path.is_file() or sha256_file(teacher_path) != base.expected_teacher_weights_sha256:
        raise ValueError("Frozen teacher SHA-256 mismatch.")
    stage1_path, stage1 = _load_stage1_reference(protocol)
    population, population_path, device_receipt = _load_device_model(
        device_model_path, base
    )
    p0 = _load_exact_p0(p0_path, protocol=base, population=population)
    pulse_population = array_population_as_pulse_population(population)
    store = RunStore.create(
        output_root=request.output_dir,
        experiment_id=spec.experiment_id,
        resolved_config=to_plain_data(spec),
        command=request.command,
        repo_root=_ROOT,
        input_artifacts=(
            _input("teacher_weights", teacher_path),
            _input("exact_target_p0", p0_path),
            _input("paired_target_device_model", device_model_path),
            _input("published_target_population", population_path),
            _input("stage1_partial_summary_reference", stage1_path),
        ),
        resume_capability="unsupported",
    )
    try:
        student = spec.student
        if student.runtime.device != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Hybrid-fraction P&V recovery requires CUDA.")
        torch.manual_seed(student.runtime.seed)
        torch.cuda.manual_seed_all(student.runtime.seed)
        loaders = build_mnist_loaders(
            student.data,
            data_seed=student.runtime.data_seed,
            calibration_examples=student.mapping.calibration_examples,
            calibration_batch_size=student.mapping.calibration_batch_size,
        )
        teacher, teacher_metadata = _load_teacher(
            teacher_path, device=torch.device("cuda"), spec=student
        )
        stack = build_student_stack(student, enable_measured=False)
        stack.cost.gain = base.recovery.fixed_logit_gain
        masks_raw, ranking_payload = _rank_partial_masks(
            p0=p0,
            population=population,
            pulse_population=pulse_population,
            stack=stack,
            teacher=teacher,
            loaders=loaders,
            protocol=protocol,
            mask_builder=build_hybrid_fraction_update_masks,
            mask_builder_kwargs={
                "random_selector_seed": protocol.partial.random_selector_seed
            },
        )
        masks = {arm_id: masks_raw[arm_id] for arm_id in ARM_IDS}
        generated_artifacts, mask_receipt = _save_mask_artifacts(
            store=store,
            p0_sha256=base.p0.sha256,
            masks=masks,
            ranking_payload=ranking_payload,
            provenance={
                "teacher_weights_sha256": sha256_file(teacher_path),
                "exact_p0_sha256": sha256_file(p0_path),
                "source_hwa_checkpoint_sha256": base.p0.source_hwa_checkpoint_sha256,
                "paired_device_model_receipt_sha256": sha256_file(device_model_path),
                "published_population_sha256": sha256_file(population_path),
                "published_population_fingerprint": population.fingerprint,
                "stage1_summary_path": str(stage1_path),
                "stage1_summary_sha256": sha256_file(stage1_path),
            },
            stage1=stage1,
            protocol=protocol,
        )
        arm_reports: dict[str, Mapping[str, Any]] = {}
        selected_payloads: dict[str, Mapping[str, Any]] = {}
        w2_epoch_parity: Mapping[str, Any] | None = None
        for arm_id in ARM_IDS:
            contract = next(
                value for value in protocol.partial.arms if value.arm_id == arm_id
            )
            mask = masks[arm_id]
            observed = mask.report()
            if (
                observed["logical_weights_by_layer"]
                != [
                    contract.expected_w1_logical_quads,
                    contract.expected_w2_logical_quads,
                ]
                or int(observed["physical_cells"])
                != contract.expected_physical_cells
            ):
                raise RuntimeError("Frozen hybrid-fraction mask count differs from config.")
            report, selected_payload = _run_arm(
                CLOSED_WRITER_ARM_ID,
                artifact_arm_id=arm_id,
                trainable_cell_mask=mask.flat_physical_mask,
                p0=p0,
                population=population,
                pulse_population=pulse_population,
                stack=stack,
                teacher=teacher,
                loaders=loaders,
                protocol=base,
                store=store,
                generated_artifacts=generated_artifacts,
            )
            arm_reports[arm_id] = report
            selected_payloads[arm_id] = selected_payload
            if arm_id == "w2_only":
                w2_epoch_parity = _assert_w2_historical_epoch_parity(
                    report, stage1=stage1
                )
        order_hashes = {
            arm_id: [
                row["train_generator_state_sha256"]
                for row in report["epoch_reports"]
            ]
            for arm_id, report in arm_reports.items()
        }
        if any(value != order_hashes["w2_only"] for value in order_hashes.values()):
            raise RuntimeError("Hybrid-fraction arms consumed different minibatch orders.")

        # Freeze every validation-only epoch choice before choosing the smallest
        # qualifying union or opening the current run's test loader.
        selected_union, qualification = _select_smallest_qualifying_union(
            arm_reports, stage1=stage1, protocol=protocol
        )
        test_arm_ids = ["w2_only"]
        if selected_union is not None:
            test_arm_ids.append(selected_union)
        terminal_tests: dict[str, Mapping[str, Any]] = {}
        for arm_id in test_arm_ids:
            terminal_tests[arm_id] = _evaluate_payload_test(
                selected_payloads[arm_id],
                population=population,
                pulse_population=pulse_population,
                stack=stack,
                teacher=teacher,
                test_loader=loaders.test,
                maximum_random_draws=base.recovery.maximum_random_draws,
            )
        w2_test_parity = _assert_w2_historical_test_parity(
            terminal_tests["w2_only"], stage1=stage1
        )
        for arm_id in ARM_IDS:
            arm_reports[arm_id] = {
                **dict(arm_reports[arm_id]),
                "selected_test": terminal_tests.get(arm_id),
                "test_opened": arm_id in terminal_tests,
            }

        full_reference = stage1["comparison"]["by_arm"]["full"]
        w2_selected = _selected_epoch_row(arm_reports["w2_only"])
        w2_val_accuracy = float(w2_selected["validation"]["student_accuracy"])
        w2_val_kl = float(w2_selected["validation"]["kl_teacher_student"])
        full_val_accuracy = float(full_reference["validation_accuracy"])
        full_test_accuracy = float(full_reference["test_accuracy"])
        full_pulses = int(
            full_reference["selected_checkpoint_cost"]["issued_commands"]
        )
        p0_historical_test_accuracy = float(
            stage1["exact_p0"]["test"]["student_accuracy"]
        )
        comparisons: dict[str, Mapping[str, Any]] = {}
        for arm_id in ARM_IDS:
            selected = _selected_epoch_row(arm_reports[arm_id])
            rows = list(arm_reports[arm_id]["epoch_reports"])
            selected_epoch = int(arm_reports[arm_id]["selected_epoch"])
            selected_rows = [row for row in rows if int(row["epoch"]) <= selected_epoch]
            final_row = rows[-1]
            val_accuracy = float(selected["validation"]["student_accuracy"])
            val_kl = float(selected["validation"]["kl_teacher_student"])
            pulses = int(selected["issued_commands_cumulative"])
            selected_api_calls = sum(
                int(row["api_full_port_verify_calls_epoch"]) for row in selected_rows
            )
            full_horizon_api_calls = sum(
                int(row["api_full_port_verify_calls_epoch"]) for row in rows
            )
            test = terminal_tests.get(arm_id)
            comparisons[arm_id] = {
                "selected_epoch": selected_epoch,
                "validation_correct": int(selected["validation"]["student_correct"]),
                "validation_examples": int(selected["validation"]["examples"]),
                "validation_accuracy": val_accuracy,
                "validation_KL": val_kl,
                "validation_accuracy_gain_over_exact_w2": val_accuracy - w2_val_accuracy,
                "validation_KL_minus_exact_w2": val_kl - w2_val_kl,
                "validation_minus_stage1_full": val_accuracy - full_val_accuracy,
                "selected_checkpoint_cost": {
                    "issued_commands": pulses,
                    "effective_state_change_events": int(
                        selected["effective_state_change_events_cumulative"]
                    ),
                    "api_full_port_verify_calls": selected_api_calls,
                    "software_full_port_values_materialized": (
                        selected_api_calls * population.size
                    ),
                    "conceptual_pulsed_cell_observations": sum(
                        int(row["conceptual_post_pulse_cell_observations_epoch"])
                        for row in selected_rows
                    ),
                },
                "executed_three_epoch_selection_cost": {
                    "issued_commands": int(final_row["issued_commands_cumulative"]),
                    "effective_state_change_events": int(
                        final_row["effective_state_change_events_cumulative"]
                    ),
                    "api_full_port_verify_calls": full_horizon_api_calls,
                    "software_full_port_values_materialized": (
                        full_horizon_api_calls * population.size
                    ),
                    "conceptual_pulsed_cell_observations": sum(
                        int(row["conceptual_post_pulse_cell_observations_epoch"])
                        for row in rows
                    ),
                    "discarded_post_selection_issued_commands": (
                        int(final_row["issued_commands_cumulative"]) - pulses
                    ),
                },
                "selected_issued_commands_fraction_of_stage1_full": pulses / full_pulses,
                "qualification": qualification.get(arm_id),
                "test_opened": test is not None,
                "test": (
                    None
                    if test is None
                    else {
                        "correct": int(test["student_correct"]),
                        "examples": int(test["examples"]),
                        "accuracy": float(test["student_accuracy"]),
                        "KL": float(test["kl_teacher_student"]),
                        "teacher_agreement": float(test["teacher_agreement"]),
                        "prediction_sha256": test["prediction_sha256"],
                        "minus_stage1_full": (
                            float(test["student_accuracy"]) - full_test_accuracy
                        ),
                        "recovered_stage1_full_gain_fraction": (
                            (float(test["student_accuracy"]) - p0_historical_test_accuracy)
                            / (full_test_accuracy - p0_historical_test_accuracy)
                        ),
                    }
                ),
            }
        summary = {
            "schema": SUMMARY_SCHEMA,
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "evidence_tier": EVIDENCE_TIER,
            "status": "completed",
            "question": (
                "What is the smallest nested P0-gradient-ranked W1 prefix that, "
                "when added to all of W2, materially improves exact-P0 recovery?"
            ),
            "teacher": {
                "path": str(teacher_path),
                "sha256": sha256_file(teacher_path),
                **teacher_metadata,
            },
            "exact_p0": {
                "path": str(p0_path),
                "sha256": sha256_file(p0_path),
                "assignment_seed": base.p0.assignment_seed,
                "endpoint_seed": base.p0.endpoint_seed,
                "historical_stage1_test_reference": stage1["exact_p0"]["test"],
                "current_run_p0_test_opened": False,
                "initial_programming_cost": {
                    "issued_pulses": int(p0["programming"]["total_pulses"].sum().item()),
                    "verify_observations": int(
                        p0["programming"]["verify_count"].sum().item()
                    ),
                    "excluded_from_recovery_arm_cost_ratios": True,
                },
            },
            "device_model": {
                "receipt_path": str(device_model_path),
                "receipt_sha256": sha256_file(device_model_path),
                "population_path": str(population_path),
                "population_sha256": sha256_file(population_path),
                "population_fingerprint": population.fingerprint,
                "corrupt_cells": int(population.corrupt.sum().item()),
                "receipt_role": device_receipt["role"],
            },
            "stage1_reference": {
                "path": str(stage1_path),
                "sha256": sha256_file(stage1_path),
                "full_arm": full_reference,
                "full_arm_rerun": False,
                "W2_epoch_parity": w2_epoch_parity,
                "W2_test_parity": w2_test_parity,
            },
            "mask_artifact": mask_receipt,
            "protocol": {
                "arms": list(ARM_IDS),
                "mask_unit": protocol.partial.mask_unit,
                "frozen_cell_policy": protocol.partial.frozen_cell_policy,
                "physical_conductance_formula": protocol.partial.physical_conductance_formula,
                "objective": base.recovery.objective,
                "optimizer": base.recovery.optimizer,
                "controller": protocol.partial.controller,
                "epochs": base.recovery.epochs,
                "learning_rate_raw_x": base.recovery.learning_rate_raw_x,
                "pulse_cap": base.recovery.pulse_cap,
                "test_policy": protocol.partial.test_policy,
            },
            "arms": arm_reports,
            "comparison": {
                "by_arm": comparisons,
                "validation_gate": qualification,
                "selected_smallest_qualifying_union": selected_union,
                "test_opened_arms": test_arm_ids,
                "random500_is_validation_only": True,
                "selection_is_validation_only_after_all_five_epoch_choices_frozen": True,
                "stage1_full_is_hashed_reference_not_rerun": True,
                "verify_API_and_software_full_port_materialization_reported_separately": True,
            },
            "matched_gates": {
                "same_exact_p0": True,
                "same_three_epoch_minibatch_order_hashes": True,
                "stage1_ranking_hashes_exact": True,
                "stage1_W2_mask_hash_exact": True,
                "stage1_W2_epoch_trajectory_hash_exact": True,
                "stage1_W2_terminal_test_hash_exact": True,
                "all_masks_frozen_before_any_arm_validation_or_test": True,
                "all_five_validation_epoch_selections_frozen_before_test": True,
                "all_four_cells_per_selected_logical_weight": True,
                "all_W2_logical_weights_trainable_in_every_arm": True,
                "frozen_cells_exact_in_plant_rng_and_optimizer_state": True,
                "fault_mask_not_used_by_selector_or_controller": True,
                "random_selector_uses_only_seed_and_W1_flat_address": True,
                "persistent_full_G_nonnegative": True,
                "published_corrupt_cells_immobile": True,
                "random500_test_never_opened": True,
                "stage1_full_not_rerun": True,
            },
            "limitations": [
                "exploratory_noncanonical_single_previously_inspected_target_endpoint",
                "model_based_aihwkit_1.1.0_IBM_OM_fitted_preset_not_raw_measured_traces",
                "static_P0_gradient_ranking_may_not_capture_later_update_turnover",
                "one_deterministic_random500_control_seed_only",
                "nested_addresses_are_not_realized_pulse_cost_matched",
                "digital_full_BPTT_and_Adam_are_not_sparse_or_autonomous_on_chip",
                "vectorized_verify_call_cost_need_not_scale_with_trainable_cell_count",
                "target_accumulation_and_[0,1]_projection_remain_controller_cointerventions",
                "three_epochs_are_a_fixed_horizon_not_a_convergence_claim",
                "only_W2_and_the_smallest_qualifying_ranked_union_open_current_test",
            ],
        }
        summary_path = store.run_dir / "artifacts/scientific_summary.json"
        atomic_write_json(summary_path, summary)
        generated_artifacts.append({"path": str(summary_path), "kind": "scientific_summary"})
        terminal = {
            "evidence_tier": EVIDENCE_TIER,
            "stage1_full_test_accuracy_reference": full_test_accuracy,
            "selected_smallest_qualifying_union": selected_union,
            "test_opened_arms": test_arm_ids,
            "test_accuracy_by_opened_arm": {
                arm_id: float(test["student_accuracy"])
                for arm_id, test in terminal_tests.items()
            },
        }
        store.append_metric({"mode": "train_terminal", **terminal})
        store.complete(
            metrics=terminal,
            artifacts=_artifact_records(store, generated_artifacts),
        )
        print(f"run_dir={store.run_dir}", flush=True)
        gc.collect()
        torch.cuda.empty_cache()
        return 0
    except BaseException as error:
        store.fail(error)
        raise


__all__ = [
    "_assert_w2_historical_epoch_parity",
    "_assert_w2_historical_test_parity",
    "_epoch_trajectory_sha256",
    "_load_stage1_reference",
    "_select_smallest_qualifying_union",
    "_strict_protocol",
    "run_train",
]
