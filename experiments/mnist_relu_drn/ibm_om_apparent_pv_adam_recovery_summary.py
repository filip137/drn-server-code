"""Strictly aggregate the eight completed apparent-state Adam recovery arms."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import fmean
import sys
from typing import Any, Mapping, Sequence

import torch

from experiments.artifacts import RunStore, atomic_write_json, sha256_file
from experiments.mnist_relu_drn.ibm_om_apparent_pv_adam_recovery import (
    EXPERIMENT_ID,
    FORWARD_STATE,
    MAIN_TRAINING_EXAMPLES,
    PERSISTENT_ROLE,
    SCHEMA_VERSION,
    UPDATE_RULE,
    main_arms,
)
from experiments.mnist_relu_drn.ibm_om_apparent_pv_adam_recovery_experiment import (
    _load_learning_rate_receipt,
)


_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_OUTPUT = _ROOT / "simulation_results" / "ibm_om_apparent_pv_adam_recovery"


def _json_object(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Expected readable JSON at {path}.") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected a JSON object at {path}.")
    return value


def _local_checkpoint(summary_path: Path, reported_path: str) -> Path:
    reported = Path(reported_path)
    if reported.is_file():
        return reported.resolve()
    imported = summary_path.parent / "checkpoints" / reported.name
    if not imported.is_file():
        raise FileNotFoundError(
            f"Neither reported nor imported checkpoint exists for {summary_path}."
        )
    return imported.resolve()


def _metric_pair(report: Mapping[str, Any], phase: str) -> Mapping[str, Any]:
    value = report.get(phase)
    if not isinstance(value, Mapping):
        raise ValueError(f"Arm report lacks {phase} metrics.")
    apparent = value.get("apparent")
    persistent = value.get("persistent_secondary")
    if (
        value.get("primary_state") != FORWARD_STATE
        or value.get("persistent_role") != PERSISTENT_ROLE
        or not isinstance(apparent, Mapping)
        or not isinstance(persistent, Mapping)
    ):
        raise ValueError(f"Arm {phase} violates the apparent-primary contract.")
    return {
        "apparent_accuracy": float(apparent["student_accuracy"]),
        "apparent_kl": float(apparent["kl_teacher_student"]),
        "apparent_prediction_sha256": apparent["prediction_sha256"],
        "persistent_accuracy": float(persistent["student_accuracy"]),
        "persistent_kl": float(persistent["kl_teacher_student"]),
        "persistent_prediction_sha256": persistent["prediction_sha256"],
        "apparent_minus_persistent_accuracy": (
            float(apparent["student_accuracy"])
            - float(persistent["student_accuracy"])
        ),
    }


def _validate_arm(
    path: Path,
    *,
    receipt_sha256: str,
    learning_rate: float,
    source_contract: Mapping[str, Any],
) -> tuple[str, Mapping[str, Any]]:
    summary = _json_object(path)
    report = summary.get("result")
    arm = summary.get("arm")
    receipt = summary.get("learning_rate_receipt")
    if (
        summary.get("schema") != EXPERIMENT_ID
        or summary.get("schema_version") != SCHEMA_VERSION
        or summary.get("status") != "complete"
        or summary.get("mode") != "main_arm"
        or summary.get("execution_kind") != "main_full_epoch"
        or summary.get("claim_scope") != "model-based_hybrid_not_measured_hardware"
        or summary.get("primary_state") != FORWARD_STATE
        or summary.get("persistent_state") != PERSISTENT_ROLE
        or summary.get("updates") != UPDATE_RULE
        or not isinstance(report, Mapping)
        or not isinstance(arm, Mapping)
        or not isinstance(receipt, Mapping)
        or receipt.get("sha256") != receipt_sha256
        or float(summary.get("learning_rate_progress", -1.0)) != learning_rate
    ):
        raise ValueError(f"Main-arm summary contract failed at {path}.")
    slug = arm.get("slug")
    if not isinstance(slug, str) or report.get("arm") != arm:
        raise ValueError(f"Main-arm identity mismatch at {path}.")
    training = report.get("training")
    pulses = report.get("pulses")
    state = report.get("state_diagnostics")
    source_p0 = report.get("source_p0")
    optimizer = report.get("optimizer")
    if (
        not isinstance(training, Mapping)
        or training.get("forward_state") != FORWARD_STATE
        or training.get("persistent_forward_used_for_gradients") is not False
        or training.get("examples") != MAIN_TRAINING_EXAMPLES
        or training.get("batches") != math.ceil(MAIN_TRAINING_EXAMPLES / 16)
        or not isinstance(pulses, Mapping)
        or pulses.get("verify_reads") != 0
        or not isinstance(state, Mapping)
        or state.get("corrupt_persistent_immobility_verified") is not True
        or not isinstance(source_p0, Mapping)
        or source_p0.get("sha256")
        != source_contract.get("P0_sha256", {}).get(slug)
        or not isinstance(optimizer, Mapping)
        or optimizer.get("authoritative_weight_shadow") is not None
    ):
        raise ValueError(f"Main-arm execution invariant failed for {slug}.")

    checkpoint = _local_checkpoint(path, str(report["checkpoint"]))
    if sha256_file(checkpoint) != report.get("checkpoint_sha256"):
        raise ValueError(f"Checkpoint hash mismatch for {slug}.")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (
        payload.get("schema")
        != "ebl.mnist_relu_drn.ibm_om_apparent_pv_adam_recovery_state"
        or payload.get("arm", {}).get("slug") != slug
        or payload.get("source_p0", {}).get("sha256") != source_p0.get("sha256")
        or payload.get("forward_contract", {}).get("primary_state") != FORWARD_STATE
        or payload.get("forward_contract", {}).get("persistent_state")
        != PERSISTENT_ROLE
        or payload.get("forward_contract", {}).get("updates") != UPDATE_RULE
        or payload.get("forward_contract", {}).get("verify_reads_during_recovery")
        != 0
        or payload.get("forward_contract", {}).get("authoritative_weight_shadow")
        is not None
        or int(payload.get("optimizer_state", {}).get("step", -1))
        != math.ceil(MAIN_TRAINING_EXAMPLES / 16)
    ):
        raise ValueError(f"Checkpoint semantic round-trip failed for {slug}.")
    manifest = _json_object(path.parent / "manifest.json")
    before_validation = _metric_pair(report, "before_validation")
    after_validation = _metric_pair(report, "after_validation")
    before_test = _metric_pair(report, "before_test")
    after_test = _metric_pair(report, "after_test")
    return slug, {
        "arm": dict(arm),
        "host": manifest.get("runtime", {}).get("hostname"),
        "source_p0": dict(source_p0),
        "training": dict(training),
        "pulses": dict(pulses),
        "state_diagnostics": dict(state),
        "optimizer": dict(optimizer),
        "before_validation": before_validation,
        "after_validation": after_validation,
        "before_test": before_test,
        "after_test": after_test,
        "apparent_test_accuracy_change": (
            after_test["apparent_accuracy"] - before_test["apparent_accuracy"]
        ),
        "persistent_test_accuracy_change": (
            after_test["persistent_accuracy"]
            - before_test["persistent_accuracy"]
        ),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "source_summary": str(path.resolve()),
        "source_summary_sha256": sha256_file(path),
    }


def run(args: argparse.Namespace) -> Path:
    receipt_path = args.learning_rate_receipt.expanduser().resolve()
    learning_rate, receipt = _load_learning_rate_receipt(receipt_path)
    receipt_sha256 = sha256_file(receipt_path)
    paths = tuple(path.expanduser().resolve() for path in args.arm_summary)
    if len(paths) != 8 or len(set(paths)) != 8:
        raise ValueError("Expected eight distinct explicit arm summaries.")
    expected = {arm.slug for arm in main_arms()}
    records = {}
    for path in paths:
        slug, record = _validate_arm(
            path,
            receipt_sha256=receipt_sha256,
            learning_rate=learning_rate,
            source_contract=receipt["source_contract"],
        )
        if slug in records:
            raise ValueError(f"Duplicate main arm: {slug}.")
        records[slug] = record
    if set(records) != expected:
        raise ValueError(
            f"Arm matrix mismatch: missing={sorted(expected-set(records))}, "
            f"extra={sorted(set(records)-expected)}."
        )
    ordered = {arm.slug: records[arm.slug] for arm in main_arms()}
    provenance_paths = tuple(
        path.expanduser().resolve() for path in args.execution_provenance
    )
    if len(provenance_paths) != 5 or len(set(provenance_paths)) != 5:
        raise ValueError("Expected five distinct explicit execution-provenance files.")
    for path in provenance_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Execution provenance is missing: {path}.")
    if (provenance_paths[-1].read_text(encoding="utf-8").strip()) != "0":
        raise ValueError("Remote launcher exit-code provenance is not zero.")
    execution_provenance = [
        {
            "path": str(path),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in provenance_paths
    ]
    changes = [
        float(record["apparent_test_accuracy_change"])
        for record in ordered.values()
    ]
    margins = [
        float(record["after_test"]["apparent_minus_persistent_accuracy"])
        for record in ordered.values()
    ]
    overview = {
        "arms": 8,
        "arms_with_improved_apparent_test_accuracy": sum(
            value > 0.0 for value in changes
        ),
        "minimum_apparent_test_accuracy_change": min(changes),
        "mean_apparent_test_accuracy_change": fmean(changes),
        "maximum_apparent_test_accuracy_change": max(changes),
        "arms_with_apparent_above_persistent_after_recovery": sum(
            value > 0.0 for value in margins
        ),
        "minimum_apparent_minus_persistent_after_test": min(margins),
        "mean_apparent_minus_persistent_after_test": fmean(margins),
        "maximum_apparent_minus_persistent_after_test": max(margins),
        "total_applied_recovery_pulses": sum(
            int(record["pulses"]["applied"]) for record in ordered.values()
        ),
        "total_probability_clipped_cell_events": sum(
            int(record["pulses"]["probability_clipped_cells_across_steps"])
            for record in ordered.values()
        ),
        "total_cells_at_pulse_cap": sum(
            int(record["state_diagnostics"]["cells_at_pulse_cap"])
            for record in ordered.values()
        ),
        "total_verify_reads_during_recovery": sum(
            int(record["pulses"]["verify_reads"])
            for record in ordered.values()
        ),
    }
    config = {
        "schema": EXPERIMENT_ID,
        "schema_version": SCHEMA_VERSION,
        "mode": "eight_arm_aggregate",
        "learning_rate_progress": learning_rate,
        "learning_rate_receipt_sha256": receipt_sha256,
        "explicit_arm_summaries": [str(path) for path in paths],
        "execution_provenance": execution_provenance,
        "primary_state": FORWARD_STATE,
        "persistent_state": PERSISTENT_ROLE,
    }
    inputs = [
        {
            "role": "frozen_common_learning_rate_receipt",
            "path": str(receipt_path),
            "sha256": receipt_sha256,
        }
    ]
    for slug, record in ordered.items():
        inputs.extend(
            (
                {
                    "role": f"arm_summary_{slug}",
                    "path": record["source_summary"],
                    "sha256": record["source_summary_sha256"],
                },
                {
                    "role": f"arm_checkpoint_{slug}",
                    "path": record["checkpoint"],
                    "sha256": record["checkpoint_sha256"],
                },
            )
        )
    for index, record in enumerate(execution_provenance):
        inputs.append(
            {
                "role": f"execution_provenance_{index}_{Path(record['path']).name}",
                "path": record["path"],
                "sha256": record["sha256"],
            }
        )
    store = RunStore.create(
        output_root=args.output_dir / "aggregate",
        experiment_id=EXPERIMENT_ID,
        resolved_config=config,
        command=sys.argv,
        repo_root=_ROOT,
        input_artifacts=inputs,
        resume_capability="unsupported",
        run_id=args.run_id,
    )
    try:
        summary = {
            "schema": EXPERIMENT_ID,
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "mode": "eight_arm_aggregate",
            "evidence_tier": "exploratory_noncanonical",
            "claim_scope": "model-based_hybrid_not_measured_hardware",
            "learning_rate_progress": learning_rate,
            "learning_rate_receipt": {
                "path": str(receipt_path),
                "sha256": receipt_sha256,
                "screen_run_id": receipt["screen_run_id"],
            },
            "primary_state": FORWARD_STATE,
            "persistent_state": PERSISTENT_ROLE,
            "updates": UPDATE_RULE,
            "execution_provenance": execution_provenance,
            "overview": overview,
            "arms": ordered,
        }
        summary_path = store.run_dir / "scientific_summary.json"
        atomic_write_json(summary_path, summary)
        result = store.complete(
            metrics=summary,
            artifacts=(
                store.artifact_record(summary_path, kind="scientific_summary"),
            ),
        )
        print(f"aggregate complete result={result}", flush=True)
        return result
    except BaseException as error:
        store.fail(error)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learning-rate-receipt", type=Path, required=True)
    parser.add_argument(
        "--arm-summary",
        type=Path,
        action="append",
        required=True,
        help="Repeat exactly eight times; paths are never auto-discovered.",
    )
    parser.add_argument(
        "--execution-provenance",
        type=Path,
        action="append",
        required=True,
        help=(
            "Repeat exactly five times in this order: remote code bundle, input "
            "bundle, launcher log, launcher command, launcher exit code."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--run-id", type=str)
    return parser


def main() -> int:
    run(_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
