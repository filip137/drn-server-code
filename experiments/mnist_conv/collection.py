"""Single-writer, fail-closed collection for canonical Conv sweeps."""

from __future__ import annotations

import json
import math
import os
import fcntl
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .identity import sha256_file
from .io import atomic_write_csv, atomic_write_json, read_json, relative_posix
from .layout import ResultLayout
from .manifest import load_manifest
from .protocol import final_protocol_approval_reason
from .runner import InvalidBundleError, validate_bundle
from .specs import RunSpec, pointer_get


COLLECTION_SCHEMA_VERSION = "mnist-conv-collection/v1"

APPROVED_AMPLIFICATIONS = frozenset({(1.0, 1.0), (4.0, 1.0), (4.0, 0.25)})
FROZEN_NORMALIZATION = {"mean": 0.1307, "std": 0.3081, "scale": 0.3}
FROZEN_MEDIUM_AFFINE = {
    "enabled": True,
    "preset": "medium",
    "degrees": 25.0,
    "translate": [0.2, 0.2],
    "scale": [0.8, 1.2],
    "shear": 0.0,
    "seed": 1729,
    "interpolation": "bilinear",
    "fill": 0.0,
}

SUMMARY_COLUMNS = [
    "job_index", "logical_key", "case_id", "run_id", "status",
    "protocol_id", "category", "paper_eligible", "eligibility_reasons",
    "git_revision", "dirty_source_digest", "code_fingerprint",
    "seed", "replicate_id", "architecture", "non_linearity",
    "voltage_amp", "current_amp", "input_gain", "batch_size",
    "optimizer", "momentum", "weight_decay",
    "preprocessing", "affine", "v_off",
    "calibration_id",
    "target_initial_saturation", "measured_initial_saturation",
    "target_initial_occupancy", "measured_initial_occupancy",
    "inference_iterations", "training_iterations", "epochs", "learning_rate",
    "checkpoint_rule", "batch_state_policy", "best_epoch", "best_test_accuracy",
    "final_test_accuracy", "final_train_loss", "final_test_loss",
    "run_relpath", "best_checkpoint_path", "final_checkpoint_path",
    "weights_best_path", "weights_final_path",
    "best_checkpoint_sha256", "final_checkpoint_sha256",
]

CASE_SUMMARY_COLUMNS = [
    "case_id", "group_key", "group_values", "required", "complete",
    "expected_seeds", "complete_seeds", "missing_seeds",
    "run_count", "complete_count", "best_test_accuracy_mean",
    "final_test_accuracy_mean", "paper_eligible", "eligibility_reasons",
]


class IncompleteSweepError(RuntimeError):
    pass


class CollectionBusyError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _acquire_lock(path: Path):
    """Acquire a crash-safe advisory collector lock.

    The file is descriptive, while the kernel lock is authoritative and is
    released automatically if a collector process exits unexpectedly.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise CollectionBusyError(
            f"Expected at most one sweep collector. Provided live lock: {path}."
        ) from exc
    handle.seek(0)
    handle.truncate()
    json.dump({"pid": os.getpid(), "host": os.uname().nodename, "created_at": _utc_now()}, handle)
    handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    return handle


def _release_lock(path: Path, handle) -> None:
    try:
        try:
            if path.exists() and path.stat().st_ino == os.fstat(handle.fileno()).st_ino:
                path.unlink()
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


def _latest_attempt_state(layout: ResultLayout, run_id: str) -> str:
    root = layout.attempt_root(run_id)
    if not root.exists():
        return "missing"
    statuses: list[tuple[str, str]] = []
    for path in root.glob("*/status.json"):
        try:
            value = read_json(path)
            statuses.append((path.parent.name, str(value.get("state", "invalid"))))
        except (OSError, ValueError):
            continue
    return sorted(statuses)[-1][1] if statuses else "missing"


def _frozen_science_reasons(spec: RunSpec) -> list[str]:
    value, run = spec.data, spec.data["run"]
    model, dataset = run["model"], run["dataset"]
    architecture, calibration = run["architecture"], run["calibration"]
    reasons: list[str] = []
    protocol_reason = final_protocol_approval_reason(spec)
    if protocol_reason is not None:
        reasons.append(protocol_reason)
    if value["category"] != "final":
        reasons.append("category_not_final")
    if dataset["affine"] != FROZEN_MEDIUM_AFFINE:
        reasons.append("affine_not_frozen_medium")
    if dataset["normalization"] != FROZEN_NORMALIZATION:
        reasons.append("normalization_not_frozen")
    if (float(model["voltage_amp"]), float(model["current_amp"])) not in APPROVED_AMPLIFICATIONS:
        reasons.append("amplification_not_approved")
    if architecture["output_dim"] != 20 or architecture["pooling"] != {"mode": "none"}:
        reasons.append("architecture_not_output20_no_pool")
    if run["training"]["pruning"]["enabled"]:
        reasons.append("pruning_enabled")

    common_frozen = (
        calibration["scope"] == "first_hidden_layer"
        and calibration["sample_count"] == 256
        and calibration["batch_size"] == 64
        and calibration["model_seed"] == 0
        and calibration["affine_seed"] == 1729
        and calibration["settling_iterations"] == 64
        and calibration["adaptive_equilibrium"] is False
    )
    if not common_frozen:
        reasons.append("calibration_provenance_not_frozen")
    if model["non_linearity"] == "hard_sigmoid":
        hard = model["hard_sigmoid_param"]
        if not (
            calibration["kind"] == "hard_sigmoid_saturation"
            and math.isclose(calibration["target_initial_saturation"], 0.30)
            and math.isclose(calibration["v_off"], 4.0)
            and math.isclose(hard["g_on"], 100.0)
            and math.isclose(hard["g_off"], 0.0)
            and calibration["g_on"] == hard["g_on"]
            and calibration["g_off"] == hard["g_off"]
        ):
            reasons.append("hard_sigmoid_calibration_not_frozen")
    elif not (
        calibration["kind"] == "perfect_diode_clamped_occupancy"
        and math.isclose(calibration["target_initial_occupancy"], 0.30)
        and math.isclose(calibration["clamp_epsilon"], 1e-8)
    ):
        reasons.append("perfect_diode_calibration_not_frozen")
    return reasons


def _collection_group(spec: RunSpec, pointers: list[str]) -> tuple[str, dict[str, Any]]:
    """Return a deterministic grouping key and its human-readable values."""

    document = spec.to_dict()
    values = {pointer: pointer_get(document, pointer) for pointer in pointers}
    key = json.dumps(values, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return key, values


def _comparison_table_key(spec: RunSpec) -> tuple[str, str]:
    """Axes across which an amplification grid is never allowed to mix."""

    run = spec.data["run"]
    return run["architecture"]["profile"], run["model"]["non_linearity"]


def _canonical_value(value: Any) -> str:
    """Return a stable equality key for normalized scientific JSON."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _calibration_protocol(calibration: dict[str, Any]) -> dict[str, Any]:
    """Return the fixed calibration method, excluding row measurements.

    A calibrated gain, its calibration id, and the measurements made for that
    gain belong to one amplification case.  The method that produced them
    (cohort, target, settling rule, conductances/clamp, and scope) must remain
    identical across the comparison table.
    """

    return {
        key: value
        for key, value in calibration.items()
        if key != "calibration_id"
        and key != "layer_measurements"
        and not key.startswith("measured_")
    }


def _comparison_contract_fields(spec: RunSpec) -> dict[str, Any]:
    """Return paper-table fields that must match across amplification cases.

    The explicit numeric learning-rate vector is deliberately absent: it may
    be selected separately for each amplification case.  The optimizer and LR
    decay policy remain fixed.  Amplification, input gain, calibration id, and
    measured calibration values are likewise checked for consistency *within*
    a case by :func:`_case_linked_fields` rather than across cases.
    """

    value, run = spec.data, spec.data["run"]
    dataset, model = run["dataset"], run["model"]
    solver, training = run["solver"], run["training"]
    checkpoint = run["initialization"]["checkpoint"]
    if checkpoint is not None:
        checkpoint = {key: item for key, item in checkpoint.items() if key != "path"}
    return {
        "comparison_protocol_mismatch": {
            "protocol_id": value["protocol_id"],
            "category": value["category"],
        },
        "comparison_architecture_mismatch": run["architecture"],
        "comparison_preprocessing_mismatch": {
            key: item for key, item in dataset.items() if key != "batch_size"
        },
        "comparison_nonlinearity_mismatch": {
            "type": model["type"],
            "non_linearity": model["non_linearity"],
            "quadratic_diode_param": model["quadratic_diode_param"],
            "exponential_diode_param": model["exponential_diode_param"],
            "hard_sigmoid_param": model["hard_sigmoid_param"],
        },
        "comparison_model_initialization_mismatch": {
            "weight_gains": model["weight_gains"],
            "weight_min": model["weight_min"],
            "weight_max": model["weight_max"],
            "weight_init_mode": model["weight_init_mode"],
            "trainable_parameters": model["trainable_parameters"],
            "amplification_min": model["amplification_min"],
            "amplification_max": model["amplification_max"],
            "checkpoint": checkpoint,
        },
        "comparison_batch_size_mismatch": dataset["batch_size"],
        "comparison_epoch_budget_mismatch": training["epochs"],
        "comparison_checkpoint_rule_mismatch": training["checkpoint_rule"],
        "comparison_solver_tk_mismatch": {
            "inference_iterations": solver["inference_iterations"],
            "training_iterations": solver["training_iterations"],
        },
        "comparison_minimizer_mismatch": {
            "energy_mode": solver["energy_mode"],
            "minimizer": solver["minimizer"],
        },
        "comparison_lr_policy_mismatch": {
            "optimizer": training["optimizer"],
            "lr_decay": training["lr_decay"],
        },
        "comparison_training_policy_mismatch": {
            "algorithm": training["algorithm"],
            "beta": training["beta"],
            "pruning": training["pruning"],
            "batch_state_policy": training["batch_state_policy"],
        },
        "comparison_calibration_protocol_mismatch": _calibration_protocol(
            run["calibration"]
        ),
    }


def _case_linked_fields(spec: RunSpec) -> dict[str, Any]:
    """Return fields allowed to differ between, but never within, cases."""

    run = spec.data["run"]
    model, calibration = run["model"], run["calibration"]
    measurements = {
        key: value
        for key, value in calibration.items()
        if key == "calibration_id"
        or key == "layer_measurements"
        or key.startswith("measured_")
    }
    return {
        "case_amplification_mismatch": {
            "voltage_amp": model["voltage_amp"],
            "current_amp": model["current_amp"],
        },
        "case_input_gain_mismatch": model["input_gain"],
        "case_learning_rate_mismatch": run["training"]["learning_rate"],
        "case_calibration_binding_mismatch": measurements,
    }


def _paper_comparison_reasons(
    manifest: dict[str, Any],
    specs_by_run: dict[str, RunSpec],
) -> tuple[dict[str, set[str]], bool, bool]:
    """Validate fixed table axes and matched stochastic comparison structure."""

    entries_by_table: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for entry in manifest["entries"]:
        entries_by_table[_comparison_table_key(specs_by_run[entry["run_id"]])].append(entry)

    reasons: dict[str, set[str]] = defaultdict(set)
    fixed_contract = True
    fixed_seed_structure = True
    expected_seeds = set(manifest["collection"]["expected_seeds"])

    # Compare the set of fixed settings represented by each linked case before
    # splitting a sweep into architecture/nonlinearity tables.  This catches a
    # malformed sweep that assigns, for example, Conv1 to one amplification
    # case and Conv2 to another.  A genuine independent axis is still allowed:
    # every case then contains the same set of values.
    all_entries_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in manifest["entries"]:
        all_entries_by_case[entry["case_id"]].append(entry)
    all_run_ids = [entry["run_id"] for entry in manifest["entries"]]
    first_spec = specs_by_run[all_run_ids[0]]
    for reason in _comparison_contract_fields(first_spec):
        value_sets = {
            _canonical_value(sorted({
                _canonical_value(_comparison_contract_fields(specs_by_run[entry["run_id"]])[reason])
                for entry in entries
            }))
            for entries in all_entries_by_case.values()
        }
        if len(value_sets) > 1:
            fixed_contract = False
            for run_id in all_run_ids:
                reasons[run_id].add(reason)

    for entries in entries_by_table.values():
        run_ids = [entry["run_id"] for entry in entries]
        specs = [specs_by_run[run_id] for run_id in run_ids]

        for reason in _comparison_contract_fields(specs[0]):
            values = {
                _canonical_value(_comparison_contract_fields(spec)[reason])
                for spec in specs
            }
            if len(values) > 1:
                fixed_contract = False
                for run_id in run_ids:
                    reasons[run_id].add(reason)

        entries_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for entry in entries:
            entries_by_case[entry["case_id"]].append(entry)

        case_amplifications: dict[str, set[tuple[float, float]]] = {}
        for case_id, case_entries in entries_by_case.items():
            case_specs = [specs_by_run[entry["run_id"]] for entry in case_entries]
            for reason in _case_linked_fields(case_specs[0]):
                values = {
                    _canonical_value(_case_linked_fields(spec)[reason])
                    for spec in case_specs
                }
                if len(values) > 1:
                    fixed_contract = False
                    for run_id in run_ids:
                        reasons[run_id].add(reason)
            case_amplifications[case_id] = {
                (
                    float(spec.data["run"]["model"]["voltage_amp"]),
                    float(spec.data["run"]["model"]["current_amp"]),
                )
                for spec in case_specs
            }

        cases_by_amplification: dict[tuple[float, float], set[str]] = defaultdict(set)
        for case_id, amplifications in case_amplifications.items():
            for amplification in amplifications:
                cases_by_amplification[amplification].add(case_id)
        if (
            any(len(amplifications) != 1 for amplifications in case_amplifications.values())
            or any(len(case_ids) != 1 for case_ids in cases_by_amplification.values())
        ):
            fixed_contract = False
            for run_id in run_ids:
                reasons[run_id].add("amplification_case_mapping_mismatch")

        # A comparison slot is every non-seed independent axis plus an explicit
        # replicate id.  Each seed must have the same complete slot set, and
        # every amplification case must use that same stochastic structure.
        slots_by_case_seed: dict[str, dict[int, list[str]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for entry in entries:
            spec = specs_by_run[entry["run_id"]]
            slot = {
                "replicate_id": spec.data["replicate_id"],
                "axes": {
                    pointer: value
                    for pointer, value in entry["axes"].items()
                    if pointer != "/seed"
                },
            }
            slots_by_case_seed[entry["case_id"]][spec.data["seed"]].append(
                _canonical_value(slot)
            )

        case_structures: list[str] = []
        seed_structure_mismatch = False
        for by_seed in slots_by_case_seed.values():
            if set(by_seed) != expected_seeds:
                seed_structure_mismatch = True
                continue
            ordered_slots = {
                seed: tuple(sorted(by_seed[seed])) for seed in expected_seeds
            }
            if len(set(ordered_slots.values())) != 1:
                seed_structure_mismatch = True
            case_structures.append(_canonical_value(ordered_slots[min(expected_seeds)]))
        if len(set(case_structures)) > 1:
            seed_structure_mismatch = True
        if seed_structure_mismatch:
            fixed_seed_structure = False
            for run_id in run_ids:
                reasons[run_id].add("comparison_seed_structure_mismatch")

    return reasons, fixed_contract, fixed_seed_structure


def _summary_row(
    entry: dict[str, Any],
    spec: RunSpec,
    status: str,
    metrics: dict[str, Any] | None,
    run_relpath: str,
    reasons: list[str],
    source_provenance: dict[str, Any],
    code_fingerprint: str,
    run_dir: Path | None,
) -> dict[str, Any]:
    value, run = spec.data, spec.data["run"]
    model, solver, training = run["model"], run["solver"], run["training"]
    metrics = metrics or {}
    calibration, dataset = run["calibration"], run["dataset"]
    hard_sigmoid = model["non_linearity"] == "hard_sigmoid"
    return {
        "job_index": entry["job_index"], "logical_key": entry["logical_key"],
        "case_id": entry["case_id"], "run_id": entry["run_id"], "status": status,
        "protocol_id": value["protocol_id"], "category": value["category"],
        "paper_eligible": False, "eligibility_reasons": json.dumps(reasons, separators=(",", ":")),
        "git_revision": source_provenance["git_revision"] or "",
        "dirty_source_digest": source_provenance["dirty_source_digest"] or "",
        "code_fingerprint": code_fingerprint,
        "seed": value["seed"], "replicate_id": value["replicate_id"],
        "architecture": run["architecture"]["profile"], "non_linearity": model["non_linearity"],
        "voltage_amp": model["voltage_amp"], "current_amp": model["current_amp"],
        "input_gain": model["input_gain"], "batch_size": run["dataset"]["batch_size"],
        "optimizer": training["optimizer"]["name"],
        "momentum": training["optimizer"]["momentum"],
        "weight_decay": training["optimizer"]["weight_decay"],
        "preprocessing": json.dumps(dataset["normalization"], sort_keys=True, separators=(",", ":")),
        "affine": json.dumps(dataset["affine"], sort_keys=True, separators=(",", ":")),
        "v_off": calibration["v_off"] if hard_sigmoid else "",
        "calibration_id": calibration["calibration_id"],
        "target_initial_saturation": calibration["target_initial_saturation"] if hard_sigmoid else "",
        "measured_initial_saturation": calibration["measured_initial_saturation"] if hard_sigmoid else "",
        "target_initial_occupancy": "" if hard_sigmoid else calibration["target_initial_occupancy"],
        "measured_initial_occupancy": "" if hard_sigmoid else calibration["measured_initial_occupancy"],
        "inference_iterations": solver["inference_iterations"],
        "training_iterations": solver["training_iterations"], "epochs": training["epochs"],
        "learning_rate": json.dumps(training["learning_rate"], separators=(",", ":")),
        "checkpoint_rule": training["checkpoint_rule"],
        "batch_state_policy": training["batch_state_policy"],
        "best_epoch": metrics.get("best_epoch", ""),
        "best_test_accuracy": metrics.get("best_test_accuracy", ""),
        "final_test_accuracy": metrics.get("final_test_accuracy", ""),
        "final_train_loss": metrics.get("final_train_loss", ""),
        "final_test_loss": metrics.get("final_test_loss", ""),
        "run_relpath": run_relpath,
        "best_checkpoint_path": metrics.get("best_checkpoint_path", "checkpoints/best.pt"),
        "final_checkpoint_path": metrics.get("checkpoint_path", "checkpoints/final.pt"),
        "weights_best_path": metrics.get("weights_best_path", "weights/best.npz"),
        "weights_final_path": metrics.get("weights_final_path", "weights/final.npz"),
        "best_checkpoint_sha256": sha256_file(run_dir / "checkpoints/best.pt") if run_dir is not None and status == "complete" else "",
        "final_checkpoint_sha256": sha256_file(run_dir / "checkpoints/final.pt") if run_dir is not None and status == "complete" else "",
    }


def collect_sweep(
    manifest_path: str | Path,
    layout: ResultLayout,
    *,
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    manifest_file = Path(manifest_path).expanduser().resolve()
    manifest = load_manifest(manifest_file)
    sweep_dir = layout.find_sweep_dir(manifest["sweep_id"]) or layout.sweep_dir(
        manifest["sweep_name"], manifest["sweep_id"]
    )
    sweep_dir.mkdir(parents=True, exist_ok=True)
    canonical_manifest = sweep_dir / "manifest.json"
    if canonical_manifest.exists():
        if load_manifest(canonical_manifest) != manifest:
            raise ValueError(f"Expected canonical sweep manifest to match input. Provided value: {manifest_file}.")
    else:
        atomic_write_json(canonical_manifest, manifest, canonical=True)

    lock_path = sweep_dir / ".collect.lock"
    lock_handle = _acquire_lock(lock_path)
    try:
        rows: list[dict[str, Any]] = []
        coverage: dict[str, int] = {
            key: 0 for key in ("complete", "missing", "running", "failed", "pruned", "stale", "invalid")
        }
        input_completions: list[dict[str, Any]] = []
        seeds_by_group: dict[tuple[str, str], set[int]] = defaultdict(set)
        planned_runs_by_group_seed: dict[tuple[tuple[str, str], int], set[str]] = defaultdict(set)
        complete_runs_by_group_seed: dict[tuple[tuple[str, str], int], set[str]] = defaultdict(set)
        group_values: dict[tuple[str, str], dict[str, Any]] = {}
        group_order: list[tuple[str, str]] = []
        table_by_run: dict[str, tuple[str, str]] = {}
        amplifications_by_table: dict[tuple[str, str], set[tuple[float, float]]] = defaultdict(set)
        reasons_by_run: dict[str, list[str]] = {}
        specs_by_run: dict[str, RunSpec] = {}
        grouping_pointers = list(manifest["collection"]["group_by"])

        for entry in manifest["entries"]:
            spec = RunSpec.from_dict(entry["run_spec"])
            group_key, values = _collection_group(spec, grouping_pointers)
            grouped_key = (entry["case_id"], group_key)
            if grouped_key not in group_values:
                group_values[grouped_key] = values
                group_order.append(grouped_key)
            seed = spec.data["seed"]
            seeds_by_group[grouped_key].add(seed)
            planned_runs_by_group_seed[(grouped_key, seed)].add(entry["run_id"])
            specs_by_run[entry["run_id"]] = spec
            table_key = _comparison_table_key(spec)
            table_by_run[entry["run_id"]] = table_key
            amplifications_by_table[table_key].add(
                (
                    float(spec.data["run"]["model"]["voltage_amp"]),
                    float(spec.data["run"]["model"]["current_amp"]),
                )
            )
            run_dir = layout.find_run_dir(entry["run_id"])
            expected_dir = layout.run_dir(spec.data["label"], entry["run_id"])
            metrics: dict[str, Any] | None = None
            if run_dir is not None:
                try:
                    validate_bundle(run_dir, entry["run_id"])
                    metrics = read_json(run_dir / "metrics.json")
                    if not isinstance(metrics, dict):
                        raise InvalidBundleError(f"Expected metrics object. Provided value: {metrics!r}.")
                    status = "complete"
                    complete_runs_by_group_seed[(grouped_key, seed)].add(entry["run_id"])
                    input_completions.append({
                        "run_id": entry["run_id"],
                        "manifest_sha256": sha256_file(run_dir / "manifest.json"),
                    })
                except (InvalidBundleError, OSError, json.JSONDecodeError, ValueError):
                    status = "invalid"
            else:
                status = _latest_attempt_state(layout, entry["run_id"])
                if status not in coverage:
                    status = "invalid"
            coverage[status] += 1
            reasons = _frozen_science_reasons(spec)
            if status != "complete":
                reasons.append("run_not_complete")
            reasons_by_run[entry["run_id"]] = reasons
            rows.append(_summary_row(
                entry, spec, status, metrics,
                relative_posix(run_dir or expected_dir, sweep_dir), reasons,
                manifest["code_provenance"], manifest["code_fingerprint"], run_dir,
            ))

        expected_seeds = list(manifest["collection"]["expected_seeds"])
        required_cases = list(manifest["collection"]["required_cases"])
        expected_seed_set = set(expected_seeds)
        required_groups = [key for key in group_order if key[0] in required_cases]
        complete_seeds_by_group: dict[tuple[str, str], set[int]] = {
            key: {
                seed
                for seed in expected_seed_set
                if planned_runs_by_group_seed[(key, seed)]
                and complete_runs_by_group_seed[(key, seed)]
                == planned_runs_by_group_seed[(key, seed)]
            }
            for key in required_groups
        }
        planned_coverage_ok = (
            {key[0] for key in required_groups} == set(required_cases)
            and all(seeds_by_group[key] == expected_seed_set for key in required_groups)
        )
        complete_coverage_ok = (
            planned_coverage_ok
            and all(complete_seeds_by_group[key] == expected_seed_set for key in required_groups)
        )
        all_complete = coverage["complete"] == len(manifest["entries"]) and planned_coverage_ok and complete_coverage_ok
        table_science_complete = bool(amplifications_by_table) and all(
            coverage == APPROVED_AMPLIFICATIONS
            for coverage in amplifications_by_table.values()
        )
        comparison_reasons, comparison_contract_fixed, seed_structure_fixed = (
            _paper_comparison_reasons(manifest, specs_by_run)
        )

        for row in rows:
            reasons = reasons_by_run[row["run_id"]]
            reasons.extend(comparison_reasons[row["run_id"]])
            if not complete_coverage_ok:
                reasons.append("seed_or_case_coverage_incomplete")
            if amplifications_by_table[table_by_run[row["run_id"]]] != APPROVED_AMPLIFICATIONS:
                reasons.append("amplification_grid_incomplete")
            # The approved registry is currently empty, so this remains false.
            row["paper_eligible"] = not reasons and all_complete
            row["eligibility_reasons"] = json.dumps(sorted(set(reasons)), separators=(",", ":"))

        grouped: list[dict[str, Any]] = []
        for grouped_key in required_groups:
            case_id, group_key = grouped_key
            run_ids = {
                entry["run_id"]
                for entry in manifest["entries"]
                if entry["case_id"] == case_id
                and _collection_group(RunSpec.from_dict(entry["run_spec"]), grouping_pointers)[0] == group_key
            }
            case_rows = [row for row in rows if row["run_id"] in run_ids]
            complete_rows = [row for row in case_rows if row["status"] == "complete"]
            case_reasons = sorted({
                reason
                for row in case_rows
                for reason in json.loads(row["eligibility_reasons"])
            })
            grouped.append({
                "case_id": case_id,
                "group_key": group_key,
                "group_values": json.dumps(group_values[grouped_key], sort_keys=True, separators=(",", ":"), allow_nan=False),
                "required": True,
                "complete": len(complete_rows) == len(case_rows) and complete_seeds_by_group[grouped_key] == expected_seed_set,
                "expected_seeds": json.dumps(expected_seeds, separators=(",", ":")),
                "complete_seeds": json.dumps([seed for seed in expected_seeds if seed in complete_seeds_by_group[grouped_key]], separators=(",", ":")),
                "missing_seeds": json.dumps([seed for seed in expected_seeds if seed not in complete_seeds_by_group[grouped_key]], separators=(",", ":")),
                "run_count": len(case_rows), "complete_count": len(complete_rows),
                "best_test_accuracy_mean": (
                    sum(float(row["best_test_accuracy"]) for row in complete_rows) / len(complete_rows)
                    if complete_rows else ""
                ),
                "final_test_accuracy_mean": (
                    sum(float(row["final_test_accuracy"]) for row in complete_rows) / len(complete_rows)
                    if complete_rows else ""
                ),
                "paper_eligible": bool(complete_rows) and all(bool(row["paper_eligible"]) for row in case_rows),
                "eligibility_reasons": json.dumps(case_reasons, separators=(",", ":")),
            })

        collection = {
            "schema_version": COLLECTION_SCHEMA_VERSION,
            "sweep_id": manifest["sweep_id"],
            "manifest_relpath": relative_posix(canonical_manifest, sweep_dir),
            "generated_at": _utc_now(), "complete": all_complete,
            "paper_eligible": all_complete and bool(rows) and all(bool(row["paper_eligible"]) for row in rows),
            "coverage": coverage,
            "coverage_checks": {
                "planned_required_cases_and_seeds": planned_coverage_ok,
                "complete_required_cases_and_seeds": complete_coverage_ok,
                "approved_amplification_grid": table_science_complete,
                "fixed_comparison_contract": comparison_contract_fixed,
                "fixed_seed_comparison_structure": seed_structure_fixed,
            },
            "varying_fields": manifest["varying_fields"],
            "input_completions": input_completions, "rows": rows, "cases": grouped,
        }
        atomic_write_json(sweep_dir / "collection.json", collection)
        summary_name = "summary.csv" if all_complete else "summary.partial.csv"
        atomic_write_csv(sweep_dir / summary_name, SUMMARY_COLUMNS, rows)
        other_summary = sweep_dir / ("summary.partial.csv" if all_complete else "summary.csv")
        other_summary.unlink(missing_ok=True)
        atomic_write_csv(sweep_dir / "summary_by_case.csv", CASE_SUMMARY_COLUMNS, grouped)
        if not all_complete and not allow_incomplete:
            raise IncompleteSweepError(
                "Expected every required case/seed manifest run to be complete before canonical collection. "
                f"Provided coverage: {coverage!r}; partial summary: {sweep_dir / summary_name}."
            )
        return collection
    finally:
        _release_lock(lock_path, lock_handle)
