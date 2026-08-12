#!/usr/bin/env python3
"""Collate and plot the trained Conv3 high-beta/high-T/K EqProp diagnostic.

The inputs are completed canonical true-float32/float64 shadow bundles.  The
six core bundles compare baseline and ours at injected beta ``{.01,.1,1}``
under each scheme's native T/K and at T/K=64/64.  Optional adaptive
confirmation bundles may add a stronger beta or another T/K coordinate.

This is a read-only ordinary-MNIST diagnostic.  Biases remain active in the
dynamics but bias gradients are excluded, no optimizer step is taken, and the
official MNIST test split is not read.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments import reporting  # noqa: E402


DEFAULT_STUDY_ROOT = (
    REPOSITORY_ROOT
    / "results/perfectdiode-conv3-baseline-ours-eqprop-high-beta-high-tk-seed0-20260811-v1"
)
SCHEMA = "conv3-eqprop-high-beta-high-tk-analysis/v1"
SCHEMES = ("baseline", "ours")
PRECISIONS = ("float32", "float64")
PARAMETERS = ("ConvWeight_0", "ConvWeight_1", "ConvWeight_2", "DenseWeight_0")
LAYER_LABELS = {
    "ConvWeight_0": "C0",
    "ConvWeight_1": "C1",
    "ConvWeight_2": "C2",
    "DenseWeight_0": "D",
}
STATE_LAYER_BY_PARAMETER = {
    "ConvWeight_0": "Layer_1",
    "ConvWeight_1": "Layer_2",
    "ConvWeight_2": "Layer_3",
    "DenseWeight_0": "Layer_4",
}
CORE_RUN_SPECS: dict[str, dict[str, Any]] = {
    "native-injected-1em2": {"tk_mode": "native", "injected_beta": 1.0e-2},
    "native-injected-1em1": {"tk_mode": "native", "injected_beta": 1.0e-1},
    "native-injected-1": {"tk_mode": "native", "injected_beta": 1.0},
    "tk64-injected-1em2": {"tk_mode": "tk64", "injected_beta": 1.0e-2},
    "tk64-injected-1em1": {"tk_mode": "tk64", "injected_beta": 1.0e-1},
    "tk64-injected-1": {"tk_mode": "tk64", "injected_beta": 1.0},
}
COMPLETION_GUARDS = (
    "criteria_met",
    "all_iteration_contracts_passed",
    "all_runtime_dtypes_proven",
    "all_finite_differences_computed_in_native_dtype",
    "all_bptt_gradients_computed_in_native_dtype",
    "bptt_zero_and_positive_started_from_identical_post_T",
    "fixed_cohort_and_batch_reproduced",
    "source_bundle_files_unchanged",
    "source_checkpoint_bytes_unchanged",
    "parameter_tensors_unchanged",
    "bias_gradients_excluded",
)
PARAMETER_REQUIRED_COLUMNS = {
    "architecture",
    "scheme",
    "checkpoint_role",
    "T",
    "K",
    "source_native_T",
    "source_native_K",
    "native_context",
    "tk_source",
    "batch_index",
    "batch_payload_sha256",
    "parameter_name",
    "parameter_type",
    "bias_excluded",
    "actual_beta",
    "effective_beta",
    "beta_hat_requested",
    "beta_capped",
    "source_grid_cap_bypassed",
    "beta_source",
    "residual_limited",
    *(f"{precision}_eqprop_gradient_dtype" for precision in PRECISIONS),
    *(f"{precision}_bptt_gradient_dtype" for precision in PRECISIONS),
    *(f"{precision}_eqprop_l2" for precision in PRECISIONS),
    *(f"{precision}_bptt_l2" for precision in PRECISIONS),
    *(f"{precision}_exact_zero_fraction" for precision in PRECISIONS),
    *(f"{precision}_eqprop_vs_bptt_cosine" for precision in PRECISIONS),
}
SCALING_REQUIRED_COLUMNS = {
    "architecture",
    "scheme",
    "checkpoint_role",
    "T",
    "K",
    "source_native_T",
    "source_native_K",
    "native_context",
    "tk_source",
    "actual_beta",
    "effective_beta",
    "beta_hat_requested",
    "beta_capped",
    "source_grid_cap_bypassed",
    "beta_source",
}
RESIDUAL_REQUIRED_COLUMNS = {
    "architecture",
    "scheme",
    "checkpoint_role",
    "T",
    "K",
    "source_native_T",
    "source_native_K",
    "native_context",
    "tk_source",
    "precision",
    "batch_index",
    "batch_payload_sha256",
    "phase",
    "state_layer_name",
    "coverage_complete",
    "outcome",
    "selected_max_p90",
    "selected_maximum",
    "gate_passed",
    "hard_failure",
}
DISPLACEMENT_REQUIRED_COLUMNS = {
    "architecture",
    "scheme",
    "checkpoint_role",
    "T",
    "K",
    "source_native_T",
    "source_native_K",
    "native_context",
    "tk_source",
    "precision",
    "batch_index",
    "batch_payload_sha256",
    "phase",
    "actual_beta",
    "reference_kind",
    "outcome",
    "state_layer_name",
    "relative_displacement",
    "active_set_transition_fraction",
}
OUTPUT_FIELDS = (
    "schema",
    "run_id",
    "run_kind",
    "input_result_sha256",
    "architecture",
    "scheme",
    "precision",
    "checkpoint_role",
    "T",
    "K",
    "source_native_T",
    "source_native_K",
    "native_context",
    "tk_source",
    "batch_index",
    "batch_payload_sha256",
    "actual_base_beta",
    "injected_beta",
    "beta_hat_requested",
    "beta_capped",
    "source_grid_cap_bypassed",
    "beta_source",
    "parameter_name",
    "layer",
    "state_layer_name",
    "bias_excluded",
    "eqprop_vs_bptt_cosine",
    "gradient_status",
    "eqprop_l2",
    "bptt_l2",
    "exact_zero_fraction",
    "eqprop_over_bptt_norm_ratio",
    "eqprop_vs_bptt_symmetric_norm_delta",
    "residual_limited",
    "all_residual_gates_passed",
    "post_t_free_worst_selected_max_p90",
    "zero_worst_selected_max_p90",
    "positive_worst_selected_max_p90",
    "worst_residual_selected_max_p90",
    "worst_residual_selected_maximum",
    "residual_hard_failure",
    "matched_zero_relative_state_displacement",
    "post_t_relative_state_displacement",
    "matched_zero_aggregate_relative_displacement",
    "post_t_aggregate_relative_displacement",
    "matched_zero_active_set_transition_fraction",
)
LIMITATIONS = (
    "Ordinary MNIST diagnostic evidence only; it is not a deterministic medium-affine paper result.",
    "One fixed validation minibatch (batch 0, 16 examples) is replayed, so no minibatch spread is estimated.",
    "The replay is read-only: no optimizer is constructed or stepped and source checkpoint bytes must remain unchanged.",
    "Biases are active in the network dynamics, but bias gradients are excluded from every scored comparison.",
    "The official MNIST test split is not read.",
    "Injected beta values above the source 0.01 cap are diagnostic extrapolations, not proposed training nudges.",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        return [dict(row) for row in reader]


def _parse_bool(value: object) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no", ""}:
        return False
    raise ValueError(f"Expected a boolean, got {value!r}.")


def _optional_float(value: object) -> float | None:
    normalized = str(value).strip().lower()
    if normalized in {"", "none", "null", "nan"}:
        return None
    result = float(normalized)
    if not math.isfinite(result):
        raise ValueError(f"Expected a finite float or blank, got {value!r}.")
    return result


def _single(values: Iterable[Any], *, label: str) -> Any:
    unique = set(values)
    if len(unique) != 1:
        raise ValueError(f"Expected one {label}, observed {sorted(unique, key=str)}.")
    return next(iter(unique))


def _close(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1.0e-10, abs_tol=1.0e-15)


def _gradient_status(row: Mapping[str, str], precision: str) -> str:
    cosine = _optional_float(row[f"{precision}_eqprop_vs_bptt_cosine"])
    if cosine is not None:
        if not -1.000001 <= cosine <= 1.000001:
            raise ValueError(f"Cosine outside [-1,1]: {cosine}.")
        return "numeric"
    eqprop_l2 = float(row[f"{precision}_eqprop_l2"])
    zero_fraction = float(row[f"{precision}_exact_zero_fraction"])
    if eqprop_l2 == 0.0 and math.isclose(zero_fraction, 1.0, abs_tol=1.0e-12):
        return "exact_zero"
    return "undefined"


def _validate_context_fields(
    rows: Sequence[Mapping[str, str]], *, path: Path, allow_aggregate: bool = False
) -> None:
    allowed_state_names = {*STATE_LAYER_BY_PARAMETER.values(), "__all__"}
    for row in rows:
        if row["architecture"] != "conv3":
            raise ValueError(f"Non-Conv3 row in {path}.")
        if row["scheme"] not in SCHEMES:
            raise ValueError(f"Unexpected scheme {row['scheme']!r} in {path}.")
        if row["checkpoint_role"] != "best_validation":
            raise ValueError(f"Non-best checkpoint row in {path}.")
        if int(row["T"]) <= 0 or int(row["K"]) <= 0:
            raise ValueError(f"Nonpositive T/K in {path}.")
        if "batch_index" in row and (
            int(row["batch_index"]) != 0 or not row.get("batch_payload_sha256")
        ):
            raise ValueError(f"Expected guarded batch zero in {path}.")
        if allow_aggregate and row.get("state_layer_name") not in allowed_state_names:
            raise ValueError(f"Unexpected state layer in {path}: {row.get('state_layer_name')!r}.")


def _bundle_metadata(
    run_dir: Path,
    *,
    run_kind: str,
    expected: Mapping[str, Any] | None,
) -> dict[str, Any]:
    validation_errors = reporting.validate_run(run_dir)
    if validation_errors:
        raise ValueError(
            f"Invalid canonical bundle {run_dir}: " + "; ".join(validation_errors)
        )
    manifest = _read_json(run_dir / "manifest.json")
    result = _read_json(run_dir / "result.json")
    if bool(manifest.get("smoke")) or bool(result.get("smoke")):
        raise ValueError(f"Production analysis cannot include smoke bundle {run_dir}.")
    if result.get("dataset") != "ordinary_mnist":
        raise ValueError(f"Expected ordinary_mnist in {run_dir}.")
    if result.get("run_id") != run_dir.name or manifest.get("run_id") != run_dir.name:
        raise ValueError(f"Run identity differs from directory name for {run_dir}.")
    completion = result.get("completion", {})
    failed_guards = [field for field in COMPLETION_GUARDS if completion.get(field) is not True]
    if failed_guards:
        raise ValueError(f"Completion guards failed in {run_dir}: {failed_guards}.")
    if completion.get("optimizer_steps_applied") is not False:
        raise ValueError(f"Optimizer-step guard failed in {run_dir}.")
    if completion.get("official_test_read") is not False:
        raise ValueError(f"Official-test guard failed in {run_dir}.")

    parameter_path = run_dir / "parameter_precision_comparison.csv"
    scaling_path = run_dir / "beta_scaling.csv"
    residual_path = run_dir / "equilibrium_residual_summary.csv"
    displacement_path = run_dir / "state_displacement.csv"
    parameter_rows = _read_csv(parameter_path, PARAMETER_REQUIRED_COLUMNS)
    scaling_rows = _read_csv(scaling_path, SCALING_REQUIRED_COLUMNS)
    residual_rows = _read_csv(residual_path, RESIDUAL_REQUIRED_COLUMNS)
    displacement_rows = _read_csv(displacement_path, DISPLACEMENT_REQUIRED_COLUMNS)
    if not parameter_rows or not scaling_rows or not residual_rows or not displacement_rows:
        raise ValueError(f"Empty scientific artifact in {run_dir}.")
    _validate_context_fields(parameter_rows, path=parameter_path)
    _validate_context_fields(scaling_rows, path=scaling_path)
    _validate_context_fields(residual_rows, path=residual_path, allow_aggregate=True)
    _validate_context_fields(displacement_rows, path=displacement_path, allow_aggregate=True)

    schemes = sorted({row["scheme"] for row in parameter_rows})
    if expected is not None and schemes != sorted(SCHEMES):
        raise ValueError(f"Core bundle {run_dir} does not contain both schemes.")
    for scheme in schemes:
        rows = [row for row in parameter_rows if row["scheme"] == scheme]
        names = [row["parameter_name"] for row in rows]
        if sorted(names) != sorted(PARAMETERS) or len(names) != len(set(names)):
            raise ValueError(f"Incomplete or duplicate weight layers for {run_dir}/{scheme}.")
        if not all(_parse_bool(row["bias_excluded"]) for row in rows):
            raise ValueError(f"Bias gradients are not excluded in {run_dir}/{scheme}.")
        for precision in PRECISIONS:
            if any(
                row[f"{precision}_eqprop_gradient_dtype"] != f"torch.{precision}"
                or row[f"{precision}_bptt_gradient_dtype"] != f"torch.{precision}"
                for row in rows
            ):
                raise ValueError(f"True-{precision} gradient guard failed in {run_dir}/{scheme}.")
        scaling = [row for row in scaling_rows if row["scheme"] == scheme]
        if len(scaling) != 1:
            raise ValueError(f"Expected one scaling row for {run_dir}/{scheme}.")
        scaling_row = scaling[0]
        effective_beta = float(_single((row["effective_beta"] for row in rows), label="injected beta"))
        if not _close(effective_beta, float(scaling_row["effective_beta"])):
            raise ValueError(f"Parameter/scaling injected beta mismatch in {run_dir}/{scheme}.")
        context_fields = (
            "T",
            "K",
            "source_native_T",
            "source_native_K",
            "native_context",
            "tk_source",
            "batch_payload_sha256",
        )
        for field in context_fields:
            if len({row[field] for row in rows}) != 1:
                raise ValueError(f"Inconsistent {field} in {run_dir}/{scheme}.")
        if expected is not None:
            if not _close(effective_beta, float(expected["injected_beta"])):
                raise ValueError(
                    f"Injected beta mismatch for {run_dir}/{scheme}: {effective_beta}."
                )
            native_context = _parse_bool(rows[0]["native_context"])
            if expected["tk_mode"] == "native":
                if not native_context or (
                    int(rows[0]["T"]), int(rows[0]["K"])
                ) != (
                    int(rows[0]["source_native_T"]),
                    int(rows[0]["source_native_K"]),
                ):
                    raise ValueError(f"Native T/K contract failed in {run_dir}/{scheme}.")
            elif (
                int(rows[0]["T"]), int(rows[0]["K"]), native_context
            ) != (64, 64, False):
                raise ValueError(f"T/K=64/64 contract failed in {run_dir}/{scheme}.")

    observed_contexts = {
        (row["scheme"], row["precision"])
        for row in residual_rows
    }
    expected_contexts = {(scheme, precision) for scheme in schemes for precision in PRECISIONS}
    if observed_contexts != expected_contexts:
        raise ValueError(f"Residual precision coverage differs in {run_dir}.")
    for key in expected_contexts:
        rows = [row for row in residual_rows if (row["scheme"], row["precision"]) == key]
        if not {"post_T_free", "zero", "positive"}.issubset({row["phase"] for row in rows}):
            raise ValueError(f"Residual phase coverage differs in {run_dir}/{key}.")
        if any(not _parse_bool(row["coverage_complete"]) or row["outcome"] != "ok" for row in rows):
            raise ValueError(f"Residual coverage guard failed in {run_dir}/{key}.")

    displacement_contexts = {
        (row["scheme"], row["precision"])
        for row in displacement_rows
    }
    if displacement_contexts != expected_contexts:
        raise ValueError(f"Displacement precision coverage differs in {run_dir}.")
    required_state_names = {*STATE_LAYER_BY_PARAMETER.values(), "__all__"}
    for key in expected_contexts:
        rows = [row for row in displacement_rows if (row["scheme"], row["precision"]) == key]
        if {row["reference_kind"] for row in rows} != {"post_T_free", "matched_zero_K"}:
            raise ValueError(f"Displacement reference coverage differs in {run_dir}/{key}.")
        for reference in ("post_T_free", "matched_zero_K"):
            names = {
                row["state_layer_name"]
                for row in rows
                if row["reference_kind"] == reference
            }
            if names != required_state_names:
                raise ValueError(f"Displacement layer coverage differs in {run_dir}/{key}/{reference}.")
        if any(row["outcome"] != "ok" for row in rows):
            raise ValueError(f"Displacement outcome failed in {run_dir}/{key}.")

    return {
        "run_dir": run_dir,
        "run_id": run_dir.name,
        "run_kind": run_kind,
        "result_sha256": reporting.sha256_file(run_dir / "result.json"),
        "manifest_sha256": reporting.sha256_file(run_dir / "manifest.json"),
        "metrics_sha256": reporting.sha256_file(run_dir / "metrics.jsonl"),
        "parameter_rows": parameter_rows,
        "scaling_rows": scaling_rows,
        "residual_rows": residual_rows,
        "displacement_rows": displacement_rows,
        "schemes": schemes,
    }


def discover_bundles(
    study_root: Path,
    *,
    confirmation_runs: Sequence[Path] = (),
    auto_confirmations: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    study_root = study_root.expanduser().resolve()
    bundles: list[dict[str, Any]] = []
    for name, expected in CORE_RUN_SPECS.items():
        run_dir = study_root / name
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Missing required core bundle: {run_dir}")
        bundles.append(_bundle_metadata(run_dir, run_kind="core", expected=expected))

    optional_paths: list[Path] = []
    excluded: list[dict[str, Any]] = []
    if auto_confirmations and study_root.is_dir():
        optional_paths.extend(
            path
            for path in sorted(study_root.iterdir())
            if path.is_dir()
            and path.name not in CORE_RUN_SPECS
            and (path / "manifest.json").is_file()
            and (path / "status.json").is_file()
        )
    for supplied in confirmation_runs:
        path = supplied.expanduser()
        if not path.is_absolute():
            path = study_root / path
        optional_paths.append(path.resolve())
    seen = {bundle["run_dir"] for bundle in bundles}
    for run_dir in optional_paths:
        if run_dir in seen:
            continue
        if not run_dir.is_dir():
            raise FileNotFoundError(run_dir)
        manifest = _read_json(run_dir / "manifest.json")
        status = _read_json(run_dir / "status.json")
        if bool(manifest.get("smoke")):
            continue
        if status.get("state") == "failed":
            excluded.append({
                "run_id": run_dir.name,
                "path": str(run_dir),
                "state": "failed",
                "reason": "failed_operational_attempt_no_result",
                "error": status.get("error"),
                "manifest_sha256": reporting.sha256_file(run_dir / "manifest.json"),
                "status_sha256": reporting.sha256_file(run_dir / "status.json"),
            })
            continue
        if status.get("state") != "complete":
            raise ValueError(f"Unresolved optional bundle {run_dir}: {status.get('state')!r}.")
        bundles.append(
            _bundle_metadata(run_dir, run_kind="adaptive_confirmation", expected=None)
        )
        seen.add(run_dir)
    return bundles, excluded


def _residual_metrics(
    rows: Sequence[Mapping[str, str]], *, scheme: str, precision: str
) -> dict[str, Any]:
    selected = [
        row for row in rows if row["scheme"] == scheme and row["precision"] == precision
    ]
    if not selected:
        raise ValueError(f"Missing residual rows for {scheme}/{precision}.")

    def phase_max(phase: str) -> float | None:
        values = [
            _optional_float(row["selected_max_p90"])
            for row in selected
            if row["phase"] == phase
        ]
        finite = [value for value in values if value is not None]
        return max(finite) if finite else None

    p90_values = [_optional_float(row["selected_max_p90"]) for row in selected]
    maxima = [_optional_float(row["selected_maximum"]) for row in selected]
    return {
        "all_residual_gates_passed": all(_parse_bool(row["gate_passed"]) for row in selected),
        "post_t_free_worst_selected_max_p90": phase_max("post_T_free"),
        "zero_worst_selected_max_p90": phase_max("zero"),
        "positive_worst_selected_max_p90": phase_max("positive"),
        "worst_residual_selected_max_p90": max(value for value in p90_values if value is not None),
        "worst_residual_selected_maximum": max(value for value in maxima if value is not None),
        "residual_hard_failure": any(_parse_bool(row["hard_failure"]) for row in selected),
    }


def _displacement_index(
    rows: Sequence[Mapping[str, str]], *, scheme: str, precision: str
) -> dict[tuple[str, str], Mapping[str, str]]:
    selected = [
        row for row in rows if row["scheme"] == scheme and row["precision"] == precision
    ]
    index: dict[tuple[str, str], Mapping[str, str]] = {}
    for row in selected:
        key = (row["reference_kind"], row["state_layer_name"])
        if key in index:
            raise ValueError(f"Duplicate displacement row for {scheme}/{precision}/{key}.")
        index[key] = row
    return index


def normalized_rows(bundles: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for bundle in bundles:
        for parameter_row in bundle["parameter_rows"]:
            scheme = parameter_row["scheme"]
            parameter = parameter_row["parameter_name"]
            state_layer = STATE_LAYER_BY_PARAMETER[parameter]
            for precision in PRECISIONS:
                residual = _residual_metrics(
                    bundle["residual_rows"], scheme=scheme, precision=precision
                )
                displacement = _displacement_index(
                    bundle["displacement_rows"], scheme=scheme, precision=precision
                )
                matched = displacement[("matched_zero_K", state_layer)]
                post_t = displacement[("post_T_free", state_layer)]
                matched_all = displacement[("matched_zero_K", "__all__")]
                post_t_all = displacement[("post_T_free", "__all__")]
                status = _gradient_status(parameter_row, precision)
                item = {
                    "schema": SCHEMA,
                    "run_id": bundle["run_id"],
                    "run_kind": bundle["run_kind"],
                    "input_result_sha256": bundle["result_sha256"],
                    "architecture": "conv3",
                    "scheme": scheme,
                    "precision": precision,
                    "checkpoint_role": parameter_row["checkpoint_role"],
                    "T": int(parameter_row["T"]),
                    "K": int(parameter_row["K"]),
                    "source_native_T": int(parameter_row["source_native_T"]),
                    "source_native_K": int(parameter_row["source_native_K"]),
                    "native_context": _parse_bool(parameter_row["native_context"]),
                    "tk_source": parameter_row["tk_source"],
                    "batch_index": int(parameter_row["batch_index"]),
                    "batch_payload_sha256": parameter_row["batch_payload_sha256"],
                    "actual_base_beta": float(parameter_row["actual_beta"]),
                    "injected_beta": float(parameter_row["effective_beta"]),
                    "beta_hat_requested": float(parameter_row["beta_hat_requested"]),
                    "beta_capped": _parse_bool(parameter_row["beta_capped"]),
                    "source_grid_cap_bypassed": _parse_bool(
                        parameter_row["source_grid_cap_bypassed"]
                    ),
                    "beta_source": parameter_row["beta_source"],
                    "parameter_name": parameter,
                    "layer": LAYER_LABELS[parameter],
                    "state_layer_name": state_layer,
                    "bias_excluded": True,
                    "eqprop_vs_bptt_cosine": _optional_float(
                        parameter_row[f"{precision}_eqprop_vs_bptt_cosine"]
                    ),
                    "gradient_status": status,
                    "eqprop_l2": float(parameter_row[f"{precision}_eqprop_l2"]),
                    "bptt_l2": float(parameter_row[f"{precision}_bptt_l2"]),
                    "exact_zero_fraction": float(
                        parameter_row[f"{precision}_exact_zero_fraction"]
                    ),
                    "eqprop_over_bptt_norm_ratio": _optional_float(
                        parameter_row.get(
                            f"{precision}_eqprop_over_bptt_norm_ratio", ""
                        )
                    ),
                    "eqprop_vs_bptt_symmetric_norm_delta": _optional_float(
                        parameter_row.get(
                            f"{precision}_eqprop_vs_bptt_symmetric_norm_delta", ""
                        )
                    ),
                    "residual_limited": _parse_bool(parameter_row["residual_limited"]),
                    **residual,
                    "matched_zero_relative_state_displacement": float(
                        matched["relative_displacement"]
                    ),
                    "post_t_relative_state_displacement": float(
                        post_t["relative_displacement"]
                    ),
                    "matched_zero_aggregate_relative_displacement": float(
                        matched_all["relative_displacement"]
                    ),
                    "post_t_aggregate_relative_displacement": float(
                        post_t_all["relative_displacement"]
                    ),
                    "matched_zero_active_set_transition_fraction": float(
                        matched["active_set_transition_fraction"]
                    ),
                }
                output.append(item)

    cohort_hashes = {row["batch_payload_sha256"] for row in output}
    if len(cohort_hashes) != 1 or {row["batch_index"] for row in output} != {0}:
        raise ValueError("Input bundles do not share the exact fixed batch zero.")
    keys: set[tuple[Any, ...]] = set()
    for row in output:
        key = (
            row["scheme"],
            row["precision"],
            row["T"],
            row["K"],
            row["injected_beta"],
            row["parameter_name"],
        )
        if key in keys:
            raise ValueError(f"Duplicate scientific context: {key}.")
        keys.add(key)
    return sorted(
        output,
        key=lambda row: (
            SCHEMES.index(row["scheme"]),
            PRECISIONS.index(row["precision"]),
            not row["native_context"],
            row["T"],
            row["K"],
            row["injected_beta"],
            PARAMETERS.index(row["parameter_name"]),
        ),
    )


def write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _display_gradient(row: Mapping[str, Any]) -> str:
    if row["gradient_status"] == "exact_zero":
        return "ZERO"
    if row["gradient_status"] == "undefined":
        return "undef"
    cosine = f"{float(row['eqprop_vs_bptt_cosine']):.5f}"
    ratio = row["eqprop_over_bptt_norm_ratio"]
    delta = row["eqprop_vs_bptt_symmetric_norm_delta"]
    extras = []
    if ratio is not None:
        extras.append(f"r={float(ratio):.3g}")
    if delta is not None:
        extras.append(f"d={float(delta):.3g}")
    return cosine + (" (" + ", ".join(extras) + ")" if extras else "")


def _contexts(rows: Sequence[Mapping[str, Any]]) -> list[list[Mapping[str, Any]]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (
            row["run_id"],
            row["scheme"],
            row["precision"],
            row["T"],
            row["K"],
            row["injected_beta"],
        )
        groups.setdefault(key, []).append(row)
    contexts = list(groups.values())
    if any({row["parameter_name"] for row in group} != set(PARAMETERS) for group in contexts):
        raise ValueError("A report context lacks complete layer coverage.")
    return sorted(
        contexts,
        key=lambda group: (
            SCHEMES.index(group[0]["scheme"]),
            PRECISIONS.index(group[0]["precision"]),
            not group[0]["native_context"],
            group[0]["T"],
            group[0]["K"],
            group[0]["injected_beta"],
        ),
    )


def write_markdown(
    rows: Sequence[Mapping[str, Any]],
    *,
    bundles: Sequence[Mapping[str, Any]],
    excluded_attempts: Sequence[Mapping[str, Any]],
    path: Path,
) -> None:
    lines = [
        "# Conv3 high-beta/high-T/K true-dtype EqProp diagnostic",
        "",
        "## Input provenance",
        "",
        "| run | kind | result SHA-256 | schemes |",
        "|---|---|---|---|",
    ]
    for bundle in bundles:
        lines.append(
            f"| `{bundle['run_id']}` | {bundle['run_kind']} | `{bundle['result_sha256']}` | "
            f"{', '.join(bundle['schemes'])} |"
        )
    if excluded_attempts:
        lines.extend(["", "## Excluded attempts", ""])
        for attempt in excluded_attempts:
            lines.append(
                f"- `{attempt['run_id']}`: excluded from scientific curves because it ended "
                f"in state `{attempt['state']}` ({attempt['reason']}); status SHA-256 "
                f"`{attempt['status_sha256']}`."
            )
    lines.extend(
        [
            "",
            "## Layerwise EqProp versus BPTT",
            "",
            "Each numeric cell is `cosine (r=EqProp/BPTT norm ratio, d=symmetric norm delta)`. "
            "`ZERO` is categorical: the EqProp gradient is exactly zero and its cosine is undefined.",
            "",
            "| scheme | precision | T/K | injected beta | C0 | C1 | C2 | D | residual gate | worst p90 |",
            "|---|---|---:|---:|---|---|---|---|:---:|---:|",
        ]
    )
    contexts = _contexts(rows)
    for group in contexts:
        by_parameter = {row["parameter_name"]: row for row in group}
        anchor = group[0]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(anchor["scheme"]),
                    str(anchor["precision"]),
                    f"{anchor['T']}/{anchor['K']}",
                    f"{float(anchor['injected_beta']):.6g}",
                    *[_display_gradient(by_parameter[name]) for name in PARAMETERS],
                    "pass" if anchor["all_residual_gates_passed"] else "FAIL",
                    f"{float(anchor['worst_residual_selected_max_p90']):.6g}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Matched-zero relative state displacement",
            "",
            "| scheme | precision | T/K | injected beta | H0 | H1 | H2 | output | aggregate |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for group in contexts:
        by_parameter = {row["parameter_name"]: row for row in group}
        anchor = group[0]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(anchor["scheme"]),
                    str(anchor["precision"]),
                    f"{anchor['T']}/{anchor['K']}",
                    f"{float(anchor['injected_beta']):.6g}",
                    *[
                        f"{float(by_parameter[name]['matched_zero_relative_state_displacement']):.6g}"
                        for name in PARAMETERS
                    ],
                    f"{float(anchor['matched_zero_aggregate_relative_displacement']):.6g}",
                ]
            )
            + " |"
        )
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {limitation}" for limitation in LIMITATIONS)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _series_label(row: Mapping[str, Any]) -> str:
    if row["native_context"]:
        return f"native {row['T']}/{row['K']}"
    return f"T/K={row['T']}/{row['K']}"


def _style_map(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    labels = sorted({_series_label(row) for row in rows})
    colors = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9")
    markers = ("o", "s", "D", "^", "v")
    output: dict[str, dict[str, Any]] = {}
    for index, label in enumerate(labels):
        if label.startswith("native"):
            color, marker, linestyle = "#0072B2", "o", "-"
        elif label == "T/K=64/64":
            color, marker, linestyle = "#E69F00", "s", "--"
        else:
            color = colors[(index + 2) % len(colors)]
            marker = markers[(index + 2) % len(markers)]
            linestyle = ":"
        output[label] = {"color": color, "marker": marker, "linestyle": linestyle}
    return output


def render_plot(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    figure, axes = plt.subplots(
        8,
        4,
        figsize=(17.5, 24.0),
        sharex=True,
        sharey="row",
        squeeze=False,
    )
    style_map = _style_map(rows)
    zero_rail = -0.075
    positive_displacements = [
        float(row["matched_zero_relative_state_displacement"])
        for row in rows
        if float(row["matched_zero_relative_state_displacement"]) > 0.0
    ]
    displacement_min = max(min(positive_displacements) * 0.5, 1.0e-18)
    displacement_max = max(positive_displacements) * 2.0
    beta_values = [float(row["injected_beta"]) for row in rows]
    beta_min = min(beta_values) / 1.4
    beta_max = max(beta_values) * 1.4

    for column, (scheme, precision) in enumerate(
        (scheme_precision for scheme_precision in ((s, p) for s in SCHEMES for p in PRECISIONS))
    ):
        column_rows = [
            row for row in rows if row["scheme"] == scheme and row["precision"] == precision
        ]
        axes[0, column].set_title(f"{scheme} — {precision}", fontsize=12)
        for parameter_index, parameter in enumerate(PARAMETERS):
            layer_rows = [row for row in column_rows if row["parameter_name"] == parameter]
            cosine_axis = axes[parameter_index, column]
            displacement_axis = axes[parameter_index + len(PARAMETERS), column]
            labels = sorted({_series_label(row) for row in layer_rows})
            for label in labels:
                series = sorted(
                    [row for row in layer_rows if _series_label(row) == label],
                    key=lambda row: row["injected_beta"],
                )
                style = style_map[label]
                numeric = [row for row in series if row["gradient_status"] == "numeric"]
                cosine_axis.plot(
                    [row["injected_beta"] for row in numeric],
                    [row["eqprop_vs_bptt_cosine"] for row in numeric],
                    label=label,
                    linewidth=1.7,
                    markersize=5.0,
                    **style,
                )
                exact_zero = [row for row in series if row["gradient_status"] == "exact_zero"]
                if exact_zero:
                    cosine_axis.scatter(
                        [row["injected_beta"] for row in exact_zero],
                        [zero_rail] * len(exact_zero),
                        color=style["color"],
                        marker="X",
                        s=48,
                        zorder=5,
                    )
                undefined = [row for row in series if row["gradient_status"] == "undefined"]
                if undefined:
                    cosine_axis.scatter(
                        [row["injected_beta"] for row in undefined],
                        [zero_rail] * len(undefined),
                        facecolors="none",
                        edgecolors=style["color"],
                        marker="D",
                        s=40,
                        zorder=5,
                    )
                displacement_axis.plot(
                    [row["injected_beta"] for row in series],
                    [row["matched_zero_relative_state_displacement"] for row in series],
                    label=label,
                    linewidth=1.7,
                    markersize=5.0,
                    **style,
                )
                failing = [row for row in series if not row["all_residual_gates_passed"]]
                if failing:
                    cosine_axis.scatter(
                        [row["injected_beta"] for row in failing],
                        [
                            row["eqprop_vs_bptt_cosine"]
                            if row["gradient_status"] == "numeric"
                            else zero_rail
                            for row in failing
                        ],
                        facecolors="none",
                        edgecolors="#D55E00",
                        linewidths=1.4,
                        s=85,
                        zorder=6,
                    )
                    displacement_axis.scatter(
                        [row["injected_beta"] for row in failing],
                        [row["matched_zero_relative_state_displacement"] for row in failing],
                        facecolors="none",
                        edgecolors="#D55E00",
                        linewidths=1.4,
                        s=85,
                        zorder=6,
                    )

            cosine_axis.axhline(0.99, color="#555555", linestyle=":", linewidth=1.0)
            cosine_axis.axhline(zero_rail, color="#777777", linestyle=":", linewidth=0.8)
            cosine_axis.set_ylim(-0.115, 1.035)
            cosine_axis.set_yticks((zero_rail, 0.0, 0.5, 0.9, 0.99, 1.0))
            cosine_axis.set_yticklabels(("ZERO", "0", "0.5", "0.9", "0.99", "1"))
            displacement_axis.set_yscale("log")
            displacement_axis.set_ylim(displacement_min, displacement_max)
            for axis in (cosine_axis, displacement_axis):
                axis.set_xscale("log")
                axis.set_xlim(beta_min, beta_max)
                axis.grid(True, which="both", alpha=0.16)
            if column == 0:
                cosine_axis.set_ylabel(f"{LAYER_LABELS[parameter]} cosine")
                displacement_axis.set_ylabel(
                    f"{LAYER_LABELS[parameter]} state\nrelative displacement"
                )
            if parameter_index == len(PARAMETERS) - 1:
                displacement_axis.set_xlabel("injected beta")

    legend_handles: list[Line2D] = []
    for label, style in style_map.items():
        legend_handles.append(Line2D([0], [0], label=label, linewidth=1.7, **style))
    legend_handles.extend(
        [
            Line2D([0], [0], marker="X", color="#333333", linestyle="none", label="exact-zero EqProp"),
            Line2D(
                [0],
                [0],
                marker="o",
                markerfacecolor="none",
                markeredgecolor="#D55E00",
                color="none",
                label="residual gate failed",
            ),
        ]
    )
    figure.suptitle(
        "Conv3 best-checkpoint one-sided frozen-current EqProp versus BPTT\n"
        "true float32/float64; baseline/ours; native and high T/K; weights only",
        fontsize=15,
        y=0.997,
    )
    figure.legend(
        legend_handles,
        [handle.get_label() for handle in legend_handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=4,
        frameon=False,
    )
    figure.text(0.01, 0.755, "EqProp–BPTT cosine", rotation=90, va="center", fontsize=12)
    figure.text(
        0.01,
        0.285,
        "Matched-zero relative state displacement",
        rotation=90,
        va="center",
        fontsize=12,
    )
    # Reserve a dedicated header band for the two-line title and two-row legend.
    figure.tight_layout(rect=(0.025, 0.015, 1.0, 0.905), h_pad=1.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _context_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for group in _contexts(rows):
        anchor = group[0]
        by_parameter = {row["parameter_name"]: row for row in group}
        output.append(
            {
                "run_id": anchor["run_id"],
                "scheme": anchor["scheme"],
                "precision": anchor["precision"],
                "T": anchor["T"],
                "K": anchor["K"],
                "native_context": anchor["native_context"],
                "injected_beta": anchor["injected_beta"],
                "layer_status": {
                    LAYER_LABELS[name]: {
                        "status": by_parameter[name]["gradient_status"],
                        "cosine": by_parameter[name]["eqprop_vs_bptt_cosine"],
                        "norm_ratio": by_parameter[name]["eqprop_over_bptt_norm_ratio"],
                        "symmetric_norm_delta": by_parameter[name][
                            "eqprop_vs_bptt_symmetric_norm_delta"
                        ],
                        "matched_zero_relative_state_displacement": by_parameter[name][
                            "matched_zero_relative_state_displacement"
                        ],
                    }
                    for name in PARAMETERS
                },
                "all_residual_gates_passed": anchor["all_residual_gates_passed"],
                "worst_residual_selected_max_p90": anchor[
                    "worst_residual_selected_max_p90"
                ],
                "matched_zero_aggregate_relative_displacement": anchor[
                    "matched_zero_aggregate_relative_displacement"
                ],
            }
        )
    return output


def write_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    bundles: Sequence[Mapping[str, Any]],
    excluded_attempts: Sequence[Mapping[str, Any]],
    study_root: Path,
    output_csv: Path,
    output_plot: Path,
    output_markdown: Path,
    path: Path,
) -> None:
    input_bundles = [
        {
            "run_id": bundle["run_id"],
            "run_kind": bundle["run_kind"],
            "path": str(bundle["run_dir"]),
            "result_sha256": bundle["result_sha256"],
            "manifest_sha256": bundle["manifest_sha256"],
            "metrics_sha256": bundle["metrics_sha256"],
            "schemes": bundle["schemes"],
        }
        for bundle in bundles
    ]
    summary = {
        "schema": SCHEMA,
        "study_root": str(study_root.expanduser().resolve()),
        "dataset": "ordinary_mnist",
        "evidence_class": "ordinary_mnist_learning_algorithm_gradient_diagnostic",
        "architecture": "conv3",
        "checkpoint_role": "best_validation",
        "precisions": list(PRECISIONS),
        "schemes": list(SCHEMES),
        "biases_active_in_dynamics": True,
        "bias_gradients_excluded": True,
        "read_only": True,
        "optimizer_constructed": False,
        "optimizer_steps_applied": False,
        "official_test_read": False,
        "fixed_batch": {
            "batch_index": 0,
            "example_count": 16,
            "payload_sha256": _single(
                (row["batch_payload_sha256"] for row in rows), label="batch payload SHA"
            ),
        },
        "required_core_run_ids": list(CORE_RUN_SPECS),
        "adaptive_confirmation_run_ids": [
            bundle["run_id"]
            for bundle in bundles
            if bundle["run_kind"] == "adaptive_confirmation"
        ],
        "input_bundles": input_bundles,
        "excluded_attempts": list(excluded_attempts),
        "row_count": len(rows),
        "exact_zero_cell_count": sum(row["gradient_status"] == "exact_zero" for row in rows),
        "undefined_nonzero_cell_count": sum(
            row["gradient_status"] == "undefined" for row in rows
        ),
        "residual_failed_context_count": sum(
            not group[0]["all_residual_gates_passed"] for group in _contexts(rows)
        ),
        "contexts": _context_summary(rows),
        "limitations": list(LIMITATIONS),
        "outputs": {
            "csv": {
                "path": str(output_csv.resolve()),
                "sha256": reporting.sha256_file(output_csv),
            },
            "figure": {
                "path": str(output_plot.resolve()),
                "sha256": reporting.sha256_file(output_plot),
            },
            "markdown": {
                "path": str(output_markdown.resolve()),
                "sha256": reporting.sha256_file(output_markdown),
            },
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_analysis(
    *,
    study_root: Path,
    analysis_dir: Path,
    confirmation_runs: Sequence[Path] = (),
    auto_confirmations: bool = True,
) -> dict[str, Any]:
    bundles, excluded_attempts = discover_bundles(
        study_root,
        confirmation_runs=confirmation_runs,
        auto_confirmations=auto_confirmations,
    )
    rows = normalized_rows(bundles)
    analysis_dir = analysis_dir.expanduser().resolve()
    output_csv = analysis_dir / "conv3_eqprop_high_beta_high_tk.csv"
    output_plot = analysis_dir / "conv3_eqprop_high_beta_high_tk.png"
    output_markdown = analysis_dir / "conv3_eqprop_high_beta_high_tk.md"
    output_summary = analysis_dir / "conv3_eqprop_high_beta_high_tk.json"
    write_csv(rows, output_csv)
    write_markdown(rows, bundles=bundles, excluded_attempts=excluded_attempts, path=output_markdown)
    render_plot(rows, output_plot)
    write_summary(
        rows,
        bundles=bundles,
        excluded_attempts=excluded_attempts,
        study_root=study_root,
        output_csv=output_csv,
        output_plot=output_plot,
        output_markdown=output_markdown,
        path=output_summary,
    )
    return {
        "row_count": len(rows),
        "bundle_count": len(bundles),
        "output_csv": str(output_csv),
        "output_plot": str(output_plot),
        "output_markdown": str(output_markdown),
        "output_summary": str(output_summary),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=DEFAULT_STUDY_ROOT)
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        help="Defaults to <study-root>/analysis.",
    )
    parser.add_argument(
        "--confirmation-run",
        type=Path,
        action="append",
        default=[],
        help="Optional adaptive confirmation bundle; relative paths resolve under the study root.",
    )
    parser.add_argument("--no-auto-confirmations", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    study_root = args.study_root.expanduser().resolve()
    analysis_dir = (
        args.analysis_dir.expanduser().resolve()
        if args.analysis_dir is not None
        else study_root / "analysis"
    )
    result = run_analysis(
        study_root=study_root,
        analysis_dir=analysis_dir,
        confirmation_runs=args.confirmation_run,
        auto_confirmations=not args.no_auto_confirmations,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
