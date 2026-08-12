#!/usr/bin/env python3
"""Compare one- and two-sided current-nudge EqProp at high beta on Conv3.

The primary inputs are completed canonical true-float32/float64 replay bundles
under the higher-beta one-versus-two-sided study.  Completed one-sided bundles
from the preceding high-beta study are admitted as lower-beta anchors only
when their scientific metadata matches the primary T/K coordinate, checkpoint,
fixed minibatch, and precision contract.

Run directory names are provenance labels only.  EqProp variant, beta, T/K,
scheme, checkpoint role, and cohort identity are resolved from canonical bundle
metadata and scientific CSVs.  Biases remain active in the replay dynamics but
are excluded from every scored gradient.
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
    / "results/perfectdiode-conv3-baseline-ours-eqprop-one-vs-two-sided-higher-beta-seed0-20260811-v1"
)
DEFAULT_ANCHOR_ROOT = (
    REPOSITORY_ROOT
    / "results/perfectdiode-conv3-baseline-ours-eqprop-high-beta-high-tk-seed0-20260811-v1"
)
SCHEMA = "conv3-eqprop-one-vs-two-sided-higher-beta-analysis/v1"
SCHEMES = ("baseline", "ours")
VARIANTS = ("positive_one_sided", "centered")
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
STATE_LABELS = {
    "Layer_1": "H0",
    "Layer_2": "H1",
    "Layer_3": "H2",
    "Layer_4": "output",
}
COMMON_COMPLETION_GUARDS = (
    "criteria_met",
    "completed_all_selected_cases",
    "all_weight_layer_precision_comparisons_present",
    "all_runtime_dtypes_proven",
    "all_iteration_contracts_passed",
    "all_finite_differences_computed_in_native_dtype",
    "all_bptt_gradients_computed_in_native_dtype",
    "all_eqprop_vs_bptt_layer_metrics_present",
    "fixed_cohort_and_batch_reproduced",
    "source_bundle_files_unchanged",
    "source_checkpoint_bytes_unchanged",
    "parameter_tensors_unchanged",
    "bias_gradients_excluded",
)
PARAMETER_REQUIRED_COLUMNS = {
    "schema",
    "architecture",
    "scheme",
    "checkpoint_role",
    "T",
    "K",
    "batch_index",
    "batch_payload_sha256",
    "parameter_name",
    "parameter_type",
    "bias_excluded",
    "actual_beta",
    "effective_beta",
    "amplification_factor",
    "beta_source",
    "residual_limited",
    *(f"{precision}_eqprop_gradient_dtype" for precision in PRECISIONS),
    *(f"{precision}_bptt_gradient_dtype" for precision in PRECISIONS),
    *(f"{precision}_eqprop_l2" for precision in PRECISIONS),
    *(f"{precision}_bptt_l2" for precision in PRECISIONS),
    *(f"{precision}_exact_zero_fraction" for precision in PRECISIONS),
    *(f"{precision}_eqprop_vs_bptt_cosine" for precision in PRECISIONS),
}
PARAMETER_SCHEMA_VARIANTS = {
    "perfectdiode-conv-positive-eqprop-float64-shadow-audit/v2": {
        "positive_one_sided"
    },
    "perfectdiode-conv-current-eqprop-true-dtype-audit/v3": set(VARIANTS),
}
DISPLACEMENT_REQUIRED_COLUMNS = {
    "architecture",
    "scheme",
    "checkpoint_role",
    "T",
    "K",
    "precision",
    "batch_index",
    "batch_payload_sha256",
    "batch_source_indices_sha256",
    "phase",
    "actual_beta",
    "reference_kind",
    "outcome",
    "state_layer_name",
    "displacement_l2",
    "relative_displacement",
    "active_set_transition_fraction",
}
OUTPUT_FIELDS = (
    "schema",
    "source_kind",
    "run_id",
    "input_result_sha256",
    "eqprop_variant",
    "architecture",
    "scheme",
    "precision",
    "checkpoint_role",
    "T",
    "K",
    "batch_index",
    "batch_payload_sha256",
    "batch_source_indices_sha256",
    "source_selection_row_sha256",
    "best_checkpoint_sha256",
    "actual_base_beta",
    "injected_beta",
    "beta_source",
    "parameter_name",
    "layer",
    "state_layer_name",
    "bias_excluded",
    "gradient_status",
    "eqprop_vs_bptt_cosine",
    "eqprop_l2",
    "bptt_l2",
    "exact_zero_fraction",
    "eqprop_over_bptt_norm_ratio",
    "eqprop_vs_bptt_symmetric_norm_delta",
    "residual_limited",
    "matched_zero_negative_relative_displacement",
    "matched_zero_negative_displacement_l2",
    "matched_zero_negative_active_set_transition_fraction",
    "matched_zero_positive_relative_displacement",
    "matched_zero_positive_displacement_l2",
    "matched_zero_positive_active_set_transition_fraction",
    "signed_span_present",
    "signed_span_relative_displacement",
    "signed_span_displacement_l2",
    "signed_span_active_set_transition_fraction",
)
LIMITATIONS = (
    "Ordinary-MNIST checkpoint replay is diagnostic evidence, not a deterministic medium-affine paper result.",
    "One fixed validation minibatch (batch 0, 16 examples) is replayed; minibatch spread is not measured.",
    "The checkpoints were trained with BPTT. This analysis compares read-only EqProp gradient estimates and does not perform EqProp training.",
    "Biases are active in the dynamics but bias gradients are excluded from every cosine.",
    "Betas above the source 0.01 injected-current cap are explicitly recorded diagnostic extrapolations.",
    "The signed-span field is the reported L2 displacement from the negative endpoint to the positive endpoint; it is a phase-oriented span, not a signed scalar coordinate.",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"Scientific artifact is empty: {path}.")
    return rows


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
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


def _variant_from_metadata(
    manifest: Mapping[str, Any],
    result: Mapping[str, Any],
    parameter_rows: Sequence[Mapping[str, str]],
) -> str:
    """Resolve the estimator variant without using a directory or run name."""

    candidates: list[str] = []
    terminal = result.get("terminal_metrics", {})
    if terminal.get("eqprop_variant"):
        candidates.append(str(terminal["eqprop_variant"]))
    resolved = manifest.get("configuration", {}).get("resolved", {})
    if resolved.get("eqprop_variant"):
        candidates.append(str(resolved["eqprop_variant"]))
    row_variants = {
        str(row.get("eqprop_variant", "")).strip()
        for row in parameter_rows
        if str(row.get("eqprop_variant", "")).strip()
    }
    candidates.extend(sorted(row_variants))
    arm_id = str(manifest.get("arm_id", ""))
    if "conv_centered_eqprop" in arm_id:
        candidates.append("centered")
    elif "conv_positive_eqprop" in arm_id:
        candidates.append("positive_one_sided")
    observed = set(candidates)
    if len(observed) != 1:
        raise ValueError(f"EqProp variant metadata is missing or inconsistent: {candidates}.")
    variant = next(iter(observed))
    if variant not in VARIANTS:
        raise ValueError(f"Unsupported EqProp variant {variant!r}.")
    return variant


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


def _validate_parameter_schema(
    rows: Sequence[Mapping[str, str]], *, variant: str, path: Path
) -> str:
    schema = str(_single((row.get("schema", "") for row in rows), label="row schema"))
    allowed_variants = PARAMETER_SCHEMA_VARIANTS.get(schema)
    if allowed_variants is None:
        raise ValueError(f"Unsupported true-dtype parameter schema {schema!r} in {path}.")
    if variant not in allowed_variants:
        raise ValueError(f"Schema {schema!r} cannot represent {variant!r} in {path}.")
    if schema.endswith("/v3") and any(
        str(row.get("eqprop_variant", "")).strip() != variant for row in rows
    ):
        raise ValueError(f"V3 rows must explicitly declare {variant!r} in {path}.")
    return schema


def _validate_displacement_parameter_context(
    parameter_rows: Sequence[Mapping[str, str]],
    displacement_rows: Sequence[Mapping[str, str]],
    *,
    variant: str,
    path: Path,
) -> None:
    allowed_phase_references = {
        ("positive", "post_T_free"),
        ("positive", "matched_zero_K"),
    }
    if variant == "centered":
        allowed_phase_references.update(
            {
                ("negative", "post_T_free"),
                ("negative", "matched_zero_K"),
                ("positive_minus_negative", "negative_phase"),
            }
        )
    for scheme in {row["scheme"] for row in parameter_rows}:
        parameters = [row for row in parameter_rows if row["scheme"] == scheme]
        displacements = [row for row in displacement_rows if row["scheme"] == scheme]
        expected_context = {
            "T": str(_single((row["T"] for row in parameters), label=f"{scheme} T")),
            "K": str(_single((row["K"] for row in parameters), label=f"{scheme} K")),
            "batch_index": str(
                _single((row["batch_index"] for row in parameters), label=f"{scheme} batch")
            ),
            "batch_payload_sha256": str(
                _single(
                    (row["batch_payload_sha256"] for row in parameters),
                    label=f"{scheme} batch payload",
                )
            ),
            "checkpoint_role": str(
                _single(
                    (row["checkpoint_role"] for row in parameters),
                    label=f"{scheme} checkpoint role",
                )
            ),
        }
        for row in displacements:
            for field, expected in expected_context.items():
                if str(row[field]) != expected:
                    raise ValueError(
                        f"Displacement {field} differs from parameter context in "
                        f"{path}/{scheme}: {row[field]!r} != {expected!r}."
                    )
            phase_reference = (row["phase"], row["reference_kind"])
            if phase_reference not in allowed_phase_references:
                raise ValueError(
                    f"Unexpected displacement phase/reference {phase_reference} in {path}."
                )
            base_beta = float(parameters[0]["actual_beta"])
            expected_beta = -base_beta if row["phase"] == "negative" else base_beta
            if not _close(float(row["actual_beta"]), expected_beta):
                raise ValueError(
                    f"Displacement beta sign/magnitude mismatch in {path}/{scheme}/"
                    f"{row['phase']}."
                )


def _completion_guard(result: Mapping[str, Any], variant: str, run_dir: Path) -> None:
    completion = result.get("completion", {})
    failed = [
        field for field in COMMON_COMPLETION_GUARDS if completion.get(field) is not True
    ]
    phase_guard = completion.get(
        "bptt_and_configured_eqprop_phases_started_from_identical_post_T"
    )
    if phase_guard is None and variant == "positive_one_sided":
        phase_guard = completion.get(
            "bptt_zero_and_positive_started_from_identical_post_T"
        )
    if phase_guard is not True:
        failed.append("configured_phase_start_state_guard")
    if failed:
        raise ValueError(f"Completion guards failed in {run_dir}: {sorted(set(failed))}.")
    if completion.get("optimizer_steps_applied") is not False:
        raise ValueError(f"Optimizer-step guard failed in {run_dir}.")
    if completion.get("official_test_read") is not False:
        raise ValueError(f"Official-test guard failed in {run_dir}.")


def _validate_common_rows(
    rows: Sequence[Mapping[str, str]], *, path: Path, variant: str
) -> None:
    for row in rows:
        if row["architecture"] != "conv3":
            raise ValueError(f"Non-Conv3 row in {path}.")
        if row["scheme"] not in SCHEMES:
            raise ValueError(f"Unexpected scheme {row['scheme']!r} in {path}.")
        if row["checkpoint_role"] != "best_validation":
            raise ValueError(f"Non-best checkpoint row in {path}.")
        if int(row["T"]) <= 0 or int(row["K"]) <= 0:
            raise ValueError(f"Nonpositive T/K in {path}.")
        if int(row["batch_index"]) != 0 or not row["batch_payload_sha256"]:
            raise ValueError(f"Expected guarded batch zero in {path}.")
        declared_variant = str(row.get("eqprop_variant", "")).strip()
        if declared_variant and declared_variant != variant:
            raise ValueError(f"Row variant differs from bundle metadata in {path}.")


def _load_bundle(run_dir: Path, *, source_kind: str) -> dict[str, Any]:
    validation_errors = reporting.validate_run(run_dir)
    if validation_errors:
        raise ValueError(
            f"Invalid canonical bundle {run_dir}: " + "; ".join(validation_errors)
        )
    manifest = _read_json(run_dir / "manifest.json")
    result = _read_json(run_dir / "result.json")
    if bool(manifest.get("smoke")) or bool(result.get("smoke")):
        raise ValueError(f"Production analysis cannot include smoke bundle {run_dir}.")
    if manifest.get("run_id") != run_dir.name or result.get("run_id") != run_dir.name:
        raise ValueError(f"Run identity differs from directory name for {run_dir}.")
    if result.get("dataset") != "ordinary_mnist":
        raise ValueError(f"Expected ordinary_mnist in {run_dir}.")

    parameter_path = run_dir / "parameter_precision_comparison.csv"
    displacement_path = run_dir / "state_displacement.csv"
    parameter_rows = _read_csv(parameter_path, PARAMETER_REQUIRED_COLUMNS)
    displacement_rows = _read_csv(displacement_path, DISPLACEMENT_REQUIRED_COLUMNS)
    variant = _variant_from_metadata(manifest, result, parameter_rows)
    _validate_parameter_schema(parameter_rows, variant=variant, path=parameter_path)
    _completion_guard(result, variant, run_dir)
    replay = manifest.get("replay", {})
    if replay.get("nudging_mode") != "current":
        raise ValueError(f"Expected current nudging in {run_dir}.")
    if list(replay.get("precisions", ())) != list(PRECISIONS):
        raise ValueError(f"Expected true float32/float64 replay in {run_dir}.")
    if replay.get("finite_difference_arithmetic") != (
        "native_runtime_dtype_before_diagnostic_cast"
    ):
        raise ValueError(f"Finite-difference arithmetic contract failed in {run_dir}.")
    if replay.get("optimizer_steps_applied") is not False:
        raise ValueError(f"Replay optimizer-step guard failed in {run_dir}.")
    if replay.get("official_test_read") is not False:
        raise ValueError(f"Replay official-test guard failed in {run_dir}.")
    fixed_batch = replay.get("fixed_batch", {})
    fixed_source_indices_sha = str(fixed_batch.get("source_indices_sha256", ""))
    if not fixed_source_indices_sha:
        raise ValueError(f"Missing fixed source-index SHA in {run_dir}.")
    if any(
        row["batch_source_indices_sha256"] != fixed_source_indices_sha
        for row in displacement_rows
    ):
        raise ValueError(f"Manifest/displacement source-index mismatch in {run_dir}.")
    resolved = manifest.get("configuration", {}).get("resolved", {})
    selected_cases = resolved.get("selected_cases", ())
    source_cases = resolved.get("source_beta_config", {}).get("cases", ())
    case_identity_by_scheme: dict[str, dict[str, str]] = {}
    for scheme in sorted({row["scheme"] for row in parameter_rows}):
        selected = [
            row
            for row in selected_cases
            if str(row.get("architecture")) == "conv3"
            and str(row.get("scheme")) == scheme
            and str(row.get("checkpoint_role")) == "best_validation"
        ]
        sources = [
            row
            for row in source_cases
            if str(row.get("architecture")) == "conv3"
            and str(row.get("scheme")) == scheme
        ]
        if len(selected) != 1 or len(sources) != 1:
            raise ValueError(f"Could not resolve one source identity for {run_dir}/{scheme}.")
        selection_sha = str(selected[0].get("source_selection_row_sha256", ""))
        checkpoint_sha = str(
            sources[0].get(
                "best_checkpoint_sha256", sources[0].get("best_model_pt_sha256", "")
            )
        )
        if not selection_sha or not checkpoint_sha:
            raise ValueError(f"Missing source/checkpoint SHA for {run_dir}/{scheme}.")
        case_identity_by_scheme[scheme] = {
            "source_selection_row_sha256": selection_sha,
            "best_checkpoint_sha256": checkpoint_sha,
        }
    _validate_common_rows(parameter_rows, path=parameter_path, variant=variant)
    _validate_common_rows(displacement_rows, path=displacement_path, variant=variant)
    _validate_displacement_parameter_context(
        parameter_rows,
        displacement_rows,
        variant=variant,
        path=displacement_path,
    )

    schemes = sorted({row["scheme"] for row in parameter_rows})
    for scheme in schemes:
        scheme_rows = [row for row in parameter_rows if row["scheme"] == scheme]
        names = [row["parameter_name"] for row in scheme_rows]
        if sorted(names) != sorted(PARAMETERS) or len(names) != len(set(names)):
            raise ValueError(f"Incomplete or duplicate weight layers in {run_dir}/{scheme}.")
        if not all(_parse_bool(row["bias_excluded"]) for row in scheme_rows):
            raise ValueError(f"Bias gradients are not excluded in {run_dir}/{scheme}.")
        for precision in PRECISIONS:
            if any(
                row[f"{precision}_eqprop_gradient_dtype"] != f"torch.{precision}"
                or row[f"{precision}_bptt_gradient_dtype"] != f"torch.{precision}"
                for row in scheme_rows
            ):
                raise ValueError(
                    f"True-{precision} gradient guard failed in {run_dir}/{scheme}."
                )
        for field in (
            "T",
            "K",
            "batch_index",
            "batch_payload_sha256",
            "actual_beta",
            "effective_beta",
        ):
            _single((row[field] for row in scheme_rows), label=f"{scheme} {field}")
        actual_beta = float(scheme_rows[0]["actual_beta"])
        injected_beta = float(scheme_rows[0]["effective_beta"])
        amplification_factor = float(
            _single(
                (row["amplification_factor"] for row in scheme_rows),
                label=f"{scheme} amplification factor",
            )
        )
        if actual_beta <= 0.0 or injected_beta <= 0.0 or amplification_factor <= 0.0:
            raise ValueError(f"Nonpositive beta/scaling metadata in {run_dir}/{scheme}.")
        if not _close(injected_beta, actual_beta * amplification_factor):
            raise ValueError(f"Injected-beta scaling mismatch in {run_dir}/{scheme}.")
        manifest_payload_sha = fixed_batch.get(
            "payload_sha256", fixed_batch.get("batch_payload_sha256")
        )
        if manifest_payload_sha != scheme_rows[0]["batch_payload_sha256"]:
            raise ValueError(f"Manifest/CSV fixed-batch mismatch in {run_dir}/{scheme}.")

    required_layers = {*STATE_LAYER_BY_PARAMETER.values(), "__all__"}
    expected_contexts = {
        (scheme, precision) for scheme in schemes for precision in PRECISIONS
    }
    observed_contexts = {
        (row["scheme"], row["precision"]) for row in displacement_rows
    }
    if observed_contexts != expected_contexts:
        raise ValueError(f"Displacement precision coverage differs in {run_dir}.")
    for scheme, precision in expected_contexts:
        context = [
            row
            for row in displacement_rows
            if row["scheme"] == scheme and row["precision"] == precision
        ]
        phases = ("positive",) if variant == "positive_one_sided" else ("negative", "positive")
        for phase in phases:
            for reference in ("post_T_free", "matched_zero_K"):
                selected = [
                    row
                    for row in context
                    if row["phase"] == phase and row["reference_kind"] == reference
                ]
                if {row["state_layer_name"] for row in selected} != required_layers:
                    raise ValueError(
                        f"Displacement coverage differs in {run_dir}/{scheme}/{precision}/"
                        f"{phase}/{reference}."
                    )
                if any(row["outcome"] != "ok" for row in selected):
                    raise ValueError(f"Displacement outcome failed in {run_dir}.")
        if variant == "centered":
            span = [
                row
                for row in context
                if row["phase"] == "positive_minus_negative"
                and row["reference_kind"] == "negative_phase"
            ]
            if span and {row["state_layer_name"] for row in span} != required_layers:
                raise ValueError(f"Incomplete signed-span displacement in {run_dir}.")
            if any(row["outcome"] != "ok" for row in span):
                raise ValueError(f"Signed-span displacement outcome failed in {run_dir}.")

    return {
        "run_dir": run_dir,
        "run_id": run_dir.name,
        "source_kind": source_kind,
        "variant": variant,
        "result_sha256": reporting.sha256_file(run_dir / "result.json"),
        "manifest_sha256": reporting.sha256_file(run_dir / "manifest.json"),
        "metrics_sha256": reporting.sha256_file(run_dir / "metrics.jsonl"),
        "parameter_rows": parameter_rows,
        "displacement_rows": displacement_rows,
        "schemes": schemes,
        "case_identity_by_scheme": case_identity_by_scheme,
        "batch_source_indices_sha256": fixed_source_indices_sha,
    }


def discover_completed_bundles(
    root: Path, *, source_kind: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    bundles: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir() or run_dir.name == "analysis":
            continue
        manifest_path = run_dir / "manifest.json"
        status_path = run_dir / "status.json"
        if not manifest_path.is_file() and not status_path.is_file():
            continue
        if not manifest_path.is_file() or not status_path.is_file():
            raise ValueError(f"Partial canonical metadata in {run_dir}.")
        manifest = _read_json(manifest_path)
        status = _read_json(status_path)
        state = str(status.get("state", ""))
        if bool(manifest.get("smoke")):
            excluded.append(
                {
                    "source_kind": source_kind,
                    "run_id": run_dir.name,
                    "state": state,
                    "reason": "smoke_bundle",
                    "manifest_sha256": reporting.sha256_file(manifest_path),
                    "status_sha256": reporting.sha256_file(status_path),
                }
            )
            continue
        if state == "failed":
            excluded.append(
                {
                    "source_kind": source_kind,
                    "run_id": run_dir.name,
                    "state": state,
                    "reason": "failed_attempt_no_scientific_result",
                    "manifest_sha256": reporting.sha256_file(manifest_path),
                    "status_sha256": reporting.sha256_file(status_path),
                }
            )
            continue
        if state != "complete":
            raise ValueError(f"Unresolved non-smoke bundle {run_dir}: state={state!r}.")
        bundles.append(_load_bundle(run_dir, source_kind=source_kind))
    return bundles, excluded


def _displacement_index(
    rows: Sequence[Mapping[str, str]], *, scheme: str, precision: str
) -> dict[tuple[str, str, str], Mapping[str, str]]:
    index: dict[tuple[str, str, str], Mapping[str, str]] = {}
    for row in rows:
        if row["scheme"] != scheme or row["precision"] != precision:
            continue
        key = (row["phase"], row["reference_kind"], row["state_layer_name"])
        if key in index:
            raise ValueError(f"Duplicate displacement row for {scheme}/{precision}/{key}.")
        index[key] = row
    return index


def _displacement_values(
    index: Mapping[tuple[str, str, str], Mapping[str, str]],
    *,
    phase: str,
    reference: str,
    state_layer: str,
) -> dict[str, Any]:
    row = index.get((phase, reference, state_layer))
    if row is None:
        return {
            "relative": None,
            "l2": None,
            "transition_fraction": None,
        }
    return {
        "relative": float(row["relative_displacement"]),
        "l2": float(row["displacement_l2"]),
        "transition_fraction": float(row["active_set_transition_fraction"]),
    }


def normalize_bundle(bundle: Mapping[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    variant = str(bundle["variant"])
    for parameter_row in bundle["parameter_rows"]:
        parameter = parameter_row["parameter_name"]
        if parameter not in PARAMETERS:
            raise ValueError(f"Unexpected scored parameter {parameter!r}.")
        scheme = parameter_row["scheme"]
        state_layer = STATE_LAYER_BY_PARAMETER[parameter]
        for precision in PRECISIONS:
            displacement = _displacement_index(
                bundle["displacement_rows"], scheme=scheme, precision=precision
            )
            negative = _displacement_values(
                displacement,
                phase="negative",
                reference="matched_zero_K",
                state_layer=state_layer,
            )
            positive = _displacement_values(
                displacement,
                phase="positive",
                reference="matched_zero_K",
                state_layer=state_layer,
            )
            span = _displacement_values(
                displacement,
                phase="positive_minus_negative",
                reference="negative_phase",
                state_layer=state_layer,
            )
            if positive["relative"] is None:
                raise ValueError(
                    f"Missing positive matched-zero displacement in {bundle['run_id']}."
                )
            if variant == "centered" and negative["relative"] is None:
                raise ValueError(
                    f"Missing negative matched-zero displacement in {bundle['run_id']}."
                )
            status = _gradient_status(parameter_row, precision)
            output.append(
                {
                    "schema": SCHEMA,
                    "source_kind": bundle["source_kind"],
                    "run_id": bundle["run_id"],
                    "input_result_sha256": bundle["result_sha256"],
                    "eqprop_variant": variant,
                    "architecture": "conv3",
                    "scheme": scheme,
                    "precision": precision,
                    "checkpoint_role": parameter_row["checkpoint_role"],
                    "T": int(parameter_row["T"]),
                    "K": int(parameter_row["K"]),
                    "batch_index": int(parameter_row["batch_index"]),
                    "batch_payload_sha256": parameter_row["batch_payload_sha256"],
                    "batch_source_indices_sha256": bundle[
                        "batch_source_indices_sha256"
                    ],
                    **bundle["case_identity_by_scheme"][scheme],
                    "actual_base_beta": float(parameter_row["actual_beta"]),
                    "injected_beta": float(parameter_row["effective_beta"]),
                    "beta_source": parameter_row["beta_source"],
                    "parameter_name": parameter,
                    "layer": LAYER_LABELS[parameter],
                    "state_layer_name": state_layer,
                    "bias_excluded": True,
                    "gradient_status": status,
                    "eqprop_vs_bptt_cosine": _optional_float(
                        parameter_row[f"{precision}_eqprop_vs_bptt_cosine"]
                    ),
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
                    "matched_zero_negative_relative_displacement": negative["relative"],
                    "matched_zero_negative_displacement_l2": negative["l2"],
                    "matched_zero_negative_active_set_transition_fraction": negative[
                        "transition_fraction"
                    ],
                    "matched_zero_positive_relative_displacement": positive["relative"],
                    "matched_zero_positive_displacement_l2": positive["l2"],
                    "matched_zero_positive_active_set_transition_fraction": positive[
                        "transition_fraction"
                    ],
                    "signed_span_present": span["relative"] is not None,
                    "signed_span_relative_displacement": span["relative"],
                    "signed_span_displacement_l2": span["l2"],
                    "signed_span_active_set_transition_fraction": span[
                        "transition_fraction"
                    ],
                }
            )
    return output


def _infer_tk(
    primary_rows: Sequence[Mapping[str, Any]], override: Sequence[int] | None
) -> tuple[int, int]:
    if override is not None:
        if len(override) != 2 or any(int(value) <= 0 for value in override):
            raise ValueError("T/K override must contain two positive integers.")
        return int(override[0]), int(override[1])
    coordinates = {(int(row["T"]), int(row["K"])) for row in primary_rows}
    if len(coordinates) != 1:
        raise ValueError(
            "Primary production bundles must resolve to one T/K coordinate, or pass --tk T K; "
            f"observed={sorted(coordinates)}."
        )
    return next(iter(coordinates))


def _beta_set(rows: Sequence[Mapping[str, Any]]) -> set[float]:
    return {float(row["injected_beta"]) for row in rows}


def validate_scientific_coverage(rows: Sequence[Mapping[str, Any]]) -> None:
    primary = [row for row in rows if row["source_kind"] == "primary"]
    anchors = [row for row in rows if row["source_kind"] == "anchor"]
    if not primary:
        raise ValueError("No completed primary production rows were found.")
    if not anchors:
        raise ValueError("No matching prior one-sided anchor rows were found.")
    if len({row["batch_payload_sha256"] for row in rows}) != 1:
        raise ValueError("Primary and anchor bundles do not share the exact fixed minibatch.")
    if len({row["batch_source_indices_sha256"] for row in rows}) != 1:
        raise ValueError("Primary and anchor bundles do not share fixed source indices.")
    for scheme in SCHEMES:
        scheme_primary = [row for row in primary if row["scheme"] == scheme]
        if {row["eqprop_variant"] for row in scheme_primary} != set(VARIANTS):
            raise ValueError(f"Primary {scheme} coverage must contain both EqProp variants.")
        for precision in PRECISIONS:
            for variant in VARIANTS:
                selected = [
                    row
                    for row in scheme_primary
                    if row["precision"] == precision
                    and row["eqprop_variant"] == variant
                ]
                # Multiple beta points are allowed, so validate every beta
                # independently rather than only checking the union of names.
                betas = _beta_set(selected)
                if not betas:
                    raise ValueError(
                        f"Missing primary {scheme}/{precision}/{variant} coverage."
                    )
                for beta in betas:
                    names = {
                        row["parameter_name"]
                        for row in selected
                        if _close(float(row["injected_beta"]), beta)
                    }
                    if names != set(PARAMETERS):
                        raise ValueError(
                            f"Incomplete layers for {scheme}/{precision}/{variant}/beta={beta}."
                        )
        primary_one_sided_betas = _beta_set(
            row
            for row in scheme_primary
            if row["precision"] == "float64"
            and row["eqprop_variant"] == "positive_one_sided"
        )
        primary_centered_betas = _beta_set(
            row
            for row in scheme_primary
            if row["precision"] == "float64" and row["eqprop_variant"] == "centered"
        )
        if not primary_one_sided_betas.issubset(primary_centered_betas):
            raise ValueError(
                f"Every new one-sided beta must also have a centered primary replay for {scheme}: "
                f"one-sided={sorted(primary_one_sided_betas)}, "
                f"centered={sorted(primary_centered_betas)}."
            )
        scheme_anchors = [row for row in anchors if row["scheme"] == scheme]
        if not scheme_anchors or any(
            row["eqprop_variant"] != "positive_one_sided" for row in scheme_anchors
        ):
            raise ValueError(f"Missing clean one-sided anchors for {scheme}.")
        all_one_sided_betas = primary_one_sided_betas | _beta_set(
            row
            for row in scheme_anchors
            if row["precision"] == "float64"
        )
        if not primary_centered_betas.issubset(all_one_sided_betas):
            raise ValueError(
                f"Centered betas lack matched one-sided primary/anchor evidence for {scheme}: "
                f"centered={sorted(primary_centered_betas)}, "
                f"available one-sided={sorted(all_one_sided_betas)}."
            )
        for precision in PRECISIONS:
            for parameter in PARAMETERS:
                bptt_values = [
                    float(row["bptt_l2"])
                    for row in rows
                    if row["scheme"] == scheme
                    and row["precision"] == precision
                    and row["parameter_name"] == parameter
                ]
                if not bptt_values or any(
                    not math.isclose(
                        value,
                        bptt_values[0],
                        rel_tol=1.0e-6,
                        abs_tol=1.0e-12,
                    )
                    for value in bptt_values[1:]
                ):
                    raise ValueError(
                        f"BPTT reference changed across variants/betas for "
                        f"{scheme}/{precision}/{parameter}."
                    )
        source_selection_shas = {
            row["source_selection_row_sha256"]
            for row in rows
            if row["scheme"] == scheme
        }
        checkpoint_shas = {
            row["best_checkpoint_sha256"]
            for row in rows
            if row["scheme"] == scheme
        }
        if len(source_selection_shas) != 1 or len(checkpoint_shas) != 1:
            raise ValueError(
                f"Source checkpoint identity changed across primary/anchor rows for {scheme}."
            )


def _scientific_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row["eqprop_variant"],
        row["scheme"],
        row["precision"],
        int(row["T"]),
        int(row["K"]),
        format(float(row["injected_beta"]), ".15g"),
        row["parameter_name"],
    )


def merge_rows(
    primary_rows: Sequence[Mapping[str, Any]],
    anchor_rows: Sequence[Mapping[str, Any]],
    *,
    tk: tuple[int, int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected_primary = [
        dict(row)
        for row in primary_rows
        if (int(row["T"]), int(row["K"])) == tk
    ]
    if not selected_primary:
        raise ValueError(f"No primary rows match T/K={tk[0]}/{tk[1]}.")
    missing_schemes = [
        scheme
        for scheme in SCHEMES
        if not any(row["scheme"] == scheme for row in selected_primary)
    ]
    if missing_schemes:
        raise ValueError(f"Primary rows at T/K={tk[0]}/{tk[1]} miss {missing_schemes}.")
    maximum_primary_beta = {
        scheme: max(
            float(row["injected_beta"])
            for row in selected_primary
            if row["scheme"] == scheme
        )
        for scheme in SCHEMES
    }
    selected_anchors = [
        dict(row)
        for row in anchor_rows
        if row["eqprop_variant"] == "positive_one_sided"
        and (int(row["T"]), int(row["K"])) == tk
        and float(row["injected_beta"]) <= maximum_primary_beta[row["scheme"]]
    ]
    output: dict[tuple[Any, ...], dict[str, Any]] = {}
    duplicate_anchors: list[dict[str, Any]] = []
    for row in selected_primary:
        key = _scientific_key(row)
        if key in output:
            raise ValueError(f"Duplicate primary scientific context: {key}.")
        output[key] = row
    primary_keys = set(output)
    for row in selected_anchors:
        key = _scientific_key(row)
        if key in primary_keys:
            duplicate_anchors.append(
                {"run_id": row["run_id"], "reason": "superseded_by_primary", "key": key}
            )
            continue
        if key in output:
            raise ValueError(f"Duplicate anchor scientific context: {key}.")
        output[key] = row
    merged = sorted(
        output.values(),
        key=lambda row: (
            SCHEMES.index(row["scheme"]),
            PRECISIONS.index(row["precision"]),
            VARIANTS.index(row["eqprop_variant"]),
            float(row["injected_beta"]),
            PARAMETERS.index(row["parameter_name"]),
        ),
    )
    validate_scientific_coverage(merged)
    return merged, duplicate_anchors


def write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _contexts(rows: Sequence[Mapping[str, Any]]) -> list[list[Mapping[str, Any]]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (
            row["source_kind"],
            row["run_id"],
            row["eqprop_variant"],
            row["scheme"],
            row["precision"],
            row["T"],
            row["K"],
            row["injected_beta"],
        )
        groups.setdefault(key, []).append(row)
    contexts = list(groups.values())
    for group in contexts:
        if {row["parameter_name"] for row in group} != set(PARAMETERS):
            raise ValueError("A report context lacks complete layer coverage.")
    return sorted(
        contexts,
        key=lambda group: (
            SCHEMES.index(group[0]["scheme"]),
            PRECISIONS.index(group[0]["precision"]),
            VARIANTS.index(group[0]["eqprop_variant"]),
            float(group[0]["injected_beta"]),
        ),
    )


def _display_gradient(row: Mapping[str, Any]) -> str:
    if row["gradient_status"] == "exact_zero":
        return "ZERO"
    if row["gradient_status"] == "undefined":
        return "undef"
    return f"{float(row['eqprop_vs_bptt_cosine']):.6f}"


def _display_displacement(row: Mapping[str, Any]) -> str:
    negative = row["matched_zero_negative_relative_displacement"]
    positive = row["matched_zero_positive_relative_displacement"]
    span = row["signed_span_relative_displacement"]
    return "/".join(
        (
            "—" if negative is None else f"{float(negative):.4g}",
            f"{float(positive):.4g}",
            "—" if span is None else f"{float(span):.4g}",
        )
    )


def write_markdown(
    rows: Sequence[Mapping[str, Any]],
    *,
    bundles: Sequence[Mapping[str, Any]],
    excluded: Sequence[Mapping[str, Any]],
    tk: tuple[int, int],
    path: Path,
) -> None:
    lines = [
        "# Conv3 one-sided versus two-sided higher-beta EqProp",
        "",
        f"All scientific rows use the best-validation Conv3 checkpoints, fixed batch 0, and T/K `{tk[0]}/{tk[1]}`. Betas below are injected-current magnitudes.",
        "",
        "## Input provenance",
        "",
        "| source | run | variant | schemes | result SHA-256 |",
        "|---|---|---|---|---|",
    ]
    for bundle in bundles:
        lines.append(
            f"| {bundle['source_kind']} | `{bundle['run_id']}` | {bundle['variant']} | "
            f"{', '.join(bundle['schemes'])} | `{bundle['result_sha256']}` |"
        )
    if excluded:
        lines.extend(["", "Excluded attempts/smokes:", ""])
        for item in excluded:
            lines.append(
                f"- `{item['source_kind']}:{item['run_id']}` — {item['reason']} "
                f"(state `{item['state']}`)."
            )
    lines.extend(
        [
            "",
            "## Layerwise EqProp versus BPTT cosine",
            "",
            "`ZERO` means the EqProp gradient is exactly zero, so cosine is undefined.",
            "",
            "| source | scheme | precision | variant | base beta | injected beta | C0 | C1 | C2 | D | residual-limited |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|:---:|",
        ]
    )
    for group in _contexts(rows):
        anchor = group[0]
        by_parameter = {row["parameter_name"]: row for row in group}
        lines.append(
            "| "
            + " | ".join(
                [
                    str(anchor["source_kind"]),
                    str(anchor["scheme"]),
                    str(anchor["precision"]),
                    str(anchor["eqprop_variant"]),
                    f"{float(anchor['actual_base_beta']):.6g}",
                    f"{float(anchor['injected_beta']):.6g}",
                    *[_display_gradient(by_parameter[name]) for name in PARAMETERS],
                    "yes" if any(row["residual_limited"] for row in group) else "no",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Layer displacement",
            "",
            "Each cell is `negative / positive / signed-span`. The first two values are relative to the matched zero-K state. Signed-span is the reported positive-minus-negative endpoint displacement relative to the negative endpoint; `—` means that phase does not exist for the estimator or the optional span was not emitted.",
            "",
            "| source | scheme | precision | variant | base beta | injected beta | H0 | H1 | H2 | output |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for group in _contexts(rows):
        anchor = group[0]
        by_parameter = {row["parameter_name"]: row for row in group}
        lines.append(
            "| "
            + " | ".join(
                [
                    str(anchor["source_kind"]),
                    str(anchor["scheme"]),
                    str(anchor["precision"]),
                    str(anchor["eqprop_variant"]),
                    f"{float(anchor['actual_base_beta']):.6g}",
                    f"{float(anchor['injected_beta']):.6g}",
                    *[_display_displacement(by_parameter[name]) for name in PARAMETERS],
                ]
            )
            + " |"
        )
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {value}" for value in LIMITATIONS)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_plot(rows: Sequence[Mapping[str, Any]], path: Path, *, tk: tuple[int, int]) -> None:
    figure, axes = plt.subplots(4, 2, figsize=(14.5, 15.5), sharex="col", squeeze=False)
    layer_colors = {
        "ConvWeight_0": "#0072B2",
        "ConvWeight_1": "#E69F00",
        "ConvWeight_2": "#009E73",
        "DenseWeight_0": "#CC79A7",
    }
    variant_styles = {
        "positive_one_sided": {"linestyle": "-", "marker": "o"},
        "centered": {"linestyle": "--", "marker": "s"},
    }
    displacement_styles = {
        "one-sided +": ("-", "o"),
        "centered -": (":", "v"),
        "centered +": ("--", "^"),
        "centered span": ("-.", "x"),
    }
    zero_rail = -1.08
    beta_values = [float(row["injected_beta"]) for row in rows]
    beta_min = min(beta_values) / 1.5
    beta_max = max(beta_values) * 1.5

    for scheme_index, scheme in enumerate(SCHEMES):
        cosine_row = scheme_index * 2
        displacement_row = cosine_row + 1
        for precision_index, precision in enumerate(PRECISIONS):
            cosine_axis = axes[cosine_row, precision_index]
            displacement_axis = axes[displacement_row, precision_index]
            selected = [
                row
                for row in rows
                if row["scheme"] == scheme and row["precision"] == precision
            ]
            for parameter in PARAMETERS:
                color = layer_colors[parameter]
                for variant in VARIANTS:
                    series = sorted(
                        [
                            row
                            for row in selected
                            if row["parameter_name"] == parameter
                            and row["eqprop_variant"] == variant
                        ],
                        key=lambda row: float(row["injected_beta"]),
                    )
                    numeric = [row for row in series if row["gradient_status"] == "numeric"]
                    if numeric:
                        cosine_axis.plot(
                            [row["injected_beta"] for row in numeric],
                            [row["eqprop_vs_bptt_cosine"] for row in numeric],
                            color=color,
                            linewidth=1.6,
                            markersize=4.8,
                            **variant_styles[variant],
                        )
                    nonnumeric = [row for row in series if row["gradient_status"] != "numeric"]
                    if nonnumeric:
                        cosine_axis.scatter(
                            [row["injected_beta"] for row in nonnumeric],
                            [zero_rail] * len(nonnumeric),
                            color=color,
                            marker="X" if all(row["gradient_status"] == "exact_zero" for row in nonnumeric) else "D",
                            s=42,
                            zorder=5,
                        )

                one_sided = sorted(
                    [
                        row
                        for row in selected
                        if row["parameter_name"] == parameter
                        and row["eqprop_variant"] == "positive_one_sided"
                    ],
                    key=lambda row: float(row["injected_beta"]),
                )
                centered = sorted(
                    [
                        row
                        for row in selected
                        if row["parameter_name"] == parameter
                        and row["eqprop_variant"] == "centered"
                    ],
                    key=lambda row: float(row["injected_beta"]),
                )
                phase_series = (
                    (
                        "one-sided +",
                        one_sided,
                        "matched_zero_positive_relative_displacement",
                    ),
                    (
                        "centered -",
                        centered,
                        "matched_zero_negative_relative_displacement",
                    ),
                    (
                        "centered +",
                        centered,
                        "matched_zero_positive_relative_displacement",
                    ),
                    ("centered span", centered, "signed_span_relative_displacement"),
                )
                for label, series, field in phase_series:
                    values = [row for row in series if row[field] is not None]
                    if not values:
                        continue
                    linestyle, marker = displacement_styles[label]
                    displacement_axis.plot(
                        [row["injected_beta"] for row in values],
                        [row[field] for row in values],
                        color=color,
                        linestyle=linestyle,
                        marker=marker,
                        linewidth=1.45,
                        markersize=4.5,
                    )

            cosine_axis.axhline(0.99, color="#555555", linestyle=":", linewidth=1.0)
            cosine_axis.axhline(zero_rail, color="#777777", linestyle=":", linewidth=0.8)
            cosine_axis.set_ylim(-1.12, 1.035)
            cosine_axis.set_yticks((zero_rail, -1.0, -0.5, 0.0, 0.5, 0.9, 0.99, 1.0))
            cosine_axis.set_yticklabels(
                ("ZERO", "-1", "-0.5", "0", "0.5", "0.9", "0.99", "1")
            )
            displacement_axis.set_yscale("symlog", linthresh=1.0e-14, linscale=0.6)
            displacement_axis.set_ylim(bottom=0.0)
            for axis in (cosine_axis, displacement_axis):
                axis.set_xscale("log")
                axis.set_xlim(beta_min, beta_max)
                axis.grid(True, which="both", alpha=0.18)
            cosine_axis.set_title(f"{scheme} — {precision}")
            cosine_axis.set_ylabel("EqProp–BPTT cosine")
            displacement_axis.set_ylabel("relative state displacement\n(phase-specific reference)")
            displacement_axis.set_xlabel("injected beta magnitude")

    layer_handles = [
        Line2D([0], [0], color=layer_colors[name], linewidth=2.0, label=LAYER_LABELS[name])
        for name in PARAMETERS
    ]
    estimator_handles = [
        Line2D(
            [0],
            [0],
            color="#333333",
            linewidth=1.6,
            label=("one-sided cosine" if variant == "positive_one_sided" else "centered cosine"),
            **style,
        )
        for variant, style in variant_styles.items()
    ]
    phase_handles = [
        Line2D(
            [0],
            [0],
            color="#333333",
            linestyle=style[0],
            marker=style[1],
            linewidth=1.45,
            label=label,
        )
        for label, style in displacement_styles.items()
    ]
    figure.suptitle(
        "Conv3 best-checkpoint current-nudge EqProp: one-sided versus centered\n"
        f"true float32/float64, T/K={tk[0]}/{tk[1]}, weights only",
        fontsize=15,
        y=0.995,
    )
    handles = [*layer_handles, *estimator_handles, *phase_handles]
    figure.legend(
        handles,
        [handle.get_label() for handle in handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=5,
        frameon=False,
    )
    figure.tight_layout(rect=(0.02, 0.02, 1.0, 0.90), h_pad=1.4)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    bundles: Sequence[Mapping[str, Any]],
    excluded: Sequence[Mapping[str, Any]],
    duplicate_anchors: Sequence[Mapping[str, Any]],
    tk: tuple[int, int],
    output_csv: Path,
    output_markdown: Path,
    output_plot: Path,
    path: Path,
) -> None:
    summary = {
        "schema": SCHEMA,
        "dataset": "ordinary_mnist",
        "evidence_class": "ordinary_mnist_learning_algorithm_gradient_diagnostic",
        "architecture": "conv3",
        "checkpoint_role": "best_validation",
        "T": tk[0],
        "K": tk[1],
        "schemes": list(SCHEMES),
        "precisions": list(PRECISIONS),
        "eqprop_variants": list(VARIANTS),
        "fixed_batch": {
            "batch_index": 0,
            "payload_sha256": _single(
                (row["batch_payload_sha256"] for row in rows),
                label="fixed batch payload SHA",
            ),
            "source_indices_sha256": _single(
                (row["batch_source_indices_sha256"] for row in rows),
                label="fixed source-index SHA",
            ),
        },
        "bias_gradients_excluded": True,
        "read_only": True,
        "row_count": len(rows),
        "input_bundles": [
            {
                "source_kind": bundle["source_kind"],
                "run_id": bundle["run_id"],
                "path": str(bundle["run_dir"]),
                "eqprop_variant": bundle["variant"],
                "schemes": bundle["schemes"],
                "result_sha256": bundle["result_sha256"],
                "manifest_sha256": bundle["manifest_sha256"],
                "metrics_sha256": bundle["metrics_sha256"],
            }
            for bundle in bundles
        ],
        "excluded": list(excluded),
        "duplicate_anchors": list(duplicate_anchors),
        "primary_injected_betas": {
            scheme: sorted(
                {
                    float(row["injected_beta"])
                    for row in rows
                    if row["source_kind"] == "primary" and row["scheme"] == scheme
                }
            )
            for scheme in SCHEMES
        },
        "anchor_injected_betas": {
            scheme: sorted(
                {
                    float(row["injected_beta"])
                    for row in rows
                    if row["source_kind"] == "anchor" and row["scheme"] == scheme
                }
            )
            for scheme in SCHEMES
        },
        "centered_signed_span_complete": all(
            row["signed_span_present"]
            for row in rows
            if row["eqprop_variant"] == "centered"
        ),
        "limitations": list(LIMITATIONS),
        "outputs": {
            "csv": {"path": str(output_csv), "sha256": reporting.sha256_file(output_csv)},
            "markdown": {
                "path": str(output_markdown),
                "sha256": reporting.sha256_file(output_markdown),
            },
            "plot": {"path": str(output_plot), "sha256": reporting.sha256_file(output_plot)},
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_analysis(
    *,
    study_root: Path,
    anchor_root: Path,
    analysis_dir: Path,
    tk_override: Sequence[int] | None = None,
) -> dict[str, Any]:
    primary_bundles, primary_excluded = discover_completed_bundles(
        study_root, source_kind="primary"
    )
    anchor_bundles, anchor_excluded = discover_completed_bundles(
        anchor_root, source_kind="anchor"
    )
    if not primary_bundles:
        raise ValueError("No completed non-smoke primary production bundles exist yet.")
    primary_rows = [row for bundle in primary_bundles for row in normalize_bundle(bundle)]
    anchor_rows = [row for bundle in anchor_bundles for row in normalize_bundle(bundle)]
    tk = _infer_tk(primary_rows, tk_override)
    rows, duplicate_anchors = merge_rows(primary_rows, anchor_rows, tk=tk)

    used_ids = {row["run_id"] for row in rows}
    used_bundles = [
        bundle
        for bundle in [*primary_bundles, *anchor_bundles]
        if bundle["run_id"] in used_ids
    ]
    analysis_dir = analysis_dir.expanduser().resolve()
    output_csv = analysis_dir / "conv3_eqprop_one_vs_two_sided_higher_beta.csv"
    output_markdown = analysis_dir / "conv3_eqprop_one_vs_two_sided_higher_beta.md"
    output_plot = analysis_dir / "conv3_eqprop_one_vs_two_sided_higher_beta.png"
    output_summary = analysis_dir / "conv3_eqprop_one_vs_two_sided_higher_beta.json"
    write_csv(rows, output_csv)
    write_markdown(
        rows,
        bundles=used_bundles,
        excluded=[*primary_excluded, *anchor_excluded],
        tk=tk,
        path=output_markdown,
    )
    render_plot(rows, output_plot, tk=tk)
    write_summary(
        rows,
        bundles=used_bundles,
        excluded=[*primary_excluded, *anchor_excluded],
        duplicate_anchors=duplicate_anchors,
        tk=tk,
        output_csv=output_csv,
        output_markdown=output_markdown,
        output_plot=output_plot,
        path=output_summary,
    )
    return {
        "row_count": len(rows),
        "bundle_count": len(used_bundles),
        "T": tk[0],
        "K": tk[1],
        "output_csv": str(output_csv),
        "output_markdown": str(output_markdown),
        "output_plot": str(output_plot),
        "output_summary": str(output_summary),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=DEFAULT_STUDY_ROOT)
    parser.add_argument("--anchor-root", type=Path, default=DEFAULT_ANCHOR_ROOT)
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        help="Defaults to <study-root>/analysis.",
    )
    parser.add_argument(
        "--tk",
        nargs=2,
        type=int,
        metavar=("T", "K"),
        help="Select one positive T/K coordinate; otherwise infer a unique primary coordinate.",
    )
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
        anchor_root=args.anchor_root,
        analysis_dir=analysis_dir,
        tk_override=args.tk,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
