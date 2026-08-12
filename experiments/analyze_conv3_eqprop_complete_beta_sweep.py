#!/usr/bin/env python3
"""Validate and summarize the complete Conv3 current-nudge EqProp beta sweep.

This analyzer is intentionally fail closed.  It accepts exactly the 22
non-smoke bundles declared by the study plan (11 equal injected-current beta
values times the positive one-sided and centered estimators), validates their
canonical reporting records and replay guards, and joins each weight-layer
EqProp/BPTT comparison to the corresponding matched-zero state displacement.

The primary beta coordinate is the current actually injected at the output,
``effective_beta``.  The scheme-dependent CLI/base beta is retained only as
provenance.  Empty cosines caused by an exact-zero vector remain undefined;
they are never converted to zero.
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
from matplotlib import pyplot as plt  # noqa: E402


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments import reporting  # noqa: E402


DEFAULT_CONFIG = (
    REPOSITORY_ROOT
    / "configs/conv/perfectdiode_conv3_eqprop_beta_1em3_to_1e2_init_best_true_dtype_seed0_20260811_v1.json"
)
DEFAULT_LAUNCHER = (
    REPOSITORY_ROOT / "experiments/run_conv3_eqprop_complete_beta_sweep_local.sh"
)
SCHEMA = "perfectdiode-conv3-current-eqprop-complete-beta-analysis/v1"
BUNDLE_PARAMETER_SCHEMA = "perfectdiode-conv-current-eqprop-true-dtype-audit/v3"
RESIDUAL_SCHEMA = "perfectdiode-conv-eqprop-bptt-beta-tk-displacement/v1"
SCHEMES = ("baseline", "ours", "legacy")
CHECKPOINT_ROLES = ("reconstructed_initialization", "best_validation")
VARIANTS = ("positive_one_sided", "centered")
PRECISIONS = ("float32", "float64")
PARAMETERS = ("ConvWeight_0", "ConvWeight_1", "ConvWeight_2", "DenseWeight_0")
PARAMETER_LABELS = {
    "ConvWeight_0": "C0",
    "ConvWeight_1": "C1",
    "ConvWeight_2": "C2",
    "DenseWeight_0": "Dense",
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
ALL_STATE_NAMES = (*STATE_LABELS, "__all__")
BETA_LABELS = (
    "1em3",
    "3em3",
    "1em2",
    "3em2",
    "1em1",
    "3em1",
    "1",
    "3",
    "10",
    "30",
    "100",
)
EXPECTED_BETAS = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
EXPECTED_TK = (64, 64)
EXPECTED_BATCH_INDEX = 0
EXPECTED_BATCH_SIZE = 16
EXPECTED_PRODUCTION_BUNDLES = 22
EXPECTED_LOGICAL_CONTEXTS = 132
EXPECTED_DTYPE_CONTEXTS = 264
EXPECTED_SUMMARY_ROWS = 1056
EXPECTED_RESIDUAL_ROWS = 3696
EXPECTED_ONE_SIDED_RESIDUAL_ROWS = 144
EXPECTED_CENTERED_RESIDUAL_ROWS = 192
RESIDUAL_LAYER_CONTRACT = {
    "Layer_1": {
        "layer_index": 0,
        "state_layer_type": "ConvLayer",
        "layer_role": "hidden_0",
        "residual_mode": "projected_kkt",
    },
    "Layer_2": {
        "layer_index": 1,
        "state_layer_type": "ConvLayer",
        "layer_role": "hidden_1",
        "residual_mode": "projected_kkt",
    },
    "Layer_3": {
        "layer_index": 2,
        "state_layer_type": "ConvLayer",
        "layer_role": "hidden_2",
        "residual_mode": "projected_kkt",
    },
    "Layer_4": {
        "layer_index": 3,
        "state_layer_type": "LinearLayer",
        "layer_role": "output",
        "residual_mode": "raw",
    },
}
TRUE_COMPLETION_GUARDS = (
    "criteria_met",
    "completed_all_selected_cases",
    "all_weight_layer_precision_comparisons_present",
    "scientific_precision_gate_outcome_recorded",
    "all_runtime_dtypes_proven",
    "all_iteration_contracts_passed",
    "all_finite_differences_computed_in_native_dtype",
    "all_bptt_gradients_computed_in_native_dtype",
    "all_scored_bptt_weight_tensors_archived",
    "all_eqprop_vs_bptt_layer_metrics_present",
    "bptt_and_configured_eqprop_phases_started_from_identical_post_T",
    "amplification_scaled_beta_metadata_recorded",
    "exact_residuals_recorded_for_both_precisions",
    "displacement_recorded_for_both_precisions",
    "phase_diagnostics_recorded_for_both_precisions",
    "fixed_cohort_and_batch_reproduced",
    "source_bundle_files_unchanged",
    "source_checkpoint_bytes_unchanged",
    "parameter_tensors_unchanged",
    "bias_gradients_excluded",
)
TRUE_READ_ONLY_GUARDS = (
    "all_dtype_proofs_passed",
    "all_iteration_contracts_passed",
    "all_parameter_and_arithmetic_guards_passed",
    "bias_gradients_excluded",
    "biases_active_in_dynamics",
    "cohort_exact",
    "source_bundle_files_unchanged",
    "source_checkpoint_bytes_unchanged",
    "source_config_semantically_exact",
)
SUMMARY_FIELDS = (
    "schema",
    "plan_study_id",
    "bundle_manifest_study_id",
    "run_id",
    "input_result_sha256",
    "eqprop_variant",
    "scheme",
    "checkpoint_role",
    "precision",
    "T",
    "K",
    "batch_index",
    "batch_payload_sha256",
    "batch_source_indices_sha256",
    "beta_index",
    "actual_base_beta",
    "injected_beta",
    "amplification_factor",
    "parameter_name",
    "gradient_layer",
    "state_layer_name",
    "state_layer",
    "bias_excluded",
    "residual_limited",
    "gradient_outcome",
    "eqprop_gradient_dtype",
    "bptt_gradient_dtype",
    "eqprop_l2",
    "bptt_l2",
    "eqprop_exact_zero_fraction",
    "bptt_exact_zero_fraction",
    "eqprop_exact_zero",
    "bptt_exact_zero",
    "cosine_defined",
    "cosine_undefined_reason",
    "eqprop_vs_bptt_cosine",
    "eqprop_over_bptt_norm_ratio",
    "eqprop_vs_bptt_symmetric_norm_delta",
    "matched_zero_positive_relative_displacement",
    "matched_zero_positive_delta_rms",
    "matched_zero_negative_relative_displacement",
    "matched_zero_negative_delta_rms",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing CSV header in {path}.")
        return [dict(row) for row in reader]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in SUMMARY_FIELDS})


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _parse_bool(value: Any, *, label: str) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0"}:
        return False
    raise ValueError(f"Invalid boolean for {label}: {value!r}.")


def _finite_float(value: Any, *, label: str) -> float:
    if value is None or str(value).strip() == "":
        raise ValueError(f"Missing numeric value for {label}.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Non-finite numeric value for {label}: {value!r}.")
    return number


def _optional_float(value: Any, *, label: str) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    return _finite_float(value, label=label)


def _close(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1.0e-10, abs_tol=1.0e-14)


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return path.resolve()


def _require_columns(
    rows: Sequence[Mapping[str, str]], required: Iterable[str], *, path: Path
) -> None:
    if not rows:
        raise ValueError(f"No rows in {path}.")
    missing = set(required) - set(rows[0])
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}.")


def _beta_index(value: float, expected: Sequence[float]) -> int:
    matches = [index for index, beta in enumerate(expected) if _close(value, beta)]
    if len(matches) != 1:
        raise ValueError(f"Injected beta {value!r} is outside the exact study grid.")
    return matches[0]


def _expected_run_id(variant: str, beta_index: int) -> str:
    estimator = "one-sided" if variant == "positive_one_sided" else "centered"
    return f"{estimator}-B{BETA_LABELS[beta_index]}-tk64"


def _validate_plan(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != "perfectdiode-conv3-current-eqprop-complete-beta-sweep/v1":
        raise ValueError("Unexpected complete-beta study schema.")
    source = config.get("source", {})
    replay = config.get("replay", {})
    beta = config.get("beta_contract", {})
    completion = config.get("completion", {})
    if tuple(source.get("schemes", ())) != SCHEMES:
        raise ValueError("Study plan must declare baseline/ours/legacy in canonical order.")
    if tuple(source.get("checkpoint_roles", ())) != CHECKPOINT_ROLES:
        raise ValueError("Study plan must declare initialization and best checkpoint roles.")
    if tuple(replay.get("precisions", ())) != PRECISIONS:
        raise ValueError("Study plan must declare float32 and float64.")
    if tuple(replay.get("eqprop_variants", ())) != VARIANTS:
        raise ValueError("Study plan must declare one-sided and centered estimators.")
    if (int(replay.get("T", -1)), int(replay.get("K", -1))) != EXPECTED_TK:
        raise ValueError("Study plan must use T=K=64.")
    if list(replay.get("batch_indices", ())) != [EXPECTED_BATCH_INDEX]:
        raise ValueError("Study plan must use only fixed batch index 0.")
    if int(replay.get("batch_size", -1)) != EXPECTED_BATCH_SIZE:
        raise ValueError("Study plan must use 16 examples.")
    observed_betas = tuple(float(value) for value in beta.get("injected_betas", ()))
    if len(observed_betas) != len(EXPECTED_BETAS) or any(
        not _close(left, right) for left, right in zip(observed_betas, EXPECTED_BETAS)
    ):
        raise ValueError("Study plan injected-beta grid differs from the frozen 11 values.")
    divisors = beta.get("base_beta_divisor", {})
    if any(not _close(float(divisors.get(name, -1.0)), expected) for name, expected in {
        "baseline": 1.0,
        "ours": 64.0,
        "legacy": 4096.0,
    }.items()):
        raise ValueError("Study plan amplification/base-beta divisors are invalid.")
    expected_counts = {
        "expected_run_bundles": EXPECTED_PRODUCTION_BUNDLES,
        "expected_scheme_checkpoint_beta_variant_contexts": EXPECTED_LOGICAL_CONTEXTS,
        "expected_true_dtype_contexts": EXPECTED_DTYPE_CONTEXTS,
    }
    for key, expected in expected_counts.items():
        if int(completion.get(key, -1)) != expected:
            raise ValueError(f"Study plan completion count {key} differs from {expected}.")


def _validate_source_identity(
    *,
    manifest: Mapping[str, Any],
    source_config: Mapping[str, Any],
    source_config_path: Path,
    source_run: Path,
    plan: Mapping[str, Any],
) -> str:
    expected_bundle_study_id = f"{source_config['study_id']}--float64-shadow"
    if manifest.get("study_id") != expected_bundle_study_id:
        raise ValueError(
            f"Bundle {manifest.get('run_id')} has unexpected inherited study_id "
            f"{manifest.get('study_id')!r}."
        )
    configuration = manifest.get("configuration", {})
    if configuration.get("sha256") != reporting.sha256_file(source_config_path):
        raise ValueError(f"Source config hash mismatch in {manifest.get('run_id')}.")
    resolved = configuration.get("resolved", {})
    if resolved.get("source_beta_config") != source_config:
        raise ValueError(f"Source config semantics differ in {manifest.get('run_id')}.")
    source_record = manifest.get("source_native_beta_run", {})
    expected_names = (
        "result.json",
        "manifest.json",
        "config.resolved.json",
        "cohort.json",
        "context_beta_selection.csv",
        "parameter_summary.csv",
    )
    expected_files = {
        name: reporting.sha256_file(source_run / name)
        for name in expected_names
        if (source_run / name).is_file()
    }
    if source_record.get("source_files") != expected_files:
        raise ValueError(f"Source bundle hashes differ in {manifest.get('run_id')}.")
    if source_record.get("result_sha256") != expected_files.get("result.json"):
        raise ValueError(f"Source result hash differs in {manifest.get('run_id')}.")
    runtime_source = manifest.get("runtime", {}).get("runtime_source", {})
    if runtime_source.get("head") != plan["source"]["runtime_source_head"]:
        raise ValueError(f"Exact runtime source HEAD differs in {manifest.get('run_id')}.")
    return expected_bundle_study_id


def _validate_completion(result: Mapping[str, Any], *, run_dir: Path) -> None:
    completion = result.get("completion", {})
    for key in TRUE_COMPLETION_GUARDS:
        if completion.get(key) is not True:
            raise ValueError(f"Completion guard {key} failed in {run_dir}.")
    if completion.get("official_test_read") is not False:
        raise ValueError(f"Official test read in {run_dir}.")
    if completion.get("optimizer_steps_applied") is not False:
        raise ValueError(f"Optimizer step recorded in {run_dir}.")


def _validate_read_only_guards(guards: Mapping[str, Any], *, run_dir: Path) -> None:
    for key in TRUE_READ_ONLY_GUARDS:
        if guards.get(key) is not True:
            raise ValueError(f"Read-only guard {key} failed in {run_dir}.")
    if guards.get("official_test_read") is not False:
        raise ValueError(f"Read-only guard reports official test access in {run_dir}.")
    if guards.get("optimizer_constructed") is not False:
        raise ValueError(f"Read-only replay constructed an optimizer in {run_dir}.")
    if guards.get("optimizer_steps_applied") is not False:
        raise ValueError(f"Read-only replay applied an optimizer step in {run_dir}.")
    if guards.get("source_bundle_validation_errors") not in ([], None):
        raise ValueError(f"Source bundle validation failed in {run_dir}.")
    replay = guards.get("bptt_replay", {})
    if replay.get("all_bptt_and_configured_eqprop_phases_started_from_common_post_T") is not True:
        raise ValueError(f"BPTT/EqProp common-state guard failed in {run_dir}.")


def _validate_cohort(
    *, run_dir: Path, plan: Mapping[str, Any], source_cohort_path: Path
) -> tuple[str, str]:
    source_copy = run_dir / "source_cohort.json"
    materialized = run_dir / "materialized_cohort.json"
    expected_sha = reporting.sha256_file(source_cohort_path)
    if reporting.sha256_file(source_copy) != expected_sha:
        raise ValueError(f"Source cohort copy differs in {run_dir}.")
    if reporting.sha256_file(materialized) != expected_sha:
        raise ValueError(f"Materialized cohort differs in {run_dir}.")
    cohort = _read_json(materialized)
    replay = plan["replay"]
    if int(cohort.get("batch_size", -1)) != EXPECTED_BATCH_SIZE:
        raise ValueError(f"Cohort batch size differs in {run_dir}.")
    batches = cohort.get("batches", [])
    selected = [row for row in batches if int(row.get("batch_index", -1)) == 0]
    if len(selected) != 1:
        raise ValueError(f"Cohort lacks unique batch 0 in {run_dir}.")
    batch = selected[0]
    payload = str(batch.get("payload_sha256", ""))
    indices = str(batch.get("source_indices_sha256", ""))
    if payload != replay["batch_payload_sha256"]:
        raise ValueError(f"Fixed batch payload differs in {run_dir}.")
    if indices != replay["batch_source_indices_sha256"]:
        raise ValueError(f"Fixed batch source indices differ in {run_dir}.")
    if cohort.get("official_test_read") is not False:
        raise ValueError(f"Cohort reports official test access in {run_dir}.")
    return payload, indices


def _validate_beta_scaling(
    rows: Sequence[Mapping[str, str]],
    *,
    plan: Mapping[str, Any],
    variant: str,
    run_dir: Path,
) -> tuple[int, float]:
    required = {
        "architecture",
        "scheme",
        "checkpoint_role",
        "eqprop_variant",
        "T",
        "K",
        "actual_beta",
        "requested_actual_beta",
        "effective_beta",
        "amplification_factor",
        "amplification_depth_L",
        "amplification_exponent_convention",
        "scored_interaction_count",
        "beta_capped",
    }
    _require_columns(rows, required, path=run_dir / "beta_scaling.csv")
    if len(rows) != len(SCHEMES) * len(CHECKPOINT_ROLES):
        raise ValueError(f"Expected six beta-scaling rows in {run_dir}.")
    effective_values = {
        _finite_float(row["effective_beta"], label="effective_beta") for row in rows
    }
    if len(effective_values) != 1:
        raise ValueError(f"Schemes do not share one injected beta in {run_dir}.")
    injected_beta = next(iter(effective_values))
    beta_index = _beta_index(injected_beta, plan["beta_contract"]["injected_betas"])
    expected_keys = {(scheme, role) for scheme in SCHEMES for role in CHECKPOINT_ROLES}
    observed_keys: set[tuple[str, str]] = set()
    divisors = plan["beta_contract"]["base_beta_divisor"]
    for row in rows:
        key = (row["scheme"], row["checkpoint_role"])
        if key in observed_keys:
            raise ValueError(f"Duplicate beta-scaling row {key} in {run_dir}.")
        observed_keys.add(key)
        scheme, _ = key
        if row["architecture"] != "conv3" or row["eqprop_variant"] != variant:
            raise ValueError(f"Beta-scaling context mismatch in {run_dir}.")
        if (int(row["T"]), int(row["K"])) != EXPECTED_TK:
            raise ValueError(f"Beta-scaling T/K mismatch in {run_dir}.")
        divisor = float(divisors[scheme])
        actual = _finite_float(row["actual_beta"], label="actual_beta")
        requested = _finite_float(row["requested_actual_beta"], label="requested_actual_beta")
        factor = _finite_float(row["amplification_factor"], label="amplification_factor")
        if not _close(actual, injected_beta / divisor):
            raise ValueError(f"Base beta scaling mismatch for {scheme} in {run_dir}.")
        if not _close(requested, actual) or not _close(factor, divisor):
            raise ValueError(f"Requested beta/factor mismatch for {scheme} in {run_dir}.")
        if int(row["amplification_depth_L"]) != 3:
            raise ValueError(f"Output-row amplification depth is not 3 in {run_dir}.")
        if row["amplification_exponent_convention"] != "output_bias_current_row":
            raise ValueError(f"Wrong amplification exponent convention in {run_dir}.")
        if int(row["scored_interaction_count"]) != 4:
            raise ValueError(f"Wrong scored interaction count in {run_dir}.")
        # ``beta_capped`` is source-grid provenance, not evidence that this
        # replay's requested current was changed.  At the exact B=0.01 source
        # boundary the ours/legacy grid rows legitimately carry ``True`` even
        # though requested_actual_beta, actual_beta, and effective_beta are
        # unchanged.  The exact equalities above are the fail-closed no-silent-
        # cap guard; retain but do not reinterpret this metadata bit.
        _parse_bool(row["beta_capped"], label="beta_capped")
    if observed_keys != expected_keys:
        raise ValueError(f"Incomplete beta-scaling contexts in {run_dir}.")
    return beta_index, injected_beta


def _validate_selected_cases(
    *,
    manifest: Mapping[str, Any],
    plan: Mapping[str, Any],
    variant: str,
    injected_beta: float,
    run_dir: Path,
) -> None:
    resolved = manifest["configuration"]["resolved"]
    if resolved.get("eqprop_variant") != variant:
        raise ValueError(f"Manifest EqProp variant mismatch in {run_dir}.")
    if tuple(resolved.get("tk_override", ())) != EXPECTED_TK:
        raise ValueError(f"Manifest T/K override mismatch in {run_dir}.")
    if int(resolved.get("batch_index", -1)) != EXPECTED_BATCH_INDEX:
        raise ValueError(f"Manifest batch index mismatch in {run_dir}.")
    cases = resolved.get("selected_cases", [])
    if len(cases) != 6:
        raise ValueError(f"Manifest does not contain six selected cases in {run_dir}.")
    divisors = plan["beta_contract"]["base_beta_divisor"]
    expected = {(scheme, role) for scheme in SCHEMES for role in CHECKPOINT_ROLES}
    observed: set[tuple[str, str]] = set()
    for case in cases:
        key = (str(case.get("scheme")), str(case.get("checkpoint_role")))
        if key in observed:
            raise ValueError(f"Duplicate manifest case {key} in {run_dir}.")
        observed.add(key)
        scheme, _ = key
        actual = _finite_float(case.get("actual_beta"), label="manifest actual_beta")
        effective = _finite_float(
            case.get("resolved_beta_row", {}).get("beta_effective"),
            label="manifest beta_effective",
        )
        if case.get("architecture") != "conv3" or (int(case["T"]), int(case["K"])) != EXPECTED_TK:
            raise ValueError(f"Manifest case context mismatch in {run_dir}.")
        if not bool(case.get("beta_explicit_cli")):
            raise ValueError(f"Manifest case beta is not explicit in {run_dir}.")
        if not _close(actual, injected_beta / float(divisors[scheme])):
            raise ValueError(f"Manifest base beta mismatch for {scheme} in {run_dir}.")
        if not _close(effective, injected_beta):
            raise ValueError(f"Manifest injected beta mismatch for {scheme} in {run_dir}.")
    if observed != expected:
        raise ValueError(f"Manifest case coverage incomplete in {run_dir}.")


def _validate_residual_stat_family(
    row: Mapping[str, str],
    *,
    prefix: str,
    run_dir: Path,
    allow_blank: bool = False,
) -> dict[str, float] | None:
    fields_by_stat = {
        "mean": f"{prefix}_mean",
        "median": f"{prefix}_median",
        "p90": f"{prefix}_p90",
        "p99": f"{prefix}_p99",
        "maximum": (
            f"{prefix}imum"
            if prefix in {"selected_max", "raw_max"}
            else f"{prefix}_maximum"
        ),
    }
    fields = tuple(fields_by_stat.values())
    values = [row.get(field, "") for field in fields]
    if allow_blank and all(str(value).strip() == "" for value in values):
        return None
    if any(str(value).strip() == "" for value in values):
        raise ValueError(f"Partially missing {prefix} residual statistics in {run_dir}.")
    parsed = {
        stat: _finite_float(row[field], label=field)
        for stat, field in fields_by_stat.items()
    }
    if any(value < 0.0 for value in parsed.values()):
        raise ValueError(f"Negative {prefix} residual statistic in {run_dir}.")
    if not (
        parsed["median"] <= parsed["p90"]
        <= parsed["p99"]
        <= parsed["maximum"]
    ):
        raise ValueError(f"Non-monotonic {prefix} residual quantiles in {run_dir}.")
    if parsed["mean"] > parsed["maximum"]:
        raise ValueError(f"{prefix} residual mean exceeds its maximum in {run_dir}.")
    return parsed


def _validate_residual_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    plan: Mapping[str, Any],
    source_config: Mapping[str, Any],
    variant: str,
    beta_index: int,
    injected_beta: float,
    payload_sha: str,
    indices_sha: str,
    run_id: str,
    run_dir: Path,
) -> dict[str, Any]:
    required = {
        "schema",
        "architecture",
        "scheme",
        "checkpoint_role",
        "precision",
        "eqprop_variant",
        "T",
        "K",
        "batch_index",
        "batch_payload_sha256",
        "batch_source_indices_sha256",
        "phase",
        "beta",
        "layer_index",
        "state_layer_name",
        "state_layer_type",
        "layer_role",
        "residual_mode",
        "example_count",
        "expected_example_count",
        "unique_source_index_count",
        "coverage_complete",
        "outcome",
        "per_sample_selected_max_sha256",
        "selected_max_mean",
        "selected_max_median",
        "selected_max_p90",
        "selected_max_p99",
        "selected_maximum",
        "raw_max_mean",
        "raw_max_median",
        "raw_max_p90",
        "raw_max_p99",
        "raw_maximum",
        "clamp_occupancy_mean",
        "clamp_occupancy_median",
        "clamp_occupancy_p90",
        "clamp_occupancy_p99",
        "clamp_occupancy_maximum",
        "gate_threshold",
        "gate_comparison",
        "gate_passed",
        "uniform_equilibrium_max_threshold",
        "uniform_equilibrium",
        "hard_failure_maximum_threshold",
        "hard_failure",
    }
    path = run_dir / "equilibrium_residual_summary.csv"
    _require_columns(rows, required, path=path)
    phases = (
        ("post_T_free", "zero", "positive")
        if variant == "positive_one_sided"
        else ("post_T_free", "zero", "negative", "positive")
    )
    expected_row_count = (
        EXPECTED_ONE_SIDED_RESIDUAL_ROWS
        if variant == "positive_one_sided"
        else EXPECTED_CENTERED_RESIDUAL_ROWS
    )
    if len(rows) != expected_row_count:
        raise ValueError(
            f"Expected {expected_row_count} residual rows in {run_dir}, got {len(rows)}."
        )
    expected_keys = {
        (scheme, role, precision, phase, layer)
        for scheme in SCHEMES
        for role in CHECKPOINT_ROLES
        for precision in PRECISIONS
        for phase in phases
        for layer in RESIDUAL_LAYER_CONTRACT
    }
    observed: set[tuple[str, str, str, str, str]] = set()
    residual_contract = source_config["equilibrium_residual_contract"]
    expected_gate_threshold = float(
        residual_contract["per_layer_sample_max_p90_threshold"]
    )
    expected_uniform_threshold = float(
        residual_contract["uniform_endpoint_maximum_threshold"]
    )
    expected_hard_threshold = float(
        residual_contract["hard_failure_maximum_threshold"]
    )
    expected_comparison = str(residual_contract["comparison"])
    divisors = plan["beta_contract"]["base_beta_divisor"]
    failure_coordinates: list[dict[str, Any]] = []
    context_counts: dict[tuple[str, str, str], dict[str, int]] = {}
    for row in rows:
        key = (
            row["scheme"],
            row["checkpoint_role"],
            row["precision"],
            row["phase"],
            row["state_layer_name"],
        )
        if key in observed:
            raise ValueError(f"Duplicate equilibrium-residual row {key} in {run_dir}.")
        observed.add(key)
        scheme, role, precision, phase, layer = key
        if row["schema"] != RESIDUAL_SCHEMA or row["architecture"] != "conv3":
            raise ValueError(f"Residual schema/architecture mismatch in {run_dir}.")
        if row["eqprop_variant"] != variant or (int(row["T"]), int(row["K"])) != EXPECTED_TK:
            raise ValueError(f"Residual estimator or T/K mismatch in {run_dir}.")
        if int(row["batch_index"]) != 0:
            raise ValueError(f"Residual batch index mismatch in {run_dir}.")
        if row["batch_payload_sha256"] != payload_sha or row["batch_source_indices_sha256"] != indices_sha:
            raise ValueError(f"Residual cohort identity mismatch in {run_dir}.")
        expected_base = injected_beta / float(divisors[scheme])
        expected_beta = (
            -expected_base
            if phase == "negative"
            else expected_base
            if phase == "positive"
            else 0.0
        )
        actual_beta = _finite_float(row["beta"], label="residual beta")
        if not _close(actual_beta, expected_beta):
            raise ValueError(f"Residual phase beta mismatch in {run_dir}.")
        layer_contract = RESIDUAL_LAYER_CONTRACT[layer]
        for field in ("layer_index", "state_layer_type", "layer_role", "residual_mode"):
            observed_value: Any = int(row[field]) if field == "layer_index" else row[field]
            if observed_value != layer_contract[field]:
                raise ValueError(
                    f"Residual layer contract {field} mismatch for {layer} in {run_dir}."
                )
        if (
            int(row["example_count"]) != EXPECTED_BATCH_SIZE
            or int(row["expected_example_count"]) != EXPECTED_BATCH_SIZE
            or int(row["unique_source_index_count"]) != EXPECTED_BATCH_SIZE
        ):
            raise ValueError(f"Residual example/index coverage is not 16 in {run_dir}.")
        if not _parse_bool(row["coverage_complete"], label="residual coverage_complete"):
            raise ValueError(f"Residual sample coverage incomplete in {run_dir}.")
        if row["outcome"] != "ok":
            raise ValueError(f"Residual outcome failed in {run_dir}.")
        digest = row["per_sample_selected_max_sha256"]
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(f"Invalid residual per-sample SHA-256 in {run_dir}.")
        selected = _validate_residual_stat_family(
            row, prefix="selected_max", run_dir=run_dir
        )
        raw = _validate_residual_stat_family(row, prefix="raw_max", run_dir=run_dir)
        assert selected is not None and raw is not None
        occupancy = _validate_residual_stat_family(
            row,
            prefix="clamp_occupancy",
            run_dir=run_dir,
            allow_blank=layer_contract["residual_mode"] == "raw",
        )
        if layer_contract["residual_mode"] == "raw":
            if occupancy is not None:
                raise ValueError(f"Output-layer clamp occupancy must be undefined in {run_dir}.")
            if any(not _close(selected[name], raw[name]) for name in selected):
                raise ValueError(f"Output selected/raw residual statistics differ in {run_dir}.")
        else:
            if occupancy is None or any(value > 1.0 for value in occupancy.values()):
                raise ValueError(f"Hidden-layer clamp occupancy is invalid in {run_dir}.")
        gate_threshold = _finite_float(row["gate_threshold"], label="residual gate threshold")
        uniform_threshold = _finite_float(
            row["uniform_equilibrium_max_threshold"],
            label="uniform residual threshold",
        )
        hard_threshold = _finite_float(
            row["hard_failure_maximum_threshold"],
            label="hard-failure residual threshold",
        )
        if (
            not _close(gate_threshold, expected_gate_threshold)
            or not _close(uniform_threshold, expected_uniform_threshold)
            or not _close(hard_threshold, expected_hard_threshold)
            or row["gate_comparison"] != expected_comparison
            or row["gate_comparison"] != "strictly_less_than"
        ):
            raise ValueError(f"Residual threshold/comparison contract mismatch in {run_dir}.")
        hard_failure = _parse_bool(row["hard_failure"], label="residual hard_failure")
        expected_hard_failure = selected["maximum"] >= hard_threshold
        if hard_failure != expected_hard_failure:
            raise ValueError(f"Residual hard-failure semantics mismatch in {run_dir}.")
        if hard_failure:
            raise ValueError(f"Residual hard failure recorded in {run_dir}.")
        uniform = _parse_bool(
            row["uniform_equilibrium"], label="uniform_equilibrium"
        )
        if uniform != (selected["maximum"] < uniform_threshold):
            raise ValueError(f"Uniform-equilibrium residual semantics mismatch in {run_dir}.")
        gate_passed = _parse_bool(row["gate_passed"], label="residual gate_passed")
        expected_gate_passed = selected["p90"] < gate_threshold and not hard_failure
        if gate_passed != expected_gate_passed:
            raise ValueError(f"Residual p90 gate semantics mismatch in {run_dir}.")
        context_key = (scheme, role, precision)
        counts = context_counts.setdefault(
            context_key, {"row_count": 0, "gate_pass_count": 0, "gate_failure_count": 0}
        )
        counts["row_count"] += 1
        counts["gate_pass_count" if gate_passed else "gate_failure_count"] += 1
        if not gate_passed:
            failure_coordinates.append(
                {
                    "run_id": run_id,
                    "eqprop_variant": variant,
                    "beta_index": beta_index,
                    "injected_beta": injected_beta,
                    "actual_base_beta": actual_beta,
                    "scheme": scheme,
                    "checkpoint_role": role,
                    "precision": precision,
                    "phase": phase,
                    "state_layer_name": layer,
                    "state_layer": STATE_LABELS[layer],
                    "selected_max_p90": selected["p90"],
                    "selected_maximum": selected["maximum"],
                    "gate_threshold": gate_threshold,
                    "hard_failure": False,
                }
            )
    if observed != expected_keys:
        missing = expected_keys - observed
        extra = observed - expected_keys
        raise ValueError(
            f"Residual coverage mismatch in {run_dir}: "
            f"missing={len(missing)}, extra={len(extra)}."
        )
    return {
        "row_count": len(rows),
        "gate_failure_coordinates": failure_coordinates,
        "context_counts": context_counts,
    }


def _validate_displacement_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    plan: Mapping[str, Any],
    variant: str,
    injected_beta: float,
    payload_sha: str,
    indices_sha: str,
    run_dir: Path,
) -> dict[tuple[str, str, str, str, str], Mapping[str, str]]:
    required = {
        "architecture",
        "scheme",
        "checkpoint_role",
        "precision",
        "eqprop_variant",
        "T",
        "K",
        "batch_index",
        "batch_payload_sha256",
        "batch_source_indices_sha256",
        "phase",
        "actual_beta",
        "reference_kind",
        "outcome",
        "state_layer_name",
        "relative_displacement",
        "delta_rms",
    }
    _require_columns(rows, required, path=run_dir / "state_displacement.csv")
    phase_references = (
        (("positive", "post_T_free"), ("positive", "matched_zero_K"))
        if variant == "positive_one_sided"
        else (
            ("negative", "post_T_free"),
            ("negative", "matched_zero_K"),
            ("positive", "post_T_free"),
            ("positive", "matched_zero_K"),
            ("positive_minus_negative", "negative_phase"),
        )
    )
    expected_keys = {
        (scheme, role, precision, phase, reference, state)
        for scheme in SCHEMES
        for role in CHECKPOINT_ROLES
        for precision in PRECISIONS
        for phase, reference in phase_references
        for state in ALL_STATE_NAMES
    }
    observed: dict[tuple[str, str, str, str, str, str], Mapping[str, str]] = {}
    divisors = plan["beta_contract"]["base_beta_divisor"]
    for row in rows:
        key = (
            row["scheme"],
            row["checkpoint_role"],
            row["precision"],
            row["phase"],
            row["reference_kind"],
            row["state_layer_name"],
        )
        if key in observed:
            raise ValueError(f"Duplicate state-displacement row {key} in {run_dir}.")
        observed[key] = row
        if row["architecture"] != "conv3" or row["eqprop_variant"] != variant:
            raise ValueError(f"State-displacement context mismatch in {run_dir}.")
        if (int(row["T"]), int(row["K"])) != EXPECTED_TK or int(row["batch_index"]) != 0:
            raise ValueError(f"State-displacement T/K or batch mismatch in {run_dir}.")
        if row["batch_payload_sha256"] != payload_sha or row["batch_source_indices_sha256"] != indices_sha:
            raise ValueError(f"State-displacement cohort mismatch in {run_dir}.")
        if row["outcome"] != "ok":
            raise ValueError(f"State-displacement outcome failed in {run_dir}.")
        expected_base = injected_beta / float(divisors[row["scheme"]])
        signed_expected = -expected_base if row["phase"] == "negative" else expected_base
        if not _close(_finite_float(row["actual_beta"], label="state actual_beta"), signed_expected):
            raise ValueError(f"State-displacement beta sign/magnitude mismatch in {run_dir}.")
        for field in ("relative_displacement", "delta_rms"):
            value = _finite_float(row[field], label=f"state {field}")
            if value < 0.0:
                raise ValueError(f"Negative state metric {field} in {run_dir}.")
    if set(observed) != expected_keys:
        missing = expected_keys - set(observed)
        extra = set(observed) - expected_keys
        raise ValueError(
            f"State-displacement coverage mismatch in {run_dir}: "
            f"missing={len(missing)}, extra={len(extra)}."
        )
    primary: dict[tuple[str, str, str, str, str], Mapping[str, str]] = {}
    for scheme in SCHEMES:
        for role in CHECKPOINT_ROLES:
            for precision in PRECISIONS:
                for phase in (("positive",) if variant == "positive_one_sided" else ("positive", "negative")):
                    for state in STATE_LABELS:
                        source_key = (scheme, role, precision, phase, "matched_zero_K", state)
                        primary[(scheme, role, precision, phase, state)] = observed[source_key]
    return primary


def _cosine_status(
    *, cosine: float | None, eqprop_l2: float, bptt_l2: float, eqprop_zero_fraction: float, bptt_zero_fraction: float
) -> tuple[bool, bool, bool, str]:
    eqprop_zero = eqprop_l2 == 0.0 and eqprop_zero_fraction == 1.0
    bptt_zero = bptt_l2 == 0.0 and bptt_zero_fraction == 1.0
    if cosine is not None:
        if eqprop_l2 == 0.0 or bptt_l2 == 0.0:
            raise ValueError("A cosine is defined despite an exact-zero vector.")
        return True, eqprop_zero, bptt_zero, ""
    if not (eqprop_zero or bptt_zero):
        raise ValueError("Undefined cosine is not explained by an exact-zero vector.")
    if eqprop_zero and bptt_zero:
        reason = "both_exact_zero"
    elif eqprop_zero:
        reason = "eqprop_exact_zero"
    else:
        reason = "bptt_exact_zero"
    return False, eqprop_zero, bptt_zero, reason


def _normalize_parameter_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    displacement: Mapping[tuple[str, str, str, str, str], Mapping[str, str]],
    plan: Mapping[str, Any],
    bundle_study_id: str,
    run_id: str,
    result_sha256: str,
    variant: str,
    beta_index: int,
    injected_beta: float,
    payload_sha: str,
    indices_sha: str,
    run_dir: Path,
) -> list[dict[str, Any]]:
    required = {
        "schema",
        "architecture",
        "scheme",
        "checkpoint_role",
        "eqprop_variant",
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
        "residual_limited",
        "outcome",
        *(f"{precision}_eqprop_gradient_dtype" for precision in PRECISIONS),
        *(f"{precision}_bptt_gradient_dtype" for precision in PRECISIONS),
        *(f"{precision}_eqprop_l2" for precision in PRECISIONS),
        *(f"{precision}_bptt_l2" for precision in PRECISIONS),
        *(f"{precision}_exact_zero_fraction" for precision in PRECISIONS),
        *(f"{precision}_bptt_exact_zero_fraction" for precision in PRECISIONS),
        *(f"{precision}_eqprop_vs_bptt_cosine" for precision in PRECISIONS),
        *(f"{precision}_eqprop_over_bptt_norm_ratio" for precision in PRECISIONS),
        *(f"{precision}_eqprop_vs_bptt_symmetric_norm_delta" for precision in PRECISIONS),
    }
    _require_columns(rows, required, path=run_dir / "parameter_precision_comparison.csv")
    expected_keys = {(scheme, role, parameter) for scheme in SCHEMES for role in CHECKPOINT_ROLES for parameter in PARAMETERS}
    observed: set[tuple[str, str, str]] = set()
    output: list[dict[str, Any]] = []
    divisors = plan["beta_contract"]["base_beta_divisor"]
    for row in rows:
        key = (row["scheme"], row["checkpoint_role"], row["parameter_name"])
        if key in observed:
            raise ValueError(f"Duplicate parameter row {key} in {run_dir}.")
        observed.add(key)
        scheme, role, parameter = key
        if row["schema"] != BUNDLE_PARAMETER_SCHEMA or row["architecture"] != "conv3":
            raise ValueError(f"Parameter schema/architecture mismatch in {run_dir}.")
        if row["eqprop_variant"] != variant or (int(row["T"]), int(row["K"])) != EXPECTED_TK:
            raise ValueError(f"Parameter estimator or T/K mismatch in {run_dir}.")
        if int(row["batch_index"]) != 0 or row["batch_payload_sha256"] != payload_sha:
            raise ValueError(f"Parameter fixed-batch identity mismatch in {run_dir}.")
        if not _parse_bool(row["bias_excluded"], label="bias_excluded"):
            raise ValueError(f"A scored bias row is present in {run_dir}.")
        divisor = float(divisors[scheme])
        actual = _finite_float(row["actual_beta"], label="parameter actual_beta")
        effective = _finite_float(row["effective_beta"], label="parameter effective_beta")
        factor = _finite_float(row["amplification_factor"], label="parameter amplification_factor")
        if not _close(actual, injected_beta / divisor) or not _close(effective, injected_beta) or not _close(factor, divisor):
            raise ValueError(f"Parameter beta scaling mismatch in {run_dir}.")
        state_name = STATE_LAYER_BY_PARAMETER[parameter]
        for precision in PRECISIONS:
            eqprop_l2 = _finite_float(row[f"{precision}_eqprop_l2"], label="eqprop_l2")
            bptt_l2 = _finite_float(row[f"{precision}_bptt_l2"], label="bptt_l2")
            eqprop_zero_fraction = _finite_float(row[f"{precision}_exact_zero_fraction"], label="eqprop zero fraction")
            bptt_zero_fraction = _finite_float(row[f"{precision}_bptt_exact_zero_fraction"], label="bptt zero fraction")
            cosine = _optional_float(row[f"{precision}_eqprop_vs_bptt_cosine"], label="EqProp/BPTT cosine")
            defined, eqprop_zero, bptt_zero, undefined_reason = _cosine_status(
                cosine=cosine,
                eqprop_l2=eqprop_l2,
                bptt_l2=bptt_l2,
                eqprop_zero_fraction=eqprop_zero_fraction,
                bptt_zero_fraction=bptt_zero_fraction,
            )
            positive = displacement[(scheme, role, precision, "positive", state_name)]
            negative = displacement.get((scheme, role, precision, "negative", state_name))
            output.append(
                {
                    "schema": SCHEMA,
                    "plan_study_id": plan["study_id"],
                    "bundle_manifest_study_id": bundle_study_id,
                    "run_id": run_id,
                    "input_result_sha256": result_sha256,
                    "eqprop_variant": variant,
                    "scheme": scheme,
                    "checkpoint_role": role,
                    "precision": precision,
                    "T": 64,
                    "K": 64,
                    "batch_index": 0,
                    "batch_payload_sha256": payload_sha,
                    "batch_source_indices_sha256": indices_sha,
                    "beta_index": beta_index,
                    "actual_base_beta": actual,
                    "injected_beta": injected_beta,
                    "amplification_factor": factor,
                    "parameter_name": parameter,
                    "gradient_layer": PARAMETER_LABELS[parameter],
                    "state_layer_name": state_name,
                    "state_layer": STATE_LABELS[state_name],
                    "bias_excluded": True,
                    "residual_limited": _parse_bool(row["residual_limited"], label="residual_limited"),
                    "gradient_outcome": row["outcome"],
                    "eqprop_gradient_dtype": row[f"{precision}_eqprop_gradient_dtype"],
                    "bptt_gradient_dtype": row[f"{precision}_bptt_gradient_dtype"],
                    "eqprop_l2": eqprop_l2,
                    "bptt_l2": bptt_l2,
                    "eqprop_exact_zero_fraction": eqprop_zero_fraction,
                    "bptt_exact_zero_fraction": bptt_zero_fraction,
                    "eqprop_exact_zero": eqprop_zero,
                    "bptt_exact_zero": bptt_zero,
                    "cosine_defined": defined,
                    "cosine_undefined_reason": undefined_reason,
                    "eqprop_vs_bptt_cosine": cosine,
                    "eqprop_over_bptt_norm_ratio": _optional_float(
                        row[f"{precision}_eqprop_over_bptt_norm_ratio"], label="norm ratio"
                    ),
                    "eqprop_vs_bptt_symmetric_norm_delta": _optional_float(
                        row[f"{precision}_eqprop_vs_bptt_symmetric_norm_delta"], label="symmetric norm delta"
                    ),
                    "matched_zero_positive_relative_displacement": _finite_float(
                        positive["relative_displacement"], label="positive relative displacement"
                    ),
                    "matched_zero_positive_delta_rms": _finite_float(
                        positive["delta_rms"], label="positive delta RMS"
                    ),
                    "matched_zero_negative_relative_displacement": (
                        None
                        if negative is None
                        else _finite_float(negative["relative_displacement"], label="negative relative displacement")
                    ),
                    "matched_zero_negative_delta_rms": (
                        None if negative is None else _finite_float(negative["delta_rms"], label="negative delta RMS")
                    ),
                }
            )
    if observed != expected_keys:
        raise ValueError(f"Parameter-layer coverage mismatch in {run_dir}.")
    if len(output) != 48:
        raise ValueError(f"Expected 48 dtype/layer rows in {run_dir}, got {len(output)}.")
    return output


def _load_bundle(
    run_dir: Path,
    *,
    plan: Mapping[str, Any],
    source_config: Mapping[str, Any],
    source_config_path: Path,
    source_run: Path,
) -> dict[str, Any]:
    validation_errors = reporting.validate_run(run_dir)
    if validation_errors:
        raise ValueError(f"Invalid canonical bundle {run_dir}: {validation_errors}.")
    manifest = _read_json(run_dir / "manifest.json")
    status = _read_json(run_dir / "status.json")
    result = _read_json(run_dir / "result.json")
    if status.get("state") != "complete":
        raise ValueError(f"Production bundle is not complete: {run_dir}.")
    result_sha = reporting.sha256_file(run_dir / "result.json")
    if status.get("result_sha256") != result_sha:
        raise ValueError(f"Status/result hash mismatch in {run_dir}.")
    if bool(manifest.get("smoke")) or bool(result.get("smoke")):
        raise ValueError(f"Smoke reached production loader: {run_dir}.")
    _validate_completion(result, run_dir=run_dir)
    guards = _read_json(run_dir / "read_only_guards.json")
    _validate_read_only_guards(guards, run_dir=run_dir)
    bundle_study_id = _validate_source_identity(
        manifest=manifest,
        source_config=source_config,
        source_config_path=source_config_path,
        source_run=source_run,
        plan=plan,
    )
    payload_sha, indices_sha = _validate_cohort(
        run_dir=run_dir,
        plan=plan,
        source_cohort_path=source_run / "cohort.json",
    )
    variant = str(manifest.get("configuration", {}).get("resolved", {}).get("eqprop_variant", ""))
    if variant not in VARIANTS:
        raise ValueError(f"Unexpected EqProp variant in {run_dir}: {variant!r}.")
    scaling_rows = _read_csv(run_dir / "beta_scaling.csv")
    beta_index, injected_beta = _validate_beta_scaling(
        scaling_rows, plan=plan, variant=variant, run_dir=run_dir
    )
    expected_run_id = _expected_run_id(variant, beta_index)
    if run_dir.name != expected_run_id or manifest.get("run_id") != expected_run_id:
        raise ValueError(
            f"Production run ID does not match launcher contract in {run_dir}: "
            f"expected {expected_run_id!r}."
        )
    _validate_selected_cases(
        manifest=manifest,
        plan=plan,
        variant=variant,
        injected_beta=injected_beta,
        run_dir=run_dir,
    )
    terminal = result.get("terminal_metrics", {})
    if terminal.get("eqprop_variant") != variant:
        raise ValueError(f"Terminal estimator differs in {run_dir}.")
    if int(terminal.get("checkpoint_cases", -1)) != 6 or int(terminal.get("precision_comparisons", -1)) != 24:
        raise ValueError(f"Terminal case/layer counts differ in {run_dir}.")
    if int(terminal.get("batch_index", -1)) != 0 or int(terminal.get("batch_examples", -1)) != 16:
        raise ValueError(f"Terminal fixed-batch context differs in {run_dir}.")
    if terminal.get("bias_gradients_excluded") is not True:
        raise ValueError(f"Terminal metrics include bias gradients in {run_dir}.")
    residual_rows = _read_csv(run_dir / "equilibrium_residual_summary.csv")
    residual_audit = _validate_residual_rows(
        residual_rows,
        plan=plan,
        source_config=source_config,
        variant=variant,
        beta_index=beta_index,
        injected_beta=injected_beta,
        payload_sha=payload_sha,
        indices_sha=indices_sha,
        run_id=expected_run_id,
        run_dir=run_dir,
    )
    displacement_rows = _read_csv(run_dir / "state_displacement.csv")
    displacement = _validate_displacement_rows(
        displacement_rows,
        plan=plan,
        variant=variant,
        injected_beta=injected_beta,
        payload_sha=payload_sha,
        indices_sha=indices_sha,
        run_dir=run_dir,
    )
    parameter_rows = _read_csv(run_dir / "parameter_precision_comparison.csv")
    normalized = _normalize_parameter_rows(
        parameter_rows,
        displacement=displacement,
        plan=plan,
        bundle_study_id=bundle_study_id,
        run_id=expected_run_id,
        result_sha256=result_sha,
        variant=variant,
        beta_index=beta_index,
        injected_beta=injected_beta,
        payload_sha=payload_sha,
        indices_sha=indices_sha,
        run_dir=run_dir,
    )
    return {
        "run_id": expected_run_id,
        "run_dir": str(run_dir),
        "variant": variant,
        "beta_index": beta_index,
        "injected_beta": injected_beta,
        "manifest_study_id": bundle_study_id,
        "manifest_sha256": reporting.sha256_file(run_dir / "manifest.json"),
        "result_sha256": result_sha,
        "status_sha256": reporting.sha256_file(run_dir / "status.json"),
        "residual_row_count": residual_audit["row_count"],
        "residual_gate_failure_count": len(
            residual_audit["gate_failure_coordinates"]
        ),
        "residual_gate_failure_coordinates": residual_audit[
            "gate_failure_coordinates"
        ],
        "residual_context_counts": residual_audit["context_counts"],
        "rows": normalized,
    }


def _discover_bundles(
    study_root: Path,
    *,
    plan: Mapping[str, Any],
    source_config: Mapping[str, Any],
    source_config_path: Path,
    source_run: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not study_root.is_dir():
        raise FileNotFoundError(study_root)
    bundles: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for run_dir in sorted(study_root.iterdir()):
        if not run_dir.is_dir() or run_dir.name == "analysis":
            continue
        manifest_path = run_dir / "manifest.json"
        status_path = run_dir / "status.json"
        if not manifest_path.exists() and not status_path.exists():
            continue
        if not manifest_path.is_file() or not status_path.is_file():
            raise ValueError(f"Partial canonical bundle metadata in {run_dir}.")
        manifest = _read_json(manifest_path)
        status = _read_json(status_path)
        if bool(manifest.get("smoke")):
            excluded.append(
                {
                    "run_id": run_dir.name,
                    "reason": "smoke_bundle",
                    "state": status.get("state"),
                    "manifest_sha256": reporting.sha256_file(manifest_path),
                    "status_sha256": reporting.sha256_file(status_path),
                }
            )
            continue
        if status.get("state") != "complete":
            raise ValueError(
                f"Every non-smoke production bundle must be complete: {run_dir} "
                f"has state={status.get('state')!r}."
            )
        bundles.append(
            _load_bundle(
                run_dir,
                plan=plan,
                source_config=source_config,
                source_config_path=source_config_path,
                source_run=source_run,
            )
        )
    if len(bundles) != EXPECTED_PRODUCTION_BUNDLES:
        raise ValueError(
            f"Expected exactly {EXPECTED_PRODUCTION_BUNDLES} production bundles, "
            f"found {len(bundles)}."
        )
    expected_coordinates = {(variant, index) for variant in VARIANTS for index in range(len(EXPECTED_BETAS))}
    observed_coordinates = {(bundle["variant"], bundle["beta_index"]) for bundle in bundles}
    if len(observed_coordinates) != len(bundles) or observed_coordinates != expected_coordinates:
        raise ValueError("Production beta/estimator bundle coverage is not exactly 11 x 2.")
    return bundles, excluded


def _validate_complete_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    if len(rows) != EXPECTED_SUMMARY_ROWS:
        raise ValueError(f"Expected {EXPECTED_SUMMARY_ROWS} normalized rows, got {len(rows)}.")
    expected = {
        (variant, beta_index, scheme, role, precision, parameter)
        for variant in VARIANTS
        for beta_index in range(len(EXPECTED_BETAS))
        for scheme in SCHEMES
        for role in CHECKPOINT_ROLES
        for precision in PRECISIONS
        for parameter in PARAMETERS
    }
    observed = {
        (
            row["eqprop_variant"],
            int(row["beta_index"]),
            row["scheme"],
            row["checkpoint_role"],
            row["precision"],
            row["parameter_name"],
        )
        for row in rows
    }
    if len(observed) != len(rows) or observed != expected:
        raise ValueError("Normalized 3 x 2 x 11 x 2 x 2 x 4 coverage is incomplete.")
    for scheme in SCHEMES:
        for role in CHECKPOINT_ROLES:
            for precision in PRECISIONS:
                for parameter in PARAMETERS:
                    selected = [
                        row
                        for row in rows
                        if row["scheme"] == scheme
                        and row["checkpoint_role"] == role
                        and row["precision"] == precision
                        and row["parameter_name"] == parameter
                    ]
                    bptt = [float(row["bptt_l2"]) for row in selected]
                    if any(
                        not math.isclose(value, bptt[0], rel_tol=1.0e-6, abs_tol=1.0e-12)
                        for value in bptt[1:]
                    ):
                        raise ValueError(
                            f"BPTT reference changed across beta/estimator for "
                            f"{scheme}/{role}/{precision}/{parameter}."
                        )
    for beta_index in range(len(EXPECTED_BETAS)):
        for scheme in SCHEMES:
            for role in CHECKPOINT_ROLES:
                for precision in PRECISIONS:
                    for parameter in PARAMETERS:
                        pair = [
                            row
                            for row in rows
                            if int(row["beta_index"]) == beta_index
                            and row["scheme"] == scheme
                            and row["checkpoint_role"] == role
                            and row["precision"] == precision
                            and row["parameter_name"] == parameter
                        ]
                        if len(pair) != 2:
                            raise ValueError("Missing one-/two-sided paired context.")
                        positive_values = {
                            (
                                float(row["matched_zero_positive_relative_displacement"]),
                                float(row["matched_zero_positive_delta_rms"]),
                            )
                            for row in pair
                        }
                        if len(positive_values) != 1:
                            raise ValueError(
                                f"Positive endpoint changed across estimators for "
                                f"B={EXPECTED_BETAS[beta_index]}/{scheme}/{role}/{precision}/{parameter}."
                            )


def _layer_summaries(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for scheme in SCHEMES:
        for role in CHECKPOINT_ROLES:
            for variant in VARIANTS:
                for precision in PRECISIONS:
                    for parameter in PARAMETERS:
                        selected = [
                            row
                            for row in rows
                            if row["scheme"] == scheme
                            and row["checkpoint_role"] == role
                            and row["eqprop_variant"] == variant
                            and row["precision"] == precision
                            and row["parameter_name"] == parameter
                        ]
                        defined = [row for row in selected if row["cosine_defined"]]
                        best = max(defined, key=lambda row: float(row["eqprop_vs_bptt_cosine"])) if defined else None
                        output.append(
                            {
                                "scheme": scheme,
                                "checkpoint_role": role,
                                "eqprop_variant": variant,
                                "precision": precision,
                                "gradient_layer": PARAMETER_LABELS[parameter],
                                "state_layer": STATE_LABELS[STATE_LAYER_BY_PARAMETER[parameter]],
                                "defined_beta_count": len(defined),
                                "undefined_exact_zero_count": sum(
                                    not bool(row["cosine_defined"]) for row in selected
                                ),
                                "maximum_cosine": None if best is None else best["eqprop_vs_bptt_cosine"],
                                "beta_at_maximum_cosine": None if best is None else best["injected_beta"],
                                "positive_relative_displacement_at_maximum": (
                                    None if best is None else best["matched_zero_positive_relative_displacement"]
                                ),
                                "positive_delta_rms_at_maximum": (
                                    None if best is None else best["matched_zero_positive_delta_rms"]
                                ),
                            }
                        )
    return output


def _context_summaries(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for scheme in SCHEMES:
        for role in CHECKPOINT_ROLES:
            for variant in VARIANTS:
                for precision in PRECISIONS:
                    candidates = []
                    for beta_index, beta in enumerate(EXPECTED_BETAS):
                        selected = [
                            row
                            for row in rows
                            if row["scheme"] == scheme
                            and row["checkpoint_role"] == role
                            and row["eqprop_variant"] == variant
                            and row["precision"] == precision
                            and int(row["beta_index"]) == beta_index
                        ]
                        cosines = [row["eqprop_vs_bptt_cosine"] for row in selected]
                        candidates.append(
                            {
                                "injected_beta": beta,
                                "all_layer_cosines_defined": all(value is not None for value in cosines),
                                "minimum_layer_cosine": (
                                    min(float(value) for value in cosines if value is not None)
                                    if all(value is not None for value in cosines)
                                    else None
                                ),
                            }
                        )
                    eligible = [row for row in candidates if row["minimum_layer_cosine"] is not None]
                    best = max(eligible, key=lambda row: row["minimum_layer_cosine"]) if eligible else None
                    output.append(
                        {
                            "scheme": scheme,
                            "checkpoint_role": role,
                            "eqprop_variant": variant,
                            "precision": precision,
                            "beta_with_defined_all_layer_cosine_count": len(eligible),
                            "best_worst_layer_cosine": None if best is None else best["minimum_layer_cosine"],
                            "beta_at_best_worst_layer_cosine": None if best is None else best["injected_beta"],
                        }
                    )
    return output


def _aggregate_residual_audit(
    bundles: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    total_rows = sum(int(bundle["residual_row_count"]) for bundle in bundles)
    if total_rows != EXPECTED_RESIDUAL_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_RESIDUAL_ROWS} residual rows across the study, "
            f"got {total_rows}."
        )
    by_variant = {
        variant: sum(
            int(bundle["residual_row_count"])
            for bundle in bundles
            if bundle["variant"] == variant
        )
        for variant in VARIANTS
    }
    expected_by_variant = {
        "positive_one_sided": len(EXPECTED_BETAS)
        * EXPECTED_ONE_SIDED_RESIDUAL_ROWS,
        "centered": len(EXPECTED_BETAS) * EXPECTED_CENTERED_RESIDUAL_ROWS,
    }
    if by_variant != expected_by_variant:
        raise ValueError(
            f"Residual estimator row totals differ: observed={by_variant}, "
            f"expected={expected_by_variant}."
        )
    aggregate: dict[tuple[str, str, str], dict[str, int]] = {}
    for bundle in bundles:
        for key, counts in bundle["residual_context_counts"].items():
            target = aggregate.setdefault(
                key,
                {
                    "row_count": 0,
                    "gate_pass_count": 0,
                    "gate_failure_count": 0,
                },
            )
            for field in target:
                target[field] += int(counts[field])
    expected_contexts = {
        (scheme, role, precision)
        for scheme in SCHEMES
        for role in CHECKPOINT_ROLES
        for precision in PRECISIONS
    }
    if set(aggregate) != expected_contexts:
        raise ValueError("Residual dtype/context summary coverage is incomplete.")
    expected_rows_per_context = len(EXPECTED_BETAS) * (3 + 4) * len(
        RESIDUAL_LAYER_CONTRACT
    )
    context_summaries: list[dict[str, Any]] = []
    failures = [
        coordinate
        for bundle in bundles
        for coordinate in bundle["residual_gate_failure_coordinates"]
    ]
    for scheme in SCHEMES:
        for role in CHECKPOINT_ROLES:
            for precision in PRECISIONS:
                key = (scheme, role, precision)
                counts = aggregate[key]
                if counts["row_count"] != expected_rows_per_context:
                    raise ValueError(
                        f"Residual row count for {key} is {counts['row_count']}, "
                        f"expected {expected_rows_per_context}."
                    )
                selected_failures = [
                    row
                    for row in failures
                    if row["scheme"] == scheme
                    and row["checkpoint_role"] == role
                    and row["precision"] == precision
                ]
                if len(selected_failures) != counts["gate_failure_count"]:
                    raise ValueError(f"Residual failure summary count mismatch for {key}.")
                context_summaries.append(
                    {
                        "scheme": scheme,
                        "checkpoint_role": role,
                        "precision": precision,
                        **counts,
                        "all_gates_passed": counts["gate_failure_count"] == 0,
                        "failed_state_layers": sorted(
                            {row["state_layer"] for row in selected_failures}
                        ),
                        "failed_variants": sorted(
                            {row["eqprop_variant"] for row in selected_failures},
                            key=VARIANTS.index,
                        ),
                        "failed_phases": sorted(
                            {row["phase"] for row in selected_failures}
                        ),
                    }
                )
    failures.sort(
        key=lambda row: (
            VARIANTS.index(row["eqprop_variant"]),
            int(row["beta_index"]),
            SCHEMES.index(row["scheme"]),
            CHECKPOINT_ROLES.index(row["checkpoint_role"]),
            PRECISIONS.index(row["precision"]),
            row["phase"],
            int(row["state_layer_name"].split("_")[-1]),
        )
    )
    return {
        "row_count": total_rows,
        "row_count_by_variant": by_variant,
        "gate_pass_count": total_rows - len(failures),
        "gate_failure_count": len(failures),
        "hard_failure_count": 0,
        "all_float64_gates_passed": not any(
            row["precision"] == "float64" for row in failures
        ),
        "context_summaries": context_summaries,
        "gate_failure_coordinates": failures,
    }


def _panel_rows(
    rows: Sequence[Mapping[str, Any]], *, scheme: str, role: str, precision: str, variant: str, parameter: str
) -> list[Mapping[str, Any]]:
    return sorted(
        (
            row
            for row in rows
            if row["scheme"] == scheme
            and row["checkpoint_role"] == role
            and row["precision"] == precision
            and row["eqprop_variant"] == variant
            and row["parameter_name"] == parameter
        ),
        key=lambda row: int(row["beta_index"]),
    )


def _plot_metric(
    rows: Sequence[Mapping[str, Any]],
    *,
    precision: str,
    field: str,
    ylabel: str,
    output: Path,
    cosine: bool = False,
) -> None:
    colors = {
        "ConvWeight_0": "#1f77b4",
        "ConvWeight_1": "#ff7f0e",
        "ConvWeight_2": "#2ca02c",
        "DenseWeight_0": "#d62728",
    }
    styles = {"positive_one_sided": "-", "centered": "--"}
    fig, axes = plt.subplots(3, 2, figsize=(13.5, 11.0), sharex=True, sharey=True)
    for row_index, scheme in enumerate(SCHEMES):
        for column_index, role in enumerate(CHECKPOINT_ROLES):
            axis = axes[row_index][column_index]
            undefined = 0
            for parameter in PARAMETERS:
                for variant in VARIANTS:
                    selected = _panel_rows(
                        rows,
                        scheme=scheme,
                        role=role,
                        precision=precision,
                        variant=variant,
                        parameter=parameter,
                    )
                    x = [float(row["injected_beta"]) for row in selected]
                    y = [
                        math.nan if row[field] is None else float(row[field])
                        for row in selected
                    ]
                    undefined += sum(math.isnan(value) for value in y)
                    label = f"{PARAMETER_LABELS[parameter]} · {'one-sided' if variant == 'positive_one_sided' else 'centered'}"
                    axis.plot(
                        x,
                        y,
                        color=colors[parameter],
                        linestyle=styles[variant],
                        marker="o" if variant == "positive_one_sided" else "s",
                        markersize=3.2,
                        linewidth=1.35,
                        label=label,
                    )
            axis.set_xscale("log")
            axis.grid(True, which="both", alpha=0.25)
            axis.set_title(f"{scheme} · {'initialization' if role == 'reconstructed_initialization' else 'best validation'}")
            if cosine:
                axis.axhline(0.99, color="black", linewidth=0.8, alpha=0.55)
                axis.set_ylim(-1.05, 1.05)
                if undefined:
                    axis.text(
                        0.02,
                        0.03,
                        f"{undefined} undefined exact-zero",
                        transform=axis.transAxes,
                        fontsize=8,
                    )
            else:
                axis.set_yscale("symlog", linthresh=1.0e-16)
            if row_index == len(SCHEMES) - 1:
                axis.set_xlabel("Injected output current B")
            if column_index == 0:
                axis.set_ylabel(ylabel)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8)
    fig.suptitle(f"Conv3 {ylabel} ({precision}); blank cosine segments remain undefined")
    fig.tight_layout(rect=(0.0, 0.075, 1.0, 0.965))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _write_plots(rows: Sequence[Mapping[str, Any]], output_dir: Path) -> list[Path]:
    outputs: list[Path] = []
    for precision in PRECISIONS:
        specifications = (
            (
                "eqprop_vs_bptt_cosine",
                "EqProp vs BPTT cosine",
                f"cosine_vs_injected_beta_{precision}.png",
                True,
            ),
            (
                "matched_zero_positive_relative_displacement",
                "Matched-zero positive relative displacement",
                f"relative_displacement_vs_injected_beta_{precision}.png",
                False,
            ),
            (
                "matched_zero_positive_delta_rms",
                "Matched-zero positive delta RMS",
                f"delta_rms_vs_injected_beta_{precision}.png",
                False,
            ),
        )
        for field, ylabel, filename, is_cosine in specifications:
            path = output_dir / "plots" / filename
            _plot_metric(
                rows,
                precision=precision,
                field=field,
                ylabel=ylabel,
                output=path,
                cosine=is_cosine,
            )
            outputs.append(path)
    return outputs


def _format_optional(value: Any) -> str:
    if value is None:
        return "undefined"
    return f"{float(value):.6g}"


def _write_report(
    path: Path,
    *,
    plan: Mapping[str, Any],
    row_count: int,
    bundle_count: int,
    undefined_count: int,
    layer_summaries: Sequence[Mapping[str, Any]],
    manifest_study_id: str,
    residual_audit: Mapping[str, Any],
) -> None:
    failed_contexts = [
        row
        for row in residual_audit["context_summaries"]
        if not row["all_gates_passed"]
    ]
    failed_context_text = "; ".join(
        "{scheme}/{role}/{precision}/{layers} ({count} phase rows)".format(
            scheme=row["scheme"],
            role=(
                "init"
                if row["checkpoint_role"] == "reconstructed_initialization"
                else "best"
            ),
            precision=row["precision"],
            layers=",".join(row["failed_state_layers"]),
            count=row["gate_failure_count"],
        )
        for row in failed_contexts
    ) or "none"
    lines = [
        "# Conv3 complete current-nudge EqProp beta sweep",
        "",
        "Evidence tier: exploratory ordinary-MNIST checkpoint replay; not paper-facing training evidence.",
        "",
        f"Validated exactly `{bundle_count}` production bundles and `{row_count}` joined dtype/layer rows at `T=K=64` on fixed batch 0 (16 examples). The primary beta coordinate is the injected output current `effective_beta`, not `beta_hat_requested`.",
        "",
        f"Undefined EqProp/BPTT cosines caused by an exact-zero vector are retained as blank/`null`: `{undefined_count}` rows. They are not reported as zero.",
        "",
        "## Equilibrium residual audit",
        "",
        f"Validated exactly `{residual_audit['row_count']}` layer/phase residual rows: `{residual_audit['row_count_by_variant']['positive_one_sided']}` one-sided and `{residual_audit['row_count_by_variant']['centered']}` centered. Every row covers 16 unique examples, has finite statistics and `outcome=ok`, obeys the strict `selected_max_p90 < 0.01` gate and maximum-based uniform/hard-failure semantics, and has `hard_failure=false`.",
        "",
        f"Residual gate failures are retained as scientific outcomes: `{residual_audit['gate_failure_count']}` rows. Failed dtype/contexts: {failed_context_text}. Float64 all-pass: `{str(residual_audit['all_float64_gates_passed']).lower()}`. Exact coordinates are in `summary.json`.",
        "",
        "## Provenance deviation",
        "",
        f"The runner manifests inherit the generic source-compatible study ID `{manifest_study_id}` rather than the enclosing plan ID `{plan['study_id']}`. The analysis binds and hashes the plan config and launcher explicitly in `summary.json`; this known reporting deviation does not change the scientific coordinates.",
        "",
        "## Layerwise maximum cosine over the tested beta grid",
        "",
        "| Scheme | Checkpoint | Estimator | Precision | Gradient/state | Defined B | Undefined exact-zero | Maximum cosine | Injected B | Relative displacement | Delta RMS |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in layer_summaries:
        checkpoint = "init" if row["checkpoint_role"] == "reconstructed_initialization" else "best"
        estimator = "one-sided" if row["eqprop_variant"] == "positive_one_sided" else "centered"
        lines.append(
            "| {scheme} | {checkpoint} | {estimator} | {precision} | {gradient}/{state} | {defined} | {undefined} | {cosine} | {beta} | {relative} | {rms} |".format(
                scheme=row["scheme"],
                checkpoint=checkpoint,
                estimator=estimator,
                precision=row["precision"],
                gradient=row["gradient_layer"],
                state=row["state_layer"],
                defined=row["defined_beta_count"],
                undefined=row["undefined_exact_zero_count"],
                cosine=_format_optional(row["maximum_cosine"]),
                beta=_format_optional(row["beta_at_maximum_cosine"]),
                relative=_format_optional(row["positive_relative_displacement_at_maximum"]),
                rms=_format_optional(row["positive_delta_rms_at_maximum"]),
            )
        )
    lines.extend(
        [
            "",
            "Full layer-by-layer values, including centered negative-phase displacement, are in `summary.csv`. Plots are under `plots/`.",
            "",
            "Limitations: one fixed ordinary-MNIST validation minibatch, read-only replay only, no EqProp optimizer step, and no official-test read.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(
    *,
    config_path: Path = DEFAULT_CONFIG,
    launcher_path: Path = DEFAULT_LAUNCHER,
    study_root: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    config_path = config_path.expanduser().resolve()
    launcher_path = launcher_path.expanduser().resolve()
    plan = _read_json(config_path)
    _validate_plan(plan)
    if not launcher_path.is_file():
        raise FileNotFoundError(launcher_path)
    configured_root = _resolve_repo_path(plan["execution"]["output_root"])
    study_root = configured_root if study_root is None else study_root.expanduser().resolve()
    output_dir = study_root / "analysis" if output_dir is None else output_dir.expanduser().resolve()
    try:
        output_dir.relative_to(study_root)
    except ValueError as error:
        raise ValueError("Analysis output must remain beneath the study root.") from error
    source_config_path = _resolve_repo_path(plan["source"]["config"])
    source_run = _resolve_repo_path(plan["source"]["run"])
    source_config = _read_json(source_config_path)
    bundles, excluded = _discover_bundles(
        study_root,
        plan=plan,
        source_config=source_config,
        source_config_path=source_config_path,
        source_run=source_run,
    )
    rows = [row for bundle in bundles for row in bundle["rows"]]
    rows.sort(
        key=lambda row: (
            SCHEMES.index(row["scheme"]),
            CHECKPOINT_ROLES.index(row["checkpoint_role"]),
            VARIANTS.index(row["eqprop_variant"]),
            PRECISIONS.index(row["precision"]),
            PARAMETERS.index(row["parameter_name"]),
            int(row["beta_index"]),
        )
    )
    _validate_complete_rows(rows)
    manifest_study_ids = {bundle["manifest_study_id"] for bundle in bundles}
    if len(manifest_study_ids) != 1:
        raise ValueError("Production bundles do not share one inherited manifest study ID.")
    manifest_study_id = next(iter(manifest_study_ids))
    layer_summaries = _layer_summaries(rows)
    context_summaries = _context_summaries(rows)
    residual_audit = _aggregate_residual_audit(bundles)
    undefined_coordinates = [
        {
            "eqprop_variant": row["eqprop_variant"],
            "scheme": row["scheme"],
            "checkpoint_role": row["checkpoint_role"],
            "precision": row["precision"],
            "injected_beta": row["injected_beta"],
            "gradient_layer": row["gradient_layer"],
            "reason": row["cosine_undefined_reason"],
        }
        for row in rows
        if not row["cosine_defined"]
    ]
    summary_csv = output_dir / "summary.csv"
    summary_json = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    _write_csv(summary_csv, rows)
    plot_paths = _write_plots(rows, output_dir)
    summary = {
        "schema": SCHEMA,
        "study_id": plan["study_id"],
        "evidence_tier": plan["evidence_tier"],
        "evidence_class": plan["evidence_class"],
        "primary_beta_coordinate": "effective_beta (injected output current B)",
        "input_contract": {
            "plan_config": {
                "path": str(config_path),
                "sha256": reporting.sha256_file(config_path),
            },
            "launcher": {
                "path": str(launcher_path),
                "sha256": reporting.sha256_file(launcher_path),
            },
            "source_config": {
                "path": str(source_config_path),
                "sha256": reporting.sha256_file(source_config_path),
            },
            "source_run": {
                "path": str(source_run),
                "result_sha256": reporting.sha256_file(source_run / "result.json"),
                "cohort_sha256": reporting.sha256_file(source_run / "cohort.json"),
            },
        },
        "manifest_study_id_deviation": {
            "present": manifest_study_id != plan["study_id"],
            "plan_study_id": plan["study_id"],
            "observed_bundle_manifest_study_id": manifest_study_id,
            "reason": "audit runner inherited its generic source-compatible float64-shadow study ID; plan and launcher are bound here by SHA-256",
        },
        "coverage": {
            "production_bundle_count": len(bundles),
            "expected_production_bundle_count": EXPECTED_PRODUCTION_BUNDLES,
            "logical_checkpoint_context_count": EXPECTED_LOGICAL_CONTEXTS,
            "dtype_context_count": EXPECTED_DTYPE_CONTEXTS,
            "joined_layer_row_count": len(rows),
            "expected_joined_layer_row_count": EXPECTED_SUMMARY_ROWS,
            "undefined_exact_zero_cosine_count": len(undefined_coordinates),
            "equilibrium_residual_row_count": residual_audit["row_count"],
            "expected_equilibrium_residual_row_count": EXPECTED_RESIDUAL_ROWS,
            "equilibrium_residual_gate_failure_count": residual_audit[
                "gate_failure_count"
            ],
            "equilibrium_residual_hard_failure_count": residual_audit[
                "hard_failure_count"
            ],
        },
        "guards": {
            "all_canonical_bundles_validate": True,
            "all_result_status_and_artifact_hashes_validate": True,
            "exact_11_beta_x_2_estimator_bundle_coverage": True,
            "exact_3_scheme_x_2_checkpoint_x_2_dtype_x_4_layer_coverage": True,
            "fixed_cohort_reproduced": True,
            "T_K": [64, 64],
            "bias_gradients_excluded": True,
            "optimizer_steps_applied": False,
            "official_test_read": False,
            "undefined_cosines_preserved": True,
            "positive_endpoint_identical_across_estimators": True,
            "exact_equilibrium_residual_phase_layer_coverage": True,
            "all_equilibrium_residual_rows_cover_16_unique_examples": True,
            "all_equilibrium_residual_metrics_finite": True,
            "all_equilibrium_residual_threshold_semantics_valid": True,
            "all_equilibrium_residual_hard_failures_false": True,
            "all_float64_equilibrium_residual_gates_passed": residual_audit[
                "all_float64_gates_passed"
            ],
        },
        "equilibrium_residual_audit": residual_audit,
        "injected_betas": list(EXPECTED_BETAS),
        "included_runs": [
            {
                key: value
                for key, value in bundle.items()
                if key
                not in {
                    "rows",
                    "residual_gate_failure_coordinates",
                    "residual_context_counts",
                }
            }
            for bundle in sorted(bundles, key=lambda row: (VARIANTS.index(row["variant"]), row["beta_index"]))
        ],
        "excluded_runs": excluded,
        "undefined_cosine_coordinates": undefined_coordinates,
        "layer_summaries": layer_summaries,
        "context_summaries": context_summaries,
        "outputs": {
            "summary_csv": str(summary_csv),
            "summary_json": str(summary_json),
            "report": str(report_path),
            "plots": [str(path) for path in plot_paths],
        },
        "limitations": [
            "Exploratory ordinary-MNIST checkpoint replay; not paper-facing training evidence.",
            "One fixed 16-example validation minibatch; minibatch spread is not measured.",
            "Read-only gradient diagnostic; no EqProp optimizer step or accuracy training sweep.",
        ],
    }
    _write_json(summary_json, summary)
    _write_report(
        report_path,
        plan=plan,
        row_count=len(rows),
        bundle_count=len(bundles),
        undefined_count=len(undefined_coordinates),
        layer_summaries=layer_summaries,
        manifest_study_id=manifest_study_id,
        residual_audit=residual_audit,
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--launcher", type=Path, default=DEFAULT_LAUNCHER)
    parser.add_argument("--study-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = analyze(
        config_path=args.config,
        launcher_path=args.launcher,
        study_root=args.study_root,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "state": "complete",
                "production_bundles": summary["coverage"]["production_bundle_count"],
                "joined_rows": summary["coverage"]["joined_layer_row_count"],
                "undefined_cosines": summary["coverage"]["undefined_exact_zero_cosine_count"],
                "output": summary["outputs"]["summary_json"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
