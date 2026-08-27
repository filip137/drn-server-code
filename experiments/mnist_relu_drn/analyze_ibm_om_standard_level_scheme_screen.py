"""Integrity-checked post-run analysis for the IBM OM standard-level screen.

The screen summary is the immutable input to this module.  The analyzer
rehashes every declared source and sampled population, recomputes the selected
development candidates and held-out aggregates, and writes only to an
explicitly supplied output directory.
"""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import io
from itertools import product
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from experiments.artifacts import atomic_write_json, content_hash, sha256_file
from experiments.mnist_relu_drn.ibm_om_standard_level_scheme_screen import (
    load_standard_level_contract,
)
from training.ibm_reram_hwa import (
    commission_ibm_reram_raw_reset_means,
    load_om_array_population,
)


SOURCE_SCHEMA = "ebl.mnist_relu_drn.ibm_om_standard_level_scheme_screen"
SOURCE_SCHEMA_VERSION = 1
ANALYSIS_SCHEMA = (
    "ebl.mnist_relu_drn.ibm_om_standard_level_scheme_screen.post_run_analysis"
)
ANALYSIS_SCHEMA_VERSION = 1

_ROOT = Path(__file__).resolve().parents[2]
_POPULATION_RECEIPT_SCHEMA = "ebl.ibm_reram.om_array_population_receipt"
_FROZEN_POPULATION_IMPLEMENTATION_SHA256 = (
    "337ff76e0b3e3acda9ab6c1728f35141ef399ab2b8cbc2f57e9ce8d39ceb6206"
)
_FROZEN_SAMPLER_SOURCE_SHA256 = (
    "f938908c70135abe4308e67cb9d366651c81ecef3701b29d42c416e20c02fd2e"
)
_POPULATION_RECEIPT_KEYS = {
    "schema",
    "schema_version",
    "backend",
    "python_executable",
    "python_version",
    "torch_version",
    "aihwkit_version",
    "request",
    "num_cells",
    "population_fingerprint",
    "population_sha256",
    "sampler_source_sha256",
    "population_implementation_sha256",
}
_POPULATION_REQUEST_KEYS = {
    "preset",
    "assignment_seed",
    "corruption_policy",
    "binding_keys",
    "binding_shapes",
    "required_aihwkit_version",
}
_EXPECTED_CALIBRATION_POLICY = {
    "source": "per_scheme_development_assignment",
    "selection_domain": "development_assignment_after_standard_level_mapping",
    "selection_metric": "mapped_accuracy_then_calibrated_kl_then_scale_pair",
    "heldout_application": "freeze_selected_scale_pair_and_gain_per_scheme",
}
_EXPECTED_SCHEME_CONTRACT_V1 = {
    "without_fixed_r_origin": "sampled RESET/lower state",
    "with_fixed_r_origin": "exact intrinsic sampled r",
    "active_level_spacing": "4 * nominal dw_min/2 in x=(a+1)/2",
    "four_fixed_r_positive": "G++=G--=a; G+-=G-+=r",
    "four_fixed_r_negative": "G++=G--=r; G+-=G-+=a",
    "four_edge_transfer_and_loading": "D=G; S=G",
    "eight_edge_transfer": "D=G_a-G_r",
    "eight_edge_loading": "S=G_a+G_r",
}
_EXPECTED_SCHEME_CONTRACT_V2 = {
    **_EXPECTED_SCHEME_CONTRACT_V1,
    "without_fixed_r_origin": (
        "bounded per-cell arithmetic mean of eight sequential apparent raw-a "
        "RESET/read observations"
    ),
    "without_fixed_r_cross_cell_pooling": "none",
    "without_fixed_r_standard_error_guard": 0.0,
}
_EXPECTED_CLAIM_BOUNDARY_V1 = (
    "AIHWKit 1.1.0 normalized OM fitted-model control with repaired "
    "identities and analyst-standard uniform levels separated by four "
    "nominal increments. No stochastic write, program-and-verify, HWA, "
    "training, absolute conductance calibration, or fabricated-device claim."
)
_EXPECTED_CLAIM_BOUNDARY_V2 = (
    "AIHWKit 1.1.0 normalized OM fitted-model control with repaired "
    "identities, per-cell raw-a baselines commissioned from eight stochastic "
    "RESET/read observations in the no-r arms, and analyst-standard uniform "
    "levels separated by four nominal increments. Level deployment itself is "
    "ideal and noiseless. No program-and-verify, HWA, training, absolute "
    "conductance calibration, or fabricated-device claim."
)
_MAIN_LAUNCH_KEYS = {
    "schema_version",
    "evidence_tier",
    "screen_id",
    "started_at_utc",
    "source_commit",
    "launcher",
    "command",
    "expected_coverage",
    "paths",
    "progress_contract",
    "expected_runtime",
    "safe_retry",
}
_REPLAY_LAUNCH_KEYS = {
    "schema_version",
    "evidence_tier",
    "screen_id",
    "started_at_utc",
    "source_commit",
    "launcher",
    "command",
    "expected_coverage",
    "reference_summary",
    "terminal_result",
    "success_criterion",
}
_REPLAY_SUCCESS_CRITERION = (
    "Terminal exit zero, complete declared coverage, and byte-identical "
    "summary SHA-256 to the reference execution."
)
_V2_LAUNCH_COMMON_KEYS = {
    "schema_version",
    "evidence_tier",
    "screen_id",
    "started_at_utc",
    "source_commit",
    "source_context",
    "launcher",
    "command",
    "expected_coverage",
    "population_reuse",
    "paths",
    "progress_contract",
    "expected_runtime",
    "safe_retry",
}
_V2_REPLAY_SUCCESS_CRITERION = (
    "Semantic equivalence of every scientific field, population/commissioning "
    "content hash, target hash, calibration, prediction hash, metric, and "
    "diagnostic after normalizing only result-root-dependent artifact path "
    "strings. Raw summary byte identity is not expected because paths are "
    "serialized."
)
_V2_POPULATION_REUSE = (
    "Byte-identical v1 assignment populations and production sampling receipts "
    "copied before launch so v1-to-v2 changes only the declared no-r "
    "commissioning intervention."
)
_V2_SOURCE_CONTEXT_KEYS = {
    "numerical_source_frozen_in_commit",
    "screen_config_sha256",
    "working_tree_exception",
}
_V2_WORKING_TREE_EXCEPTION = (
    "Only the independent untracked post-run analyzer and its tests were under "
    "review at launch; neither is imported by the numerical screen."
)
_RESET_COMMISSIONING_SCHEMA = (
    "ebl.mnist_relu_drn.ibm_om_per_cell_raw_reset_commissioning"
)
_RESET_COMMISSIONING_RECEIPT_KEYS = {
    "schema",
    "schema_version",
    "algorithm",
    "population_fingerprint",
    "assignment_seed",
    "topology",
    "commissioning_seed",
    "read_samples",
    "reset_pulses_per_cell",
    "total_reset_pulses",
    "baseline_estimator",
    "cross_cell_pooling",
    "standard_error_guard",
    "read_coordinate",
    "reference_consumed_by_commissioner",
    "artifact",
    "artifact_sha256",
    "reset_mean_raw_a_sha256",
    "reset_standard_error_raw_a_sha256",
    "commissioner_report",
}
_RESET_COMMISSIONING_SUMMARY_KEYS = (
    _RESET_COMMISSIONING_RECEIPT_KEYS
    - {"commissioner_report"}
    | {
        "path",
        "sha256",
        "receipt",
        "receipt_sha256",
        "reset_mean_raw_a",
        "reset_standard_error_raw_a",
    }
)
_RESET_COMMISSIONING_ARTIFACT_FIELDS = {
    "schema",
    "schema_version",
    "population_fingerprint",
    "assignment_seed",
    "commissioning_seed",
    "read_samples",
    "binding_keys_json",
    "binding_shapes_json",
    "reset_mean_raw_a",
    "reset_standard_error_raw_a",
}

_SCHEMES: dict[str, tuple[int, bool]] = {
    "four_without_fixed_r": (4, False),
    "four_with_fixed_r": (4, True),
    "eight_without_fixed_r": (8, False),
    "eight_with_fixed_r": (8, True),
}
_SCHEME_ORDER = tuple(_SCHEMES)
_SUMMARY_KEYS = {
    "schema",
    "schema_version",
    "status",
    "screen_id",
    "claim_boundary",
    "source_precision",
    "deployment_precision",
    "screen_config",
    "screen_config_sha256",
    "model_config",
    "model_config_sha256",
    "teacher_weights",
    "teacher_weights_sha256",
    "teacher_architecture",
    "device",
    "sample_limit",
    "development_assignment_seed",
    "heldout_assignment_seeds",
    "calibration_policy",
    "scheme_contract",
    "development_populations",
    "development_calibration",
    "heldout",
    "aggregate",
    "matched_comparisons",
}
_TEST_METRIC_KEYS = {
    "examples",
    "kl_teacher_student",
    "raw_kl_teacher_student",
    "student_accuracy",
    "teacher_accuracy",
    "teacher_agreement",
    "raw_score_rms",
    "calibrated_score_rms",
    "teacher_logit_rms",
    "fixed_logit_gain",
    "prediction_sha256",
    "voltage",
}
_OUTPUT_NAMES = {
    "post_run_analysis.json",
    "accuracy_by_assignment.csv",
    "mechanism_by_scheme_layer.csv",
    "post_run_analysis.md",
}


class AnalysisIntegrityError(ValueError):
    """Raised when a completed screen artifact violates its frozen schema."""


def _fail(message: str) -> None:
    raise AnalysisIntegrityError(message)


def _object(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        _fail(f"Expected {label} to be an object.")
    return value


def _list(value: Any, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        _fail(f"Expected {label} to be a list.")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    if set(value) != expected:
        _fail(
            f"Expected {label} keys to be exactly {sorted(expected)!r}; "
            f"missing={sorted(expected - set(value))!r}, "
            f"unknown={sorted(set(value) - expected)!r}."
        )


def _number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"Expected {label} to be numeric.")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"Expected {label} to be finite.")
    return result


def _integer(value: Any, *, label: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"Expected {label} to be an integer.")
    if minimum is not None and value < minimum:
        _fail(f"Expected {label} to be >= {minimum}.")
    return value


def _probability(value: Any, *, label: str) -> float:
    result = _number(value, label=label)
    if result < 0.0 or result > 1.0:
        _fail(f"Expected {label} to be in [0, 1].")
    return result


def _sha(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"Expected {label} to be a lowercase SHA-256 digest.")
    return value


def _same(left: float, right: float, *, label: str) -> None:
    if not math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-12):
        _fail(f"Expected {label} to match; observed {left!r} != {right!r}.")


def _load_object(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AnalysisIntegrityError(f"Expected readable {label}: {path}.") from error
    return _object(value, label=label)


def canonical_semantic_summary(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return a comparison payload without result-root-dependent metadata.

    Population and receipt digests, fingerprints, device reports, mappings,
    metrics, and every other scientific field remain present.  Only population
    and receipt paths are normalized.  Receipt bytes do not contain a result
    root and therefore must remain byte-identical in a deterministic replay.
    """

    payload = json.loads(json.dumps(value, allow_nan=False))

    def normalize_population(raw: Any) -> None:
        population = _object(raw, label="semantic population record")
        if "path" in population:
            population["path"] = "<result-root>/populations/<population>.npz"
        if "receipt" in population:
            population["receipt"] = (
                "<result-root>/populations/<population>.receipt.json"
            )
        if "reset_baseline_commissioning" in population:
            commissioning = _object(
                population["reset_baseline_commissioning"],
                label="semantic RESET commissioning record",
            )
            commissioning["path"] = (
                "<result-root>/commissioning/<commissioning>.npz"
            )
            commissioning["receipt"] = (
                "<result-root>/commissioning/<commissioning>.receipt.json"
            )
    for population in _list(
        payload.get("development_populations"),
        label="semantic development populations",
    ):
        normalize_population(population)
    for block in _list(payload.get("heldout"), label="semantic heldout blocks"):
        normalize_population(
            _object(block, label="semantic heldout block").get("population")
        )
    return payload


def semantic_summary_sha256(value: Mapping[str, Any]) -> str:
    """Hash all scientific summary fields modulo output-root metadata."""

    return content_hash(canonical_semantic_summary(value))


def _verify_file(path_value: Any, digest_value: Any, *, label: str) -> Path:
    if not isinstance(path_value, str) or not path_value:
        _fail(f"Expected {label} path to be a non-empty string.")
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        _fail(f"Expected {label} file to exist: {path}.")
    expected = _sha(digest_value, label=f"{label} SHA-256")
    observed = sha256_file(path)
    if observed != expected:
        _fail(
            f"Expected {label} SHA-256 {expected}, observed {observed} for {path}."
        )
    return path


def _validate_source_contract(
    screen_config_path: Path, summary: Mapping[str, Any], *, label: str
) -> Any:
    try:
        contract = load_standard_level_contract(screen_config_path)
    except (OSError, ValueError) as error:
        raise AnalysisIntegrityError(
            f"Expected a strict production {label} screen contract."
        ) from error
    contract_version = contract.contract_schema_version
    if contract_version == 1:
        expected_claim_boundary = _EXPECTED_CLAIM_BOUNDARY_V1
        expected_scheme_contract = _EXPECTED_SCHEME_CONTRACT_V1
        expected_reset_samples = None
    elif contract_version == 2:
        expected_claim_boundary = _EXPECTED_CLAIM_BOUNDARY_V2
        expected_scheme_contract = _EXPECTED_SCHEME_CONTRACT_V2
        expected_reset_samples = 8
    else:  # The strict loader should make this unreachable.
        _fail(f"Unsupported {label} screen-contract version {contract_version!r}.")
    if (
        contract.screen_id != summary.get("screen_id")
        or contract.sample_limit is not None
        or contract.development_assignment_seed
        != summary.get("development_assignment_seed")
        or list(contract.heldout_assignment_seeds)
        != summary.get("heldout_assignment_seeds")
        or contract.teacher_sha256 != summary.get("teacher_weights_sha256")
        or contract.model_config_sha256 != summary.get("model_config_sha256")
        or contract.spacing_delta_multiples != 4
        or contract.reset_read_samples != expected_reset_samples
        or contract.continuous_envelope_control is not True
        or summary.get("claim_boundary") != expected_claim_boundary
        or summary.get("source_precision") != "fp32_teacher_and_drn_solver"
        or summary.get("deployment_precision")
        != "uniform_four_delta_standard_level_grid"
        or summary.get("teacher_architecture") != "bias_free_relu_784_50_10"
        or summary.get("device") != "cpu"
        or summary.get("sample_limit") is not None
        or summary.get("calibration_policy") != _EXPECTED_CALIBRATION_POLICY
        or summary.get("scheme_contract") != expected_scheme_contract
    ):
        _fail(f"Expected {label} summary to match the complete frozen source contract.")
    return contract


def _linear_quantile(sorted_values: Sequence[float], quantile: float) -> float:
    position = (len(sorted_values) - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def _float_summary(values: Sequence[float]) -> dict[str, float]:
    if not values or not all(math.isfinite(value) for value in values):
        _fail("Expected a non-empty finite sequence for aggregate recomputation.")
    ordered = sorted(float(value) for value in values)
    return {
        "minimum": ordered[0],
        "p01": _linear_quantile(ordered, 0.01),
        "p10": _linear_quantile(ordered, 0.10),
        "median": _linear_quantile(ordered, 0.50),
        "mean": sum(ordered) / len(ordered),
        "p90": _linear_quantile(ordered, 0.90),
        "p99": _linear_quantile(ordered, 0.99),
        "maximum": ordered[-1],
        "rms": math.sqrt(sum(value * value for value in ordered) / len(ordered)),
    }


def _validate_float_summary(
    observed_value: Any,
    expected: Mapping[str, float],
    *,
    label: str,
) -> None:
    observed = _object(observed_value, label=label)
    _exact_keys(observed, set(expected), label=label)
    for key, expected_value in expected.items():
        _same(_number(observed[key], label=f"{label}.{key}"), expected_value, label=f"{label}.{key}")


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator != 0.0 else None


def _mean(values: Sequence[float]) -> float:
    if not values:
        _fail("Expected at least one value when computing a mean.")
    return sum(values) / len(values)


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _csv_text(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(fields), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _remove_exact_output_directory(path: Path, *, staging: bool) -> None:
    if not path.is_dir() or path.is_symlink():
        _fail(f"Expected a regular analyzer output directory: {path}.")
    entries = list(path.iterdir())
    if not staging and {entry.name for entry in entries} != _OUTPUT_NAMES:
        _fail(
            "Refusing to replace a non-owned output directory; expected exactly "
            f"{sorted(_OUTPUT_NAMES)!r}: {path}."
        )
    if not staging:
        marker = _load_object(path / "post_run_analysis.json", label="existing analysis marker")
        if (
            marker.get("schema") != ANALYSIS_SCHEMA
            or marker.get("schema_version") != ANALYSIS_SCHEMA_VERSION
        ):
            _fail(f"Refusing to replace an output directory without analyzer ownership: {path}.")
    for entry in entries:
        if not entry.is_file() or entry.is_symlink():
            _fail(f"Refusing to remove unexpected analyzer output entry: {entry}.")
        entry.unlink()
    path.rmdir()


def _publish_staged_output(staging: Path, destination: Path) -> None:
    """Publish one complete directory, rolling back a prior owned bundle on error."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    backup: Path | None = None
    if destination.exists() or destination.is_symlink():
        if not destination.is_dir() or destination.is_symlink():
            _fail(f"Refusing to replace a non-directory output destination: {destination}.")
        # Validate ownership before moving anything.
        entries = list(destination.iterdir())
        if not entries:
            destination.rmdir()
        elif {entry.name for entry in entries} != _OUTPUT_NAMES:
            _fail(
                "Refusing to replace a non-owned output directory; expected exactly "
                f"{sorted(_OUTPUT_NAMES)!r}: {destination}."
            )
        else:
            marker = _load_object(
                destination / "post_run_analysis.json",
                label="existing analysis marker",
            )
            if (
                marker.get("schema") != ANALYSIS_SCHEMA
                or marker.get("schema_version") != ANALYSIS_SCHEMA_VERSION
            ):
                _fail(
                    "Refusing to replace an output directory without analyzer "
                    f"ownership: {destination}."
                )
            reserved = Path(
                tempfile.mkdtemp(
                    dir=destination.parent,
                    prefix=f".{destination.name}.previous.",
                )
            )
            reserved.rmdir()
            backup = reserved
            os.replace(destination, backup)
    try:
        os.replace(staging, destination)
    except BaseException:
        if backup is not None and backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise
    if backup is not None:
        _remove_exact_output_directory(backup, staging=False)


def _validate_test_metrics(value: Any, *, label: str, expected_gain: float) -> Mapping[str, Any]:
    metrics = _object(value, label=label)
    _exact_keys(metrics, _TEST_METRIC_KEYS, label=label)
    if _integer(metrics["examples"], label=f"{label}.examples", minimum=1) != 10_000:
        _fail(f"Expected {label} to cover the complete 10,000-example MNIST test set.")
    for name in (
        "student_accuracy",
        "teacher_accuracy",
        "teacher_agreement",
    ):
        _probability(metrics[name], label=f"{label}.{name}")
    for name in (
        "kl_teacher_student",
        "raw_kl_teacher_student",
        "raw_score_rms",
        "calibrated_score_rms",
        "teacher_logit_rms",
    ):
        if _number(metrics[name], label=f"{label}.{name}") < 0.0:
            _fail(f"Expected {label}.{name} to be nonnegative.")
    _same(
        _number(metrics["fixed_logit_gain"], label=f"{label}.fixed_logit_gain"),
        expected_gain,
        label=f"{label}.fixed_logit_gain",
    )
    _sha(metrics["prediction_sha256"], label=f"{label}.prediction_sha256")
    voltage = _list(metrics["voltage"], label=f"{label}.voltage")
    if len(voltage) != 3:
        _fail(f"Expected {label}.voltage to contain the three DRN state layers.")
    for layer_index, raw_layer in enumerate(voltage):
        layer = _object(raw_layer, label=f"{label}.voltage[{layer_index}]")
        if _integer(layer.get("layer"), label=f"{label}.voltage[{layer_index}].layer") != layer_index:
            _fail(f"Expected ordered voltage layers in {label}.")
        for name in ("mean", "rms", "standard_deviation", "minimum", "maximum"):
            _number(layer.get(name), label=f"{label}.voltage[{layer_index}].{name}")
        _integer(layer.get("values"), label=f"{label}.voltage[{layer_index}].values", minimum=1)
    return metrics


def _tensor_sha256(value: torch.Tensor) -> str:
    cpu = value.detach().contiguous().cpu()
    digest = sha256()
    digest.update(str(cpu.dtype).encode("utf-8"))
    digest.update(json.dumps(list(cpu.shape), separators=(",", ":")).encode("utf-8"))
    digest.update(cpu.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _tensor_float_summary(value: torch.Tensor) -> dict[str, float]:
    flat = value.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    if flat.numel() == 0 or not bool(torch.isfinite(flat).all()):
        _fail("Expected a non-empty finite commissioning tensor.")
    prior_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(4)
        quantiles = torch.quantile(
            flat,
            torch.tensor((0.01, 0.1, 0.5, 0.9, 0.99), dtype=torch.float64),
        )
        return {
            "minimum": float(flat.min().item()),
            "p01": float(quantiles[0].item()),
            "p10": float(quantiles[1].item()),
            "median": float(quantiles[2].item()),
            "mean": float(flat.mean().item()),
            "p90": float(quantiles[3].item()),
            "p99": float(quantiles[4].item()),
            "maximum": float(flat.max().item()),
            "rms": float(flat.square().mean().sqrt().item()),
        }
    finally:
        torch.set_num_threads(prior_threads)


def _load_reset_commissioning_artifact(path: Path) -> dict[str, Any]:
    try:
        with np.load(path, allow_pickle=False) as source:
            if set(source.files) != _RESET_COMMISSIONING_ARTIFACT_FIELDS:
                _fail("Expected exact RESET commissioning NPZ fields.")
            values = {name: source[name].copy() for name in source.files}
    except AnalysisIntegrityError:
        raise
    except (OSError, ValueError) as error:
        raise AnalysisIntegrityError(
            f"Expected a readable RESET commissioning artifact: {path}."
        ) from error

    def scalar(name: str) -> Any:
        value = values[name]
        if value.shape != () or value.dtype.hasobject:
            _fail(f"Expected scalar non-object RESET commissioning field {name!r}.")
        return value.item()

    for name in (
        "schema_version",
        "assignment_seed",
        "commissioning_seed",
        "read_samples",
    ):
        if values[name].dtype != np.dtype(np.int64):
            _fail(f"Expected int64 RESET commissioning field {name!r}.")
    for name in (
        "schema",
        "population_fingerprint",
        "binding_keys_json",
        "binding_shapes_json",
    ):
        if values[name].dtype.kind not in {"U", "S"}:
            _fail(f"Expected string RESET commissioning field {name!r}.")
    if (
        str(scalar("schema")) != _RESET_COMMISSIONING_SCHEMA
        or int(scalar("schema_version")) != 1
    ):
        _fail("Expected RESET commissioning artifact schema version 1.")
    mean = values["reset_mean_raw_a"]
    standard_error = values["reset_standard_error_raw_a"]
    for name, tensor in (
        ("reset_mean_raw_a", mean),
        ("reset_standard_error_raw_a", standard_error),
    ):
        if (
            tensor.dtype != np.dtype(np.float32)
            or tensor.ndim != 1
            or tensor.size < 1
            or not bool(np.isfinite(tensor).all())
        ):
            _fail(f"Expected finite one-dimensional float32 RESET field {name!r}.")
    if mean.shape != standard_error.shape or bool(np.any(standard_error < 0.0)):
        _fail("Expected matching RESET mean/SE vectors and nonnegative SE.")
    try:
        binding_keys = json.loads(str(scalar("binding_keys_json")))
        binding_shapes = json.loads(str(scalar("binding_shapes_json")))
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise AnalysisIntegrityError(
            "Expected canonical RESET commissioning binding metadata."
        ) from error
    if (
        not isinstance(binding_keys, list)
        or not binding_keys
        or any(not isinstance(key, str) or not key for key in binding_keys)
        or len(set(binding_keys)) != len(binding_keys)
        or not isinstance(binding_shapes, list)
        or len(binding_shapes) != len(binding_keys)
        or any(
            not isinstance(shape, list)
            or not shape
            or any(
                isinstance(dimension, bool)
                or not isinstance(dimension, int)
                or dimension < 1
                for dimension in shape
            )
            for shape in binding_shapes
        )
    ):
        _fail("Expected canonical RESET commissioning binding keys and shapes.")
    shapes = tuple(tuple(shape) for shape in binding_shapes)
    if sum(math.prod(shape) for shape in shapes) != int(mean.size):
        _fail("Expected RESET commissioning binding shapes to cover every cell.")
    return {
        "population_fingerprint": str(scalar("population_fingerprint")),
        "assignment_seed": int(scalar("assignment_seed")),
        "commissioning_seed": int(scalar("commissioning_seed")),
        "read_samples": int(scalar("read_samples")),
        "binding_keys": tuple(binding_keys),
        "binding_shapes": shapes,
        "binding_keys_json": str(scalar("binding_keys_json")),
        "binding_shapes_json": str(scalar("binding_shapes_json")),
        "reset_mean_raw_a": torch.from_numpy(mean),
        "reset_standard_error_raw_a": torch.from_numpy(standard_error),
    }


def _validate_v2_grid_linkage(
    record: Mapping[str, Any],
    *,
    population: Any,
    topology: int,
    reset_mean_raw_a: torch.Tensor,
) -> dict[str, Any]:
    grids = _object(record.get("standard_level_grids"), label="population standard-level grids")
    no_r_scheme = f"{'four' if topology == 4 else 'eight'}_without_fixed_r"
    with_r_scheme = f"{'four' if topology == 4 else 'eight'}_with_fixed_r"
    _exact_keys(grids, {no_r_scheme, with_r_scheme}, label="population standard-level grids")
    no_r_grid = _object(grids[no_r_scheme], label=f"grid.{no_r_scheme}")
    with_r_grid = _object(grids[with_r_scheme], label=f"grid.{with_r_scheme}")
    if (
        no_r_grid.get("policy")
        != "bounded_per_cell_mean_apparent_raw_a_reset_origin"
        or no_r_grid.get("use_fixed_reference") is not False
        or with_r_grid.get("policy")
        != "exact_sampled_reference_center_with_active_a_projection_only"
        or with_r_grid.get("use_fixed_reference") is not True
        or with_r_grid.get("reset_mean_commissioning") is not None
    ):
        _fail("Expected the v2 no-r RESET-mean and fixed-r grid policies.")
    for grid_name, grid in ((no_r_scheme, no_r_grid), (with_r_scheme, with_r_grid)):
        if (
            _integer(
                grid.get("spacing_delta_multiples"),
                label=f"grid.{grid_name}.spacing_delta_multiples",
                minimum=1,
            )
            != 4
        ):
            _fail("Expected every v2 grid to retain four-delta spacing.")
        _same(
            _number(grid.get("native_delta"), label=f"grid.{grid_name}.native_delta"),
            float(population.nominal_dw_min),
            label=f"grid.{grid_name}.native_delta",
        )
        _same(
            _number(
                grid.get("unit_coordinate_spacing"),
                label=f"grid.{grid_name}.unit_coordinate_spacing",
            ),
            2.0 * float(population.nominal_dw_min),
            label=f"grid.{grid_name}.unit_coordinate_spacing",
        )

    reset_mean = reset_mean_raw_a.detach().to(device="cpu", dtype=torch.float64)
    lower = ((population.min_bound.detach().cpu().to(torch.float64) + 1.0) / 2.0).clamp(0.0, 1.0)
    upper = ((population.max_bound.detach().cpu().to(torch.float64) + 1.0) / 2.0).clamp(0.0, 1.0)
    before_bounds = (reset_mean + 1.0) / 2.0
    public_origin = before_bounds.clamp(0.0, 1.0)
    after_bounds = torch.minimum(torch.maximum(public_origin, lower), upper)
    projection = after_bounds - before_bounds
    projected = torch.abs(projection) > 1e-12
    expected_reset_report = {
        "apparent_raw_a_mean": _tensor_float_summary(reset_mean),
        "mapped_mean_before_bounds": _tensor_float_summary(before_bounds),
        "mapped_mean_after_bounds": _tensor_float_summary(after_bounds),
        "mapped_mean_projection": _tensor_float_summary(projection),
        "projected_cell_count": int(projected.sum().item()),
        "projected_cell_fraction": float(projected.to(torch.float64).mean().item()),
        "hashes": {
            "apparent_raw_a_mean": _tensor_sha256(reset_mean.to(torch.float32)),
            "mapped_mean_before_bounds": _tensor_sha256(before_bounds.to(torch.float32)),
            "mapped_mean_after_bounds": _tensor_sha256(after_bounds.to(torch.float32)),
        },
    }
    if no_r_grid.get("reset_mean_commissioning") != expected_reset_report:
        _fail("Expected the no-r grid to link exactly to the per-cell RESET means and sampled bounds.")
    grid_hashes = _object(no_r_grid.get("hashes"), label=f"grid.{no_r_scheme}.hashes")
    expected_origin_hash = expected_reset_report["hashes"]["mapped_mean_after_bounds"]
    if (
        grid_hashes.get("origin") != expected_origin_hash
        or grid_hashes.get("active_zero") != expected_origin_hash
    ):
        _fail("Expected no-r grid origin/active-zero hashes to link to bounded RESET means.")
    expected_origin_summary = _tensor_float_summary(after_bounds.to(torch.float32))
    if (
        no_r_grid.get("origin") != expected_origin_summary
        or no_r_grid.get("active_zero") != expected_origin_summary
    ):
        _fail("Expected no-r grid origin summaries to match the bounded RESET means.")
    return {
        "mapped_mean_before_bounds": expected_reset_report["mapped_mean_before_bounds"],
        "mapped_mean_after_bounds": expected_reset_report["mapped_mean_after_bounds"],
        "mapped_mean_projection": expected_reset_report["mapped_mean_projection"],
        "projected_cell_count": expected_reset_report["projected_cell_count"],
        "projected_cell_fraction": expected_reset_report["projected_cell_fraction"],
        "grid_hashes": expected_reset_report["hashes"],
    }


def _validate_v2_commissioning(
    record: Mapping[str, Any],
    *,
    population: Any,
    topology: int,
    seed: int,
    result_root: Path,
) -> dict[str, Any]:
    report = _object(
        record.get("reset_baseline_commissioning"),
        label=f"RESET commissioning topology={topology} seed={seed}",
    )
    _exact_keys(
        report,
        _RESET_COMMISSIONING_SUMMARY_KEYS,
        label=f"RESET commissioning topology={topology} seed={seed}",
    )
    stem = f"{topology}-device-assignment-{seed}"
    artifact_path = _verify_file(report.get("path"), report.get("sha256"), label="RESET commissioning artifact")
    receipt_path = _verify_file(
        report.get("receipt"),
        report.get("receipt_sha256"),
        label="RESET commissioning receipt",
    )
    expected_directory = (result_root / "commissioning").resolve()
    if (
        artifact_path != expected_directory / f"{stem}.npz"
        or receipt_path != expected_directory / f"{stem}.receipt.json"
        or report.get("path") != str(artifact_path)
        or report.get("receipt") != str(receipt_path)
    ):
        _fail("Expected exact topology/assignment commissioning artifact paths.")
    stored = _load_reset_commissioning_artifact(artifact_path)
    # The frozen production command used four Torch CPU threads.  Tensor bytes
    # are thread-count invariant here, but the receipt's float32 diagnostic
    # reduction is not bitwise invariant across reduction trees.
    prior_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(4)
        expected = commission_ibm_reram_raw_reset_means(population, read_samples=8)
    finally:
        torch.set_num_threads(prior_threads)
    canonical_keys_json = json.dumps(list(population.binding_keys), separators=(",", ":"))
    canonical_shapes_json = json.dumps(
        [list(shape) for shape in population.binding_shapes], separators=(",", ":")
    )
    if (
        stored["population_fingerprint"] != population.fingerprint
        or stored["assignment_seed"] != seed
        or stored["commissioning_seed"] != expected.commissioning_seed
        or stored["read_samples"] != 8
        or stored["binding_keys"] != population.binding_keys
        or stored["binding_shapes"] != population.binding_shapes
        or stored["binding_keys_json"] != canonical_keys_json
        or stored["binding_shapes_json"] != canonical_shapes_json
        or not torch.equal(stored["reset_mean_raw_a"], expected.reset_mean_raw_a)
        or not torch.equal(
            stored["reset_standard_error_raw_a"],
            expected.reset_standard_error_raw_a,
        )
    ):
        _fail("Expected RESET commissioning NPZ to replay exactly from its population.")
    receipt = _load_object(receipt_path, label="RESET commissioning receipt")
    _exact_keys(
        receipt,
        _RESET_COMMISSIONING_RECEIPT_KEYS,
        label="RESET commissioning receipt",
    )
    artifact_sha = sha256_file(artifact_path)
    expected_receipt = {
        "schema": _RESET_COMMISSIONING_SCHEMA,
        "schema_version": 1,
        "algorithm": "sequential_reset_pulse_read_per_cell_mean_raw_a",
        "population_fingerprint": population.fingerprint,
        "assignment_seed": seed,
        "topology": topology,
        "commissioning_seed": expected.commissioning_seed,
        "read_samples": 8,
        "reset_pulses_per_cell": 8,
        "total_reset_pulses": population.size * 8,
        "baseline_estimator": "per_cell_arithmetic_mean",
        "cross_cell_pooling": "none",
        "standard_error_guard": 0.0,
        "read_coordinate": "apparent_raw_active_a",
        "reference_consumed_by_commissioner": False,
        "artifact": artifact_path.name,
        "artifact_sha256": artifact_sha,
        "reset_mean_raw_a_sha256": _tensor_sha256(expected.reset_mean_raw_a),
        "reset_standard_error_raw_a_sha256": _tensor_sha256(
            expected.reset_standard_error_raw_a
        ),
        "commissioner_report": expected.report,
    }
    if receipt != expected_receipt:
        _fail("Expected the exact deterministic RESET commissioning receipt.")
    expected_summary = {
        "path": str(artifact_path),
        "sha256": artifact_sha,
        "receipt": str(receipt_path),
        "receipt_sha256": sha256_file(receipt_path),
        **{
            key: value
            for key, value in expected_receipt.items()
            if key != "commissioner_report"
        },
        "reset_mean_raw_a": _tensor_float_summary(expected.reset_mean_raw_a),
        "reset_standard_error_raw_a": _tensor_float_summary(
            expected.reset_standard_error_raw_a
        ),
    }
    if report != expected_summary:
        _fail("Expected summary commissioning metadata to match its exact receipt and NPZ.")
    grid = _validate_v2_grid_linkage(
        record,
        population=population,
        topology=topology,
        reset_mean_raw_a=expected.reset_mean_raw_a,
    )
    return {
        "topology": topology,
        "assignment_seed": seed,
        "cells": population.size,
        "commissioning_seed": expected.commissioning_seed,
        "read_samples": 8,
        "reset_mean_raw_a": expected_summary["reset_mean_raw_a"],
        "reset_standard_error_raw_a": expected_summary[
            "reset_standard_error_raw_a"
        ],
        **grid,
        "artifact_sha256": artifact_sha,
        "receipt_sha256": expected_summary["receipt_sha256"],
    }


def _validate_population(
    raw: Any,
    *,
    expected_topology: int,
    expected_seed: int,
    contract_version: int,
    result_root: Path,
) -> tuple[tuple[int, int], dict[str, Any]]:
    record = _object(raw, label=f"population topology={expected_topology} seed={expected_seed}")
    topology = _integer(record.get("topology"), label="population.topology")
    seed = _integer(record.get("assignment_seed"), label="population.assignment_seed")
    if topology != expected_topology or seed != expected_seed:
        _fail("Expected population topology and assignment seed to match its placement.")
    population_path = _verify_file(record.get("path"), record.get("sha256"), label="population")
    receipt_path = _verify_file(
        record.get("receipt"), record.get("receipt_sha256"), label="population receipt"
    )
    fingerprint = _sha(
        record.get("population_fingerprint"), label="population fingerprint"
    )
    if (
        _integer(
            record.get("final_corrupt_cells"),
            label="population.final_corrupt_cells",
            minimum=0,
        )
        != 0
    ):
        _fail("Expected the declared counterfactual-repaired population to contain no final corrupt cells.")
    receipt = _load_object(receipt_path, label="population receipt")
    _exact_keys(receipt, _POPULATION_RECEIPT_KEYS, label="population receipt")
    request = _object(receipt.get("request"), label="population receipt request")
    _exact_keys(request, _POPULATION_REQUEST_KEYS, label="population receipt request")
    expected_keys = (
        ("base.dense_weight.0", "base.dense_weight.1")
        if expected_topology == 4
        else (
            "base.conductance_plus.0",
            "base.conductance_minus.0",
            "base.conductance_plus.1",
            "base.conductance_minus.1",
        )
    )
    try:
        population = load_om_array_population(population_path)
    except ValueError as error:
        raise AnalysisIntegrityError(
            f"Expected an independently valid OM population: {population_path}."
        ) from error
    expected_population_sha = sha256_file(population_path)
    expected_receipt_values = {
        "schema": _POPULATION_RECEIPT_SCHEMA,
        "schema_version": 1,
        "backend": "external_pinned_aihwkit_python",
        "aihwkit_version": "1.1.0",
        "population_sha256": expected_population_sha,
        "population_fingerprint": population.fingerprint,
        "num_cells": population.size,
        "sampler_source_sha256": _FROZEN_SAMPLER_SOURCE_SHA256,
        "population_implementation_sha256": (
            _FROZEN_POPULATION_IMPLEMENTATION_SHA256
        ),
    }
    for name, expected_value in expected_receipt_values.items():
        if receipt.get(name) != expected_value:
            _fail(f"Expected production population receipt field {name!r} to match.")
    for name in ("python_executable", "python_version", "torch_version"):
        if not isinstance(receipt.get(name), str) or not receipt[name]:
            _fail(f"Expected non-empty population receipt field {name!r}.")
    if (
        request.get("assignment_seed") != seed
        or request.get("corruption_policy") != "counterfactual_repaired"
        or request.get("preset") != "reram_array_om"
        or request.get("required_aihwkit_version") != "1.1.0"
        or tuple(request.get("binding_keys", ())) != expected_keys
        or tuple(tuple(shape) for shape in request.get("binding_shapes", ()))
        != population.binding_shapes
        or population.binding_keys != expected_keys
        or population.assignment_seed != seed
        or population.corruption_policy != "counterfactual_repaired"
        or population.fingerprint != fingerprint
    ):
        _fail("Expected population receipt request to match the frozen OM contract.")
    if _integer(record.get("cells"), label="population.cells", minimum=1) != population.size:
        _fail("Expected population cell count to match the loaded population.")
    if (
        _integer(
            record.get("published_corrupt_cells_repaired"),
            label="population.published_corrupt_cells_repaired",
            minimum=0,
        )
        != int(population.published_corrupt.sum().item())
        or int(population.corrupt.sum().item()) != 0
    ):
        _fail("Expected repaired and final corrupt counts to match the population.")
    if contract_version == 1:
        noise_fields = (
            ("declared_dw_min_std_but_disabled", "dw_min_std"),
            ("declared_write_noise_std_but_disabled", "write_noise_std"),
        )
        commissioning = None
    elif contract_version == 2:
        if (
            "declared_dw_min_std_but_disabled" in record
            or "declared_write_noise_std_but_disabled" in record
            or record.get("noise_application")
            != {
                "reset_commissioning": (
                    "preset cycle-to-cycle and apparent write noise enabled"
                ),
                "ideal_level_deployment": "both disabled",
            }
        ):
            _fail("Expected the renamed v2 population noise fields and application contract.")
        noise_fields = (
            ("declared_dw_min_std", "dw_min_std"),
            ("declared_write_noise_std", "write_noise_std"),
        )
        commissioning = _validate_v2_commissioning(
            record,
            population=population,
            topology=topology,
            seed=seed,
            result_root=result_root,
        )
    else:
        _fail(f"Unsupported population contract version {contract_version!r}.")
    for report_name, population_name in (
        ("nominal_dw_min", "nominal_dw_min"),
        *noise_fields,
    ):
        _same(
            _number(record.get(report_name), label=f"population.{report_name}"),
            float(getattr(population, population_name)),
            label=f"population.{report_name}",
        )
    return (topology, seed), {
        "population_sha256": expected_population_sha,
        "receipt_sha256": sha256_file(receipt_path),
        "population_fingerprint": fingerprint,
        "commissioning": commissioning,
    }


def _declared_path(value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        _fail(f"Expected {label} to be a non-empty path string.")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else _ROOT / path).resolve()


def _validate_v2_launch_contract(
    path: Path,
    *,
    summary_path: Path,
    screen_id: str,
    screen_config_sha256: str,
    replay: bool,
    expected_source_commit: str | None = None,
) -> tuple[Mapping[str, Any], str]:
    label = "v2 replay launch contract" if replay else "v2 main launch contract"
    launch = _load_object(path, label=label)
    expected_keys = set(_V2_LAUNCH_COMMON_KEYS)
    if replay:
        expected_keys.add("replay_success_criterion")
    _exact_keys(launch, expected_keys, label=label)
    expected_coverage = {
        "development_assignments": [86001],
        "heldout_assignments": [87001, 87002, 87003],
        "schemes": list(_SCHEME_ORDER),
        "development_calibrations": 4,
        "heldout_arm_reports": 12,
        "population_receipts": 8,
        "reset_commissioning_receipts": 8,
        "test_examples_per_arm": 10_000,
    }
    source_commit = launch.get("source_commit")
    source_context = _object(launch.get("source_context"), label=f"{label} source context")
    _exact_keys(source_context, _V2_SOURCE_CONTEXT_KEYS, label=f"{label} source context")
    launcher = _object(launch.get("launcher"), label=f"{label} launcher")
    _exact_keys(launcher, {"type", "handle"}, label=f"{label} launcher")
    paths = _object(launch.get("paths"), label=f"{label} paths")
    _exact_keys(
        paths,
        {"result_root", "log", "exit_code", "runtime_contract", "terminal_result"},
        label=f"{label} paths",
    )
    result_root = summary_path.parent.parent
    expected_paths = {
        "result_root": result_root,
        "log": result_root / "run.log",
        "exit_code": result_root / "exit_code.txt",
        "runtime_contract": result_root / "screen_contract.json",
        "terminal_result": summary_path,
    }
    if any(
        _declared_path(paths.get(name), label=f"{label} {name}") != expected
        for name, expected in expected_paths.items()
    ):
        _fail(f"Expected {label} paths to identify its exact result bundle.")
    exit_code_path = expected_paths["exit_code"]
    if not exit_code_path.is_file() or exit_code_path.read_text(encoding="utf-8").strip() != "0":
        _fail(f"Expected {label} to have terminal exit code zero.")
    if not expected_paths["log"].is_file() or not expected_paths["runtime_contract"].is_file():
        _fail(f"Expected {label} runtime contract and log to exist.")
    if (
        launch.get("schema_version") != 1
        or launch.get("evidence_tier") != "exploratory_noncanonical"
        or launch.get("screen_id") != screen_id
        or not isinstance(source_commit, str)
        or len(source_commit) != 40
        or any(character not in "0123456789abcdef" for character in source_commit)
        or (expected_source_commit is not None and source_commit != expected_source_commit)
        or launch.get("expected_coverage") != expected_coverage
        or launch.get("population_reuse") != _V2_POPULATION_REUSE
        or source_context
        != {
            "numerical_source_frozen_in_commit": True,
            "screen_config_sha256": screen_config_sha256,
            "working_tree_exception": _V2_WORKING_TREE_EXCEPTION,
        }
        or launcher.get("type") != "local_tmux"
        or not isinstance(launcher.get("handle"), str)
        or not launcher["handle"]
        or not isinstance(launch.get("started_at_utc"), str)
        or not launch["started_at_utc"]
        or any(
            not isinstance(launch.get(name), str) or not launch[name]
            for name in ("command", "progress_contract", "expected_runtime", "safe_retry")
        )
        or (
            replay
            and launch.get("replay_success_criterion")
            != _V2_REPLAY_SUCCESS_CRITERION
        )
    ):
        _fail(f"Expected {label} to match the complete frozen v2 launch contract.")
    return launch, str(source_commit)


def _launch_contract_adjudication(
    *,
    source_summary: Path,
    source_summary_sha256: str,
    screen_id: str,
    contract_version: int,
    screen_config_sha256: str,
    comparison_summary: Path | None,
    comparison_summary_sha256: str | None,
    semantic_equivalence: bool | None,
) -> dict[str, Any]:
    main_path = source_summary.parent.parent / "launch_contract.json"
    if contract_version == 2:
        _main, source_commit = _validate_v2_launch_contract(
            main_path,
            summary_path=source_summary,
            screen_id=screen_id,
            screen_config_sha256=screen_config_sha256,
            replay=False,
        )
        result: dict[str, Any] = {
            "main": {
                "path": str(main_path),
                "sha256": sha256_file(main_path),
                "source_commit": source_commit,
                "declared_coverage_verified": True,
                "summary_sha256": source_summary_sha256,
            },
            "replay": None,
        }
        if comparison_summary is None:
            return result
        if comparison_summary_sha256 is None or semantic_equivalence is None:
            _fail("Expected complete v2 replay adjudication inputs.")
        replay_path = comparison_summary.parent.parent / "launch_contract.json"
        _replay, _source_commit = _validate_v2_launch_contract(
            replay_path,
            summary_path=comparison_summary,
            screen_id=screen_id,
            screen_config_sha256=screen_config_sha256,
            replay=True,
            expected_source_commit=source_commit,
        )
        byte_identical = comparison_summary_sha256 == source_summary_sha256
        result["replay"] = {
            "path": str(replay_path),
            "sha256": sha256_file(replay_path),
            "source_commit": source_commit,
            "summary_sha256": comparison_summary_sha256,
            "declared_success_criterion": _V2_REPLAY_SUCCESS_CRITERION,
            "declared_byte_identity_required": False,
            "observed_byte_identity": byte_identical,
            "semantic_equivalence_satisfied": semantic_equivalence,
            "declared_replay_success": semantic_equivalence,
            "adjudication": (
                "declared_semantic_equivalence_passed"
                if semantic_equivalence
                else "declared_semantic_equivalence_failed"
            ),
        }
        return result
    if contract_version != 1:
        _fail(f"Unsupported launch-contract version {contract_version!r}.")
    main = _load_object(main_path, label="main launch contract")
    _exact_keys(main, _MAIN_LAUNCH_KEYS, label="main launch contract")
    expected_coverage = {
        "development_assignments": [86001],
        "heldout_assignments": [87001, 87002, 87003],
        "schemes": list(_SCHEME_ORDER),
        "development_calibrations": 4,
        "heldout_arm_reports": 12,
        "population_receipts": 8,
    }
    source_commit = main.get("source_commit")
    main_paths = _object(main.get("paths"), label="main launch paths")
    if (
        main.get("schema_version") != 1
        or main.get("evidence_tier") != "exploratory_noncanonical"
        or main.get("screen_id") != screen_id
        or not isinstance(source_commit, str)
        or len(source_commit) != 40
        or any(character not in "0123456789abcdef" for character in source_commit)
        or main.get("expected_coverage") != expected_coverage
        or _declared_path(
            main_paths.get("terminal_result"), label="main terminal result"
        )
        != source_summary
    ):
        _fail("Expected the main launch contract to match the completed screen.")
    result: dict[str, Any] = {
        "main": {
            "path": str(main_path),
            "sha256": sha256_file(main_path),
            "source_commit": source_commit,
            "declared_coverage_verified": True,
            "summary_sha256": source_summary_sha256,
        },
        "replay": None,
    }
    if comparison_summary is None:
        return result
    if comparison_summary_sha256 is None or semantic_equivalence is None:
        _fail("Expected complete replay adjudication inputs.")
    replay_path = comparison_summary.parent.parent / "launch_contract.json"
    replay = _load_object(replay_path, label="replay launch contract")
    _exact_keys(replay, _REPLAY_LAUNCH_KEYS, label="replay launch contract")
    byte_identical = comparison_summary_sha256 == source_summary_sha256
    if (
        replay.get("schema_version") != 1
        or replay.get("evidence_tier")
        != "exploratory_noncanonical_deterministic_replay"
        or replay.get("screen_id") != screen_id
        or replay.get("source_commit") != source_commit
        or replay.get("expected_coverage")
        != {
            "development_calibrations": 4,
            "heldout_arm_reports": 12,
            "population_receipts": 8,
        }
        or replay.get("success_criterion") != _REPLAY_SUCCESS_CRITERION
        or _declared_path(
            replay.get("reference_summary"), label="replay reference summary"
        )
        != source_summary
        or _declared_path(
            replay.get("terminal_result"), label="replay terminal result"
        )
        != comparison_summary
    ):
        _fail("Expected the replay launch contract to match its declared comparison.")
    result["replay"] = {
        "path": str(replay_path),
        "sha256": sha256_file(replay_path),
        "source_commit": source_commit,
        "summary_sha256": comparison_summary_sha256,
        "declared_success_criterion": _REPLAY_SUCCESS_CRITERION,
        "declared_byte_identity_satisfied": byte_identical,
        "semantic_equivalence_satisfied": semantic_equivalence,
        "declared_replay_success": byte_identical,
        "adjudication": (
            "declared_byte_identity_and_semantic_equivalence_passed"
            if byte_identical and semantic_equivalence
            else "declared_byte_identity_failed_semantic_equivalence_passed"
            if semantic_equivalence
            else "semantic_equivalence_failed"
        ),
    }
    return result


def _validate_comparison_provenance(
    summary_path: Path, summary: Mapping[str, Any]
) -> dict[str, Any]:
    """Verify replay-owned sources, contract, and all population receipts."""

    screen_config_path = _verify_file(
        summary.get("screen_config"),
        summary.get("screen_config_sha256"),
        label="comparison screen config",
    )
    _verify_file(
        summary.get("model_config"),
        summary.get("model_config_sha256"),
        label="comparison model config",
    )
    _verify_file(
        summary.get("teacher_weights"),
        summary.get("teacher_weights_sha256"),
        label="comparison teacher weights",
    )
    comparison_contract = _validate_source_contract(
        screen_config_path, summary, label="comparison"
    )
    contract_version = comparison_contract.contract_schema_version
    result_root = summary_path.parent.parent
    screen_config = comparison_contract.raw
    contract = _load_object(
        summary_path.parent.parent / "screen_contract.json",
        label="comparison screen contract",
    )
    _exact_keys(
        contract,
        {
            "screen_config",
            "screen_config_sha256",
            "model_config",
            "model_config_sha256",
            "teacher_weights",
            "teacher_weights_sha256",
            "aihwkit_python",
            "device",
            "sample_limit",
            "contract",
        },
        label="comparison screen contract",
    )
    for name in (
        "screen_config",
        "screen_config_sha256",
        "model_config",
        "model_config_sha256",
        "teacher_weights",
        "teacher_weights_sha256",
        "device",
        "sample_limit",
    ):
        if contract.get(name) != summary.get(name):
            _fail(f"Expected comparison screen contract {name} to match its summary.")
    if contract.get("contract") != screen_config:
        _fail("Expected comparison screen contract to embed its exact screen config.")

    development_seed = _integer(
        summary.get("development_assignment_seed"),
        label="comparison development assignment seed",
    )
    heldout_seeds = [
        _integer(value, label="comparison held-out assignment seed")
        for value in _list(
            summary.get("heldout_assignment_seeds"),
            label="comparison held-out assignment seeds",
        )
    ]
    verified: set[tuple[int, int]] = set()
    development = _list(
        summary.get("development_populations"),
        label="comparison development populations",
    )
    if len(development) != 2:
        _fail("Expected two comparison development populations.")
    for topology, raw_population in zip((4, 8), development):
        key, _record = _validate_population(
            raw_population,
            expected_topology=topology,
            expected_seed=development_seed,
            contract_version=contract_version,
            result_root=result_root,
        )
        verified.add(key)
    blocks = _list(summary.get("heldout"), label="comparison heldout")
    expected_blocks = {
        (seed, topology) for seed in heldout_seeds for topology in (4, 8)
    }
    observed_blocks: set[tuple[int, int]] = set()
    for raw_block in blocks:
        block = _object(raw_block, label="comparison held-out block")
        seed = _integer(
            block.get("assignment_seed"),
            label="comparison held-out assignment seed",
        )
        topology = _integer(
            block.get("topology"), label="comparison held-out topology"
        )
        block_key = (seed, topology)
        if block_key not in expected_blocks or block_key in observed_blocks:
            _fail(f"Unexpected or duplicate comparison held-out block {block_key}.")
        observed_blocks.add(block_key)
        population_key, _record = _validate_population(
            block.get("population"),
            expected_topology=topology,
            expected_seed=seed,
            contract_version=contract_version,
            result_root=result_root,
        )
        if population_key in verified:
            _fail(f"Duplicate comparison population {population_key}.")
        verified.add(population_key)
    if observed_blocks != expected_blocks or len(verified) != 8:
        _fail("Expected all six comparison held-out blocks and eight populations.")
    return {
        "source_files_verified": 3,
        "screen_contract_verified": True,
        "population_receipt_pairs_verified": len(verified),
        "reset_commissioning_receipt_pairs_verified": (
            len(verified) if contract_version == 2 else 0
        ),
    }


def _target_hashes(
    value: Any, *, topology: int, label: str
) -> dict[str, list[str]]:
    hashes = _object(value, label=label)
    names = {"standard_level_targets", "continuous_envelope_targets"}
    _exact_keys(hashes, names, label=label)
    target_count = topology // 2
    result: dict[str, list[str]] = {}
    for name in sorted(names):
        raw = _list(hashes[name], label=f"{label}.{name}")
        if len(raw) != target_count:
            _fail(f"Expected {target_count} {label}.{name} hashes.")
        result[name] = [
            _sha(digest, label=f"{label}.{name}[{index}]")
            for index, digest in enumerate(raw)
        ]
    return result


def _validate_mapping_header(
    value: Any,
    *,
    scheme: str,
    topology: int,
    use_fixed_r: bool,
    scale_pair: tuple[float, float],
    contract_version: int,
    label: str,
) -> tuple[Mapping[str, Any], dict[str, list[str]]]:
    mapping = _object(value, label=label)
    expected_origin = (
        "exact_sampled_reference_center_with_active_a_projection_only"
        if use_fixed_r
        else (
            "bounded_per_cell_mean_apparent_raw_a_reset_origin"
            if contract_version == 2
            else "sampled_min_bound_reset_origin"
        )
    )
    if (
        mapping.get("scheme") != scheme
        or mapping.get("encoding")
        != ("single" if topology == 4 else "differential")
        or mapping.get("device_count_per_logical_weight") != topology
        or mapping.get("use_fixed_reference") is not use_fixed_r
        or mapping.get("level_origin_policy") != expected_origin
        or tuple(mapping.get("scale_fractions", ())) != scale_pair
        or mapping.get("standard_level_spacing_delta_multiples") != 4
        or _number(
            mapping.get("standard_level_spacing"),
            label=f"{label}.standard_level_spacing",
        )
        <= 0.0
    ):
        _fail(f"Expected complete mapping header contract for {label}.")
    hashes = _target_hashes(
        mapping.get("hashes"), topology=topology, label=f"{label}.hashes"
    )
    layers = _list(mapping.get("layers"), label=f"{label}.layers")
    if len(layers) != 2:
        _fail(f"Expected two mapping layers for {label}.")
    return mapping, hashes


def _selected_development(
    summary: Mapping[str, Any],
    *,
    scale_values: Sequence[float],
    contract_version: int,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    raw_development = _object(
        summary["development_calibration"], label="development_calibration"
    )
    if set(raw_development) != set(_SCHEMES):
        _fail("Expected development calibration for exactly the four declared schemes.")
    expected_pairs = set(product(scale_values, repeat=2))
    selected: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for scheme in _SCHEME_ORDER:
        topology, use_fixed_r = _SCHEMES[scheme]
        report = _object(raw_development[scheme], label=f"development.{scheme}")
        if (
            report.get("scheme") != scheme
            or report.get("topology") != topology
            or report.get("selection_domain")
            != "development_assignment_after_standard_level_mapping"
            or report.get("selection_metric")
            != "mapped_accuracy_then_calibrated_kl_then_scale_pair"
            or report.get("per_scheme_refit_performed") is not True
        ):
            _fail(f"Expected matched per-scheme development calibration for {scheme}.")
        candidates = _list(report.get("candidates"), label=f"development.{scheme}.candidates")
        if len(candidates) != len(expected_pairs):
            _fail(f"Expected the full Cartesian scale grid for {scheme}.")
        observed_pairs: set[tuple[float, float]] = set()
        normalized_candidates: list[tuple[tuple[float, float], Mapping[str, Any]]] = []
        for candidate_index, raw_candidate in enumerate(candidates):
            candidate = _object(raw_candidate, label=f"{scheme}.candidate[{candidate_index}]")
            pair_raw = _list(candidate.get("scale_fractions"), label=f"{scheme}.candidate.scale_fractions")
            if len(pair_raw) != 2:
                _fail(f"Expected two layer scale fractions for {scheme}.")
            pair = tuple(_number(value, label=f"{scheme}.candidate.scale") for value in pair_raw)
            observed_pairs.add((pair[0], pair[1]))
            calibration = _object(candidate.get("calibration"), label=f"{scheme}.candidate.calibration")
            accuracy = _probability(calibration.get("student_accuracy"), label=f"{scheme}.candidate.accuracy")
            calibrated_kl = _number(calibration.get("calibrated_kl"), label=f"{scheme}.candidate.calibrated_kl")
            if calibrated_kl < 0.0:
                _fail(f"Expected nonnegative calibrated KL for {scheme}.")
            _target_hashes(
                candidate.get("target_hashes"),
                topology=topology,
                label=f"{scheme}.candidate[{candidate_index}].target_hashes",
            )
            normalized_candidates.append(((pair[0], pair[1]), candidate))
        if observed_pairs != expected_pairs:
            _fail(f"Expected every scale pair exactly once for {scheme}.")
        expected_index = min(
            range(len(candidates)),
            key=lambda index: (
                -float(candidates[index]["calibration"]["student_accuracy"]),
                float(candidates[index]["calibration"]["calibrated_kl"]),
                tuple(float(value) for value in candidates[index]["scale_fractions"]),
            ),
        )
        selected_index = _integer(report.get("selected_index"), label=f"{scheme}.selected_index", minimum=0)
        if selected_index != expected_index or selected_index >= len(candidates):
            _fail(f"Expected the declared selection rule to choose candidate {expected_index} for {scheme}.")
        selected_candidate = _object(report.get("selected"), label=f"{scheme}.selected")
        if selected_candidate != candidates[selected_index]:
            _fail(f"Expected selected candidate payload to match candidates[{selected_index}] for {scheme}.")
        pair = tuple(float(value) for value in selected_candidate["scale_fractions"])
        calibration = _object(selected_candidate["calibration"], label=f"{scheme}.selected.calibration")
        gain = _number(calibration.get("gain"), label=f"{scheme}.selected.gain")
        if gain <= 0.0 or calibration.get("gain_was_fit_on_this_scheme") is not True:
            _fail(f"Expected a positive per-scheme fitted gain for {scheme}.")
        continuous = _object(
            report.get("continuous_envelope_calibration_at_selected_gain"),
            label=f"{scheme}.continuous_envelope_calibration",
        )
        _same(
            _number(continuous.get("gain"), label=f"{scheme}.continuous.gain"),
            gain,
            label=f"{scheme}.continuous.gain",
        )
        _mapping, mapping_hashes = _validate_mapping_header(
            report.get("mapping"),
            scheme=scheme,
            topology=topology,
            use_fixed_r=use_fixed_r,
            scale_pair=(pair[0], pair[1]),
            contract_version=contract_version,
            label=f"{scheme}.mapping",
        )
        selected_hashes = _target_hashes(
            selected_candidate.get("target_hashes"),
            topology=topology,
            label=f"{scheme}.selected.target_hashes",
        )
        if mapping_hashes != selected_hashes:
            _fail(
                f"Expected selected candidate target hashes to match the mapping for {scheme}."
            )
        selected[scheme] = {
            "scale_fractions": pair,
            "gain": gain,
            "development_accuracy": _probability(
                calibration.get("student_accuracy"), label=f"{scheme}.development_accuracy"
            ),
            "development_calibrated_kl": _number(
                calibration.get("calibrated_kl"), label=f"{scheme}.development_calibrated_kl"
            ),
            "continuous_development_accuracy": _probability(
                continuous.get("student_accuracy"), label=f"{scheme}.continuous_development_accuracy"
            ),
        }
        rows.append(
            {
                "scheme": scheme,
                "topology": topology,
                "use_fixed_r": use_fixed_r,
                "scale_layer_0": pair[0],
                "scale_layer_1": pair[1],
                "gain": gain,
                "development_accuracy": selected[scheme]["development_accuracy"],
                "continuous_development_accuracy": selected[scheme]["continuous_development_accuracy"],
                "development_calibrated_kl": selected[scheme]["development_calibrated_kl"],
            }
        )
    return selected, rows


def _layer_mechanism_row(
    *,
    scheme: str,
    topology: int,
    use_fixed_r: bool,
    seed: int,
    layer_index: int,
    mapping_layer: Mapping[str, Any],
    standard_test: Mapping[str, Any],
    continuous_test: Mapping[str, Any],
) -> dict[str, Any]:
    if mapping_layer.get("layer") != layer_index:
        _fail(f"Expected ordered mapping layers for {scheme} assignment {seed}.")
    if mapping_layer.get("standard_level_spacing_delta_multiples") != 4:
        _fail(f"Expected four-delta standard levels for {scheme} assignment {seed}.")
    spacing = _number(mapping_layer.get("standard_level_spacing"), label="standard_level_spacing")
    if spacing <= 0.0:
        _fail("Expected positive standard-level spacing.")
    outside = _integer(
        mapping_layer.get("active_target_outside_bounds_count"),
        label="active_target_outside_bounds_count",
        minimum=0,
    )
    if not use_fixed_r and outside != 0:
        _fail(f"Expected no RESET-anchored target outside active bounds for {scheme}.")
    zero_fraction = _probability(
        mapping_layer.get("zero_positive_capacity_quad_fraction"),
        label="zero_positive_capacity_quad_fraction",
    )
    if use_fixed_r and outside > 0 and zero_fraction <= 0.0:
        _fail("Expected every fixed-r out-of-bound structural target to expose zero capacity.")
    level_summary = _object(
        mapping_layer.get("per_quad_common_signed_logical_level_count"),
        label="signed logical level count",
    )
    error_summary = _object(
        mapping_layer.get("active_quantization_error"), label="active quantization error"
    )
    logical_summary = _object(mapping_layer.get("logical_contrast"), label="logical contrast")
    continuous_logical = _object(
        mapping_layer.get("continuous_logical_contrast"), label="continuous logical contrast"
    )
    baseline_summary = _object(
        mapping_layer.get("baseline_logical_contrast"), label="baseline logical contrast"
    )
    level_p10 = _number(level_summary.get("p10"), label="level count p10")
    level_median = _number(level_summary.get("median"), label="level count median")
    error_rms = _number(error_summary.get("rms"), label="quantization error RMS")
    logical_rms = _number(logical_summary.get("rms"), label="logical contrast RMS")
    continuous_rms = _number(continuous_logical.get("rms"), label="continuous logical contrast RMS")
    baseline_rms = _number(baseline_summary.get("rms"), label="baseline logical contrast RMS")
    loading = _number(
        mapping_layer.get("mean_edge_denominator_loading"), label="mean denominator loading"
    )
    exact_source_voltage = _object(
        standard_test["voltage"][layer_index], label="standard source voltage"
    )
    exact_destination_voltage = _object(
        standard_test["voltage"][layer_index + 1],
        label="standard destination voltage",
    )
    continuous_source_voltage = _object(
        continuous_test["voltage"][layer_index], label="continuous source voltage"
    )
    continuous_destination_voltage = _object(
        continuous_test["voltage"][layer_index + 1],
        label="continuous destination voltage",
    )
    return {
        "scheme": scheme,
        "topology": topology,
        "use_fixed_r": use_fixed_r,
        "assignment_seed": seed,
        "layer": layer_index,
        "signed_level_count_p10": level_p10,
        "signed_level_count_median": level_median,
        "zero_capacity_fraction": zero_fraction,
        "active_quantization_error_rms_over_spacing": error_rms / spacing,
        "logical_sign_flip_fraction": _probability(
            mapping_layer.get("logical_sign_flip_fraction_nonzero"),
            label="logical sign flip fraction",
        ),
        "continuous_logical_sign_flip_fraction": _probability(
            mapping_layer.get("continuous_logical_sign_flip_fraction_nonzero"),
            label="continuous logical sign flip fraction",
        ),
        "baseline_logical_contrast_rms": baseline_rms,
        "logical_contrast_rms": logical_rms,
        "continuous_logical_contrast_rms": continuous_rms,
        "baseline_rms_over_logical_rms": _ratio(baseline_rms, logical_rms),
        "mean_edge_denominator_loading": loading,
        "logical_contrast_rms_over_mean_loading": _ratio(logical_rms, loading),
        "normalized_total_conductance_proxy": _number(
            mapping_layer.get("normalized_total_conductance_proxy"),
            label="normalized total conductance proxy",
        ),
        "standard_source_voltage_rms": _number(
            exact_source_voltage.get("rms"), label="standard source voltage RMS"
        ),
        "standard_destination_voltage_rms": _number(
            exact_destination_voltage.get("rms"),
            label="standard destination voltage RMS",
        ),
        "continuous_source_voltage_rms": _number(
            continuous_source_voltage.get("rms"), label="continuous source voltage RMS"
        ),
        "continuous_destination_voltage_rms": _number(
            continuous_destination_voltage.get("rms"),
            label="continuous destination voltage RMS",
        ),
        "active_target_outside_bounds_count": outside,
    }


def _validate_heldout(
    summary: Mapping[str, Any],
    *,
    selected: Mapping[str, Mapping[str, Any]],
    heldout_seeds: Sequence[int],
    contract_version: int,
    result_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[tuple[int, int], tuple[tuple[int, int], dict[str, Any]]]]:
    blocks = _list(summary["heldout"], label="heldout")
    expected_blocks = {(seed, topology) for seed in heldout_seeds for topology in (4, 8)}
    seen_blocks: set[tuple[int, int]] = set()
    accuracy_rows: list[dict[str, Any]] = []
    mechanism_rows: list[dict[str, Any]] = []
    population_records: dict[tuple[int, int], tuple[tuple[int, int], dict[str, Any]]] = {}
    teacher_accuracy: float | None = None
    for block_index, raw_block in enumerate(blocks):
        block = _object(raw_block, label=f"heldout[{block_index}]")
        seed = _integer(block.get("assignment_seed"), label="heldout.assignment_seed")
        topology = _integer(block.get("topology"), label="heldout.topology")
        block_key = (seed, topology)
        if block_key not in expected_blocks or block_key in seen_blocks:
            _fail(f"Unexpected or duplicate held-out block {block_key}.")
        seen_blocks.add(block_key)
        population_records[block_key] = _validate_population(
            block.get("population"),
            expected_topology=topology,
            expected_seed=seed,
            contract_version=contract_version,
            result_root=result_root,
        )
        expected_schemes = {
            scheme for scheme, (scheme_topology, _fixed) in _SCHEMES.items() if scheme_topology == topology
        }
        arms = _list(block.get("arms"), label=f"heldout {block_key}.arms")
        if {arm.get("scheme") for arm in arms if isinstance(arm, dict)} != expected_schemes or len(arms) != 2:
            _fail(f"Expected exactly the two topology-{topology} arms for assignment {seed}.")
        for raw_arm in arms:
            arm = _object(raw_arm, label=f"heldout {block_key} arm")
            scheme = arm.get("scheme")
            if scheme not in expected_schemes:
                _fail(f"Unexpected held-out scheme {scheme!r} in block {block_key}.")
            use_fixed_r = _SCHEMES[str(scheme)][1]
            if (
                arm.get("assignment_seed") != seed
                or arm.get("calibration_source")
                != "frozen_per_scheme_development_assignment"
                or arm.get("per_scheme_refit_performed") is not True
            ):
                _fail(f"Expected held-out assignment/refit provenance for {scheme}.")
            frozen = selected[str(scheme)]
            pair = tuple(float(value) for value in _list(arm.get("scale_fractions"), label="arm.scale_fractions"))
            if pair != tuple(frozen["scale_fractions"]):
                _fail(f"Expected frozen development scale pair for {scheme} assignment {seed}.")
            gain = _number(arm.get("fixed_logit_gain"), label="arm.fixed_logit_gain")
            _same(gain, float(frozen["gain"]), label=f"{scheme} frozen gain")
            standard = _validate_test_metrics(
                arm.get("standard_level_test"), label=f"{scheme}.{seed}.standard", expected_gain=gain
            )
            continuous = _validate_test_metrics(
                arm.get("continuous_envelope_test"), label=f"{scheme}.{seed}.continuous", expected_gain=gain
            )
            for metric in (standard, continuous):
                observed_teacher = float(metric["teacher_accuracy"])
                if teacher_accuracy is None:
                    teacher_accuracy = observed_teacher
                else:
                    _same(observed_teacher, teacher_accuracy, label="teacher accuracy across held-out arms")
            flip_fraction = _probability(
                arm.get("continuous_to_standard_prediction_flip_fraction"),
                label="continuous-to-standard prediction flip fraction",
            )
            flip_count = _integer(
                arm.get("continuous_to_standard_prediction_flip_count"),
                label="continuous-to-standard prediction flip count",
                minimum=0,
            )
            _same(flip_fraction, flip_count / 10_000.0, label="prediction flip count/fraction")
            standard_accuracy = float(standard["student_accuracy"])
            continuous_accuracy = float(continuous["student_accuracy"])
            accuracy_rows.append(
                {
                    "scheme": scheme,
                    "topology": topology,
                    "use_fixed_r": use_fixed_r,
                    "assignment_seed": seed,
                    "scale_layer_0": pair[0],
                    "scale_layer_1": pair[1],
                    "fixed_logit_gain": gain,
                    "standard_accuracy": standard_accuracy,
                    "continuous_accuracy": continuous_accuracy,
                    "standard_minus_continuous_accuracy": standard_accuracy - continuous_accuracy,
                    "standard_teacher_agreement": float(standard["teacher_agreement"]),
                    "continuous_teacher_agreement": float(continuous["teacher_agreement"]),
                    "standard_kl_teacher_student": float(standard["kl_teacher_student"]),
                    "continuous_kl_teacher_student": float(continuous["kl_teacher_student"]),
                    "prediction_flip_fraction": flip_fraction,
                }
            )
            mapping, _mapping_hashes = _validate_mapping_header(
                arm.get("mapping"),
                scheme=str(scheme),
                topology=topology,
                use_fixed_r=use_fixed_r,
                scale_pair=(pair[0], pair[1]),
                contract_version=contract_version,
                label=f"{scheme}.{seed}.mapping",
            )
            layers = _list(mapping.get("layers"), label=f"{scheme}.{seed}.mapping.layers")
            for layer_index, raw_layer in enumerate(layers):
                mechanism_rows.append(
                    _layer_mechanism_row(
                        scheme=str(scheme),
                        topology=topology,
                        use_fixed_r=use_fixed_r,
                        seed=seed,
                        layer_index=layer_index,
                        mapping_layer=_object(raw_layer, label="mapping layer"),
                        standard_test=standard,
                        continuous_test=continuous,
                    )
                )
    if seen_blocks != expected_blocks:
        _fail(f"Missing held-out blocks: {sorted(expected_blocks - seen_blocks)!r}.")
    accuracy_rows.sort(key=lambda row: (_SCHEME_ORDER.index(str(row["scheme"])), int(row["assignment_seed"])))
    mechanism_rows.sort(
        key=lambda row: (
            _SCHEME_ORDER.index(str(row["scheme"])),
            int(row["assignment_seed"]),
            int(row["layer"]),
        )
    )
    return accuracy_rows, mechanism_rows, population_records


def _validate_aggregates(
    summary: Mapping[str, Any], accuracy_rows: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source_aggregate = _object(summary["aggregate"], label="aggregate")
    if set(source_aggregate) != set(_SCHEMES):
        _fail("Expected aggregate entries for exactly the four schemes.")
    derived: dict[str, Any] = {}
    for scheme in _SCHEME_ORDER:
        rows = [row for row in accuracy_rows if row["scheme"] == scheme]
        standard = [float(row["standard_accuracy"]) for row in rows]
        continuous = [float(row["continuous_accuracy"]) for row in rows]
        delta = [left - right for left, right in zip(standard, continuous)]
        expected = {
            "assignments": [int(row["assignment_seed"]) for row in rows],
            "standard_level_accuracy": _float_summary(standard),
            "continuous_envelope_accuracy": _float_summary(continuous),
            "standard_minus_continuous_accuracy": _float_summary(delta),
            "passes_90_percent_mean_gate": _mean(standard) >= 0.9,
        }
        observed = _object(source_aggregate[scheme], label=f"aggregate.{scheme}")
        _exact_keys(observed, set(expected), label=f"aggregate.{scheme}")
        if observed["assignments"] != expected["assignments"]:
            _fail(f"Expected aggregate assignment ordering to match held-out rows for {scheme}.")
        if observed["passes_90_percent_mean_gate"] is not expected["passes_90_percent_mean_gate"]:
            _fail(f"Expected recomputed 90% gate for {scheme}.")
        for name in (
            "standard_level_accuracy",
            "continuous_envelope_accuracy",
            "standard_minus_continuous_accuracy",
        ):
            _validate_float_summary(observed[name], expected[name], label=f"aggregate.{scheme}.{name}")
        derived[scheme] = {
            **expected,
            "prediction_flip_fraction": _float_summary(
                [float(row["prediction_flip_fraction"]) for row in rows]
            ),
            "standard_accuracy_range": [min(standard), max(standard)],
            "continuous_accuracy_range": [min(continuous), max(continuous)],
        }

    lookup = {
        (str(row["scheme"]), int(row["assignment_seed"])): float(row["standard_accuracy"])
        for row in accuracy_rows
    }
    comparisons = [
        ("fixed_r_minus_no_r_four_devices", "four_with_fixed_r", "four_without_fixed_r"),
        ("fixed_r_minus_no_r_eight_devices", "eight_with_fixed_r", "eight_without_fixed_r"),
        ("eight_minus_four_without_fixed_r", "eight_without_fixed_r", "four_without_fixed_r"),
        ("eight_minus_four_with_fixed_r", "eight_with_fixed_r", "four_with_fixed_r"),
    ]
    paired_rows: list[dict[str, Any]] = []
    for comparison, left, right in comparisons:
        values = [lookup[(left, seed)] - lookup[(right, seed)] for seed in derived[left]["assignments"]]
        for seed, value in zip(derived[left]["assignments"], values):
            paired_rows.append(
                {"comparison": comparison, "assignment_seed": seed, "accuracy_delta": value}
            )
        paired_rows.append(
            {"comparison": comparison, "assignment_seed": "mean", "accuracy_delta": _mean(values)}
        )

    source_comparisons = _object(summary["matched_comparisons"], label="matched_comparisons")
    expected_source = {
        "fixed_r_minus_no_r": {
            "four_devices": _mean(
                [row["accuracy_delta"] for row in paired_rows if row["comparison"] == "fixed_r_minus_no_r_four_devices" and row["assignment_seed"] != "mean"]
            ),
            "eight_devices": _mean(
                [row["accuracy_delta"] for row in paired_rows if row["comparison"] == "fixed_r_minus_no_r_eight_devices" and row["assignment_seed"] != "mean"]
            ),
        },
        "eight_minus_four": {
            "without_fixed_r": _mean(
                [row["accuracy_delta"] for row in paired_rows if row["comparison"] == "eight_minus_four_without_fixed_r" and row["assignment_seed"] != "mean"]
            ),
            "with_fixed_r": _mean(
                [row["accuracy_delta"] for row in paired_rows if row["comparison"] == "eight_minus_four_with_fixed_r" and row["assignment_seed"] != "mean"]
            ),
        },
    }
    if set(source_comparisons) != set(expected_source):
        _fail("Expected the two declared matched-comparison axes.")
    for axis, expected_values in expected_source.items():
        observed_values = _object(source_comparisons[axis], label=f"matched_comparisons.{axis}")
        _exact_keys(observed_values, set(expected_values), label=f"matched_comparisons.{axis}")
        for name, expected_value in expected_values.items():
            _same(
                _number(observed_values[name], label=f"matched_comparisons.{axis}.{name}"),
                expected_value,
                label=f"matched_comparisons.{axis}.{name}",
            )
    return derived, paired_rows


def _aggregate_mechanisms(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        "signed_level_count_p10",
        "signed_level_count_median",
        "zero_capacity_fraction",
        "active_quantization_error_rms_over_spacing",
        "logical_sign_flip_fraction",
        "continuous_logical_sign_flip_fraction",
        "baseline_logical_contrast_rms",
        "logical_contrast_rms",
        "continuous_logical_contrast_rms",
        "mean_edge_denominator_loading",
        "normalized_total_conductance_proxy",
        "standard_source_voltage_rms",
        "standard_destination_voltage_rms",
        "continuous_source_voltage_rms",
        "continuous_destination_voltage_rms",
    )
    result = []
    for scheme in _SCHEME_ORDER:
        for layer in (0, 1):
            group = [row for row in rows if row["scheme"] == scheme and row["layer"] == layer]
            item: dict[str, Any] = {"scheme": scheme, "layer": layer}
            for field in fields:
                item[field] = _mean([float(row[field]) for row in group])
            item["baseline_rms_over_logical_rms"] = _ratio(
                float(item["baseline_logical_contrast_rms"]),
                float(item["logical_contrast_rms"]),
            )
            item["logical_contrast_rms_over_mean_loading"] = _ratio(
                float(item["logical_contrast_rms"]),
                float(item["mean_edge_denominator_loading"]),
            )
            item["ratio_aggregation"] = (
                "ratio_of_assignment_mean_components_not_mean_of_assignment_ratios"
            )
            result.append(item)
    return result


def _aggregate_commissioning(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    if not rows:
        return None
    result: dict[str, Any] = {}
    for topology in (4, 8):
        group = [row for row in rows if row["topology"] == topology]
        if len(group) != 4:
            _fail(f"Expected four commissioned populations for topology {topology}.")
        total_cells = sum(int(row["cells"]) for row in group)
        projected_cells = sum(int(row["projected_cell_count"]) for row in group)
        result[str(topology)] = {
            "population_count": len(group),
            "assignment_seeds": [int(row["assignment_seed"]) for row in group],
            "cells": total_cells,
            "reset_mean_raw_a_population_mean": _float_summary(
                [float(row["reset_mean_raw_a"]["mean"]) for row in group]
            ),
            "reset_standard_error_raw_a_population_mean": _float_summary(
                [
                    float(row["reset_standard_error_raw_a"]["mean"])
                    for row in group
                ]
            ),
            "projected_cell_fraction_unweighted": _float_summary(
                [float(row["projected_cell_fraction"]) for row in group]
            ),
            "projected_cell_count": projected_cells,
            "projected_cell_fraction_cell_weighted": projected_cells / total_cells,
        }
    return result


def _markdown_report(analysis: Mapping[str, Any]) -> str:
    semantic_comparison = analysis["semantic_comparison"]
    contract_version = analysis["source_contract_schema_version"]
    commissioning_clause = (
        " and all eight exact commissioning NPZ/receipt pairs"
        if contract_version == 2
        else ""
    )
    if semantic_comparison is None:
        integrity_statement = (
            "Internal consistency checks passed for all source files, eight "
            f"production population/receipt pairs{commissioning_clause}, the development selection rule, "
            "twelve complete-test arms, aggregates, and matched comparisons. No "
            "independent replay was supplied, so replay equivalence is not verified."
        )
    elif contract_version == 1:
        byte_state = (
                "passed"
            if semantic_comparison["declared_byte_identity_satisfied"]
            else "failed"
        )
        integrity_statement = (
            "Internal consistency and production-artifact checks passed, and the "
            "independent replay matched every scientific summary field after "
            "normalizing only population and receipt paths. The replay's predeclared "
            f"byte-identical-summary criterion {byte_state}."
        )
    else:
        integrity_statement = (
            "Internal consistency and production-artifact checks passed for both "
            "executions, including all population and commissioning receipts. The "
            "independent replay passed its predeclared scientific-field semantic-"
            "equivalence criterion after normalizing only result-root-dependent "
            "population and commissioning path strings. Raw byte identity was not "
            "a v2 success criterion."
        )
    lines = [
        f"# IBM OM standard-level post-run analysis — {analysis['screen_id']}",
        "",
        integrity_statement,
        "",
        "## Held-out accuracy",
        "",
        "| Scheme | Standard mean (range) | Continuous mean (range) | Standard − continuous | Prediction flips |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    aggregate = analysis["accuracy_aggregate"]
    for scheme in _SCHEME_ORDER:
        item = aggregate[scheme]
        standard = item["standard_level_accuracy"]
        continuous = item["continuous_envelope_accuracy"]
        delta = item["standard_minus_continuous_accuracy"]
        flips = item["prediction_flip_fraction"]
        lines.append(
            f"| `{scheme}` | {100*standard['mean']:.2f}% "
            f"({100*standard['minimum']:.2f}–{100*standard['maximum']:.2f}%) | "
            f"{100*continuous['mean']:.2f}% "
            f"({100*continuous['minimum']:.2f}–{100*continuous['maximum']:.2f}%) | "
            f"{100*delta['mean']:+.2f} pp | {100*flips['mean']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Frozen development calibration",
            "",
            "| Scheme | Layer scales | Gain | Development accuracy | Continuous accuracy | Calibrated KL |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in analysis["development_calibration"]:
        lines.append(
            f"| `{row['scheme']}` | {row['scale_layer_0']:g}, {row['scale_layer_1']:g} | "
            f"{row['gain']:.6g} | {100*row['development_accuracy']:.2f}% | "
            f"{100*row['continuous_development_accuracy']:.2f}% | "
            f"{row['development_calibrated_kl']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Matched accuracy differences",
            "",
            "| Comparison | 87001 | 87002 | 87003 | Mean |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    paired = analysis["paired_accuracy_differences"]
    for comparison_name in dict.fromkeys(row["comparison"] for row in paired):
        rows = [row for row in paired if row["comparison"] == comparison_name]
        values = {str(row["assignment_seed"]): float(row["accuracy_delta"]) for row in rows}
        lines.append(
            f"| `{comparison_name}` | {100*values['87001']:+.2f} pp | "
            f"{100*values['87002']:+.2f} pp | {100*values['87003']:+.2f} pp | "
            f"{100*values['mean']:+.2f} pp |"
        )
    lines.extend(
        [
            "",
            "## Mechanism diagnostics (assignment-mean components)",
            "",
            "RMS/loading ratios below are formed after averaging their numerator and denominator across assignments; they are not means of per-assignment ratios.",
            "",
            "| Scheme / layer | Median signed levels | Zero capacity | Quantization RMS / spacing | Baseline RMS / logical RMS | Logical RMS / sum loading |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in analysis["mechanism_aggregate"]:
        baseline_ratio = row["baseline_rms_over_logical_rms"]
        signal_ratio = row["logical_contrast_rms_over_mean_loading"]
        lines.append(
            f"| `{row['scheme']}` / {row['layer']} | {row['signed_level_count_median']:.2f} | "
            f"{100*row['zero_capacity_fraction']:.3f}% | "
            f"{row['active_quantization_error_rms_over_spacing']:.4f} | "
            f"{'n/a' if baseline_ratio is None else f'{baseline_ratio:.4f}'} | "
            f"{'n/a' if signal_ratio is None else f'{signal_ratio:.4f}'} |"
        )
    commissioning_rows = analysis["commissioning_by_population"]
    if commissioning_rows:
        lines.extend(
            [
                "",
                "## Per-cell RESET commissioning",
                "",
                "Each row is one independently verified topology/assignment artifact; projection is the fraction whose mapped raw-a mean was moved by the public coordinate and sampled active bounds.",
                "",
                "| Topology | Assignment | Cells | Raw-a mean | SE mean | Bounded origin mean | Projected cells |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in commissioning_rows:
            lines.append(
                f"| {row['topology']} | {row['assignment_seed']} | {row['cells']} | "
                f"{row['reset_mean_raw_a']['mean']:.6f} | "
                f"{row['reset_standard_error_raw_a']['mean']:.6f} | "
                f"{row['mapped_mean_after_bounds']['mean']:.6f} | "
                f"{row['projected_cell_count']} ({100*row['projected_cell_fraction']:.3f}%) |"
            )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "A low continuous-envelope accuracy points to representation, baseline-offset, capacity, or loading effects already present before rounding. A large standard-versus-continuous change with a strong prediction-flip fraction points to standard-level quantization. Development calibration accuracy uses 1,024 training examples whereas held-out accuracy uses the 10,000-example test set, so their change combines assignment shift with evaluation-cohort shift and cannot isolate assignment-sensitive calibration.",
            "",
            "The reported `difference_rms_over_mean_loading` is intentionally not compared across four- and eight-device topologies: its numerator is raw `G` for four-device edges but `G_a-G_r` for eight-device edges. The table instead uses effective logical-contrast RMS divided by conductance-sum loading. These three assignment seeds support matched descriptive comparisons, not a significance claim or fabricated-device conclusion.",
            "",
        ]
    )
    if semantic_comparison is not None:
        replay = analysis["launch_contract_adjudication"]["replay"]
        if contract_version == 1:
            lines.extend(
                [
                    "## Replay adjudication",
                    "",
                    f"- Scientific-field semantic equivalence: {'passed' if replay['semantic_equivalence_satisfied'] else 'failed'}.",
                    f"- Predeclared byte-identical summary SHA-256: {'passed' if replay['declared_byte_identity_satisfied'] else 'failed'}.",
                    "",
                    (
                        "Semantic equivalence does not override the frozen byte-identity "
                        "criterion; `declared_replay_success` remains false when byte "
                        "identity failed."
                    ),
                    "",
                ]
            )
        else:
            lines.extend(
                [
                    "## Replay adjudication",
                    "",
                    f"- Predeclared scientific-field semantic equivalence: {'passed' if replay['semantic_equivalence_satisfied'] else 'failed'}.",
                    f"- Observed raw summary byte identity (not required): {'yes' if replay['observed_byte_identity'] else 'no'}.",
                    "",
                ]
            )
    return "\n".join(lines)


def analyze_standard_level_screen(
    summary_path: Path,
    output_dir: Path,
    *,
    compare_summary_path: Path | None = None,
) -> dict[str, Any]:
    """Validate one completed summary and emit derived artifacts."""

    source = summary_path.expanduser().resolve()
    destination = output_dir.expanduser().resolve()
    if not source.is_file():
        _fail(f"Expected a completed screen summary: {source}.")
    source_digest_before = sha256_file(source)
    summary = _load_object(source, label="screen summary")
    _exact_keys(summary, _SUMMARY_KEYS, label="screen summary")
    if (
        summary["schema"] != SOURCE_SCHEMA
        or summary["schema_version"] != SOURCE_SCHEMA_VERSION
        or summary["status"] != "identity_aware_ideal_standard_level_screen_complete"
    ):
        _fail("Expected a complete schema-v1 IBM OM standard-level screen summary.")
    if summary["sample_limit"] is not None:
        _fail("Expected the evidentiary post-run analysis to use the complete test set.")
    semantic_digest = semantic_summary_sha256(summary)
    comparison_report: dict[str, Any] | None = None
    comparison_path: Path | None = None
    comparison_summary_digest: str | None = None
    if compare_summary_path is not None:
        comparison_path = compare_summary_path.expanduser().resolve()
        comparison = _load_object(comparison_path, label="comparison screen summary")
        _exact_keys(comparison, _SUMMARY_KEYS, label="comparison screen summary")
        if (
            comparison.get("schema") != SOURCE_SCHEMA
            or comparison.get("schema_version") != SOURCE_SCHEMA_VERSION
            or comparison.get("status")
            != "identity_aware_ideal_standard_level_screen_complete"
        ):
            _fail("Expected a complete schema-v1 comparison screen summary.")
        comparison_digest = semantic_summary_sha256(comparison)
        if comparison_digest != semantic_digest:
            _fail(
                "Expected replay summary scientific fields to match after "
                "normalizing only output-location-dependent population metadata; "
                f"observed {semantic_digest} != {comparison_digest}."
            )
        comparison_provenance = _validate_comparison_provenance(
            comparison_path, comparison
        )
        comparison_report = {
            "summary": str(comparison_path),
            "summary_sha256": sha256_file(comparison_path),
            "semantic_summary_sha256": comparison_digest,
            "scientific_fields_match": True,
            "artifact_provenance_verified": True,
            "provenance_verified": True,
            **comparison_provenance,
            "normalized_fields": [
                "development_populations[].path",
                "development_populations[].receipt",
                "heldout[].population.path",
                "heldout[].population.receipt",
                *(
                    [
                        "development_populations[].reset_baseline_commissioning.path",
                        "development_populations[].reset_baseline_commissioning.receipt",
                        "heldout[].population.reset_baseline_commissioning.path",
                        "heldout[].population.reset_baseline_commissioning.receipt",
                    ]
                    if comparison_provenance[
                        "reset_commissioning_receipt_pairs_verified"
                    ]
                    else []
                ),
            ],
        }
        comparison_summary_digest = sha256_file(comparison_path)

    screen_config_path = _verify_file(
        summary["screen_config"], summary["screen_config_sha256"], label="screen config"
    )
    _verify_file(summary["model_config"], summary["model_config_sha256"], label="model config")
    _verify_file(summary["teacher_weights"], summary["teacher_weights_sha256"], label="teacher weights")
    source_contract = _validate_source_contract(
        screen_config_path, summary, label="source"
    )
    screen_config = source_contract.raw
    scale_values = source_contract.scale_fractions
    contract_version = source_contract.contract_schema_version

    result_root = source.parent.parent
    contract_path = result_root / "screen_contract.json"
    screen_contract = _load_object(contract_path, label="screen contract")
    _exact_keys(
        screen_contract,
        {
            "screen_config",
            "screen_config_sha256",
            "model_config",
            "model_config_sha256",
            "teacher_weights",
            "teacher_weights_sha256",
            "aihwkit_python",
            "device",
            "sample_limit",
            "contract",
        },
        label="screen contract",
    )
    for name in ("screen_config", "screen_config_sha256", "model_config", "model_config_sha256", "teacher_weights", "teacher_weights_sha256", "device", "sample_limit"):
        if screen_contract.get(name) != summary.get(name):
            _fail(f"Expected screen_contract.json {name} to match summary.json.")
    if screen_contract.get("contract") != screen_config:
        _fail("Expected screen_contract.json to embed the exact frozen screen config.")

    development_seed = _integer(summary["development_assignment_seed"], label="development_assignment_seed")
    heldout_seeds = [
        _integer(value, label="heldout assignment seed")
        for value in _list(summary["heldout_assignment_seeds"], label="heldout assignment seeds")
    ]
    if heldout_seeds != [87001, 87002, 87003] or development_seed != 86001:
        _fail("Expected the frozen development and three held-out assignment seeds.")

    development_populations = _list(
        summary["development_populations"], label="development_populations"
    )
    if len(development_populations) != 2:
        _fail("Expected one four-device and one eight-device development population.")
    populations: dict[tuple[int, int], dict[str, Any]] = {}
    for topology, raw_population in zip((4, 8), development_populations):
        key, verified = _validate_population(
            raw_population,
            expected_topology=topology,
            expected_seed=development_seed,
            contract_version=contract_version,
            result_root=result_root,
        )
        populations[key] = verified

    selected, calibration_rows = _selected_development(
        summary,
        scale_values=scale_values,
        contract_version=contract_version,
    )
    accuracy_rows, mechanism_rows, heldout_population_records = _validate_heldout(
        summary,
        selected=selected,
        heldout_seeds=heldout_seeds,
        contract_version=contract_version,
        result_root=result_root,
    )
    for _block_key, (validated_key, verified) in heldout_population_records.items():
        if validated_key in populations:
            _fail(f"Duplicate population assignment/topology record {validated_key}.")
        populations[validated_key] = verified
    expected_population_keys = {
        (topology, seed)
        for topology in (4, 8)
        for seed in (development_seed, *heldout_seeds)
    }
    if set(populations) != expected_population_keys:
        _fail("Expected exactly eight topology-by-assignment populations.")
    commissioning_rows = sorted(
        [
            verified["commissioning"]
            for verified in populations.values()
            if verified["commissioning"] is not None
        ],
        key=lambda row: (int(row["topology"]), int(row["assignment_seed"])),
    )
    if len(commissioning_rows) != (8 if contract_version == 2 else 0):
        _fail("Expected v2 to verify exactly eight commissioning pairs and v1 none.")
    commissioning_aggregate = _aggregate_commissioning(commissioning_rows)

    accuracy_aggregate, paired_rows = _validate_aggregates(summary, accuracy_rows)
    mechanism_aggregate = _aggregate_mechanisms(mechanism_rows)
    launch_adjudication = _launch_contract_adjudication(
        source_summary=source,
        source_summary_sha256=source_digest_before,
        screen_id=str(summary["screen_id"]),
        contract_version=contract_version,
        screen_config_sha256=str(summary["screen_config_sha256"]),
        comparison_summary=comparison_path,
        comparison_summary_sha256=comparison_summary_digest,
        semantic_equivalence=(True if comparison_report is not None else None),
    )
    replay_launch = launch_adjudication["replay"]
    if comparison_report is None:
        status = "complete_internal_consistency_post_run_analysis"
    elif contract_version == 2:
        status = "complete_semantic_replay_verified"
    elif replay_launch["declared_byte_identity_satisfied"]:
        status = "complete_replay_and_declared_byte_identity_verified"
    else:
        status = "complete_semantic_replay_match_declared_byte_identity_failed"
    if comparison_report is not None:
        if contract_version == 1:
            comparison_report.update(
                {
                    "declared_byte_identity_satisfied": replay_launch[
                        "declared_byte_identity_satisfied"
                    ],
                    "declared_replay_success": replay_launch[
                        "declared_replay_success"
                    ],
                    "adjudication": replay_launch["adjudication"],
                }
            )
        else:
            comparison_report.update(
                {
                    "declared_byte_identity_required": False,
                    "observed_byte_identity": replay_launch[
                        "observed_byte_identity"
                    ],
                    "declared_replay_success": replay_launch[
                        "declared_replay_success"
                    ],
                    "adjudication": replay_launch["adjudication"],
                }
            )
    analysis: dict[str, Any] = {
        "schema": ANALYSIS_SCHEMA,
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "status": status,
        "screen_id": summary["screen_id"],
        "source_contract_schema_version": contract_version,
        "source_summary": str(source),
        "source_summary_sha256": source_digest_before,
        "semantic_summary_sha256": semantic_digest,
        "semantic_comparison": comparison_report,
        "launch_contract_adjudication": launch_adjudication,
        "claim_boundary": summary["claim_boundary"],
        "integrity": {
            "passed": True,
            "verification_scope": (
                "internal_consistency_only"
                if comparison_report is None
                else "internal_consistency_and_semantic_replay"
            ),
            "independent_replay_supplied": comparison_report is not None,
            "source_summary_unchanged": True,
            "complete_test_examples_per_arm": 10_000,
            "development_schemes_verified": 4,
            "heldout_arms_verified": len(accuracy_rows),
            "population_receipt_pairs_verified": len(populations),
            "reset_commissioning_receipt_pairs_verified": len(
                commissioning_rows
            ),
            "assignment_seeds": heldout_seeds,
        },
        "development_calibration": calibration_rows,
        "accuracy_by_assignment": accuracy_rows,
        "accuracy_aggregate": accuracy_aggregate,
        "paired_accuracy_differences": paired_rows,
        "mechanism_by_assignment_layer": mechanism_rows,
        "mechanism_aggregate": mechanism_aggregate,
        "commissioning_by_population": commissioning_rows,
        "commissioning_aggregate": commissioning_aggregate,
        "interpretation_contract": {
            "continuous_already_low": "representation_baseline_capacity_or_loading_before_rounding",
            "standard_changes_from_continuous": "standard_level_quantization_diagnostic",
            "development_to_heldout_change": (
                "joint_assignment_and_calibration_to_test_cohort_shift_not_an_"
                "assignment_only_diagnostic"
            ),
            "cross_topology_signal_loading_metric": "logical_contrast_rms_over_mean_edge_denominator_loading",
            "forbidden_cross_topology_metric": "difference_rms_over_mean_loading",
            "statistical_scope": "three_matched_assignment_seeds_descriptive_not_significance_test",
        },
    }

    if source == destination or destination in source.parents:
        _fail("Refusing to replace a directory containing the immutable source summary.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            dir=destination.parent,
            prefix=f".{destination.name}.staging.",
        )
    )
    try:
        outputs = {
            "json": staging / "post_run_analysis.json",
            "accuracy_csv": staging / "accuracy_by_assignment.csv",
            "mechanism_csv": staging / "mechanism_by_scheme_layer.csv",
            "markdown": staging / "post_run_analysis.md",
        }
        atomic_write_json(outputs["json"], analysis)
        _atomic_write_text(
            outputs["accuracy_csv"],
            _csv_text(
                accuracy_rows,
                (
                    "scheme",
                    "topology",
                    "use_fixed_r",
                    "assignment_seed",
                    "scale_layer_0",
                    "scale_layer_1",
                    "fixed_logit_gain",
                    "standard_accuracy",
                    "continuous_accuracy",
                    "standard_minus_continuous_accuracy",
                    "standard_teacher_agreement",
                    "continuous_teacher_agreement",
                    "standard_kl_teacher_student",
                    "continuous_kl_teacher_student",
                    "prediction_flip_fraction",
                ),
            ),
        )
        mechanism_fields = tuple(mechanism_rows[0])
        _atomic_write_text(
            outputs["mechanism_csv"], _csv_text(mechanism_rows, mechanism_fields)
        )
        _atomic_write_text(outputs["markdown"], _markdown_report(analysis))
        if {entry.name for entry in staging.iterdir()} != _OUTPUT_NAMES:
            _fail("Expected a complete staged analyzer output bundle.")
        if sha256_file(source) != source_digest_before:
            _fail("Source summary changed during post-run analysis.")
        _publish_staged_output(staging, destination)
    except BaseException:
        if staging.exists():
            _remove_exact_output_directory(staging, staging=True)
        raise
    return analysis


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and analyze a completed IBM OM standard-level screen."
    )
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--compare-summary",
        type=Path,
        default=None,
        help=(
            "Optional replay summary. Scientific fields must match after "
            "normalizing only result-root-dependent population metadata."
        ),
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    analyze_standard_level_screen(
        args.summary,
        args.output_dir,
        compare_summary_path=args.compare_summary,
    )


if __name__ == "__main__":
    main()
