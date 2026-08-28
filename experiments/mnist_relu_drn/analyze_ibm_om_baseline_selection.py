"""Fail-closed analysis of the four-device IBM OM baseline-selection study.

The analyzer intentionally replays no network inference.  It audits the twelve
native RunStore bundles, checks every registered byte hash, and independently
recomputes the physical ``G = B + d`` and exact-count facts needed by the
predeclared comparison.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import fields
from hashlib import sha256
import io
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from experiments.artifacts import (
    atomic_write_json,
    canonical_json_bytes,
    content_hash,
    sha256_file,
)
from experiments.mnist_relu_drn.ibm_om_baseline_selection import (
    BASELINE_POLICIES,
    INDEPENDENT_CELL_RESET_MEAN,
    LAYOUTS,
    MAPPING_SCHEMA,
    MAPPING_SCHEMA_VERSION,
    LayerPhysicalMapping,
    baseline_group_violation_mask,
    hardware_instance_fingerprint,
    quad_column_loading,
    quad_contrast,
    quad_loading,
    quad_row_loading,
)
from experiments.mnist_relu_drn.ibm_om_baseline_selection_config import (
    CONTINUOUS_METRIC_DEFINITION,
    DEVELOPMENT_ASSIGNMENT_SEED,
    EXPECTED_WEIGHTS_SHA256,
    EXPERIMENT_ID,
    HELDOUT_ASSIGNMENT_SEEDS,
    SPACING_DELTA_MULTIPLES,
    STANDARD4DELTA_METRIC_DEFINITION,
)
from experiments.mnist_relu_drn.ibm_om_baseline_selection_runtime import (
    BASELINE_RESIDUAL_SCHEMA,
    BASELINE_RESIDUAL_SCHEMA_VERSION,
    JOINT_SCHEMA,
    JOINT_SCHEMA_VERSION,
    PREDICTION_SCHEMA,
    PREDICTION_SCHEMA_VERSION,
    SUMMARY_SCHEMA,
    SUMMARY_SCHEMA_VERSION,
)
from experiments.study_workflow import load_study_record
from experiments.mnist_relu_drn.ibm_om_standard_level_scheme_screen import (
    _load_reset_commissioning_arrays,
)
from training.ibm_reram_hwa import load_om_array_population


ANALYSIS_SCHEMA = "ebl.mnist_relu_drn.ibm_om_baseline_selection_analysis"
ANALYSIS_SCHEMA_VERSION = 1
STUDY_ID = "mnist-ibm-om-four-device-baseline-selection-20260828-v1"
EXPECTED_EXAMPLES = 10_000
EXPECTED_RUNS = len(BASELINE_POLICIES) * len(HELDOUT_ASSIGNMENT_SEEDS)
OUTPUT_JSON = "baseline_selection_analysis.json"
OUTPUT_CSV = "baseline_selection_runs.csv"
OUTPUT_MARKDOWN = "baseline_selection_analysis.md"
WEIGHT_ERROR_SCHEMA = "ebl.mnist_relu_drn.ibm_om_baseline_weight_errors"
CALIBRATION_SCHEMA = (
    "ebl.mnist_relu_drn.ibm_om_baseline_development_calibration"
)
ARM_POLICIES = {
    "independent-cell-reset-mean": "independent_cell_reset_mean",
    "shared-quad-reset-max": "shared_quad_reset_max",
    "shared-destination-columns-reset-max": (
        "shared_destination_columns_reset_max"
    ),
    "reference-enforced-destination-columns": (
        "reference_enforced_destination_columns"
    ),
}
_IDENTITY_FIELDS = (
    "max_bound",
    "min_bound",
    "dwmin_up",
    "dwmin_down",
    "reference",
    "corrupt",
    "published_corrupt",
)
_TENSOR_FIELDS = tuple(
    field.name
    for field in fields(LayerPhysicalMapping)
    if field.name
    not in {
        "layer_index",
        "layout",
        "policy",
        "scale_fraction",
        "teacher_absmax",
        "level_spacing_unit",
        "level_spacing_physical",
    }
)
_WEIGHT_FIELDS = (
    "reconstructed_continuous_weight",
    "reconstructed_quantized_weight",
    "baseline_error",
    "envelope_error",
    "continuous_total_error",
    "quantization_error",
    "total_error",
    "quantization_error_levels",
)
_ROOT = Path(__file__).resolve().parents[2]


class BaselineSelectionAnalysisError(ValueError):
    """Raised when a study bundle cannot support the declared comparison."""


def _fail(message: str) -> None:
    raise BaselineSelectionAnalysisError(message)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"Duplicate JSON key {key!r}.")
        result[key] = value
    return result


def _reject_constant(value: str) -> Any:
    _fail(f"Non-finite JSON number {value!r}.")


def _json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (OSError, json.JSONDecodeError) as error:
        raise BaselineSelectionAnalysisError(
            f"Expected readable strict JSON for {label}: {path}."
        ) from error
    if not isinstance(value, dict):
        _fail(f"Expected {label} to contain a JSON object: {path}.")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    if set(value) != expected:
        _fail(
            f"Expected exact {label} keys; missing={sorted(expected - set(value))!r}, "
            f"unknown={sorted(set(value) - expected)!r}."
        )


def _scalar(arrays: Mapping[str, np.ndarray], name: str) -> Any:
    if name not in arrays or arrays[name].shape != ():
        _fail(f"Expected scalar NPZ field {name!r}.")
    return arrays[name].item()


def _npz(path: Path, *, label: str) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as source:
            if len(source.files) != len(set(source.files)):
                _fail(f"Expected unique fields in {label}: {path}.")
            return {name: np.asarray(source[name]) for name in source.files}
    except (OSError, ValueError) as error:
        raise BaselineSelectionAnalysisError(
            f"Expected readable pickle-free NPZ for {label}: {path}."
        ) from error


def _torch_hash(array: np.ndarray) -> str:
    value = torch.from_numpy(np.ascontiguousarray(array))
    digest = sha256()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(
        json.dumps(list(value.shape), separators=(",", ":")).encode("utf-8")
    )
    digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _raw_hash(array: np.ndarray) -> str:
    return sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def _summary(array: np.ndarray) -> dict[str, float]:
    value = torch.from_numpy(np.ascontiguousarray(array)).to(torch.float64).reshape(-1)
    if value.numel() == 0 or not bool(torch.isfinite(value).all()):
        _fail("Expected a finite, non-empty tensor summary input.")
    return {
        "minimum": float(value.min().item()),
        "mean": float(value.mean().item()),
        "rms": float(value.square().mean().sqrt().item()),
        "maximum": float(value.max().item()),
    }


def _error_summary(value: torch.Tensor) -> dict[str, float]:
    flat = value.detach().to(torch.float64).reshape(-1)
    absolute = flat.abs()
    return {
        "signed_mean": float(flat.mean().item()),
        "mae": float(absolute.mean().item()),
        "rmse": float(flat.square().mean().sqrt().item()),
        "p50_absolute": float(torch.quantile(absolute, 0.5).item()),
        "p95_absolute": float(torch.quantile(absolute, 0.95).item()),
        "maximum_absolute": float(absolute.max().item()),
    }


def _equal(left: Any, right: Any, *, label: str) -> None:
    if left != right:
        _fail(f"Expected exact {label}; provided {left!r}, expected {right!r}.")


def _digest(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"Expected lowercase SHA-256 for {label}.")
    return value


def _git_commit(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"Expected lowercase Git SHA-1 for {label}.")
    return value


def _validate_data_provenance(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail("Expected dataset provenance object.")
    _exact_keys(
        value,
        {
            "dataset",
            "test_examples",
            "test_data_sha256",
            "test_targets_sha256",
            "training_examples",
            "training_data_sha256",
            "training_targets_sha256",
            "calibration_examples",
            "calibration_indices_sha256",
            "calibration_raw_images_sha256",
            "calibration_raw_targets_sha256",
            "calibration_labels_sha256",
            "test_split_labels_used_for_mapping_or_calibration",
        },
        label="dataset provenance",
    )
    _equal(value["dataset"], "MNIST", label="dataset")
    _equal(value["test_examples"], EXPECTED_EXAMPLES, label="dataset test examples")
    _equal(value["training_examples"], 60_000, label="dataset training examples")
    _equal(value["calibration_examples"], 1024, label="calibration examples")
    _equal(
        value["test_split_labels_used_for_mapping_or_calibration"],
        False,
        label="test-label exclusion",
    )
    for name in (
        "test_data_sha256",
        "test_targets_sha256",
        "training_data_sha256",
        "training_targets_sha256",
        "calibration_indices_sha256",
        "calibration_raw_images_sha256",
        "calibration_raw_targets_sha256",
        "calibration_labels_sha256",
    ):
        _digest(value[name], label=f"dataset {name}")
    _equal(
        value["calibration_raw_targets_sha256"],
        value["calibration_labels_sha256"],
        label="raw versus loader calibration labels",
    )
    return dict(value)


def _validate_baseline_residual(
    path: Path,
    receipt_path: Path,
    *,
    logical_output_sizes: Sequence[int],
) -> dict[str, Any]:
    arrays = _npz(path, label="baseline rail-voltage residual")
    receipt = _json(receipt_path, label="baseline rail-voltage residual receipt")
    _exact_keys(
        arrays,
        {"schema", "schema_version", "examples", "layer_0_rail_voltage_residual", "layer_1_rail_voltage_residual"},
        label="baseline residual NPZ",
    )
    _exact_keys(
        receipt,
        {"schema", "schema_version", "artifact", "artifact_sha256", "report", "tensor_hashes"},
        label="baseline residual receipt",
    )
    _equal(str(_scalar(arrays, "schema")), BASELINE_RESIDUAL_SCHEMA, label="baseline residual schema")
    _equal(int(_scalar(arrays, "schema_version")), BASELINE_RESIDUAL_SCHEMA_VERSION, label="baseline residual version")
    _equal(int(_scalar(arrays, "examples")), EXPECTED_EXAMPLES, label="baseline residual NPZ examples")
    _equal(receipt["schema"], BASELINE_RESIDUAL_SCHEMA, label="baseline residual receipt schema")
    _equal(receipt["schema_version"], BASELINE_RESIDUAL_SCHEMA_VERSION, label="baseline residual receipt version")
    _equal(receipt["artifact"], path.name, label="baseline residual artifact name")
    _equal(receipt["artifact_sha256"], sha256_file(path), label="baseline residual artifact hash")
    value = receipt["report"]
    if not isinstance(value, dict):
        _fail("Expected baseline rail-voltage residual report object.")
    _exact_keys(value, {"definition", "examples", "layers", "residual_hashes"}, label="baseline rail-voltage residual report")
    _equal(
        value["definition"],
        "paired_rail_voltage_difference_with_all_offsets_d_equal_zero",
        label="baseline residual definition",
    )
    _equal(value["examples"], EXPECTED_EXAMPLES, label="baseline residual examples")
    layers = value["layers"]
    hashes = value["residual_hashes"]
    if not isinstance(layers, list) or len(layers) != 2 or not isinstance(hashes, list) or len(hashes) != 2:
        _fail("Expected two baseline residual layers and hashes.")
    for layer, layout in enumerate(LAYOUTS):
        tensor_name = f"layer_{layer}_rail_voltage_residual"
        residual = arrays[tensor_name]
        if residual.dtype != np.float32 or residual.shape != (
            EXPECTED_EXAMPLES,
            int(logical_output_sizes[layer]),
        ) or not bool(np.isfinite(residual).all()):
            _fail("Expected finite float32 baseline rail residual tensors with exact shape.")
        item = layers[layer]
        if not isinstance(item, dict):
            _fail("Expected baseline residual layer object.")
        _exact_keys(
            item,
            {"layer", "layout", "values", "signed_mean", "rms", "p95_absolute", "maximum_absolute"},
            label="baseline residual layer",
        )
        _equal(item["layer"], layer, label="baseline residual layer index")
        _equal(item["layout"], layout, label="baseline residual layout")
        torch_value = torch.from_numpy(np.ascontiguousarray(residual)).to(torch.float64)
        absolute = torch_value.abs().reshape(-1)
        expected_report = {
            "layer": layer,
            "layout": layout,
            "values": int(torch_value.numel()),
            "signed_mean": float(torch_value.mean().item()),
            "rms": float(torch_value.square().mean().sqrt().item()),
            "p95_absolute": float(torch.quantile(absolute, 0.95).item()),
            "maximum_absolute": float(absolute.max().item()),
        }
        _equal(item, expected_report, label="recomputed baseline residual report")
        tensor_hash = _torch_hash(residual)
        _equal(hashes[layer], tensor_hash, label="baseline residual report tensor hash")
        _equal(receipt["tensor_hashes"].get(tensor_name), tensor_hash, label="baseline residual receipt tensor hash")
    _equal(set(receipt["tensor_hashes"]), {"layer_0_rail_voltage_residual", "layer_1_rail_voltage_residual"}, label="baseline residual tensor-hash fields")
    return dict(value)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _artifact_index(run_dir: Path, result: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    records = result.get("artifacts")
    if not isinstance(records, list) or not records:
        _fail(f"Expected registered artifacts in {run_dir}.")
    indexed: dict[str, dict[str, Any]] = {}
    root = run_dir.resolve()
    for index, raw in enumerate(records):
        if not isinstance(raw, dict):
            _fail(f"Expected artifact record {index} to be an object in {run_dir}.")
        _exact_keys(raw, {"path", "sha256", "size_bytes", "kind"}, label="artifact")
        relative = raw["path"]
        if not isinstance(relative, str) or relative in indexed:
            _fail(f"Expected unique relative artifact paths in {run_dir}.")
        candidate = (run_dir / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            _fail(f"Artifact escapes its run directory: {relative!r}.")
        if not candidate.is_file():
            _fail(f"Registered artifact is missing: {candidate}.")
        if candidate.stat().st_size != raw["size_bytes"]:
            _fail(f"Registered artifact size mismatch: {candidate}.")
        if sha256_file(candidate) != raw["sha256"]:
            _fail(f"Registered artifact hash mismatch: {candidate}.")
        indexed[relative] = dict(raw)
    return indexed


def _registered(
    run_dir: Path,
    artifacts: Mapping[str, Mapping[str, Any]],
    relative: str,
    *,
    kind: str,
) -> Path:
    record = artifacts.get(relative)
    if record is None or record.get("kind") != kind:
        _fail(f"Expected registered {kind!r} artifact {relative!r} in {run_dir}.")
    return run_dir / relative


def _kind_paths(
    run_dir: Path,
    artifacts: Mapping[str, Mapping[str, Any]],
    *,
    kind: str,
    prefix: str,
) -> list[Path]:
    return [
        run_dir / relative
        for relative, record in sorted(artifacts.items())
        if record.get("kind") == kind and relative.startswith(prefix)
    ]


def _validate_population(
    path: Path,
    receipt_path: Path,
    *,
    expected_assignment_seed: int | None,
    expected_corruption_policy: str,
    expected_summary: Any,
) -> Any:
    try:
        population = load_om_array_population(path)
    except (OSError, ValueError) as error:
        raise BaselineSelectionAnalysisError(
            f"Invalid semantic OM population artifact: {path}."
        ) from error
    receipt = _json(receipt_path, label="OM population receipt")
    _exact_keys(
        receipt,
        {
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
        },
        label="OM population receipt",
    )
    _equal(receipt["schema"], "ebl.ibm_reram.om_array_population_receipt", label="population receipt schema")
    _equal(receipt["schema_version"], 1, label="population receipt version")
    _equal(receipt["backend"], "external_pinned_aihwkit_python", label="population backend")
    _equal(receipt["aihwkit_version"], "1.1.0", label="population AIHWKit version")
    _equal(population.aihwkit_version, "1.1.0", label="population artifact AIHWKit version")
    _equal(receipt["population_sha256"], sha256_file(path), label="population artifact hash")
    _equal(receipt["population_fingerprint"], population.fingerprint, label="population fingerprint")
    _equal(receipt["num_cells"], population.size, label="population cell count")
    if expected_assignment_seed is not None:
        _equal(population.assignment_seed, expected_assignment_seed, label="population assignment seed")
    _equal(population.corruption_policy, expected_corruption_policy, label="population corruption policy")
    request = receipt["request"]
    expected_request = {
        "preset": "reram_array_om",
        "assignment_seed": population.assignment_seed,
        "corruption_policy": expected_corruption_policy,
        "binding_keys": list(population.binding_keys),
        "binding_shapes": [list(shape) for shape in population.binding_shapes],
        "required_aihwkit_version": "1.1.0",
    }
    _equal(request, expected_request, label="population sampling request")
    _equal(
        receipt["sampler_source_sha256"],
        sha256_file(_ROOT / "experiments/reram_program_verify/hwa_population_sampler.py"),
        label="population sampler source hash",
    )
    _equal(
        receipt["population_implementation_sha256"],
        sha256_file(_ROOT / "training/ibm_reram_hwa.py"),
        label="population implementation source hash",
    )
    if not isinstance(expected_summary, dict):
        _fail("Expected population summary in scientific hardware report.")
    for key, expected in (
        ("assignment_seed", population.assignment_seed),
        ("sha256", sha256_file(path)),
        ("receipt_sha256", sha256_file(receipt_path)),
        ("population_fingerprint", population.fingerprint),
        ("cells", population.size),
    ):
        _equal(expected_summary.get(key), expected, label=f"population summary {key}")
    if "request" in expected_summary:
        _equal(expected_summary["request"], request, label="population summary request")
    noise = expected_summary.get("noise_partition")
    _equal(
        noise,
        {
            "commissioning_cycle_to_cycle_noise": "enabled",
            "commissioning_apparent_write_noise": "enabled",
            "mapped_target_write_noise": "disabled",
            "declared_dw_min_std": population.dw_min_std,
            "declared_write_noise_std": population.write_noise_std,
        },
        label="population noise partition",
    )
    return population


def _validate_commissioning(
    path: Path,
    receipt_path: Path,
    *,
    population: Any,
    expected_summary: Any,
) -> Mapping[str, Any]:
    try:
        stored = _load_reset_commissioning_arrays(path)
    except (OSError, RuntimeError, ValueError) as error:
        raise BaselineSelectionAnalysisError(
            f"Invalid RESET commissioning artifact: {path}."
        ) from error
    receipt = _json(receipt_path, label="RESET commissioning receipt")
    _exact_keys(
        receipt,
        {
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
        },
        label="RESET commissioning receipt",
    )
    _equal(receipt["schema"], "ebl.mnist_relu_drn.ibm_om_per_cell_raw_reset_commissioning", label="commissioning schema")
    _equal(receipt["schema_version"], 1, label="commissioning version")
    _equal(receipt["algorithm"], "sequential_reset_pulse_read_per_cell_mean_raw_a", label="commissioning algorithm")
    _equal(receipt["population_fingerprint"], population.fingerprint, label="commissioning population fingerprint")
    _equal(receipt["assignment_seed"], population.assignment_seed, label="commissioning assignment seed")
    _equal(receipt["topology"], 4, label="commissioning topology")
    _equal(receipt["read_samples"], 8, label="commissioning read samples")
    _equal(receipt["reset_pulses_per_cell"], 8, label="commissioning pulses per cell")
    _equal(receipt["total_reset_pulses"], population.size * 8, label="commissioning total pulses")
    _equal(receipt["artifact"], path.name, label="commissioning artifact name")
    _equal(receipt["artifact_sha256"], sha256_file(path), label="commissioning artifact hash")
    _equal(stored["population_fingerprint"], population.fingerprint, label="commissioning NPZ population fingerprint")
    _equal(stored["assignment_seed"], population.assignment_seed, label="commissioning NPZ seed")
    _equal(stored["read_samples"], 8, label="commissioning NPZ reads")
    _equal(stored["binding_keys"], population.binding_keys, label="commissioning binding keys")
    _equal(stored["binding_shapes"], population.binding_shapes, label="commissioning binding shapes")
    mean_hash = _torch_hash(stored["reset_mean_raw_a"].numpy())
    se_hash = _torch_hash(stored["reset_standard_error_raw_a"].numpy())
    _equal(receipt["reset_mean_raw_a_sha256"], mean_hash, label="RESET mean tensor hash")
    _equal(receipt["reset_standard_error_raw_a_sha256"], se_hash, label="RESET SE tensor hash")
    if not isinstance(expected_summary, dict):
        _fail("Expected commissioning summary in scientific hardware report.")
    for key, expected in (
        ("sha256", sha256_file(path)),
        ("receipt_sha256", sha256_file(receipt_path)),
        ("population_fingerprint", population.fingerprint),
        ("assignment_seed", population.assignment_seed),
        ("read_samples", 8),
        ("artifact_sha256", sha256_file(path)),
        ("reset_mean_raw_a_sha256", mean_hash),
        ("reset_standard_error_raw_a_sha256", se_hash),
    ):
        _equal(expected_summary.get(key), expected, label=f"commissioning summary {key}")
    return stored


def _validate_hardware_sources(
    run_dir: Path,
    artifacts: Mapping[str, Mapping[str, Any]],
    *,
    role: str,
    assignment_seed: int,
    protocol: Mapping[str, Any],
    hardware_summary: Mapping[str, Any],
    joint_report: Mapping[str, Any],
) -> None:
    prefix = f"artifacts/{role}/base/"
    stem = f"4-device-assignment-{assignment_seed}"
    population_path = _registered(
        run_dir,
        artifacts,
        f"{prefix}populations/{stem}.npz",
        kind="ibm_om_base_population",
    )
    population_receipt = _registered(
        run_dir,
        artifacts,
        f"{prefix}populations/{stem}.receipt.json",
        kind="ibm_om_base_population_receipt",
    )
    commissioning_path = _registered(
        run_dir,
        artifacts,
        f"{prefix}commissioning/{stem}.npz",
        kind="ibm_om_base_reset_commissioning",
    )
    commissioning_receipt = _registered(
        run_dir,
        artifacts,
        f"{prefix}commissioning/{stem}.receipt.json",
        kind="ibm_om_base_reset_commissioning_receipt",
    )
    population = _validate_population(
        population_path,
        population_receipt,
        expected_assignment_seed=assignment_seed,
        expected_corruption_policy=protocol["device"]["corruption_policy"],
        expected_summary=hardware_summary.get("population"),
    )
    _validate_commissioning(
        commissioning_path,
        commissioning_receipt,
        population=population,
        expected_summary=hardware_summary.get("base_commissioning"),
    )
    _equal(joint_report.get("base_population_fingerprint"), population.fingerprint, label="joint/base population binding")
    donor_summary = hardware_summary.get("donor_population")
    donor_commissioning_summary = hardware_summary.get("donor_commissioning")
    donor_paths = _kind_paths(run_dir, artifacts, kind="ibm_om_donor_population", prefix=f"artifacts/{role}/donor/")
    donor_receipts = _kind_paths(run_dir, artifacts, kind="ibm_om_donor_population_receipt", prefix=f"artifacts/{role}/donor/")
    donor_commissioning_paths = _kind_paths(run_dir, artifacts, kind="ibm_om_donor_reset_commissioning", prefix=f"artifacts/{role}/donor/")
    donor_commissioning_receipts = _kind_paths(run_dir, artifacts, kind="ibm_om_donor_reset_commissioning_receipt", prefix=f"artifacts/{role}/donor/")
    if donor_summary is None:
        if donor_commissioning_summary is not None or any((donor_paths, donor_receipts, donor_commissioning_paths, donor_commissioning_receipts)):
            _fail("Donor summary/artifacts must be jointly absent.")
        _equal(joint_report.get("donor_population_fingerprint"), None, label="absent donor binding")
        return
    if not all(len(values) == 1 for values in (donor_paths, donor_receipts, donor_commissioning_paths, donor_commissioning_receipts)):
        _fail("Expected exactly one complete donor population/commissioning artifact pair.")
    donor = _validate_population(
        donor_paths[0],
        donor_receipts[0],
        expected_assignment_seed=None,
        expected_corruption_policy=protocol["device"]["corruption_policy"],
        expected_summary=donor_summary,
    )
    _validate_commissioning(
        donor_commissioning_paths[0],
        donor_commissioning_receipts[0],
        population=donor,
        expected_summary=donor_commissioning_summary,
    )
    _equal(joint_report.get("donor_population_fingerprint"), donor.fingerprint, label="joint/donor population binding")


def _validate_joint(
    path: Path,
    receipt_path: Path,
    *,
    assignment_seed: int,
    assignment_role: str,
) -> dict[str, Any]:
    arrays = _npz(path, label="joint assignment")
    receipt = _json(receipt_path, label="joint-assignment receipt")
    expected_scalars = {
        "schema",
        "schema_version",
        "assignment_seed",
        "hardware_instance_id",
        "base_population_fingerprint",
        "donor_population_fingerprint",
        "maximum_attempts",
    }
    expected_vectors = {"failure_layer_index", "failure_flat_index", "selected_attempt"}
    expected_layers = set()
    for layer in range(2):
        expected_layers.update(
            {
                f"layer_{layer}_selected_attempt_grid",
                f"layer_{layer}_reset_mean_raw_a",
                f"layer_{layer}_reset_standard_error_raw_a",
                *(f"layer_{layer}_{name}" for name in _IDENTITY_FIELDS),
            }
        )
    _exact_keys(
        arrays,
        expected_scalars | expected_vectors | expected_layers,
        label="joint-assignment NPZ",
    )
    _exact_keys(
        receipt,
        {
            "schema",
            "schema_version",
            "assignment_seed",
            "hardware_instance_id",
            "artifact",
            "artifact_sha256",
            "report",
            "tensor_hashes",
        },
        label="joint-assignment receipt",
    )
    for source in (arrays, receipt):
        _equal(source["schema"] if source is receipt else _scalar(source, "schema"), JOINT_SCHEMA, label="joint schema")
        _equal(source["schema_version"] if source is receipt else int(_scalar(source, "schema_version")), JOINT_SCHEMA_VERSION, label="joint schema version")
        _equal(source["assignment_seed"] if source is receipt else int(_scalar(source, "assignment_seed")), assignment_seed, label="joint assignment seed")
    _equal(receipt["artifact"], path.name, label="joint receipt artifact name")
    _equal(receipt["artifact_sha256"], sha256_file(path), label="joint receipt artifact hash")
    tensor_hashes = {
        name: _raw_hash(value) for name, value in arrays.items() if value.ndim > 0
    }
    _equal(receipt["tensor_hashes"], tensor_hashes, label="joint tensor hashes")
    report = receipt["report"]
    if not isinstance(report, dict):
        _fail("Expected a joint-assignment report object.")
    hardware_id = str(_scalar(arrays, "hardware_instance_id"))
    _equal(receipt["hardware_instance_id"], hardware_id, label="joint hardware ID")
    _equal(report.get("hardware_instance_id"), hardware_id, label="joint report hardware ID")
    _equal(report.get("assignment_role"), assignment_role, label="joint assignment role")
    _equal(report.get("layouts"), list(LAYOUTS), label="joint layouts")
    _equal(report.get("maximum_attempts"), int(_scalar(arrays, "maximum_attempts")), label="joint attempt budget")
    for vector in expected_vectors:
        if arrays[vector].ndim != 1 or arrays[vector].dtype != np.int64:
            _fail(f"Expected int64 vector {vector!r} in joint assignment.")
    _equal(report.get("selected_attempts"), arrays["selected_attempt"].tolist(), label="selected attempts")
    _equal(report.get("failure_layer_index"), arrays["failure_layer_index"].tolist(), label="failure layers")
    _equal(report.get("failure_flat_index"), arrays["failure_flat_index"].tolist(), label="failure indices")
    coverage = report.get("final_policy_coverage")
    if (
        not isinstance(coverage, dict)
        or coverage.get("feasible_quads") != coverage.get("total_quads")
        or not isinstance(coverage.get("total_quads"), list)
        or len(coverage["total_quads"]) != 2
    ):
        _fail("Joint assignment does not have complete all-policy coverage.")
    semantic_keys = (
        "schema",
        "schema_version",
        "assignment_seed",
        "assignment_role",
        "base_population_fingerprint",
        "donor_population_fingerprint",
        "binding_keys",
        "binding_shapes",
        "layouts",
        "read_samples",
        "maximum_attempts",
        "selected_attempts",
        "failure_layer_index",
        "failure_flat_index",
    )
    metadata = {key: report.get(key) for key in semantic_keys}
    _equal(metadata["schema"], JOINT_SCHEMA, label="joint report schema")
    _equal(metadata["schema_version"], JOINT_SCHEMA_VERSION, label="joint report version")
    _equal(metadata["assignment_seed"], assignment_seed, label="joint report seed")
    _equal(metadata["base_population_fingerprint"], str(_scalar(arrays, "base_population_fingerprint")), label="base population fingerprint")
    donor = str(_scalar(arrays, "donor_population_fingerprint"))
    _equal(metadata["donor_population_fingerprint"], None if donor == "none" else donor, label="donor population fingerprint")
    fingerprint_tensors: dict[str, torch.Tensor] = {}
    corrupt_by_layer: list[int] = []
    published_corrupt_by_layer: list[int] = []
    for layer in range(2):
        reset_shape = arrays[f"layer_{layer}_reset_mean_raw_a"].shape
        if len(reset_shape) != 2 or reset_shape[0] % 2 or reset_shape[1] % 2:
            _fail("Expected even rank-2 commissioned RESET arrays.")
        if arrays[f"layer_{layer}_reset_standard_error_raw_a"].shape != reset_shape:
            _fail("Expected matched RESET mean/SE shapes.")
        grid = arrays[f"layer_{layer}_selected_attempt_grid"]
        if grid.shape != (reset_shape[0] // 2, reset_shape[1] // 2):
            _fail("Expected selected-attempt grid to match logical quad shape.")
        for name in (*_IDENTITY_FIELDS, "reset_mean_raw_a", "reset_standard_error_raw_a"):
            key = f"layer_{layer}_{name}"
            if arrays[key].shape != reset_shape:
                _fail(f"Expected joint tensor {key!r} to match layer shape.")
            fingerprint_tensors[key] = torch.from_numpy(np.ascontiguousarray(arrays[key]))
        corrupt_by_layer.append(
            int(np.count_nonzero(arrays[f"layer_{layer}_corrupt"]))
        )
        published_corrupt_by_layer.append(
            int(np.count_nonzero(arrays[f"layer_{layer}_published_corrupt"]))
        )
    _equal(report.get("final_corrupt_count_by_layer"), corrupt_by_layer, label="final corrupt counts by layer")
    _equal(report.get("final_corrupt_count"), sum(corrupt_by_layer), label="final corrupt count")
    _equal(report.get("published_corrupt_count_by_layer"), published_corrupt_by_layer, label="published corrupt counts by layer")
    _equal(report.get("published_corrupt_count"), sum(published_corrupt_by_layer), label="published corrupt count")
    if sum(corrupt_by_layer) != 0:
        _fail("Joint repaired population retains corrupt cells.")
    recomputed_id = hardware_instance_fingerprint(metadata, fingerprint_tensors)
    _equal(hardware_id, recomputed_id, label="recomputed hardware instance ID")
    return {"hardware_instance_id": hardware_id, "report": report}


def _validate_mapping(
    path: Path,
    receipt_path: Path,
    *,
    policy: str,
    assignment_seed: int,
    assignment_role: str,
    hardware_instance_id: str,
    nominal_dw_min: float,
) -> dict[str, Any]:
    arrays = _npz(path, label="physical mapping")
    receipt = _json(receipt_path, label="physical-mapping receipt")
    scalar_fields = {
        "schema",
        "schema_version",
        "policy",
        "assignment_seed",
        "assignment_role",
        "hardware_instance_id",
        "conductance_min",
        "conductance_max",
    }
    tensor_fields = {
        f"layer_{layer}_{name}" for layer in range(2) for name in _TENSOR_FIELDS
    }
    _exact_keys(arrays, scalar_fields | tensor_fields, label="physical-mapping NPZ")
    _exact_keys(
        receipt,
        {
            "schema",
            "schema_version",
            "policy",
            "assignment_seed",
            "assignment_role",
            "hardware_instance_id",
            "artifact",
            "artifact_sha256",
            "tensor_hashes",
            "mapping",
        },
        label="physical-mapping receipt",
    )
    expected = {
        "schema": MAPPING_SCHEMA,
        "schema_version": MAPPING_SCHEMA_VERSION,
        "policy": policy,
        "assignment_seed": assignment_seed,
        "assignment_role": assignment_role,
        "hardware_instance_id": hardware_instance_id,
    }
    for key, value in expected.items():
        _equal(receipt[key], value, label=f"mapping receipt {key}")
        actual = _scalar(arrays, key)
        if key in {"schema_version", "assignment_seed"}:
            actual = int(actual)
        else:
            actual = str(actual)
        _equal(actual, value, label=f"mapping NPZ {key}")
    _equal(receipt["artifact"], path.name, label="mapping artifact name")
    _equal(receipt["artifact_sha256"], sha256_file(path), label="mapping artifact hash")
    _equal(
        receipt["tensor_hashes"],
        {name: _torch_hash(arrays[name]) for name in sorted(tensor_fields)},
        label="physical-mapping tensor hashes",
    )
    conductance_min = float(_scalar(arrays, "conductance_min"))
    conductance_max = float(_scalar(arrays, "conductance_max"))
    if not (math.isfinite(conductance_min) and conductance_min >= 0.0 and conductance_max > conductance_min):
        _fail("Expected a finite nonnegative physical conductance range.")
    span = conductance_max - conductance_min
    level_spacing = SPACING_DELTA_MULTIPLES * (float(nominal_dw_min) / 2.0) * span
    zero_tolerance = 1e-9 * level_spacing
    mapping_report = receipt["mapping"]
    if not isinstance(mapping_report, dict) or mapping_report.get("policy") != policy:
        _fail("Expected mapping report to identify its policy.")
    _equal(mapping_report.get("scale_fractions") is not None, True, label="mapping scales presence")
    reports = mapping_report.get("layers")
    if not isinstance(reports, list) or len(reports) != 2:
        _fail("Expected exactly two mapping layer reports.")
    diagnostics = []
    tensors_by_layer: list[dict[str, np.ndarray]] = []
    for layer, layout in enumerate(LAYOUTS):
        values = {name: arrays[f"layer_{layer}_{name}"] for name in _TENSOR_FIELDS}
        tensors_by_layer.append(values)
        shape = values["baseline"].shape
        if len(shape) != 2 or shape[0] % 2 or shape[1] % 2:
            _fail("Expected even rank-2 mapping conductance tensors.")
        for name, value in values.items():
            if name.endswith("_contrast") or name.endswith("_loading") or name in {
                "source_weight", "normalized_weight", "selected_level_capacity",
                "positive_level_capacity", "negative_level_capacity",
                "integer_level_number", "selected_headroom_unit",
                "positive_headroom_unit", "negative_headroom_unit",
            }:
                continue
            if value.shape != shape:
                _fail(f"Expected physical mapping field {name!r} to match rail shape.")
        baseline = torch.from_numpy(np.ascontiguousarray(values["baseline"]))
        continuous_offset = torch.from_numpy(np.ascontiguousarray(values["continuous_offset"]))
        quantized_offset = torch.from_numpy(np.ascontiguousarray(values["quantized_offset"]))
        continuous = torch.from_numpy(np.ascontiguousarray(values["continuous_conductance"]))
        quantized = torch.from_numpy(np.ascontiguousarray(values["quantized_conductance"]))
        if not torch.equal(continuous, baseline + continuous_offset):
            _fail("Continuous mapping violates exact G=B+d.")
        if not torch.equal(quantized, baseline + quantized_offset):
            _fail("Four-delta mapping violates exact G=B+d.")
        lower = conductance_min + span * torch.from_numpy(np.ascontiguousarray(values["cell_lower_unit"]))
        upper = conductance_min + span * torch.from_numpy(np.ascontiguousarray(values["cell_upper_unit"]))
        tolerance = max(1e-12, span * 1e-6)
        for endpoint, target in (("baseline", baseline), ("continuous", continuous), ("standard4delta", quantized)):
            if (
                not bool(torch.isfinite(target).all())
                or bool(torch.any(target < -tolerance))
                or bool(torch.any(target < lower - tolerance))
                or bool(torch.any(target > upper + tolerance))
            ):
                _fail(f"Expected finite, nonnegative, per-cell bounded {endpoint} G.")
        recomputed = {
            "baseline_contrast": quad_contrast(baseline, layout=layout),
            "continuous_contrast": quad_contrast(continuous, layout=layout),
            "quantized_contrast": quad_contrast(quantized, layout=layout),
            "baseline_loading": quad_loading(baseline, layout=layout),
            "continuous_loading": quad_loading(continuous, layout=layout),
            "quantized_loading": quad_loading(quantized, layout=layout),
            "baseline_row_loading": quad_row_loading(baseline, layout=layout),
            "baseline_column_loading": quad_column_loading(baseline, layout=layout),
            "continuous_row_loading": quad_row_loading(continuous, layout=layout),
            "continuous_column_loading": quad_column_loading(continuous, layout=layout),
            "quantized_row_loading": quad_row_loading(quantized, layout=layout),
            "quantized_column_loading": quad_column_loading(quantized, layout=layout),
        }
        for name, value in recomputed.items():
            saved = torch.from_numpy(np.ascontiguousarray(values[name]))
            if not torch.equal(value, saved):
                _fail(f"Saved {name} does not equal recomputation from full G.")
        layer_report = reports[layer]
        _equal(layer_report.get("layer"), layer, label="mapping report layer")
        _equal(layer_report.get("layout"), layout, label="mapping report layout")
        for name in ("baseline_contrast", "continuous_contrast", "quantized_contrast", "baseline_loading", "selected_headroom_unit"):
            _equal(layer_report.get(name), _summary(values[name]), label=f"mapping report {name}")
        reported_hashes = layer_report.get("hashes")
        expected_hashes = {
            name: _torch_hash(values[name])
            for name in (
                "baseline", "continuous_offset", "quantized_offset",
                "continuous_conductance", "quantized_conductance",
                "baseline_contrast", "continuous_contrast", "quantized_contrast",
            )
        }
        _equal(reported_hashes, expected_hashes, label="mapping report hashes")
        zero_max = float(recomputed["baseline_contrast"].abs().max().item())
        zero_required = policy != INDEPENDENT_CELL_RESET_MEAN
        if zero_required and zero_max > zero_tolerance:
            _fail(f"Policy {policy!r} violates its exact-zero contrast gate.")
        group_violation_count = int(
            baseline_group_violation_mask(
                baseline, layout=layout, policy=policy
            )
            .sum()
            .item()
        )
        if group_violation_count:
            _fail(
                f"Policy {policy!r} violates its declared baseline grouping "
                f"in {group_violation_count} logical quads."
            )
        capacity = values["selected_level_capacity"].astype(np.float64, copy=False)
        column = values["baseline_column_loading"].astype(np.float64, copy=False)
        diagnostics.append(
            {
                "layer": layer,
                "layout": layout,
                "baseline_contrast": _summary(values["baseline_contrast"]),
                "baseline_loading": _summary(values["baseline_loading"]),
                "baseline_destination_column_loading_imbalance": _summary(column[..., 0] - column[..., 1]),
                "continuous_loading": _summary(values["continuous_loading"]),
                "standard4delta_loading": _summary(values["quantized_loading"]),
                "selected_headroom_unit": _summary(values["selected_headroom_unit"]),
                "selected_level_capacity": _summary(capacity),
                "zero_selected_capacity_count": int(np.count_nonzero(capacity == 0)),
                "exact_zero_required": zero_required,
                "exact_zero_tolerance": zero_tolerance,
                "exact_zero_passed": (not zero_required) or zero_max <= zero_tolerance,
                "declared_baseline_group_violation_count": group_violation_count,
                "continuous_decomposition_max_abs_residual": 0.0,
                "standard4delta_decomposition_max_abs_residual": 0.0,
            }
        )
    return {
        "mapping": mapping_report,
        "layers": diagnostics,
        "tensors": tensors_by_layer,
        "conductance_min": conductance_min,
        "conductance_max": conductance_max,
        "level_spacing_physical": level_spacing,
    }


def _validate_predictions(
    path: Path,
    receipt_path: Path,
    *,
    policy: str,
    assignment_seed: int,
) -> dict[str, Any]:
    arrays = _npz(path, label="prediction artifact")
    receipt = _json(receipt_path, label="prediction receipt")
    fields_expected = {
        "schema", "schema_version", "assignment_seed", "policy", "labels",
        "teacher_prediction", "continuous_prediction", "standard4delta_prediction",
    }
    _exact_keys(arrays, fields_expected, label="prediction NPZ")
    _exact_keys(
        receipt,
        {"schema", "schema_version", "assignment_seed", "policy", "examples", "artifact", "artifact_sha256", "correct", "labels_sha256", "prediction_hashes"},
        label="prediction receipt",
    )
    expected = {
        "schema": PREDICTION_SCHEMA,
        "schema_version": PREDICTION_SCHEMA_VERSION,
        "assignment_seed": assignment_seed,
        "policy": policy,
    }
    for key, value in expected.items():
        _equal(receipt[key], value, label=f"prediction receipt {key}")
        actual = _scalar(arrays, key)
        actual = int(actual) if key in {"schema_version", "assignment_seed"} else str(actual)
        _equal(actual, value, label=f"prediction NPZ {key}")
    _equal(receipt["artifact"], path.name, label="prediction artifact name")
    _equal(receipt["artifact_sha256"], sha256_file(path), label="prediction artifact hash")
    names = ("labels", "teacher_prediction", "continuous_prediction", "standard4delta_prediction")
    for name in names:
        value = arrays[name]
        if value.dtype != np.int64 or value.shape != (EXPECTED_EXAMPLES,):
            _fail(f"Expected {name} to be int64[{EXPECTED_EXAMPLES}].")
        if np.any(value < 0) or np.any(value > 9):
            _fail(f"Expected MNIST class values in {name}.")
    labels = arrays["labels"]
    correct = {
        "teacher": int(np.count_nonzero(arrays["teacher_prediction"] == labels)),
        "continuous": int(np.count_nonzero(arrays["continuous_prediction"] == labels)),
        "standard4delta": int(np.count_nonzero(arrays["standard4delta_prediction"] == labels)),
    }
    hashes = {
        "teacher": _torch_hash(arrays["teacher_prediction"]),
        "continuous": _torch_hash(arrays["continuous_prediction"]),
        "standard4delta": _torch_hash(arrays["standard4delta_prediction"]),
    }
    _equal(receipt["examples"], EXPECTED_EXAMPLES, label="prediction example count")
    _equal(receipt["correct"], correct, label="prediction exact correct counts")
    _equal(receipt["labels_sha256"], _torch_hash(labels), label="prediction label hash")
    _equal(receipt["prediction_hashes"], hashes, label="prediction hashes")
    return {
        "examples": EXPECTED_EXAMPLES,
        "correct": correct,
        "hashes": hashes,
        "labels_sha256": receipt["labels_sha256"],
        "teacher_agreement": {
            "continuous": int(np.count_nonzero(arrays["continuous_prediction"] == arrays["teacher_prediction"])),
            "standard4delta": int(np.count_nonzero(arrays["standard4delta_prediction"] == arrays["teacher_prediction"])),
        },
    }


def _validate_weight_errors(
    path: Path,
    receipt_path: Path,
    *,
    policy: str,
    assignment_seed: int,
    mapping: Mapping[str, Any],
    analysis_scales: Sequence[float],
) -> list[dict[str, Any]]:
    arrays = _npz(path, label="weight-error artifact")
    receipt = _json(receipt_path, label="weight-error receipt")
    scalar_fields = {"schema", "schema_version", "assignment_seed", "policy"}
    tensor_fields = {f"layer_{layer}_{name}" for layer in range(2) for name in _WEIGHT_FIELDS}
    _exact_keys(arrays, scalar_fields | tensor_fields, label="weight-error NPZ")
    _exact_keys(
        receipt,
        {"schema", "schema_version", "assignment_seed", "policy", "artifact", "artifact_sha256", "analysis_scales", "layers"},
        label="weight-error receipt",
    )
    expected = {
        "schema": WEIGHT_ERROR_SCHEMA,
        "schema_version": 1,
        "assignment_seed": assignment_seed,
        "policy": policy,
    }
    for key, value in expected.items():
        _equal(receipt[key], value, label=f"weight receipt {key}")
        actual = _scalar(arrays, key)
        actual = int(actual) if key in {"schema_version", "assignment_seed"} else str(actual)
        _equal(actual, value, label=f"weight NPZ {key}")
    _equal(receipt["artifact"], path.name, label="weight artifact name")
    _equal(receipt["artifact_sha256"], sha256_file(path), label="weight artifact hash")
    _equal(receipt["analysis_scales"], list(analysis_scales), label="frozen analysis scales")
    reports = receipt["layers"]
    if not isinstance(reports, list) or len(reports) != 2:
        _fail("Expected two weight-error reports.")
    verified = []
    for layer in range(2):
        saved = {name: torch.from_numpy(np.ascontiguousarray(arrays[f"layer_{layer}_{name}"])) for name in _WEIGHT_FIELDS}
        source = torch.from_numpy(np.ascontiguousarray(mapping["tensors"][layer]["source_weight"])).to(torch.float64)
        c0 = torch.from_numpy(np.ascontiguousarray(mapping["tensors"][layer]["baseline_contrast"])).to(torch.float64)
        cc = torch.from_numpy(np.ascontiguousarray(mapping["tensors"][layer]["continuous_contrast"])).to(torch.float64)
        cq = torch.from_numpy(np.ascontiguousarray(mapping["tensors"][layer]["quantized_contrast"])).to(torch.float64)
        if source.ndim != 2 or source.shape != c0.shape:
            _fail("Expected logical source and contrast shapes to match.")
        maximum = float(source.abs().max().item())
        if maximum <= 0.0:
            _fail("Expected nonzero source weights.")
        normalized = source / maximum
        scale = float(analysis_scales[layer])
        expected_tensors = {
            "reconstructed_continuous_weight": maximum * cc / scale,
            "reconstructed_quantized_weight": maximum * cq / scale,
            "baseline_error": maximum * c0 / scale,
            "envelope_error": maximum * ((cc - c0) / scale - normalized),
            "continuous_total_error": maximum * (cc / scale - normalized),
            "quantization_error": maximum * (cq - cc) / scale,
            "total_error": maximum * (cq / scale - normalized),
            "quantization_error_levels": (cq - cc) / mapping["level_spacing_physical"],
        }
        for name, expected_tensor in expected_tensors.items():
            if not torch.equal(saved[name], expected_tensor.to(torch.float32)):
                _fail(f"Weight-error tensor {name!r} does not match full-G reconstruction.")
        residual = saved["total_error"].to(torch.float64) - sum(
            saved[name].to(torch.float64)
            for name in ("baseline_error", "envelope_error", "quantization_error")
        )
        tolerance = max(1e-12, maximum * 1e-6)
        if float(residual.abs().max().item()) > tolerance:
            _fail("Weight-error components do not sum to total error.")
        target_norm = float(source.norm().item())
        reconstructed_continuous = expected_tensors["reconstructed_continuous_weight"]
        reconstructed_quantized = expected_tensors["reconstructed_quantized_weight"]
        continuous_norm = float(reconstructed_continuous.norm().item())
        quantized_norm = float(reconstructed_quantized.norm().item())
        continuous_cosine = float((source * reconstructed_continuous).sum().item()) / (target_norm * continuous_norm) if target_norm and continuous_norm else 0.0
        quantized_cosine = float((source * reconstructed_quantized).sum().item()) / (target_norm * quantized_norm) if target_norm and quantized_norm else 0.0
        sign_tolerance = max(1e-12, 1e-9 * mapping["level_spacing_physical"])
        nonzero = source != 0.0
        continuous_erased = nonzero & (cc.abs() <= sign_tolerance)
        continuous_opposite = nonzero & ~continuous_erased & (torch.sign(cc) != torch.sign(source))
        quantized_erased = nonzero & (cq.abs() <= sign_tolerance)
        quantized_opposite = nonzero & ~quantized_erased & (torch.sign(cq) != torch.sign(source))
        raw = {
            "baseline": _error_summary(c0),
            "envelope": _error_summary((cc - c0) - scale * normalized),
            "continuous_total": _error_summary(cc - scale * normalized),
            "quantization": _error_summary(cq - cc),
            "standard4delta_total": _error_summary(cq - scale * normalized),
        }
        teacher = {
            "baseline": _error_summary(expected_tensors["baseline_error"]),
            "envelope": _error_summary(expected_tensors["envelope_error"]),
            "continuous_total": _error_summary(expected_tensors["continuous_total_error"]),
            "quantization": _error_summary(expected_tensors["quantization_error"]),
            "standard4delta_total": _error_summary(expected_tensors["total_error"]),
        }
        normalized_units = {
            "baseline": _error_summary(c0 / scale),
            "envelope": _error_summary((cc - c0) / scale - normalized),
            "continuous_total": _error_summary(cc / scale - normalized),
            "quantization": _error_summary((cq - cc) / scale),
            "standard4delta_total": _error_summary(cq / scale - normalized),
        }
        conductance_span = mapping["conductance_max"] - mapping["conductance_min"]
        normalized_conductance_units = {
            "baseline": _error_summary(c0 / conductance_span),
            "envelope": _error_summary(((cc - c0) - scale * normalized) / conductance_span),
            "continuous_total": _error_summary((cc - scale * normalized) / conductance_span),
            "quantization": _error_summary((cq - cc) / conductance_span),
            "standard4delta_total": _error_summary((cq - scale * normalized) / conductance_span),
        }
        expected_report = {
            "layer": layer,
            "analysis_scale": scale,
            "raw_contrast_error": raw,
            "normalized_conductance_coordinate_error": normalized_conductance_units,
            "normalized_relu_weight_error": normalized_units,
            "teacher_weight_error": teacher,
            "continuous_endpoint": {
                "relative_l2": float(expected_tensors["continuous_total_error"].norm().item()) / target_norm if target_norm else 0.0,
                "cosine_similarity": continuous_cosine,
                "opposite_sign_count": int(continuous_opposite.sum().item()),
                "erased_to_zero_count": int(continuous_erased.sum().item()),
                "sign_error_union_count": int((continuous_opposite | continuous_erased).sum().item()),
            },
            "standard4delta_endpoint": {
                "relative_l2": float(expected_tensors["total_error"].norm().item()) / target_norm if target_norm else 0.0,
                "cosine_similarity": quantized_cosine,
                "opposite_sign_count": int(quantized_opposite.sum().item()),
                "erased_to_zero_count": int(quantized_erased.sum().item()),
                "sign_error_union_count": int((quantized_opposite | quantized_erased).sum().item()),
            },
            "nonzero_source_count": int(nonzero.sum().item()),
            "quantization_error_in_levels": _error_summary(expected_tensors["quantization_error_levels"]),
            "component_sum_max_abs_residual": float(
                (expected_tensors["total_error"] - (
                    expected_tensors["baseline_error"] + expected_tensors["envelope_error"] + expected_tensors["quantization_error"]
                )).abs().max().item()
            ),
        }
        _equal(reports[layer], expected_report, label="weight-error report")
        verified.append(expected_report)
    return verified


def _metric_from_predictions(
    metric: Mapping[str, Any],
    predictions: Mapping[str, Any],
    endpoint: str,
) -> None:
    key = "continuous" if endpoint == "continuous" else "standard4delta"
    expected_keys = {
        "student_correct", "examples", "student_accuracy", "teacher_correct",
        "teacher_accuracy", "teacher_agreement_count", "teacher_agreement",
        "prediction_sha256",
    }
    if not expected_keys.issubset(metric):
        _fail(f"Expected complete {endpoint} metric fields.")
    student_correct = predictions["correct"][key]
    teacher_correct = predictions["correct"]["teacher"]
    agreement = predictions["teacher_agreement"][key]
    expected = {
        "student_correct": student_correct,
        "examples": EXPECTED_EXAMPLES,
        "student_accuracy": student_correct / EXPECTED_EXAMPLES,
        "teacher_correct": teacher_correct,
        "teacher_accuracy": teacher_correct / EXPECTED_EXAMPLES,
        "teacher_agreement_count": agreement,
        "teacher_agreement": agreement / EXPECTED_EXAMPLES,
        "prediction_sha256": predictions["hashes"][key],
    }
    for name, value in expected.items():
        _equal(metric[name], value, label=f"{endpoint} metric {name}")


def _validate_run(
    run_dir: Path,
    *,
    study: Mapping[str, Any],
    arm: Mapping[str, Any],
    expected_config_hash: str,
    expected_policy: str,
) -> dict[str, Any]:
    manifest = _json(run_dir / "manifest.json", label="run manifest")
    status = _json(run_dir / "status.json", label="run status")
    result = _json(run_dir / "result.json", label="run result")
    config = _json(run_dir / "config.resolved.json", label="resolved config")
    run_id = run_dir.name
    for value, label in ((manifest, "manifest"), (status, "status"), (result, "result")):
        _equal(value.get("schema"), "ebl.run", label=f"{label} schema")
        _equal(value.get("schema_version"), 1, label=f"{label} schema version")
        _equal(value.get("run_id"), run_id, label=f"{label} run ID")
    _equal(status.get("status"), "complete", label="run status")
    _equal(result.get("status"), "complete", label="result status")
    _equal(result.get("error"), None, label="result error")
    _equal(manifest.get("experiment_id"), EXPERIMENT_ID, label="manifest experiment")
    _equal(result.get("experiment_id"), EXPERIMENT_ID, label="result experiment")
    run_source = manifest.get("source")
    if not isinstance(run_source, dict):
        _fail("Expected a source-controlled run manifest.")
    _exact_keys(
        run_source,
        {"available", "commit", "dirty", "dirty_hash"},
        label="manifest source state",
    )
    _equal(run_source.get("available"), True, label="manifest source availability")
    _git_commit(run_source.get("commit"), label="manifest source commit")
    _equal(run_source.get("dirty"), False, label="manifest clean source state")
    _equal(run_source.get("dirty_hash"), None, label="manifest dirty hash")
    _equal(manifest.get("config", {}).get("sha256"), content_hash(config), label="resolved config hash")
    study_link = manifest.get("study")
    if not isinstance(study_link, dict):
        _fail("Expected workflow-managed study linkage in manifest.")
    _equal(study_link.get("study_id"), study["study_id"], label="manifest study ID")
    _equal(study_link.get("arm_id"), arm["arm_id"], label="manifest arm ID")
    _equal(study_link.get("study_sha256"), sha256_file(Path(study["_root"]) / "study.json"), label="manifest study hash")
    _equal(study_link.get("source_plan_sha256"), study["source_plan"]["sha256"], label="manifest plan hash")
    _equal(study_link.get("source_config_sha256"), expected_config_hash, label="manifest source-config hash")
    _equal(config.get("experiment_id"), EXPERIMENT_ID, label="resolved experiment")
    protocol = config.get("protocol")
    student = config.get("student")
    if not isinstance(protocol, dict) or not isinstance(student, dict):
        _fail("Expected resolved baseline-selection protocol and student config.")
    _equal(protocol.get("execution", {}).get("profile"), "production", label="production profile")
    _equal(student.get("runtime", {}).get("device"), "cuda", label="frozen production CUDA device")
    _equal(student.get("settings", {}).get("sample_limit"), None, label="production sample limit")
    _equal(protocol.get("source", {}).get("expected_weights_sha256"), EXPECTED_WEIGHTS_SHA256, label="source contract hash")
    _equal(protocol.get("baseline", {}).get("selected_policy"), expected_policy, label="selected policy")
    assignments = protocol.get("assignments", {})
    _equal(assignments.get("development_seed"), DEVELOPMENT_ASSIGNMENT_SEED, label="development seed")
    heldout_seed = assignments.get("heldout_seed")
    if heldout_seed not in HELDOUT_ASSIGNMENT_SEEDS:
        _fail(f"Unexpected held-out assignment seed {heldout_seed!r}.")
    inputs = manifest.get("inputs")
    if not isinstance(inputs, list):
        _fail("Expected manifest input artifacts.")
    weights_inputs = [item for item in inputs if isinstance(item, dict) and item.get("role") == "weights"]
    if len(weights_inputs) != 1:
        _fail("Expected exactly one frozen weights input.")
    _equal(weights_inputs[0].get("sha256"), EXPECTED_WEIGHTS_SHA256, label="weights input hash")
    artifacts = _artifact_index(run_dir, result)
    required = {
        "summary": _registered(run_dir, artifacts, "artifacts/scientific_summary.json", kind="scientific_summary"),
        "calibration": _registered(run_dir, artifacts, "artifacts/development/calibration.json", kind="development_calibration"),
        "dev_joint": _registered(run_dir, artifacts, "artifacts/development/joint_assignment.npz", kind="ibm_om_joint_assignment"),
        "dev_joint_receipt": _registered(run_dir, artifacts, "artifacts/development/joint_assignment.receipt.json", kind="ibm_om_joint_assignment_receipt"),
        "dev_mapping": _registered(run_dir, artifacts, "artifacts/development/selected_physical_mapping.npz", kind="ibm_om_physical_mapping"),
        "dev_mapping_receipt": _registered(run_dir, artifacts, "artifacts/development/selected_physical_mapping.receipt.json", kind="ibm_om_physical_mapping_receipt"),
        "held_joint": _registered(run_dir, artifacts, "artifacts/heldout/joint_assignment.npz", kind="ibm_om_joint_assignment"),
        "held_joint_receipt": _registered(run_dir, artifacts, "artifacts/heldout/joint_assignment.receipt.json", kind="ibm_om_joint_assignment_receipt"),
        "held_mapping": _registered(run_dir, artifacts, "artifacts/heldout/physical_mapping.npz", kind="ibm_om_physical_mapping"),
        "held_mapping_receipt": _registered(run_dir, artifacts, "artifacts/heldout/physical_mapping.receipt.json", kind="ibm_om_physical_mapping_receipt"),
        "predictions": _registered(run_dir, artifacts, "artifacts/heldout/predictions.npz", kind="predictions"),
        "predictions_receipt": _registered(run_dir, artifacts, "artifacts/heldout/predictions.receipt.json", kind="predictions_receipt"),
        "weight_errors": _registered(run_dir, artifacts, "artifacts/heldout/weight_errors.npz", kind="weight_error_decomposition"),
        "weight_errors_receipt": _registered(run_dir, artifacts, "artifacts/heldout/weight_errors.receipt.json", kind="weight_error_decomposition_receipt"),
        "baseline_residual": _registered(run_dir, artifacts, "artifacts/heldout/baseline_rail_residual.npz", kind="baseline_rail_residual"),
        "baseline_residual_receipt": _registered(run_dir, artifacts, "artifacts/heldout/baseline_rail_residual.receipt.json", kind="baseline_rail_residual_receipt"),
    }
    summary = _json(required["summary"], label="scientific summary")
    _equal(summary.get("schema"), SUMMARY_SCHEMA, label="scientific-summary schema")
    _equal(summary.get("schema_version"), SUMMARY_SCHEMA_VERSION, label="scientific-summary version")
    _equal(summary.get("status"), "complete", label="scientific-summary status")
    _equal(summary.get("policy"), expected_policy, label="scientific-summary policy")
    _equal(summary.get("contract"), protocol, label="scientific-summary contract")
    _equal(summary.get("source", {}).get("sha256"), EXPECTED_WEIGHTS_SHA256, label="scientific-summary source hash")
    _equal(summary.get("source", {}).get("optimizer_updates"), 0, label="source optimizer updates")
    data_provenance = _validate_data_provenance(summary.get("data"))
    expected_metric_ids = {"primary": CONTINUOUS_METRIC_DEFINITION, "diagnostic": STANDARD4DELTA_METRIC_DEFINITION}
    _equal(summary.get("metric_definition_ids"), expected_metric_ids, label="metric definition IDs")
    expected_validity_gates = {
        "development_joint_policy_coverage_complete": True,
        "heldout_joint_policy_coverage_complete": True,
        "development_repaired_population_has_no_corrupt_cells": True,
        "heldout_repaired_population_has_no_corrupt_cells": True,
        "development_full_conductance_invariants": True,
        "heldout_full_conductance_invariants": True,
        "development_candidate_count_is_16": True,
        "standard4delta_uses_continuous_selected_calibration": True,
        "expected_evaluation_examples": True,
        "optimizer_updates_are_zero": True,
        "target_write_noise_is_zero": True,
        "inference_read_noise_is_zero": True,
    }
    observed_examples = {
        "continuous": EXPECTED_EXAMPLES,
        "standard4delta": EXPECTED_EXAMPLES,
        "baseline_only": EXPECTED_EXAMPLES,
        "baseline_rail_residual": EXPECTED_EXAMPLES,
        "predictions": EXPECTED_EXAMPLES,
    }
    validity = summary.get("validity")
    _equal(
        validity,
        {
            "coverage_valid": True,
            "gates": expected_validity_gates,
            "expected_examples": EXPECTED_EXAMPLES,
            "observed_examples": observed_examples,
            "optimizer_updates": 0,
            "program_verify_enabled": False,
            "hardware_aware_training_enabled": False,
            "target_write_noise": 0.0,
            "inference_read_noise": 0.0,
            "retention_or_drift": False,
            "noise_partition": {
                "commissioning_reset_pulse_and_apparent_read_noise": "enabled_and_frozen",
                "mapped_target_write_noise": "disabled",
                "inference_read_noise": "disabled",
            },
        },
        label="ideal-mapped validity boundary",
    )
    calibration = _json(required["calibration"], label="development calibration")
    _equal(calibration.get("schema"), CALIBRATION_SCHEMA, label="calibration schema")
    _equal(calibration.get("schema_version"), 1, label="calibration version")
    _equal(calibration.get("policy"), expected_policy, label="calibration policy")
    _equal(calibration.get("assignment_seed"), DEVELOPMENT_ASSIGNMENT_SEED, label="calibration seed")
    _equal(calibration.get("test_labels_used"), False, label="calibration test-label exclusion")
    _equal(calibration.get("candidate_count"), 16, label="calibration candidate count")
    if not isinstance(calibration.get("candidates"), list) or len(calibration["candidates"]) != 16:
        _fail("Expected exactly sixteen development calibration candidates.")
    for index, candidate in enumerate(calibration["candidates"]):
        if not isinstance(candidate, dict):
            _fail(f"Expected calibration candidate {index} to be an object.")
        _exact_keys(
            candidate,
            {"scale_fractions", "calibration", "target_hashes"},
            label=f"calibration candidate {index}",
        )
        scales = candidate["scale_fractions"]
        details = candidate["calibration"]
        hashes = candidate["target_hashes"]
        if (
            not isinstance(scales, list)
            or len(scales) != 2
            or any(value not in protocol["mapping"]["scale_fractions"] for value in scales)
            or not isinstance(details, dict)
            or isinstance(details.get("student_correct"), bool)
            or not isinstance(details.get("student_correct"), int)
            or not isinstance(details.get("calibrated_kl"), (int, float))
            or not math.isfinite(float(details["calibrated_kl"]))
            or not isinstance(hashes, list)
            or len(hashes) != 2
        ):
            _fail(f"Invalid calibration candidate {index}.")
        for digest in hashes:
            _digest(digest, label="calibration target tensor")
    recomputed_selected_index = min(
        range(len(calibration["candidates"])),
        key=lambda index: (
            -int(calibration["candidates"][index]["calibration"]["student_correct"]),
            float(calibration["candidates"][index]["calibration"]["calibrated_kl"]),
            tuple(float(value) for value in calibration["candidates"][index]["scale_fractions"]),
        ),
    )
    _equal(calibration.get("selected_index"), recomputed_selected_index, label="recomputed calibration selected index")
    _equal(calibration.get("selected"), calibration["candidates"][recomputed_selected_index], label="selected calibration candidate")
    _equal(calibration.get("calibration_labels_sha256"), data_provenance["calibration_labels_sha256"], label="calibration-label provenance")
    _equal(calibration.get("calibration_indices_sha256"), data_provenance["calibration_indices_sha256"], label="calibration-index provenance")
    if not isinstance(calibration.get("analysis_scales"), list) or len(calibration["analysis_scales"]) != 2 or any(not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0 for value in calibration["analysis_scales"]):
        _fail("Expected two finite positive frozen analysis scales.")
    dev_joint = _validate_joint(required["dev_joint"], required["dev_joint_receipt"], assignment_seed=DEVELOPMENT_ASSIGNMENT_SEED, assignment_role="development")
    held_joint = _validate_joint(required["held_joint"], required["held_joint_receipt"], assignment_seed=int(heldout_seed), assignment_role="heldout")
    _equal(calibration.get("hardware_instance_id"), dev_joint["hardware_instance_id"], label="calibration hardware ID")
    development = summary.get("development", {})
    heldout = summary.get("heldout", {})
    _equal(development.get("hardware", {}).get("hardware_instance_id"), dev_joint["hardware_instance_id"], label="summary development hardware ID")
    _equal(heldout.get("hardware", {}).get("hardware_instance_id"), held_joint["hardware_instance_id"], label="summary held-out hardware ID")
    _equal(development.get("hardware", {}).get("repair"), dev_joint["report"], label="summary development repair")
    _equal(heldout.get("hardware", {}).get("repair"), held_joint["report"], label="summary held-out repair")
    _validate_hardware_sources(
        run_dir,
        artifacts,
        role="development",
        assignment_seed=DEVELOPMENT_ASSIGNMENT_SEED,
        protocol=protocol,
        hardware_summary=development["hardware"],
        joint_report=dev_joint["report"],
    )
    _validate_hardware_sources(
        run_dir,
        artifacts,
        role="heldout",
        assignment_seed=int(heldout_seed),
        protocol=protocol,
        hardware_summary=heldout["hardware"],
        joint_report=held_joint["report"],
    )
    nominal_dev = float(development.get("hardware", {}).get("nominal_dw_min"))
    nominal_held = float(heldout.get("hardware", {}).get("nominal_dw_min"))
    if nominal_dev <= 0 or nominal_held <= 0:
        _fail("Expected positive OM nominal dw_min values.")
    dev_mapping = _validate_mapping(required["dev_mapping"], required["dev_mapping_receipt"], policy=expected_policy, assignment_seed=DEVELOPMENT_ASSIGNMENT_SEED, assignment_role="development", hardware_instance_id=dev_joint["hardware_instance_id"], nominal_dw_min=nominal_dev)
    held_mapping = _validate_mapping(required["held_mapping"], required["held_mapping_receipt"], policy=expected_policy, assignment_seed=int(heldout_seed), assignment_role="heldout", hardware_instance_id=held_joint["hardware_instance_id"], nominal_dw_min=nominal_held)
    _equal(development.get("mapping"), dev_mapping["mapping"], label="summary development mapping")
    _equal(heldout.get("mapping"), held_mapping["mapping"], label="summary held-out mapping")
    _equal(calibration.get("continuous_mapping_artifact_sha256"), sha256_file(required["dev_mapping"]), label="selected development mapping hash")
    selected_scales = calibration.get("selected", {}).get("scale_fractions")
    selected_gain = calibration.get("selected", {}).get("calibration", {}).get("gain")
    _equal(heldout.get("scale_fractions"), selected_scales, label="held-out frozen scale pair")
    _equal(heldout.get("fixed_logit_gain"), selected_gain, label="held-out fixed gain")
    predictions = _validate_predictions(required["predictions"], required["predictions_receipt"], policy=expected_policy, assignment_seed=int(heldout_seed))
    _equal(predictions["labels_sha256"], data_provenance["test_targets_sha256"], label="prediction labels versus MNIST targets")
    continuous = heldout.get("ideal_bounded_continuous_init")
    standard = heldout.get("ideal_bounded_standard4delta_init")
    if not isinstance(continuous, dict) or not isinstance(standard, dict):
        _fail("Expected continuous and four-delta held-out metrics.")
    _metric_from_predictions(continuous, predictions, "continuous")
    _metric_from_predictions(standard, predictions, "standard4delta")
    logical_output_sizes = [
        int(held_mapping["tensors"][layer]["source_weight"].shape[1])
        for layer in range(2)
    ]
    baseline_residual = _validate_baseline_residual(
        required["baseline_residual"],
        required["baseline_residual_receipt"],
        logical_output_sizes=logical_output_sizes,
    )
    _equal(heldout.get("baseline_rail_voltage_residual"), baseline_residual, label="summary baseline rail residual")
    if not isinstance(heldout.get("baseline_only_full_circuit"), dict) or heldout["baseline_only_full_circuit"].get("examples") != EXPECTED_EXAMPLES:
        _fail("Expected baseline-only full-circuit metrics for all examples.")
    weight_reports = _validate_weight_errors(required["weight_errors"], required["weight_errors_receipt"], policy=expected_policy, assignment_seed=int(heldout_seed), mapping=held_mapping, analysis_scales=calibration["analysis_scales"])
    _equal(heldout.get("weight_errors"), weight_reports, label="summary weight errors")
    metrics = result.get("metrics")
    if not isinstance(metrics, dict):
        _fail("Expected terminal result metrics.")
    _equal(metrics.get("metric_definition_ids"), expected_metric_ids, label="terminal metric IDs")
    _equal(metrics.get("policy"), expected_policy, label="terminal policy")
    _equal(metrics.get("heldout_assignment_seed"), heldout_seed, label="terminal held-out seed")
    _equal(metrics.get("development_hardware_instance_id"), dev_joint["hardware_instance_id"], label="terminal development hardware ID")
    _equal(metrics.get("heldout_hardware_instance_id"), held_joint["hardware_instance_id"], label="terminal held-out hardware ID")
    _equal(metrics.get("selected_scale_fractions"), selected_scales, label="terminal scale pair")
    _equal(metrics.get("fixed_logit_gain"), selected_gain, label="terminal gain")
    _equal(metrics.get("coverage_valid"), True, label="terminal coverage")
    _equal(metrics.get("validity_gates"), expected_validity_gates, label="terminal validity gates")
    _equal(metrics.get("expected_examples"), EXPECTED_EXAMPLES, label="terminal expected examples")
    _equal(metrics.get("optimizer_updates"), 0, label="terminal optimizer updates")
    terminal_keys = ("student_correct", "examples", "student_accuracy", "teacher_correct", "teacher_accuracy", "teacher_agreement_count", "teacher_agreement", "prediction_sha256")
    _equal(metrics.get("ideal_bounded_continuous_init"), {key: continuous[key] for key in terminal_keys}, label="terminal continuous metric")
    _equal(metrics.get("ideal_bounded_standard4delta_init"), {key: standard[key] for key in terminal_keys}, label="terminal four-delta metric")
    semantic_calibration = {key: value for key, value in calibration.items() if key != "continuous_mapping_artifact_sha256"}
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "arm_id": arm["arm_id"],
        "policy": expected_policy,
        "heldout_assignment_seed": int(heldout_seed),
        "run_source": run_source,
        "development_hardware_instance_id": dev_joint["hardware_instance_id"],
        "heldout_hardware_instance_id": held_joint["hardware_instance_id"],
        "source_sha256": EXPECTED_WEIGHTS_SHA256,
        "source_logical_weight_hashes": summary.get("source", {}).get("logical_weight_hashes"),
        "data_provenance": data_provenance,
        "selected_scale_fractions": selected_scales,
        "fixed_logit_gain": selected_gain,
        "continuous": {key: continuous[key] for key in terminal_keys},
        "standard4delta": {key: standard[key] for key in terminal_keys},
        "development_calibration_semantic": semantic_calibration,
        "development_mapping": {"layers": dev_mapping["layers"]},
        "heldout_mapping": {"layers": held_mapping["layers"]},
        "weight_errors": weight_reports,
        "registered_artifact_count": len(artifacts),
    }


def _discover(study_dir: Path, study: dict[str, Any]) -> list[dict[str, Any]]:
    study["_root"] = str(study_dir)
    if study["study_id"] != STUDY_ID:
        _fail(f"Expected study_id {STUDY_ID!r}.")
    if len(study["arms"]) != len(ARM_POLICIES):
        _fail("Expected exactly four declared study arms.")
    current_plan = Path(study["source_plan"]["path"])
    if not current_plan.is_file() or sha256_file(current_plan) != study["source_plan"]["sha256"]:
        _fail("Prepared study source-plan hash no longer matches its tracked file.")
    complete: list[tuple[Path, Mapping[str, Any]]] = []
    for arm in study["arms"]:
        arm_id = arm["arm_id"]
        if arm_id not in ARM_POLICIES or arm.get("experiment_id") != EXPERIMENT_ID or arm.get("mode") != "validate" or len(arm.get("configs", [])) != 3:
            _fail(f"Unexpected arm contract for {arm_id!r}.")
        arm_dir = study_dir / "runs" / arm_id
        if not arm_dir.is_dir():
            _fail(f"Missing run arm directory {arm_dir}.")
        for candidate in sorted(path for path in arm_dir.iterdir() if path.is_dir()):
            status_path = candidate / "status.json"
            if not status_path.is_file():
                _fail(f"Expected every run directory to contain status.json: {candidate}.")
            status = _json(status_path, label="run status")
            state = status.get("status")
            if state == "complete":
                complete.append((candidate, arm))
            elif state == "running":
                _fail(f"Study still has a running bundle: {candidate}.")
            elif state != "failed":
                _fail(f"Invalid run status {state!r}: {candidate}.")
    if len(complete) != EXPECTED_RUNS:
        _fail(f"Expected exactly {EXPECTED_RUNS} complete RunStore bundles; found {len(complete)}.")
    config_owner: dict[str, tuple[Mapping[str, Any], str]] = {}
    for arm in study["arms"]:
        for config in arm["configs"]:
            config_path = Path(config["resolved_path"])
            if not config_path.is_file() or sha256_file(config_path) != config["sha256"]:
                _fail(f"Declared config hash no longer matches {config_path}.")
            config_owner[config["sha256"]] = (arm, config["sha256"])
    seen_configs: set[str] = set()
    records = []
    for run_dir, directory_arm in complete:
        manifest = _json(run_dir / "manifest.json", label="run manifest")
        config_hash = manifest.get("study", {}).get("source_config_sha256")
        owner = config_owner.get(config_hash)
        if owner is None or owner[0]["arm_id"] != directory_arm["arm_id"]:
            _fail(f"Complete run is not linked to a declared config in its arm: {run_dir}.")
        if config_hash in seen_configs:
            _fail(f"More than one complete run claims config hash {config_hash}.")
        seen_configs.add(config_hash)
        records.append(_validate_run(run_dir, study=study, arm=directory_arm, expected_config_hash=config_hash, expected_policy=ARM_POLICIES[directory_arm["arm_id"]]))
    if seen_configs != set(config_owner):
        _fail("Complete run coverage does not equal the twelve declared configs.")
    coverage = {(record["policy"], record["heldout_assignment_seed"]) for record in records}
    expected_coverage = {(policy, seed) for policy in BASELINE_POLICIES for seed in HELDOUT_ASSIGNMENT_SEEDS}
    if coverage != expected_coverage:
        _fail("Run policy/held-out-seed coverage is not exactly 4 x 3.")
    dev_ids = {record["development_hardware_instance_id"] for record in records}
    if len(dev_ids) != 1:
        _fail("Cross-arm development hardware_instance_id values differ.")
    source_states = {canonical_json_bytes(record["source_logical_weight_hashes"]) for record in records}
    if len(source_states) != 1:
        _fail("Frozen logical source hashes differ across runs.")
    run_source_states = {
        canonical_json_bytes(record["run_source"]) for record in records
    }
    if len(run_source_states) != 1:
        _fail("Run source commit or clean-state provenance differs across runs.")
    data_states = {canonical_json_bytes(record["data_provenance"]) for record in records}
    if len(data_states) != 1:
        _fail("MNIST test/calibration provenance differs across runs.")
    for seed in HELDOUT_ASSIGNMENT_SEEDS:
        ids = {record["heldout_hardware_instance_id"] for record in records if record["heldout_assignment_seed"] == seed}
        if len(ids) != 1:
            _fail(f"Cross-arm held-out hardware_instance_id differs at seed {seed}.")
    for policy in BASELINE_POLICIES:
        semantics = {canonical_json_bytes(record["development_calibration_semantic"]) for record in records if record["policy"] == policy}
        if len(semantics) != 1:
            _fail(f"Development calibration is not semantically identical across held-out runs for {policy}.")
    return sorted(records, key=lambda value: (BASELINE_POLICIES.index(value["policy"]), value["heldout_assignment_seed"]))


def _aggregate(records: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    policies: dict[str, Any] = {}
    for policy in BASELINE_POLICIES:
        rows = [record for record in records if record["policy"] == policy]
        continuous_correct = [int(record["continuous"]["student_correct"]) for record in rows]
        standard_correct = [int(record["standard4delta"]["student_correct"]) for record in rows]
        continuous_accuracy = [value / EXPECTED_EXAMPLES for value in continuous_correct]
        standard_accuracy = [value / EXPECTED_EXAMPLES for value in standard_correct]
        policies[policy] = {
            "heldout_assignment_seeds": [record["heldout_assignment_seed"] for record in rows],
            "continuous": {
                "correct": continuous_correct,
                "examples_each": EXPECTED_EXAMPLES,
                "accuracy_mean": sum(continuous_accuracy) / len(rows),
                "accuracy_minimum": min(continuous_accuracy),
                "accuracy_maximum": max(continuous_accuracy),
                "accuracy_range": max(continuous_accuracy) - min(continuous_accuracy),
            },
            "standard4delta_diagnostic": {
                "correct": standard_correct,
                "examples_each": EXPECTED_EXAMPLES,
                "accuracy_mean": sum(standard_accuracy) / len(rows),
                "accuracy_minimum": min(standard_accuracy),
                "accuracy_maximum": max(standard_accuracy),
                "accuracy_range": max(standard_accuracy) - min(standard_accuracy),
            },
            "weight_error_relative_l2_mean_by_layer": [
                sum(record["weight_errors"][layer]["continuous_endpoint"]["relative_l2"] for record in rows) / len(rows)
                for layer in range(2)
            ],
            "baseline_contrast_rms_mean_by_layer": [
                sum(record["heldout_mapping"]["layers"][layer]["baseline_contrast"]["rms"] for record in rows) / len(rows)
                for layer in range(2)
            ],
            "baseline_loading_mean_by_layer": [
                sum(record["heldout_mapping"]["layers"][layer]["baseline_loading"]["mean"] for record in rows) / len(rows)
                for layer in range(2)
            ],
            "selected_level_capacity_mean_by_layer": [
                sum(record["heldout_mapping"]["layers"][layer]["selected_level_capacity"]["mean"] for record in rows) / len(rows)
                for layer in range(2)
            ],
        }
    ordered = sorted(BASELINE_POLICIES, key=lambda policy: (-policies[policy]["continuous"]["accuracy_mean"], policy))
    best, runner = ordered[:2]
    strict_each_seed = all(
        next(record for record in records if record["policy"] == best and record["heldout_assignment_seed"] == seed)["continuous"]["student_correct"]
        > max(record["continuous"]["student_correct"] for record in records if record["policy"] != best and record["heldout_assignment_seed"] == seed)
        for seed in HELDOUT_ASSIGNMENT_SEEDS
    )
    mean_lead = policies[best]["continuous"]["accuracy_mean"] - policies[runner]["continuous"]["accuracy_mean"]
    unique = strict_each_seed and mean_lead >= 0.01
    winner = best if unique else None
    best_mean = policies[best]["continuous"]["accuracy_mean"]
    gate_passed = best_mean >= 0.90
    decision = {
        "ranking_metric": CONTINUOUS_METRIC_DEFINITION,
        "best_observed_policy": best,
        "runner_up_policy": runner,
        "best_mean_accuracy": best_mean,
        "mean_lead_over_runner_up": mean_lead,
        "strictly_more_correct_on_all_three_paired_assignments": strict_each_seed,
        "minimum_mean_lead": 0.01,
        "predeclared_unique_winner": winner,
        "winner_status": "winner" if winner is not None else "inconclusive",
        "initialization_gate": {
            "threshold": 0.90,
            "evaluated_policy": winner,
            "observed_best_policy": best,
            "observed_best_mean_accuracy": best_mean,
            "passed": bool(winner is not None and gate_passed),
            "status": (
                "passed" if winner is not None and gate_passed
                else "failed" if winner is not None
                else "inconclusive"
            ),
        },
        "overall_status": (
            "winner_and_initialization_gate_passed"
            if winner is not None and gate_passed
            else "winner_but_initialization_gate_failed"
            if winner is not None
            else "inconclusive"
        ),
        "standard4delta_role": "diagnostic_only_not_used_for_ranking",
    }
    return policies, decision


def _csv(records: Sequence[Mapping[str, Any]]) -> str:
    columns = [
        "policy", "arm_id", "heldout_assignment_seed", "run_id",
        "continuous_correct", "continuous_examples", "continuous_accuracy",
        "standard4delta_correct", "standard4delta_examples", "standard4delta_accuracy",
        "teacher_correct", "continuous_teacher_agreement_count",
        "development_hardware_instance_id", "heldout_hardware_instance_id",
        "selected_scale_layer0", "selected_scale_layer1", "fixed_logit_gain",
        "layer0_baseline_contrast_rms", "layer1_baseline_contrast_rms",
        "layer0_baseline_loading_mean", "layer1_baseline_loading_mean",
        "layer0_selected_capacity_mean", "layer1_selected_capacity_mean",
        "layer0_weight_total_rmse", "layer1_weight_total_rmse",
        "layer0_weight_relative_l2", "layer1_weight_relative_l2",
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for record in records:
        writer.writerow({
            "policy": record["policy"], "arm_id": record["arm_id"],
            "heldout_assignment_seed": record["heldout_assignment_seed"], "run_id": record["run_id"],
            "continuous_correct": record["continuous"]["student_correct"], "continuous_examples": EXPECTED_EXAMPLES,
            "continuous_accuracy": record["continuous"]["student_accuracy"],
            "standard4delta_correct": record["standard4delta"]["student_correct"], "standard4delta_examples": EXPECTED_EXAMPLES,
            "standard4delta_accuracy": record["standard4delta"]["student_accuracy"],
            "teacher_correct": record["continuous"]["teacher_correct"],
            "continuous_teacher_agreement_count": record["continuous"]["teacher_agreement_count"],
            "development_hardware_instance_id": record["development_hardware_instance_id"],
            "heldout_hardware_instance_id": record["heldout_hardware_instance_id"],
            "selected_scale_layer0": record["selected_scale_fractions"][0],
            "selected_scale_layer1": record["selected_scale_fractions"][1],
            "fixed_logit_gain": record["fixed_logit_gain"],
            **{f"layer{layer}_baseline_contrast_rms": record["heldout_mapping"]["layers"][layer]["baseline_contrast"]["rms"] for layer in range(2)},
            **{f"layer{layer}_baseline_loading_mean": record["heldout_mapping"]["layers"][layer]["baseline_loading"]["mean"] for layer in range(2)},
            **{f"layer{layer}_selected_capacity_mean": record["heldout_mapping"]["layers"][layer]["selected_level_capacity"]["mean"] for layer in range(2)},
            **{f"layer{layer}_weight_total_rmse": record["weight_errors"][layer]["teacher_weight_error"]["continuous_total"]["rmse"] for layer in range(2)},
            **{f"layer{layer}_weight_relative_l2": record["weight_errors"][layer]["continuous_endpoint"]["relative_l2"] for layer in range(2)},
        })
    return stream.getvalue()


def _markdown(study: Mapping[str, Any], policies: Mapping[str, Any], decision: Mapping[str, Any]) -> str:
    lines = [
        f"# {study['title']} — audited analysis", "",
        "All 12 production RunStore bundles passed registered-hash, schema, source, matched-hardware, full-conductance, bound, exact-zero, prediction-count, and frozen-calibration checks.", "",
        "| Policy | Continuous correct / 10000 | Continuous mean (range) | 4delta mean (range) |",
        "|---|---:|---:|---:|",
    ]
    for policy in BASELINE_POLICIES:
        value = policies[policy]
        continuous = value["continuous"]
        quantized = value["standard4delta_diagnostic"]
        lines.append(
            f"| `{policy}` | {', '.join(str(number) for number in continuous['correct'])} | "
            f"{100 * continuous['accuracy_mean']:.2f}% ({100 * continuous['accuracy_minimum']:.2f}–{100 * continuous['accuracy_maximum']:.2f}%) | "
            f"{100 * quantized['accuracy_mean']:.2f}% ({100 * quantized['accuracy_minimum']:.2f}–{100 * quantized['accuracy_maximum']:.2f}%) |"
        )
    lines.extend(["", "## Predeclared decision", ""])
    if decision["predeclared_unique_winner"] is None:
        lines.append("The baseline-policy comparison is **inconclusive** under the predeclared unique-winner rule.")
    else:
        lines.append(
            f"The predeclared unique winner is `{decision['predeclared_unique_winner']}` with a "
            f"{100 * decision['mean_lead_over_runner_up']:.2f}-percentage-point mean lead."
        )
    gate = decision["initialization_gate"]
    lines.extend([
        "",
        f"Initialization gate: **{gate['status']}** (threshold 90.00%; observed best mean {100 * gate['observed_best_mean_accuracy']:.2f}%).",
        "",
        "The four-delta result is a frozen diagnostic and was not used to choose or rank the baseline policy.",
        "",
    ])
    return "\n".join(lines)


def analyze_study(study_dir: Path | str, output_dir: Path | str | None = None) -> dict[str, Any]:
    root = Path(study_dir).expanduser().resolve()
    study = load_study_record(root)
    records = _discover(root, study)
    policies, decision = _aggregate(records)
    output = (root / "analysis") if output_dir is None else Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    public_records = [{key: value for key, value in record.items() if key != "development_calibration_semantic"} for record in records]
    report = {
        "schema": ANALYSIS_SCHEMA,
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "study_id": study["study_id"],
        "claim_boundary": "four-device_model_based_ideal_bounded_initialization_without_training_or_write_read_noise",
        "audit": {
            "complete_run_count": len(records),
            "expected_run_count": EXPECTED_RUNS,
            "production_profile": "production",
            "runtime_device": "cuda",
            "examples_per_run": EXPECTED_EXAMPLES,
            "source_sha256": EXPECTED_WEIGHTS_SHA256,
            "run_source_commit": records[0]["run_source"]["commit"],
            "development_assignment_seed": DEVELOPMENT_ASSIGNMENT_SEED,
            "heldout_assignment_seeds": list(HELDOUT_ASSIGNMENT_SEEDS),
            "registered_artifacts_verified": True,
            "full_g_recomputed": True,
            "prediction_counts_and_hashes_recomputed": True,
            "cross_arm_hardware_matched": True,
            "within_policy_development_calibration_matched": True,
        },
        "runs": public_records,
        "policies": policies,
        "decision": decision,
        "outputs": {"json": OUTPUT_JSON, "csv": OUTPUT_CSV, "markdown": OUTPUT_MARKDOWN},
    }
    atomic_write_json(output / OUTPUT_JSON, report)
    _atomic_text(output / OUTPUT_CSV, _csv(records))
    _atomic_text(output / OUTPUT_MARKDOWN, _markdown(study, policies, decision))
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    analyze_study(arguments.study_dir, arguments.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ANALYSIS_SCHEMA",
    "ANALYSIS_SCHEMA_VERSION",
    "BaselineSelectionAnalysisError",
    "analyze_study",
    "main",
]
