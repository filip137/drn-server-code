#!/usr/bin/env python3
"""Directional Hessian response metrics for legacy MNIST DRN-XS amplification runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
LABS_DIR = REPO_ROOT / "labs"
TOOLS_DIR = LABS_DIR / "tools"
for path in (REPO_ROOT, LABS_DIR, TOOLS_DIR, Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import evaluate_mnist_bp_hessian_lpw_amplification_sweep as hess_eval  # noqa: E402


DEFAULT_INPUT_ROOT = (
    REPO_ROOT / "results" / "mnist_bp_hessian_lpw_amplification_sweep_legacy_drnxs"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "results" / "mnist_bp_hessian_directional_response_legacy_drnxs"
)
RUN_ORDER = hess_eval.RUN_ORDER
EPS = 1e-12

SAMPLE_COLUMNS = [
    "run_name",
    "seed",
    "voltage_amp",
    "current_amp",
    "sample_index",
    "label",
    "prediction",
    "correct",
    "lpw_active_hidden_count",
    "lpw_active_hidden_fraction",
    "clean_margin",
    "current_gain_all_rms",
    "current_gain_hidden_rms",
    "current_gain_output_rms",
    "current_gain_all_spectral",
    "current_hidden_state_gain_all_rms",
    "input_noise_gain_rms",
    "input_noise_gain_spectral",
    "write_abs_gain_rms",
    "write_abs_gain_spectral",
    "write_rel_gain_rms",
    "write_rel_gain_spectral",
    "teaching_hidden_gain_rms",
    "teaching_hidden_gain_spectral",
    "teaching_output_gain_rms",
    "teaching_output_gain_spectral",
    "teaching_hidden_over_current_hidden_state",
    "teaching_output_over_current_output",
    "margin_norm_current_all",
    "margin_norm_current_hidden",
    "margin_norm_current_output",
    "margin_norm_input_noise",
    "margin_norm_write_abs",
    "margin_norm_write_rel",
]

SUMMARY_METRIC_KEYS = [
    "lpw_active_hidden_fraction",
    "clean_margin",
    "current_gain_all_rms",
    "current_gain_hidden_rms",
    "current_gain_output_rms",
    "current_gain_all_spectral",
    "current_hidden_state_gain_all_rms",
    "input_noise_gain_rms",
    "input_noise_gain_spectral",
    "write_abs_gain_rms",
    "write_abs_gain_spectral",
    "write_rel_gain_rms",
    "write_rel_gain_spectral",
    "teaching_hidden_gain_rms",
    "teaching_hidden_gain_spectral",
    "teaching_output_gain_rms",
    "teaching_output_gain_spectral",
    "teaching_hidden_over_current_hidden_state",
    "teaching_output_over_current_output",
    "margin_norm_current_all",
    "margin_norm_current_hidden",
    "margin_norm_current_output",
    "margin_norm_input_noise",
    "margin_norm_write_abs",
    "margin_norm_write_rel",
]


@dataclass(frozen=True)
class DirectionalRunSpec:
    run_name: str
    seed: int
    voltage_amp: float
    current_amp: float
    checkpoint_path: Path
    source_config_path: Path
    resolved_config_path: Path
    hessian_run_dir: Path
    hessian_linear_path: Path
    lpw_masks_path: Path
    hessian_summary_path: Path


def finite_float(value: object) -> float:
    return hess_eval.finite_float(value)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    hess_eval.write_csv(path, rows, fieldnames)


def stats(values: np.ndarray, prefix: str) -> dict[str, float]:
    return hess_eval.stats(values, prefix)


def summary_sort_key(row: dict[str, object]) -> tuple[int, int]:
    run_name = str(row.get("run_name", ""))
    try:
        run_idx = RUN_ORDER.index(run_name)
    except ValueError:
        run_idx = len(RUN_ORDER)
    return run_idx, int(row.get("seed", 0))


def read_run_specs(
    input_root: Path,
    *,
    run_names: set[str] | None,
    seeds: set[int] | None,
) -> list[DirectionalRunSpec]:
    summary_path = input_root / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Expected Hessian summary.csv at {summary_path}.")
    rows = sorted(read_csv(summary_path), key=summary_sort_key)
    specs: list[DirectionalRunSpec] = []
    for row in rows:
        run_name = row["run_name"]
        seed = int(row["seed"])
        if run_names is not None and run_name not in run_names:
            continue
        if seeds is not None and seed not in seeds:
            continue
        hessian_summary_path = Path(row["summary_path"])
        specs.append(
            DirectionalRunSpec(
                run_name=run_name,
                seed=seed,
                voltage_amp=float(row["voltage_amp"]),
                current_amp=float(row["current_amp"]),
                checkpoint_path=Path(row["checkpoint_path"]),
                source_config_path=Path(row["source_config_path"]),
                resolved_config_path=Path(row["resolved_config_path"]),
                hessian_run_dir=hessian_summary_path.parent,
                hessian_linear_path=Path(row["hessian_linear_path"]),
                lpw_masks_path=Path(row["lpw_masks_path"]),
                hessian_summary_path=hessian_summary_path,
            )
        )
    if not specs:
        raise ValueError("No run specs matched the requested filters.")
    return specs


def as_hessian_run_spec(spec: DirectionalRunSpec) -> hess_eval.RunSpec:
    return hess_eval.RunSpec(
        run_name=spec.run_name,
        seed=spec.seed,
        voltage_amp=spec.voltage_amp,
        current_amp=spec.current_amp,
        run_dir=spec.checkpoint_path.parent,
        checkpoint_path=spec.checkpoint_path,
        source_config_path=spec.source_config_path,
        resolved_config_path=spec.resolved_config_path,
    )


def run_output_dir(output_root: Path, spec: DirectionalRunSpec) -> Path:
    return output_root / spec.run_name / f"seed_{spec.seed}"


def run_complete(output_root: Path, spec: DirectionalRunSpec) -> bool:
    run_dir = run_output_dir(output_root, spec)
    required = [
        "directional_sample_metrics.csv",
        "directional_summary.json",
        "operator_metrics_by_mask.npz",
        "config.json",
    ]
    return all((run_dir / name).exists() for name in required)


def raw_output_logits(output: torch.Tensor) -> torch.Tensor:
    if output.shape[1] == 10:
        return output
    if output.shape[1] == 20:
        return output[:, :10] - output[:, 10:]
    raise ValueError(f"Unsupported output shape {tuple(output.shape)}.")


def collect_test_states(context: dict, max_test_samples: int | None) -> dict[str, np.ndarray]:
    network = context["network"]
    minimizer = context["minimizer"]
    energy_fn = context["energy_fn"]
    input_layer = energy_fn.layers()[0]
    free_layers = context["free_layers"]
    output_layer = context["output_layer"]
    test_loader = context["test_loader"]

    inputs_all: list[np.ndarray] = []
    hidden_all: list[np.ndarray] = []
    outputs_all: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    preds_all: list[np.ndarray] = []
    correct_all: list[np.ndarray] = []
    seen = 0

    for images, labels in test_loader:
        if max_test_samples is not None and seen >= max_test_samples:
            break
        remaining = None if max_test_samples is None else max_test_samples - seen
        if remaining is not None and images.shape[0] > remaining:
            images = images[:remaining]
            labels = labels[:remaining]
        images = images.to(energy_fn._device)
        labels = labels.to(energy_fn._device)

        network.set_input(images, reset=True)
        minimizer.compute_equilibrium()

        output_state = output_layer.state.detach()
        logits = raw_output_logits(output_state)
        preds = torch.argmax(logits, dim=1)
        correct = preds.eq(labels)

        inputs_all.append(
            input_layer.state.detach().cpu().numpy().astype(np.float64).reshape(images.shape[0], -1)
        )
        hidden_all.append(
            free_layers[0].state.detach().cpu().numpy().astype(np.float64).reshape(images.shape[0], -1)
        )
        outputs_all.append(logits.detach().cpu().numpy().astype(np.float64))
        labels_all.append(labels.detach().cpu().numpy().astype(np.int64))
        preds_all.append(preds.detach().cpu().numpy().astype(np.int64))
        correct_all.append(correct.detach().cpu().numpy().astype(bool))
        seen += int(images.shape[0])

    if seen == 0:
        raise ValueError("No test samples were evaluated.")

    return {
        "inputs": np.concatenate(inputs_all, axis=0),
        "hidden": np.concatenate(hidden_all, axis=0),
        "outputs": np.concatenate(outputs_all, axis=0),
        "labels": np.concatenate(labels_all, axis=0),
        "predictions": np.concatenate(preds_all, axis=0),
        "correct": np.concatenate(correct_all, axis=0),
    }


def load_hessian_artifacts(spec: DirectionalRunSpec) -> dict[str, np.ndarray | list[int] | float]:
    if not spec.hessian_linear_path.exists():
        raise FileNotFoundError(f"Expected hessian_linear.npz at {spec.hessian_linear_path}.")
    if not spec.lpw_masks_path.exists():
        raise FileNotFoundError(f"Expected lpw_conductance_masks.npz at {spec.lpw_masks_path}.")
    hessian_npz = np.load(spec.hessian_linear_path)
    masks_npz = np.load(spec.lpw_masks_path)
    hessian_linear = np.asarray(hessian_npz["hessian"], dtype=np.float64)
    free_dims = [int(v) for v in np.asarray(hessian_npz["free_dims"]).tolist()]
    metadata = json.loads(str(masks_npz["metadata_json"]))
    return {
        "hessian_linear": hessian_linear,
        "free_dims": free_dims,
        "saved_hidden_mask": np.asarray(masks_npz["hidden_active_mask"], dtype=bool),
        "saved_labels": np.asarray(masks_npz["labels"], dtype=np.int64),
        "saved_predictions": np.asarray(masks_npz["predictions"], dtype=np.int64),
        "saved_correct": np.asarray(masks_npz["correct"], dtype=bool),
        "unique_hidden_mask": np.asarray(masks_npz["unique_hidden_active_mask"], dtype=bool),
        "inverse_unique_mask": np.asarray(masks_npz["inverse_unique_mask"], dtype=np.int64),
        "lpw_layer_conductance": float(metadata["lpw_layer_conductance"]),
    }


def validate_saved_states(states: dict[str, np.ndarray], artifacts: dict, max_test_samples: int | None) -> None:
    n = states["labels"].shape[0]
    if max_test_samples is not None:
        expected_n = min(max_test_samples, artifacts["saved_labels"].shape[0])
        if n != expected_n:
            raise ValueError(f"Expected {expected_n} samples from max_test_samples, got {n}.")
    saved_slice = slice(0, n)
    if not np.array_equal(states["labels"], artifacts["saved_labels"][saved_slice]):
        raise ValueError("Recomputed labels do not match saved Hessian labels.")
    if not np.array_equal(states["predictions"], artifacts["saved_predictions"][saved_slice]):
        raise ValueError("Recomputed predictions do not match saved Hessian predictions.")
    if not np.array_equal(states["correct"], artifacts["saved_correct"][saved_slice]):
        raise ValueError("Recomputed correctness does not match saved Hessian correctness.")


def validate_masks(states: dict[str, np.ndarray], artifacts: dict) -> np.ndarray:
    hidden_mask = hess_eval.lpw_hidden_mask(states["hidden"])
    saved_mask = artifacts["saved_hidden_mask"][: hidden_mask.shape[0]]
    if not np.array_equal(hidden_mask, saved_mask):
        mismatch = int(np.count_nonzero(hidden_mask != saved_mask))
        raise ValueError(f"Recomputed LPW masks do not match saved Hessian masks; mismatched entries={mismatch}.")
    return hidden_mask


def dense_weights(context: dict) -> tuple[np.ndarray, np.ndarray]:
    weights = hess_eval.dense_weight_mats(context["energy_fn"])
    if len(weights) != 2:
        raise ValueError(f"Expected two dense weights for DRN-XS, got {len(weights)}.")
    return weights[0], weights[1]


def hessian_with_lpw_diag(
    hessian_linear: np.ndarray,
    hidden_mask: np.ndarray,
    *,
    lpw_layer_conductance: float,
    output_dim: int,
) -> np.ndarray:
    hessian = hessian_linear.copy()
    diag = np.concatenate(
        [
            hidden_mask.astype(np.float64) * lpw_layer_conductance,
            np.zeros(output_dim, dtype=np.float64),
        ]
    )
    hessian[np.diag_indices_from(hessian)] += diag
    return hessian


def symmetrized(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)


def max_eigval_psd(cov: np.ndarray) -> float:
    eigvals = np.linalg.eigvalsh(symmetrized(cov))
    return float(max(eigvals[-1], 0.0))


def cov_metrics(cov: np.ndarray, denom: float) -> tuple[float, float]:
    denom = float(denom)
    if denom <= 0.0:
        return math.nan, math.nan
    trace = max(float(np.trace(cov)), 0.0)
    rms = math.sqrt(trace / denom)
    spectral = math.sqrt(max_eigval_psd(cov))
    return rms, spectral


def operator_cov_metrics(operator: np.ndarray, denom: float) -> tuple[np.ndarray, float, float]:
    cov = operator @ operator.T
    rms, spectral = cov_metrics(cov, denom)
    return cov, rms, spectral


def margin_metrics(logits: np.ndarray, label: int, cov_y: np.ndarray) -> tuple[float, float]:
    num_classes = logits.shape[0]
    competitors = [idx for idx in range(num_classes) if idx != label]
    margins = np.asarray([logits[label] - logits[idx] for idx in competitors], dtype=np.float64)
    stds = []
    for idx in competitors:
        variance = cov_y[label, label] + cov_y[idx, idx] - 2.0 * cov_y[label, idx]
        stds.append(math.sqrt(max(float(variance), 0.0)))
    stds_arr = np.asarray(stds, dtype=np.float64)
    clean_margin = float(np.min(margins))
    normalized = float(np.min(margins / (stds_arr + EPS)))
    return clean_margin, normalized


def response_cov_from_residual_cov(h_inv: np.ndarray, cov_b: np.ndarray, output_slice: slice) -> np.ndarray:
    output_response = h_inv[output_slice, :]
    return output_response @ cov_b @ output_response.T


def conductance_residual_cov(
    x: np.ndarray,
    h: np.ndarray,
    y: np.ndarray,
    weight0: np.ndarray,
    weight1: np.ndarray,
    *,
    voltage_amp: float,
    current_amp: float,
    relative: bool,
) -> tuple[np.ndarray, float]:
    hidden_dim, output_dim = weight1.shape
    cov_b = np.zeros((hidden_dim + output_dim, hidden_dim + output_dim), dtype=np.float64)
    hidden_slice = slice(0, hidden_dim)
    output_slice = slice(hidden_dim, hidden_dim + output_dim)

    var0 = np.square(weight0) if relative else np.ones_like(weight0, dtype=np.float64)
    var1 = np.square(weight1) if relative else np.ones_like(weight1, dtype=np.float64)
    denom = float(np.sum(var0) + np.sum(var1))

    coef0 = (current_amp**2) * h[np.newaxis, :] - current_amp * x[:, np.newaxis]
    cov_b[hidden_slice, hidden_slice] += np.diag(np.sum(var0 * np.square(coef0), axis=0))

    p = current_amp * voltage_amp * h[:, np.newaxis] - (current_amp**2) * y[np.newaxis, :]
    q = -(current_amp**2) * h[:, np.newaxis] + ((current_amp**3) / voltage_amp) * y[np.newaxis, :]
    cov_b[hidden_slice, hidden_slice] += np.diag(np.sum(var1 * np.square(p), axis=1))
    cov_b[output_slice, output_slice] += np.diag(np.sum(var1 * np.square(q), axis=0))
    cross = var1 * p * q
    cov_b[hidden_slice, output_slice] += cross
    cov_b[output_slice, hidden_slice] += cross.T
    return cov_b, denom


def compute_mask_operators(
    h_inv: np.ndarray,
    weight0: np.ndarray,
    *,
    current_amp: float,
    hidden_dim: int,
    output_dim: int,
) -> dict[str, object]:
    hidden_slice = slice(0, hidden_dim)
    output_slice = slice(hidden_dim, hidden_dim + output_dim)
    all_dim = hidden_dim + output_dim

    output_all = h_inv[output_slice, :]
    output_hidden = h_inv[output_slice, hidden_slice]
    output_output = h_inv[output_slice, output_slice]
    hidden_all = h_inv[hidden_slice, :]

    cov_current_all, current_all_rms, current_all_spectral = operator_cov_metrics(output_all, all_dim)
    cov_current_hidden, current_hidden_rms, current_hidden_spectral = operator_cov_metrics(
        output_hidden, hidden_dim
    )
    cov_current_output, current_output_rms, current_output_spectral = operator_cov_metrics(
        output_output, output_dim
    )
    _, hidden_state_all_rms, hidden_state_all_spectral = operator_cov_metrics(hidden_all, all_dim)

    b_x = np.zeros((all_dim, weight0.shape[0]), dtype=np.float64)
    b_x[:hidden_dim, :] = -current_amp * weight0.T
    input_operator = output_all @ b_x
    cov_input, input_rms, input_spectral = operator_cov_metrics(input_operator, weight0.shape[0])

    teaching_hidden = h_inv[hidden_slice, output_slice]
    teaching_output = h_inv[output_slice, output_slice]
    _, teaching_hidden_rms, teaching_hidden_spectral = operator_cov_metrics(teaching_hidden, output_dim)
    _, teaching_output_rms, teaching_output_spectral = operator_cov_metrics(teaching_output, output_dim)

    return {
        "cov_current_all": cov_current_all,
        "cov_current_hidden": cov_current_hidden,
        "cov_current_output": cov_current_output,
        "cov_input": cov_input,
        "current_gain_all_rms": current_all_rms,
        "current_gain_hidden_rms": current_hidden_rms,
        "current_gain_output_rms": current_output_rms,
        "current_gain_all_spectral": current_all_spectral,
        "current_gain_hidden_spectral": current_hidden_spectral,
        "current_gain_output_spectral": current_output_spectral,
        "current_hidden_state_gain_all_rms": hidden_state_all_rms,
        "current_hidden_state_gain_all_spectral": hidden_state_all_spectral,
        "input_noise_gain_rms": input_rms,
        "input_noise_gain_spectral": input_spectral,
        "teaching_hidden_gain_rms": teaching_hidden_rms,
        "teaching_hidden_gain_spectral": teaching_hidden_spectral,
        "teaching_output_gain_rms": teaching_output_rms,
        "teaching_output_gain_spectral": teaching_output_spectral,
        "teaching_hidden_over_current_hidden_state": teaching_hidden_rms / (hidden_state_all_rms + EPS),
        "teaching_output_over_current_output": teaching_output_rms / (current_output_rms + EPS),
    }


def sample_write_metrics(
    h_inv: np.ndarray,
    x: np.ndarray,
    h: np.ndarray,
    y: np.ndarray,
    weight0: np.ndarray,
    weight1: np.ndarray,
    *,
    voltage_amp: float,
    current_amp: float,
    output_slice: slice,
    logits: np.ndarray,
    label: int,
    relative: bool,
) -> dict[str, float]:
    cov_b, denom = conductance_residual_cov(
        x,
        h,
        y,
        weight0,
        weight1,
        voltage_amp=voltage_amp,
        current_amp=current_amp,
        relative=relative,
    )
    cov_y = response_cov_from_residual_cov(h_inv, cov_b, output_slice)
    gain_rms, gain_spectral = cov_metrics(cov_y, denom)
    _, margin_norm = margin_metrics(logits, label, cov_y)
    prefix = "write_rel" if relative else "write_abs"
    return {
        f"{prefix}_gain_rms": gain_rms,
        f"{prefix}_gain_spectral": gain_spectral,
        f"margin_norm_{prefix}": margin_norm,
    }


def run_summary_from_samples(
    spec: DirectionalRunSpec,
    sample_rows: list[dict],
    *,
    output_paths: dict[str, Path],
    source_paths: dict[str, Path],
) -> dict:
    rows_by_key = {
        key: np.asarray([finite_float(row[key]) for row in sample_rows], dtype=np.float64)
        for key in SUMMARY_METRIC_KEYS
    }
    correct = np.asarray([int(row["correct"]) for row in sample_rows], dtype=np.float64)
    summary = {
        "run_name": spec.run_name,
        "seed": spec.seed,
        "voltage_amp": spec.voltage_amp,
        "current_amp": spec.current_amp,
        "sample_count": len(sample_rows),
        "test_accuracy_recomputed": float(np.mean(correct)) if correct.size else math.nan,
        "checkpoint_path": str(spec.checkpoint_path),
        "source_config_path": str(spec.source_config_path),
        "resolved_config_path": str(spec.resolved_config_path),
        "hessian_linear_path": str(spec.hessian_linear_path),
        "lpw_masks_path": str(spec.lpw_masks_path),
        "directional_sample_metrics_path": str(output_paths["sample_metrics"]),
        "operator_metrics_by_mask_path": str(output_paths["operator_metrics"]),
        "directional_summary_path": str(output_paths["summary"]),
        "config_path": str(output_paths["config"]),
        **{f"{key}_{stat_key.rsplit('_', 1)[-1]}": value for key, values in rows_by_key.items() for stat_key, value in stats(values, key).items()},
    }
    correct_mask = correct.astype(bool)
    for key in [
        "clean_margin",
        "margin_norm_current_all",
        "margin_norm_input_noise",
        "margin_norm_write_abs",
        "margin_norm_write_rel",
    ]:
        values = rows_by_key[key][correct_mask]
        summary.update(stats(values, f"{key}_correct"))
    summary.update({f"source_{name}": str(path) for name, path in source_paths.items()})
    return summary


def analyze_run(
    spec: DirectionalRunSpec,
    args: argparse.Namespace,
    *,
    output_root: Path,
    device: torch.device,
) -> dict:
    out_dir = run_output_dir(output_root, spec)
    if run_complete(output_root, spec) and not args.rerun_complete:
        print(f"[skip] {spec.run_name} seed={spec.seed} already complete at {out_dir}")
        return json.loads((out_dir / "directional_summary.json").read_text())

    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts = load_hessian_artifacts(spec)
    hessian_linear = artifacts["hessian_linear"]
    free_dims = artifacts["free_dims"]
    if free_dims != [100, 10]:
        raise ValueError(f"Expected DRN-XS free_dims [100, 10], got {free_dims}.")
    if hessian_linear.shape != (110, 110):
        raise ValueError(f"Expected H_linear 110x110, got {hessian_linear.shape}.")

    context = hess_eval.build_eval_context(
        as_hessian_run_spec(spec),
        device=device,
        batch_size=args.batch_size,
        dataset_root=args.dataset_root,
        no_download=args.no_download,
    )
    weight0, weight1 = dense_weights(context)
    hidden_dim, output_dim = free_dims
    if weight0.shape != (1568, hidden_dim):
        raise ValueError(f"Expected DenseWeight_0 flattened shape (1568, {hidden_dim}), got {weight0.shape}.")
    if weight1.shape != (hidden_dim, output_dim):
        raise ValueError(f"Expected DenseWeight_1 shape ({hidden_dim}, {output_dim}), got {weight1.shape}.")

    states = collect_test_states(context, args.max_test_samples)
    validate_saved_states(states, artifacts, args.max_test_samples)
    hidden_mask = validate_masks(states, artifacts)
    sample_count = hidden_mask.shape[0]
    inverse = artifacts["inverse_unique_mask"][:sample_count]
    unique_masks = artifacts["unique_hidden_mask"]
    active_count = hidden_mask.sum(axis=1).astype(np.int64)
    active_fraction = hidden_mask.mean(axis=1).astype(np.float64)

    output_slice = slice(hidden_dim, hidden_dim + output_dim)
    eye = np.eye(hidden_dim + output_dim, dtype=np.float64)
    sample_rows: list[dict | None] = [None] * sample_count

    mask_metric_rows: list[dict] = []
    mask_metric_arrays: dict[str, list[float]] = {
        key: []
        for key in [
            "current_gain_all_rms",
            "current_gain_hidden_rms",
            "current_gain_output_rms",
            "current_gain_all_spectral",
            "current_hidden_state_gain_all_rms",
            "input_noise_gain_rms",
            "input_noise_gain_spectral",
            "teaching_hidden_gain_rms",
            "teaching_hidden_gain_spectral",
            "teaching_output_gain_rms",
            "teaching_output_gain_spectral",
            "teaching_hidden_over_current_hidden_state",
            "teaching_output_over_current_output",
        ]
    }

    unique_seen = sorted(set(int(idx) for idx in inverse.tolist()))
    for unique_idx in unique_seen:
        mask = unique_masks[unique_idx]
        hessian = hessian_with_lpw_diag(
            hessian_linear,
            mask,
            lpw_layer_conductance=float(artifacts["lpw_layer_conductance"]),
            output_dim=output_dim,
        )
        if not np.allclose(hessian, hessian.T, rtol=1e-9, atol=1e-12):
            raise ValueError(f"H_lpw_sample is not symmetric for unique mask {unique_idx}.")
        h_inv = np.linalg.solve(hessian, eye)
        ops = compute_mask_operators(
            h_inv,
            weight0,
            current_amp=spec.current_amp,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
        )
        mask_metric_row = {
            "unique_mask_index": unique_idx,
            "active_hidden_count": int(mask.sum()),
            "active_hidden_fraction": float(mask.mean()),
        }
        for key in mask_metric_arrays:
            value = float(ops[key])
            mask_metric_row[key] = value
            mask_metric_arrays[key].append(value)
        mask_metric_rows.append(mask_metric_row)

        sample_indices = np.flatnonzero(inverse == unique_idx)
        for sample_index in sample_indices:
            logits = states["outputs"][sample_index]
            label = int(states["labels"][sample_index])
            clean_margin, margin_current_all = margin_metrics(logits, label, ops["cov_current_all"])
            _, margin_current_hidden = margin_metrics(logits, label, ops["cov_current_hidden"])
            _, margin_current_output = margin_metrics(logits, label, ops["cov_current_output"])
            _, margin_input = margin_metrics(logits, label, ops["cov_input"])

            write_abs = sample_write_metrics(
                h_inv,
                states["inputs"][sample_index],
                states["hidden"][sample_index],
                states["outputs"][sample_index],
                weight0,
                weight1,
                voltage_amp=spec.voltage_amp,
                current_amp=spec.current_amp,
                output_slice=output_slice,
                logits=logits,
                label=label,
                relative=False,
            )
            write_rel = sample_write_metrics(
                h_inv,
                states["inputs"][sample_index],
                states["hidden"][sample_index],
                states["outputs"][sample_index],
                weight0,
                weight1,
                voltage_amp=spec.voltage_amp,
                current_amp=spec.current_amp,
                output_slice=output_slice,
                logits=logits,
                label=label,
                relative=True,
            )
            row = {
                "run_name": spec.run_name,
                "seed": spec.seed,
                "voltage_amp": spec.voltage_amp,
                "current_amp": spec.current_amp,
                "sample_index": int(sample_index),
                "label": label,
                "prediction": int(states["predictions"][sample_index]),
                "correct": int(states["correct"][sample_index]),
                "lpw_active_hidden_count": int(active_count[sample_index]),
                "lpw_active_hidden_fraction": float(active_fraction[sample_index]),
                "clean_margin": clean_margin,
                "current_gain_all_rms": float(ops["current_gain_all_rms"]),
                "current_gain_hidden_rms": float(ops["current_gain_hidden_rms"]),
                "current_gain_output_rms": float(ops["current_gain_output_rms"]),
                "current_gain_all_spectral": float(ops["current_gain_all_spectral"]),
                "current_hidden_state_gain_all_rms": float(ops["current_hidden_state_gain_all_rms"]),
                "input_noise_gain_rms": float(ops["input_noise_gain_rms"]),
                "input_noise_gain_spectral": float(ops["input_noise_gain_spectral"]),
                "write_abs_gain_rms": write_abs["write_abs_gain_rms"],
                "write_abs_gain_spectral": write_abs["write_abs_gain_spectral"],
                "write_rel_gain_rms": write_rel["write_rel_gain_rms"],
                "write_rel_gain_spectral": write_rel["write_rel_gain_spectral"],
                "teaching_hidden_gain_rms": float(ops["teaching_hidden_gain_rms"]),
                "teaching_hidden_gain_spectral": float(ops["teaching_hidden_gain_spectral"]),
                "teaching_output_gain_rms": float(ops["teaching_output_gain_rms"]),
                "teaching_output_gain_spectral": float(ops["teaching_output_gain_spectral"]),
                "teaching_hidden_over_current_hidden_state": float(
                    ops["teaching_hidden_over_current_hidden_state"]
                ),
                "teaching_output_over_current_output": float(ops["teaching_output_over_current_output"]),
                "margin_norm_current_all": margin_current_all,
                "margin_norm_current_hidden": margin_current_hidden,
                "margin_norm_current_output": margin_current_output,
                "margin_norm_input_noise": margin_input,
                "margin_norm_write_abs": write_abs["margin_norm_write_abs"],
                "margin_norm_write_rel": write_rel["margin_norm_write_rel"],
            }
            sample_rows[int(sample_index)] = row

    if any(row is None for row in sample_rows):
        missing = [idx for idx, row in enumerate(sample_rows) if row is None][:10]
        raise RuntimeError(f"Internal error: missing sample rows, first missing indices={missing}.")
    typed_rows = [row for row in sample_rows if row is not None]

    sample_metrics_path = out_dir / "directional_sample_metrics.csv"
    summary_path = out_dir / "directional_summary.json"
    operator_metrics_path = out_dir / "operator_metrics_by_mask.npz"
    config_path = out_dir / "config.json"

    write_csv(sample_metrics_path, typed_rows, SAMPLE_COLUMNS)

    np.savez_compressed(
        operator_metrics_path,
        unique_hidden_active_mask=unique_masks[np.asarray(unique_seen, dtype=np.int64)],
        unique_mask_index=np.asarray(unique_seen, dtype=np.int64),
        **{
            key: np.asarray(values, dtype=np.float64)
            for key, values in mask_metric_arrays.items()
        },
        metadata_json=np.asarray(
            json.dumps(
                {
                    "run_name": spec.run_name,
                    "seed": spec.seed,
                    "voltage_amp": spec.voltage_amp,
                    "current_amp": spec.current_amp,
                    "lpw_layer_conductance": float(artifacts["lpw_layer_conductance"]),
                    "free_dims": free_dims,
                    "node_order": "hidden 0:100, output 100:110",
                    "source_hessian_root": str(args.input_root),
                },
                sort_keys=True,
            )
        ),
    )

    output_paths = {
        "sample_metrics": sample_metrics_path,
        "summary": summary_path,
        "operator_metrics": operator_metrics_path,
        "config": config_path,
    }
    source_paths = {
        "hessian_summary_path": spec.hessian_summary_path,
        "hessian_run_dir": spec.hessian_run_dir,
    }
    summary = run_summary_from_samples(
        spec,
        typed_rows,
        output_paths=output_paths,
        source_paths=source_paths,
    )
    summary["unique_lpw_masks_used"] = len(unique_seen)
    summary["lpw_layer_conductance"] = float(artifacts["lpw_layer_conductance"])
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    config = {
        "input_root": str(args.input_root),
        "output_root": str(output_root),
        "device": str(device),
        "batch_size": args.batch_size,
        "dataset_root_override": args.dataset_root,
        "max_test_samples": args.max_test_samples,
        "no_download": bool(args.no_download),
        "source_checkpoint": str(spec.checkpoint_path),
        "source_config": str(spec.source_config_path),
        "resolved_config": str(spec.resolved_config_path),
        "hessian_linear_path": str(spec.hessian_linear_path),
        "lpw_masks_path": str(spec.lpw_masks_path),
        "lpw_layer_conductance": float(artifacts["lpw_layer_conductance"]),
        "response_metric_convention": {
            "additive_current": "unit independent current perturbation in selected free nodes",
            "input_noise": "unit independent perturbation in signed/preprocessed input coordinates",
            "write_abs": "unit independent absolute conductance perturbation on dense weights",
            "write_rel": "unit independent multiplier times each dense conductance weight",
            "rms_gain": "sqrt(trace(output_covariance) / expected_probe_norm_squared)",
            "margin_norm": "minimum true-vs-competitor clean margin divided by induced margin std",
        },
    }
    config_path.write_text(json.dumps(hess_eval._json_sanitize(config), indent=2, sort_keys=True) + "\n")
    print(
        f"[run] {spec.run_name} seed={spec.seed} samples={sample_count} "
        f"unique_masks={len(unique_seen)} acc={summary['test_accuracy_recomputed']:.4f}"
    )
    return summary


def collect_run_summaries(output_root: Path, specs: list[DirectionalRunSpec]) -> list[dict]:
    rows: list[dict] = []
    for spec in specs:
        path = run_output_dir(output_root, spec) / "directional_summary.json"
        if path.exists():
            rows.append(json.loads(path.read_text()))
    return sorted(rows, key=summary_sort_key)


def sample_values(sample_rows: list[dict[str, str]], key: str) -> np.ndarray:
    return np.asarray([finite_float(row.get(key, math.nan)) for row in sample_rows], dtype=np.float64)


def aggregate_by_amp(summary_rows: list[dict]) -> list[dict]:
    out: list[dict] = []
    for run_name in RUN_ORDER:
        rows = [row for row in summary_rows if row["run_name"] == run_name]
        if not rows:
            continue
        sample_rows: list[dict[str, str]] = []
        for row in rows:
            path = Path(row["directional_sample_metrics_path"])
            if path.exists():
                sample_rows.extend(read_csv(path))
        acc = np.asarray([finite_float(row["test_accuracy_recomputed"]) for row in rows], dtype=np.float64)
        result = {
            "run_name": run_name,
            "voltage_amp": rows[0]["voltage_amp"],
            "current_amp": rows[0]["current_amp"],
            "num_seeds": len(rows),
            "sample_count": len(sample_rows),
            "test_accuracy_mean": float(np.nanmean(acc)) if acc.size else math.nan,
            "test_accuracy_std": float(np.nanstd(acc)) if acc.size else math.nan,
        }
        for key in SUMMARY_METRIC_KEYS:
            result.update(stats(sample_values(sample_rows, key), key))
        out.append(result)
    return out


def make_by_amp_plot(
    rows: list[dict],
    keys: list[str],
    ylabel: str,
    output_path: Path,
    *,
    log_y: bool = False,
) -> None:
    if not rows:
        return
    labels = [str(row["run_name"]).replace("mnist_bp_amp_", "") for row in rows]
    x = np.arange(len(rows), dtype=np.float64)
    width = min(0.8 / max(len(keys), 1), 0.35)
    fig, ax = plt.subplots(figsize=(8.8, 4.8), constrained_layout=True)
    for key_index, key in enumerate(keys):
        means = np.asarray([finite_float(row.get(f"{key}_mean", math.nan)) for row in rows], dtype=np.float64)
        stds = np.asarray([finite_float(row.get(f"{key}_std", math.nan)) for row in rows], dtype=np.float64)
        offset = (key_index - 0.5 * (len(keys) - 1)) * width
        ax.bar(x + offset, means, width=width, yerr=stds, capsize=3, label=key)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    if len(keys) > 1:
        ax.legend(fontsize=8)
    if log_y:
        ax.set_yscale("log")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_top_level_outputs(output_root: Path, summary_rows: list[dict]) -> None:
    summary_rows = sorted(summary_rows, key=summary_sort_key)
    write_csv(output_root / "summary.csv", summary_rows)
    by_amp = aggregate_by_amp(summary_rows)
    write_csv(output_root / "summary_by_amp.csv", by_amp)
    make_by_amp_plot(
        by_amp,
        ["current_gain_all_rms", "current_gain_hidden_rms", "current_gain_output_rms"],
        "Additive current RMS output gain",
        output_root / "additive_current_gain_by_amp.png",
        log_y=True,
    )
    make_by_amp_plot(
        by_amp,
        ["write_abs_gain_rms", "write_rel_gain_rms"],
        "Conductance write-noise RMS output gain",
        output_root / "conductance_write_gain_by_amp.png",
        log_y=True,
    )
    make_by_amp_plot(
        by_amp,
        ["input_noise_gain_rms"],
        "Input-noise RMS output gain",
        output_root / "input_noise_gain_by_amp.png",
        log_y=True,
    )
    make_by_amp_plot(
        by_amp,
        ["teaching_hidden_gain_rms", "teaching_output_gain_rms"],
        "EqProp teaching transfer RMS gain",
        output_root / "teaching_transfer_by_amp.png",
        log_y=True,
    )
    make_by_amp_plot(
        by_amp,
        ["margin_norm_current_all", "margin_norm_input_noise", "margin_norm_write_rel"],
        "Margin-normalized response",
        output_root / "margin_normalized_response_by_amp.png",
    )
    make_by_amp_plot(
        by_amp,
        ["teaching_hidden_over_current_hidden_state", "teaching_output_over_current_output"],
        "Teaching/current-noise selectivity proxy",
        output_root / "selectivity_by_amp.png",
        log_y=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate directional Hessian responses for legacy MNIST DRN-XS amplification runs."
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--run-name", action="append", choices=RUN_ORDER)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--rerun-complete", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument(
        "--no-summary-after-run",
        action="store_true",
        help="Only write per-run outputs; useful for parallel launchers that collect once at the end.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    args.input_root = input_root
    args.output_root = output_root

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested CUDA device {device}, but CUDA is not available.")

    specs = read_run_specs(
        input_root,
        run_names=set(args.run_name) if args.run_name else None,
        seeds=set(args.seeds) if args.seeds else None,
    )

    if args.summary_only:
        all_specs = read_run_specs(
            input_root,
            run_names=None,
            seeds=set(args.seeds) if args.seeds else None,
        )
        summary_rows = collect_run_summaries(output_root, all_specs)
        write_top_level_outputs(output_root, summary_rows)
        print(f"[summary] wrote {output_root / 'summary.csv'}")
        print(f"[summary] wrote {output_root / 'summary_by_amp.csv'}")
        return

    print(f"[directional] input_root={input_root}")
    print(f"[directional] output_root={output_root}")
    print(f"[directional] selected_runs={len(specs)} device={device}")
    for spec in specs:
        analyze_run(spec, args, output_root=output_root, device=device)

    if args.no_summary_after_run:
        print("[done] per-run outputs written; top-level summary skipped by request")
        return

    all_specs = read_run_specs(
        input_root,
        run_names=None,
        seeds=set(args.seeds) if args.seeds else None,
    )
    summary_rows = collect_run_summaries(output_root, all_specs)
    write_top_level_outputs(output_root, summary_rows)
    print(f"[done] summary={output_root / 'summary.csv'}")


if __name__ == "__main__":
    main()
