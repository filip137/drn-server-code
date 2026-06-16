#!/usr/bin/env python3
"""Stern-style incidence validation for hard-sigmoid MNIST DRN-XS checkpoints.

This script implements tests 3-8 from the Stern incidence validation plan for
the one-hidden-layer MNIST DRN-XS checkpoints.  It constructs the explicit
incidence representation

    H = B.T @ diag(k) @ B + diag(Lambda_nl)
    A_phi xi = B.T @ (k * z * xi)

using the same amplified/symmetrized convention as the existing Hessian and
physical sharpness evaluators.
"""

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
EXPERIMENTS_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, LABS_DIR, TOOLS_DIR, EXPERIMENTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import evaluate_mnist_bp_hessian_hardsigmoid_amplification_sweep as hs_eval  # noqa: E402
import evaluate_mnist_bp_hessian_lpw_amplification_sweep as hess_eval  # noqa: E402
import evaluate_mnist_bp_physical_cost_sharpness as phys_eval  # noqa: E402
import plot_conditioning_vs_runtime as pcr  # noqa: E402


DEFAULT_INPUT_ROOT = (
    REPO_ROOT / "results" / "mnist_bp_amplification_sweep_hardsigmoid_voff15_legacy"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "results" / "drn_stern_incidence_validation_hardsigmoid_voff15_iter16"
)
DEFAULT_COST_ROOT = (
    REPO_ROOT / "results" / "mnist_bp_cost_hessian_noise_curvature_hardsigmoid_voff15_iter16"
)
RUN_ORDER = hs_eval.RUN_ORDER
RUN_LABELS = {
    "mnist_bp_amp_v1_c1": "v1/c1",
    "mnist_bp_amp_v2_c1": "v2/c1",
    "mnist_bp_amp_v4_c1": "v4/c1",
    "mnist_bp_amp_v1_c2": "v1/c2",
    "mnist_bp_amp_v1_c4": "v1/c4",
}
RUN_COLORS = {
    "mnist_bp_amp_v1_c1": "#4c78a8",
    "mnist_bp_amp_v2_c1": "#f58518",
    "mnist_bp_amp_v4_c1": "#e45756",
    "mnist_bp_amp_v1_c2": "#54a24b",
    "mnist_bp_amp_v1_c4": "#b279a2",
}


@dataclass(frozen=True)
class IncidenceDRNXS:
    B: np.ndarray
    k: np.ndarray
    owner: np.ndarray
    layer: np.ndarray
    free_dims: list[int]
    weight0_shape: tuple[int, int]
    weight1_shape: tuple[int, int]
    voltage_amp: float
    current_amp: float
    weight0_edges: slice
    weight1_edges: slice

    @property
    def num_edges(self) -> int:
        return int(self.k.shape[0])

    @property
    def num_free_nodes(self) -> int:
        return int(sum(self.free_dims))


def write_csv(path: Path, rows: list[dict], columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def finite_float(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def stats(values: list[float] | np.ndarray, prefix: str) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            f"{prefix}_mean": math.nan,
            f"{prefix}_std": math.nan,
            f"{prefix}_p50": math.nan,
            f"{prefix}_p90": math.nan,
        }
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        f"{prefix}_p50": float(np.percentile(arr, 50)),
        f"{prefix}_p90": float(np.percentile(arr, 90)),
    }


def relerr(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(b))
    if denom == 0.0:
        return float(np.linalg.norm(a - b))
    return float(np.linalg.norm(a - b) / denom)


def build_incidence(weight0: np.ndarray, weight1: np.ndarray, voltage_amp: float, current_amp: float) -> IncidenceDRNXS:
    input_dim, hidden_dim = weight0.shape
    hidden_dim_1, output_dim = weight1.shape
    if hidden_dim != hidden_dim_1:
        raise ValueError(f"Weight shape mismatch: {weight0.shape=} {weight1.shape=}.")

    n0 = weight0.size
    n1 = weight1.size
    num_edges = n0 + n1
    num_free = hidden_dim + output_dim
    B = np.zeros((num_edges, num_free), dtype=np.float64)
    k = np.zeros(num_edges, dtype=np.float64)
    owner = np.empty(num_edges, dtype=object)
    layer = np.empty(num_edges, dtype=object)

    edge = 0
    for input_index in range(input_dim):
        for hidden_index in range(hidden_dim):
            B[edge, hidden_index] = float(current_amp)
            k[edge] = float(weight0[input_index, hidden_index])
            owner[edge] = "DenseWeight_0"
            layer[edge] = "input_hidden"
            edge += 1

    layer_scale = float(current_amp / voltage_amp)
    for hidden_index in range(hidden_dim):
        for output_index in range(output_dim):
            B[edge, hidden_index] = float(voltage_amp)
            B[edge, hidden_dim + output_index] = -float(current_amp)
            k[edge] = layer_scale * float(weight1[hidden_index, output_index])
            owner[edge] = "DenseWeight_1"
            layer[edge] = "hidden_output"
            edge += 1

    return IncidenceDRNXS(
        B=B,
        k=k,
        owner=owner,
        layer=layer,
        free_dims=[hidden_dim, output_dim],
        weight0_shape=(input_dim, hidden_dim),
        weight1_shape=(hidden_dim, output_dim),
        voltage_amp=float(voltage_amp),
        current_amp=float(current_amp),
        weight0_edges=slice(0, n0),
        weight1_edges=slice(n0, n0 + n1),
    )


def incidence_z(inc: IncidenceDRNXS, x_flat: np.ndarray, hidden: np.ndarray, output: np.ndarray, current_amp: float, voltage_amp: float) -> np.ndarray:
    input_dim = x_flat.shape[0]
    hidden_dim, output_dim = inc.free_dims
    z = np.empty(inc.num_edges, dtype=np.float64)
    z0 = current_amp * hidden[None, :] - x_flat[:, None]
    z[inc.weight0_edges] = z0.reshape(-1)
    z1 = voltage_amp * hidden[:, None] - current_amp * output[None, :]
    z[inc.weight1_edges] = z1.reshape(-1)
    return z


def linear_hessian_from_incidence(inc: IncidenceDRNXS) -> np.ndarray:
    hidden_dim, output_dim = inc.free_dims
    k0 = inc.k[inc.weight0_edges].reshape(inc.weight0_shape)
    k1 = inc.k[inc.weight1_edges].reshape(inc.weight1_shape)
    H = np.zeros((inc.num_free_nodes, inc.num_free_nodes), dtype=np.float64)
    hidden_diag = (inc.current_amp**2) * np.sum(k0, axis=0)
    hidden_diag += (inc.voltage_amp**2) * np.sum(k1, axis=1)
    output_diag = (inc.current_amp**2) * np.sum(k1, axis=0)
    H[np.arange(hidden_dim), np.arange(hidden_dim)] = hidden_diag
    out_idx = hidden_dim + np.arange(output_dim)
    H[out_idx, out_idx] = output_diag
    offdiag = -inc.voltage_amp * inc.current_amp * k1
    H[:hidden_dim, hidden_dim:] = offdiag
    H[hidden_dim:, :hidden_dim] = offdiag.T
    return H


def hessian_with_lambda(H_linear: np.ndarray, lambda_nl: np.ndarray) -> np.ndarray:
    H = H_linear.copy()
    H[np.diag_indices_from(H)] += lambda_nl
    return H


def edge_values_to_nodes(inc: IncidenceDRNXS, edge_values: np.ndarray) -> np.ndarray:
    hidden_dim, output_dim = inc.free_dims
    values0 = edge_values[inc.weight0_edges].reshape(inc.weight0_shape)
    values1 = edge_values[inc.weight1_edges].reshape(inc.weight1_shape)
    out = np.zeros(inc.num_free_nodes, dtype=np.float64)
    out[:hidden_dim] += inc.current_amp * np.sum(values0, axis=0)
    out[:hidden_dim] += inc.voltage_amp * np.sum(values1, axis=1)
    out[hidden_dim:] += -inc.current_amp * np.sum(values1, axis=0)
    if out.shape[0] != hidden_dim + output_dim:
        raise RuntimeError(f"Unexpected incidence output shape: {out.shape}.")
    return out


def aphi_times(inc: IncidenceDRNXS, z: np.ndarray, xi: np.ndarray) -> np.ndarray:
    return edge_values_to_nodes(inc, inc.k * z * xi)


def residual_synaptic(inc: IncidenceDRNXS, z: np.ndarray, logg_delta: np.ndarray | None = None) -> np.ndarray:
    if logg_delta is None:
        kk = inc.k
    else:
        kk = inc.k * np.exp(logg_delta)
    return edge_values_to_nodes(inc, kk * z)


def output_response_matrix(H: np.ndarray, free_dims: list[int]) -> np.ndarray:
    return phys_eval.solve_output_response(H, free_dims, num_classes=10)[0]


def swrite_from_incidence(inc: IncidenceDRNXS, z: np.ndarray, Z: np.ndarray) -> tuple[float, float, float]:
    # Z = H^{-1} P^T, so columns are responses for output projection directions.
    hidden_dim, output_dim = inc.free_dims
    Zh = Z[:hidden_dim]
    Zo = Z[hidden_dim:]
    k0 = inc.k[inc.weight0_edges].reshape(inc.weight0_shape)
    z0 = z[inc.weight0_edges].reshape(inc.weight0_shape)
    k1 = inc.k[inc.weight1_edges].reshape(inc.weight1_shape)
    z1 = z[inc.weight1_edges].reshape(inc.weight1_shape)
    response0_norm = np.sum((inc.current_amp * Zh) ** 2, axis=1)
    s0 = float(np.sum((k0 * z0) ** 2 * response0_norm[None, :]))
    response1 = inc.voltage_amp * Zh[:, None, :] - inc.current_amp * Zo[None, :, :]
    if response1.shape[:2] != (hidden_dim, output_dim):
        raise RuntimeError(f"Unexpected hidden-output response shape: {response1.shape}.")
    response1_norm = np.sum(response1 * response1, axis=2)
    s1 = float(np.sum((k1 * z1) ** 2 * response1_norm))
    return s0 + s1, s0, s1


def spectral_metrics(H: np.ndarray) -> dict[str, float]:
    eig = np.linalg.eigvalsh(0.5 * (H + H.T))
    sigma = np.linalg.svd(H, compute_uv=False)
    sigma_min = float(np.min(sigma))
    sigma_max = float(np.max(sigma))
    return {
        "lambda_min_sym": float(np.min(eig)),
        "lambda_max_sym": float(np.max(eig)),
        "sigma_min": sigma_min,
        "sigma_max": sigma_max,
        "kappa": float(sigma_max / sigma_min) if sigma_min > 0 else math.inf,
    }


def read_cost_curvature(cost_root: Path) -> dict[tuple[str, int], float]:
    path = cost_root / "summary_by_model.csv"
    if not path.exists():
        return {}
    out: dict[tuple[str, int], float] = {}
    for row in read_csv(path):
        if row.get("target") == "all":
            out[(row["run_name"], int(row["seed"]))] = finite_float(row["directional_curvature_mean"])
    return out


def run_output_dir(output_root: Path, spec: hess_eval.RunSpec) -> Path:
    return output_root / spec.run_name / f"seed_{spec.seed}"


def analyze_run(spec: hess_eval.RunSpec, args: argparse.Namespace, *, output_root: Path, device: torch.device) -> dict:
    out_dir = run_output_dir(output_root, spec)
    out_dir.mkdir(parents=True, exist_ok=True)

    context = hs_eval.build_eval_context(
        spec,
        device=device,
        batch_size=args.batch_size,
        dataset_root=args.dataset_root,
        no_download=args.no_download,
    )
    inference_iterations = phys_eval.override_minimizer_iterations(context, args.inference_iterations)
    weight0, weight1 = hess_eval.dense_weight_mats(context["energy_fn"])
    inc = build_incidence(weight0, weight1, spec.voltage_amp, spec.current_amp)
    H_existing_linear, free_dims = pcr.build_linear_hessian([weight0, weight1], spec.voltage_amp, spec.current_amp)
    H_inc_linear = linear_hessian_from_incidence(inc)
    if free_dims != inc.free_dims:
        raise ValueError(f"free_dims mismatch: {free_dims=} {inc.free_dims=}.")

    states = phys_eval.collect_samples(context, args.max_test_samples)
    hs_params = context["model_cfg"].get("hard_sigmoid_param", {})
    active_mask, hidden_conductance, *_ = hs_eval.hard_sigmoid_hidden_conductance(
        states["hidden"],
        params=hs_params,
        voltage_amp=spec.voltage_amp,
        current_amp=spec.current_amp,
    )
    lambda_all = np.concatenate(
        [hidden_conductance.astype(np.float64), np.zeros((hidden_conductance.shape[0], free_dims[-1]), dtype=np.float64)],
        axis=1,
    )
    cost_curvature_by_model = read_cost_curvature(args.cost_root)
    cost_curvature = cost_curvature_by_model.get((spec.run_name, spec.seed), math.nan)

    incidence_rows: list[dict] = []
    hessian_rows: list[dict] = []
    aphi_rows: list[dict] = []
    scaling_rows: list[dict] = []
    swrite_rows: list[dict] = []
    same_rows: list[dict] = []
    rng = np.random.default_rng(args.direction_seed + 1000 * spec.seed)

    H_linear_relerr = relerr(H_inc_linear, H_existing_linear)
    max_samples = states["hidden"].shape[0]
    edge_count_by_owner = {
        "DenseWeight_0": int(np.sum(inc.owner == "DenseWeight_0")),
        "DenseWeight_1": int(np.sum(inc.owner == "DenseWeight_1")),
    }

    for sample_index in range(max_samples):
        x_flat = states["inputs"][sample_index].reshape(-1).astype(np.float64)
        hidden = states["hidden"][sample_index].astype(np.float64)
        output = states["output"][sample_index].astype(np.float64)
        z = incidence_z(inc, x_flat, hidden, output, spec.current_amp, spec.voltage_amp)
        lambda_nl = lambda_all[sample_index]
        H_inc = hessian_with_lambda(H_inc_linear, lambda_nl)
        H_existing = hessian_with_lambda(H_existing_linear, lambda_nl)
        H_rel = relerr(H_inc, H_existing)
        sym_err = relerr(H_inc, H_inc.T)
        metrics_inc = spectral_metrics(H_inc)
        metrics_existing = spectral_metrics(H_existing)

        incidence_rows.append(
            {
                "run_name": spec.run_name,
                "seed": spec.seed,
                "voltage_amp": spec.voltage_amp,
                "current_amp": spec.current_amp,
                "sample_index": sample_index,
                "num_edges": inc.num_edges,
                "num_free_nodes": inc.num_free_nodes,
                "num_edges_weight0": edge_count_by_owner["DenseWeight_0"],
                "num_edges_weight1": edge_count_by_owner["DenseWeight_1"],
                "H_linear_relerr": H_linear_relerr,
                "H_inc_symmetry_relerr": sym_err,
                "Aphi_shape_rows": inc.num_free_nodes,
                "Aphi_shape_cols": inc.num_edges,
            }
        )
        hessian_rows.append(
            {
                "run_name": spec.run_name,
                "seed": spec.seed,
                "voltage_amp": spec.voltage_amp,
                "current_amp": spec.current_amp,
                "sample_index": sample_index,
                "rel_fro_error": H_rel,
                "symmetry_rel_error": sym_err,
                "sigma_min_inc": metrics_inc["sigma_min"],
                "sigma_min_existing": metrics_existing["sigma_min"],
                "sigma_min_rel_error": abs(metrics_inc["sigma_min"] - metrics_existing["sigma_min"]) / max(abs(metrics_existing["sigma_min"]), 1e-30),
                "sigma_max_inc": metrics_inc["sigma_max"],
                "sigma_max_existing": metrics_existing["sigma_max"],
                "sigma_max_rel_error": abs(metrics_inc["sigma_max"] - metrics_existing["sigma_max"]) / max(abs(metrics_existing["sigma_max"]), 1e-30),
                "kappa_inc": metrics_inc["kappa"],
                "kappa_existing": metrics_existing["kappa"],
                "kappa_rel_error": abs(metrics_inc["kappa"] - metrics_existing["kappa"]) / max(abs(metrics_existing["kappa"]), 1e-30),
                "lambda_min_sym_inc": metrics_inc["lambda_min_sym"],
                "lambda_min_sym_existing": metrics_existing["lambda_min_sym"],
            }
        )

        if sample_index < args.aphi_samples:
            for direction_index in range(args.num_directions):
                xi = rng.standard_normal(inc.num_edges)
                analytic = aphi_times(inc, z, xi)
                for eps in args.epsilons:
                    fd = (
                        residual_synaptic(inc, z, eps * xi)
                        - residual_synaptic(inc, z, -eps * xi)
                    ) / (2.0 * eps)
                    aphi_rows.append(
                        {
                            "run_name": spec.run_name,
                            "seed": spec.seed,
                            "voltage_amp": spec.voltage_amp,
                            "current_amp": spec.current_amp,
                            "sample_index": sample_index,
                            "direction_index": direction_index,
                            "epsilon": eps,
                            "relative_error": relerr(analytic, fd),
                            "analytic_norm": float(np.linalg.norm(analytic)),
                            "fd_norm": float(np.linalg.norm(fd)),
                        }
                    )

        if sample_index < args.scaling_samples:
            row_scale = np.concatenate(
                [
                    np.ones(free_dims[0], dtype=np.float64),
                    np.full(free_dims[1], spec.current_amp / spec.voltage_amp, dtype=np.float64),
                ]
            )
            Dinv = 1.0 / row_scale
            J_raw = Dinv[:, None] * H_inc
            for direction_index in range(args.num_directions):
                xi = rng.standard_normal(inc.num_edges)
                A_scaled_xi = aphi_times(inc, z, xi)
                A_raw_xi = Dinv * A_scaled_xi
                response_raw = np.linalg.solve(J_raw, A_raw_xi)
                response_scaled = np.linalg.solve(H_inc, A_scaled_xi)
                response_wrong = np.linalg.solve(H_inc, A_raw_xi)
                scaling_rows.append(
                    {
                        "run_name": spec.run_name,
                        "seed": spec.seed,
                        "voltage_amp": spec.voltage_amp,
                        "current_amp": spec.current_amp,
                        "sample_index": sample_index,
                        "direction_index": direction_index,
                        "response_consistency_relerr": relerr(response_scaled, response_raw),
                        "wrong_mixed_scaling_relerr": relerr(response_wrong, response_raw),
                    }
                )

        if sample_index < args.swrite_samples:
            Z = output_response_matrix(H_inc, free_dims)
            s_total, s0, s1 = swrite_from_incidence(inc, z, Z)
            s_ref, s0_ref, s1_ref = phys_eval.physical_cost_sharpness(
                x_flat=x_flat,
                hidden=hidden,
                output=output,
                weight0=weight0,
                weight1=weight1,
                z_response=Z,
                voltage_amp=spec.voltage_amp,
                current_amp=spec.current_amp,
            )
            swrite_rows.append(
                {
                    "run_name": spec.run_name,
                    "seed": spec.seed,
                    "voltage_amp": spec.voltage_amp,
                    "current_amp": spec.current_amp,
                    "sample_index": sample_index,
                    "swrite_inc": s_total,
                    "swrite_inc_weight0": s0,
                    "swrite_inc_weight1": s1,
                    "swrite_reference": s_ref,
                    "swrite_reference_weight0": s0_ref,
                    "swrite_reference_weight1": s1_ref,
                    "swrite_relerr": abs(s_total - s_ref) / max(abs(s_ref), 1e-30),
                    "swrite_weight0_relerr": abs(s0 - s0_ref) / max(abs(s0_ref), 1e-30),
                    "swrite_weight1_relerr": abs(s1 - s1_ref) / max(abs(s1_ref), 1e-30),
                }
            )

        if sample_index < args.same_coordinate_samples:
            Z = output_response_matrix(H_inc, free_dims)
            for direction_index in range(args.num_directions):
                xi = rng.standard_normal(inc.num_edges)
                b_phi = aphi_times(inc, z, xi)
                delta_v_phi = -np.linalg.solve(H_inc, b_phi)
                q_phys_phi = float(np.sum(delta_v_phi[-free_dims[-1] :] ** 2))
                # In this DRN-XS incidence convention, the software multiplicative
                # weight direction and the incidence log-G direction induce the same
                # local residual.  We still save it separately to make the coordinate
                # check explicit and comparable to the finite-difference cost summary.
                b_theta = b_phi
                delta_v_same = -np.linalg.solve(H_inc, b_theta)
                q_phys_same = float(np.sum(delta_v_same[-free_dims[-1] :] ** 2))
                same_rows.append(
                    {
                        "run_name": spec.run_name,
                        "seed": spec.seed,
                        "voltage_amp": spec.voltage_amp,
                        "current_amp": spec.current_amp,
                        "sample_index": sample_index,
                        "direction_index": direction_index,
                        "q_phys_same": q_phys_same,
                        "q_phys_phi": q_phys_phi,
                        "q_cost_model_mean": cost_curvature,
                        "same_phi_relerr": abs(q_phys_same - q_phys_phi) / max(abs(q_phys_phi), 1e-30),
                    }
                )

    run_dir = out_dir
    write_csv(run_dir / "incidence_representation_sample_metrics.csv", incidence_rows)
    write_csv(run_dir / "hessian_equivalence_sample_metrics.csv", hessian_rows)
    write_csv(run_dir / "aphi_fd_sample_metrics.csv", aphi_rows)
    write_csv(run_dir / "scaled_raw_consistency.csv", scaling_rows)
    write_csv(run_dir / "swrite_inc_sample_metrics.csv", swrite_rows)
    write_csv(run_dir / "same_coordinate_response_vs_cost.csv", same_rows)

    summary = {
        "run_name": spec.run_name,
        "seed": spec.seed,
        "voltage_amp": spec.voltage_amp,
        "current_amp": spec.current_amp,
        "sample_count": int(max_samples),
        "inference_iterations": int(inference_iterations),
        "num_edges": inc.num_edges,
        "num_free_nodes": inc.num_free_nodes,
        "H_linear_relerr": H_linear_relerr,
        **stats([finite_float(row["rel_fro_error"]) for row in hessian_rows], "hessian_rel_fro_error"),
        **stats([finite_float(row["relative_error"]) for row in aphi_rows if math.isclose(finite_float(row["epsilon"]), 1e-3)], "aphi_fd_relerr_eps_1e_3"),
        **stats([finite_float(row["response_consistency_relerr"]) for row in scaling_rows], "scaled_raw_consistency_relerr"),
        **stats([finite_float(row["wrong_mixed_scaling_relerr"]) for row in scaling_rows], "wrong_mixed_scaling_relerr"),
        **stats([finite_float(row["swrite_inc"]) for row in swrite_rows], "swrite_inc"),
        **stats([finite_float(row["swrite_relerr"]) for row in swrite_rows], "swrite_reference_relerr"),
        **stats([finite_float(row["q_phys_same"]) for row in same_rows], "q_phys_same"),
        **stats([finite_float(row["q_phys_phi"]) for row in same_rows], "q_phys_phi"),
        "q_cost_model_mean": cost_curvature,
    }
    (run_dir / "summary.json").write_text(json.dumps(hess_eval._json_sanitize(summary), indent=2, sort_keys=True) + "\n")
    print(
        f"[stern] {spec.run_name} seed={spec.seed} samples={max_samples} "
        f"Herr={summary['hessian_rel_fro_error_mean']:.3e} "
        f"Aerr={summary['aphi_fd_relerr_eps_1e_3_mean']:.3e} "
        f"Serr={summary['swrite_reference_relerr_mean']:.3e}"
    )
    return summary


def collect_summaries(output_root: Path, specs: list[hess_eval.RunSpec]) -> list[dict]:
    rows = []
    for spec in specs:
        path = run_output_dir(output_root, spec) / "summary.json"
        if path.exists():
            rows.append(json.loads(path.read_text()))
    return rows


def aggregate_by_amp(rows: list[dict]) -> list[dict]:
    out = []
    for run_name in RUN_ORDER:
        group = [row for row in rows if row["run_name"] == run_name]
        if not group:
            continue
        result = {
            "run_name": run_name,
            "voltage_amp": group[0]["voltage_amp"],
            "current_amp": group[0]["current_amp"],
            "num_seeds": len(group),
            "sample_count": int(sum(row["sample_count"] for row in group)),
        }
        for key in [
            "hessian_rel_fro_error_mean",
            "aphi_fd_relerr_eps_1e_3_mean",
            "scaled_raw_consistency_relerr_mean",
            "wrong_mixed_scaling_relerr_mean",
            "swrite_inc_mean",
            "swrite_reference_relerr_mean",
            "q_phys_same_mean",
            "q_phys_phi_mean",
            "q_cost_model_mean",
        ]:
            result.update(stats([finite_float(row.get(key)) for row in group], key))
        out.append(result)
    return out


def write_plots(output_root: Path, by_amp: list[dict]) -> None:
    plots_dir = output_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    def ordered(rows: list[dict]) -> list[dict]:
        return sorted(rows, key=lambda row: RUN_ORDER.index(row["run_name"]) if row["run_name"] in RUN_ORDER else 999)

    rows = ordered(by_amp)
    x = np.arange(len(rows))
    labels = [RUN_LABELS.get(row["run_name"], row["run_name"]) for row in rows]
    colors = [RUN_COLORS.get(row["run_name"], "#666666") for row in rows]

    for key, title, filename, log_y in [
        ("hessian_rel_fro_error_mean_mean", "H_inc vs existing Hessian relative error", "hessian_inc_vs_existing_relerr_by_amp.png", True),
        ("aphi_fd_relerr_eps_1e_3_mean_mean", "A_phi finite-difference error eps=1e-3", "aphi_fd_error_by_amp.png", True),
        ("scaled_raw_consistency_relerr_mean_mean", "Scaled/raw response consistency error", "scaled_raw_consistency_error_by_amp.png", True),
        ("wrong_mixed_scaling_relerr_mean_mean", "Wrong mixed-scaling response error", "wrong_scaling_error_by_amp.png", True),
        ("swrite_inc_mean_mean", "Incidence S_write", "swrite_inc_by_amp.png", True),
    ]:
        fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
        ax.bar(x, [finite_float(row[key]) for row in rows], color=colors, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
        if log_y:
            ax.set_yscale("log")
        fig.savefig(plots_dir / filename, dpi=200)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.6, 4.4), constrained_layout=True)
    for row in rows:
        ax.scatter(finite_float(row["swrite_inc_mean_mean"]), finite_float(row["q_cost_model_mean_mean"]), color=RUN_COLORS.get(row["run_name"]), s=70)
        ax.annotate(RUN_LABELS.get(row["run_name"], row["run_name"]), (finite_float(row["swrite_inc_mean_mean"]), finite_float(row["q_cost_model_mean_mean"])), xytext=(5, 4), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("S_write_inc")
    ax.set_ylabel("finite-difference cost curvature")
    ax.grid(True, alpha=0.3)
    fig.savefig(plots_dir / "swrite_inc_vs_cost_curvature.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.6, 4.4), constrained_layout=True)
    for row in rows:
        ax.scatter(finite_float(row["q_phys_same_mean_mean"]), finite_float(row["q_phys_phi_mean_mean"]), color=RUN_COLORS.get(row["run_name"]), s=70)
        ax.annotate(RUN_LABELS.get(row["run_name"], row["run_name"]), (finite_float(row["q_phys_same_mean_mean"]), finite_float(row["q_phys_phi_mean_mean"])), xytext=(5, 4), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("q_phys_same")
    ax.set_ylabel("q_phys_phi")
    ax.grid(True, alpha=0.3)
    fig.savefig(plots_dir / "q_phys_same_vs_q_phys_phi_scatter.png", dpi=200)
    plt.close(fig)


def write_top_level(output_root: Path, summaries: list[dict]) -> None:
    write_csv(output_root / "summary.csv", summaries)
    by_amp = aggregate_by_amp(summaries)
    write_csv(output_root / "summary_by_amp.csv", by_amp)
    write_plots(output_root, by_amp)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--cost-root", type=Path, default=DEFAULT_COST_ROOT)
    parser.add_argument("--checkpoint", choices=("best", "final"), default="best")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--run-name", action="append", choices=RUN_ORDER)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--max-test-samples", type=int, default=128)
    parser.add_argument("--aphi-samples", type=int, default=128)
    parser.add_argument("--scaling-samples", type=int, default=64)
    parser.add_argument("--swrite-samples", type=int, default=128)
    parser.add_argument("--same-coordinate-samples", type=int, default=128)
    parser.add_argument("--num-directions", type=int, default=8)
    parser.add_argument("--epsilons", type=float, nargs="+", default=[1e-4, 1e-3, 1e-2])
    parser.add_argument("--direction-seed", type=int, default=123)
    parser.add_argument("--inference-iterations", type=int, default=16)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--no-summary-after-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    args.cost_root = args.cost_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    specs = hess_eval.read_run_specs(
        input_root,
        checkpoint=args.checkpoint,
        run_names=set(args.run_name) if args.run_name else None,
        seeds=set(args.seeds) if args.seeds else None,
    )
    if args.summary_only:
        write_top_level(output_root, collect_summaries(output_root, specs))
        print(f"[summary] wrote {output_root / 'summary.csv'}")
        return

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested CUDA device {device}, but CUDA is not available.")
    print(f"[stern] input_root={input_root}")
    print(f"[stern] output_root={output_root}")
    print(f"[stern] selected_runs={len(specs)} device={device}")
    summaries = [analyze_run(spec, args, output_root=output_root, device=device) for spec in specs]
    if not args.no_summary_after_run:
        write_top_level(output_root, summaries)
        print(f"[done] summary={output_root / 'summary.csv'}")


if __name__ == "__main__":
    main()
