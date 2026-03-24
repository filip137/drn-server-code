#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import plot_conditioning_vs_runtime as pcr


FAMILY_STYLE = {
    "single_diode_exponential": ("Single diode", "#1f77b4"),
    "double_diode_exponential": ("Double diode", "#d62728"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute per-sample odd-even spectral-radius proxies and compare them to per-sample outer iteration counts."
    )
    parser.add_argument(
        "--metadata",
        action="append",
        required=True,
        type=Path,
        help="Path to a validation_metadata.json file. Can be provided multiple times.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where the per-sample CSV/plots will be written.",
    )
    parser.add_argument(
        "--max-power-iters",
        type=int,
        default=40,
        help="Maximum number of power-iteration steps per sample.",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-8,
        help="Relative convergence tolerance for the power iteration.",
    )
    return parser.parse_args()


def infer_nonlinearity(meta_path: Path, meta: dict) -> str:
    haystacks = [str(meta_path), str(meta.get("weights_path", "")), str(meta.get("config_path", ""))]
    for text in haystacks:
        if "single_diode_exponential" in text:
            return "single_diode_exponential"
        if "double_diode_exponential" in text:
            return "double_diode_exponential"
    raise ValueError(f"Could not infer nonlinearity from metadata: {meta_path}")


def samplewise_nonlinear_diag(nonlinearity: str, states_path: Path, cfg: dict, free_dims: list[int]) -> np.ndarray:
    states = np.load(states_path)
    params = cfg["exponential_diode_param"]
    i_s = float(params["I_s"])
    v_t = float(params["V_t"])
    v_off = float(params["V_off"])
    voltage_amp = float(cfg["voltage_amp"])
    current_amp = float(cfg["current_amp"])

    pieces: list[np.ndarray] = []
    hidden_count = len(free_dims) - 1
    sample_count = None

    for layer_idx in range(1, hidden_count + 1):
        layer = states[f"Layer_{layer_idx}"].astype(np.float64)
        if sample_count is None:
            sample_count = layer.shape[0]
        scaled_i_s = i_s * (current_amp / voltage_amp) ** (layer_idx - 1)

        if nonlinearity == "single_diode_exponential":
            half = layer.shape[1] // 2
            fwd = layer[:, :half]
            rev = layer[:, half:]
            deriv_fwd = np.zeros_like(fwd)
            deriv_rev = np.zeros_like(rev)
            mask_fwd = fwd > v_off
            mask_rev = rev < -v_off
            deriv_fwd[mask_fwd] = (scaled_i_s / v_t) * np.exp((fwd[mask_fwd] - v_off) / v_t)
            deriv_rev[mask_rev] = (scaled_i_s / v_t) * np.exp(((-rev[mask_rev]) - v_off) / v_t)
            deriv = np.concatenate([deriv_fwd, deriv_rev], axis=1)
        elif nonlinearity == "double_diode_exponential":
            deriv = np.zeros_like(layer)
            excess = np.abs(layer) - v_off
            mask = excess > 0.0
            deriv[mask] = (scaled_i_s / v_t) * np.exp(excess[mask] / v_t)
        else:
            raise ValueError(f"Unsupported nonlinearity: {nonlinearity}")

        pieces.append(deriv)

    if sample_count is None:
        raise ValueError(f"No hidden-layer states found in {states_path}")
    pieces.append(np.zeros((sample_count, free_dims[-1]), dtype=np.float64))
    return np.concatenate(pieces, axis=1)


def chol_solve(chol_factor: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    y = np.linalg.solve(chol_factor, rhs)
    return np.linalg.solve(chol_factor.T, y)


def apply_symmetric_sweep_operator(vec: np.ndarray, h_oo: np.ndarray, h_oe: np.ndarray, h_eo: np.ndarray, chol_ee: np.ndarray) -> np.ndarray:
    tmp_even = np.linalg.solve(chol_ee.T, vec)
    tmp_odd_rhs = h_oe @ tmp_even
    tmp_odd = np.linalg.solve(h_oo, tmp_odd_rhs)
    tmp_even_rhs = h_eo @ tmp_odd
    return np.linalg.solve(chol_ee, tmp_even_rhs)


def rho_power_iteration(hessian: np.ndarray, free_dims: list[int], max_iters: int, tol: float) -> float:
    odd_idx, even_idx = pcr.odd_even_partition_indices(free_dims)
    reordered_idx = np.concatenate([odd_idx, even_idx])
    reordered = hessian[np.ix_(reordered_idx, reordered_idx)]
    n_odd = len(odd_idx)
    h_oo = reordered[:n_odd, :n_odd]
    h_oe = reordered[:n_odd, n_odd:]
    h_eo = reordered[n_odd:, :n_odd]
    h_ee = reordered[n_odd:, n_odd:]

    chol_ee = np.linalg.cholesky(h_ee)
    n_even = h_ee.shape[0]
    vec = np.ones(n_even, dtype=np.float64) / math.sqrt(n_even)
    lam_prev = math.nan

    for _ in range(max_iters):
        out = apply_symmetric_sweep_operator(vec, h_oo, h_oe, h_eo, chol_ee)
        norm_out = float(np.linalg.norm(out))
        if norm_out == 0.0:
            return 0.0
        vec = out / norm_out
        image = apply_symmetric_sweep_operator(vec, h_oo, h_oe, h_eo, chol_ee)
        lam = float(vec @ image)
        if math.isfinite(lam_prev) and abs(lam - lam_prev) <= tol * max(1.0, abs(lam)):
            return lam
        lam_prev = lam
    return lam_prev if math.isfinite(lam_prev) else 0.0


def load_iteration_counts(states_path: Path) -> np.ndarray:
    counts_path = states_path.with_name("validation_iteration_counts.npz")
    data = np.load(counts_path)
    return data["iteration_counts"].astype(np.int64)


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    return pcr.pearson_corr(x.astype(np.float64), y.astype(np.float64))


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    return pcr.spearman_corr(x.astype(np.float64), y.astype(np.float64))


def analyze_run(meta_path: Path, output_dir: Path, max_power_iters: int, tol: float) -> dict:
    meta = pcr.load_metadata(meta_path)
    cfg_path = pcr.resolve_config_path(meta)
    cfg = pcr.load_config(cfg_path)
    weights_path = Path(meta["weights_path"])
    states_path = Path(meta["validation_states"])
    nonlinearity = infer_nonlinearity(meta_path, meta)
    mats = pcr.load_weight_mats(weights_path)
    hessian_lin, free_dims = pcr.build_linear_hessian(mats, float(cfg["voltage_amp"]), float(cfg["current_amp"]))
    sample_diags = samplewise_nonlinear_diag(nonlinearity, states_path, cfg, free_dims)
    iter_counts = load_iteration_counts(states_path)

    diag_idx = np.diag_indices_from(hessian_lin)
    rho_samples = np.empty(sample_diags.shape[0], dtype=np.float64)
    for sample_idx, diag_vec in enumerate(sample_diags):
        hessian = hessian_lin.copy()
        hessian[diag_idx] += diag_vec
        rho_samples[sample_idx] = rho_power_iteration(hessian, free_dims, max_iters=max_power_iters, tol=tol)

    mean_diag, _layer_stats = pcr.mean_nonlinear_diag(nonlinearity, states_path, cfg, free_dims)
    hessian_mean = hessian_lin.copy()
    hessian_mean[diag_idx] += mean_diag
    rho_mean_diag = pcr.odd_even_spectral_radius(hessian_mean, free_dims)

    label = f"{nonlinearity}_h{len(free_dims)-1}_w{free_dims[0]}"
    family_label, color = FAMILY_STYLE[nonlinearity]
    run_dir = output_dir / label
    run_dir.mkdir(parents=True, exist_ok=True)

    csv_path = run_dir / "per_sample_rho.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_index", "iteration_count", "rho_sample"])
        writer.writeheader()
        for sample_index, (count, rho_val) in enumerate(zip(iter_counts, rho_samples)):
            writer.writerow(
                {
                    "sample_index": sample_index,
                    "iteration_count": int(count),
                    "rho_sample": float(rho_val),
                }
            )

    scatter_path = run_dir / "per_sample_rho_vs_iterations.png"
    fig, ax = plt.subplots(figsize=(6.2, 4.8), constrained_layout=True)
    ax.scatter(rho_samples, iter_counts, color=color, s=14, alpha=0.7)
    pearson = pearson_corr(rho_samples, iter_counts)
    spearman = spearman_corr(rho_samples, iter_counts)
    p_txt = "nan" if math.isnan(pearson) else f"{pearson:.3f}"
    s_txt = "nan" if math.isnan(spearman) else f"{spearman:.3f}"
    ax.set_title(f"{family_label}, H={len(free_dims)-1}, width={free_dims[0]}\nPearson={p_txt}, Spearman={s_txt}")
    ax.set_xlabel(r"Per-sample $\rho(T_s)$")
    ax.set_ylabel("Outer sweeps")
    ax.grid(True, alpha=0.3)
    fig.savefig(scatter_path, dpi=200)
    plt.close(fig)

    summary = {
        "nonlinearity": nonlinearity,
        "hidden_layers": len(free_dims) - 1,
        "width": free_dims[0],
        "sample_count": int(rho_samples.size),
        "pearson": pearson,
        "spearman": spearman,
        "rho_mean_diag": float(rho_mean_diag),
        "rho_sample_mean": float(np.mean(rho_samples)),
        "rho_sample_std": float(np.std(rho_samples)),
        "rho_sample_min": float(np.min(rho_samples)),
        "rho_sample_max": float(np.max(rho_samples)),
        "iter_mean": float(np.mean(iter_counts)),
        "iter_std": float(np.std(iter_counts)),
        "metadata_path": str(meta_path),
        "config_path": str(cfg_path),
        "weights_path": str(weights_path),
        "states_path": str(states_path),
        "csv_path": str(csv_path),
        "scatter_path": str(scatter_path),
    }

    summary_path = run_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    summary["summary_path"] = str(summary_path)
    return summary


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for meta_path in args.metadata:
        summaries.append(analyze_run(meta_path, args.output_dir, args.max_power_iters, args.tol))

    summary_csv = args.output_dir / "summary.csv"
    fieldnames = [
        "nonlinearity",
        "hidden_layers",
        "width",
        "sample_count",
        "pearson",
        "spearman",
        "rho_mean_diag",
        "rho_sample_mean",
        "rho_sample_std",
        "rho_sample_min",
        "rho_sample_max",
        "iter_mean",
        "iter_std",
        "metadata_path",
        "config_path",
        "weights_path",
        "states_path",
        "csv_path",
        "scatter_path",
        "summary_path",
    ]
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)

    print(f"Wrote {summary_csv}")
    for summary in summaries:
        print(f"Wrote {summary['csv_path']}")
        print(f"Wrote {summary['scatter_path']}")
        print(f"Wrote {summary['summary_path']}")


if __name__ == "__main__":
    main()
