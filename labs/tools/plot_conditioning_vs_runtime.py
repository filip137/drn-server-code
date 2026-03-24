#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


CASE_RUNS_BY_DEPTH = {
    1: {
        "single_diode_exponential": {
            64: Path(
                "/home/filip/server_code/labs/figures_for_paper_digits/timings/"
                "single_diode_exponential/hidden_1/hidden_64/20260317-170659_single_diode_exponential/"
                "codex_cpu_rerun/20260319-151912_single_diode_exponential/validation_metadata.json"
            ),
            128: Path(
                "/home/filip/server_code/labs/figures_for_paper_digits/timings/"
                "single_diode_exponential/hidden_1/hidden_128/20260316-172522_single_diode_exponential/"
                "codex_cpu_rerun/20260319-151919_single_diode_exponential/validation_metadata.json"
            ),
            256: Path(
                "/home/filip/server_code/labs/figures_for_paper_digits/timings/"
                "single_diode_exponential/hidden_1/hidden_256/20260311-170349_single_diode_exponential/"
                "codex_cpu_rerun/20260319-151927_single_diode_exponential/validation_metadata.json"
            ),
            512: Path(
                "/home/filip/server_code/labs/figures_for_paper_digits/timings/"
                "single_diode_exponential/hidden_1/hidden_512/20260311-170405_single_diode_exponential/"
                "codex_cpu_rerun/20260319-151934_single_diode_exponential/validation_metadata.json"
            ),
            1024: Path(
                "/home/filip/server_code/labs/figures_for_paper_digits/timings/"
                "single_diode_exponential/hidden_1/hidden_1024/20260311-170419_single_diode_exponential/"
                "codex_cpu_rerun/20260319-151941_single_diode_exponential/validation_metadata.json"
            ),
        },
        "double_diode_exponential": {
            64: Path(
                "/home/filip/server_code/labs/figures_for_paper_digits/timings/"
                "double_diode_exponential/hidden_1/hidden_64/20260311-170020_double_diode_exponential/"
                "codex_cpu_rerun_float_32/20260319-150015_double_diode_exponential/validation_metadata.json"
            ),
            128: Path(
                "/home/filip/server_code/labs/figures_for_paper_digits/timings/"
                "double_diode_exponential/hidden_1/hidden_128/20260311-162256_double_diode_exponential/"
                "codex_cpu_rerun_float_32/20260319-150024_double_diode_exponential/validation_metadata.json"
            ),
            256: Path(
                "/home/filip/server_code/labs/figures_for_paper_digits/timings/"
                "double_diode_exponential/hidden_1/hidden_256/20260311-170041_double_diode_exponential/"
                "codex_cpu_rerun_float_32/20260319-150031_double_diode_exponential/validation_metadata.json"
            ),
            512: Path(
                "/home/filip/server_code/labs/figures_for_paper_digits/timings/"
                "double_diode_exponential/hidden_1/hidden_512/20260311-170102_double_diode_exponential/"
                "codex_cpu_rerun_float_32/20260319-150040_double_diode_exponential/validation_metadata.json"
            ),
            1024: Path(
                "/home/filip/server_code/labs/figures_for_paper_digits/timings/"
                "double_diode_exponential/hidden_1/hidden_1024/20260311-170123_double_diode_exponential/"
                "codex_cpu_rerun_float_32/20260319-150049_double_diode_exponential/validation_metadata.json"
            ),
        },
        "experimental": {
            64: Path(
                "/home/filip/server_code/labs/figures_for_paper_digits/timings/"
                "experimental/hidden_1/hidden_64/20260317-113144_experimental/"
                "codex_cpu_rerun/20260320-132204_experimental/validation_metadata.json"
            ),
            128: Path(
                "/home/filip/server_code/labs/figures_for_paper_digits/timings/"
                "experimental/hidden_1/hidden_128/20260317-113144_experimental/"
                "codex_cpu_rerun/20260320-132213_experimental/validation_metadata.json"
            ),
            256: Path(
                "/home/filip/server_code/labs/figures_for_paper_digits/timings/"
                "experimental/hidden_1/hidden_256/20260317-113706_experimental/"
                "codex_cpu_rerun/20260320-132218_experimental/validation_metadata.json"
            ),
            512: Path(
                "/home/filip/server_code/labs/figures_for_paper_digits/timings/"
                "experimental/hidden_1/hidden_512/20260317-113706_experimental/"
                "codex_cpu_rerun/20260320-132225_experimental/validation_metadata.json"
            ),
            1024: Path(
                "/home/filip/server_code/labs/figures_for_paper_digits/timings/"
                "experimental/hidden_1/hidden_1024/20260317-113706_experimental/"
                "codex_cpu_rerun/20260320-132232_experimental/validation_metadata.json"
            ),
        },
    },
    2: {
        "single_diode_exponential": {
            64: Path(
                "/home/filip/server_code/labs/figures_for_paper_digits/timings/"
                "single_diode_exponential/hidden_2/hidden_64/20260317-103528_single_diode_exponential/"
                "codex_cpu_rerun/20260319-114011_single_diode_exponential/validation_metadata.json"
            ),
            128: Path(
                "/home/filip/server_code/labs/figures_for_paper_digits/timings/"
                "single_diode_exponential/hidden_2/hidden_128/20260314-160910_single_diode_exponential/"
                "codex_cpu_rerun/20260319-114106_single_diode_exponential/validation_metadata.json"
            ),
            256: Path(
                "/home/filip/server_code/labs/figures_for_paper_digits/timings/"
                "single_diode_exponential/hidden_2/hidden_256/20260317-094836_single_diode_exponential/"
                "codex_cpu_rerun/20260319-114203_single_diode_exponential/validation_metadata.json"
            ),
            512: Path(
                "/home/filip/server_code/labs/figures_for_paper_digits/timings/"
                "single_diode_exponential/hidden_2/hidden_512/20260317-105922_single_diode_exponential/"
                "codex_cpu_rerun/20260319-114318_single_diode_exponential/validation_metadata.json"
            ),
            1024: Path(
                "/home/filip/server_code/labs/figures_for_paper_digits/timings/"
                "single_diode_exponential/hidden_2/hidden_1024/20260317-110028_single_diode_exponential/"
                "codex_cpu_rerun/20260319-155124_single_diode_exponential/validation_metadata.json"
            ),
        },
        "double_diode_exponential": {
            64: Path(
                "/home/filip/server_code/labs/figures_for_paper_digits/timings/"
                "double_diode_exponential/hidden_2/hidden_64/20260311-170549_double_diode_exponential/"
                "codex_cpu_rerun_float_32/20260319-150100_double_diode_exponential/validation_metadata.json"
            ),
            128: Path(
                "/home/filip/server_code/labs/figures_for_paper_digits/timings/"
                "double_diode_exponential/hidden_2/hidden_128/super_high_accuracy/20260316-145914_double_diode_exponential/"
                "codex_cpu_rerun_float_32/20260319-150110_double_diode_exponential/validation_metadata.json"
            ),
            256: Path(
                "/home/filip/server_code/labs/figures_for_paper_digits/timings/"
                "double_diode_exponential/hidden_2/hidden_256/20260311-170612_double_diode_exponential/"
                "codex_cpu_rerun_float_32/20260319-150126_double_diode_exponential/validation_metadata.json"
            ),
            512: Path(
                "/home/filip/server_code/labs/figures_for_paper_digits/timings/"
                "double_diode_exponential/hidden_2/hidden_512/20260311-170652_double_diode_exponential/"
                "codex_cpu_rerun_float_32/20260319-150141_double_diode_exponential/validation_metadata.json"
            ),
            1024: Path(
                "/home/filip/server_code/labs/figures_for_paper_digits/timings/"
                "double_diode_exponential/hidden_2/hidden_1024/20260311-170758_double_diode_exponential/"
                "codex_cpu_rerun_float_32/20260319-150205_double_diode_exponential/validation_metadata.json"
            ),
        },
        "experimental": {
            64: Path(
                "/home/filip/server_code/labs/figures_for_paper_digits/timings/"
                "experimental/hidden_2/hidden_64/20260313-150228_experimental/"
                "codex_cpu_rerun/20260320-133359_experimental/validation_metadata.json"
            ),
            128: Path(
                "/home/filip/server_code/labs/figures_for_paper_digits/timings/"
                "experimental/hidden_2/hidden_128/20260313-150244_experimental/"
                "codex_cpu_rerun/20260320-133407_experimental/validation_metadata.json"
            ),
            256: Path(
                "/home/filip/server_code/labs/figures_for_paper_digits/timings/"
                "experimental/hidden_2/hidden_256/20260313-150302_experimental/"
                "codex_cpu_rerun/20260320-133415_experimental/validation_metadata.json"
            ),
            512: Path(
                "/home/filip/server_code/labs/figures_for_paper_digits/timings/"
                "experimental/hidden_2/hidden_512/20260313-150320_experimental/"
                "codex_cpu_rerun/20260320-133425_experimental/validation_metadata.json"
            ),
            1024: Path(
                "/home/filip/server_code/labs/figures_for_paper_digits/timings/"
                "experimental/hidden_2/hidden_1024/20260313-150347_experimental/"
                "codex_cpu_rerun/20260320-133440_experimental/validation_metadata.json"
            ),
        },
    },
    3: {
        "single_diode_exponential": {
            64: Path(
                "/home/filip/server_code/labs/figures_for_paper_digits/timings/"
                "single_diode_exponential/hidden_3/hidden_64/codex_cpu_rerun/"
                "20260319-100810_single_diode_exponential/validation_metadata.json"
            ),
            128: Path(
                "/home/filip/server_code/labs/figures_for_paper_digits/timings/"
                "single_diode_exponential/hidden_3/hidden_128/codex_cpu_rerun/"
                "20260319-155148_single_diode_exponential/validation_metadata.json"
            ),
            256: Path(
                "/home/filip/server_code/labs/figures_for_paper_digits/timings/"
                "single_diode_exponential/hidden_3/hidden_256/codex_cpu_rerun/"
                "20260319-155207_single_diode_exponential/validation_metadata.json"
            ),
            512: Path(
                "/home/filip/server_code/labs/figures_for_paper_digits/timings/"
                "single_diode_exponential/hidden_3/hidden_512/codex_cpu_rerun/"
                "20260319-155447_single_diode_exponential/validation_metadata.json"
            ),
        },
        "double_diode_exponential": {
            64: Path(
                "/home/filip/server_code/labs/figures_for_paper_digits/timings/"
                "double_diode_exponential/hidden_3/hidden_64/"
                "20260311-185700_double_diode_exponential/codex_cpu_rerun_float_32/"
                "20260319-150239_double_diode_exponential/validation_metadata.json"
            ),
            128: Path(
                "/home/filip/server_code/labs/figures_for_paper_digits/timings/"
                "double_diode_exponential/hidden_3/hidden_128/"
                "20260313-155212_double_diode_exponential/codex_cpu_rerun_float_32/"
                "20260319-150302_double_diode_exponential/validation_metadata.json"
            ),
            256: Path(
                "/home/filip/server_code/labs/figures_for_paper_digits/timings/"
                "double_diode_exponential/hidden_3/hidden_256/validation/"
                "20260317-161243_double_diode_exponential/codex_cpu_rerun_float_32/"
                "20260319-150325_double_diode_exponential/validation_metadata.json"
            ),
            512: Path(
                "/home/filip/server_code/labs/figures_for_paper_digits/timings/"
                "double_diode_exponential/hidden_3/hidden_512/validation/"
                "20260311-185941_double_diode_exponential/codex_cpu_rerun_float_32/"
                "20260319-150350_double_diode_exponential/validation_metadata.json"
            ),
        },
        "experimental": {
            64: Path(
                "/home/filip/server_code/labs/figures_for_paper_digits/timings/"
                "experimental/hidden_3/hidden_64/validation/20260313-151007_experimental/"
                "codex_cpu_rerun/20260320-133506_experimental/validation_metadata.json"
            ),
            128: Path(
                "/home/filip/server_code/labs/figures_for_paper_digits/timings/"
                "experimental/hidden_3/hidden_128/amp_4/validation/20260313-151035_experimental/"
                "codex_cpu_rerun/20260320-133516_experimental/validation_metadata.json"
            ),
            256: Path(
                "/home/filip/server_code/labs/figures_for_paper_digits/timings/"
                "experimental/hidden_3/hidden_256/validation/20260313-151121_experimental/"
                "codex_cpu_rerun/20260320-133532_experimental/validation_metadata.json"
            ),
            512: Path(
                "/home/filip/server_code/labs/figures_for_paper_digits/timings/"
                "experimental/hidden_3/hidden_512/validation/20260313-171929_experimental/"
                "codex_cpu_rerun/20260320-133540_experimental/validation_metadata.json"
            ),
        },
    },
}


FAMILY_STYLE = [
    ("single_diode_exponential", "Single diode", "#1f77b4", "S"),
    ("double_diode_exponential", "Double diode", "#d62728", "D"),
    ("experimental", "Experimental I-V", "#2ca02c", "E"),
]

SCATTER_METRICS = [
    ("rho_full_mean_diag", r"$\rho(T)$", False),
    ("inv_neg_log_rho", r"$1/(-\log \rho(T))$", True),
    ("inv_one_minus_rho", r"$1/(1-\rho(T))$", True),
    ("one_minus_sigma_over_L_block_min", r"$1 - \sigma / \min(L_{\mathrm{odd}}, L_{\mathrm{even}})$", False),
    ("sigma", r"$\sigma$", True),
    ("inv_neg_log_sigma", r"$1/(-\log \sigma)$", False),
    ("kappa_full", r"$\kappa_{\mathrm{full}}$", True),
    ("kappa_block_min", r"$\kappa_{\mathrm{block,min}}$", True),
    ("kappa_block_max", r"$\kappa_{\mathrm{block,max}}$", True),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare average outer iterations against frozen-Hessian metrics "
            "for matched single- and double-diode CPU timing runs."
        )
    )
    parser.add_argument(
        "--hidden-layers",
        type=int,
        choices=sorted(CASE_RUNS_BY_DEPTH.keys()),
        default=3,
        help="Number of hidden layers to analyze.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/filip/server_code/labs/cases/conditioning_vs_runtime"),
        help="Directory where the plots and CSV summary will be written.",
    )
    return parser.parse_args()


def load_metadata(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def resolve_config_path(meta: dict) -> Path:
    config_path = meta.get("config_path")
    if config_path:
        return Path(config_path)
    cli_command = meta.get("cli_command", "")
    if cli_command:
        parts = shlex.split(cli_command)
        if "--config" in parts:
            return Path(parts[parts.index("--config") + 1])
    weights_parent = Path(meta["weights_path"]).parent
    configs = sorted(weights_parent.glob("small_network*.json"))
    if not configs:
        raise FileNotFoundError(f"No config found next to weights: {meta['weights_path']}")
    return configs[0]


def load_config(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def resolve_iv_curve_path(meta: dict, cfg: dict) -> Path:
    for key in ("iv_data_path", "LABS_IV_CURVE_PATH"):
        value = meta.get(key)
        if value:
            return Path(value)
    for key in ("iv_data_path", "LABS_IV_CURVE_PATH", "experimental_iv_curve_path"):
        value = cfg.get(key)
        if value:
            return Path(value)
    raise FileNotFoundError("No experimental IV-curve path found in metadata or config.")


def load_iv_curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    if "iv" in data:
        iv = np.asarray(data["iv"], dtype=np.float64)
        return iv[0], iv[1]
    if "i" in data and "v" in data:
        return np.asarray(data["i"], dtype=np.float64), np.asarray(data["v"], dtype=np.float64)
    raise ValueError(f"Expected IV NPZ with 'iv' or both 'i' and 'v'. Got {path}.")


def load_weight_mats(path: Path) -> list[np.ndarray]:
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    tensors = list(state.values()) if isinstance(state, dict) else list(state)
    mats = [
        tensor.detach().cpu().numpy().astype(np.float64)
        for tensor in tensors
        if isinstance(tensor, torch.Tensor) and tensor.ndim == 2
    ]
    if not mats:
        raise ValueError(f"No 2D weight tensors found in {path}")
    return mats


def build_linear_hessian(mats: list[np.ndarray], voltage_amp: float, current_amp: float) -> tuple[np.ndarray, list[int]]:
    layer_dims = [mats[0].shape[0]] + [mat.shape[1] for mat in mats]
    free_dims = layer_dims[1:]
    offsets = np.cumsum([0] + free_dims)
    hessian = np.zeros((offsets[-1], offsets[-1]), dtype=np.float64)

    hessian[offsets[0] : offsets[1], offsets[0] : offsets[1]] += (current_amp**2) * np.diag(mats[0].sum(axis=0))

    for edge_index, weight in enumerate(mats[1:], start=1):
        scale = (current_amp / voltage_amp) ** edge_index
        diag_pre = scale * (voltage_amp**2) * np.diag(weight.sum(axis=1))
        diag_post = scale * (current_amp**2) * np.diag(weight.sum(axis=0))
        off_diag = -scale * (voltage_amp * current_amp) * weight

        pre_start, pre_end = offsets[edge_index - 1], offsets[edge_index]
        post_start, post_end = offsets[edge_index], offsets[edge_index + 1]
        hessian[pre_start:pre_end, pre_start:pre_end] += diag_pre
        hessian[post_start:post_end, post_start:post_end] += diag_post
        hessian[pre_start:pre_end, post_start:post_end] += off_diag
        hessian[post_start:post_end, pre_start:pre_end] += off_diag.T

    return hessian, free_dims


def mean_nonlinear_diag(
    nonlinearity: str,
    states_path: Path,
    cfg: dict,
    free_dims: list[int],
    meta: dict,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    states = np.load(states_path)
    params = cfg["exponential_diode_param"]
    i_s = float(params["I_s"])
    v_t = float(params["V_t"])
    v_off = float(params["V_off"])
    voltage_amp = float(cfg["voltage_amp"])
    current_amp = float(cfg["current_amp"])

    pieces: list[np.ndarray] = []
    layer_stats: list[dict[str, float]] = []
    hidden_count = len(free_dims) - 1

    for layer_idx in range(1, hidden_count + 1):
        layer = states[f"Layer_{layer_idx}"].astype(np.float64)
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
        elif nonlinearity == "experimental":
            iv_path = resolve_iv_curve_path(meta, cfg)
            i_data, v_data = load_iv_curve(iv_path)
            slope = np.diff(i_data) / np.diff(v_data)
            idx = np.searchsorted(v_data, layer, side="left") - 1
            idx = np.clip(idx, 0, len(v_data) - 2)
            deriv = slope[idx]
        else:
            raise ValueError(f"Unsupported nonlinearity: {nonlinearity}")

        mean_diag = deriv.mean(axis=0)
        pieces.append(mean_diag)
        layer_stats.append(
            {
                "mean": float(mean_diag.mean()),
                "min": float(mean_diag.min()),
                "max": float(mean_diag.max()),
                "nonzero_fraction": float((mean_diag > 0.0).mean()),
            }
        )

    pieces.append(np.zeros(free_dims[-1], dtype=np.float64))
    return np.concatenate(pieces), layer_stats


def symmetric_eigvalsh_clipped(matrix: np.ndarray, rel_tol: float = 1e-10, abs_tol: float = 1e-12) -> np.ndarray:
    eigvals = np.linalg.eigvalsh(matrix)
    scale = max(float(np.max(np.abs(eigvals))), 1.0)
    tol = max(abs_tol, rel_tol * scale)
    clipped = eigvals.copy()
    near_zero_neg = (clipped < 0.0) & (clipped > -tol)
    clipped[near_zero_neg] = 0.0
    return clipped


def odd_even_partition_indices(free_dims: list[int]) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.cumsum([0] + free_dims)
    odd_blocks: list[np.ndarray] = []
    even_blocks: list[np.ndarray] = []
    for layer_idx in range(1, len(free_dims) + 1):
        block = np.arange(offsets[layer_idx - 1], offsets[layer_idx])
        if layer_idx % 2 == 1:
            odd_blocks.append(block)
        else:
            even_blocks.append(block)
    odd_idx = np.concatenate(odd_blocks) if odd_blocks else np.array([], dtype=int)
    even_idx = np.concatenate(even_blocks) if even_blocks else np.array([], dtype=int)
    return odd_idx, even_idx


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0.0 else math.nan


def one_minus_safe_ratio(numerator: float, denominator: float) -> float:
    ratio = safe_ratio(numerator, denominator)
    return float(1.0 - ratio) if math.isfinite(ratio) else math.nan


def inv_neg_log(value: float) -> float:
    value = float(value)
    if not (value > 0.0):
        return math.nan
    log_value = math.log(value)
    if abs(log_value) < 1e-12:
        return math.nan
    return -1.0 / log_value


def stopping_time_proxies(rho: float) -> tuple[float, float]:
    rho = float(rho)
    if not (0.0 < rho < 1.0):
        return math.nan, math.nan
    return inv_neg_log(rho), 1.0 / (1.0 - rho)


def block_curvature_metrics(hessian: np.ndarray, free_dims: list[int]) -> dict[str, float]:
    odd_idx, even_idx = odd_even_partition_indices(free_dims)
    reordered_idx = np.concatenate([odd_idx, even_idx])
    reordered = hessian[np.ix_(reordered_idx, reordered_idx)]
    n_odd = len(odd_idx)
    h_oo = reordered[:n_odd, :n_odd]
    h_ee = reordered[n_odd:, n_odd:]

    eig_full = symmetric_eigvalsh_clipped(reordered)
    eig_odd = symmetric_eigvalsh_clipped(h_oo)
    eig_even = symmetric_eigvalsh_clipped(h_ee)

    sigma = float(eig_full[0])
    l_full = float(eig_full[-1])
    l_odd = float(eig_odd[-1])
    l_even = float(eig_even[-1])

    min_block = min(l_odd, l_even)
    max_block = max(l_odd, l_even)
    return {
        "L_odd": l_odd,
        "L_even": l_even,
        "sigma": sigma,
        "L_full": l_full,
        "kappa_full": safe_ratio(l_full, sigma),
        "kappa_block_min": safe_ratio(min_block, sigma),
        "kappa_block_max": safe_ratio(max_block, sigma),
        "sigma_over_L_full": safe_ratio(sigma, l_full),
        "sigma_over_L_block_min": safe_ratio(sigma, min_block),
        "sigma_over_L_block_max": safe_ratio(sigma, max_block),
        "one_minus_sigma_over_L_full": one_minus_safe_ratio(sigma, l_full),
        "one_minus_sigma_over_L_block_min": one_minus_safe_ratio(sigma, min_block),
        "one_minus_sigma_over_L_block_max": one_minus_safe_ratio(sigma, max_block),
    }


def odd_even_spectral_radius(hessian: np.ndarray, free_dims: list[int]) -> float:
    odd_idx, even_idx = odd_even_partition_indices(free_dims)
    h_oo = hessian[np.ix_(odd_idx, odd_idx)]
    h_oe = hessian[np.ix_(odd_idx, even_idx)]
    h_eo = hessian[np.ix_(even_idx, odd_idx)]
    h_ee = hessian[np.ix_(even_idx, even_idx)]
    odd_half = np.linalg.solve(h_oo, h_oe)
    even_half = np.linalg.solve(h_ee, h_eo)
    eigvals = np.linalg.eigvals(even_half @ odd_half)
    return float(np.max(np.abs(eigvals)))


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return math.nan
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_vals = values[order]
    ranks = np.zeros(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_vals[end] == sorted_vals[start]:
            end += 1
        avg_rank = 0.5 * (start + end - 1) + 1.0
        ranks[start:end] = avg_rank
        start = end
    out = np.empty_like(ranks)
    out[order] = ranks
    return out


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return math.nan
    return pearson_corr(rankdata(x), rankdata(y))


def correlation_summary(rows: list[dict], metrics: list[str]) -> dict[str, dict[str, dict[str, float]]]:
    by_family = {"all_runs": rows}
    for family_key, _label, _color, _prefix in FAMILY_STYLE:
        by_family[family_key] = [row for row in rows if row["nonlinearity"] == family_key]

    summary: dict[str, dict[str, dict[str, float]]] = {}
    for group_name, group_rows in by_family.items():
        group_summary: dict[str, dict[str, float]] = {}
        y = np.asarray([row["avg_outer_iterations"] for row in group_rows], dtype=np.float64)
        for metric in metrics:
            x = np.asarray([row[metric] for row in group_rows], dtype=np.float64)
            finite_mask = np.isfinite(x) & np.isfinite(y)
            x_f = x[finite_mask]
            y_f = y[finite_mask]
            group_summary[metric] = {
                "n": int(x_f.size),
                "pearson": pearson_corr(x_f, y_f),
                "spearman": spearman_corr(x_f, y_f),
            }
        summary[group_name] = group_summary
    return summary


def analyze_case(nonlinearity: str, width: int, meta_path: Path) -> dict:
    meta = load_metadata(meta_path)
    cfg_path = resolve_config_path(meta)
    cfg = load_config(cfg_path)
    mats = load_weight_mats(Path(meta["weights_path"]))
    hessian_lin, free_dims = build_linear_hessian(mats, float(cfg["voltage_amp"]), float(cfg["current_amp"]))
    mean_diag, layer_stats = mean_nonlinear_diag(
        nonlinearity, Path(meta["validation_states"]), cfg, free_dims, meta
    )
    hessian_full = hessian_lin + np.diag(mean_diag)
    metrics = block_curvature_metrics(hessian_full, free_dims)

    rho_lin = odd_even_spectral_radius(hessian_lin, free_dims)
    rho_full = odd_even_spectral_radius(hessian_full, free_dims)
    inv_neg_log_rho, inv_one_minus_rho = stopping_time_proxies(rho_full)
    inv_neg_log_sigma = inv_neg_log(metrics["sigma"])

    row = {
        "nonlinearity": nonlinearity,
        "width": width,
        "avg_outer_iterations": float(meta["equilibrium_iterations"]["avg_iterations"]),
        "avg_inner_iterations": float(meta.get("newton_iteration_stats", {}).get("avg_iterations", math.nan)),
        "lambda_min_positive": metrics["sigma"],
        "lambda_max": metrics["L_full"],
        "condition_ratio": metrics["kappa_full"],
        "rho_lin": rho_lin,
        "rho_full_mean_diag": rho_full,
        "inv_neg_log_rho": inv_neg_log_rho,
        "inv_one_minus_rho": inv_one_minus_rho,
        "L_odd": metrics["L_odd"],
        "L_even": metrics["L_even"],
        "sigma": metrics["sigma"],
        "inv_neg_log_sigma": inv_neg_log_sigma,
        "L_full": metrics["L_full"],
        "kappa_full": metrics["kappa_full"],
        "kappa_block_min": metrics["kappa_block_min"],
        "kappa_block_max": metrics["kappa_block_max"],
        "sigma_over_L_full": metrics["sigma_over_L_full"],
        "sigma_over_L_block_min": metrics["sigma_over_L_block_min"],
        "sigma_over_L_block_max": metrics["sigma_over_L_block_max"],
        "one_minus_sigma_over_L_full": metrics["one_minus_sigma_over_L_full"],
        "one_minus_sigma_over_L_block_min": metrics["one_minus_sigma_over_L_block_min"],
        "one_minus_sigma_over_L_block_max": metrics["one_minus_sigma_over_L_block_max"],
        "config_path": str(cfg_path),
        "weights_path": meta["weights_path"],
        "states_path": meta["validation_states"],
        "output_dim": int(meta["dims"][-1]),
    }
    for layer_number in range(1, 4):
        if layer_number <= len(layer_stats):
            stats = layer_stats[layer_number - 1]
            row[f"nl_layer{layer_number}_mean"] = stats["mean"]
            row[f"nl_layer{layer_number}_nonzero_fraction"] = stats["nonzero_fraction"]
        else:
            row[f"nl_layer{layer_number}_mean"] = math.nan
            row[f"nl_layer{layer_number}_nonzero_fraction"] = math.nan
    return row


def write_csv(rows: list[dict], output_path: Path) -> None:
    fieldnames = [
        "nonlinearity",
        "width",
        "avg_outer_iterations",
        "avg_inner_iterations",
        "lambda_min_positive",
        "lambda_max",
        "condition_ratio",
        "rho_lin",
        "rho_full_mean_diag",
        "inv_neg_log_rho",
        "inv_one_minus_rho",
        "L_odd",
        "L_even",
        "sigma",
        "inv_neg_log_sigma",
        "L_full",
        "kappa_full",
        "kappa_block_min",
        "kappa_block_max",
        "sigma_over_L_full",
        "sigma_over_L_block_min",
        "sigma_over_L_block_max",
        "one_minus_sigma_over_L_full",
        "one_minus_sigma_over_L_block_min",
        "one_minus_sigma_over_L_block_max",
        "output_dim",
        "nl_layer1_mean",
        "nl_layer2_mean",
        "nl_layer3_mean",
        "nl_layer1_nonzero_fraction",
        "nl_layer2_nonzero_fraction",
        "nl_layer3_nonzero_fraction",
        "config_path",
        "weights_path",
        "states_path",
    ]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_metric_plot(rows: list[dict], metric_key: str, metric_title: str, metric_ylabel: str, output_path: Path, log_metric: bool = False) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    widths_all = sorted({row["width"] for row in rows})

    for family_key, label, color, _prefix in FAMILY_STYLE:
        family_rows = sorted([row for row in rows if row["nonlinearity"] == family_key], key=lambda row: row["width"])
        widths = [row["width"] for row in family_rows]
        avg_iters = [row["avg_outer_iterations"] for row in family_rows]
        metric_vals = [row[metric_key] for row in family_rows]

        axes[0].plot(widths, avg_iters, marker="o", linewidth=2, color=color, label=label)
        axes[1].plot(widths, metric_vals, marker="o", linewidth=2, color=color, label=label)

    axes[0].set_title("Average Outer Iterations")
    axes[0].set_xlabel("Hidden Width")
    axes[0].set_ylabel("Average outer sweeps")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title(metric_title)
    axes[1].set_xlabel("Hidden Width")
    axes[1].set_ylabel(metric_ylabel)
    if log_metric:
        axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.3, which="both")

    for ax in axes:
        ax.set_xticks(widths_all)
        ax.legend(frameon=False)

    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def make_rho_transform_plot(rows: list[dict], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(18.5, 4.8), constrained_layout=True)
    widths_all = sorted({row["width"] for row in rows})
    metric_specs = [
        ("avg_outer_iterations", "Average Outer Iterations", "Average outer sweeps", False),
        ("rho_full_mean_diag", r"$\rho(T)$", r"$\rho(T)$", False),
        ("inv_neg_log_rho", r"$1/(-\log \rho(T))$", r"$1/(-\log \rho(T))$", True),
        ("inv_one_minus_rho", r"$1/(1-\rho(T))$", r"$1/(1-\rho(T))$", True),
    ]

    for ax, (metric_key, title, ylabel, use_log_y) in zip(axes, metric_specs):
        for family_key, label, color, _prefix in FAMILY_STYLE:
            family_rows = sorted([row for row in rows if row["nonlinearity"] == family_key], key=lambda row: row["width"])
            widths = [row["width"] for row in family_rows]
            vals = [row[metric_key] for row in family_rows]
            ax.plot(widths, vals, marker="o", linewidth=2, color=color, label=label)
        ax.set_title(title)
        ax.set_xlabel("Hidden Width")
        ax.set_ylabel(ylabel)
        if use_log_y:
            ax.set_yscale("log")
        ax.set_xticks(widths_all)
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(frameon=False)

    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def make_scatter_plots(rows: list[dict], output_path: Path) -> None:
    correlations = correlation_summary(rows, [metric for metric, _label, _log in SCATTER_METRICS])
    fig, axes = plt.subplots(3, 3, figsize=(15.5, 11.0), constrained_layout=True)
    flat_axes = axes.ravel()

    for ax, (metric_key, label, use_log_x) in zip(flat_axes, SCATTER_METRICS):
        for family_key, _family_label, color, prefix in FAMILY_STYLE:
            family_rows = [row for row in rows if row["nonlinearity"] == family_key]
            x = np.asarray([row[metric_key] for row in family_rows], dtype=np.float64)
            y = np.asarray([row["avg_outer_iterations"] for row in family_rows], dtype=np.float64)
            ax.scatter(x, y, color=color, s=50, alpha=0.9)
            for row, x_val, y_val in zip(family_rows, x, y):
                ax.annotate(f"{prefix}{row['width']}", (x_val, y_val), textcoords="offset points", xytext=(4, 4), fontsize=8, color=color)
        if use_log_x:
            ax.set_xscale("log")
        ax.set_xlabel(label)
        ax.set_ylabel("Average outer sweeps")
        corr = correlations["all_runs"][metric_key]
        pearson = corr["pearson"]
        spearman = corr["spearman"]
        p_txt = "nan" if math.isnan(pearson) else f"{pearson:.3f}"
        s_txt = "nan" if math.isnan(spearman) else f"{spearman:.3f}"
        ax.set_title(f"{label}\nPearson={p_txt}, Spearman={s_txt}")
        ax.grid(True, alpha=0.3, which="both")

    if len(flat_axes) > len(SCATTER_METRICS):
        flat_axes[-1].axis("off")
        handles = [
            plt.Line2D([], [], color=color, marker="o", linestyle="", label=label)
            for _family_key, label, color, _prefix in FAMILY_STYLE
        ]
        flat_axes[-1].legend(handles=handles, loc="center", frameon=False)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_correlations(correlations: dict, output_path: Path) -> None:
    with output_path.open("w") as f:
        json.dump(correlations, f, indent=2)
        f.write("\n")


def output_paths(output_dir: Path, hidden_layers: int) -> dict[str, Path]:
    suffix = f"hidden{hidden_layers}"
    return {
        "csv": output_dir / f"conditioning_vs_runtime_{suffix}.csv",
        "condition_png": output_dir / f"conditioning_vs_runtime_{suffix}.png",
        "rho_png": output_dir / f"rho_vs_runtime_{suffix}.png",
        "sigma_png": output_dir / f"inv_neg_log_sigma_vs_runtime_{suffix}.png",
        "scatter_png": output_dir / f"scatter_metrics_vs_runtime_{suffix}.png",
        "corr_json": output_dir / f"scatter_metric_correlations_{suffix}.json",
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    case_runs = CASE_RUNS_BY_DEPTH[args.hidden_layers]
    rows = []
    for nonlinearity, width_map in case_runs.items():
        for width, meta_path in sorted(width_map.items()):
            rows.append(analyze_case(nonlinearity, width, meta_path))

    paths = output_paths(args.output_dir, args.hidden_layers)
    write_csv(rows, paths["csv"])
    make_metric_plot(
        rows,
        metric_key="condition_ratio",
        metric_title=r"Conditioning Proxy: $\lambda_{\max} / \lambda_{\min}^{+}$",
        metric_ylabel=r"$\lambda_{\max} / \lambda_{\min}^{+}$",
        output_path=paths["condition_png"],
        log_metric=True,
    )
    make_rho_transform_plot(rows, paths["rho_png"])
    make_metric_plot(
        rows,
        metric_key="inv_neg_log_sigma",
        metric_title=r"Stopping-Time Proxy: $1/(-\log \sigma)$",
        metric_ylabel=r"$1/(-\log \sigma)$",
        output_path=paths["sigma_png"],
        log_metric=False,
    )
    correlations = correlation_summary(rows, [metric for metric, _label, _log in SCATTER_METRICS])
    make_scatter_plots(rows, paths["scatter_png"])
    write_correlations(correlations, paths["corr_json"])

    print(f"Wrote {paths['csv']}")
    print(f"Wrote {paths['condition_png']}")
    print(f"Wrote {paths['rho_png']}")
    print(f"Wrote {paths['sigma_png']}")
    print(f"Wrote {paths['scatter_png']}")
    print(f"Wrote {paths['corr_json']}")


if __name__ == "__main__":
    main()
