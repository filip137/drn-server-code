#!/usr/bin/env python3
"""Output decomposition and physical log-G noise for hard-sigmoid MNIST DRNs.

This diagnostic checks whether the finite-difference test-cost curvature is
coming from the Gauss-Newton output response term or from the residual/nonlinear
term.  For the same random software-weight directions used by
``evaluate_mnist_bp_cost_hessian_noise_curvature.py`` it evaluates

    q_cost = (L(theta + eps d) - 2 L(theta) + L(theta - eps d)) / eps**2
    q_gn = ||(y_plus - y_minus) / (2 eps)||**2
    q_residual = (y0 - t)^T (y_plus - 2 y0 + y_minus) / eps**2

It also repeats the decomposition and noisy inference in the incidence
coordinate, i.e. per-edge conductance/log-G perturbations
``G <- G * exp(sigma * xi)`` on DenseWeight parameters only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from evaluate_mnist_bp_cost_hessian_noise_curvature import (  # noqa: E402
    DEFAULT_NOISE_ROOT,
    RUN_COLORS,
    RUN_LABELS,
    RUN_ORDER,
    _apply_direction,
    _finite_stats,
    _make_direction,
    _selected_param_indices,
)
from evaluate_mnist_bp_physical_cost_sharpness import DEFAULT_OUTPUT_ROOT as DEFAULT_SWRITE_ROOT  # noqa: E402
from evaluate_mnist_bp_write_noise_sweep import (  # noqa: E402
    DEFAULT_INPUT_ROOT,
    RunRow,
    _build_eval_context,
    _json_sanitize,
    _read_summary_rows,
    _reset_params,
    _scores,
    _select_shard,
    _write_json,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "mnist_bp_output_decomp_physical_logg_noise_hardsigmoid_iter16"
DEFAULT_SIGMAS = [0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0]

DECOMP_COLUMNS = [
    "run_name",
    "seed",
    "voltage_amp",
    "current_amp",
    "checkpoint_path",
    "coordinate",
    "epsilon",
    "direction_index",
    "direction_seed",
    "num_examples",
    "clean_loss",
    "clean_accuracy",
    "loss_plus",
    "loss_minus",
    "accuracy_plus",
    "accuracy_minus",
    "q_cost",
    "q_gn",
    "q_residual",
    "q_decomposition",
    "q_closure_error",
    "q_closure_relerr",
    "q_residual_fraction",
    "direction_l2_sq",
    "normalized_q_cost",
]

DECOMP_MODEL_COLUMNS = [
    "run_name",
    "seed",
    "voltage_amp",
    "current_amp",
    "coordinate",
    "epsilon",
    "num_directions",
    "clean_loss",
    "clean_accuracy",
    "q_cost_mean",
    "q_cost_std",
    "q_cost_p10",
    "q_cost_p50",
    "q_cost_p90",
    "q_gn_mean",
    "q_gn_std",
    "q_gn_p10",
    "q_gn_p50",
    "q_gn_p90",
    "q_residual_mean",
    "q_residual_std",
    "q_residual_p10",
    "q_residual_p50",
    "q_residual_p90",
    "q_closure_relerr_mean",
    "q_residual_fraction_mean",
    "direction_l2_sq_mean",
    "normalized_q_cost_mean",
    "s_write_mean",
]

DECOMP_AMP_COLUMNS = [
    "run_name",
    "voltage_amp",
    "current_amp",
    "coordinate",
    "epsilon",
    "num_models",
    "num_directions",
    "q_cost_mean",
    "q_cost_std_across_models",
    "q_gn_mean",
    "q_gn_std_across_models",
    "q_residual_mean",
    "q_residual_std_across_models",
    "q_closure_relerr_mean",
    "q_residual_fraction_mean",
    "normalized_q_cost_mean",
    "s_write_mean",
    "q_gn_over_s_write",
    "q_cost_over_s_write",
]

NOISE_COLUMNS = [
    "run_name",
    "seed",
    "voltage_amp",
    "current_amp",
    "checkpoint_path",
    "coordinate",
    "sigma",
    "noise_seed",
    "num_examples",
    "clean_loss",
    "clean_accuracy",
    "noisy_loss",
    "noisy_accuracy",
    "loss_increase",
    "accuracy_drop",
    "clamp_noisy",
    "s_write_mean",
    "pred_loss_increase_swrite",
]

NOISE_MODEL_COLUMNS = [
    "run_name",
    "seed",
    "voltage_amp",
    "current_amp",
    "coordinate",
    "sigma",
    "num_noise_seeds",
    "clean_loss",
    "clean_accuracy",
    "mean_noisy_loss",
    "std_noisy_loss",
    "mean_noisy_accuracy",
    "std_noisy_accuracy",
    "mean_loss_increase",
    "mean_accuracy_drop",
    "s_write_mean",
    "pred_loss_increase_swrite",
]

NOISE_AMP_COLUMNS = [
    "run_name",
    "voltage_amp",
    "current_amp",
    "coordinate",
    "sigma",
    "num_models",
    "num_noise_seeds",
    "clean_loss_mean",
    "clean_accuracy_mean",
    "mean_noisy_loss",
    "mean_noisy_accuracy",
    "mean_loss_increase",
    "mean_accuracy_drop",
    "s_write_mean",
    "pred_loss_increase_swrite",
    "loss_increase_over_prediction",
]


def _write_rows(path: Path, columns: list[str], rows: list[dict], *, append: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and append
    mode = "a" if append else "w"
    with path.open(mode, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({column: _json_sanitize(row.get(column, "")) for column in columns})


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _finite_float(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def _mean(values: list[float]) -> float:
    arr = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    return float(np.mean(arr)) if arr.size else math.nan


def _swrite_by_model(swrite_root: Path) -> dict[tuple[str, int], float]:
    path = swrite_root / "summary.csv"
    if not path.exists():
        return {}
    out: dict[tuple[str, int], float] = {}
    for row in _read_rows(path):
        out[(row["run_name"], int(row["seed"]))] = _finite_float(row.get("s_write_mean"))
    return out


def _collect_outputs(context: dict, *, max_eval_batches: int | None = None) -> dict[str, np.ndarray | float | int]:
    network = context["network"]
    minimizer = context["minimizer"]
    cost_fn = context["cost_fn"]
    output_layer = context["output_layer"]
    test_loader = context["test_loader"]
    device = context["base_states"][0].device
    num_classes = context["num_classes"]

    ys: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    losses: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    pred_all: list[np.ndarray] = []

    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(test_loader):
            if max_eval_batches is not None and batch_index >= max_eval_batches:
                break
            images = images.to(device)
            labels = labels.to(device)
            network.set_input(images, reset=True)
            minimizer.compute_equilibrium()
            cost_fn.set_target(labels)
            scores = _scores(output_layer.state.detach(), num_classes)
            target = F.one_hot(labels, num_classes=num_classes).to(dtype=scores.dtype)
            loss = 0.5 * ((scores - target) ** 2).sum(dim=1)
            pred = torch.argmax(scores, dim=1)

            ys.append(scores.detach().cpu().to(torch.float64).numpy())
            targets.append(target.detach().cpu().to(torch.float64).numpy())
            losses.append(loss.detach().cpu().to(torch.float64).numpy())
            labels_all.append(labels.detach().cpu().numpy())
            pred_all.append(pred.detach().cpu().numpy())

    if not ys:
        raise ValueError("Evaluation saw zero examples.")
    y = np.concatenate(ys, axis=0)
    target = np.concatenate(targets, axis=0)
    losses_arr = np.concatenate(losses, axis=0)
    labels_arr = np.concatenate(labels_all, axis=0)
    pred_arr = np.concatenate(pred_all, axis=0)
    return {
        "y": y,
        "target": target,
        "losses": losses_arr,
        "labels": labels_arr,
        "predictions": pred_arr,
        "loss": float(np.mean(losses_arr)),
        "accuracy": float(np.mean(pred_arr == labels_arr)),
        "num_examples": int(y.shape[0]),
    }


def _make_logg_xi(
    context: dict,
    *,
    generator: torch.Generator,
    selected_indices: set[int],
) -> tuple[list[torch.Tensor | None], float]:
    xis: list[torch.Tensor | None] = []
    l2_sq = 0.0
    for index, base in enumerate(context["base_states"]):
        if index not in selected_indices:
            xis.append(None)
            continue
        xi = torch.randn(base.shape, generator=generator, device=base.device, dtype=base.dtype)
        xis.append(xi)
        l2_sq += float(torch.sum((base * xi).detach() ** 2).item())
    return xis, l2_sq


def _apply_logg(
    context: dict,
    xis: list[torch.Tensor | None],
    *,
    magnitude: float,
    sign: float,
    clamp: bool,
) -> None:
    for param, base, xi in zip(context["params"], context["base_states"], xis):
        if xi is None:
            param.state = base.detach().clone()
            continue
        param.state = (base * torch.exp(float(sign) * float(magnitude) * xi)).detach().clone()
        if clamp:
            param.clamp_()


def _decomposition_from_outputs(clean: dict, plus: dict, minus: dict, epsilon: float) -> dict[str, float]:
    y0 = clean["y"]
    yp = plus["y"]
    ym = minus["y"]
    target = clean["target"]
    losses0 = clean["losses"]
    lossesp = plus["losses"]
    lossesm = minus["losses"]
    eps = float(epsilon)
    delta_y_lin = (yp - ym) / (2.0 * eps)
    delta_y_quad = (yp - 2.0 * y0 + ym) / (eps * eps)
    q_cost_sample = (lossesp - 2.0 * losses0 + lossesm) / (eps * eps)
    q_gn_sample = np.sum(delta_y_lin * delta_y_lin, axis=1)
    q_res_sample = np.sum((y0 - target) * delta_y_quad, axis=1)
    q_decomp_sample = q_gn_sample + q_res_sample
    closure = q_cost_sample - q_decomp_sample
    q_cost = float(np.mean(q_cost_sample))
    q_gn = float(np.mean(q_gn_sample))
    q_res = float(np.mean(q_res_sample))
    q_decomp = float(np.mean(q_decomp_sample))
    return {
        "loss_plus": float(plus["loss"]),
        "loss_minus": float(minus["loss"]),
        "accuracy_plus": float(plus["accuracy"]),
        "accuracy_minus": float(minus["accuracy"]),
        "q_cost": q_cost,
        "q_gn": q_gn,
        "q_residual": q_res,
        "q_decomposition": q_decomp,
        "q_closure_error": float(np.mean(closure)),
        "q_closure_relerr": abs(q_cost - q_decomp) / max(abs(q_cost), 1e-30),
        "q_residual_fraction": q_res / q_cost if q_cost != 0.0 else math.nan,
    }


def _run_decomposition(
    run: RunRow,
    *,
    context: dict,
    args: argparse.Namespace,
    raw_path: Path,
    s_write_mean: float,
) -> None:
    selected_indices = set(_selected_param_indices(context, include_biases=False))
    clean = _collect_outputs(context, max_eval_batches=args.max_eval_batches)
    rows: list[dict] = []
    for direction_index in range(args.num_directions):
        direction_seed = int(args.direction_seed_offset + 100000 * run.seed + direction_index)
        for coordinate in args.decomposition_coordinates:
            generator = torch.Generator(device=str(context["base_states"][0].device))
            generator.manual_seed(direction_seed)
            if coordinate == "software_theta_linear":
                directions, direction_l2_sq = _make_direction(
                    context,
                    generator=generator,
                    selected_indices=selected_indices,
                    target_index=None,
                )
                apply_fn = lambda sign, eps: _apply_direction(  # noqa: E731
                    context,
                    directions,
                    epsilon=float(eps),
                    sign=float(sign),
                    clamp_perturbed=args.clamp_perturbed,
                )
            elif coordinate == "physical_logg_exponential":
                xis, direction_l2_sq = _make_logg_xi(
                    context,
                    generator=generator,
                    selected_indices=selected_indices,
                )
                apply_fn = lambda sign, eps: _apply_logg(  # noqa: E731
                    context,
                    xis,
                    magnitude=float(eps),
                    sign=float(sign),
                    clamp=args.clamp_perturbed,
                )
            else:
                raise ValueError(f"Unknown decomposition coordinate {coordinate!r}.")

            for epsilon in args.epsilons:
                apply_fn(+1.0, float(epsilon))
                plus = _collect_outputs(context, max_eval_batches=args.max_eval_batches)
                apply_fn(-1.0, float(epsilon))
                minus = _collect_outputs(context, max_eval_batches=args.max_eval_batches)
                _reset_params(context)
                metrics = _decomposition_from_outputs(clean, plus, minus, float(epsilon))
                q_cost = metrics["q_cost"]
                rows.append(
                    {
                        "run_name": run.run_name,
                        "seed": run.seed,
                        "voltage_amp": run.voltage_amp,
                        "current_amp": run.current_amp,
                        "checkpoint_path": str(run.checkpoint_path),
                        "coordinate": coordinate,
                        "epsilon": float(epsilon),
                        "direction_index": direction_index,
                        "direction_seed": direction_seed,
                        "num_examples": clean["num_examples"],
                        "clean_loss": clean["loss"],
                        "clean_accuracy": clean["accuracy"],
                        "direction_l2_sq": direction_l2_sq,
                        "normalized_q_cost": q_cost / direction_l2_sq if direction_l2_sq > 0 else math.nan,
                        **metrics,
                    }
                )
                if len(rows) >= 16:
                    _write_rows(raw_path, DECOMP_COLUMNS, rows)
                    rows = []
    if rows:
        _write_rows(raw_path, DECOMP_COLUMNS, rows)


def _run_physical_logg_noise(
    run: RunRow,
    *,
    context: dict,
    args: argparse.Namespace,
    raw_path: Path,
    s_write_mean: float,
) -> None:
    selected_indices = set(_selected_param_indices(context, include_biases=False))
    clean = _collect_outputs(context, max_eval_batches=args.max_eval_batches)
    rows: list[dict] = []
    for sigma in args.sigmas:
        for noise_seed in args.noise_seeds:
            generator = torch.Generator(device=str(context["base_states"][0].device))
            generator.manual_seed(int(noise_seed))
            xis, _ = _make_logg_xi(context, generator=generator, selected_indices=selected_indices)
            _apply_logg(
                context,
                xis,
                magnitude=float(sigma),
                sign=1.0,
                clamp=args.clamp_noisy,
            )
            noisy = _collect_outputs(context, max_eval_batches=args.max_eval_batches)
            _reset_params(context)
            rows.append(
                {
                    "run_name": run.run_name,
                    "seed": run.seed,
                    "voltage_amp": run.voltage_amp,
                    "current_amp": run.current_amp,
                    "checkpoint_path": str(run.checkpoint_path),
                    "coordinate": "physical_logg_exponential",
                    "sigma": float(sigma),
                    "noise_seed": int(noise_seed),
                    "num_examples": clean["num_examples"],
                    "clean_loss": clean["loss"],
                    "clean_accuracy": clean["accuracy"],
                    "noisy_loss": noisy["loss"],
                    "noisy_accuracy": noisy["accuracy"],
                    "loss_increase": float(noisy["loss"] - clean["loss"]),
                    "accuracy_drop": float(clean["accuracy"] - noisy["accuracy"]),
                    "clamp_noisy": bool(args.clamp_noisy),
                    "s_write_mean": s_write_mean,
                    "pred_loss_increase_swrite": 0.5 * float(sigma) * float(sigma) * s_write_mean,
                }
            )
            if len(rows) >= 16:
                _write_rows(raw_path, NOISE_COLUMNS, rows)
                rows = []
    if rows:
        _write_rows(raw_path, NOISE_COLUMNS, rows)


def _summarize_decomposition(output_root: Path, swrite: dict[tuple[str, int], float]) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(output_root.glob("raw_decomposition*.csv")):
        rows.extend(_read_rows(path))
    if not rows:
        return []
    grouped: dict[tuple[str, int, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["run_name"], int(row["seed"]), row["coordinate"], row["epsilon"])].append(row)

    model_rows: list[dict] = []
    for (run_name, seed, coordinate, epsilon), values in sorted(grouped.items()):
        first = values[0]
        q_cost = [_finite_float(row["q_cost"]) for row in values]
        q_gn = [_finite_float(row["q_gn"]) for row in values]
        q_res = [_finite_float(row["q_residual"]) for row in values]
        closure = [_finite_float(row["q_closure_relerr"]) for row in values]
        res_frac = [_finite_float(row["q_residual_fraction"]) for row in values]
        norms = [_finite_float(row["direction_l2_sq"]) for row in values]
        normalized = [_finite_float(row["normalized_q_cost"]) for row in values]
        q_cost_stats = _finite_stats(q_cost)
        q_gn_stats = _finite_stats(q_gn)
        q_res_stats = _finite_stats(q_res)
        model_rows.append(
            {
                "run_name": run_name,
                "seed": seed,
                "voltage_amp": first["voltage_amp"],
                "current_amp": first["current_amp"],
                "coordinate": coordinate,
                "epsilon": float(epsilon),
                "num_directions": len(values),
                "clean_loss": first["clean_loss"],
                "clean_accuracy": first["clean_accuracy"],
                "q_cost_mean": q_cost_stats["mean"],
                "q_cost_std": q_cost_stats["std"],
                "q_cost_p10": q_cost_stats["p10"],
                "q_cost_p50": q_cost_stats["p50"],
                "q_cost_p90": q_cost_stats["p90"],
                "q_gn_mean": q_gn_stats["mean"],
                "q_gn_std": q_gn_stats["std"],
                "q_gn_p10": q_gn_stats["p10"],
                "q_gn_p50": q_gn_stats["p50"],
                "q_gn_p90": q_gn_stats["p90"],
                "q_residual_mean": q_res_stats["mean"],
                "q_residual_std": q_res_stats["std"],
                "q_residual_p10": q_res_stats["p10"],
                "q_residual_p50": q_res_stats["p50"],
                "q_residual_p90": q_res_stats["p90"],
                "q_closure_relerr_mean": _mean(closure),
                "q_residual_fraction_mean": _mean(res_frac),
                "direction_l2_sq_mean": _mean(norms),
                "normalized_q_cost_mean": _mean(normalized),
                "s_write_mean": swrite.get((run_name, seed), math.nan),
            }
        )
    _write_rows(output_root / "summary_decomposition_by_model.csv", DECOMP_MODEL_COLUMNS, model_rows, append=False)

    amp_grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in model_rows:
        amp_grouped[(row["run_name"], row["coordinate"], str(row["epsilon"]))].append(row)
    amp_rows: list[dict] = []
    for (run_name, coordinate, epsilon), values in sorted(amp_grouped.items()):
        first = values[0]
        q_cost = [_finite_float(row["q_cost_mean"]) for row in values]
        q_gn = [_finite_float(row["q_gn_mean"]) for row in values]
        q_res = [_finite_float(row["q_residual_mean"]) for row in values]
        sw = [_finite_float(row["s_write_mean"]) for row in values]
        q_cost_stats = _finite_stats(q_cost)
        q_gn_stats = _finite_stats(q_gn)
        q_res_stats = _finite_stats(q_res)
        sw_mean = _mean(sw)
        amp_rows.append(
            {
                "run_name": run_name,
                "voltage_amp": first["voltage_amp"],
                "current_amp": first["current_amp"],
                "coordinate": coordinate,
                "epsilon": float(epsilon),
                "num_models": len(values),
                "num_directions": int(sum(int(row["num_directions"]) for row in values)),
                "q_cost_mean": q_cost_stats["mean"],
                "q_cost_std_across_models": q_cost_stats["std"],
                "q_gn_mean": q_gn_stats["mean"],
                "q_gn_std_across_models": q_gn_stats["std"],
                "q_residual_mean": q_res_stats["mean"],
                "q_residual_std_across_models": q_res_stats["std"],
                "q_closure_relerr_mean": _mean([_finite_float(row["q_closure_relerr_mean"]) for row in values]),
                "q_residual_fraction_mean": _mean([_finite_float(row["q_residual_fraction_mean"]) for row in values]),
                "normalized_q_cost_mean": _mean([_finite_float(row["normalized_q_cost_mean"]) for row in values]),
                "s_write_mean": sw_mean,
                "q_gn_over_s_write": q_gn_stats["mean"] / sw_mean if sw_mean > 0 else math.nan,
                "q_cost_over_s_write": q_cost_stats["mean"] / sw_mean if sw_mean > 0 else math.nan,
            }
        )
    _write_rows(output_root / "summary_decomposition_by_amp.csv", DECOMP_AMP_COLUMNS, amp_rows, append=False)
    return amp_rows


def _summarize_noise(output_root: Path, swrite: dict[tuple[str, int], float]) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(output_root.glob("raw_physical_logg_noise*.csv")):
        rows.extend(_read_rows(path))
    if not rows:
        return []
    grouped: dict[tuple[str, int, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["run_name"], int(row["seed"]), row["coordinate"], row["sigma"])].append(row)

    model_rows: list[dict] = []
    for (run_name, seed, coordinate, sigma), values in sorted(grouped.items()):
        first = values[0]
        noisy_loss = [_finite_float(row["noisy_loss"]) for row in values]
        noisy_acc = [_finite_float(row["noisy_accuracy"]) for row in values]
        loss_inc = [_finite_float(row["loss_increase"]) for row in values]
        acc_drop = [_finite_float(row["accuracy_drop"]) for row in values]
        sw = swrite.get((run_name, seed), _finite_float(first.get("s_write_mean")))
        sigma_f = float(sigma)
        model_rows.append(
            {
                "run_name": run_name,
                "seed": seed,
                "voltage_amp": first["voltage_amp"],
                "current_amp": first["current_amp"],
                "coordinate": coordinate,
                "sigma": sigma_f,
                "num_noise_seeds": len(values),
                "clean_loss": first["clean_loss"],
                "clean_accuracy": first["clean_accuracy"],
                "mean_noisy_loss": _mean(noisy_loss),
                "std_noisy_loss": _finite_stats(noisy_loss)["std"],
                "mean_noisy_accuracy": _mean(noisy_acc),
                "std_noisy_accuracy": _finite_stats(noisy_acc)["std"],
                "mean_loss_increase": _mean(loss_inc),
                "mean_accuracy_drop": _mean(acc_drop),
                "s_write_mean": sw,
                "pred_loss_increase_swrite": 0.5 * sigma_f * sigma_f * sw,
            }
        )
    _write_rows(output_root / "summary_physical_logg_noise_by_model_sigma.csv", NOISE_MODEL_COLUMNS, model_rows, append=False)

    amp_grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in model_rows:
        amp_grouped[(row["run_name"], row["coordinate"], str(row["sigma"]))].append(row)
    amp_rows: list[dict] = []
    for (run_name, coordinate, sigma), values in sorted(amp_grouped.items()):
        first = values[0]
        sigma_f = float(sigma)
        sw = _mean([_finite_float(row["s_write_mean"]) for row in values])
        pred = 0.5 * sigma_f * sigma_f * sw
        loss_inc = _mean([_finite_float(row["mean_loss_increase"]) for row in values])
        amp_rows.append(
            {
                "run_name": run_name,
                "voltage_amp": first["voltage_amp"],
                "current_amp": first["current_amp"],
                "coordinate": coordinate,
                "sigma": sigma_f,
                "num_models": len(values),
                "num_noise_seeds": int(sum(int(row["num_noise_seeds"]) for row in values)),
                "clean_loss_mean": _mean([_finite_float(row["clean_loss"]) for row in values]),
                "clean_accuracy_mean": _mean([_finite_float(row["clean_accuracy"]) for row in values]),
                "mean_noisy_loss": _mean([_finite_float(row["mean_noisy_loss"]) for row in values]),
                "mean_noisy_accuracy": _mean([_finite_float(row["mean_noisy_accuracy"]) for row in values]),
                "mean_loss_increase": loss_inc,
                "mean_accuracy_drop": _mean([_finite_float(row["mean_accuracy_drop"]) for row in values]),
                "s_write_mean": sw,
                "pred_loss_increase_swrite": pred,
                "loss_increase_over_prediction": loss_inc / pred if pred > 0 else math.nan,
            }
        )
    _write_rows(output_root / "summary_physical_logg_noise_by_amp_sigma.csv", NOISE_AMP_COLUMNS, amp_rows, append=False)
    return amp_rows


def _ordered_amp_rows(rows: list[dict], *, coordinate: str | None = None) -> list[dict]:
    if coordinate is not None:
        rows = [row for row in rows if row.get("coordinate") == coordinate]
    return sorted(rows, key=lambda row: RUN_ORDER.index(row["run_name"]) if row["run_name"] in RUN_ORDER else 999)


def _plot_decomposition(output_root: Path, rows: list[dict]) -> None:
    if not rows:
        return
    plots_dir = output_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    eps = min(float(row["epsilon"]) for row in rows)
    for coordinate in sorted({row["coordinate"] for row in rows}):
        selected = _ordered_amp_rows(
            [row for row in rows if row["coordinate"] == coordinate and float(row["epsilon"]) == eps]
        )
        if not selected:
            continue
        x = np.arange(len(selected))
        width = 0.26
        fig, ax = plt.subplots(figsize=(7.4, 4.5), constrained_layout=True)
        for offset, key, label in [
            (-width, "q_cost_mean", "q_cost"),
            (0.0, "q_gn_mean", "q_GN"),
            (width, "q_residual_mean", "q_residual"),
        ]:
            ax.bar(x + offset, [_finite_float(row[key]) for row in selected], width=width, label=label)
        ax.set_xticks(x)
        ax.set_xticklabels([RUN_LABELS.get(row["run_name"], row["run_name"]) for row in selected])
        ax.set_ylabel("mean curvature term")
        ax.set_title(f"Output decomposition ({coordinate}, eps={eps:g})")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(frameon=False)
        fig.savefig(plots_dir / f"output_decomposition_{coordinate}.png", dpi=200)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.8, 4.5), constrained_layout=True)
    for row in rows:
        if float(row["epsilon"]) != eps:
            continue
        color = RUN_COLORS.get(row["run_name"], "#666666")
        marker = "o" if row["coordinate"] == "physical_logg_exponential" else "s"
        ax.scatter(_finite_float(row["s_write_mean"]), _finite_float(row["q_gn_mean"]), color=color, marker=marker, s=70)
        ax.annotate(
            f"{RUN_LABELS.get(row['run_name'], row['run_name'])}/{row['coordinate'].split('_')[0]}",
            (_finite_float(row["s_write_mean"]), _finite_float(row["q_gn_mean"])),
            fontsize=8,
            xytext=(4, 3),
            textcoords="offset points",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("S_write")
    ax.set_ylabel("q_GN from finite output difference")
    ax.grid(True, alpha=0.3)
    fig.savefig(plots_dir / "q_gn_vs_swrite.png", dpi=200)
    plt.close(fig)


def _plot_noise(output_root: Path, rows: list[dict]) -> None:
    if not rows:
        return
    plots_dir = output_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    sigmas = sorted({float(row["sigma"]) for row in rows})
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    for run_name in RUN_ORDER:
        selected = sorted([row for row in rows if row["run_name"] == run_name], key=lambda row: float(row["sigma"]))
        if not selected:
            continue
        ax.plot(
            [float(row["sigma"]) for row in selected],
            [_finite_float(row["mean_noisy_accuracy"]) for row in selected],
            marker="o",
            label=RUN_LABELS.get(run_name, run_name),
            color=RUN_COLORS.get(run_name),
        )
    ax.set_xlabel("physical log-G noise sigma")
    ax.set_ylabel("test accuracy")
    ax.set_title("Physical-coordinate noisy inference")
    ax.set_xticks(sigmas[:: max(1, len(sigmas) // 8)])
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.savefig(plots_dir / "physical_logg_noise_accuracy_vs_sigma.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    for run_name in RUN_ORDER:
        selected = sorted([row for row in rows if row["run_name"] == run_name], key=lambda row: float(row["sigma"]))
        if not selected:
            continue
        ax.plot(
            [float(row["sigma"]) for row in selected],
            [_finite_float(row["mean_loss_increase"]) for row in selected],
            marker="o",
            label=f"{RUN_LABELS.get(run_name, run_name)} actual",
            color=RUN_COLORS.get(run_name),
        )
        ax.plot(
            [float(row["sigma"]) for row in selected],
            [_finite_float(row["pred_loss_increase_swrite"]) for row in selected],
            linestyle="--",
            color=RUN_COLORS.get(run_name),
            alpha=0.65,
        )
    ax.set_xlabel("physical log-G noise sigma")
    ax.set_ylabel("test loss increase")
    ax.set_title("Actual loss increase vs 0.5 sigma^2 S_write")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(plots_dir / "physical_logg_noise_loss_increase_vs_swrite_prediction.png", dpi=200)
    plt.close(fig)


def write_summaries_and_plots(output_root: Path, *, swrite_root: Path) -> None:
    swrite = _swrite_by_model(swrite_root)
    decomp_rows = _summarize_decomposition(output_root, swrite)
    noise_rows = _summarize_noise(output_root, swrite)
    _plot_decomposition(output_root, decomp_rows)
    _plot_noise(output_root, noise_rows)


def _run_single(run: RunRow, *, args: argparse.Namespace, device: torch.device, decomp_path: Path, noise_path: Path, swrite: dict[tuple[str, int], float]) -> None:
    context = _build_eval_context(
        run,
        device=device,
        eval_batch_size=args.eval_batch_size,
        no_download=args.no_download,
        inference_iterations_override=args.inference_iterations,
    )
    s_write_mean = swrite.get((run.run_name, run.seed), math.nan)
    if not args.no_decomposition:
        _reset_params(context)
        _run_decomposition(run, context=context, args=args, raw_path=decomp_path, s_write_mean=s_write_mean)
    if not args.no_physical_logg_noise:
        _reset_params(context)
        _run_physical_logg_noise(run, context=context, args=args, raw_path=noise_path, s_write_mean=s_write_mean)
    _reset_params(context)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default=str(DEFAULT_INPUT_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--swrite-root", default=str(DEFAULT_SWRITE_ROOT))
    parser.add_argument("--noise-root", default=str(DEFAULT_NOISE_ROOT))
    parser.add_argument("--device", default=None)
    parser.add_argument("--checkpoint-kind", choices=("best", "final"), default="best")
    parser.add_argument("--run-name", action="append")
    parser.add_argument("--training-seeds", type=int, nargs="+")
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--inference-iterations", type=int, default=16)
    parser.add_argument("--max-eval-batches", type=int)
    parser.add_argument("--num-directions", type=int, default=16)
    parser.add_argument("--direction-seed-offset", type=int, default=12345)
    parser.add_argument("--epsilons", type=float, nargs="+", default=[0.02])
    parser.add_argument(
        "--decomposition-coordinates",
        nargs="+",
        choices=("software_theta_linear", "physical_logg_exponential"),
        default=["software_theta_linear", "physical_logg_exponential"],
    )
    parser.add_argument("--sigmas", type=float, nargs="+", default=DEFAULT_SIGMAS)
    parser.add_argument("--noise-seeds", type=int, nargs="+", default=list(range(30)))
    parser.add_argument("--clamp-perturbed", action="store_true")
    parser.add_argument("--clamp-noisy", action="store_true")
    parser.add_argument("--no-decomposition", action="store_true")
    parser.add_argument("--no-physical-logg-noise", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--raw-decomposition-name", default=None)
    parser.add_argument("--raw-noise-name", default=None)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    swrite_root = Path(args.swrite_root).expanduser().resolve()
    noise_root = Path(args.noise_root).expanduser().resolve() if args.noise_root else None
    output_root.mkdir(parents=True, exist_ok=True)

    if args.summary_only:
        write_summaries_and_plots(output_root, swrite_root=swrite_root)
        print(f"[summary] wrote decomposition/noise summaries under {output_root}")
        return

    if args.device is not None:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"Requested CUDA device {args.device!r}, but CUDA is unavailable.")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = _read_summary_rows(
        input_root,
        checkpoint_kind=args.checkpoint_kind,
        run_names=set(args.run_name) if args.run_name else None,
        training_seeds=set(args.training_seeds) if args.training_seeds else None,
    )
    selected = _select_shard(rows, args.num_shards, args.shard_index)
    decomp_name = args.raw_decomposition_name or f"raw_decomposition_shard_{args.shard_index}.csv"
    noise_name = args.raw_noise_name or f"raw_physical_logg_noise_shard_{args.shard_index}.csv"
    decomp_path = output_root / decomp_name
    noise_path = output_root / noise_name
    if decomp_path.exists() and not args.dry_run:
        decomp_path.unlink()
    if noise_path.exists() and not args.dry_run:
        noise_path.unlink()

    config_payload = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "swrite_root": str(swrite_root),
        "noise_root": str(noise_root) if noise_root is not None else None,
        "checkpoint_kind": args.checkpoint_kind,
        "device": str(device),
        "eval_batch_size": args.eval_batch_size,
        "inference_iterations": args.inference_iterations,
        "max_eval_batches": args.max_eval_batches,
        "num_directions": args.num_directions,
        "direction_seed_offset": args.direction_seed_offset,
        "epsilons": [float(value) for value in args.epsilons],
        "decomposition_coordinates": args.decomposition_coordinates,
        "sigmas": [float(value) for value in args.sigmas],
        "noise_seeds": [int(value) for value in args.noise_seeds],
        "clamp_perturbed": bool(args.clamp_perturbed),
        "clamp_noisy": bool(args.clamp_noisy),
        "no_decomposition": bool(args.no_decomposition),
        "no_physical_logg_noise": bool(args.no_physical_logg_noise),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "selected_runs": [f"{row.run_name}/seed_{row.seed}" for row in selected],
    }
    _write_json(output_root / f"config_shard_{args.shard_index}.json", config_payload)

    print(f"[output-decomp] input_root={input_root}")
    print(f"[output-decomp] output_root={output_root}")
    print(f"[output-decomp] selected_runs={len(selected)}/{len(rows)} shard={args.shard_index}/{args.num_shards}")
    if args.dry_run:
        for row in selected:
            print(f"[dry-run] {row.run_name} seed={row.seed} checkpoint={row.checkpoint_path}")
        return

    swrite = _swrite_by_model(swrite_root)
    for run in selected:
        print(f"[run] {run.run_name} seed={run.seed} checkpoint={run.checkpoint_path}")
        _run_single(run, args=args, device=device, decomp_path=decomp_path, noise_path=noise_path, swrite=swrite)

    write_summaries_and_plots(output_root, swrite_root=swrite_root)
    print(f"[done] output={output_root}")


if __name__ == "__main__":
    main()
