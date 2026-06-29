#!/usr/bin/env python3
"""Evaluate hard-sigmoid conv DRN saturation and EP/BP cosine versus input gain."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from evaluate_mnist_bp_conv_ep_bp_cosine_vs_iterations import (
    RUN_LABELS,
    RUN_ORDER,
    _collect_batches,
    _compute_bp_gradient,
    _compute_ep_gradient,
    _float,
    _make_augmented_minimizer,
    _param_group,
    _param_name,
    _stats,
    _vector_stats,
)
from evaluate_mnist_bp_write_noise_sweep import (
    RunRow,
    _build_eval_context,
    _json_sanitize,
    _read_summary_rows,
    _select_shard,
    _write_json,
)


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    return plt
from labs.mnist_train import _build_tracking_minimizer, load_config


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONV1_ROOT = REPO_ROOT / "results" / "mnist_bp_conv1_amplification_sweep_64ch_s2_valid"
DEFAULT_CONV2_ROOT = (
    REPO_ROOT
    / "results"
    / "mnist_bp_conv2_epbpK_large_50epoch_64_128ch_s2_valid_seed0_local_distributed"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "mnist_bp_conv_input_gain_saturation_cosine_hardsigmoid"


RAW_COSINE_COLUMNS = [
    "model_label",
    "conv_depth",
    "split",
    "non_linearity",
    "run_name",
    "seed",
    "voltage_amp",
    "current_amp",
    "checkpoint_kind",
    "checkpoint_path",
    "trained_input_gain",
    "eval_input_gain",
    "trained_v_min",
    "trained_v_max",
    "trained_v_off",
    "eval_v_min",
    "eval_v_max",
    "eval_v_off",
    "iteration_count",
    "beta",
    "batch_index",
    "scope",
    "param_index",
    "param_name",
    "param_class",
    "param_group",
    "num_values",
    "loss",
    "accuracy",
    "bp_grad_l2",
    "ep_grad_l2",
    "grad_dot",
    "cosine",
    "relative_l2_error",
]

RAW_SATURATION_COLUMNS = [
    "model_label",
    "conv_depth",
    "split",
    "non_linearity",
    "run_name",
    "seed",
    "voltage_amp",
    "current_amp",
    "checkpoint_kind",
    "checkpoint_path",
    "trained_input_gain",
    "eval_input_gain",
    "trained_v_min",
    "trained_v_max",
    "trained_v_off",
    "eval_v_min",
    "eval_v_max",
    "eval_v_off",
    "iteration_count",
    "batch_index",
    "layer_index",
    "layer_name",
    "layer_role",
    "v_min",
    "v_max",
    "num_values",
    "loss",
    "accuracy",
    "below_fraction",
    "inside_fraction",
    "above_fraction",
    "outside_fraction",
    "state_mean",
    "state_std",
    "state_min",
    "state_p10",
    "state_p50",
    "state_p90",
    "state_max",
]

SUMMARY_COSINE_COLUMNS = [
    "model_label",
    "conv_depth",
    "split",
    "non_linearity",
    "run_name",
    "seed",
    "voltage_amp",
    "current_amp",
    "checkpoint_kind",
    "trained_input_gain",
    "eval_input_gain",
    "trained_v_min",
    "trained_v_max",
    "trained_v_off",
    "eval_v_min",
    "eval_v_max",
    "eval_v_off",
    "iteration_count",
    "beta",
    "scope",
    "param_name",
    "param_class",
    "param_group",
    "num_batches",
    "num_values",
    "loss_mean",
    "accuracy_mean",
    "cosine_mean",
    "cosine_std",
    "cosine_p10",
    "cosine_p50",
    "cosine_p90",
    "bp_grad_l2_mean",
    "ep_grad_l2_mean",
    "relative_l2_error_mean",
]

SUMMARY_SATURATION_COLUMNS = [
    "model_label",
    "conv_depth",
    "split",
    "non_linearity",
    "run_name",
    "seed",
    "voltage_amp",
    "current_amp",
    "checkpoint_kind",
    "trained_input_gain",
    "eval_input_gain",
    "trained_v_min",
    "trained_v_max",
    "trained_v_off",
    "eval_v_min",
    "eval_v_max",
    "eval_v_off",
    "iteration_count",
    "layer_index",
    "layer_name",
    "layer_role",
    "v_min",
    "v_max",
    "num_batches",
    "num_values",
    "loss_mean",
    "accuracy_mean",
    "below_fraction_mean",
    "inside_fraction_mean",
    "above_fraction_mean",
    "outside_fraction_mean",
    "state_mean_mean",
    "state_std_mean",
    "state_min_mean",
    "state_p10_mean",
    "state_p50_mean",
    "state_p90_mean",
    "state_max_mean",
]

COMBINED_SUMMARY_COLUMNS = [
    "model_label",
    "conv_depth",
    "split",
    "non_linearity",
    "run_name",
    "seed",
    "voltage_amp",
    "current_amp",
    "checkpoint_kind",
    "trained_input_gain",
    "eval_input_gain",
    "trained_v_min",
    "trained_v_max",
    "trained_v_off",
    "eval_v_min",
    "eval_v_max",
    "eval_v_off",
    "iteration_count",
    "beta",
    "num_batches",
    "loss_mean",
    "accuracy_mean",
    "layer0_below_fraction_mean",
    "layer0_inside_fraction_mean",
    "layer0_above_fraction_mean",
    "layer0_outside_fraction_mean",
    "layer0_state_p10_mean",
    "layer0_state_p50_mean",
    "layer0_state_p90_mean",
    "cosine_all_params_mean",
    "cosine_weights_mean",
    "cosine_biases_mean",
    "cosine_convweight0_mean",
    "bp_grad_l2_all_params_mean",
    "ep_grad_l2_all_params_mean",
    "relative_l2_error_all_params_mean",
    "bp_grad_l2_convweight0_mean",
    "ep_grad_l2_convweight0_mean",
    "relative_l2_error_convweight0_mean",
]

SELECTED_COLUMNS = [
    "model_label",
    "conv_depth",
    "split",
    "non_linearity",
    "run_name",
    "seed",
    "voltage_amp",
    "current_amp",
    "checkpoint_kind",
    "primary_beta",
    "selected_input_gain",
    "selected_v_off",
    "selected_iteration_count",
    "selected_accuracy_mean",
    "selected_layer0_outside_fraction_mean",
    "selected_cosine_convweight0_mean",
    "selected_cosine_all_params_mean",
    "reference_input_gain",
    "reference_v_off",
    "reference_iteration_count",
    "reference_accuracy_mean",
    "reference_layer0_outside_fraction_mean",
    "reference_cosine_convweight0_mean",
    "max_accuracy_mean",
    "accuracy_tolerance",
    "max_layer0_outside",
    "eligible",
    "selection_rule",
]


@dataclass(frozen=True)
class InputRootSpec:
    conv_depth: int
    root: Path
    label: str


@dataclass(frozen=True)
class EvalSpec:
    conv_depth: int
    root_label: str
    run: RunRow


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


def _load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def _seed_from_dir(seed_dir: Path) -> int:
    try:
        return int(seed_dir.name.split("_", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Expected seed directory name like seed_0, got {seed_dir.name!r}.") from exc


def _config_model_cfg(seed_dir: Path) -> dict:
    source_config = seed_dir / "source_config.json"
    if source_config.exists():
        config = load_config(source_config)
        model_key = config.get("lab", {}).get("model_key", "mnist_bp_conv_amp")
        return {**config["model_base"], **config["model_overrides"][model_key]}

    config_json = seed_dir / "config.json"
    if not config_json.exists():
        raise FileNotFoundError(f"Expected source_config.json or config.json under {seed_dir}.")
    config = _load_json(config_json)
    if "architecture" in config:
        return {
            "input_gain": config["architecture"].get("input_gain"),
            "voltage_amp": config.get("amplification", {}).get("voltage_amp"),
            "current_amp": config.get("amplification", {}).get("current_amp"),
            "non_linearity": config["architecture"].get("non_linearity"),
        }
    model_key = config.get("lab", {}).get("model_key", "mnist_bp_conv_amp")
    return {**config["model_base"], **config["model_overrides"][model_key]}


def _discover_directory_rows(
    input_root: Path,
    *,
    checkpoint_kind: str,
    run_names: set[str] | None,
    training_seeds: set[int] | None,
    non_linearities: set[str] | None,
) -> list[RunRow]:
    rows: list[RunRow] = []
    for seed_dir in sorted(input_root.glob("*/*/seed_*")):
        if not seed_dir.is_dir():
            continue
        non_linearity = seed_dir.parent.parent.name
        run_name = seed_dir.parent.name
        seed = _seed_from_dir(seed_dir)
        if run_names is not None and run_name not in run_names:
            continue
        if training_seeds is not None and seed not in training_seeds:
            continue
        if non_linearities is not None and non_linearity.lower() not in non_linearities:
            continue

        checkpoint_path = seed_dir / ("best_model.pt" if checkpoint_kind == "best" else "final_model.pt")
        if not checkpoint_path.exists():
            continue
        metrics_path = seed_dir / "metrics.json"
        metrics = _load_json(metrics_path) if metrics_path.exists() else {}
        model_cfg = _config_model_cfg(seed_dir)
        if checkpoint_kind == "best":
            clean_accuracy = float(metrics.get("best_test_accuracy", math.nan))
        else:
            clean_accuracy = float(metrics.get("final_test_accuracy", math.nan))
        rows.append(
            RunRow(
                non_linearity=non_linearity,
                run_name=run_name,
                seed=seed,
                voltage_amp=float(metrics.get("voltage_amp", model_cfg.get("voltage_amp"))),
                current_amp=float(metrics.get("current_amp", model_cfg.get("current_amp"))),
                checkpoint_path=checkpoint_path,
                clean_accuracy=clean_accuracy,
                run_dir=seed_dir,
            )
        )
    return rows


def _read_rows_for_root(
    input_root: Path,
    *,
    checkpoint_kind: str,
    run_names: set[str] | None,
    training_seeds: set[int] | None,
    non_linearities: set[str] | None,
) -> list[RunRow]:
    by_key: dict[tuple[str, str, int], RunRow] = {}
    try:
        for row in _read_summary_rows(
            input_root,
            checkpoint_kind=checkpoint_kind,
            run_names=run_names,
            training_seeds=training_seeds,
        ):
            if non_linearities is not None and row.non_linearity.lower() not in non_linearities:
                continue
            if row.checkpoint_path.exists():
                by_key[(row.non_linearity, row.run_name, row.seed)] = row
    except (FileNotFoundError, ValueError):
        pass

    for row in _discover_directory_rows(
        input_root,
        checkpoint_kind=checkpoint_kind,
        run_names=run_names,
        training_seeds=training_seeds,
        non_linearities=non_linearities,
    ):
        by_key[(row.non_linearity, row.run_name, row.seed)] = row

    rows = list(by_key.values())
    return sorted(
        rows,
        key=lambda row: (
            row.non_linearity,
            RUN_ORDER.index(row.run_name) if row.run_name in RUN_ORDER else 999,
            row.seed,
        ),
    )


def _input_roots(args: argparse.Namespace) -> list[InputRootSpec]:
    roots: list[InputRootSpec] = []
    if args.input_root_conv1:
        roots.append(InputRootSpec(1, Path(args.input_root_conv1).expanduser().resolve(), "conv1"))
    elif not args.conv_depth or 1 in args.conv_depth:
        roots.append(InputRootSpec(1, DEFAULT_CONV1_ROOT.resolve(), "conv1"))

    if args.input_root_conv2:
        roots.append(InputRootSpec(2, Path(args.input_root_conv2).expanduser().resolve(), "conv2"))
    elif not args.conv_depth or 2 in args.conv_depth:
        roots.append(InputRootSpec(2, DEFAULT_CONV2_ROOT.resolve(), "conv2"))

    if args.conv_depth:
        wanted = set(args.conv_depth)
        roots = [root for root in roots if root.conv_depth in wanted]
    return roots


def _scope_rows(
    *,
    base_row: dict,
    params: list,
    bp_grads: list[torch.Tensor],
    ep_grads: list[torch.Tensor],
) -> list[dict]:
    rows: list[dict] = []

    def add_scope(scope: str, name: str, indices: list[int], param_class: str, param_group: str) -> None:
        if not indices:
            return
        stats = _vector_stats([bp_grads[index] for index in indices], [ep_grads[index] for index in indices])
        row = dict(base_row)
        row.update(
            {
                "scope": scope,
                "param_index": "" if scope != "param" else indices[0],
                "param_name": name,
                "param_class": param_class,
                "param_group": param_group,
                **stats,
            }
        )
        rows.append(row)

    all_indices = list(range(len(params)))
    weight_indices = [index for index, param in enumerate(params) if _param_group(param) == "weights"]
    bias_indices = [index for index, param in enumerate(params) if _param_group(param) == "biases"]
    add_scope("aggregate", "all_params", all_indices, "all", "all_params")
    add_scope("aggregate", "weights", weight_indices, "Weight", "weights")
    add_scope("aggregate", "biases", bias_indices, "Bias", "biases")

    for index, param in enumerate(params):
        add_scope(
            "param",
            _param_name(param, index),
            [index],
            param.__class__.__name__,
            _param_group(param),
        )
    return rows


def _layer_role(index: int, num_layers: int) -> str:
    return "output" if index == num_layers - 1 else f"hidden_{index}"


def _window_from_params(params: dict) -> tuple[float, float, float]:
    v_min = float(params.get("v_min", -1.5))
    v_max = float(params.get("v_max", 1.5))
    v_off = 0.5 * (abs(v_min) + abs(v_max))
    return v_min, v_max, v_off


def _window_metadata(context: dict) -> dict[str, float]:
    trained_v_min, trained_v_max, trained_v_off = _window_from_params(
        context.get("trained_hard_sigmoid_param", {})
    )
    eval_v_min, eval_v_max, eval_v_off = _window_from_params(
        context.get("eval_hard_sigmoid_param", context["model_cfg"].get("hard_sigmoid_param", {}))
    )
    return {
        "trained_v_min": trained_v_min,
        "trained_v_max": trained_v_max,
        "trained_v_off": trained_v_off,
        "eval_v_min": eval_v_min,
        "eval_v_max": eval_v_max,
        "eval_v_off": eval_v_off,
    }


def _v_off_values(args: argparse.Namespace) -> list[float | None]:
    if args.v_offs:
        return [float(value) for value in args.v_offs]
    return [None]


def _quantiles(flat: torch.Tensor) -> tuple[float, float, float]:
    if flat.numel() == 0:
        return math.nan, math.nan, math.nan
    q = torch.quantile(flat.float(), torch.tensor([0.1, 0.5, 0.9], device=flat.device))
    return float(q[0].item()), float(q[1].item()), float(q[2].item())


def _saturation_rows(context: dict, base_row: dict, *, loss: float, accuracy: float) -> list[dict]:
    v_min, v_max, _v_off = _window_from_params(context["model_cfg"].get("hard_sigmoid_param", {}))
    free_layers = context["free_layers"]
    rows: list[dict] = []
    for index, layer in enumerate(free_layers):
        state = layer.state.detach().float()
        flat = state.reshape(-1)
        total = int(flat.numel())
        if total == 0:
            below_fraction = inside_fraction = above_fraction = outside_fraction = math.nan
            state_mean = state_std = state_min = state_p10 = state_p50 = state_p90 = state_max = math.nan
        else:
            below = flat < v_min
            above = flat > v_max
            below_fraction = float(below.float().mean().item())
            above_fraction = float(above.float().mean().item())
            outside_fraction = below_fraction + above_fraction
            inside_fraction = 1.0 - outside_fraction
            state_p10, state_p50, state_p90 = _quantiles(flat)
            state_mean = float(flat.mean().item())
            state_std = float(flat.std(unbiased=False).item())
            state_min = float(flat.min().item())
            state_max = float(flat.max().item())
        row = dict(base_row)
        row.update(
            {
                "layer_index": index,
                "layer_name": str(getattr(layer, "name", f"free_layer_{index}")),
                "layer_role": _layer_role(index, len(free_layers)),
                "v_min": v_min,
                "v_max": v_max,
                "num_values": total,
                "loss": loss,
                "accuracy": accuracy,
                "below_fraction": below_fraction,
                "inside_fraction": inside_fraction,
                "above_fraction": above_fraction,
                "outside_fraction": outside_fraction,
                "state_mean": state_mean,
                "state_std": state_std,
                "state_min": state_min,
                "state_p10": state_p10,
                "state_p50": state_p50,
                "state_p90": state_p90,
                "state_max": state_max,
            }
        )
        rows.append(row)
    return rows


def _run_single(spec: EvalSpec, *, args: argparse.Namespace, device: torch.device) -> None:
    run = spec.run
    iteration_counts = _iteration_counts_for_depth(args, spec.conv_depth)
    for eval_input_gain in args.input_gains:
        for eval_v_off in _v_off_values(args):
            hs_override = None
            if eval_v_off is not None:
                hs_override = {"v_min": -float(eval_v_off), "v_max": float(eval_v_off)}
            context = _build_eval_context(
                run,
                device=device,
                eval_batch_size=args.batch_size,
                no_download=args.no_download,
                inference_iterations_override=max(iteration_counts),
                input_gain_override=float(eval_input_gain),
                hard_sigmoid_param_override=hs_override,
                dataset_root_override=args.dataset_root,
            )
            batches = _collect_batches(context, args.split, args.max_batches)
            params = context["params"]
            trained_input_gain = float(context["trained_input_gain"])
            eval_gain = float(context["eval_input_gain"])
            window_metadata = _window_metadata(context)
            for iteration_count in iteration_counts:
                context["minimizer"] = _build_tracking_minimizer(
                    context["energy_fn"],
                    context["free_layers"],
                    context["model_cfg"],
                    context["config"]["energy_minimizer"]["mode"],
                    num_iterations=int(iteration_count),
                    voltage_amp=context["energy_fn"]._voltage_amp,
                    current_amp=context["energy_fn"]._current_amp,
                )
                augmented_fn, augmented_minimizer = _make_augmented_minimizer(context, int(iteration_count))
                cosine_buffer: list[dict] = []
                saturation_buffer: list[dict] = []
                for batch_index, (images, labels) in enumerate(batches):
                    images = images.to(device)
                    labels = labels.to(device)
                    loss, accuracy, bp_grads = _compute_bp_gradient(context, images, labels)
                    base_row = {
                        "model_label": args.model_label,
                        "conv_depth": spec.conv_depth,
                        "split": args.split,
                        "non_linearity": run.non_linearity,
                        "run_name": run.run_name,
                        "seed": run.seed,
                        "voltage_amp": run.voltage_amp,
                        "current_amp": run.current_amp,
                        "checkpoint_kind": args.checkpoint_kind,
                        "checkpoint_path": str(run.checkpoint_path),
                        "trained_input_gain": trained_input_gain,
                        "eval_input_gain": eval_gain,
                        **window_metadata,
                        "iteration_count": int(iteration_count),
                        "batch_index": batch_index,
                    }
                    saturation_buffer.extend(_saturation_rows(context, base_row, loss=loss, accuracy=accuracy))
                    for beta in args.betas:
                        ep_grads = _compute_ep_gradient(
                            context,
                            augmented_fn,
                            augmented_minimizer,
                            images,
                            labels,
                            float(beta),
                        )
                        cosine_row = dict(base_row)
                        cosine_row.update(
                            {
                                "beta": float(beta),
                                "loss": loss,
                                "accuracy": accuracy,
                            }
                        )
                        cosine_buffer.extend(
                            _scope_rows(
                                base_row=cosine_row,
                                params=params,
                                bp_grads=bp_grads,
                                ep_grads=ep_grads,
                            )
                        )
                    if len(cosine_buffer) >= 128:
                        _write_rows(args.raw_cosine_path, RAW_COSINE_COLUMNS, cosine_buffer)
                        cosine_buffer = []
                    if len(saturation_buffer) >= 128:
                        _write_rows(args.raw_saturation_path, RAW_SATURATION_COLUMNS, saturation_buffer)
                        saturation_buffer = []
                if cosine_buffer:
                    _write_rows(args.raw_cosine_path, RAW_COSINE_COLUMNS, cosine_buffer)
                if saturation_buffer:
                    _write_rows(args.raw_saturation_path, RAW_SATURATION_COLUMNS, saturation_buffer)


def _read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _summarize_cosine(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row.get("model_label", ""),
                int(float(row.get("conv_depth", 0))),
                row.get("split", ""),
                row.get("non_linearity", ""),
                row.get("run_name", ""),
                int(float(row.get("seed", 0))),
                row.get("checkpoint_kind", ""),
                float(row.get("trained_input_gain", math.nan)),
                float(row.get("eval_input_gain", math.nan)),
                float(row.get("trained_v_off", math.nan)),
                float(row.get("eval_v_off", math.nan)),
                int(float(row.get("iteration_count", 0))),
                float(row.get("beta", 0.0)),
                row.get("scope", ""),
                row.get("param_name", ""),
            )
        ].append(row)

    out: list[dict] = []
    for key, values in sorted(grouped.items()):
        (
            model_label,
            conv_depth,
            split,
            non_linearity,
            run_name,
            seed,
            checkpoint_kind,
            trained_input_gain,
            eval_input_gain,
            trained_v_off,
            eval_v_off,
            iteration_count,
            beta,
            scope,
            param_name,
        ) = key
        first = values[0]
        cosine = _stats([_float(row, "cosine") for row in values])
        out.append(
            {
                "model_label": model_label,
                "conv_depth": conv_depth,
                "split": split,
                "non_linearity": non_linearity,
                "run_name": run_name,
                "seed": seed,
                "voltage_amp": first.get("voltage_amp", ""),
                "current_amp": first.get("current_amp", ""),
                "checkpoint_kind": checkpoint_kind,
                "trained_input_gain": trained_input_gain,
                "eval_input_gain": eval_input_gain,
                "trained_v_min": first.get("trained_v_min", ""),
                "trained_v_max": first.get("trained_v_max", ""),
                "trained_v_off": trained_v_off,
                "eval_v_min": first.get("eval_v_min", ""),
                "eval_v_max": first.get("eval_v_max", ""),
                "eval_v_off": eval_v_off,
                "iteration_count": iteration_count,
                "beta": beta,
                "scope": scope,
                "param_name": param_name,
                "param_class": first.get("param_class", ""),
                "param_group": first.get("param_group", ""),
                "num_batches": len(values),
                "num_values": first.get("num_values", ""),
                "loss_mean": _stats([_float(row, "loss") for row in values])["mean"],
                "accuracy_mean": _stats([_float(row, "accuracy") for row in values])["mean"],
                "cosine_mean": cosine["mean"],
                "cosine_std": cosine["std"],
                "cosine_p10": cosine["p10"],
                "cosine_p50": cosine["p50"],
                "cosine_p90": cosine["p90"],
                "bp_grad_l2_mean": _stats([_float(row, "bp_grad_l2") for row in values])["mean"],
                "ep_grad_l2_mean": _stats([_float(row, "ep_grad_l2") for row in values])["mean"],
                "relative_l2_error_mean": _stats([_float(row, "relative_l2_error") for row in values])["mean"],
            }
        )
    return out


def _summarize_saturation(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row.get("model_label", ""),
                int(float(row.get("conv_depth", 0))),
                row.get("split", ""),
                row.get("non_linearity", ""),
                row.get("run_name", ""),
                int(float(row.get("seed", 0))),
                row.get("checkpoint_kind", ""),
                float(row.get("trained_input_gain", math.nan)),
                float(row.get("eval_input_gain", math.nan)),
                float(row.get("trained_v_off", math.nan)),
                float(row.get("eval_v_off", math.nan)),
                int(float(row.get("iteration_count", 0))),
                int(float(row.get("layer_index", 0))),
            )
        ].append(row)

    out: list[dict] = []
    for key, values in sorted(grouped.items()):
        (
            model_label,
            conv_depth,
            split,
            non_linearity,
            run_name,
            seed,
            checkpoint_kind,
            trained_input_gain,
            eval_input_gain,
            trained_v_off,
            eval_v_off,
            iteration_count,
            layer_index,
        ) = key
        first = values[0]
        out.append(
            {
                "model_label": model_label,
                "conv_depth": conv_depth,
                "split": split,
                "non_linearity": non_linearity,
                "run_name": run_name,
                "seed": seed,
                "voltage_amp": first.get("voltage_amp", ""),
                "current_amp": first.get("current_amp", ""),
                "checkpoint_kind": checkpoint_kind,
                "trained_input_gain": trained_input_gain,
                "eval_input_gain": eval_input_gain,
                "trained_v_min": first.get("trained_v_min", ""),
                "trained_v_max": first.get("trained_v_max", ""),
                "trained_v_off": trained_v_off,
                "eval_v_min": first.get("eval_v_min", ""),
                "eval_v_max": first.get("eval_v_max", ""),
                "eval_v_off": eval_v_off,
                "iteration_count": iteration_count,
                "layer_index": layer_index,
                "layer_name": first.get("layer_name", ""),
                "layer_role": first.get("layer_role", ""),
                "v_min": first.get("v_min", ""),
                "v_max": first.get("v_max", ""),
                "num_batches": len(values),
                "num_values": first.get("num_values", ""),
                "loss_mean": _stats([_float(row, "loss") for row in values])["mean"],
                "accuracy_mean": _stats([_float(row, "accuracy") for row in values])["mean"],
                "below_fraction_mean": _stats([_float(row, "below_fraction") for row in values])["mean"],
                "inside_fraction_mean": _stats([_float(row, "inside_fraction") for row in values])["mean"],
                "above_fraction_mean": _stats([_float(row, "above_fraction") for row in values])["mean"],
                "outside_fraction_mean": _stats([_float(row, "outside_fraction") for row in values])["mean"],
                "state_mean_mean": _stats([_float(row, "state_mean") for row in values])["mean"],
                "state_std_mean": _stats([_float(row, "state_std") for row in values])["mean"],
                "state_min_mean": _stats([_float(row, "state_min") for row in values])["mean"],
                "state_p10_mean": _stats([_float(row, "state_p10") for row in values])["mean"],
                "state_p50_mean": _stats([_float(row, "state_p50") for row in values])["mean"],
                "state_p90_mean": _stats([_float(row, "state_p90") for row in values])["mean"],
                "state_max_mean": _stats([_float(row, "state_max") for row in values])["mean"],
            }
        )
    return out


def _combined_summary(cosine_rows: list[dict], saturation_rows: list[dict]) -> list[dict]:
    saturation_by_key = {
        (
            row["model_label"],
            int(row["conv_depth"]),
            row["split"],
            row["non_linearity"],
            row["run_name"],
            int(row["seed"]),
            row["checkpoint_kind"],
            float(row["trained_input_gain"]),
            float(row["eval_input_gain"]),
            float(row["trained_v_off"]),
            float(row["eval_v_off"]),
            int(row["iteration_count"]),
        ): row
        for row in saturation_rows
        if int(row["layer_index"]) == 0
    }
    cosine_grouped: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for row in cosine_rows:
        key = (
            row["model_label"],
            int(row["conv_depth"]),
            row["split"],
            row["non_linearity"],
            row["run_name"],
            int(row["seed"]),
            row["checkpoint_kind"],
            float(row["trained_input_gain"]),
            float(row["eval_input_gain"]),
            float(row["trained_v_off"]),
            float(row["eval_v_off"]),
            int(row["iteration_count"]),
            float(row["beta"]),
        )
        cosine_grouped[key][row["param_name"]] = row

    out: list[dict] = []
    for key, by_param in sorted(cosine_grouped.items()):
        (
            model_label,
            conv_depth,
            split,
            non_linearity,
            run_name,
            seed,
            checkpoint_kind,
            trained_input_gain,
            eval_input_gain,
            trained_v_off,
            eval_v_off,
            iteration_count,
            beta,
        ) = key
        first = by_param.get("all_params") or next(iter(by_param.values()))
        sat = saturation_by_key.get(key[:-1], {})

        def metric(param_name: str, field: str) -> float:
            row = by_param.get(param_name, {})
            return _float(row, field)

        out.append(
            {
                "model_label": model_label,
                "conv_depth": conv_depth,
                "split": split,
                "non_linearity": non_linearity,
                "run_name": run_name,
                "seed": seed,
                "voltage_amp": first.get("voltage_amp", ""),
                "current_amp": first.get("current_amp", ""),
                "checkpoint_kind": checkpoint_kind,
                "trained_input_gain": trained_input_gain,
                "eval_input_gain": eval_input_gain,
                "trained_v_min": first.get("trained_v_min", ""),
                "trained_v_max": first.get("trained_v_max", ""),
                "trained_v_off": trained_v_off,
                "eval_v_min": first.get("eval_v_min", ""),
                "eval_v_max": first.get("eval_v_max", ""),
                "eval_v_off": eval_v_off,
                "iteration_count": iteration_count,
                "beta": beta,
                "num_batches": first.get("num_batches", ""),
                "loss_mean": first.get("loss_mean", ""),
                "accuracy_mean": first.get("accuracy_mean", ""),
                "layer0_below_fraction_mean": sat.get("below_fraction_mean", ""),
                "layer0_inside_fraction_mean": sat.get("inside_fraction_mean", ""),
                "layer0_above_fraction_mean": sat.get("above_fraction_mean", ""),
                "layer0_outside_fraction_mean": sat.get("outside_fraction_mean", ""),
                "layer0_state_p10_mean": sat.get("state_p10_mean", ""),
                "layer0_state_p50_mean": sat.get("state_p50_mean", ""),
                "layer0_state_p90_mean": sat.get("state_p90_mean", ""),
                "cosine_all_params_mean": metric("all_params", "cosine_mean"),
                "cosine_weights_mean": metric("weights", "cosine_mean"),
                "cosine_biases_mean": metric("biases", "cosine_mean"),
                "cosine_convweight0_mean": metric("ConvWeight_0", "cosine_mean"),
                "bp_grad_l2_all_params_mean": metric("all_params", "bp_grad_l2_mean"),
                "ep_grad_l2_all_params_mean": metric("all_params", "ep_grad_l2_mean"),
                "relative_l2_error_all_params_mean": metric("all_params", "relative_l2_error_mean"),
                "bp_grad_l2_convweight0_mean": metric("ConvWeight_0", "bp_grad_l2_mean"),
                "ep_grad_l2_convweight0_mean": metric("ConvWeight_0", "ep_grad_l2_mean"),
                "relative_l2_error_convweight0_mean": metric("ConvWeight_0", "relative_l2_error_mean"),
            }
        )
    return out


def _select_gain(
    combined_rows: list[dict],
    *,
    primary_beta: float,
    accuracy_tolerance: float,
    max_layer0_outside: float,
) -> list[dict]:
    primary_rows = [
        row for row in combined_rows if abs(_float(row, "beta") - primary_beta) < 1e-12
    ]
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in primary_rows:
        grouped[
            (
                row["model_label"],
                int(row["conv_depth"]),
                row["split"],
                row["non_linearity"],
                row["run_name"],
                int(row["seed"]),
                row["checkpoint_kind"],
            )
        ].append(row)

    selected_rows: list[dict] = []
    for key, values in sorted(grouped.items()):
        by_gain: dict[float, list[dict]] = defaultdict(list)
        for row in values:
            by_gain[_float(row, "eval_input_gain")].append(row)
        gain_rows = []
        for gain, rows in by_gain.items():
            finite = [row for row in rows if math.isfinite(_float(row, "cosine_all_params_mean"))]
            if not finite:
                continue
            best = max(
                finite,
                key=lambda row: (
                    _float(row, "cosine_all_params_mean"),
                    _float(row, "cosine_convweight0_mean"),
                    int(row["iteration_count"]),
                ),
            )
            gain_rows.append(best)
        if not gain_rows:
            continue
        max_accuracy = max(_float(row, "accuracy_mean") for row in gain_rows)
        reference_gain = max(_float(row, "eval_input_gain") for row in gain_rows)
        reference = next(
            (row for row in gain_rows if abs(_float(row, "eval_input_gain") - reference_gain) < 1e-12),
            gain_rows[-1],
        )
        ref_cw0 = _float(reference, "cosine_convweight0_mean")

        eligible = []
        for row in gain_rows:
            accuracy_ok = _float(row, "accuracy_mean") >= max_accuracy - accuracy_tolerance
            saturation_ok = _float(row, "layer0_outside_fraction_mean") <= max_layer0_outside
            improved_ok = _float(row, "cosine_convweight0_mean") >= ref_cw0
            row = dict(row)
            row["_eligible"] = accuracy_ok and saturation_ok and improved_ok
            eligible.append(row)
        eligible_rows = [row for row in eligible if row["_eligible"]]
        if eligible_rows:
            selected = max(eligible_rows, key=lambda row: _float(row, "eval_input_gain"))
            rule = (
                "largest input_gain with accuracy within tolerance of best, "
                "layer0 outside below threshold, and ConvWeight_0 cosine no worse than max-gain reference"
            )
        else:
            selected = max(
                eligible,
                key=lambda row: (
                    _float(row, "cosine_convweight0_mean") - _float(row, "layer0_outside_fraction_mean"),
                    _float(row, "accuracy_mean"),
                    _float(row, "eval_input_gain"),
                ),
            )
            rule = "fallback: best ConvWeight_0 cosine minus layer0 outside fraction, then accuracy"

        selected_rows.append(
            {
                "model_label": key[0],
                "conv_depth": key[1],
                "split": key[2],
                "non_linearity": key[3],
                "run_name": key[4],
                "seed": key[5],
                "voltage_amp": selected.get("voltage_amp", ""),
                "current_amp": selected.get("current_amp", ""),
                "checkpoint_kind": key[6],
                "primary_beta": primary_beta,
                "selected_input_gain": selected.get("eval_input_gain", ""),
                "selected_v_off": selected.get("eval_v_off", ""),
                "selected_iteration_count": selected.get("iteration_count", ""),
                "selected_accuracy_mean": selected.get("accuracy_mean", ""),
                "selected_layer0_outside_fraction_mean": selected.get("layer0_outside_fraction_mean", ""),
                "selected_cosine_convweight0_mean": selected.get("cosine_convweight0_mean", ""),
                "selected_cosine_all_params_mean": selected.get("cosine_all_params_mean", ""),
                "reference_input_gain": reference.get("eval_input_gain", ""),
                "reference_v_off": reference.get("eval_v_off", ""),
                "reference_iteration_count": reference.get("iteration_count", ""),
                "reference_accuracy_mean": reference.get("accuracy_mean", ""),
                "reference_layer0_outside_fraction_mean": reference.get("layer0_outside_fraction_mean", ""),
                "reference_cosine_convweight0_mean": reference.get("cosine_convweight0_mean", ""),
                "max_accuracy_mean": max_accuracy,
                "accuracy_tolerance": accuracy_tolerance,
                "max_layer0_outside": max_layer0_outside,
                "eligible": bool(selected.get("_eligible", False)),
                "selection_rule": rule,
            }
        )
    return selected_rows


def _select_voff(
    combined_rows: list[dict],
    *,
    primary_beta: float,
    accuracy_tolerance: float,
    max_layer0_outside: float,
) -> list[dict]:
    primary_rows = [
        row for row in combined_rows if abs(_float(row, "beta") - primary_beta) < 1e-12
    ]
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in primary_rows:
        grouped[
            (
                row["model_label"],
                int(row["conv_depth"]),
                row["split"],
                row["non_linearity"],
                row["run_name"],
                int(row["seed"]),
                row["checkpoint_kind"],
                float(row["eval_input_gain"]),
            )
        ].append(row)

    selected_rows: list[dict] = []
    for key, values in sorted(grouped.items()):
        by_voff: dict[float, list[dict]] = defaultdict(list)
        for row in values:
            by_voff[_float(row, "eval_v_off")].append(row)
        voff_rows = []
        for _voff, rows in by_voff.items():
            finite = [row for row in rows if math.isfinite(_float(row, "cosine_all_params_mean"))]
            if not finite:
                continue
            voff_rows.append(
                max(
                    finite,
                    key=lambda row: (
                        _float(row, "cosine_all_params_mean"),
                        _float(row, "cosine_convweight0_mean"),
                        int(row["iteration_count"]),
                    ),
                )
            )
        if not voff_rows:
            continue

        max_accuracy = max(_float(row, "accuracy_mean") for row in voff_rows)
        reference_voff = min(_float(row, "eval_v_off") for row in voff_rows)
        reference = next(
            (row for row in voff_rows if abs(_float(row, "eval_v_off") - reference_voff) < 1e-12),
            voff_rows[0],
        )
        ref_cw0 = _float(reference, "cosine_convweight0_mean")

        eligible = []
        for row in voff_rows:
            accuracy_ok = _float(row, "accuracy_mean") >= max_accuracy - accuracy_tolerance
            saturation_ok = _float(row, "layer0_outside_fraction_mean") <= max_layer0_outside
            improved_ok = _float(row, "cosine_convweight0_mean") >= ref_cw0
            row = dict(row)
            row["_eligible"] = accuracy_ok and saturation_ok and improved_ok
            eligible.append(row)
        eligible_rows = [row for row in eligible if row["_eligible"]]
        if eligible_rows:
            selected = min(eligible_rows, key=lambda row: _float(row, "eval_v_off"))
            rule = (
                "smallest v_off with accuracy within tolerance of best, "
                "layer0 outside below threshold, and ConvWeight_0 cosine no worse than baseline v_off"
            )
        else:
            selected = max(
                eligible,
                key=lambda row: (
                    _float(row, "cosine_convweight0_mean") - _float(row, "layer0_outside_fraction_mean"),
                    _float(row, "accuracy_mean"),
                    -_float(row, "eval_v_off"),
                ),
            )
            rule = "fallback: best ConvWeight_0 cosine minus layer0 outside fraction, then accuracy"

        selected_rows.append(
            {
                "model_label": key[0],
                "conv_depth": key[1],
                "split": key[2],
                "non_linearity": key[3],
                "run_name": key[4],
                "seed": key[5],
                "voltage_amp": selected.get("voltage_amp", ""),
                "current_amp": selected.get("current_amp", ""),
                "checkpoint_kind": key[6],
                "primary_beta": primary_beta,
                "selected_input_gain": selected.get("eval_input_gain", ""),
                "selected_v_off": selected.get("eval_v_off", ""),
                "selected_iteration_count": selected.get("iteration_count", ""),
                "selected_accuracy_mean": selected.get("accuracy_mean", ""),
                "selected_layer0_outside_fraction_mean": selected.get("layer0_outside_fraction_mean", ""),
                "selected_cosine_convweight0_mean": selected.get("cosine_convweight0_mean", ""),
                "selected_cosine_all_params_mean": selected.get("cosine_all_params_mean", ""),
                "reference_input_gain": reference.get("eval_input_gain", ""),
                "reference_v_off": reference.get("eval_v_off", ""),
                "reference_iteration_count": reference.get("iteration_count", ""),
                "reference_accuracy_mean": reference.get("accuracy_mean", ""),
                "reference_layer0_outside_fraction_mean": reference.get("layer0_outside_fraction_mean", ""),
                "reference_cosine_convweight0_mean": reference.get("cosine_convweight0_mean", ""),
                "max_accuracy_mean": max_accuracy,
                "accuracy_tolerance": accuracy_tolerance,
                "max_layer0_outside": max_layer0_outside,
                "eligible": bool(selected.get("_eligible", False)),
                "selection_rule": rule,
            }
        )
    return selected_rows


def _terminal_primary_rows(rows: list[dict], *, conv_depth: int, primary_beta: float) -> list[dict]:
    candidates = [
        row
        for row in rows
        if int(row["conv_depth"]) == conv_depth and abs(_float(row, "beta") - primary_beta) < 1e-12
    ]
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in candidates:
        grouped[(row["run_name"], _float(row, "eval_input_gain"), _float(row, "eval_v_off"))].append(row)
    terminal = []
    for values in grouped.values():
        terminal.append(max(values, key=lambda row: int(row["iteration_count"])))
    return terminal


def _plot_metric_by_depth(
    output_root: Path,
    rows: list[dict],
    *,
    conv_depth: int,
    primary_beta: float,
    metric: str,
    ylabel: str,
    filename: str,
) -> None:
    plot_rows = _terminal_primary_rows(rows, conv_depth=conv_depth, primary_beta=primary_beta)
    if not plot_rows:
        return
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    run_names = sorted(
        {row["run_name"] for row in plot_rows},
        key=lambda name: RUN_ORDER.index(name) if name in RUN_ORDER else 999,
    )
    for run_name in run_names:
        series = sorted(
            [row for row in plot_rows if row["run_name"] == run_name],
            key=lambda row: _float(row, "eval_input_gain"),
        )
        ax.plot(
            [_float(row, "eval_input_gain") for row in series],
            [_float(row, metric) for row in series],
            marker="o",
            linewidth=1.8,
            label=RUN_LABELS.get(run_name, run_name),
        )
    ax.set_xlabel("eval input_gain")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Conv{conv_depth} hard-sigmoid")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(output_root / filename, dpi=200)
    plt.close(fig)


def _plot_accuracy(output_root: Path, rows: list[dict], *, primary_beta: float) -> None:
    depths = sorted({int(row["conv_depth"]) for row in rows})
    if not depths:
        return
    plt = _pyplot()
    fig, axes = plt.subplots(1, len(depths), figsize=(6.2 * len(depths), 4.2), squeeze=False, constrained_layout=True)
    for ax, depth in zip(axes[0], depths):
        plot_rows = _terminal_primary_rows(rows, conv_depth=depth, primary_beta=primary_beta)
        run_names = sorted(
            {row["run_name"] for row in plot_rows},
            key=lambda name: RUN_ORDER.index(name) if name in RUN_ORDER else 999,
        )
        for run_name in run_names:
            series = sorted(
                [row for row in plot_rows if row["run_name"] == run_name],
                key=lambda row: _float(row, "eval_input_gain"),
            )
            ax.plot(
                [_float(row, "eval_input_gain") for row in series],
                [_float(row, "accuracy_mean") for row in series],
                marker="o",
                linewidth=1.8,
                label=RUN_LABELS.get(run_name, run_name),
            )
        ax.set_title(f"Conv{depth}")
        ax.set_xlabel("eval input_gain")
        ax.set_ylabel("accuracy")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False, fontsize=8)
    fig.savefig(output_root / "accuracy_vs_input_gain_by_amp.png", dpi=200)
    plt.close(fig)


def _plot_metric_by_depth_voff(
    output_root: Path,
    rows: list[dict],
    *,
    conv_depth: int,
    primary_beta: float,
    metric: str,
    ylabel: str,
    filename: str,
) -> None:
    plot_rows = _terminal_primary_rows(rows, conv_depth=conv_depth, primary_beta=primary_beta)
    if not plot_rows:
        return
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    run_names = sorted(
        {row["run_name"] for row in plot_rows},
        key=lambda name: RUN_ORDER.index(name) if name in RUN_ORDER else 999,
    )
    for run_name in run_names:
        series = sorted(
            [row for row in plot_rows if row["run_name"] == run_name],
            key=lambda row: _float(row, "eval_v_off"),
        )
        ax.plot(
            [_float(row, "eval_v_off") for row in series],
            [_float(row, metric) for row in series],
            marker="o",
            linewidth=1.8,
            label=RUN_LABELS.get(run_name, run_name),
        )
    ax.set_xlabel("eval v_off")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Conv{conv_depth} hard-sigmoid")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(output_root / filename, dpi=200)
    plt.close(fig)


def _plot_accuracy_voff(output_root: Path, rows: list[dict], *, primary_beta: float) -> None:
    depths = sorted({int(row["conv_depth"]) for row in rows})
    if not depths:
        return
    plt = _pyplot()
    fig, axes = plt.subplots(1, len(depths), figsize=(6.2 * len(depths), 4.2), squeeze=False, constrained_layout=True)
    for ax, depth in zip(axes[0], depths):
        plot_rows = _terminal_primary_rows(rows, conv_depth=depth, primary_beta=primary_beta)
        run_names = sorted(
            {row["run_name"] for row in plot_rows},
            key=lambda name: RUN_ORDER.index(name) if name in RUN_ORDER else 999,
        )
        for run_name in run_names:
            series = sorted(
                [row for row in plot_rows if row["run_name"] == run_name],
                key=lambda row: _float(row, "eval_v_off"),
            )
            ax.plot(
                [_float(row, "eval_v_off") for row in series],
                [_float(row, "accuracy_mean") for row in series],
                marker="o",
                linewidth=1.8,
                label=RUN_LABELS.get(run_name, run_name),
            )
        ax.set_title(f"Conv{depth}")
        ax.set_xlabel("eval v_off")
        ax.set_ylabel("accuracy")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False, fontsize=8)
    fig.savefig(output_root / "accuracy_vs_voff_by_amp.png", dpi=200)
    plt.close(fig)


def write_summaries_and_plots(
    output_root: Path,
    *,
    primary_beta: float,
    accuracy_tolerance: float,
    max_layer0_outside: float,
) -> None:
    raw_cosine: list[dict] = []
    for path in sorted(output_root.glob("raw_cosine*.csv")):
        raw_cosine.extend(_read_rows(path))
    raw_saturation: list[dict] = []
    for path in sorted(output_root.glob("raw_saturation*.csv")):
        raw_saturation.extend(_read_rows(path))
    if not raw_cosine or not raw_saturation:
        return

    cosine_summary = _summarize_cosine(raw_cosine)
    saturation_summary = _summarize_saturation(raw_saturation)
    combined = _combined_summary(cosine_summary, saturation_summary)
    selected = _select_gain(
        combined,
        primary_beta=primary_beta,
        accuracy_tolerance=accuracy_tolerance,
        max_layer0_outside=max_layer0_outside,
    )
    selected_voff = _select_voff(
        combined,
        primary_beta=primary_beta,
        accuracy_tolerance=accuracy_tolerance,
        max_layer0_outside=max_layer0_outside,
    )

    _write_rows(output_root / "summary_cosine_by_depth_amp_gain_k_beta.csv", SUMMARY_COSINE_COLUMNS, cosine_summary, append=False)
    _write_rows(output_root / "summary_saturation_by_depth_amp_gain_k.csv", SUMMARY_SATURATION_COLUMNS, saturation_summary, append=False)
    _write_rows(output_root / "summary_by_depth_amp_gain_k_beta.csv", COMBINED_SUMMARY_COLUMNS, combined, append=False)
    _write_rows(output_root / "summary_by_depth_amp_gain_voff_k_beta.csv", COMBINED_SUMMARY_COLUMNS, combined, append=False)
    _write_rows(output_root / "selected_gain_by_depth_amp.csv", SELECTED_COLUMNS, selected, append=False)
    _write_rows(output_root / "selected_voff_by_depth_amp.csv", SELECTED_COLUMNS, selected_voff, append=False)

    _plot_metric_by_depth(
        output_root,
        combined,
        conv_depth=1,
        primary_beta=primary_beta,
        metric="layer0_outside_fraction_mean",
        ylabel="layer 0 outside-window fraction",
        filename="layer0_saturation_vs_input_gain_conv1.png",
    )
    _plot_metric_by_depth(
        output_root,
        combined,
        conv_depth=2,
        primary_beta=primary_beta,
        metric="layer0_outside_fraction_mean",
        ylabel="layer 0 outside-window fraction",
        filename="layer0_saturation_vs_input_gain_conv2.png",
    )
    _plot_metric_by_depth(
        output_root,
        combined,
        conv_depth=1,
        primary_beta=primary_beta,
        metric="cosine_convweight0_mean",
        ylabel="ConvWeight_0 EP/BP cosine",
        filename="convweight0_cosine_vs_input_gain_conv1.png",
    )
    _plot_metric_by_depth(
        output_root,
        combined,
        conv_depth=2,
        primary_beta=primary_beta,
        metric="cosine_convweight0_mean",
        ylabel="ConvWeight_0 EP/BP cosine",
        filename="convweight0_cosine_vs_input_gain_conv2.png",
    )
    _plot_accuracy(output_root, combined, primary_beta=primary_beta)
    _plot_metric_by_depth_voff(
        output_root,
        combined,
        conv_depth=1,
        primary_beta=primary_beta,
        metric="layer0_outside_fraction_mean",
        ylabel="layer 0 outside-window fraction",
        filename="layer0_saturation_vs_voff_conv1.png",
    )
    _plot_metric_by_depth_voff(
        output_root,
        combined,
        conv_depth=2,
        primary_beta=primary_beta,
        metric="layer0_outside_fraction_mean",
        ylabel="layer 0 outside-window fraction",
        filename="layer0_saturation_vs_voff_conv2.png",
    )
    _plot_metric_by_depth_voff(
        output_root,
        combined,
        conv_depth=1,
        primary_beta=primary_beta,
        metric="cosine_convweight0_mean",
        ylabel="ConvWeight_0 EP/BP cosine",
        filename="convweight0_cosine_vs_voff_conv1.png",
    )
    _plot_metric_by_depth_voff(
        output_root,
        combined,
        conv_depth=2,
        primary_beta=primary_beta,
        metric="cosine_convweight0_mean",
        ylabel="ConvWeight_0 EP/BP cosine",
        filename="convweight0_cosine_vs_voff_conv2.png",
    )
    _plot_accuracy_voff(output_root, combined, primary_beta=primary_beta)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root-conv1", default=None)
    parser.add_argument("--input-root-conv2", default=None)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--model-label", default="conv_input_gain_saturation_cosine")
    parser.add_argument("--conv-depth", type=int, nargs="+", choices=(1, 2), default=None)
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--device", default=None)
    parser.add_argument("--checkpoint-kind", choices=("best", "final"), default="best")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--run-name", action="append", choices=RUN_ORDER)
    parser.add_argument("--non-linearity", nargs="+", default=["hard_sigmoid"])
    parser.add_argument("--training-seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--input-gains", type=float, nargs="+", default=[25.0, 50.0, 75.0, 100.0])
    parser.add_argument("--v-offs", type=float, nargs="+", default=[])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=32)
    parser.add_argument("--iteration-counts", type=int, nargs="+", default=None)
    parser.add_argument("--iteration-counts-conv1", type=int, nargs="+", default=[4, 6, 8, 12, 16])
    parser.add_argument("--iteration-counts-conv2", type=int, nargs="+", default=[4, 6, 8, 12, 16, 24, 32])
    parser.add_argument("--betas", type=float, nargs="+", default=[0.1, 0.25, 0.5, 1.0])
    parser.add_argument("--primary-beta", type=float, default=0.25)
    parser.add_argument("--selection-accuracy-tolerance", type=float, default=0.02)
    parser.add_argument("--selection-max-layer0-outside", type=float, default=0.60)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--raw-cosine-name", default=None)
    parser.add_argument("--raw-saturation-name", default=None)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _iteration_counts_for_depth(args: argparse.Namespace, conv_depth: int) -> list[int]:
    if args.iteration_counts is not None:
        return [int(value) for value in args.iteration_counts]
    if conv_depth == 1:
        return [int(value) for value in args.iteration_counts_conv1]
    if conv_depth == 2:
        return [int(value) for value in args.iteration_counts_conv2]
    raise ValueError(f"Unsupported conv depth {conv_depth}.")


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    raw_cosine_name = args.raw_cosine_name
    raw_saturation_name = args.raw_saturation_name
    if raw_cosine_name is None:
        raw_cosine_name = "raw_cosine.csv" if args.num_shards == 1 else f"raw_cosine_shard_{args.shard_index}.csv"
    if raw_saturation_name is None:
        raw_saturation_name = (
            "raw_saturation.csv"
            if args.num_shards == 1
            else f"raw_saturation_shard_{args.shard_index}.csv"
        )
    args.raw_cosine_path = output_root / raw_cosine_name
    args.raw_saturation_path = output_root / raw_saturation_name

    if args.summary_only:
        write_summaries_and_plots(
            output_root,
            primary_beta=float(args.primary_beta),
            accuracy_tolerance=float(args.selection_accuracy_tolerance),
            max_layer0_outside=float(args.selection_max_layer0_outside),
        )
        print(f"[summary] wrote input-gain saturation/cosine summaries under {output_root}")
        return

    if args.device is not None:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"Requested CUDA device {args.device!r}, but CUDA is unavailable.")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nonlinearities = {value.lower() for value in args.non_linearity}
    roots = [root for root in _input_roots(args) if root.root.exists()]
    missing_roots = [root for root in _input_roots(args) if not root.root.exists()]
    for root in missing_roots:
        print(f"[warn] skipping missing Conv{root.conv_depth} root: {root.root}")

    specs: list[EvalSpec] = []
    for root in roots:
        rows = _read_rows_for_root(
            root.root,
            checkpoint_kind=args.checkpoint_kind,
            run_names=set(args.run_name) if args.run_name else None,
            training_seeds=set(args.training_seeds) if args.training_seeds else None,
            non_linearities=nonlinearities,
        )
        specs.extend(EvalSpec(root.conv_depth, root.label, row) for row in rows)
    if not specs:
        raise ValueError("No matching conv hard-sigmoid checkpoints were found.")

    selected = _select_shard(specs, args.num_shards, args.shard_index)
    _write_json(
        output_root / f"config_shard_{args.shard_index}.json",
        {
            "output_root": str(output_root),
            "model_label": args.model_label,
            "device": str(device),
            "split": args.split,
            "checkpoint_kind": args.checkpoint_kind,
            "dataset_root": args.dataset_root,
            "conv_depth": args.conv_depth,
            "non_linearity": args.non_linearity,
            "training_seeds": args.training_seeds,
            "input_gains": [float(value) for value in args.input_gains],
            "v_offs": [float(value) for value in args.v_offs],
            "iteration_counts": (
                [int(value) for value in args.iteration_counts]
                if args.iteration_counts is not None
                else None
            ),
            "iteration_counts_conv1": [int(value) for value in args.iteration_counts_conv1],
            "iteration_counts_conv2": [int(value) for value in args.iteration_counts_conv2],
            "betas": [float(value) for value in args.betas],
            "primary_beta": float(args.primary_beta),
            "batch_size": int(args.batch_size),
            "max_batches": args.max_batches,
            "selection_accuracy_tolerance": float(args.selection_accuracy_tolerance),
            "selection_max_layer0_outside": float(args.selection_max_layer0_outside),
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "raw_cosine_path": str(args.raw_cosine_path),
            "raw_saturation_path": str(args.raw_saturation_path),
            "input_roots": [
                {"conv_depth": root.conv_depth, "label": root.label, "root": str(root.root)}
                for root in roots
            ],
            "selected_runs": [
                f"conv{spec.conv_depth}/{spec.run.non_linearity}/{spec.run.run_name}/seed_{spec.run.seed}"
                for spec in selected
            ],
            "note": "input_gain is overridden only in memory during evaluation; checkpoint weights are unchanged",
        },
    )

    print(f"[input-gain-diag] output_root={output_root}")
    print(f"[input-gain-diag] selected_runs={len(selected)}/{len(specs)} shard={args.shard_index}/{args.num_shards}")
    print(
        "[input-gain-diag] "
        f"input_gains={args.input_gains} "
        f"v_offs={args.v_offs or ['trained']} "
        f"K_global={args.iteration_counts} "
        f"K_conv1={args.iteration_counts_conv1} "
        f"K_conv2={args.iteration_counts_conv2} "
        f"betas={args.betas}"
    )
    if args.dry_run:
        for spec in selected:
            print(
                "[dry-run] "
                f"conv{spec.conv_depth} {spec.run.non_linearity} {spec.run.run_name} "
                f"seed={spec.run.seed} checkpoint={spec.run.checkpoint_path}"
            )
        return

    if args.shard_index == 0:
        for path in (args.raw_cosine_path, args.raw_saturation_path):
            if path.exists():
                path.unlink()

    for spec in selected:
        print(
            "[run] "
            f"conv{spec.conv_depth} {spec.run.non_linearity} {spec.run.run_name} "
            f"seed={spec.run.seed} checkpoint={spec.run.checkpoint_path}",
            flush=True,
        )
        _run_single(spec, args=args, device=device)

    write_summaries_and_plots(
        output_root,
        primary_beta=float(args.primary_beta),
        accuracy_tolerance=float(args.selection_accuracy_tolerance),
        max_layer0_outside=float(args.selection_max_layer0_outside),
    )
    print(f"[done] output={output_root}")


if __name__ == "__main__":
    main()
