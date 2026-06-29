#!/usr/bin/env python3
"""Measure BP gradient scale alongside hidden saturation for good Conv MNIST runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
LABS_DIR = REPO_ROOT / "labs"
for path in (REPO_ROOT, LABS_DIR, Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analyze_mnist_bp_conv_good_run_saturation import (  # noqa: E402
    RunSpec,
    _best_specs_per_group,
    _build_context,
    _discover_run_dirs,
    _layer_saturation,
    _load_specs,
    _normalize_run_dir,
    _path_part,
    _target_from_label,
)
from labs.mnist_train import _build_tracking_minimizer  # noqa: E402
from training.sgd import Backprop  # noqa: E402


OUTPUT_COLUMNS = [
    "conv_depth",
    "non_linearity",
    "run_name",
    "phase",
    "voltage_amp",
    "current_amp",
    "input_gain",
    "v_off",
    "lr_reference",
    "epochs",
    "seed",
    "run_dir",
    "best_test_accuracy",
    "final_test_accuracy",
    "target_label",
    "target_saturation",
    "split",
    "num_samples",
    "hidden1_saturation",
    "hidden2_saturation",
    "all_hidden_saturation",
    "global_param_l2",
    "global_grad_l2",
    "global_relative_grad_l2",
    "global_relative_step_l2",
    "max_param_relative_grad_l2",
    "max_param_relative_step_l2",
    "weight0_grad_l2",
    "weight0_relative_grad_l2",
    "weight0_relative_step_l2",
    "weight1_grad_l2",
    "weight1_relative_grad_l2",
    "weight1_relative_step_l2",
    "weight2_grad_l2",
    "weight2_relative_grad_l2",
    "weight2_relative_step_l2",
    "bias0_grad_l2",
    "bias0_relative_grad_l2",
    "bias1_grad_l2",
    "bias1_relative_grad_l2",
    "param_summaries_json",
]


def _learning_rates(spec: RunSpec, num_params: int, phase: str) -> list[float]:
    lr = spec.source_config.get("lr", [])
    if isinstance(lr, (int, float)):
        values = [float(lr)]
    else:
        values = [float(value) for value in lr]
    if not values:
        values = [math.nan]
    if phase == "final":
        final_lr = spec.metrics.get("final_learning_rate")
        if final_lr is not None:
            if isinstance(final_lr, (int, float)):
                values = [float(final_lr)]
            else:
                values = [float(value) for value in final_lr]
    if len(values) < num_params:
        values.extend([values[-1]] * (num_params - len(values)))
    return values[:num_params]


def _mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else math.nan


def _parameter_name(param: object, index: int) -> str:
    return str(getattr(param, "name", f"param_{index}")).strip()


def _parameter_stats(param: object, grad: torch.Tensor, lr: float) -> dict:
    state = param.state.detach()
    grad = grad.detach()
    param_l2 = float(torch.linalg.vector_norm(state).item())
    grad_l2 = float(torch.linalg.vector_norm(grad).item())
    relative = grad_l2 / param_l2 if param_l2 > 0 else math.nan
    return {
        "param_l2": param_l2,
        "grad_l2": grad_l2,
        "relative_grad_l2": relative,
        "relative_step_l2": lr * relative if math.isfinite(lr) and math.isfinite(relative) else math.nan,
        "grad_abs_mean": float(torch.mean(torch.abs(grad)).item()),
        "grad_abs_max": float(torch.max(torch.abs(grad)).item()),
        "grad_zero_fraction": float((grad.abs() <= 1e-12).float().mean().item()),
    }


def _hidden_saturation(context: dict, *, perfect_eps: float) -> dict:
    model_cfg = context["model_cfg"]
    non_linearity = str(model_cfg.get("non_linearity", ""))
    hard_sigmoid_param = dict(model_cfg.get("hard_sigmoid_param", {}))
    hidden_layers = context["energy_fn"].layers()[1:-1]
    layer_values: list[float] = []
    weighted_sum = 0.0
    total = 0
    for layer in hidden_layers:
        fraction, count = _layer_saturation(
            layer.state.detach(),
            non_linearity=non_linearity,
            hard_sigmoid_param=hard_sigmoid_param,
            perfect_eps=perfect_eps,
        )
        layer_values.append(fraction)
        if math.isfinite(fraction) and count > 0:
            weighted_sum += fraction * count
            total += count
    return {
        "hidden1_saturation": layer_values[0] if len(layer_values) > 0 else math.nan,
        "hidden2_saturation": layer_values[1] if len(layer_values) > 1 else math.nan,
        "all_hidden_saturation": weighted_sum / total if total else math.nan,
        "layer_saturations": layer_values,
    }


def _empty_param_slots() -> dict:
    row = {}
    for prefix in ("weight0", "weight1", "weight2", "bias0", "bias1"):
        row[f"{prefix}_grad_l2"] = math.nan
        row[f"{prefix}_relative_grad_l2"] = math.nan
        if prefix.startswith("weight"):
            row[f"{prefix}_relative_step_l2"] = math.nan
    return row


def _measure_phase(
    spec: RunSpec,
    *,
    phase: str,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    context = _build_context(
        spec,
        device=device,
        load_final=(phase == "final"),
        batch_size=args.batch_size,
        no_download=args.no_download,
        dataset_root=args.dataset_root,
        inference_iterations_override=args.inference_iterations,
    )
    params = context["energy_fn"].params()
    base_states = [param.state.detach().clone() for param in params]
    lrs = _learning_rates(spec, len(params), phase)
    lr_reference = lrs[0] if lrs else math.nan
    training_iterations = (
        int(args.training_iterations)
        if args.training_iterations is not None
        else int(
            context["model_cfg"].get(
                "num_iterations_training",
                context["model_cfg"]["num_iterations_inference"],
            )
        )
    )
    minimizer_training = _build_tracking_minimizer(
        context["energy_fn"],
        context["network"].free_layers(),
        context["model_cfg"],
        spec.source_config["energy_minimizer"]["mode"],
        num_iterations=training_iterations,
        voltage_amp=context["energy_fn"]._voltage_amp,
        current_amp=context["energy_fn"]._current_amp,
    )
    estimator = Backprop(params, context["free_layers"], context["cost_fn"], minimizer_training)
    loader = context["train_loader"] if args.split == "train" else context["test_loader"]

    batch_summaries: list[dict] = []
    param_accumulator: dict[int, list[dict]] = {index: [] for index in range(len(params))}
    samples = 0
    with torch.no_grad():
        pass
    for images, labels in loader:
        if samples >= args.max_samples:
            break
        if samples + int(images.shape[0]) > args.max_samples:
            images = images[: args.max_samples - samples]
            labels = labels[: args.max_samples - samples]
        for param, base in zip(params, base_states):
            param.state = base.detach().clone()
        images = images.to(device)
        labels = labels.to(device)
        context["network"].set_input(images, reset=True)
        context["minimizer"].compute_equilibrium()
        context["cost_fn"].set_target(labels)
        hidden = _hidden_saturation(context, perfect_eps=args.perfect_eps)
        grads = estimator.compute_gradient()[: len(params)]

        param_l2_sq = 0.0
        grad_l2_sq = 0.0
        relative_values: list[float] = []
        step_values: list[float] = []
        for index, (param, grad) in enumerate(zip(params, grads)):
            stats = _parameter_stats(param, grad, lrs[index])
            param_l2_sq += stats["param_l2"] ** 2
            grad_l2_sq += stats["grad_l2"] ** 2
            relative_values.append(stats["relative_grad_l2"])
            step_values.append(stats["relative_step_l2"])
            param_accumulator[index].append(stats)
        global_param_l2 = math.sqrt(param_l2_sq)
        global_grad_l2 = math.sqrt(grad_l2_sq)
        batch_summaries.append(
            {
                **hidden,
                "global_param_l2": global_param_l2,
                "global_grad_l2": global_grad_l2,
                "global_relative_grad_l2": (
                    global_grad_l2 / global_param_l2 if global_param_l2 > 0 else math.nan
                ),
                "global_relative_step_l2": (
                    lr_reference * global_grad_l2 / global_param_l2
                    if global_param_l2 > 0 and math.isfinite(lr_reference)
                    else math.nan
                ),
                "max_param_relative_grad_l2": float(np.nanmax(np.asarray(relative_values, dtype=np.float64))),
                "max_param_relative_step_l2": float(np.nanmax(np.asarray(step_values, dtype=np.float64))),
            }
        )
        samples += int(images.shape[0])
        del grads
        if device.type == "cuda":
            torch.cuda.empty_cache()

    param_summaries = []
    for index, values in param_accumulator.items():
        if not values:
            continue
        param = params[index]
        param_summaries.append(
            {
                "index": index,
                "name": _parameter_name(param, index),
                "class": param.__class__.__name__,
                "lr": lrs[index],
                "grad_l2": _mean([row["grad_l2"] for row in values]),
                "relative_grad_l2": _mean([row["relative_grad_l2"] for row in values]),
                "relative_step_l2": _mean([row["relative_step_l2"] for row in values]),
                "grad_zero_fraction": _mean([row["grad_zero_fraction"] for row in values]),
            }
        )

    row = _empty_param_slots()
    weight_slots = ["weight0", "weight1", "weight2"]
    bias_slots = ["bias0", "bias1"]
    weight_index = 0
    bias_index = 0
    for summary in param_summaries:
        param_class = summary["class"]
        slot = None
        if "Weight" in param_class and weight_index < len(weight_slots):
            slot = weight_slots[weight_index]
            weight_index += 1
        elif "Bias" in param_class and bias_index < len(bias_slots):
            slot = bias_slots[bias_index]
            bias_index += 1
        if slot is None:
            continue
        row[f"{slot}_grad_l2"] = summary["grad_l2"]
        row[f"{slot}_relative_grad_l2"] = summary["relative_grad_l2"]
        if slot.startswith("weight"):
            row[f"{slot}_relative_step_l2"] = summary["relative_step_l2"]

    model_cfg = spec.model_cfg
    target_label = _path_part(spec.run_dir, "target_")
    aggregate = {
        "conv_depth": len(model_cfg.get("conv_pipeline") or []),
        "non_linearity": model_cfg.get("non_linearity"),
        "run_name": spec.run_dir.parent.name,
        "phase": phase,
        "voltage_amp": model_cfg.get("voltage_amp", math.nan),
        "current_amp": model_cfg.get("current_amp", math.nan),
        "input_gain": model_cfg.get("input_gain", math.nan),
        "v_off": model_cfg.get("hard_sigmoid_param", {}).get("v_max", math.nan),
        "lr_reference": lr_reference,
        "epochs": spec.source_config.get("lab", {}).get("epochs", ""),
        "seed": spec.source_config.get("seed", spec.metrics.get("seed", "")),
        "run_dir": str(spec.run_dir),
        "best_test_accuracy": spec.metrics.get("best_test_accuracy", math.nan),
        "final_test_accuracy": spec.metrics.get("final_test_accuracy", math.nan),
        "target_label": target_label,
        "target_saturation": _target_from_label(target_label),
        "split": args.split,
        "num_samples": samples,
        "hidden1_saturation": _mean([batch["hidden1_saturation"] for batch in batch_summaries]),
        "hidden2_saturation": _mean([batch["hidden2_saturation"] for batch in batch_summaries]),
        "all_hidden_saturation": _mean([batch["all_hidden_saturation"] for batch in batch_summaries]),
        "global_param_l2": _mean([batch["global_param_l2"] for batch in batch_summaries]),
        "global_grad_l2": _mean([batch["global_grad_l2"] for batch in batch_summaries]),
        "global_relative_grad_l2": _mean([batch["global_relative_grad_l2"] for batch in batch_summaries]),
        "global_relative_step_l2": _mean([batch["global_relative_step_l2"] for batch in batch_summaries]),
        "max_param_relative_grad_l2": _mean(
            [batch["max_param_relative_grad_l2"] for batch in batch_summaries]
        ),
        "max_param_relative_step_l2": _mean(
            [batch["max_param_relative_step_l2"] for batch in batch_summaries]
        ),
        **row,
        "param_summaries_json": json.dumps(param_summaries),
    }
    return aggregate


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", action="append", default=[])
    parser.add_argument("--run-dir", action="append", default=[])
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--best-per-group", action="store_true")
    parser.add_argument("--phase", choices=("init", "final", "both"), default="both")
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--dataset-root", default=str(REPO_ROOT / "data"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--inference-iterations", type=int, default=None)
    parser.add_argument("--training-iterations", type=int, default=None)
    parser.add_argument("--perfect-eps", type=float, default=1e-8)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_root and not args.run_dir:
        raise ValueError("Provide at least one --input-root or --run-dir.")
    run_dirs = [_normalize_run_dir(path) for path in args.run_dir]
    run_dirs.extend(_discover_run_dirs(args.input_root))
    specs = _load_specs(run_dirs)
    if args.best_per_group:
        specs = _best_specs_per_group(specs)
    if args.phase in {"final", "both"}:
        specs = [spec for spec in specs if (spec.run_dir / "final_model.pt").exists()]
    specs = sorted(
        specs,
        key=lambda spec: (
            len(spec.model_cfg.get("conv_pipeline") or []),
            str(spec.model_cfg.get("non_linearity", "")),
            spec.run_dir.parent.name,
            str(spec.run_dir),
        ),
    )
    phases = ["init", "final"] if args.phase == "both" else [args.phase]
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(
        f"[discover] runs={len(specs)} phases={phases} device={device} "
        f"split={args.split} max_samples={args.max_samples}",
        flush=True,
    )
    if args.dry_run:
        for spec in specs:
            print(
                f"[dry-run] conv{len(spec.model_cfg.get('conv_pipeline') or [])} "
                f"{spec.model_cfg.get('non_linearity')} {spec.run_dir.parent.name} {spec.run_dir}"
            )
        return
    rows = []
    total = len(specs) * len(phases)
    count = 0
    for spec in specs:
        for phase in phases:
            count += 1
            print(
                f"[measure] {count}/{total} {phase} conv{len(spec.model_cfg.get('conv_pipeline') or [])} "
                f"{spec.model_cfg.get('non_linearity')} {spec.run_dir.parent.name}",
                flush=True,
            )
            rows.append(_measure_phase(spec, phase=phase, args=args, device=device))
    _write_csv(Path(args.output_csv), rows)
    print(f"[done] wrote {args.output_csv}", flush=True)


if __name__ == "__main__":
    main()
