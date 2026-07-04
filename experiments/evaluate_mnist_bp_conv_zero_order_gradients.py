#!/usr/bin/env python3
"""Probe Conv MNIST BP gradients with zeroth-order finite differences.

This diagnostic compares BP/autograd gradients with finite-difference
estimates that only use loss evaluations. By default it probes the truncated
from-reset loss, C(s_K(theta)); with ``--free-iteration-counts`` it first
settles a detached free state for T iterations and then probes K additional
updates from that state, matching the T/K gradient-death diagnostic protocol.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
LABS_DIR = REPO_ROOT / "labs"
for path in (REPO_ROOT, LABS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import model  # noqa: F401,E402 - anchor repo-local package before labs imports.
from diagnose_mnist_bp_conv_saturation_gradients import (  # noqa: E402
    _build_init_context,
    _discover_runs,
    _param_bounds,
    _param_name,
)
from evaluate_mnist_bp_write_noise_sweep import (  # noqa: E402
    _build_eval_context,
    _json_sanitize,
    _write_json,
)
from labs.mnist_train import _build_tracking_minimizer  # noqa: E402
from training.sgd import Backprop  # noqa: E402


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "mnist_bp_conv_zero_order_gradient_diagnostics"

RAW_COLUMNS = [
    "source_label",
    "run_dir",
    "checkpoint_kind",
    "checkpoint_path",
    "non_linearity",
    "run_name",
    "seed",
    "voltage_amp",
    "current_amp",
    "split",
    "batch_index",
    "free_iteration_count",
    "iteration_count",
    "epsilon",
    "param_index",
    "param_name",
    "param_class",
    "probe_kind",
    "probe_index",
    "flat_index",
    "param_value",
    "num_direction_values",
    "loss_base",
    "loss_plus",
    "loss_minus",
    "fd_value",
    "bp_value",
    "abs_error",
    "relative_abs_error",
    "bp_grad_l2",
    "bp_grad_abs_max",
    "bp_grad_zero_fraction",
]

SUMMARY_COLUMNS = [
    "source_label",
    "run_dir",
    "checkpoint_kind",
    "non_linearity",
    "run_name",
    "seed",
    "voltage_amp",
    "current_amp",
    "split",
    "free_iteration_count",
    "iteration_count",
    "epsilon",
    "param_index",
    "param_name",
    "param_class",
    "probe_kind",
    "num_probes",
    "loss_base_mean",
    "bp_abs_mean",
    "bp_abs_max",
    "bp_nonzero_fraction",
    "fd_abs_mean",
    "fd_abs_max",
    "fd_nonzero_fraction",
    "abs_error_mean",
    "relative_abs_error_mean",
    "sign_agreement_fraction",
    "bp_grad_l2_mean",
    "bp_grad_abs_max_mean",
    "bp_grad_zero_fraction_mean",
]


def _write_rows(path: Path, columns: list[str], rows: list[dict], *, append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and append
    mode = "a" if append else "w"
    with path.open(mode, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({column: _json_sanitize(row.get(column, "")) for column in columns})


def _finite_float(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def _mean(values: list[float]) -> float:
    arr = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    return float(arr.mean()) if arr.size else math.nan


def _max_abs(values: list[float]) -> float:
    arr = np.asarray([abs(value) for value in values if math.isfinite(value)], dtype=np.float64)
    return float(arr.max()) if arr.size else math.nan


def _nonzero_fraction(values: list[float], *, tol: float) -> float:
    arr = np.asarray([abs(value) for value in values if math.isfinite(value)], dtype=np.float64)
    return float(np.mean(arr > tol)) if arr.size else math.nan


def _sign_agreement(values: list[dict], *, tol: float) -> float:
    signs: list[float] = []
    for row in values:
        fd = _finite_float(row["fd_value"])
        bp = _finite_float(row["bp_value"])
        if not math.isfinite(fd) or not math.isfinite(bp):
            continue
        if abs(fd) <= tol or abs(bp) <= tol:
            continue
        signs.append(1.0 if math.copysign(1.0, fd) == math.copysign(1.0, bp) else 0.0)
    return _mean(signs)


def _build_context(run, *, args: argparse.Namespace, device: torch.device) -> dict:
    if run.checkpoint_kind == "init":
        context = _build_init_context(
            run,
            device=device,
            eval_batch_size=args.batch_size,
            no_download=args.no_download,
            inference_iterations_override=args.iteration_counts[0],
            dataset_root_override=args.dataset_root,
        )
    else:
        context = _build_eval_context(
            run,
            device=device,
            eval_batch_size=args.batch_size,
            no_download=args.no_download,
            inference_iterations_override=args.iteration_counts[0],
            dataset_root_override=args.dataset_root,
        )
    context["adaptive_equilibrium"] = bool(args.adaptive_equilibrium)
    return context


def _make_minimizer(context: dict, iteration_count: int):
    energy_fn = context["energy_fn"]
    return _build_tracking_minimizer(
        energy_fn,
        context["free_layers"],
        context["model_cfg"],
        context["config"]["energy_minimizer"]["mode"],
        num_iterations=int(iteration_count),
        voltage_amp=energy_fn._voltage_amp,
        current_amp=energy_fn._current_amp,
        adaptive_equilibrium=context["adaptive_equilibrium"],
    )


def _reset_params(context: dict) -> None:
    for param, base in zip(context["params"], context["base_states"]):
        param.state = base.detach().clone()


def _set_free_layer_states(context: dict, states: list[torch.Tensor] | None) -> None:
    if states is None:
        return
    for layer, state in zip(context["free_layers"], states):
        layer.state = state.detach().clone()


def _set_perturbed_param(context: dict, param_index: int, value: torch.Tensor) -> None:
    context["params"][param_index].state = value.detach().clone()
    context["params"][param_index].clamp_()


def _compute_free_layer_states(
    context: dict,
    minimizer,
    images: torch.Tensor,
    labels: torch.Tensor,
) -> list[torch.Tensor] | None:
    if minimizer is None:
        return None
    _reset_params(context)
    with torch.no_grad():
        context["network"].set_input(images, reset=True)
        context["cost_fn"].set_target(labels)
        minimizer.compute_equilibrium()
    return [layer.state.detach().clone() for layer in context["free_layers"]]


def _loss_for_params(
    context: dict,
    minimizer,
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    initial_layer_states: list[torch.Tensor] | None = None,
    param_index: int | None = None,
    param_value: torch.Tensor | None = None,
) -> float:
    _reset_params(context)
    if param_index is not None:
        if param_value is None:
            raise ValueError("Expected param_value when param_index is provided.")
        _set_perturbed_param(context, param_index, param_value)

    with torch.no_grad():
        context["network"].set_input(images, reset=True)
        _set_free_layer_states(context, initial_layer_states)
        minimizer.compute_equilibrium()
        context["cost_fn"].set_target(labels)
        return float(context["cost_fn"].eval().mean().item())


def _compute_bp(
    context: dict,
    minimizer,
    images: torch.Tensor,
    labels: torch.Tensor,
    initial_layer_states: list[torch.Tensor] | None = None,
) -> tuple[float, float, list[torch.Tensor]]:
    _reset_params(context)
    context["network"].set_input(images, reset=True)
    _set_free_layer_states(context, initial_layer_states)
    context["cost_fn"].set_target(labels)
    estimator = Backprop(context["params"], context["free_layers"], context["cost_fn"], minimizer)
    grads = estimator.compute_gradient()[: len(context["params"])]
    with torch.no_grad():
        loss = float(context["cost_fn"].eval().mean().item())
        errors = context["cost_fn"].error_fn()
        accuracy = float((~errors).float().mean().item())
    return loss, accuracy, [grad.detach().clone() for grad in grads]


def _collect_batches(context: dict, split: str, max_batches: int | None) -> list[tuple[torch.Tensor, torch.Tensor]]:
    loader = context["train_loader"] if split == "train" else context["test_loader"]
    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    for batch_index, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batches.append((images.detach().clone(), labels.detach().clone()))
    if not batches:
        raise ValueError(f"No batches collected for split {split!r}.")
    return batches


def _param_indices(params: list, names: list[str] | None) -> list[int]:
    if not names:
        return [index for index, param in enumerate(params) if "Weight" in param.__class__.__name__]
    wanted = {name.strip() for name in names}
    selected = [
        index
        for index, param in enumerate(params)
        if _param_name(param, index) in wanted or param.__class__.__name__ in wanted
    ]
    if not selected:
        available = ", ".join(_param_name(param, index) for index, param in enumerate(params))
        raise ValueError(f"Expected --param-name to match one of [{available}], got {sorted(wanted)!r}.")
    return selected


def _interior_mask(param, epsilon: float) -> torch.Tensor:
    state = param.state.detach()
    mask = torch.ones_like(state, dtype=torch.bool)
    lower, upper = _param_bounds(param)
    if lower is not None and math.isfinite(float(lower)):
        mask &= state > float(lower) + float(epsilon)
    if upper is not None and math.isfinite(float(upper)):
        mask &= state < float(upper) - float(epsilon)
    return mask


def _make_direction(state: torch.Tensor, mask: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    rand = torch.empty(state.shape, device=state.device, dtype=state.dtype)
    rand.bernoulli_(0.5, generator=generator)
    direction = rand.mul_(2.0).sub_(1.0)
    return torch.where(mask, direction, torch.zeros_like(direction))


def _sample_coordinate_indices(mask: torch.Tensor, max_samples: int, generator: torch.Generator) -> list[int]:
    interior = torch.nonzero(mask.detach().reshape(-1).cpu(), as_tuple=False).reshape(-1)
    if interior.numel() == 0:
        return []
    if interior.numel() <= max_samples:
        return [int(value) for value in interior.tolist()]
    order = torch.randperm(int(interior.numel()), generator=generator)[: int(max_samples)]
    return [int(value) for value in interior[order].tolist()]


def _probe_param(
    *,
    run,
    args: argparse.Namespace,
    context: dict,
    minimizer,
    images: torch.Tensor,
    labels: torch.Tensor,
    initial_layer_states: list[torch.Tensor] | None,
    batch_index: int,
    free_iteration_count: int | None,
    iteration_count: int,
    epsilon: float,
    param_index: int,
    bp_grad: torch.Tensor,
    loss_base: float,
    generator: torch.Generator,
) -> list[dict]:
    param = context["params"][param_index]
    param_name = _param_name(param, param_index)
    base_state = context["base_states"][param_index].detach().clone()
    mask = _interior_mask(param, epsilon)
    bp_grad = bp_grad.detach()
    bp_grad_l2 = float(torch.linalg.vector_norm(bp_grad).item())
    bp_grad_abs_max = float(bp_grad.abs().max().item()) if bp_grad.numel() else math.nan
    bp_grad_zero_fraction = float((bp_grad.abs() <= args.zero_tol).float().mean().item()) if bp_grad.numel() else math.nan

    common = {
        "source_label": run.source_label,
        "run_dir": str(run.run_dir),
        "checkpoint_kind": run.checkpoint_kind,
        "checkpoint_path": str(run.checkpoint_path),
        "non_linearity": run.non_linearity,
        "run_name": run.run_name,
        "seed": run.seed,
        "voltage_amp": run.voltage_amp,
        "current_amp": run.current_amp,
        "split": args.split,
        "batch_index": batch_index,
        "free_iteration_count": "" if free_iteration_count is None else free_iteration_count,
        "iteration_count": iteration_count,
        "epsilon": epsilon,
        "param_index": param_index,
        "param_name": param_name,
        "param_class": param.__class__.__name__,
        "loss_base": loss_base,
        "bp_grad_l2": bp_grad_l2,
        "bp_grad_abs_max": bp_grad_abs_max,
        "bp_grad_zero_fraction": bp_grad_zero_fraction,
    }

    rows: list[dict] = []
    if int(mask.sum().item()) == 0:
        return rows

    for probe_index in range(args.num_directions):
        direction = _make_direction(base_state, mask, generator)
        num_values = int((direction != 0).sum().item())
        if num_values == 0:
            continue
        plus = base_state + float(epsilon) * direction
        minus = base_state - float(epsilon) * direction
        loss_plus = _loss_for_params(
            context,
            minimizer,
            images,
            labels,
            initial_layer_states=initial_layer_states,
            param_index=param_index,
            param_value=plus,
        )
        loss_minus = _loss_for_params(
            context,
            minimizer,
            images,
            labels,
            initial_layer_states=initial_layer_states,
            param_index=param_index,
            param_value=minus,
        )
        fd_value = (loss_plus - loss_minus) / (2.0 * float(epsilon))
        bp_value = float(torch.sum(bp_grad * direction).item())
        rows.append(
            {
                **common,
                "probe_kind": "direction",
                "probe_index": probe_index,
                "flat_index": "",
                "param_value": "",
                "num_direction_values": num_values,
                "loss_plus": loss_plus,
                "loss_minus": loss_minus,
                "fd_value": fd_value,
                "bp_value": bp_value,
                "abs_error": abs(fd_value - bp_value),
                "relative_abs_error": abs(fd_value - bp_value) / max(abs(bp_value), args.zero_tol),
            }
        )

    coordinate_indices = _sample_coordinate_indices(mask, args.num_coordinates, generator)
    flat_base = base_state.reshape(-1)
    flat_bp = bp_grad.reshape(-1)
    for probe_index, flat_index in enumerate(coordinate_indices):
        plus = base_state.clone().reshape(-1)
        minus = base_state.clone().reshape(-1)
        plus[flat_index] += float(epsilon)
        minus[flat_index] -= float(epsilon)
        plus_state = plus.reshape_as(base_state)
        minus_state = minus.reshape_as(base_state)
        loss_plus = _loss_for_params(
            context,
            minimizer,
            images,
            labels,
            initial_layer_states=initial_layer_states,
            param_index=param_index,
            param_value=plus_state,
        )
        loss_minus = _loss_for_params(
            context,
            minimizer,
            images,
            labels,
            initial_layer_states=initial_layer_states,
            param_index=param_index,
            param_value=minus_state,
        )
        fd_value = (loss_plus - loss_minus) / (2.0 * float(epsilon))
        bp_value = float(flat_bp[flat_index].item())
        rows.append(
            {
                **common,
                "probe_kind": "coordinate",
                "probe_index": probe_index,
                "flat_index": flat_index,
                "param_value": float(flat_base[flat_index].item()),
                "num_direction_values": 1,
                "loss_plus": loss_plus,
                "loss_minus": loss_minus,
                "fd_value": fd_value,
                "bp_value": bp_value,
                "abs_error": abs(fd_value - bp_value),
                "relative_abs_error": abs(fd_value - bp_value) / max(abs(bp_value), args.zero_tol),
            }
        )

    return rows


def _summarize(rows: list[dict], *, zero_tol: float) -> list[dict]:
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (
            row["source_label"],
            row["run_dir"],
            row["checkpoint_kind"],
            row["non_linearity"],
            row["run_name"],
            row["seed"],
            row["voltage_amp"],
            row["current_amp"],
            row["split"],
            row["free_iteration_count"],
            row["iteration_count"],
            row["epsilon"],
            row["param_index"],
            row["param_name"],
            row["param_class"],
            row["probe_kind"],
        )
        grouped.setdefault(key, []).append(row)

    summaries: list[dict] = []
    for key, values in sorted(grouped.items()):
        (
            source_label,
            run_dir,
            checkpoint_kind,
            non_linearity,
            run_name,
            seed,
            voltage_amp,
            current_amp,
            split,
            free_iteration_count,
            iteration_count,
            epsilon,
            param_index,
            param_name,
            param_class,
            probe_kind,
        ) = key
        bp_values = [_finite_float(row["bp_value"]) for row in values]
        fd_values = [_finite_float(row["fd_value"]) for row in values]
        summaries.append(
            {
                "source_label": source_label,
                "run_dir": run_dir,
                "checkpoint_kind": checkpoint_kind,
                "non_linearity": non_linearity,
                "run_name": run_name,
                "seed": seed,
                "voltage_amp": voltage_amp,
                "current_amp": current_amp,
                "split": split,
                "free_iteration_count": free_iteration_count,
                "iteration_count": iteration_count,
                "epsilon": epsilon,
                "param_index": param_index,
                "param_name": param_name,
                "param_class": param_class,
                "probe_kind": probe_kind,
                "num_probes": len(values),
                "loss_base_mean": _mean([_finite_float(row["loss_base"]) for row in values]),
                "bp_abs_mean": _mean([abs(value) for value in bp_values]),
                "bp_abs_max": _max_abs(bp_values),
                "bp_nonzero_fraction": _nonzero_fraction(bp_values, tol=zero_tol),
                "fd_abs_mean": _mean([abs(value) for value in fd_values]),
                "fd_abs_max": _max_abs(fd_values),
                "fd_nonzero_fraction": _nonzero_fraction(fd_values, tol=zero_tol),
                "abs_error_mean": _mean([_finite_float(row["abs_error"]) for row in values]),
                "relative_abs_error_mean": _mean([_finite_float(row["relative_abs_error"]) for row in values]),
                "sign_agreement_fraction": _sign_agreement(values, tol=zero_tol),
                "bp_grad_l2_mean": _mean([_finite_float(row["bp_grad_l2"]) for row in values]),
                "bp_grad_abs_max_mean": _mean([_finite_float(row["bp_grad_abs_max"]) for row in values]),
                "bp_grad_zero_fraction_mean": _mean([_finite_float(row["bp_grad_zero_fraction"]) for row in values]),
            }
        )
    return summaries


def _run_single(run, *, args: argparse.Namespace, device: torch.device) -> list[dict]:
    context = _build_context(run, args=args, device=device)
    batches = _collect_batches(context, args.split, args.max_batches)
    selected_param_indices = _param_indices(context["params"], args.param_name)
    rows: list[dict] = []

    free_iteration_counts = args.free_iteration_counts or [None]
    for free_iteration_count in free_iteration_counts:
        free_minimizer = (
            None if free_iteration_count is None else _make_minimizer(context, int(free_iteration_count))
        )
        for batch_index, (images, labels) in enumerate(batches):
            images = images.to(device)
            labels = labels.to(device)
            initial_layer_states = _compute_free_layer_states(context, free_minimizer, images, labels)
            for iteration_count in args.iteration_counts:
                minimizer = _make_minimizer(context, int(iteration_count))
                loss_base, _accuracy, bp_grads = _compute_bp(
                    context,
                    minimizer,
                    images,
                    labels,
                    initial_layer_states=initial_layer_states,
                )
                for epsilon in args.epsilons:
                    generator = torch.Generator(device=device)
                    generator.manual_seed(
                        int(args.probe_seed)
                        + 1000003 * int(iteration_count)
                        + 104729 * (0 if free_iteration_count is None else int(free_iteration_count))
                        + 9176 * int(batch_index)
                        + 271 * int(round(float(epsilon) * 1.0e8))
                    )
                    for param_index in selected_param_indices:
                        rows.extend(
                            _probe_param(
                                run=run,
                                args=args,
                                context=context,
                                minimizer=minimizer,
                                images=images,
                                labels=labels,
                                initial_layer_states=initial_layer_states,
                                batch_index=batch_index,
                                free_iteration_count=free_iteration_count,
                                iteration_count=int(iteration_count),
                                epsilon=float(epsilon),
                                param_index=param_index,
                                bp_grad=bp_grads[param_index],
                                loss_base=loss_base,
                                generator=generator,
                            )
                        )
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", action="append", required=True)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--checkpoint-kind", choices=("auto", "best", "final", "init"), default="auto")
    parser.add_argument("--run-name", action="append")
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=1)
    parser.add_argument(
        "--free-iteration-counts",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Optional pre-settling counts. When set, finite differences and BP start "
            "from the same detached free state after this many iterations, matching "
            "the T/K gradient-death diagnostic protocol."
        ),
    )
    parser.add_argument("--iteration-counts", type=int, nargs="+", default=[4, 16])
    parser.add_argument(
        "--adaptive-equilibrium",
        action="store_true",
        help="Allow minimizers to stop early by residual tolerance instead of running exact iteration counts.",
    )
    parser.add_argument("--epsilons", type=float, nargs="+", default=[1.0e-3, 1.0e-2, 1.0e-1])
    parser.add_argument("--param-name", action="append", default=None)
    parser.add_argument("--num-directions", type=int, default=4)
    parser.add_argument("--num-coordinates", type=int, default=8)
    parser.add_argument("--probe-seed", type=int, default=0)
    parser.add_argument("--zero-tol", type=float, default=1.0e-10)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.param_name is None:
        args.param_name = ["ConvWeight_0", "ConvWeight_1"]
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    input_roots = [Path(value).expanduser().resolve() for value in args.input_root]

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"Requested CUDA device {args.device!r}, but CUDA is unavailable.")

    runs = _discover_runs(input_roots, args.checkpoint_kind)
    if args.run_name:
        wanted = set(args.run_name)
        runs = [run for run in runs if run.run_name in wanted]
    if args.limit is not None:
        runs = runs[: int(args.limit)]

    _write_json(
        output_root / "diagnostic_config.json",
        {
            "input_roots": [str(path) for path in input_roots],
            "output_root": str(output_root),
            "dataset_root": args.dataset_root,
            "device": str(device),
            "checkpoint_kind": args.checkpoint_kind,
            "run_names": args.run_name,
            "split": args.split,
            "batch_size": args.batch_size,
            "max_batches": args.max_batches,
            "free_iteration_counts": args.free_iteration_counts,
            "iteration_counts": args.iteration_counts,
            "adaptive_equilibrium": args.adaptive_equilibrium,
            "epsilons": args.epsilons,
            "param_name": args.param_name,
            "num_directions": args.num_directions,
            "num_coordinates": args.num_coordinates,
            "probe_seed": args.probe_seed,
            "zero_tol": args.zero_tol,
            "num_runs": len(runs),
            "runs": [str(run.run_dir) for run in runs],
            "diagnostic_category": "zeroth_order_finite_difference_gradient_check",
        },
    )
    print(f"[zero-order] runs={len(runs)} output={output_root} device={device}")
    if args.dry_run:
        for run in runs:
            print(f"[dry-run] {run.checkpoint_kind} {run.checkpoint_path}")
        return

    raw_rows: list[dict] = []
    for index, run in enumerate(runs, start=1):
        print(f"[run {index}/{len(runs)}] {run.checkpoint_kind} {run.run_dir}", flush=True)
        rows = _run_single(run, args=args, device=device)
        raw_rows.extend(rows)
        _write_rows(output_root / "raw_zero_order_gradients.csv", RAW_COLUMNS, raw_rows, append=False)
        _write_rows(
            output_root / "summary_zero_order_gradients.csv",
            SUMMARY_COLUMNS,
            _summarize(raw_rows, zero_tol=args.zero_tol),
            append=False,
        )
    print(f"[done] wrote {len(raw_rows)} raw rows under {output_root}")


if __name__ == "__main__":
    main()
