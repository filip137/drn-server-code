#!/usr/bin/env python3
"""Small residual-current diagnostic for hard-sigmoid MNIST DRN checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

from evaluate_mnist_bp_cost_hessian_noise_curvature import (  # noqa: E402
    RUN_ORDER,
    _apply_direction,
    _make_direction,
    _selected_param_indices,
)
from evaluate_mnist_bp_output_decomposition_physical_logg_noise import (  # noqa: E402
    _apply_logg,
    _make_logg_xi,
)
from evaluate_mnist_bp_write_noise_sweep import (  # noqa: E402
    DEFAULT_INPUT_ROOT,
    _build_eval_context,
    _json_sanitize,
    _read_summary_rows,
    _reset_params,
    _scores,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "mnist_bp_residual_current_diagnostic_hardsigmoid_iter16"

SUMMARY_COLUMNS = [
    "run_name",
    "seed",
    "voltage_amp",
    "current_amp",
    "condition",
    "num_examples",
    "accuracy",
    "loss",
    "row_scale_output",
    "hidden_scaled_inf_mean",
    "hidden_scaled_inf_p90",
    "hidden_scaled_inf_max",
    "output_scaled_inf_mean",
    "output_scaled_inf_p90",
    "output_scaled_inf_max",
    "hidden_raw_inf_mean",
    "hidden_raw_inf_p90",
    "hidden_raw_inf_max",
    "output_raw_inf_mean",
    "output_raw_inf_p90",
    "output_raw_inf_max",
    "total_scaled_inf_mean",
    "total_scaled_inf_p90",
    "total_scaled_inf_max",
    "total_raw_inf_mean",
    "total_raw_inf_p90",
    "total_raw_inf_max",
]


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _json_sanitize(row.get(column, "")) for column in SUMMARY_COLUMNS})


def _stats(values: list[float] | np.ndarray, prefix: str) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {f"{prefix}_mean": math.nan, f"{prefix}_p90": math.nan, f"{prefix}_max": math.nan}
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_p90": float(np.percentile(arr, 90)),
        f"{prefix}_max": float(np.max(arr)),
    }


def _condition_specs(args: argparse.Namespace) -> list[tuple[str, str, float, int]]:
    specs: list[tuple[str, str, float, int]] = [("clean", "clean", 0.0, 0)]
    specs.append((f"software_theta_eps_{args.epsilon:g}", "software_theta", float(args.epsilon), 0))
    specs.append((f"physical_logg_eps_{args.epsilon:g}", "physical_logg", float(args.epsilon), 0))
    specs.append((f"physical_logg_sigma_{args.noise_sigma:g}_seed_{args.noise_seed}", "physical_logg", float(args.noise_sigma), int(args.noise_seed)))
    return specs


def _apply_condition(context: dict, run_seed: int, args: argparse.Namespace, *, kind: str, magnitude: float, direction_index: int) -> None:
    _reset_params(context)
    if kind == "clean":
        return
    selected = set(_selected_param_indices(context, include_biases=False))
    direction_seed = int(args.direction_seed_offset + 100000 * run_seed + direction_index)
    if kind == "software_theta":
        generator = torch.Generator(device=str(context["base_states"][0].device))
        generator.manual_seed(direction_seed)
        directions, _ = _make_direction(
            context,
            generator=generator,
            selected_indices=selected,
            target_index=None,
        )
        _apply_direction(
            context,
            directions,
            epsilon=float(magnitude),
            sign=1.0,
            clamp_perturbed=args.clamp_perturbed,
        )
        return
    if kind == "physical_logg":
        generator = torch.Generator(device=str(context["base_states"][0].device))
        generator.manual_seed(direction_seed)
        xis, _ = _make_logg_xi(context, generator=generator, selected_indices=selected)
        _apply_logg(
            context,
            xis,
            magnitude=float(magnitude),
            sign=1.0,
            clamp=args.clamp_perturbed,
        )
        return
    raise ValueError(f"Unknown condition kind {kind!r}.")


def _evaluate_residuals(context: dict, *, max_eval_batches: int | None) -> dict:
    network = context["network"]
    minimizer = context["minimizer"]
    fn = getattr(minimizer, "_fn", context["energy_fn"])
    cost_fn = context["cost_fn"]
    output_layer = context["output_layer"]
    test_loader = context["test_loader"]
    device = context["base_states"][0].device
    num_classes = context["num_classes"]
    free_layers = context["free_layers"]

    hidden_scaled: list[float] = []
    output_scaled: list[float] = []
    hidden_raw: list[float] = []
    output_raw: list[float] = []
    losses: list[np.ndarray] = []
    correct: list[np.ndarray] = []

    row_scale_output = float(context["model_cfg"]["current_amp"] / context["model_cfg"]["voltage_amp"])
    output_raw_factor = 1.0 / row_scale_output

    for batch_index, (images, labels) in enumerate(test_loader):
        if max_eval_batches is not None and batch_index >= max_eval_batches:
            break
        images = images.to(device)
        labels = labels.to(device)
        network.set_input(images, reset=True)
        minimizer.compute_equilibrium()
        cost_fn.set_target(labels)

        scores = _scores(output_layer.state.detach(), num_classes)
        pred = torch.argmax(scores, dim=1)
        losses.append(cost_fn.eval().detach().cpu().numpy().astype(np.float64))
        correct.append(pred.eq(labels).detach().cpu().numpy().astype(bool))

        grads = [fn.grad_layer_fn(layer)().detach() for layer in free_layers]
        if len(grads) != 2:
            raise ValueError(f"Expected DRN-XS with two free layers, got {len(grads)}.")
        h = grads[0].reshape(grads[0].shape[0], -1).abs().amax(dim=1).cpu().numpy()
        y = grads[1].reshape(grads[1].shape[0], -1).abs().amax(dim=1).cpu().numpy()
        hidden_scaled.extend(h.astype(np.float64).tolist())
        output_scaled.extend(y.astype(np.float64).tolist())
        hidden_raw.extend(h.astype(np.float64).tolist())
        output_raw.extend((y.astype(np.float64) * output_raw_factor).tolist())

    if not hidden_scaled:
        raise ValueError("Evaluation saw zero examples.")
    hidden_scaled_arr = np.asarray(hidden_scaled, dtype=np.float64)
    output_scaled_arr = np.asarray(output_scaled, dtype=np.float64)
    hidden_raw_arr = np.asarray(hidden_raw, dtype=np.float64)
    output_raw_arr = np.asarray(output_raw, dtype=np.float64)
    total_scaled = np.maximum(hidden_scaled_arr, output_scaled_arr)
    total_raw = np.maximum(hidden_raw_arr, output_raw_arr)
    loss_arr = np.concatenate(losses)
    correct_arr = np.concatenate(correct)
    return {
        "num_examples": int(loss_arr.shape[0]),
        "accuracy": float(np.mean(correct_arr)),
        "loss": float(np.mean(loss_arr)),
        "row_scale_output": row_scale_output,
        **_stats(hidden_scaled_arr, "hidden_scaled_inf"),
        **_stats(output_scaled_arr, "output_scaled_inf"),
        **_stats(hidden_raw_arr, "hidden_raw_inf"),
        **_stats(output_raw_arr, "output_raw_inf"),
        **_stats(total_scaled, "total_scaled_inf"),
        **_stats(total_raw, "total_raw_inf"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default=str(DEFAULT_INPUT_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--device", default=None)
    parser.add_argument("--checkpoint-kind", choices=("best", "final"), default="best")
    parser.add_argument("--run-name", action="append", choices=RUN_ORDER)
    parser.add_argument("--training-seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--max-eval-batches", type=int, default=1)
    parser.add_argument("--inference-iterations", type=int, default=16)
    parser.add_argument("--epsilon", type=float, default=0.02)
    parser.add_argument("--noise-sigma", type=float, default=0.2)
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--direction-seed-offset", type=int, default=12345)
    parser.add_argument("--clamp-perturbed", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

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
    rows = sorted(rows, key=lambda row: (RUN_ORDER.index(row.run_name), row.seed))
    config = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "device": str(device),
        "checkpoint_kind": args.checkpoint_kind,
        "training_seeds": args.training_seeds,
        "eval_batch_size": args.eval_batch_size,
        "max_eval_batches": args.max_eval_batches,
        "inference_iterations": args.inference_iterations,
        "epsilon": args.epsilon,
        "noise_sigma": args.noise_sigma,
        "noise_seed": args.noise_seed,
        "direction_seed_offset": args.direction_seed_offset,
        "clamp_perturbed": bool(args.clamp_perturbed),
    }
    (output_root / "config.json").write_text(json.dumps(_json_sanitize(config), indent=2, sort_keys=True) + "\n")

    out_rows: list[dict] = []
    for run in rows:
        print(f"[residual] {run.run_name} seed={run.seed}")
        context = _build_eval_context(
            run,
            device=device,
            eval_batch_size=args.eval_batch_size,
            no_download=args.no_download,
            inference_iterations_override=args.inference_iterations,
        )
        for condition, kind, magnitude, direction_index in _condition_specs(args):
            _apply_condition(context, run.seed, args, kind=kind, magnitude=magnitude, direction_index=direction_index)
            metrics = _evaluate_residuals(context, max_eval_batches=args.max_eval_batches)
            out_rows.append(
                {
                    "run_name": run.run_name,
                    "seed": run.seed,
                    "voltage_amp": run.voltage_amp,
                    "current_amp": run.current_amp,
                    "condition": condition,
                    **metrics,
                }
            )
        _reset_params(context)

    _write_rows(output_root / "summary.csv", out_rows)
    print(f"[done] wrote {output_root / 'summary.csv'}")


if __name__ == "__main__":
    main()
