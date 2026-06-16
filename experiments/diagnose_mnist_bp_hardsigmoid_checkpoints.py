#!/usr/bin/env python3
"""Diagnostic summaries for trained hard-sigmoid MNIST DRN checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

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


DEFAULT_INPUT_ROOT = (
    REPO_ROOT
    / "results"
    / "mnist_bp_amplification_sweep_hardsigmoid_current_amp_scratch_optuna_best_dense_amp_fix"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "results"
    / "mnist_bp_hardsigmoid_current_amp_checkpoint_diagnostics_dense_amp_fix"
)

SUMMARY_COLUMNS = [
    "run_name",
    "seed",
    "voltage_amp",
    "current_amp",
    "checkpoint_path",
    "sample_count",
    "accuracy",
    "loss_mean",
    "margin_mean",
    "margin_p10",
    "margin_p50",
    "margin_p90",
    "top1_top2_gap_mean",
    "top1_top2_gap_p10",
    "score_abs_mean",
    "score_l2_mean",
    "hidden_mean",
    "hidden_std",
    "hidden_abs_mean",
    "hidden_abs_p90",
    "hidden_abs_p99",
    "hidden_below_fraction_mean",
    "hidden_above_fraction_mean",
    "hidden_active_fraction_mean",
    "hidden_active_fraction_p10",
    "hidden_active_fraction_p50",
    "hidden_active_fraction_p90",
    "hidden_deadzone_margin_mean",
    "hidden_deadzone_margin_p10",
    "hidden_deadzone_margin_p50",
    "hidden_deadzone_margin_p90",
    "hidden_residual_l2_mean",
    "hidden_residual_l2_p90",
    "hidden_residual_inf_mean",
    "hidden_relative_residual_l2_mean",
    "output_residual_l2_mean",
    "output_residual_l2_p90",
    "output_residual_inf_mean",
    "output_relative_residual_l2_mean",
    "extra_iterations",
    "extra_accuracy",
    "extra_loss_mean",
    "extra_prediction_change_fraction",
    "extra_margin_mean",
    "extra_hidden_state_delta_l2_mean",
    "extra_output_state_delta_l2_mean",
    "extra_hidden_residual_l2_mean",
    "extra_output_residual_l2_mean",
    "sample_diagnostics_path",
    "summary_path",
]

SAMPLE_COLUMNS = [
    "run_name",
    "seed",
    "sample_index",
    "label",
    "prediction",
    "correct",
    "loss",
    "margin",
    "top1_top2_gap",
    "score_abs_mean",
    "score_l2",
    "hidden_abs_mean",
    "hidden_abs_max",
    "hidden_below_fraction",
    "hidden_above_fraction",
    "hidden_active_fraction",
    "hidden_deadzone_margin_mean",
    "hidden_deadzone_margin_min",
    "hidden_residual_l2",
    "hidden_residual_inf",
    "hidden_relative_residual_l2",
    "output_residual_l2",
    "output_residual_inf",
    "output_relative_residual_l2",
    "extra_prediction",
    "extra_correct",
    "extra_loss",
    "extra_margin",
    "extra_prediction_changed",
    "extra_hidden_state_delta_l2",
    "extra_output_state_delta_l2",
    "extra_hidden_residual_l2",
    "extra_output_residual_l2",
]


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def stats(values: np.ndarray, prefix: str) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            f"{prefix}_mean": math.nan,
            f"{prefix}_std": math.nan,
            f"{prefix}_p10": math.nan,
            f"{prefix}_p50": math.nan,
            f"{prefix}_p90": math.nan,
            f"{prefix}_p99": math.nan,
        }
    return {
        f"{prefix}_mean": float(np.mean(finite)),
        f"{prefix}_std": float(np.std(finite)),
        f"{prefix}_p10": float(np.percentile(finite, 10)),
        f"{prefix}_p50": float(np.percentile(finite, 50)),
        f"{prefix}_p90": float(np.percentile(finite, 90)),
        f"{prefix}_p99": float(np.percentile(finite, 99)),
    }


def output_scores(output: torch.Tensor, num_classes: int = 10) -> torch.Tensor:
    if output.shape[1] == num_classes:
        return output
    if output.shape[1] == 2 * num_classes:
        return output[:, :num_classes] - output[:, num_classes:]
    raise ValueError(f"Unsupported output shape {tuple(output.shape)}.")


def prediction_and_margin(scores: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    preds = torch.argmax(scores, dim=1)
    correct_scores = scores.gather(1, labels.view(-1, 1)).squeeze(1)
    other_scores = scores.clone()
    other_scores.scatter_(1, labels.view(-1, 1), -float("inf"))
    max_other = other_scores.max(dim=1).values
    margin = correct_scores - max_other
    top2 = torch.topk(scores, k=2, dim=1).values
    top_gap = top2[:, 0] - top2[:, 1]
    return preds, margin, top_gap


def layer_residuals(energy_fn, layers: list) -> dict[str, torch.Tensor]:
    values: dict[str, torch.Tensor] = {}
    for idx, layer in enumerate(layers):
        grad = energy_fn.grad_layer_fn(layer)().detach().flatten(start_dim=1)
        state = layer.state.detach().flatten(start_dim=1)
        l2 = torch.linalg.vector_norm(grad, ord=2, dim=1)
        inf_norm = torch.amax(torch.abs(grad), dim=1)
        state_l2 = torch.linalg.vector_norm(state, ord=2, dim=1)
        rel_l2 = l2 / torch.clamp(state_l2, min=1.0e-12)
        name = "hidden" if idx == 0 else "output" if idx == len(layers) - 1 else f"free_{idx}"
        values[f"{name}_residual_l2"] = l2
        values[f"{name}_residual_inf"] = inf_norm
        values[f"{name}_relative_residual_l2"] = rel_l2
    return values


def make_extra_minimizer(context: dict, extra_iterations: int):
    if extra_iterations <= 0:
        return None
    model_cfg = context["model_cfg"]
    energy_fn = context["energy_fn"]
    return hess_eval._build_tracking_minimizer(
        energy_fn,
        context["free_layers"],
        model_cfg,
        context["source_config"]["energy_minimizer"]["mode"],
        num_iterations=int(extra_iterations),
        voltage_amp=energy_fn._voltage_amp,
        current_amp=energy_fn._current_amp,
    )


def tensor_to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def append_batch(
    *,
    rows: list[dict],
    arrays: dict[str, list[np.ndarray]],
    spec: hess_eval.RunSpec,
    sample_offset: int,
    labels: torch.Tensor,
    hidden: torch.Tensor,
    output: torch.Tensor,
    losses: torch.Tensor,
    residual: dict[str, torch.Tensor],
    hs_params: dict,
    extra_payload: dict[str, torch.Tensor] | None,
) -> int:
    scores = output_scores(output)
    preds, margins, top_gaps = prediction_and_margin(scores, labels)
    correct = preds.eq(labels)

    v_min = float(hs_params["v_min"])
    v_max = float(hs_params["v_max"])
    hidden_flat = hidden.detach().flatten(start_dim=1)
    below = hidden_flat < v_min
    above = hidden_flat > v_max
    active = below | above
    deadzone_margin = torch.minimum(hidden_flat - v_min, v_max - hidden_flat)

    per_sample = {
        "loss": losses,
        "margin": margins,
        "top1_top2_gap": top_gaps,
        "score_abs_mean": torch.mean(torch.abs(scores), dim=1),
        "score_l2": torch.linalg.vector_norm(scores, ord=2, dim=1),
        "hidden_abs_mean": torch.mean(torch.abs(hidden_flat), dim=1),
        "hidden_abs_max": torch.amax(torch.abs(hidden_flat), dim=1),
        "hidden_below_fraction": below.to(torch.float32).mean(dim=1),
        "hidden_above_fraction": above.to(torch.float32).mean(dim=1),
        "hidden_active_fraction": active.to(torch.float32).mean(dim=1),
        "hidden_deadzone_margin_mean": deadzone_margin.mean(dim=1),
        "hidden_deadzone_margin_min": deadzone_margin.min(dim=1).values,
        **residual,
    }
    if extra_payload is not None:
        per_sample.update(extra_payload)

    labels_np = tensor_to_numpy(labels).astype(np.int64)
    preds_np = tensor_to_numpy(preds).astype(np.int64)
    correct_np = tensor_to_numpy(correct).astype(bool)
    for key, value in per_sample.items():
        arrays.setdefault(key, []).append(tensor_to_numpy(value))
    arrays.setdefault("label", []).append(labels_np)
    arrays.setdefault("prediction", []).append(preds_np)
    arrays.setdefault("correct", []).append(correct_np)

    extra_pred = per_sample.get("extra_prediction")
    extra_correct = per_sample.get("extra_correct")
    extra_loss = per_sample.get("extra_loss")
    extra_margin = per_sample.get("extra_margin")
    extra_changed = per_sample.get("extra_prediction_changed")
    extra_hidden_delta = per_sample.get("extra_hidden_state_delta_l2")
    extra_output_delta = per_sample.get("extra_output_state_delta_l2")
    extra_hidden_resid = per_sample.get("extra_hidden_residual_l2")
    extra_output_resid = per_sample.get("extra_output_residual_l2")

    batch_size = int(labels.shape[0])
    for local_idx in range(batch_size):
        row = {
            "run_name": spec.run_name,
            "seed": spec.seed,
            "sample_index": sample_offset + local_idx,
            "label": int(labels_np[local_idx]),
            "prediction": int(preds_np[local_idx]),
            "correct": int(correct_np[local_idx]),
        }
        for key in [
            "loss",
            "margin",
            "top1_top2_gap",
            "score_abs_mean",
            "score_l2",
            "hidden_abs_mean",
            "hidden_abs_max",
            "hidden_below_fraction",
            "hidden_above_fraction",
            "hidden_active_fraction",
            "hidden_deadzone_margin_mean",
            "hidden_deadzone_margin_min",
            "hidden_residual_l2",
            "hidden_residual_inf",
            "hidden_relative_residual_l2",
            "output_residual_l2",
            "output_residual_inf",
            "output_relative_residual_l2",
        ]:
            row[key] = float(tensor_to_numpy(per_sample[key])[local_idx])

        if extra_payload is None:
            row.update(
                {
                    "extra_prediction": "",
                    "extra_correct": "",
                    "extra_loss": "",
                    "extra_margin": "",
                    "extra_prediction_changed": "",
                    "extra_hidden_state_delta_l2": "",
                    "extra_output_state_delta_l2": "",
                    "extra_hidden_residual_l2": "",
                    "extra_output_residual_l2": "",
                }
            )
        else:
            row.update(
                {
                    "extra_prediction": int(tensor_to_numpy(extra_pred)[local_idx]),
                    "extra_correct": int(tensor_to_numpy(extra_correct)[local_idx]),
                    "extra_loss": float(tensor_to_numpy(extra_loss)[local_idx]),
                    "extra_margin": float(tensor_to_numpy(extra_margin)[local_idx]),
                    "extra_prediction_changed": int(tensor_to_numpy(extra_changed)[local_idx]),
                    "extra_hidden_state_delta_l2": float(tensor_to_numpy(extra_hidden_delta)[local_idx]),
                    "extra_output_state_delta_l2": float(tensor_to_numpy(extra_output_delta)[local_idx]),
                    "extra_hidden_residual_l2": float(tensor_to_numpy(extra_hidden_resid)[local_idx]),
                    "extra_output_residual_l2": float(tensor_to_numpy(extra_output_resid)[local_idx]),
                }
            )
        rows.append(row)
    return sample_offset + batch_size


def concatenate(arrays: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    return {key: np.concatenate(values) for key, values in arrays.items() if values}


def summarize_arrays(values: dict[str, np.ndarray], *, extra_iterations: int) -> dict:
    summary = {
        "sample_count": int(values["label"].shape[0]),
        "accuracy": float(np.mean(values["correct"])),
        "loss_mean": float(np.mean(values["loss"])),
        "extra_iterations": int(extra_iterations),
    }
    stat_specs = [
        ("margin", "margin"),
        ("top1_top2_gap", "top1_top2_gap"),
        ("score_abs_mean", "score_abs_mean"),
        ("score_l2", "score_l2"),
        ("hidden_abs_mean", "hidden_abs_mean"),
        ("hidden_active_fraction", "hidden_active_fraction"),
        ("hidden_deadzone_margin_mean", "hidden_deadzone_margin"),
        ("hidden_residual_l2", "hidden_residual_l2"),
        ("hidden_residual_inf", "hidden_residual_inf"),
        ("hidden_relative_residual_l2", "hidden_relative_residual_l2"),
        ("output_residual_l2", "output_residual_l2"),
        ("output_residual_inf", "output_residual_inf"),
        ("output_relative_residual_l2", "output_relative_residual_l2"),
    ]
    for source_key, output_prefix in stat_specs:
        summary.update(stats(values[source_key], output_prefix))

    hidden_all = values["hidden_values"]
    summary.update(
        {
            "hidden_mean": float(np.mean(hidden_all)),
            "hidden_std": float(np.std(hidden_all)),
            "hidden_abs_mean": float(np.mean(np.abs(hidden_all))),
            "hidden_abs_p90": float(np.percentile(np.abs(hidden_all), 90)),
            "hidden_abs_p99": float(np.percentile(np.abs(hidden_all), 99)),
            "hidden_below_fraction_mean": float(np.mean(values["hidden_below_fraction"])),
            "hidden_above_fraction_mean": float(np.mean(values["hidden_above_fraction"])),
        }
    )
    if "extra_correct" in values:
        summary.update(
            {
                "extra_accuracy": float(np.mean(values["extra_correct"])),
                "extra_loss_mean": float(np.mean(values["extra_loss"])),
                "extra_prediction_change_fraction": float(np.mean(values["extra_prediction_changed"])),
                "extra_margin_mean": float(np.mean(values["extra_margin"])),
                "extra_hidden_state_delta_l2_mean": float(np.mean(values["extra_hidden_state_delta_l2"])),
                "extra_output_state_delta_l2_mean": float(np.mean(values["extra_output_state_delta_l2"])),
                "extra_hidden_residual_l2_mean": float(np.mean(values["extra_hidden_residual_l2"])),
                "extra_output_residual_l2_mean": float(np.mean(values["extra_output_residual_l2"])),
            }
        )
    else:
        summary.update(
            {
                "extra_accuracy": math.nan,
                "extra_loss_mean": math.nan,
                "extra_prediction_change_fraction": math.nan,
                "extra_margin_mean": math.nan,
                "extra_hidden_state_delta_l2_mean": math.nan,
                "extra_output_state_delta_l2_mean": math.nan,
                "extra_hidden_residual_l2_mean": math.nan,
                "extra_output_residual_l2_mean": math.nan,
            }
        )
    return summary


def analyze_run(
    spec: hess_eval.RunSpec,
    args: argparse.Namespace,
    *,
    output_root: Path,
    device: torch.device,
) -> dict:
    out_dir = output_root / spec.run_name / f"seed_{spec.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    context = hs_eval.build_eval_context(
        spec,
        device=device,
        batch_size=args.batch_size,
        dataset_root=args.dataset_root,
        no_download=args.no_download,
    )
    hs_params = context["model_cfg"].get("hard_sigmoid_param", {})
    if "v_min" not in hs_params or "v_max" not in hs_params:
        raise ValueError("Expected hard_sigmoid_param to contain v_min and v_max.")
    extra_minimizer = make_extra_minimizer(context, args.extra_iterations)

    network = context["network"]
    minimizer = context["minimizer"]
    energy_fn = context["energy_fn"]
    free_layers = context["free_layers"]
    output_layer = context["output_layer"]
    cost_fn = context["cost_fn"]
    test_loader = context["test_loader"]

    sample_rows: list[dict] = []
    arrays: dict[str, list[np.ndarray]] = {}
    sample_offset = 0
    for images, labels in test_loader:
        if args.max_test_samples is not None and sample_offset >= args.max_test_samples:
            break
        remaining = None if args.max_test_samples is None else args.max_test_samples - sample_offset
        if remaining is not None and images.shape[0] > remaining:
            images = images[:remaining]
            labels = labels[:remaining]
        images = images.to(device)
        labels = labels.to(device)

        network.set_input(images, reset=True)
        minimizer.compute_equilibrium()
        cost_fn.set_target(labels)

        hidden_initial = free_layers[0].state.detach().clone()
        output_initial = output_layer.state.detach().clone()
        initial_losses = cost_fn.eval().detach().clone()
        residual = layer_residuals(energy_fn, free_layers)

        extra_payload = None
        if extra_minimizer is not None:
            extra_minimizer.compute_equilibrium()
            cost_fn.set_target(labels)
            extra_scores = output_scores(output_layer.state.detach())
            extra_preds, extra_margins, _ = prediction_and_margin(extra_scores, labels)
            extra_correct = extra_preds.eq(labels)
            extra_residual = layer_residuals(energy_fn, free_layers)
            extra_payload = {
                "extra_prediction": extra_preds,
                "extra_correct": extra_correct.to(torch.float32),
                "extra_loss": cost_fn.eval().detach(),
                "extra_margin": extra_margins,
                "extra_prediction_changed": extra_preds.ne(
                    torch.argmax(output_scores(output_initial), dim=1)
                ).to(torch.float32),
                "extra_hidden_state_delta_l2": torch.linalg.vector_norm(
                    (free_layers[0].state.detach() - hidden_initial).flatten(start_dim=1),
                    ord=2,
                    dim=1,
                ),
                "extra_output_state_delta_l2": torch.linalg.vector_norm(
                    (output_layer.state.detach() - output_initial).flatten(start_dim=1),
                    ord=2,
                    dim=1,
                ),
                "extra_hidden_residual_l2": extra_residual["hidden_residual_l2"],
                "extra_output_residual_l2": extra_residual["output_residual_l2"],
            }

        arrays.setdefault("hidden_values", []).append(
            tensor_to_numpy(hidden_initial.flatten(start_dim=1)).reshape(-1)
        )
        sample_offset = append_batch(
            rows=sample_rows,
            arrays=arrays,
            spec=spec,
            sample_offset=sample_offset,
            labels=labels,
            hidden=hidden_initial,
            output=output_initial,
            losses=initial_losses,
            residual=residual,
            hs_params=hs_params,
            extra_payload=extra_payload,
        )

    if sample_offset == 0:
        raise ValueError("No test samples were evaluated.")

    values = concatenate(arrays)
    summary = summarize_arrays(values, extra_iterations=args.extra_iterations)
    summary.update(
        {
            "run_name": spec.run_name,
            "seed": spec.seed,
            "voltage_amp": spec.voltage_amp,
            "current_amp": spec.current_amp,
            "checkpoint_path": str(spec.checkpoint_path),
            "source_config_path": str(spec.source_config_path),
            "resolved_config_path": str(spec.resolved_config_path),
            "hard_sigmoid_param": hs_params,
            "inference_iterations": int(context["model_cfg"]["num_iterations_inference"]),
            "dataset_params": context["dataset_params"],
        }
    )

    sample_path = out_dir / "sample_diagnostics.csv"
    summary_path = out_dir / "summary.json"
    arrays_path = out_dir / "diagnostic_arrays.npz"
    config_path = out_dir / "config.json"
    write_csv(sample_path, sample_rows, SAMPLE_COLUMNS)
    np.savez_compressed(arrays_path, **values)
    summary["sample_diagnostics_path"] = str(sample_path)
    summary["arrays_path"] = str(arrays_path)
    summary["summary_path"] = str(summary_path)
    summary_path.write_text(json.dumps(hess_eval._json_sanitize(summary), indent=2, sort_keys=True) + "\n")
    config_path.write_text(
        json.dumps(
            hess_eval._json_sanitize(
                {
                    "input_root": str(args.input_root),
                    "output_root": str(output_root),
                    "checkpoint": args.checkpoint,
                    "device": str(device),
                    "batch_size": args.batch_size,
                    "max_test_samples": args.max_test_samples,
                    "extra_iterations": args.extra_iterations,
                    "dataset_root_override": args.dataset_root,
                }
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(
        f"[diagnostic] {spec.run_name} seed={spec.seed} "
        f"acc={summary['accuracy']:.4f} margin_mean={summary['margin_mean']:.4g} "
        f"active={summary['hidden_active_fraction_mean']:.4f} "
        f"hidden_resid={summary['hidden_residual_l2_mean']:.4g} "
        f"output_resid={summary['output_residual_l2_mean']:.4g}"
    )
    return summary


def collect_summaries(output_root: Path, specs: list[hess_eval.RunSpec]) -> list[dict]:
    rows = []
    for spec in specs:
        path = output_root / spec.run_name / f"seed_{spec.seed}" / "summary.json"
        if path.exists():
            rows.append(json.loads(path.read_text()))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--checkpoint", choices=("best", "final"), default="best")
    parser.add_argument("--run-name", action="append", choices=hess_eval.RUN_ORDER)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--extra-iterations", type=int, default=48)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested CUDA device {device}, but CUDA is not available.")

    specs = hess_eval.read_run_specs(
        input_root,
        checkpoint=args.checkpoint,
        run_names=set(args.run_name) if args.run_name else None,
        seeds=set(args.seeds) if args.seeds else None,
    )
    if args.summary_only:
        rows = collect_summaries(output_root, specs)
    else:
        print(f"[diagnostic] input_root={input_root}")
        print(f"[diagnostic] output_root={output_root}")
        print(f"[diagnostic] selected_runs={len(specs)} device={device}")
        rows = [
            analyze_run(spec, args, output_root=output_root, device=device)
            for spec in specs
        ]

    rows = sorted(
        rows,
        key=lambda row: (
            hess_eval.RUN_ORDER.index(row["run_name"])
            if row["run_name"] in hess_eval.RUN_ORDER
            else len(hess_eval.RUN_ORDER),
            int(row["seed"]),
        ),
    )
    write_csv(output_root / "summary.csv", rows, SUMMARY_COLUMNS)
    print(f"[done] summary={output_root / 'summary.csv'}")


if __name__ == "__main__":
    main()
