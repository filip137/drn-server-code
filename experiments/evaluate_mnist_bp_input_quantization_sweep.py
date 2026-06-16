#!/usr/bin/env python3
"""Evaluate clean-trained MNIST DRNs under input quantization."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
LABS_DIR = REPO_ROOT / "labs"
for path in (REPO_ROOT, LABS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.evaluate_mnist_bp_write_noise_sweep import (  # noqa: E402
    _build_eval_context,
    _json_sanitize,
    _mean_margin,
    _read_summary_rows,
    _scores,
    _select_shard,
    _saturation_fraction,
)


DEFAULT_INPUT_ROOT = (
    REPO_ROOT / "results" / "mnist_bp_amplification_sweep_drn_xs_10epoch_legacy_preproc"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "mnist_bp_input_quantization_sweep"
RAW_COLUMNS = [
    "run_name",
    "voltage_amp",
    "current_amp",
    "training_seed",
    "checkpoint_kind",
    "checkpoint_path",
    "input_quantization_bits",
    "quantization_mode",
    "quant_min",
    "quant_max",
    "clean_checkpoint_accuracy",
    "accuracy",
    "accuracy_drop_from_clean_checkpoint",
    "test_loss",
    "mean_logit_margin",
    "fraction_saturated_nodes",
    "num_examples",
]
SUMMARY_COLUMNS = [
    "run_name",
    "voltage_amp",
    "current_amp",
    "input_quantization_bits",
    "quantization_mode",
    "num_training_seeds",
    "clean_checkpoint_accuracy_mean",
    "mean_accuracy",
    "std_accuracy",
    "mean_accuracy_drop_from_clean_checkpoint",
    "mean_test_loss",
    "mean_logit_margin",
    "mean_fraction_saturated_nodes",
]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_sanitize(payload), indent=2, sort_keys=True))


def _write_rows(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({column: _json_sanitize(row.get(column, "")) for column in columns})


def _preprocessed_mnist_range(dataset_params: dict) -> tuple[float, float]:
    normalize = bool(dataset_params.get("normalize", False))
    if not normalize:
        return 0.0, 1.0
    mean = float(dataset_params.get("normalize_mean", 0.0))
    std = float(dataset_params["normalize_std"])
    scale = float(dataset_params.get("normalize_scale", 1.0))
    return ((0.0 - mean) / std * scale, (1.0 - mean) / std * scale)


def _quantize_inputs(
    images: torch.Tensor,
    bits: str,
    *,
    quant_min: float,
    quant_max: float,
) -> torch.Tensor:
    bit_label = bits.lower()
    if bit_label in {"clean", "none", "float", "fp32"}:
        return images
    bit_depth = int(bits)
    if bit_depth < 1:
        raise ValueError(f"Expected input quantization bits >= 1; got {bits!r}.")
    levels = (1 << bit_depth) - 1
    if levels <= 0:
        return images
    qmin = torch.as_tensor(float(quant_min), dtype=images.dtype, device=images.device)
    qmax = torch.as_tensor(float(quant_max), dtype=images.dtype, device=images.device)
    scaled = (images.clamp(qmin.item(), qmax.item()) - qmin) / (qmax - qmin)
    quantized = torch.round(scaled * levels) / levels
    return quantized * (qmax - qmin) + qmin


def _evaluate_quantized(context: dict, *, bits: str, max_eval_batches: int | None = None) -> dict:
    network = context["network"]
    minimizer = context["minimizer"]
    cost_fn = context["cost_fn"]
    output_layer = context["output_layer"]
    test_loader = context["test_loader"]
    device = context["base_states"][0].device
    num_classes = context["num_classes"]
    quant_min, quant_max = _preprocessed_mnist_range(context["dataset_params"])

    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    margin_sum = 0.0
    saturated_sum = 0.0
    saturated_weight = 0

    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(test_loader):
            if max_eval_batches is not None and batch_index >= max_eval_batches:
                break
            images = images.to(device)
            labels = labels.to(device)
            images = _quantize_inputs(images, bits, quant_min=quant_min, quant_max=quant_max)
            network.set_input(images, reset=True)
            minimizer.compute_equilibrium()
            cost_fn.set_target(labels)

            batch_loss = cost_fn.eval().mean().item()
            errors = cost_fn.error_fn()
            batch_correct = int((~errors).sum().item())
            output_scores = _scores(output_layer.state.detach(), num_classes)
            margins = _mean_margin(output_scores, labels)
            saturation, saturation_count = _saturation_fraction(network.free_layers())

            batch_size = int(images.size(0))
            total_loss += batch_loss * batch_size
            total_correct += batch_correct
            total_seen += batch_size
            margin_sum += float(margins.sum().item())
            if saturation_count:
                saturated_sum += saturation * saturation_count
                saturated_weight += saturation_count

    if total_seen == 0:
        raise ValueError("Evaluation saw zero examples.")
    return {
        "accuracy": total_correct / total_seen,
        "loss": total_loss / total_seen,
        "mean_logit_margin": margin_sum / total_seen,
        "fraction_saturated_nodes": (
            saturated_sum / saturated_weight if saturated_weight else float("nan")
        ),
        "num_examples": total_seen,
        "quant_min": quant_min,
        "quant_max": quant_max,
    }


def write_summaries(output_root: Path) -> None:
    raw_paths = sorted(output_root.glob("raw_results*.csv"))
    if not raw_paths:
        return
    rows = []
    for raw_path in raw_paths:
        rows.extend(csv.DictReader(raw_path.open()))

    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (
            row["run_name"],
            row["voltage_amp"],
            row["current_amp"],
            row["input_quantization_bits"],
            row["quantization_mode"],
        )
        grouped.setdefault(key, []).append(row)

    summary_rows = []
    for key, values in sorted(grouped.items()):
        run_name, voltage_amp, current_amp, bits, quantization_mode = key
        acc = np.asarray([float(value["accuracy"]) for value in values], dtype=float)
        clean = np.asarray(
            [float(value["clean_checkpoint_accuracy"]) for value in values], dtype=float
        )
        loss = np.asarray([float(value["test_loss"]) for value in values], dtype=float)
        margin = np.asarray([float(value["mean_logit_margin"]) for value in values], dtype=float)
        saturation = np.asarray(
            [float(value["fraction_saturated_nodes"]) for value in values], dtype=float
        )
        summary_rows.append(
            {
                "run_name": run_name,
                "voltage_amp": voltage_amp,
                "current_amp": current_amp,
                "input_quantization_bits": bits,
                "quantization_mode": quantization_mode,
                "num_training_seeds": len({value["training_seed"] for value in values}),
                "clean_checkpoint_accuracy_mean": float(np.mean(clean)),
                "mean_accuracy": float(np.mean(acc)),
                "std_accuracy": float(np.std(acc, ddof=1)) if len(acc) > 1 else 0.0,
                "mean_accuracy_drop_from_clean_checkpoint": float(np.mean(clean) - np.mean(acc)),
                "mean_test_loss": float(np.mean(loss)),
                "mean_logit_margin": float(np.mean(margin)),
                "mean_fraction_saturated_nodes": float(np.nanmean(saturation)),
            }
        )

    with (output_root / "summary_by_run_bits.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default=str(DEFAULT_INPUT_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--device", default=None)
    parser.add_argument("--checkpoint-kind", choices=("best", "final"), default="best")
    parser.add_argument("--bits", nargs="+", default=["clean", "8", "4"])
    parser.add_argument(
        "--quantization-mode",
        choices=("preprocessed_raw_pixel_equivalent",),
        default="preprocessed_raw_pixel_equivalent",
        help=(
            "Uniformly quantize the post-preprocessing MNIST tensor over the range "
            "corresponding to raw pixels [0, 1], before signed-channel duplication and input_gain."
        ),
    )
    parser.add_argument("--run-name", action="append", help="Restrict to one or more run names.")
    parser.add_argument("--training-seeds", type=int, nargs="+")
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--max-eval-batches", type=int)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--raw-results-name", default=None)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    raw_results_name = args.raw_results_name or f"raw_results_shard_{args.shard_index}.csv"
    raw_results_path = output_root / raw_results_name

    if args.summary_only:
        write_summaries(output_root)
        print(f"[summary] wrote summaries under {output_root}")
        return

    requested_device = args.device
    if requested_device is not None:
        device = torch.device(requested_device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"Requested CUDA device {requested_device!r}, but CUDA is unavailable.")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = _read_summary_rows(
        input_root,
        checkpoint_kind=args.checkpoint_kind,
        run_names=set(args.run_name) if args.run_name else None,
        training_seeds=set(args.training_seeds) if args.training_seeds else None,
    )
    selected = _select_shard(rows, args.num_shards, args.shard_index)
    config_payload = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "checkpoint_kind": args.checkpoint_kind,
        "bits": [str(value) for value in args.bits],
        "quantization_mode": args.quantization_mode,
        "eval_batch_size": args.eval_batch_size,
        "device": str(device),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "selected_runs": [f"{row.run_name}/seed_{row.seed}" for row in selected],
    }
    _write_json(output_root / f"config_shard_{args.shard_index}.json", config_payload)

    print(f"[input-quant] input_root={input_root}")
    print(f"[input-quant] output_root={output_root}")
    print(f"[input-quant] selected_runs={len(selected)}/{len(rows)} shard={args.shard_index}/{args.num_shards}")
    if args.dry_run:
        for row in selected:
            print(f"[dry-run] {row.run_name} seed={row.seed} checkpoint={row.checkpoint_path}")
        return

    raw_rows = []
    for run in selected:
        print(f"[run] {run.run_name} seed={run.seed} checkpoint={run.checkpoint_path}")
        context = _build_eval_context(
            run,
            device=device,
            eval_batch_size=args.eval_batch_size,
            no_download=args.no_download,
        )
        for bits in args.bits:
            bits = str(bits)
            metrics = _evaluate_quantized(
                context,
                bits=bits,
                max_eval_batches=args.max_eval_batches,
            )
            raw_rows.append(
                {
                    "run_name": run.run_name,
                    "voltage_amp": run.voltage_amp,
                    "current_amp": run.current_amp,
                    "training_seed": run.seed,
                    "checkpoint_kind": args.checkpoint_kind,
                    "checkpoint_path": str(run.checkpoint_path),
                    "input_quantization_bits": bits,
                    "quantization_mode": args.quantization_mode,
                    "quant_min": metrics["quant_min"],
                    "quant_max": metrics["quant_max"],
                    "clean_checkpoint_accuracy": run.clean_accuracy,
                    "accuracy": metrics["accuracy"],
                    "accuracy_drop_from_clean_checkpoint": run.clean_accuracy - metrics["accuracy"],
                    "test_loss": metrics["loss"],
                    "mean_logit_margin": metrics["mean_logit_margin"],
                    "fraction_saturated_nodes": metrics["fraction_saturated_nodes"],
                    "num_examples": metrics["num_examples"],
                }
            )
            print(
                "[result] "
                f"{run.run_name} seed={run.seed} bits={bits} "
                f"accuracy={metrics['accuracy']:.4f}"
            )
        _write_rows(raw_results_path, RAW_COLUMNS, raw_rows)
        raw_rows.clear()

    write_summaries(output_root)
    print(f"[done] wrote {raw_results_path}")


if __name__ == "__main__":
    main()
