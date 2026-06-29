#!/usr/bin/env python3
"""Measure hidden saturation for trained Conv MNIST hard-sigmoid checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
LABS_DIR = REPO_ROOT / "labs"
for path in (REPO_ROOT, LABS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evaluate_mnist_bp_write_noise_sweep import RunRow, _build_eval_context  # noqa: E402


OUTPUT_COLUMNS = [
    "source_label",
    "run_dir",
    "checkpoint_kind",
    "conv_depth",
    "target_label",
    "target_saturation",
    "input_gain",
    "v_off",
    "run_name",
    "voltage_amp",
    "current_amp",
    "seed",
    "split",
    "num_batches",
    "num_samples",
    "accuracy_mean",
    "loss_mean",
    "first_hidden_saturation_mean",
    "all_hidden_saturation_mean",
    "layer_saturations_mean",
    "first_hidden_abs_mean_mean",
    "first_hidden_abs_p90_mean",
    "first_hidden_abs_max_mean",
    "best_test_accuracy",
    "final_test_accuracy",
    "best_epoch",
]


@dataclass(frozen=True)
class DiscoveredRun:
    source_label: str
    run: RunRow
    checkpoint_kind: str
    metrics: dict
    model_cfg: dict


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _path_part(run_dir: Path, prefix: str) -> str:
    for part in reversed(run_dir.parts):
        if part.startswith(prefix):
            return part[len(prefix) :]
    return ""


def _label_to_float(value: str) -> float:
    if not value:
        return math.nan
    try:
        return float(value.replace("p", ".").replace("m", "-"))
    except ValueError:
        return math.nan


def _target_from_label(label: str) -> float:
    if label.startswith("sat"):
        try:
            return int(label[3:]) / 100.0
        except ValueError:
            return math.nan
    return math.nan


def _checkpoint_path(run_dir: Path, checkpoint_kind: str) -> Path | None:
    if checkpoint_kind == "final":
        path = run_dir / "final_model.pt"
        return path if path.exists() else None
    if checkpoint_kind == "best":
        path = run_dir / "best_model.pt"
        return path if path.exists() else None
    if checkpoint_kind == "auto":
        final_path = run_dir / "final_model.pt"
        if final_path.exists():
            return final_path
        best_path = run_dir / "best_model.pt"
        return best_path if best_path.exists() else None
    raise ValueError(f"Expected checkpoint_kind final/best/auto, got {checkpoint_kind!r}.")


def _combined_model_cfg(source_config: dict) -> dict:
    model_key = source_config.get("lab", {}).get("model_key", "mnist_bp_conv_amp")
    return {
        **source_config.get("model_base", {}),
        **source_config.get("model_overrides", {}).get(model_key, {}),
    }


def _discover_runs(input_roots: list[Path], checkpoint_kind: str) -> list[DiscoveredRun]:
    runs: list[DiscoveredRun] = []
    seen: set[Path] = set()
    for input_root in input_roots:
        input_root = input_root.expanduser().resolve()
        for source_config_path in sorted(input_root.glob("**/source_config.json")):
            run_dir = source_config_path.parent.resolve()
            if run_dir in seen:
                continue
            seen.add(run_dir)
            checkpoint_path = _checkpoint_path(run_dir, checkpoint_kind)
            if checkpoint_path is None:
                continue
            selected_kind = "final" if checkpoint_path.name == "final_model.pt" else "best"
            source_config = _load_json(source_config_path)
            model_cfg = _combined_model_cfg(source_config)
            metrics = _load_json(run_dir / "metrics.json")
            seed = int(source_config.get("seed", metrics.get("seed", 0)))
            clean_accuracy = float(
                metrics.get(
                    "final_test_accuracy" if selected_kind == "final" else "best_test_accuracy",
                    math.nan,
                )
            )
            runs.append(
                DiscoveredRun(
                    source_label=input_root.name,
                    checkpoint_kind=selected_kind,
                    metrics=metrics,
                    model_cfg=model_cfg,
                    run=RunRow(
                        non_linearity=run_dir.parent.parent.name,
                        run_name=run_dir.parent.name,
                        seed=seed,
                        voltage_amp=float(model_cfg.get("voltage_amp", math.nan)),
                        current_amp=float(model_cfg.get("current_amp", math.nan)),
                        checkpoint_path=checkpoint_path,
                        clean_accuracy=clean_accuracy,
                        run_dir=run_dir,
                    ),
                )
            )
    return sorted(runs, key=lambda item: str(item.run.run_dir))


def _tensor_stats(tensor: torch.Tensor) -> dict:
    flat_abs = tensor.detach().abs().flatten()
    if flat_abs.numel() == 0:
        return {"abs_mean": math.nan, "abs_p90": math.nan, "abs_max": math.nan}
    p90 = torch.quantile(flat_abs, torch.tensor(0.9, device=flat_abs.device))
    return {
        "abs_mean": float(flat_abs.mean().item()),
        "abs_p90": float(p90.item()),
        "abs_max": float(flat_abs.max().item()),
    }


def _mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else math.nan


def _measure_one(
    discovered: DiscoveredRun,
    *,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    context = _build_eval_context(
        discovered.run,
        device=device,
        eval_batch_size=args.batch_size,
        no_download=args.no_download,
        inference_iterations_override=args.inference_iterations,
        dataset_root_override=args.dataset_root,
    )
    loader = context["train_loader"] if args.split == "train" else context["test_loader"]
    layers = context["energy_fn"].layers()
    hidden_layers = layers[1:-1]
    v_off = abs(float(discovered.model_cfg.get("hard_sigmoid_param", {}).get("v_max", math.nan)))

    batch_accs: list[float] = []
    batch_losses: list[float] = []
    first_hidden_sats: list[float] = []
    all_hidden_sats: list[float] = []
    first_abs_mean: list[float] = []
    first_abs_p90: list[float] = []
    first_abs_max: list[float] = []
    layer_saturation_sums = [0.0 for _ in hidden_layers]
    layer_saturation_counts = [0 for _ in hidden_layers]
    num_samples = 0

    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(loader):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
            images = images.to(device)
            labels = labels.to(device)
            context["network"].set_input(images, reset=True)
            context["minimizer"].compute_equilibrium()
            context["cost_fn"].set_target(labels)
            loss = float(context["cost_fn"].eval().mean().item())
            errors = context["cost_fn"].error_fn()
            accuracy = float((~errors).float().mean().item())
            batch_accs.append(accuracy)
            batch_losses.append(loss)
            num_samples += int(images.shape[0])

            saturated_total = 0.0
            hidden_total = 0
            per_layer = []
            for layer_index, layer in enumerate(hidden_layers):
                state = layer.state.detach()
                count = int(state.numel())
                sat = float((state.abs() > v_off).float().mean().item()) if math.isfinite(v_off) else math.nan
                per_layer.append(sat)
                if math.isfinite(sat):
                    saturated_total += sat * count
                    hidden_total += count
                    layer_saturation_sums[layer_index] += sat
                    layer_saturation_counts[layer_index] += 1
            if per_layer:
                first_hidden_sats.append(per_layer[0])
            all_hidden_sats.append(saturated_total / hidden_total if hidden_total else math.nan)
            if hidden_layers:
                stats = _tensor_stats(hidden_layers[0].state)
                first_abs_mean.append(stats["abs_mean"])
                first_abs_p90.append(stats["abs_p90"])
                first_abs_max.append(stats["abs_max"])

    layer_saturations_mean = [
        layer_saturation_sums[index] / layer_saturation_counts[index]
        if layer_saturation_counts[index]
        else math.nan
        for index in range(len(hidden_layers))
    ]
    conv_pipeline = discovered.model_cfg.get("conv_pipeline") or []
    target_label = _path_part(discovered.run.run_dir, "target_")
    return {
        "source_label": discovered.source_label,
        "run_dir": str(discovered.run.run_dir),
        "checkpoint_kind": discovered.checkpoint_kind,
        "conv_depth": len(conv_pipeline),
        "target_label": target_label,
        "target_saturation": _target_from_label(target_label),
        "input_gain": float(discovered.model_cfg.get("input_gain", _label_to_float(_path_part(discovered.run.run_dir, "input_gain_")))),
        "v_off": v_off,
        "run_name": discovered.run.run_name,
        "voltage_amp": discovered.run.voltage_amp,
        "current_amp": discovered.run.current_amp,
        "seed": discovered.run.seed,
        "split": args.split,
        "num_batches": len(batch_accs),
        "num_samples": num_samples,
        "accuracy_mean": _mean(batch_accs),
        "loss_mean": _mean(batch_losses),
        "first_hidden_saturation_mean": _mean(first_hidden_sats),
        "all_hidden_saturation_mean": _mean(all_hidden_sats),
        "layer_saturations_mean": json.dumps(layer_saturations_mean),
        "first_hidden_abs_mean_mean": _mean(first_abs_mean),
        "first_hidden_abs_p90_mean": _mean(first_abs_p90),
        "first_hidden_abs_max_mean": _mean(first_abs_max),
        "best_test_accuracy": discovered.metrics.get("best_test_accuracy", math.nan),
        "final_test_accuracy": discovered.metrics.get("final_test_accuracy", math.nan),
        "best_epoch": discovered.metrics.get("best_epoch", ""),
    }


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", action="append", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--dataset-root", default=str(REPO_ROOT / "data"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--checkpoint-kind", choices=("auto", "best", "final"), default="final")
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-batches", type=int, default=16)
    parser.add_argument("--inference-iterations", type=int, default=None)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--run-name", action="append", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    run_names = set(args.run_name or [])
    runs = _discover_runs([Path(value) for value in args.input_root], args.checkpoint_kind)
    if run_names:
        runs = [item for item in runs if item.run.run_name in run_names]
    if args.limit is not None:
        runs = runs[: int(args.limit)]
    print(f"[discover] runs={len(runs)} device={device} split={args.split} max_batches={args.max_batches}")
    if args.dry_run:
        for item in runs:
            print(f"[dry-run] {item.run.run_name} {item.run.run_dir}")
        return

    rows = []
    for index, item in enumerate(runs, start=1):
        print(f"[measure] {index}/{len(runs)} {item.run.run_name} {item.run.run_dir}")
        rows.append(_measure_one(item, args=args, device=device))
    _write_rows(Path(args.output_csv), rows)
    print(f"[done] wrote {args.output_csv}")


if __name__ == "__main__":
    main()
