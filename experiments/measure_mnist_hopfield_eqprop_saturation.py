#!/usr/bin/env python3
"""Measure trained saturation for Hopfield EqProp Conv MNIST checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.function.cost import SquaredError  # noqa: E402
from model.function.network import Network  # noqa: E402
from model.hopfield.minimizer import FixedPointMinimizer  # noqa: E402
from model.hopfield.network import FlexibleConvHopfieldEnergy  # noqa: E402
from model.variable.layer import Layer  # noqa: E402
from model.variable.parameter import Bias, ConvWeight, DenseWeight  # noqa: E402


DATASETS = {
    "mnist": datasets.MNIST,
    "fashion_mnist": datasets.FashionMNIST,
}
OUTPUT_COLUMNS = [
    "conv_depth",
    "seed",
    "run_group",
    "lr_multiplier",
    "checkpoint_kind",
    "split",
    "num_samples",
    "accuracy",
    "loss",
    "all_low_saturation",
    "all_high_saturation",
    "all_total_saturation",
    "sample_total_saturation_mean",
    "sample_total_saturation_p50",
    "sample_total_saturation_p90",
    "layer_low_saturation",
    "layer_high_saturation",
    "layer_total_saturation",
    "best_test_accuracy",
    "final_test_accuracy",
    "best_epoch",
    "run_dir",
    "checkpoint_path",
]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _reset_name_counters() -> None:
    Layer._counter = 0
    Bias._counter = 0
    ConvWeight._counter = 0
    DenseWeight._counter = 0


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _discover_run_dirs(input_roots: list[str], run_dirs: list[str]) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for raw in run_dirs:
        path = Path(raw).expanduser().resolve()
        if path.name == "source_config.json":
            path = path.parent
        if not (path / "source_config.json").exists():
            raise FileNotFoundError(f"Expected source_config.json under run dir: {path}")
        if path not in seen:
            seen.add(path)
            paths.append(path)
    for raw in input_roots:
        root = Path(raw).expanduser().resolve()
        candidates = [root] if (root / "source_config.json").exists() else [
            path.parent for path in root.glob("**/source_config.json")
        ]
        for candidate in candidates:
            candidate = candidate.resolve()
            if candidate not in seen:
                seen.add(candidate)
                paths.append(candidate)
    return sorted(paths)


def _build_loader(args: argparse.Namespace, split: str) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.0,), std=(1.0,)),
        ]
    )
    dataset_root = os.path.expanduser(str(args.dataset_root))
    dataset_cls = DATASETS[args.dataset]
    dataset = dataset_cls(
        dataset_root,
        train=(split == "train"),
        transform=transform,
        download=not args.no_download,
    )
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)


def _build_context(
    source_config: dict,
    *,
    device: torch.device,
    checkpoint_path: Path | None,
) -> dict:
    seed = int(source_config.get("seed", 0))
    _set_seed(seed)
    _reset_name_counters()
    model_cfg = source_config["model"]
    energy_fn = FlexibleConvHopfieldEnergy(
        layer_shapes=[tuple(shape) for shape in model_cfg["layer_shapes"]],
        weight_gains=[float(value) for value in model_cfg["weight_gains"]],
        conv_pipeline=model_cfg["conv_pipeline"],
        activation=model_cfg.get("activation", "hard-sigmoid"),
        weight_init_mode=model_cfg.get("weight_init_mode", "kaiming_uniform"),
    )
    energy_fn.set_device(device)
    if checkpoint_path is not None:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Expected checkpoint at {checkpoint_path}")
        energy_fn.load(checkpoint_path)
    network = Network(energy_fn)
    cost_fn = SquaredError(energy_fn.layers()[-1])
    minimizer = FixedPointMinimizer(
        energy_fn,
        network.free_layers(),
        num_iterations=int(source_config.get("t1", 200)),
        mode=source_config.get("mode", "asynchronous"),
    )
    return {
        "energy_fn": energy_fn,
        "network": network,
        "cost_fn": cost_fn,
        "minimizer": minimizer,
    }


def _checkpoint_path(run_dir: Path, kind: str) -> Path | None:
    if kind == "init":
        return None
    name = "best_model.pt" if kind == "best" else "final_model.pt"
    return run_dir / name


@torch.no_grad()
def _measure_context(
    context: dict,
    loader: DataLoader,
    *,
    device: torch.device,
    max_samples: int,
    eps: float,
) -> dict:
    hidden_layers = context["energy_fn"].layers()[1:-1]
    low_counts = [0 for _ in hidden_layers]
    high_counts = [0 for _ in hidden_layers]
    totals = [0 for _ in hidden_layers]
    sample_saturations: list[np.ndarray] = []
    running_loss = 0.0
    running_correct = 0
    seen = 0

    for images, labels in loader:
        if seen >= max_samples:
            break
        take = min(images.shape[0], max_samples - seen)
        images = images[:take].to(device)
        labels = labels[:take].to(device)
        context["network"].set_input(images, reset=True)
        context["minimizer"].compute_equilibrium()
        context["cost_fn"].set_target(labels)
        batch_loss = float(context["cost_fn"].eval().mean().item())
        errors = context["cost_fn"].error_fn()
        running_loss += batch_loss * images.size(0)
        running_correct += int((~errors).sum().item())

        batch_low_per_sample = torch.zeros(images.size(0), device=device)
        batch_high_per_sample = torch.zeros(images.size(0), device=device)
        batch_total_units = 0
        for index, layer in enumerate(hidden_layers):
            state = layer.state.detach()
            low = state <= eps
            high = state >= 1.0 - eps
            low_counts[index] += int(low.sum().item())
            high_counts[index] += int(high.sum().item())
            totals[index] += int(state.numel())
            flat_low = low.flatten(start_dim=1).sum(dim=1)
            flat_high = high.flatten(start_dim=1).sum(dim=1)
            batch_low_per_sample += flat_low
            batch_high_per_sample += flat_high
            batch_total_units += int(np.prod(state.shape[1:]))
        if batch_total_units:
            batch_total = (batch_low_per_sample + batch_high_per_sample) / batch_total_units
            sample_saturations.append(batch_total.detach().cpu().numpy())
        seen += int(images.size(0))

    low_fracs = [low / total if total else math.nan for low, total in zip(low_counts, totals)]
    high_fracs = [high / total if total else math.nan for high, total in zip(high_counts, totals)]
    total_fracs = [
        (low + high) / total if total else math.nan
        for low, high, total in zip(low_counts, high_counts, totals)
    ]
    all_low = sum(low_counts) / sum(totals) if sum(totals) else math.nan
    all_high = sum(high_counts) / sum(totals) if sum(totals) else math.nan
    all_total = (sum(low_counts) + sum(high_counts)) / sum(totals) if sum(totals) else math.nan
    sample_values = (
        np.concatenate(sample_saturations).astype(np.float64)
        if sample_saturations
        else np.asarray([], dtype=np.float64)
    )
    return {
        "num_samples": seen,
        "accuracy": running_correct / seen if seen else math.nan,
        "loss": running_loss / seen if seen else math.nan,
        "all_low_saturation": all_low,
        "all_high_saturation": all_high,
        "all_total_saturation": all_total,
        "sample_total_saturation_mean": float(sample_values.mean()) if sample_values.size else math.nan,
        "sample_total_saturation_p50": float(np.quantile(sample_values, 0.50)) if sample_values.size else math.nan,
        "sample_total_saturation_p90": float(np.quantile(sample_values, 0.90)) if sample_values.size else math.nan,
        "layer_low_saturation": json.dumps(low_fracs),
        "layer_high_saturation": json.dumps(high_fracs),
        "layer_total_saturation": json.dumps(total_fracs),
    }


def _measure_run(
    run_dir: Path,
    *,
    args: argparse.Namespace,
    device: torch.device,
    loaders: dict[str, DataLoader],
) -> list[dict]:
    source_config = _load_json(run_dir / "source_config.json")
    metrics = _load_json(run_dir / "metrics.json")
    rows = []
    for checkpoint_kind in args.checkpoint_kind:
        checkpoint_path = _checkpoint_path(run_dir, checkpoint_kind)
        if checkpoint_path is not None and not checkpoint_path.exists():
            continue
        context = _build_context(
            source_config,
            device=device,
            checkpoint_path=checkpoint_path,
        )
        for split in args.split:
            result = _measure_context(
                context,
                loaders[split],
                device=device,
                max_samples=args.max_samples,
                eps=args.eps,
            )
            rows.append(
                {
                    "conv_depth": source_config.get("conv_depth", metrics.get("conv_depth", "")),
                    "seed": source_config.get("seed", metrics.get("seed", "")),
                    "run_group": source_config.get("run_group", metrics.get("run_group", "")),
                    "lr_multiplier": source_config.get("lr_multiplier", metrics.get("lr_multiplier", "")),
                    "checkpoint_kind": checkpoint_kind,
                    "split": split,
                    **result,
                    "best_test_accuracy": metrics.get("best_test_accuracy", math.nan),
                    "final_test_accuracy": metrics.get("final_test_accuracy", math.nan),
                    "best_epoch": metrics.get("best_epoch", ""),
                    "run_dir": str(run_dir),
                    "checkpoint_path": "" if checkpoint_path is None else str(checkpoint_path),
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", action="append", default=[])
    parser.add_argument("--run-dir", action="append", default=[])
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="mnist")
    parser.add_argument("--dataset-root", default=str(REPO_ROOT / "data"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--split", nargs="+", choices=("train", "test"), default=["train", "test"])
    parser.add_argument("--checkpoint-kind", nargs="+", choices=("init", "best", "final"), default=["init", "best", "final"])
    parser.add_argument("--max-samples", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--eps", type=float, default=1.0e-6)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()
    if not args.input_root and not args.run_dir:
        raise ValueError("Provide at least one --input-root or --run-dir.")
    return args


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    loaders = {split: _build_loader(args, split) for split in set(args.split)}
    run_dirs = _discover_run_dirs(args.input_root, args.run_dir)
    rows = []
    print(f"[discover] runs={len(run_dirs)} device={device} splits={args.split}")
    for index, run_dir in enumerate(run_dirs, start=1):
        print(f"[measure] {index}/{len(run_dirs)} {run_dir}", flush=True)
        rows.extend(_measure_run(run_dir, args=args, device=device, loaders=loaders))
    _write_csv(Path(args.output_csv), rows)
    print(f"[done] rows={len(rows)} output={args.output_csv}")


if __name__ == "__main__":
    main()
