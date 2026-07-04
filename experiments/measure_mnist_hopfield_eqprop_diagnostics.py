#!/usr/bin/env python3
"""Measure activation-aware Hopfield EqProp hidden-state diagnostics."""

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
from model.minimizer.minimizer import ParamUpdater  # noqa: E402
from model.variable.layer import Layer  # noqa: E402
from model.variable.parameter import Bias, ConvWeight, DenseWeight  # noqa: E402
from training.sgd import AugmentedFunction  # noqa: E402


DATASETS = {
    "mnist": datasets.MNIST,
    "fashion_mnist": datasets.FashionMNIST,
}
INPUT_PREPROCESSING = {
    "identity": ((0.0,), (1.0,)),
    "centered": ((0.5,), (0.5,)),
}
OUTPUT_COLUMNS = [
    "conv_depth",
    "seed",
    "run_group",
    "lr_multiplier",
    "activation",
    "input_preprocessing",
    "checkpoint_kind",
    "split",
    "num_samples",
    "near_low_fraction",
    "near_high_fraction",
    "near_total_fraction",
    "sample_near_total_mean",
    "sample_near_total_p50",
    "sample_near_total_p90",
    "layer_names",
    "layer_near_low_fraction",
    "layer_near_high_fraction",
    "layer_near_total_fraction",
    "layer_state_mean",
    "layer_state_mean_abs",
    "layer_state_std",
    "layer_delta_plus_rms",
    "layer_delta_centered_rms",
    "param_names",
    "param_grad_rms",
    "param_grad_mean_abs",
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
        candidates = (
            [root]
            if (root / "source_config.json").exists()
            else [path.parent for path in root.glob("**/source_config.json")]
        )
        for candidate in candidates:
            candidate = candidate.resolve()
            if candidate not in seen:
                seen.add(candidate)
                paths.append(candidate)
    return sorted(paths)


def _build_loader(
    args: argparse.Namespace,
    *,
    split: str,
    input_preprocessing: str,
) -> DataLoader:
    mean, std = INPUT_PREPROCESSING[input_preprocessing]
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
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


def _loader_for(
    loaders: dict[tuple[str, str], DataLoader],
    args: argparse.Namespace,
    *,
    split: str,
    input_preprocessing: str,
) -> DataLoader:
    key = (split, input_preprocessing)
    if key not in loaders:
        loaders[key] = _build_loader(args, split=split, input_preprocessing=input_preprocessing)
    return loaders[key]


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
    activation = model_cfg.get("activation", source_config.get("activation", "hard-sigmoid"))
    energy_fn = FlexibleConvHopfieldEnergy(
        layer_shapes=[tuple(shape) for shape in model_cfg["layer_shapes"]],
        weight_gains=[float(value) for value in model_cfg["weight_gains"]],
        conv_pipeline=model_cfg["conv_pipeline"],
        activation=activation,
        weight_init_mode=model_cfg.get("weight_init_mode", "kaiming_uniform"),
    )
    energy_fn.set_device(device)
    if checkpoint_path is not None:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Expected checkpoint at {checkpoint_path}")
        energy_fn.load(checkpoint_path)
    network = Network(energy_fn)
    cost_fn = SquaredError(energy_fn.layers()[-1])
    free_layers = network.free_layers()
    augmented_fn = AugmentedFunction(energy_fn, cost_fn)
    minimizer_free = FixedPointMinimizer(
        energy_fn,
        free_layers,
        num_iterations=int(source_config.get("t1", 200)),
        mode=source_config.get("mode", "asynchronous"),
    )
    minimizer_nudged = FixedPointMinimizer(
        augmented_fn,
        free_layers,
        num_iterations=int(source_config.get("t2", 10)),
        mode=source_config.get("mode", "asynchronous"),
    )
    return {
        "activation": activation,
        "energy_fn": energy_fn,
        "network": network,
        "cost_fn": cost_fn,
        "free_layers": free_layers,
        "augmented_fn": augmented_fn,
        "minimizer_free": minimizer_free,
        "minimizer_nudged": minimizer_nudged,
        "param_updaters": [ParamUpdater(param, energy_fn) for param in energy_fn.params()],
    }


def _checkpoint_path(run_dir: Path, kind: str) -> Path | None:
    if kind == "init":
        return None
    name = "best_model.pt" if kind == "best" else "final_model.pt"
    return run_dir / name


def _near_masks(
    state: torch.Tensor,
    *,
    activation: str,
    eps: float,
    tanh_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if activation == "tanh":
        return state <= -float(tanh_threshold), state >= float(tanh_threshold)
    if activation in ("hard-sigmoid", "sigmoid"):
        return state <= float(eps), state >= 1.0 - float(eps)
    raise ValueError(f"Expected activation `hard-sigmoid', `sigmoid', or `tanh', got {activation}.")


def _set_layer_states(layers: list, states: list[torch.Tensor]) -> None:
    for layer, state in zip(layers, states):
        layer.state = state.detach().clone()


def _json_float_list(values: list[float]) -> str:
    return json.dumps([None if math.isnan(value) else value for value in values])


def _measure_context(
    context: dict,
    loader: DataLoader,
    *,
    device: torch.device,
    max_samples: int,
    eps: float,
    tanh_threshold: float,
    beta: float,
) -> dict:
    hidden_layers = context["energy_fn"].layers()[1:-1]
    free_layers = context["free_layers"]
    activation = context["activation"]
    params = context["energy_fn"].params()
    param_updaters = context["param_updaters"]

    layer_count = len(hidden_layers)
    param_count = len(params)
    low_counts = [0 for _ in hidden_layers]
    high_counts = [0 for _ in hidden_layers]
    totals = [0 for _ in hidden_layers]
    state_sums = [0.0 for _ in hidden_layers]
    state_abs_sums = [0.0 for _ in hidden_layers]
    state_sq_sums = [0.0 for _ in hidden_layers]
    delta_plus_sq_sums = [0.0 for _ in hidden_layers]
    delta_centered_sq_sums = [0.0 for _ in hidden_layers]
    delta_counts = [0 for _ in hidden_layers]
    param_grad_rms_sums = [0.0 for _ in params]
    param_grad_abs_sums = [0.0 for _ in params]
    param_grad_weights = [0 for _ in params]
    sample_near_values: list[np.ndarray] = []
    seen = 0

    for images, labels in loader:
        if seen >= max_samples:
            break
        take = min(images.shape[0], max_samples - seen)
        images = images[:take].to(device)
        labels = labels[:take].to(device)

        context["network"].set_input(images, reset=True)
        context["minimizer_free"].compute_equilibrium()
        context["cost_fn"].set_target(labels)
        free_hidden_states = [layer.state.detach().clone() for layer in hidden_layers]
        free_layer_states = [layer.state.detach().clone() for layer in free_layers]

        batch_low_per_sample = torch.zeros(images.size(0), device=device)
        batch_high_per_sample = torch.zeros(images.size(0), device=device)
        batch_total_units = 0
        for index, state in enumerate(free_hidden_states):
            low, high = _near_masks(
                state,
                activation=activation,
                eps=eps,
                tanh_threshold=tanh_threshold,
            )
            low_counts[index] += int(low.sum().item())
            high_counts[index] += int(high.sum().item())
            totals[index] += int(state.numel())
            state_sums[index] += float(state.sum().item())
            state_abs_sums[index] += float(state.abs().sum().item())
            state_sq_sums[index] += float(state.square().sum().item())
            batch_low_per_sample += low.flatten(start_dim=1).sum(dim=1)
            batch_high_per_sample += high.flatten(start_dim=1).sum(dim=1)
            batch_total_units += int(np.prod(state.shape[1:]))

        if batch_total_units:
            batch_near = (batch_low_per_sample + batch_high_per_sample) / batch_total_units
            sample_near_values.append(batch_near.detach().cpu().numpy())

        if hasattr(context["augmented_fn"], "prepare_nudging"):
            context["augmented_fn"].prepare_nudging()

        _set_layer_states(free_layers, free_layer_states)
        context["augmented_fn"].nudging = float(beta)
        context["minimizer_nudged"].compute_equilibrium()
        plus_hidden_states = [layer.state.detach().clone() for layer in hidden_layers]
        plus_layer_states = [layer.state.detach().clone() for layer in free_layers]

        _set_layer_states(free_layers, free_layer_states)
        context["augmented_fn"].nudging = -float(beta)
        context["minimizer_nudged"].compute_equilibrium()
        minus_hidden_states = [layer.state.detach().clone() for layer in hidden_layers]
        minus_layer_states = [layer.state.detach().clone() for layer in free_layers]

        for index, (free_state, plus_state, minus_state) in enumerate(
            zip(free_hidden_states, plus_hidden_states, minus_hidden_states)
        ):
            delta_plus = plus_state - free_state
            delta_centered = (plus_state - minus_state) / (2.0 * float(beta))
            delta_plus_sq_sums[index] += float(delta_plus.square().sum().item())
            delta_centered_sq_sums[index] += float(delta_centered.square().sum().item())
            delta_counts[index] += int(delta_plus.numel())

        _set_layer_states(free_layers, minus_layer_states)
        grads_minus = [updater.grad().detach().clone() for updater in param_updaters]
        _set_layer_states(free_layers, plus_layer_states)
        grads_plus = [updater.grad().detach().clone() for updater in param_updaters]
        for index, (grad_minus, grad_plus) in enumerate(zip(grads_minus, grads_plus)):
            grad = (grad_plus - grad_minus) / (2.0 * float(beta))
            param_grad_rms_sums[index] += float(torch.sqrt(grad.square().mean()).item()) * take
            param_grad_abs_sums[index] += float(grad.abs().mean().item()) * take
            param_grad_weights[index] += take

        context["augmented_fn"].nudging = 0.0
        _set_layer_states(free_layers, free_layer_states)
        seen += int(images.size(0))

    low_fracs = [low / total if total else math.nan for low, total in zip(low_counts, totals)]
    high_fracs = [high / total if total else math.nan for high, total in zip(high_counts, totals)]
    total_fracs = [
        (low + high) / total if total else math.nan
        for low, high, total in zip(low_counts, high_counts, totals)
    ]
    state_means = [total_sum / total if total else math.nan for total_sum, total in zip(state_sums, totals)]
    state_abs_means = [
        total_sum / total if total else math.nan
        for total_sum, total in zip(state_abs_sums, totals)
    ]
    state_stds = []
    for total_sum, total_sq_sum, total in zip(state_sums, state_sq_sums, totals):
        if not total:
            state_stds.append(math.nan)
            continue
        mean = total_sum / total
        variance = max(total_sq_sum / total - mean * mean, 0.0)
        state_stds.append(math.sqrt(variance))

    delta_plus_rms = [
        math.sqrt(total_sum / count) if count else math.nan
        for total_sum, count in zip(delta_plus_sq_sums, delta_counts)
    ]
    delta_centered_rms = [
        math.sqrt(total_sum / count) if count else math.nan
        for total_sum, count in zip(delta_centered_sq_sums, delta_counts)
    ]
    param_grad_rms = [
        total_sum / weight if weight else math.nan
        for total_sum, weight in zip(param_grad_rms_sums, param_grad_weights)
    ]
    param_grad_mean_abs = [
        total_sum / weight if weight else math.nan
        for total_sum, weight in zip(param_grad_abs_sums, param_grad_weights)
    ]
    sample_values = (
        np.concatenate(sample_near_values).astype(np.float64)
        if sample_near_values
        else np.asarray([], dtype=np.float64)
    )
    all_total = sum(totals)
    return {
        "num_samples": seen,
        "near_low_fraction": sum(low_counts) / all_total if all_total else math.nan,
        "near_high_fraction": sum(high_counts) / all_total if all_total else math.nan,
        "near_total_fraction": (
            (sum(low_counts) + sum(high_counts)) / all_total if all_total else math.nan
        ),
        "sample_near_total_mean": float(sample_values.mean()) if sample_values.size else math.nan,
        "sample_near_total_p50": (
            float(np.quantile(sample_values, 0.50)) if sample_values.size else math.nan
        ),
        "sample_near_total_p90": (
            float(np.quantile(sample_values, 0.90)) if sample_values.size else math.nan
        ),
        "layer_names": json.dumps([layer.name for layer in hidden_layers]),
        "layer_near_low_fraction": _json_float_list(low_fracs),
        "layer_near_high_fraction": _json_float_list(high_fracs),
        "layer_near_total_fraction": _json_float_list(total_fracs),
        "layer_state_mean": _json_float_list(state_means),
        "layer_state_mean_abs": _json_float_list(state_abs_means),
        "layer_state_std": _json_float_list(state_stds),
        "layer_delta_plus_rms": _json_float_list(delta_plus_rms),
        "layer_delta_centered_rms": _json_float_list(delta_centered_rms),
        "param_names": json.dumps([param.name for param in params]),
        "param_grad_rms": _json_float_list(param_grad_rms),
        "param_grad_mean_abs": _json_float_list(param_grad_mean_abs),
    }


def _measure_run(
    run_dir: Path,
    *,
    args: argparse.Namespace,
    device: torch.device,
    loaders: dict[tuple[str, str], DataLoader],
) -> list[dict]:
    source_config = _load_json(run_dir / "source_config.json")
    metrics = _load_json(run_dir / "metrics.json")
    model_cfg = source_config.get("model", {})
    activation = model_cfg.get("activation", source_config.get("activation", "hard-sigmoid"))
    input_preprocessing = source_config.get("input_preprocessing", "identity")
    rows = []
    beta = float(source_config.get("beta", args.beta))
    for checkpoint_kind in args.checkpoint_kind:
        checkpoint_path = _checkpoint_path(run_dir, checkpoint_kind)
        if checkpoint_path is not None and not checkpoint_path.exists():
            continue
        context = _build_context(source_config, device=device, checkpoint_path=checkpoint_path)
        for split in args.split:
            loader = _loader_for(
                loaders,
                args,
                split=split,
                input_preprocessing=input_preprocessing,
            )
            result = _measure_context(
                context,
                loader,
                device=device,
                max_samples=args.max_samples,
                eps=args.eps,
                tanh_threshold=args.tanh_threshold,
                beta=beta,
            )
            rows.append(
                {
                    "conv_depth": source_config.get("conv_depth", metrics.get("conv_depth", "")),
                    "seed": source_config.get("seed", metrics.get("seed", "")),
                    "run_group": source_config.get("run_group", metrics.get("run_group", "")),
                    "lr_multiplier": source_config.get(
                        "lr_multiplier",
                        metrics.get("lr_multiplier", ""),
                    ),
                    "activation": activation,
                    "input_preprocessing": input_preprocessing,
                    "checkpoint_kind": checkpoint_kind,
                    "split": split,
                    **result,
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
    parser.add_argument(
        "--checkpoint-kind",
        nargs="+",
        choices=("init", "best", "final"),
        default=["init", "best", "final"],
    )
    parser.add_argument("--max-samples", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eps", type=float, default=1.0e-6)
    parser.add_argument("--tanh-threshold", type=float, default=0.99)
    parser.add_argument("--beta", type=float, default=0.4)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()
    if not args.input_root and not args.run_dir:
        raise ValueError("Provide at least one --input-root or --run-dir.")
    if not 0.0 < args.tanh_threshold < 1.0:
        raise ValueError(f"Expected --tanh-threshold in (0, 1), got {args.tanh_threshold}.")
    return args


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    loaders: dict[tuple[str, str], DataLoader] = {}
    run_dirs = _discover_run_dirs(args.input_root, args.run_dir)
    rows = []
    print(f"[discover] runs={len(run_dirs)} device={device} splits={args.split}")
    for index, run_dir in enumerate(run_dirs, start=1):
        print(f"[diagnostics] {index}/{len(run_dirs)} {run_dir}", flush=True)
        rows.extend(_measure_run(run_dir, args=args, device=device, loaders=loaders))
    _write_csv(Path(args.output_csv), rows)
    print(f"[done] rows={len(rows)} output={args.output_csv}")


if __name__ == "__main__":
    main()
