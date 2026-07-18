#!/usr/bin/env python3
"""Deprecated compatibility runner for historical MNIST Conv sweeps.

New paper-facing work must use ``python -m experiments.mnist_conv sweep`` with
a complete canonical JSON specification. This file remains only to reproduce
historical diagnostic commands while their source configs are inventoried.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LABS_DIR = REPO_ROOT / "labs"
for path in (REPO_ROOT, LABS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from labs.mnist_train import train_mnist_conv  # noqa: E402
from labs.datasets import AFFINE_PRESETS, affine_config_from_preset  # noqa: E402


MODEL_KEY = "mnist_bp_conv_amp"
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "results" / "mnist_bp_conv1_amplification_sweep_64ch_s2_valid"
)
MINIMIZER_CONFIG_HELP = (
    "Path to a JSON object for model_base.minimizer. "
    "Example: labs/configs/mnist_minimizer_fixed_iterations.json"
)
DEFAULT_SEEDS = [0, 1, 2]
DEFAULT_NON_LINEARITIES = ["hard_sigmoid", "perfect_diode"]
AMPLIFICATION_GRID = [
    ("mnist_bp_amp_v1_c1", 1.0, 1.0),
    ("mnist_bp_amp_v2_c1", 2.0, 1.0),
    ("mnist_bp_amp_v4_c1", 4.0, 1.0),
    ("mnist_bp_amp_v1_c2", 1.0, 2.0),
    ("mnist_bp_amp_v1_c4", 1.0, 4.0),
    ("mnist_bp_amp_v2_c2", 2.0, 2.0),
]
LEGACY_AMPLIFICATION_GRID = [
    ("mnist_bp_amp_v4_c0p25", 4.0, 0.25),
]
SUMMARY_COLUMNS = [
    "non_linearity",
    "run_name",
    "seed",
    "voltage_amp",
    "current_amp",
    "conv_depth",
    "conv_pipeline",
    "input_gain",
    "num_iterations_inference",
    "num_iterations_training",
    "learning_rate",
    "best_test_accuracy",
    "final_test_accuracy",
    "best_epoch",
    "final_train_loss",
    "final_test_loss",
    "checkpoint_path",
    "weights_best_path",
    "weights_final_path",
]
SUMMARY_BY_AMP_COLUMNS = [
    "non_linearity",
    "run_name",
    "voltage_amp",
    "current_amp",
    "num_seeds",
    "mean_best_test_accuracy",
    "std_best_test_accuracy",
    "mean_final_test_accuracy",
    "std_final_test_accuracy",
    "mean_final_train_loss",
    "mean_final_test_loss",
]
REQUIRED_RUN_FILES = [
    "final_model.pt",
    "best_model.pt",
    "weights_final.npz",
    "weights_best.npz",
    "metrics.json",
    "accuracy_train.npy",
    "accuracy_test.npy",
    "loss_train.npy",
    "loss_test.npy",
    "config.json",
]


@dataclass(frozen=True)
class RunSpec:
    non_linearity: str
    run_name: str
    seed: int
    voltage_amp: float
    current_amp: float

    @property
    def seed_dir_name(self) -> str:
        return f"seed_{self.seed}"


def _fmt_amp(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return str(value).replace(".", "p")


def _conv_spatial(size: int, kernel: int, stride: int, padding: int) -> int:
    return (size + 2 * padding - (kernel - 1) - 1) // stride + 1


def _expanded_layer_ints(
    values: list[int] | None,
    *,
    fallback: int,
    depth: int,
    name: str,
) -> list[int]:
    raw = [int(fallback)] if values is None else [int(value) for value in values]
    if len(raw) == 1:
        return raw * depth
    if len(raw) != depth:
        raise ValueError(
            f"Expected --{name} to contain 1 value or {depth} values "
            f"for this conv depth, got {len(raw)}."
        )
    return raw


def _conv_geometry(args: argparse.Namespace) -> tuple[list[int], list[int]]:
    return (
        _expanded_layer_ints(
            args.strides, fallback=args.stride, depth=args.conv_depth, name="strides"
        ),
        _expanded_layer_ints(
            args.paddings, fallback=args.padding, depth=args.conv_depth, name="paddings"
        ),
    )


def _iteration_counts(args: argparse.Namespace) -> tuple[int, int]:
    inference = (
        int(args.num_iterations_inference)
        if args.num_iterations_inference is not None
        else int(args.num_iterations)
    )
    training = (
        int(args.num_iterations_training)
        if args.num_iterations_training is not None
        else (inference if args.num_iterations_inference is not None else int(args.num_iterations))
    )
    return inference, training


def _affine_config_from_args(args: argparse.Namespace) -> dict:
    return affine_config_from_preset(
        args.affine_preset,
        degrees=args.affine_degrees,
        translate=args.affine_translate,
        scale=args.affine_scale,
        shear=args.affine_shear,
        seed=args.affine_seed,
    )


def _architecture(args: argparse.Namespace) -> tuple[list[list[int]], list[dict]]:
    channels = [int(value) for value in args.conv_channels]
    if args.conv_depth < 1:
        raise ValueError(f"Expected --conv-depth >= 1, got {args.conv_depth}.")
    if len(channels) < args.conv_depth:
        raise ValueError(
            f"Expected at least {args.conv_depth} --conv-channels values, got {channels}."
        )

    layer_shapes: list[list[int]] = [[2, 28, 28]]
    conv_pipeline: list[dict] = []
    height = 28
    width = 28
    strides, paddings = _conv_geometry(args)
    for index in range(args.conv_depth):
        stride = strides[index]
        padding = paddings[index]
        height = _conv_spatial(height, args.kernel_size, stride, padding)
        width = _conv_spatial(width, args.kernel_size, stride, padding)
        if height <= 0 or width <= 0:
            raise ValueError(
                "Convolution geometry produced non-positive spatial dimensions: "
                f"depth={index + 1}, height={height}, width={width}."
            )
        layer_shapes.append([channels[index], height, width])
        conv_pipeline.append(
            {
                "kernel": [args.kernel_size, args.kernel_size],
                "stride": stride,
                "padding": padding,
                "mode": "convolution",
            }
        )
    layer_shapes.append([args.output_dim])
    return layer_shapes, conv_pipeline


def _num_energy_params(args: argparse.Namespace) -> int:
    # conv weights + dense readout weight + one bias per conv/free layer.
    trainable_v_off = args.conv_depth if args.trainable_hard_sigmoid_v_off else 0
    trainable_amplification = 2 if args.trainable_amplification else 0
    return args.conv_depth + 1 + args.conv_depth + trainable_v_off + trainable_amplification


def _expanded_learning_rate(args: argparse.Namespace) -> list[float]:
    expected = _num_energy_params(args)
    values = args.learning_rate or [0.006]
    if len(values) == 1:
        return [float(values[0])] * expected
    if len(values) != expected:
        raise ValueError(
            f"Expected --learning-rate to contain 1 value or {expected} values "
            f"for this conv depth, got {len(values)}."
        )
    return [float(value) for value in values]


def _expanded_weight_gains(args: argparse.Namespace) -> list[float]:
    expected = args.conv_depth + 1
    values = args.weight_gains or [1.0]
    if len(values) == 1:
        return [float(values[0])] * expected
    if len(values) != expected:
        raise ValueError(
            f"Expected --weight-gains to contain 1 value or {expected} values "
            f"for this conv depth, got {len(values)}."
        )
    return [float(value) for value in values]


def _expanded_hard_sigmoid_v_off(args: argparse.Namespace) -> float | list[float]:
    values = args.hard_sigmoid_v_off
    if values is None:
        if abs(args.hard_sigmoid_v_min + args.hard_sigmoid_v_max) > 1e-9:
            raise ValueError(
                "Expected --hard-sigmoid-v-min and --hard-sigmoid-v-max to be "
                "symmetric when --hard-sigmoid-v-off is omitted."
            )
        values = [0.5 * (args.hard_sigmoid_v_max - args.hard_sigmoid_v_min)]
    if len(values) == 1:
        return float(values[0])
    if len(values) != args.conv_depth:
        raise ValueError(
            f"Expected --hard-sigmoid-v-off to contain 1 value or {args.conv_depth} "
            f"values for this conv depth, got {len(values)}."
        )
    return [float(value) for value in values]


def _build_config(args: argparse.Namespace, spec: RunSpec) -> dict:
    layer_shapes, conv_pipeline = _architecture(args)
    learning_rate = _expanded_learning_rate(args)
    weight_gains = _expanded_weight_gains(args)
    inference_iterations, training_iterations = _iteration_counts(args)
    affine_config = _affine_config_from_args(args)
    dataset_factory = (
        "labs.datasets.AffineMnistDataset"
        if affine_config.get("enabled")
        else "labs.datasets.MnistDataset"
    )
    dataset_params = {
        "name": "mnist",
        "batch_size": args.batch_size,
        "root": os.path.expanduser(args.dataset_root),
        "train": True,
        "download": not args.no_download,
        "normalize": True,
        "normalize_mean": args.normalize_mean,
        "normalize_std": args.normalize_std,
        "normalize_scale": args.normalize_scale,
    }
    if affine_config.get("enabled"):
        dataset_params["affine_config"] = affine_config
    model_key = args.model_key
    hard_sigmoid_param = {
        "g_on": args.hard_sigmoid_g_on,
        "g_off": args.hard_sigmoid_g_off,
    }
    if args.trainable_hard_sigmoid_v_off or args.hard_sigmoid_v_off is not None:
        hard_sigmoid_param["v_off"] = _expanded_hard_sigmoid_v_off(args)
    else:
        hard_sigmoid_param["v_min"] = args.hard_sigmoid_v_min
        hard_sigmoid_param["v_max"] = args.hard_sigmoid_v_max
    if args.trainable_hard_sigmoid_v_off:
        hard_sigmoid_param["trainable_v_off"] = True
        hard_sigmoid_param["v_off_min"] = args.hard_sigmoid_v_off_min
        if args.hard_sigmoid_v_off_max is not None:
            hard_sigmoid_param["v_off_max"] = args.hard_sigmoid_v_off_max

    return {
        "lab": {
            "model_key": model_key,
            "dataset_key": "mnist",
            "epochs": args.epochs,
            "plots_dir": "plots/mnist_bp_conv_amplification_sweep",
        },
        "input_mode": "train",
        "training_algorithm": "BP",
        "seed": spec.seed,
        "lr": learning_rate,
        "beta": float(args.beta),
        "lr_decay": float(args.lr_decay),
        "log_interval": args.log_interval,
        "max_batches": args.max_batches,
        "max_test_batches": args.max_test_batches,
        "datasets": {
            "mnist": {
                "factory": dataset_factory,
                "params": dataset_params,
            },
        },
        "model_base": {
            "weight_min": args.weight_min,
            "weight_max": args.weight_max,
            "weight_init_mode": args.weight_init_mode,
            "input_gain": args.input_gain,
            "voltage_amp": spec.voltage_amp,
            "current_amp": spec.current_amp,
            "trainable_amplification": bool(args.trainable_amplification),
            "amplification_min": args.amplification_min,
            "amplification_max": args.amplification_max,
            "non_linearity": spec.non_linearity,
            "quadratic_diode_param": {
                "diode_conductance": 1.0,
                "v_min": -1.0e6,
                "v_max": 1.0e6,
            },
            "hard_sigmoid_param": hard_sigmoid_param,
            "exponential_diode_param": {
                "I_s": 1.0e-6,
                "V_t": 0.025,
                "V_off": 0.0,
            },
            "num_iterations_inference": inference_iterations,
            "num_iterations_training": training_iterations,
            "minimizer": json.loads(json.dumps(args.minimizer_config_data)),
        },
        "model_overrides": {
            model_key: {
                "layer_shapes": layer_shapes,
                "conv_pipeline": conv_pipeline,
                "weight_gains": weight_gains,
            },
        },
        "energy_minimizer": {
            "mode": "asynchronous",
        },
    }


def _load_minimizer_config(path_value: str | None) -> dict:
    if path_value is None:
        raise SystemExit(
            "Expected --minimizer-config to point to an explicit simulator/minimizer JSON object. "
            "Provided value: None."
        )
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    if not path.exists():
        raise SystemExit(
            "Expected --minimizer-config to point to an existing JSON file. "
            f"Provided value: {path}."
        )
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise SystemExit(
            "Expected --minimizer-config JSON to contain an object. "
            f"Provided value: {data!r}."
        )
    if data.get("adaptive_equilibrium") is not False:
        raise SystemExit(
            "Expected --minimizer-config adaptive_equilibrium to be false for fixed-step runs. "
            f"Provided value: {data.get('adaptive_equilibrium')!r}."
        )
    return data


def _all_specs(
    non_linearities: list[str], seeds: list[int], *, include_legacy: bool = False
) -> list[RunSpec]:
    amplification_grid = list(AMPLIFICATION_GRID)
    if include_legacy:
        amplification_grid.extend(LEGACY_AMPLIFICATION_GRID)

    specs = []
    for non_linearity in non_linearities:
        for run_name, voltage_amp, current_amp in amplification_grid:
            for seed in seeds:
                specs.append(
                    RunSpec(
                        non_linearity=str(non_linearity),
                        run_name=run_name,
                        seed=int(seed),
                        voltage_amp=float(voltage_amp),
                        current_amp=float(current_amp),
                    )
                )
    return specs


def _select_shard(specs: list[RunSpec], num_shards: int, shard_index: int) -> list[RunSpec]:
    if num_shards < 1:
        raise ValueError("--num-shards must be >= 1.")
    if not 0 <= shard_index < num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < --num-shards.")
    return [spec for idx, spec in enumerate(specs) if idx % num_shards == shard_index]


def _run_dir(output_root: Path, spec: RunSpec) -> Path:
    return output_root / spec.non_linearity / spec.run_name / spec.seed_dir_name


def _run_complete(run_dir: Path) -> bool:
    return all((run_dir / name).exists() for name in REQUIRED_RUN_FILES)


def _resolve_init_checkpoint(args: argparse.Namespace, spec: RunSpec) -> Path | None:
    if not args.init_checkpoint_root:
        return None
    root = Path(args.init_checkpoint_root).expanduser().resolve()
    checkpoint = root / spec.non_linearity / spec.run_name / spec.seed_dir_name / args.init_checkpoint_name
    if not checkpoint.exists():
        raise FileNotFoundError(
            "Expected --init-checkpoint-root to contain checkpoints at "
            "<root>/<non_linearity>/<run_name>/seed_<seed>/<checkpoint_name>. "
            f"Provided root: {root}; missing checkpoint: {checkpoint}"
        )
    return checkpoint


def _load_metrics(run_dir: Path) -> dict | None:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return None
    return json.loads(metrics_path.read_text())


def _summary_row(output_root: Path, spec: RunSpec) -> dict | None:
    run_dir = _run_dir(output_root, spec)
    metrics = _load_metrics(run_dir)
    if metrics is None:
        return None
    config = _load_json_if_exists(run_dir / "config.json")
    architecture = config.get("architecture", {})
    training = config.get("training", {})
    optimizer = config.get("optimizer", {})
    learning_rate = optimizer.get("learning_rate", "")
    if isinstance(learning_rate, list) and learning_rate:
        learning_rate = learning_rate[0]
    return {
        "non_linearity": spec.non_linearity,
        "run_name": spec.run_name,
        "seed": spec.seed,
        "voltage_amp": spec.voltage_amp,
        "current_amp": spec.current_amp,
        "input_gain": architecture.get("input_gain", ""),
        "num_iterations_inference": training.get("num_iterations_inference", ""),
        "num_iterations_training": training.get("num_iterations_training", ""),
        "learning_rate": learning_rate,
        "conv_depth": len(architecture.get("conv_pipeline", []) or []),
        "conv_pipeline": json.dumps(architecture.get("conv_pipeline", [])),
        "best_test_accuracy": metrics.get("best_test_accuracy"),
        "final_test_accuracy": metrics.get("final_test_accuracy"),
        "best_epoch": metrics.get("best_epoch"),
        "final_train_loss": metrics.get("final_train_loss"),
        "final_test_loss": metrics.get("final_test_loss"),
        "checkpoint_path": metrics.get("checkpoint_path"),
        "weights_best_path": metrics.get("weights_best_path"),
        "weights_final_path": metrics.get("weights_final_path"),
    }


def _write_csv_atomic(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)


def _load_json_if_exists(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def write_summaries(output_root: Path, specs: list[RunSpec]) -> tuple[Path, Path, Path]:
    rows = []
    for spec in specs:
        row = _summary_row(output_root, spec)
        if row is not None:
            rows.append(row)

    summary_path = output_root / "summary.csv"
    _write_csv_atomic(summary_path, SUMMARY_COLUMNS, rows)

    grouped: dict[tuple[str, str, float, float], list[dict]] = {}
    for row in rows:
        key = (
            row["non_linearity"],
            row["run_name"],
            float(row["voltage_amp"]),
            float(row["current_amp"]),
        )
        grouped.setdefault(key, []).append(row)

    def mean(values):
        return statistics.fmean(values) if values else ""

    def stdev(values):
        return statistics.stdev(values) if len(values) > 1 else 0.0

    by_amp_rows = []
    for (non_linearity, run_name, voltage_amp, current_amp), group in sorted(grouped.items()):
        best = [float(row["best_test_accuracy"]) for row in group if row["best_test_accuracy"]]
        final = [float(row["final_test_accuracy"]) for row in group if row["final_test_accuracy"]]
        train_loss = [float(row["final_train_loss"]) for row in group if row["final_train_loss"]]
        test_loss = [float(row["final_test_loss"]) for row in group if row["final_test_loss"]]
        by_amp_rows.append(
            {
                "non_linearity": non_linearity,
                "run_name": run_name,
                "voltage_amp": voltage_amp,
                "current_amp": current_amp,
                "num_seeds": len(group),
                "mean_best_test_accuracy": mean(best),
                "std_best_test_accuracy": stdev(best),
                "mean_final_test_accuracy": mean(final),
                "std_final_test_accuracy": stdev(final),
                "mean_final_train_loss": mean(train_loss),
                "mean_final_test_loss": mean(test_loss),
            }
        )

    by_amp_path = output_root / "summary_by_amp.csv"
    by_nonlinearity_amp_path = output_root / "summary_by_nonlinearity_amp.csv"
    _write_csv_atomic(by_amp_path, SUMMARY_BY_AMP_COLUMNS, by_amp_rows)
    _write_csv_atomic(by_nonlinearity_amp_path, SUMMARY_BY_AMP_COLUMNS, by_amp_rows)
    return summary_path, by_amp_path, by_nonlinearity_amp_path


def run_one(args: argparse.Namespace, output_root: Path, spec: RunSpec) -> None:
    run_dir = _run_dir(output_root, spec)
    if _run_complete(run_dir) and not args.rerun_complete:
        print(f"[skip] {spec.non_linearity} {spec.run_name} {spec.seed_dir_name} complete at {run_dir}")
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    source_config_path = run_dir / "source_config.json"
    init_checkpoint_path = _resolve_init_checkpoint(args, spec)
    source_config = _build_config(args, spec)
    if init_checkpoint_path is not None:
        source_config["init_checkpoint_path"] = str(init_checkpoint_path)
    source_config_path.write_text(json.dumps(source_config, indent=2))

    print(
        "[run] "
        f"non_linearity={spec.non_linearity} "
        f"{spec.run_name} seed={spec.seed} "
        f"voltage_amp={_fmt_amp(spec.voltage_amp)} current_amp={_fmt_amp(spec.current_amp)} "
        f"output={run_dir}"
    )
    if init_checkpoint_path is not None:
        print(f"[run] init_checkpoint={init_checkpoint_path}")
    if args.dry_run:
        return

    prune_state: dict[str, object] = {"pruned": False}

    def prune_callback(epoch_info: dict, history: dict) -> bool:
        if args.prune_after_epochs is None:
            return False
        if int(epoch_info["epoch"]) < int(args.prune_after_epochs):
            return False
        test_accuracy = [
            float(value)
            for value in history.get("test_accuracy", [])
            if value is not None and math.isfinite(float(value))
        ]
        if not test_accuracy:
            return False
        best_so_far = max(test_accuracy)
        threshold = float(args.prune_min_best_test_accuracy)
        if best_so_far >= threshold:
            return False
        prune_state.update(
            {
                "pruned": True,
                "pruned_after_epoch": int(epoch_info["epoch"]),
                "prune_min_best_test_accuracy": threshold,
                "best_test_accuracy_so_far": best_so_far,
                "current_test_accuracy": epoch_info.get("test_accuracy"),
                "current_test_loss": epoch_info.get("test_loss"),
            }
        )
        marker_path = run_dir / "pruned_after_epoch.json"
        marker_path.write_text(json.dumps(prune_state, indent=2))
        print(
            "[prune] stopping after epoch "
            f"{epoch_info['epoch']}: best_test_accuracy_so_far={best_so_far:.6f} "
            f"< threshold={threshold:.6f}",
            flush=True,
        )
        return True

    train_mnist_conv(
        config_path=source_config_path,
        epochs=args.epochs,
        lr=_expanded_learning_rate(args),
        beta=args.beta,
        log_interval=args.log_interval,
        max_batches=args.max_batches,
        max_test_batches=args.max_test_batches,
        device=args.device,
        sanity_check=False,
        dataset_key="mnist",
        output_dir=run_dir,
        model_key=args.model_key,
        training_algorithm="BP",
        seed=spec.seed,
        lr_decay=args.lr_decay,
        init_checkpoint_path=init_checkpoint_path,
        epoch_callback=prune_callback if args.prune_after_epochs is not None else None,
    )
    if prune_state.get("pruned"):
        metrics_path = run_dir / "metrics.json"
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text())
            metrics.update(prune_state)
            metrics_path.write_text(json.dumps(metrics, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--dataset-root", default="~/datasets/mnist")
    parser.add_argument("--device", default=None)
    parser.add_argument("--model-key", default=MODEL_KEY)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument(
        "--non-linearity",
        nargs="+",
        choices=[
            "perfect_diode",
            "hard_sigmoid",
            "lpw_diode",
            "double_diode_quadratic",
            "double_diode_exponential",
            "single_diode_exponential",
            "linear",
        ],
        default=DEFAULT_NON_LINEARITIES,
        help="One or more nonlinearities to include in the sweep.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-iterations", type=int, default=4)
    parser.add_argument("--num-iterations-inference", type=int, default=None)
    parser.add_argument("--num-iterations-training", type=int, default=None)
    parser.add_argument("--minimizer-config", help=MINIMIZER_CONFIG_HELP)
    parser.add_argument("--learning-rate", type=float, nargs="+", default=[0.006])
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--lr-decay", type=float, default=0.99)
    parser.add_argument("--conv-depth", type=int, default=1)
    parser.add_argument("--conv-channels", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--padding", type=int, default=0)
    parser.add_argument(
        "--strides",
        type=int,
        nargs="+",
        default=None,
        help="Per-convolution strides. One value broadcasts to all conv layers.",
    )
    parser.add_argument(
        "--paddings",
        type=int,
        nargs="+",
        default=None,
        help="Per-convolution paddings. One value broadcasts to all conv layers.",
    )
    parser.add_argument("--output-dim", type=int, default=20)
    parser.add_argument("--weight-gains", type=float, nargs="+", default=[1.0])
    parser.add_argument("--weight-min", type=float, default=0.0)
    parser.add_argument("--weight-max", type=float, default=100.0)
    parser.add_argument("--weight-init-mode", default="kaiming_uniform")
    parser.add_argument("--input-gain", type=float, default=100.0)
    parser.add_argument("--trainable-amplification", action="store_true")
    parser.add_argument("--amplification-min", type=float, default=1e-6)
    parser.add_argument("--amplification-max", type=float, default=None)
    parser.add_argument("--hard-sigmoid-g-on", type=float, default=100.0)
    parser.add_argument("--hard-sigmoid-g-off", type=float, default=0.0)
    parser.add_argument("--hard-sigmoid-v-min", type=float, default=-1.5)
    parser.add_argument("--hard-sigmoid-v-max", type=float, default=1.5)
    parser.add_argument("--hard-sigmoid-v-off", type=float, nargs="+", default=None)
    parser.add_argument("--trainable-hard-sigmoid-v-off", action="store_true")
    parser.add_argument("--hard-sigmoid-v-off-min", type=float, default=0.0)
    parser.add_argument("--hard-sigmoid-v-off-max", type=float, default=None)
    parser.add_argument("--normalize-mean", type=float, default=0.1307)
    parser.add_argument("--normalize-std", type=float, default=0.3081)
    parser.add_argument("--normalize-scale", type=float, default=0.3)
    parser.add_argument(
        "--affine-preset",
        choices=sorted(AFFINE_PRESETS),
        default="none",
        help="Deterministic per-sample affine MNIST preset.",
    )
    parser.add_argument("--affine-degrees", type=float, default=None)
    parser.add_argument("--affine-translate", type=float, nargs="+", default=None)
    parser.add_argument("--affine-scale", type=float, nargs=2, default=None)
    parser.add_argument("--affine-shear", type=float, default=None)
    parser.add_argument("--affine-seed", type=int, default=1729)
    parser.add_argument("--log-interval", type=int, default=500)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--max-test-batches", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--run-name",
        choices=[name for name, _, _ in AMPLIFICATION_GRID + LEGACY_AMPLIFICATION_GRID],
        help="Restrict the sweep to a single amplification run name.",
    )
    parser.add_argument(
        "--include-legacy-amplification",
        action="store_true",
        help="Include the legacy A=4, B=0.25 amplification setting in full sweeps.",
    )
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument(
        "--init-checkpoint-root",
        help=(
            "Optional root containing checkpoints laid out as "
            "<root>/<non_linearity>/<run_name>/seed_<seed>/<checkpoint_name>."
        ),
    )
    parser.add_argument("--init-checkpoint-name", default="final_model.pt")
    parser.add_argument(
        "--prune-after-epochs",
        type=int,
        default=None,
        help="Stop after this epoch if best test accuracy so far is below the prune threshold.",
    )
    parser.add_argument(
        "--prune-min-best-test-accuracy",
        type=float,
        default=None,
        help="Minimum best test accuracy required at --prune-after-epochs.",
    )
    parser.add_argument("--rerun-complete", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    _architecture(args)
    _iteration_counts(args)
    _expanded_learning_rate(args)
    _expanded_weight_gains(args)
    _expanded_hard_sigmoid_v_off(args)
    _affine_config_from_args(args)
    if not args.summary_only:
        args.minimizer_config_data = _load_minimizer_config(args.minimizer_config)
    else:
        args.minimizer_config_data = None
    if args.output_dim not in (10, 20):
        raise SystemExit("Expected --output-dim to be 10 or 20 for MNIST.")
    if (args.prune_after_epochs is None) != (args.prune_min_best_test_accuracy is None):
        raise SystemExit(
            "Expected --prune-after-epochs and --prune-min-best-test-accuracy "
            "to be provided together."
        )
    if args.prune_after_epochs is not None:
        if args.prune_after_epochs < 1:
            raise SystemExit("Expected --prune-after-epochs to be >= 1.")
        if not 0.0 <= args.prune_min_best_test_accuracy <= 1.0:
            raise SystemExit("Expected --prune-min-best-test-accuracy to be in [0, 1].")
    return args


def main() -> None:
    print(
        "[deprecated] experiments/train_mnist_bp_conv_amplification_sweep.py is "
        "a legacy diagnostic runner; use `python -m experiments.mnist_conv sweep` "
        "for new work.",
        file=sys.stderr,
    )
    args = parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    legacy_run_names = {name for name, _, _ in LEGACY_AMPLIFICATION_GRID}
    include_legacy = args.include_legacy_amplification or args.run_name in legacy_run_names
    all_specs = _all_specs(
        list(args.non_linearity), list(args.seeds), include_legacy=include_legacy
    )
    if args.run_name is not None:
        all_specs = [spec for spec in all_specs if spec.run_name == args.run_name]
    selected_specs = _select_shard(all_specs, args.num_shards, args.shard_index)

    print(f"[sweep] output_root={output_root}")
    print(f"[sweep] total_runs={len(all_specs)} selected_runs={len(selected_specs)}")
    print(f"[sweep] shard={args.shard_index}/{args.num_shards}")
    print(f"[sweep] layer_shapes={_architecture(args)[0]}")
    print(f"[sweep] learning_rate={_expanded_learning_rate(args)}")

    if args.summary_only:
        paths = write_summaries(output_root, all_specs)
        print("[summary] wrote " + ", ".join(str(path) for path in paths))
        return

    for spec in selected_specs:
        run_one(args, output_root, spec)
        paths = write_summaries(output_root, all_specs)
        print("[summary] wrote " + ", ".join(str(path) for path in paths))

    paths = write_summaries(output_root, all_specs)
    print("[done] summary=" + ", ".join(str(path) for path in paths))


if __name__ == "__main__":
    main()
