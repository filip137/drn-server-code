#!/usr/bin/env python3
"""Train one-or-more-convolution MNIST DRNs with BP over amplification settings."""

from __future__ import annotations

import argparse
import csv
import json
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


MODEL_KEY = "mnist_bp_conv_amp"
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "results" / "mnist_bp_conv1_amplification_sweep_64ch_s2_valid"
)
DEFAULT_SEEDS = [0, 1, 2]
DEFAULT_NON_LINEARITIES = ["hard_sigmoid", "perfect_diode"]
AMPLIFICATION_GRID = [
    ("mnist_bp_amp_v1_c1", 1.0, 1.0),
    ("mnist_bp_amp_v2_c1", 2.0, 1.0),
    ("mnist_bp_amp_v4_c1", 4.0, 1.0),
    ("mnist_bp_amp_v1_c2", 1.0, 2.0),
    ("mnist_bp_amp_v1_c4", 1.0, 4.0),
]
SUMMARY_COLUMNS = [
    "non_linearity",
    "run_name",
    "seed",
    "voltage_amp",
    "current_amp",
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
    for index in range(args.conv_depth):
        height = _conv_spatial(height, args.kernel_size, args.stride, args.padding)
        width = _conv_spatial(width, args.kernel_size, args.stride, args.padding)
        if height <= 0 or width <= 0:
            raise ValueError(
                "Convolution geometry produced non-positive spatial dimensions: "
                f"depth={index + 1}, height={height}, width={width}."
            )
        layer_shapes.append([channels[index], height, width])
        conv_pipeline.append(
            {
                "kernel": [args.kernel_size, args.kernel_size],
                "stride": args.stride,
                "padding": args.padding,
                "mode": "convolution",
            }
        )
    layer_shapes.append([args.output_dim])
    return layer_shapes, conv_pipeline


def _num_energy_params(args: argparse.Namespace) -> int:
    # conv weights + dense readout weight + one bias per conv/free layer.
    return args.conv_depth + 1 + args.conv_depth


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


def _build_config(args: argparse.Namespace, spec: RunSpec) -> dict:
    layer_shapes, conv_pipeline = _architecture(args)
    learning_rate = _expanded_learning_rate(args)
    weight_gains = _expanded_weight_gains(args)
    model_key = args.model_key
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
                "factory": "labs.datasets.MnistDataset",
                "params": {
                    "name": "mnist",
                    "batch_size": args.batch_size,
                    "root": os.path.expanduser(args.dataset_root),
                    "train": True,
                    "download": not args.no_download,
                    "normalize": True,
                    "normalize_mean": args.normalize_mean,
                    "normalize_std": args.normalize_std,
                    "normalize_scale": args.normalize_scale,
                },
            },
        },
        "model_base": {
            "weight_min": args.weight_min,
            "weight_max": args.weight_max,
            "weight_init_mode": args.weight_init_mode,
            "input_gain": args.input_gain,
            "voltage_amp": spec.voltage_amp,
            "current_amp": spec.current_amp,
            "non_linearity": spec.non_linearity,
            "quadratic_diode_param": {
                "diode_conductance": 1.0,
                "v_min": -1.0e6,
                "v_max": 1.0e6,
            },
            "hard_sigmoid_param": {
                "g_on": args.hard_sigmoid_g_on,
                "g_off": args.hard_sigmoid_g_off,
                "v_min": args.hard_sigmoid_v_min,
                "v_max": args.hard_sigmoid_v_max,
            },
            "exponential_diode_param": {
                "I_s": 1.0e-6,
                "V_t": 0.025,
                "V_off": 0.0,
            },
            "num_iterations_inference": args.num_iterations,
            "num_iterations_training": args.num_iterations,
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


def _all_specs(non_linearities: list[str], seeds: list[int]) -> list[RunSpec]:
    specs = []
    for non_linearity in non_linearities:
        for run_name, voltage_amp, current_amp in AMPLIFICATION_GRID:
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
    return {
        "non_linearity": spec.non_linearity,
        "run_name": spec.run_name,
        "seed": spec.seed,
        "voltage_amp": spec.voltage_amp,
        "current_amp": spec.current_amp,
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
    )


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
    parser.add_argument("--learning-rate", type=float, nargs="+", default=[0.006])
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--lr-decay", type=float, default=0.99)
    parser.add_argument("--conv-depth", type=int, default=1)
    parser.add_argument("--conv-channels", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--padding", type=int, default=0)
    parser.add_argument("--output-dim", type=int, default=20)
    parser.add_argument("--weight-gains", type=float, nargs="+", default=[1.0])
    parser.add_argument("--weight-min", type=float, default=0.0)
    parser.add_argument("--weight-max", type=float, default=100.0)
    parser.add_argument("--weight-init-mode", default="kaiming_uniform")
    parser.add_argument("--input-gain", type=float, default=100.0)
    parser.add_argument("--hard-sigmoid-g-on", type=float, default=100.0)
    parser.add_argument("--hard-sigmoid-g-off", type=float, default=0.0)
    parser.add_argument("--hard-sigmoid-v-min", type=float, default=-1.5)
    parser.add_argument("--hard-sigmoid-v-max", type=float, default=1.5)
    parser.add_argument("--normalize-mean", type=float, default=0.1307)
    parser.add_argument("--normalize-std", type=float, default=0.3081)
    parser.add_argument("--normalize-scale", type=float, default=0.3)
    parser.add_argument("--log-interval", type=int, default=500)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--max-test-batches", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--run-name",
        choices=[name for name, _, _ in AMPLIFICATION_GRID],
        help="Restrict the sweep to a single amplification run name.",
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
    parser.add_argument("--rerun-complete", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    _architecture(args)
    _expanded_learning_rate(args)
    _expanded_weight_gains(args)
    if args.output_dim not in (10, 20):
        raise SystemExit("Expected --output-dim to be 10 or 20 for MNIST.")
    return args


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    all_specs = _all_specs(list(args.non_linearity), list(args.seeds))
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
