#!/usr/bin/env python3
"""Train MNIST DRNs with BP while sweeping voltage/current amplification."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LABS_DIR = REPO_ROOT / "labs"
for path in (REPO_ROOT, LABS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from labs.mnist_train import train_mnist_conv  # noqa: E402


MODEL_KEY = "mnist_bp_amp"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "mnist_bp_amplification_sweep"
DEFAULT_SEEDS = [0, 1, 2]
DEFAULT_LEARNING_RATE = [0.1, 0.05, 0.0]
AMPLIFICATION_GRID = [
    ("mnist_bp_amp_v1_c1", 1.0, 1.0),
    ("mnist_bp_amp_v2_c1", 2.0, 1.0),
    ("mnist_bp_amp_v4_c1", 4.0, 1.0),
    ("mnist_bp_amp_v1_c2", 1.0, 2.0),
    ("mnist_bp_amp_v1_c4", 1.0, 4.0),
]
SUMMARY_COLUMNS = [
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


def _build_config(args: argparse.Namespace, spec: RunSpec) -> dict:
    learning_rate = args.learning_rate or DEFAULT_LEARNING_RATE
    init_checkpoint_path = _resolve_init_checkpoint(args, spec)
    model_key = args.model_key
    return {
        "lab": {
            "model_key": model_key,
            "dataset_key": "mnist",
            "epochs": args.epochs,
            "plots_dir": "plots/mnist_bp_amplification_sweep",
        },
        "input_mode": "train",
        "init_checkpoint_path": init_checkpoint_path,
        "training_algorithm": "BP",
        "seed": spec.seed,
        "lr": [float(value) for value in learning_rate],
        "beta": float(args.beta),
        "lr_decay": float(args.lr_decay),
        "log_interval": args.log_interval,
        "max_batches": args.max_batches,
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
            "non_linearity": args.non_linearity,
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
                "layer_shapes": [[2, 28, 28], [100], [10]],
                "conv_pipeline": [],
                "weight_gains": [float(value) for value in args.weight_gains],
            },
        },
        "energy_minimizer": {
            "mode": "asynchronous",
        },
    }


def _all_specs(seeds: list[int]) -> list[RunSpec]:
    specs = []
    for run_name, voltage_amp, current_amp in AMPLIFICATION_GRID:
        for seed in seeds:
            specs.append(
                RunSpec(
                    run_name=run_name,
                    seed=int(seed),
                    voltage_amp=float(voltage_amp),
                    current_amp=float(current_amp),
                )
            )
    return specs


def _resolve_init_checkpoint(args: argparse.Namespace, spec: RunSpec) -> str | None:
    if args.init_checkpoint is not None:
        return str(Path(args.init_checkpoint).expanduser())
    if args.init_checkpoint_template is None:
        return None
    return args.init_checkpoint_template.format(
        run_name=spec.run_name,
        seed=spec.seed,
        seed_dir=spec.seed_dir_name,
    )


def _select_shard(specs: list[RunSpec], num_shards: int, shard_index: int) -> list[RunSpec]:
    if num_shards < 1:
        raise ValueError("--num-shards must be >= 1.")
    if not 0 <= shard_index < num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < num-shards.")
    return [spec for idx, spec in enumerate(specs) if idx % num_shards == shard_index]


def _run_dir(output_root: Path, spec: RunSpec) -> Path:
    return output_root / spec.run_name / spec.seed_dir_name


def _run_complete(run_dir: Path) -> bool:
    return all((run_dir / name).exists() for name in REQUIRED_RUN_FILES)


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


def write_summary(output_root: Path, specs: list[RunSpec]) -> Path:
    rows = []
    for spec in specs:
        row = _summary_row(output_root, spec)
        if row is not None:
            rows.append(row)

    summary_path = output_root / "summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = summary_path.with_name(f".summary.{os.getpid()}.tmp")
    with tmp_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(summary_path)
    return summary_path


def run_one(args: argparse.Namespace, output_root: Path, spec: RunSpec) -> None:
    run_dir = _run_dir(output_root, spec)
    if _run_complete(run_dir) and not args.rerun_complete:
        print(f"[skip] {spec.run_name} {spec.seed_dir_name} already complete at {run_dir}")
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    source_config_path = run_dir / "source_config.json"
    source_config_path.write_text(json.dumps(_build_config(args, spec), indent=2))

    print(
        "[run] "
        f"{spec.run_name} seed={spec.seed} "
        f"voltage_amp={_fmt_amp(spec.voltage_amp)} current_amp={_fmt_amp(spec.current_amp)} "
        f"output={run_dir}"
    )
    if args.dry_run:
        return

    train_mnist_conv(
        config_path=source_config_path,
        epochs=args.epochs,
        lr=args.learning_rate or DEFAULT_LEARNING_RATE,
        beta=args.beta,
        log_interval=args.log_interval,
        max_batches=args.max_batches,
        device=args.device,
        sanity_check=False,
        dataset_key="mnist",
        output_dir=run_dir,
        model_key=args.model_key,
        training_algorithm="BP",
        seed=spec.seed,
        lr_decay=args.lr_decay,
        init_checkpoint_path=_resolve_init_checkpoint(args, spec),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a 2-channel MNIST DRN with one 100-unit hidden layer and BP "
            "over the fixed amplification sweep."
        )
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--dataset-root", default="~/datasets/mnist")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--model-key",
        default=MODEL_KEY,
        help="Model override key to write into the generated config.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-iterations", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, nargs="+", default=None)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--lr-decay", type=float, default=1.0)
    parser.add_argument(
        "--init-checkpoint",
        help=(
            "Optional DRN parameter checkpoint to load before training. "
            "Use only with one selected run/seed."
        ),
    )
    parser.add_argument(
        "--init-checkpoint-template",
        help=(
            "Optional checkpoint template with {run_name}, {seed}, and {seed_dir} "
            "placeholders for seed-aware continuation runs."
        ),
    )
    parser.add_argument("--weight-gains", type=float, nargs="+", default=[1.0, 1.0])
    parser.add_argument("--weight-min", type=float, default=1.0e-7)
    parser.add_argument("--weight-max", type=float, default=100.0)
    parser.add_argument("--weight-init-mode", default="kaiming_uniform")
    parser.add_argument("--input-gain", type=float, default=50.0)
    parser.add_argument(
        "--non-linearity",
        default="perfect_diode",
        choices=[
            "perfect_diode",
            "hard_sigmoid",
            "lpw_diode",
            "double_diode_quadratic",
            "double_diode_exponential",
            "single_diode_exponential",
            "linear",
        ],
        help="DRN nonlinear interaction used by the model.",
    )
    parser.add_argument("--hard-sigmoid-g-on", type=float, default=1.0)
    parser.add_argument("--hard-sigmoid-g-off", type=float, default=1.0)
    parser.add_argument("--hard-sigmoid-v-min", type=float, default=-1.0e6)
    parser.add_argument("--hard-sigmoid-v-max", type=float, default=1.0e6)
    parser.add_argument("--normalize-mean", type=float, default=0.1307)
    parser.add_argument("--normalize-std", type=float, default=0.3081)
    parser.add_argument(
        "--normalize-scale",
        type=float,
        default=0.3,
        help=(
            "Post-normalization multiplier. The legacy paper/recovered loader used "
            "Normalize(mean=0.1307,std=0.3081) followed by multiplication by 0.3."
        ),
    )
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--run-name",
        choices=[name for name, _, _ in AMPLIFICATION_GRID],
        help="Restrict the sweep to a single amplification run name.",
    )
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument(
        "--rerun-complete",
        action="store_true",
        help="Retrain runs that already have all required artifacts.",
    )
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if len(args.weight_gains) != 2:
        raise SystemExit("Expected --weight-gains to contain 2 values for [input-hidden, hidden-output].")
    learning_rate = args.learning_rate or DEFAULT_LEARNING_RATE
    if len(learning_rate) != 3:
        raise SystemExit(
            "Expected --learning-rate to contain 3 values for "
            "[input-hidden weight, hidden-output weight, hidden bias]."
        )
    if args.init_checkpoint is not None and (
        args.run_name is None or len(args.seeds) != 1
    ):
        raise SystemExit(
            "Expected --init-checkpoint to be used with one --run-name and one --seeds value. "
            "Use --init-checkpoint-template for multi-seed continuation."
        )
    return args


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    all_specs = _all_specs(list(args.seeds))
    if args.run_name is not None:
        all_specs = [spec for spec in all_specs if spec.run_name == args.run_name]
    selected_specs = _select_shard(all_specs, args.num_shards, args.shard_index)

    print(f"[sweep] output_root={output_root}")
    print(f"[sweep] total_runs={len(all_specs)} selected_runs={len(selected_specs)}")
    print(f"[sweep] shard={args.shard_index}/{args.num_shards}")

    if args.summary_only:
        summary_path = write_summary(output_root, all_specs)
        print(f"[summary] wrote {summary_path}")
        return

    for spec in selected_specs:
        run_one(args, output_root, spec)
        summary_path = write_summary(output_root, all_specs)
        print(f"[summary] wrote {summary_path}")

    summary_path = write_summary(output_root, all_specs)
    print(f"[done] summary={summary_path}")


if __name__ == "__main__":
    main()
