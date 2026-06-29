#!/usr/bin/env python3
"""Optuna search for Conv1 hard-sigmoid MNIST DRN operating points.

This is intentionally narrow: one amplification setting per study, seed 0 by
default, 10-epoch BP trials from scratch, and only three searched axes:
input_gain, v_off, and scalar learning rate.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
LABS_DIR = REPO_ROOT / "labs"
for path in (REPO_ROOT, LABS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    import optuna
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Optuna is not installed in this Python environment.") from exc

from labs.mnist_train import train_mnist_conv  # noqa: E402


MODEL_KEY = "mnist_bp_conv_amp"
AMPLIFICATIONS = {
    "mnist_bp_amp_v1_c1": (1.0, 1.0),
    "mnist_bp_amp_v1_c2": (1.0, 2.0),
}


def _storage_uri(output_root: Path, storage: str) -> str | None:
    if storage == "none":
        return None
    if storage == "auto":
        return f"sqlite:///{output_root / 'optuna.db'}"
    if "://" in storage:
        return storage
    return f"sqlite:///{Path(storage).expanduser().resolve()}"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


def _build_config(args: argparse.Namespace, params: dict[str, Any]) -> dict[str, Any]:
    voltage_amp, current_amp = AMPLIFICATIONS[args.run_name]
    v_off = float(params["v_off"])
    learning_rate = [float(params["learning_rate"])] * 3
    return {
        "lab": {
            "model_key": MODEL_KEY,
            "dataset_key": "mnist",
            "epochs": int(args.epochs),
            "plots_dir": "plots/mnist_bp_conv1_hardsigmoid_optuna",
        },
        "input_mode": "train",
        "training_algorithm": "BP",
        "seed": int(args.seed),
        "lr": learning_rate,
        "beta": 1.0,
        "lr_decay": float(args.lr_decay),
        "log_interval": int(args.log_interval),
        "max_batches": args.max_batches,
        "max_test_batches": args.max_test_batches,
        "datasets": {
            "mnist": {
                "factory": "labs.datasets.MnistDataset",
                "params": {
                    "name": "mnist",
                    "batch_size": int(args.batch_size),
                    "root": str(args.dataset_root),
                    "train": True,
                    "download": not args.no_download,
                    "normalize": True,
                    "normalize_mean": 0.1307,
                    "normalize_std": 0.3081,
                    "normalize_scale": 0.3,
                },
            },
        },
        "model_base": {
            "weight_min": 0.0,
            "weight_max": 100.0,
            "weight_init_mode": "kaiming_uniform",
            "input_gain": float(params["input_gain"]),
            "voltage_amp": voltage_amp,
            "current_amp": current_amp,
            "non_linearity": "hard_sigmoid",
            "quadratic_diode_param": {
                "diode_conductance": 1.0,
                "v_min": -1.0e6,
                "v_max": 1.0e6,
            },
            "hard_sigmoid_param": {
                "g_on": 100.0,
                "g_off": 0.0,
                "v_min": -v_off,
                "v_max": v_off,
            },
            "exponential_diode_param": {
                "I_s": 1.0e-6,
                "V_t": 0.025,
                "V_off": 0.0,
            },
            "num_iterations_inference": int(args.num_iterations),
            "num_iterations_training": int(args.num_iterations),
        },
        "model_overrides": {
            MODEL_KEY: {
                "layer_shapes": [[2, 28, 28], [64, 13, 13], [20]],
                "conv_pipeline": [
                    {
                        "kernel": [3, 3],
                        "stride": 2,
                        "padding": 0,
                        "mode": "convolution",
                    }
                ],
                "weight_gains": [1.0, 1.0],
            },
        },
        "energy_minimizer": {
            "mode": "asynchronous",
        },
    }


def _suggest_params(trial: optuna.Trial, args: argparse.Namespace) -> dict[str, float]:
    return {
        "learning_rate": trial.suggest_float(
            "learning_rate", args.lr_min, args.lr_max, log=True
        ),
        "input_gain": trial.suggest_float(
            "input_gain", args.input_gain_min, args.input_gain_max, log=True
        ),
        "v_off": trial.suggest_float("v_off", args.v_off_min, args.v_off_max),
    }


def _metric_value(summary: dict[str, Any], name: str) -> float:
    value = summary.get(name)
    if value is None:
        return float("nan")
    return float(value)


def _write_study_summary(study: optuna.Study, output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for trial in study.trials:
        row = {
            "number": trial.number,
            "state": trial.state.name,
            "value": trial.value,
        }
        row.update({f"param_{key}": value for key, value in trial.params.items()})
        row.update({f"attr_{key}": value for key, value in trial.user_attrs.items()})
        rows.append(row)

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    trials_path = output_root / "trials.csv"
    with trials_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    complete = [
        trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    if complete:
        best = study.best_trial
        best_payload = {
            "best_trial_number": best.number,
            "best_value": best.value,
            "best_params": best.params,
            "best_user_attrs": best.user_attrs,
        }
    else:
        best_payload = {
            "best_trial_number": None,
            "best_value": None,
            "best_params": {},
            "best_user_attrs": {},
        }
    (output_root / "best_trial.json").write_text(
        json.dumps(_jsonable(best_payload), indent=2)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        default=str(
            REPO_ROOT
            / "results"
            / "mnist_bp_conv1_hardsigmoid_optuna_operating_point_seed0_10epoch"
        ),
    )
    parser.add_argument("--run-name", choices=sorted(AMPLIFICATIONS), required=True)
    parser.add_argument("--study-name", default=None)
    parser.add_argument("--storage", default="auto")
    parser.add_argument("--load-if-exists", action="store_true")
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sampler-seed", type=int, default=0)
    parser.add_argument("--worker-id", default="worker0")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dataset-root",
        default=str(REPO_ROOT / "model" / "resistive" / "data"),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-iterations", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--max-test-batches", type=int, default=None)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--log-interval", type=int, default=1000)
    parser.add_argument("--lr-decay", type=float, default=0.99)
    parser.add_argument("--lr-min", type=float, default=3.0e-3)
    parser.add_argument("--lr-max", type=float, default=8.0e-2)
    parser.add_argument("--input-gain-min", type=float, default=10.0)
    parser.add_argument("--input-gain-max", type=float, default=220.0)
    parser.add_argument("--v-off-min", type=float, default=1.0)
    parser.add_argument("--v-off-max", type=float, default=5.0)
    parser.add_argument("--pruner-startup-trials", type=int, default=4)
    parser.add_argument("--pruner-warmup-epochs", type=int, default=3)
    parser.add_argument("--pruner-interval-epochs", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_output_root = Path(args.output_root).expanduser().resolve()
    output_root = base_output_root / args.run_name
    output_root.mkdir(parents=True, exist_ok=True)
    args.dataset_root = Path(args.dataset_root).expanduser().resolve()

    if args.study_name is None:
        args.study_name = f"conv1_hardsigmoid_operating_point_{args.run_name}"
    storage = _storage_uri(output_root, args.storage)
    sampler = optuna.samplers.TPESampler(
        seed=args.sampler_seed,
        multivariate=True,
        group=True,
        constant_liar=True,
    )
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=args.pruner_startup_trials,
        n_warmup_steps=args.pruner_warmup_epochs,
        interval_steps=args.pruner_interval_epochs,
    )
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        study_name=args.study_name,
        load_if_exists=args.load_if_exists or storage is not None,
    )

    worker_dir = output_root / "workers" / args.worker_id
    worker_dir.mkdir(parents=True, exist_ok=True)
    (worker_dir / "worker_config.json").write_text(
        json.dumps(_jsonable(vars(args)), indent=2)
    )

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial, args)
        trial_dir = output_root / "trials" / f"trial_{trial.number:04d}"
        run_dir = trial_dir / "run"
        trial_dir.mkdir(parents=True, exist_ok=True)
        config = _build_config(args, params)
        config_path = trial_dir / "source_config.json"
        config_path.write_text(json.dumps(_jsonable(config), indent=2))
        epoch_log_path = trial_dir / "epoch_metrics.jsonl"

        def epoch_callback(epoch_info: dict[str, Any], history: dict[str, Any]) -> bool:
            with epoch_log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(_jsonable(epoch_info)) + "\n")
            test_accuracy = epoch_info.get("test_accuracy")
            if test_accuracy is None or not math.isfinite(float(test_accuracy)):
                return False
            trial.report(float(test_accuracy), step=int(epoch_info["epoch"]))
            trial.set_user_attr("latest_test_accuracy", float(test_accuracy))
            trial.set_user_attr("latest_epoch", int(epoch_info["epoch"]))
            if trial.should_prune():
                raise optuna.TrialPruned(
                    f"Pruned at epoch {epoch_info['epoch']} with test_accuracy={test_accuracy!r}"
                )
            return False

        try:
            history = train_mnist_conv(
                config_path=config_path,
                epochs=args.epochs,
                lr=[float(params["learning_rate"])] * 3,
                beta=1.0,
                log_interval=args.log_interval,
                max_batches=args.max_batches,
                max_test_batches=args.max_test_batches,
                device=args.device,
                sanity_check=False,
                epoch_callback=epoch_callback,
                dataset_key="mnist",
                output_dir=run_dir,
                model_key=MODEL_KEY,
                training_algorithm="BP",
                seed=args.seed,
                lr_decay=float(args.lr_decay),
                init_checkpoint_path=None,
            )
        finally:
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass
            gc.collect()

        summary = history.get("summary", {})
        result = {
            "trial_number": trial.number,
            "params": params,
            "summary": summary,
            "run_dir": history.get("run_dir"),
        }
        (trial_dir / "trial_result.json").write_text(
            json.dumps(_jsonable(result), indent=2)
        )
        trial.set_user_attr("run_dir", history.get("run_dir"))
        trial.set_user_attr("best_model_path", summary.get("best_model_path"))
        trial.set_user_attr("weights_best_path", summary.get("weights_best_path"))
        trial.set_user_attr("best_epoch", summary.get("best_epoch"))
        trial.set_user_attr("best_test_accuracy", summary.get("best_test_accuracy"))
        trial.set_user_attr("final_test_accuracy", summary.get("final_test_accuracy"))
        trial.set_user_attr("final_test_loss", summary.get("final_test_loss"))
        trial.set_user_attr("final_train_loss", summary.get("final_train_loss"))
        trial.set_user_attr("learning_rate", params["learning_rate"])
        trial.set_user_attr("input_gain", params["input_gain"])
        trial.set_user_attr("v_off", params["v_off"])
        trial.set_user_attr("lr_decay", args.lr_decay)
        value = _metric_value(summary, "best_test_accuracy")
        if not math.isfinite(value):
            return 0.0
        return value

    try:
        if args.trials > 0:
            study.optimize(objective, n_trials=args.trials)
    finally:
        _write_study_summary(study, output_root)

    complete_trials = [
        trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    if complete_trials:
        print(
            json.dumps(
                _jsonable(
                    {
                        "run_name": args.run_name,
                        "best_trial_number": study.best_trial.number,
                        "best_value": study.best_value,
                        "best_params": study.best_params,
                        "best_user_attrs": study.best_trial.user_attrs,
                    }
                ),
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
