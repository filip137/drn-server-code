#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABS_DIR = PROJECT_ROOT / "labs"
for path in (PROJECT_ROOT, LABS_DIR):
    if str(path) not in sys.path:
        sys.path.append(str(path))

try:
    import optuna
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Optuna is not installed. Install it (for example, `pip install optuna`) "
        "or use an environment that already has it."
    ) from exc

from mnist_train import load_config, train_mnist_conv, train_tiny_grid  # noqa: E402


def _deep_copy_config(config: dict) -> dict:
    return json.loads(json.dumps(config))


def _merged_model_config(config: dict, model_key: str) -> dict:
    base = dict(config.get("model_base", {}))
    overrides = dict(config.get("model_overrides", {}).get(model_key, {}))
    return {**base, **overrides}


def _set_model_override(config: dict, model_key: str, key: str, value) -> None:
    model_overrides = dict(config.get("model_overrides", {}))
    model_cfg = dict(model_overrides.get(model_key, {}))
    model_cfg[key] = value
    model_overrides[model_key] = model_cfg
    config["model_overrides"] = model_overrides


def _choose_train_fn(model_key: str):
    if model_key == "tiny3x3":
        return train_tiny_grid
    return train_mnist_conv


def _normalize_dataset_key(dataset_key):
    if dataset_key is None:
        return None
    normalized = str(dataset_key).strip().lower().replace("-", "_")
    if normalized == "fashion_mnist":
        return "fmnist"
    return normalized


def _trial_metric(metric_name: str, epoch_info: dict, history: dict):
    summary = history.get("summary", {})
    if metric_name == "train_accuracy":
        return epoch_info.get("train_accuracy")
    if metric_name == "test_accuracy":
        return epoch_info.get("test_accuracy")
    if metric_name == "train_loss":
        value = epoch_info.get("train_loss")
        return None if value is None else -float(value)
    if metric_name == "test_loss":
        value = epoch_info.get("test_loss")
        return None if value is None else -float(value)
    if metric_name == "best_test_accuracy":
        return summary.get("best_test_accuracy")
    return None


def _objective_from_history(metric_name: str, history: dict) -> float:
    summary = history.get("summary", {})
    if metric_name == "best_test_accuracy":
        value = summary.get("best_test_accuracy")
    elif metric_name == "final_test_accuracy":
        value = summary.get("final_test_accuracy")
    elif metric_name == "final_train_accuracy":
        value = summary.get("final_train_accuracy")
    elif metric_name == "final_test_loss":
        value = summary.get("final_test_loss")
        return float("inf") if value is None else float(value)
    elif metric_name == "final_train_loss":
        value = summary.get("final_train_loss")
        return float("inf") if value is None else float(value)
    else:
        raise SystemExit(f"Unsupported objective metric: {metric_name!r}")

    if value is None:
        if metric_name == "best_test_accuracy":
            value = summary.get("final_train_accuracy")
        elif metric_name == "final_test_accuracy":
            value = summary.get("final_train_accuracy")
    return float(value) if value is not None else 0.0


def main() -> int:
    p = argparse.ArgumentParser(description="Optuna sweep for labs/mnist_train.py.")
    p.add_argument("--base-config", required=True, help="Path to base MNIST JSON config.")
    p.add_argument("--trials", type=int, default=50)
    p.add_argument("--epochs", type=int, default=None, help="Override config epochs for all trials.")
    p.add_argument("--log-interval", type=int, default=None, help="Override config log interval.")
    p.add_argument("--max-batches", type=int, default=None, help="Optional cap on batches per epoch.")
    p.add_argument("--device", type=str, default=None, help="Torch device string to use for all trials, e.g. 'cuda:0'.")
    p.add_argument("--dataset", type=str, default=None, help="Dataset override for all trials. Use 'mnist' or 'fmnist'.")
    p.add_argument("--sanity-check", action="store_true", help="Run the conv sanity check before each trial.")
    p.add_argument("--lr-min", type=float, default=None, help="Override minimum learning rate sampled by Optuna.")
    p.add_argument("--lr-max", type=float, default=None, help="Override maximum learning rate sampled by Optuna.")
    p.add_argument(
        "--lr-log",
        action="store_true",
        help="Sample learning rates on a log scale (use with --lr-min/--lr-max).",
    )
    p.add_argument("--tune-beta", action="store_true", help="Include beta in the search space.")
    p.add_argument("--beta-min", type=float, default=None)
    p.add_argument("--beta-max", type=float, default=None)
    p.add_argument("--beta-log", action="store_true")
    p.add_argument("--tune-input-gain", action="store_true", help="Include input_gain in the search space.")
    p.add_argument("--input-gain-min", type=float, default=None)
    p.add_argument("--input-gain-max", type=float, default=None)
    p.add_argument("--input-gain-log", action="store_true")
    p.add_argument("--tune-weight-gains", action="store_true", help="Include per-layer weight_gains in the search space.")
    p.add_argument("--output-dir", default=None, help="Directory to write trial outputs.")
    p.add_argument(
        "--storage",
        default="auto",
        help="Optuna storage URI or path. Use 'none' for in-memory. Default: 'auto' -> <output_dir>/optuna.db.",
    )
    p.add_argument("--study-name", default=None)
    p.add_argument("--load-if-exists", action="store_true")
    p.add_argument(
        "--objective-metric",
        choices=(
            "best_test_accuracy",
            "final_test_accuracy",
            "final_train_accuracy",
            "final_test_loss",
            "final_train_loss",
        ),
        default="best_test_accuracy",
    )
    p.add_argument(
        "--pruner",
        choices=("none", "median"),
        default="none",
        help="Enable Optuna pruning.",
    )
    p.add_argument(
        "--pruner-metric",
        choices=("train_accuracy", "test_accuracy", "train_loss", "test_loss"),
        default="test_accuracy",
    )
    p.add_argument("--pruner-startup-trials", type=int, default=5)
    p.add_argument("--pruner-warmup-epochs", type=int, default=5)
    p.add_argument("--pruner-interval-epochs", type=int, default=1)
    args = p.parse_args()

    base_path = Path(args.base_config).expanduser().resolve()
    base_config = load_config(base_path)
    lab_cfg = base_config.get("lab", {})
    model_key = lab_cfg.get("model_key", "mnist")
    dataset_key = _normalize_dataset_key(
        args.dataset if args.dataset is not None else lab_cfg.get("dataset_key")
    )
    if dataset_key is None:
        dataset_key = "tiny3x3" if model_key == "tiny3x3" else "mnist"
    train_fn = _choose_train_fn(model_key)
    if model_key == "tiny3x3":
        if dataset_key != "tiny3x3":
            raise SystemExit(
                f"Model {model_key!r} only supports dataset 'tiny3x3', got {dataset_key!r}."
            )
    elif dataset_key not in ("mnist", "fmnist"):
        raise SystemExit(
            f"Unsupported dataset {dataset_key!r}. Use 'mnist' or 'fmnist'."
        )

    output_root = Path(args.output_dir).expanduser().resolve() if args.output_dir else base_path.parent / "optuna_runs"
    output_root.mkdir(parents=True, exist_ok=True)

    storage = None
    storage_arg = str(args.storage).strip() if args.storage is not None else ""
    if storage_arg and storage_arg.lower() not in ("none", "memory", "in-memory"):
        if "://" in storage_arg:
            storage = storage_arg
        else:
            if storage_arg.lower() == "auto":
                storage_path = output_root / "optuna.db"
            else:
                storage_path = Path(storage_arg).expanduser().resolve()
            storage_path.parent.mkdir(parents=True, exist_ok=True)
            storage = f"sqlite:///{storage_path}"
    study_name = args.study_name or base_path.stem
    if args.study_name is None and dataset_key not in ("mnist", "tiny3x3"):
        study_name = f"{study_name}_{dataset_key}"

    epochs = int(args.epochs) if args.epochs is not None else int(lab_cfg.get("epochs", 1))
    log_interval = args.log_interval if args.log_interval is not None else base_config.get("log_interval")
    if log_interval is None:
        raise SystemExit("log_interval must be provided via config or --log-interval.")
    log_interval = int(log_interval)
    max_batches = int(args.max_batches) if args.max_batches is not None else base_config.get("max_batches")
    if max_batches is not None:
        max_batches = int(max_batches)
    device = args.device if args.device is not None else base_config.get("device")

    base_lr = base_config.get("lr")
    if base_lr is None:
        raise SystemExit("Base config must define 'lr'.")
    if isinstance(base_lr, list):
        base_learning_rate = [float(x) for x in base_lr]
    else:
        base_learning_rate = float(base_lr)

    base_beta = base_config.get("beta")
    merged_model = _merged_model_config(base_config, model_key)
    base_input_gain = float(merged_model.get("input_gain", 10.0))
    base_weight_gains = merged_model.get("weight_gains")
    if args.tune_weight_gains and not isinstance(base_weight_gains, (list, tuple)):
        raise SystemExit("weight_gains must be a list to tune them with Optuna.")

    if args.lr_min is not None or args.lr_max is not None:
        if args.lr_min is None or args.lr_max is None:
            raise SystemExit("Specify both --lr-min and --lr-max together.")
        if args.lr_log and (float(args.lr_min) <= 0.0 or float(args.lr_max) <= 0.0):
            raise SystemExit("--lr-log requires positive --lr-min and --lr-max.")

    if args.beta_min is not None or args.beta_max is not None:
        if args.beta_min is None or args.beta_max is None:
            raise SystemExit("Specify both --beta-min and --beta-max together.")
        if not args.tune_beta:
            raise SystemExit("--beta-min/--beta-max require --tune-beta.")

    if args.input_gain_min is not None or args.input_gain_max is not None:
        if args.input_gain_min is None or args.input_gain_max is None:
            raise SystemExit("Specify both --input-gain-min and --input-gain-max together.")
        if not args.tune_input_gain:
            raise SystemExit("--input-gain-min/--input-gain-max require --tune-input-gain.")

    beta_min = float(args.beta_min) if args.beta_min is not None else None
    beta_max = float(args.beta_max) if args.beta_max is not None else None
    if args.tune_beta and beta_min is None:
        if base_beta is not None and float(base_beta) > 0.0:
            beta_min = float(base_beta) * 0.1
            beta_max = float(base_beta) * 10.0 if beta_max is None else beta_max
        else:
            beta_min = 1e-3
            beta_max = 1.0 if beta_max is None else beta_max
    if args.tune_beta and beta_max is None:
        beta_max = 1.0

    input_gain_min = float(args.input_gain_min) if args.input_gain_min is not None else None
    input_gain_max = float(args.input_gain_max) if args.input_gain_max is not None else None
    if args.tune_input_gain and input_gain_min is None:
        input_gain_min = max(base_input_gain * 0.1, 1e-6)
        input_gain_max = base_input_gain * 10.0 if input_gain_max is None else input_gain_max
    if args.tune_input_gain and input_gain_max is None:
        input_gain_max = max(base_input_gain * 10.0, 1.0)

    lr_min = float(args.lr_min) if args.lr_min is not None else None
    lr_max = float(args.lr_max) if args.lr_max is not None else None

    direction = "minimize" if args.objective_metric.endswith("_loss") else "maximize"
    pruner = None
    if args.pruner == "median":
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=args.pruner_startup_trials,
            n_warmup_steps=args.pruner_warmup_epochs,
            interval_steps=args.pruner_interval_epochs,
        )

    def objective(trial: optuna.Trial) -> float:
        config = _deep_copy_config(base_config)
        trial_dir = output_root / f"trial_{trial.number:03d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        lab_override = dict(config.get("lab", {}))
        lab_override["dataset_key"] = dataset_key
        config["lab"] = lab_override
        if device is not None:
            config["device"] = device

        if isinstance(base_learning_rate, list):
            lr_value = [
                trial.suggest_float(
                    f"lr_{idx}",
                    lr_min if lr_min is not None else float(lr) * 0.1,
                    lr_max if lr_max is not None else float(lr) * 10.0,
                    log=bool(args.lr_log) if lr_min is not None else True,
                )
                for idx, lr in enumerate(base_learning_rate)
            ]
        else:
            lr_value = trial.suggest_float(
                "lr",
                lr_min if lr_min is not None else float(base_learning_rate) * 0.1,
                lr_max if lr_max is not None else float(base_learning_rate) * 10.0,
                log=bool(args.lr_log) if lr_min is not None else True,
            )

        if args.tune_beta:
            beta_value = trial.suggest_float(
                "beta",
                beta_min,
                beta_max,
                log=bool(args.beta_log),
            )
        else:
            beta_value = float(base_beta) if base_beta is not None else None

        if args.tune_input_gain:
            _set_model_override(
                config,
                model_key,
                "input_gain",
                trial.suggest_float(
                    "input_gain",
                    input_gain_min,
                    input_gain_max,
                    log=bool(args.input_gain_log),
                ),
            )

        if args.tune_weight_gains:
            tuned_weight_gains = []
            for idx, gain in enumerate(base_weight_gains):
                tuned_weight_gains.append(
                    trial.suggest_float(
                        f"weight_gain_{idx}",
                        float(gain) * 0.1,
                        float(gain) * 10.0,
                        log=True,
                    )
                )
            _set_model_override(config, model_key, "weight_gains", tuned_weight_gains)

        config_path = trial_dir / "config.json"
        config_path.write_text(json.dumps(config, indent=2))
        epoch_log_path = trial_dir / "epoch_metrics.jsonl"

        def _epoch_callback(epoch_info: dict, history: dict):
            with epoch_log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(epoch_info) + "\n")
            if pruner is None:
                return False
            score = _trial_metric(args.pruner_metric, epoch_info, history)
            if score is None:
                return False
            trial.report(score, step=int(epoch_info["epoch"]))
            if trial.should_prune():
                raise optuna.TrialPruned(
                    f"Pruned at epoch {epoch_info['epoch']} with {args.pruner_metric}={score!r}"
                )
            return False

        train_kwargs = dict(
            config_path=str(config_path),
            epochs=epochs,
            lr=lr_value,
            beta=beta_value,
            log_interval=log_interval,
            max_batches=max_batches,
            device=device,
            sanity_check=args.sanity_check,
            epoch_callback=_epoch_callback,
        )
        if model_key != "tiny3x3":
            train_kwargs["dataset_key"] = dataset_key
            train_kwargs["model_key"] = model_key

        history = train_fn(**train_kwargs)
        summary = history.get("summary", {})
        result = {
            "trial": trial.number,
            "summary": summary,
            "lr": lr_value,
            "beta": beta_value,
            "dataset_key": dataset_key,
        }
        (trial_dir / "result.json").write_text(json.dumps(result, indent=2))
        trial.set_user_attr("dataset_key", dataset_key)
        if summary.get("final_test_accuracy") is not None:
            trial.set_user_attr("final_test_accuracy", summary.get("final_test_accuracy"))
        if summary.get("best_test_accuracy") is not None:
            trial.set_user_attr("best_test_accuracy", summary.get("best_test_accuracy"))
        return _objective_from_history(args.objective_metric, history)

    study = optuna.create_study(
        direction=direction,
        pruner=pruner,
        storage=storage,
        study_name=study_name,
        load_if_exists=bool(storage) and args.load_if_exists,
    )
    study.optimize(objective, n_trials=args.trials)

    best = study.best_trial
    summary = {
        "best_value": best.value,
        "best_params": best.params,
        "best_attrs": best.user_attrs,
    }
    (output_root / "best_trial.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
