import argparse
import copy
import json
import sys
import tempfile
from pathlib import Path
from typing import Optional

import optuna

LABS_DIR = Path(__file__).resolve().parent
if str(LABS_DIR) not in sys.path:
    sys.path.append(str(LABS_DIR))

from mnist_train import train_mnist_conv  # noqa: E402

BASE_LR = [0.002, 0.005, 0.04, 0.003, 0.002, 0.006]
BASE_BETA = 0.1
TARGET_LAYER_SHAPES = [[2, 28, 28], [32, 26, 26], [64, 24, 24], [100], [20]]
TARGET_CONV_PIPELINE = [
    {"kernel": [3, 3], "stride": 1, "padding": 0, "mode": "convolution"},
    {"kernel": [3, 3], "stride": 1, "padding": 0, "mode": "convolution"},
]


def _resolve_config(path: str) -> Path:
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = LABS_DIR / cfg_path
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    return cfg_path


def _run_training_once(
    config_dict: dict,
    epochs: Optional[int],
    lr: Optional[float],
    beta: Optional[float],
    trial: Optional[optuna.Trial] = None,
) -> float:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_cfg = Path(tmpdir) / "config.json"
        tmp_cfg.write_text(json.dumps(config_dict, indent=2))
        log_interval = int(config_dict.get("log_interval", 50))
        max_batches = config_dict.get("max_batches")
        if max_batches is not None:
            max_batches = int(max_batches)

        def epoch_callback(epoch_info, _history):
            if trial is None:
                return False
            test_acc = epoch_info.get("test_accuracy")
            train_acc = epoch_info.get("train_accuracy")
            if test_acc is not None and not isinstance(test_acc, float):
                test_acc = float(test_acc)
            if train_acc is not None and not isinstance(train_acc, float):
                train_acc = float(train_acc)
            if test_acc is not None:
                metric = 1.0 - test_acc
            elif train_acc is not None:
                metric = 1.0 - train_acc
            else:
                metric = float("nan")
            trial.report(metric, step=epoch_info["epoch"])
            if trial.should_prune():
                raise optuna.TrialPruned()
            return False

        history = train_mnist_conv(
            config_path=str(tmp_cfg),
            epochs=epochs,
            lr=lr if lr is not None else config_dict.get("lr"),
            beta=beta,
            log_interval=int(log_interval),
            max_batches=max_batches,
            epoch_callback=epoch_callback if trial is not None else None,
        )
        summary = history.get("summary")
        if not summary:
            raise RuntimeError("Training history missing summary.")
        if "test_error" not in summary or summary["test_error"] is None:
            raise RuntimeError("Training did not produce a test_error summary value.")
        return float(summary["test_error"])


def run_optuna_search(
    config_path: str,
    trials: int,
    study_name: str,
    storage: Optional[str],
    model_key: str,
    epochs: Optional[int],
) -> optuna.study.Study:

    base_config = json.loads(_resolve_config(config_path).read_text())

    def objective(trial: optuna.Trial) -> float:
        cfg = copy.deepcopy(base_config)

        # Force target architecture for this search
        cfg["lab"]["model_key"] = model_key
        overrides = cfg["model_overrides"].setdefault(model_key, {})
        overrides["layer_shapes"] = TARGET_LAYER_SHAPES
        overrides["conv_pipeline"] = TARGET_CONV_PIPELINE

        # Tune shared gains / scalars near preferred values
        input_gain_scale = trial.suggest_float("input_gain_scale", 20, 200)
        cfg["model_base"]["input_gain"] *= input_gain_scale

        weight_gain_scale = trial.suggest_float("weight_gain_scale", 0.1, 1.0)
        gains = overrides.get("weight_gains", [])
        if gains:
            overrides["weight_gains"] = [g * weight_gain_scale for g in gains]

        beta_scale = trial.suggest_float("beta_scale", 0.01, 10.0)
        beta_value = BASE_BETA * beta_scale
        cfg["beta"] = beta_value

        lr_scales = []
        for idx, base in enumerate(BASE_LR):
            scale = trial.suggest_float(f"lr_scale_{idx}", 0.1, 10.0)
            lr_scales.append(base * scale)
        cfg["lr"] = lr_scales

        quad_params = cfg["model_base"].setdefault("quadratic_diode_param", {})
        v_shared = trial.suggest_float("quadratic_v_shared", 0.5, 2)
        quad_params["v_min"] = -v_shared
        quad_params["v_max"] = v_shared

        test_error = _run_training_once(cfg, epochs=epochs, lr=None, beta=beta_value, trial=trial)
        trial.set_user_attr("test_error", test_error)
        return test_error

    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        storage=storage,
        load_if_exists=bool(storage),
    )
    study.optimize(objective, n_trials=trials)
    return study


def main():
    parser = argparse.ArgumentParser(
        description="Optuna search wrapper for labs/mnist_train.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="config.json", help="Base config path.")
    parser.add_argument("--trials", type=int, default=10, help="Number of Optuna trials.")
    parser.add_argument("--study-name", default="labs_mnist_optuna", help="Optuna study name.")
    parser.add_argument("--storage", default=None, help="Optuna storage URI (optional).")
    parser.add_argument("--model-key", default="mnist", help="Model override key to tune.")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs passed to mnist_train.")

    args = parser.parse_args()
    study = run_optuna_search(
        config_path=args.config,
        trials=args.trials,
        study_name=args.study_name,
        storage=args.storage,
        model_key=args.model_key,
        epochs=args.epochs,
    )

    print("Best trial:", study.best_trial.number)
    print("Best value (test error):", study.best_value)
    print("Best params:", study.best_trial.params)


if __name__ == "__main__":
    main()
