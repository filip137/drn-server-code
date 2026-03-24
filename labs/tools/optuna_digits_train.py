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
        "Optuna is not installed. Install it (e.g. `pip install optuna`) "
        "or use your existing env that has it."
    ) from exc

from small_network_config import (  # noqa: E402
    load_json_config,
    _parse_layer_shapes,
    _parse_non_linearity_params,
    _resolve_config_value,
    _resolve_optional_config_value,
)
from small_network_core import (  # noqa: E402
    MinimizerRuntimeSettings,
    NonLinearityData,
    TrainRunSettings,
    _resolve_double_diode_runtime,
    train,
)


def _build_train_settings(config: dict) -> TrainRunSettings:
    dims_config = config.get("dims")
    if not dims_config:
        raise SystemExit("Config must define 'dims'.")
    _input_dim, hidden_dims, _output_dim, _layer_shapes = _parse_layer_shapes(dims_config)

    non_linearity = config.get("non_linearity")
    if not non_linearity:
        raise SystemExit("Config must define 'non_linearity'.")
    quadratic_params, exponential_params, hard_sigmoid_params = _parse_non_linearity_params(
        config,
        non_linearity,
    )

    num_iterations = _resolve_config_value("num_iterations", None, config)
    num_epochs = _resolve_config_value("num_epochs", None, config)
    batch_size = _resolve_optional_config_value("batch_size", None, config, default=1)
    learning_rate = _resolve_optional_config_value("learning_rate", None, config, default=None)
    nudging = _resolve_optional_config_value("nudging", None, config, default=0.0)
    early_stop_min_epochs = _resolve_optional_config_value("early_stop_min_epochs", None, config, default=None)
    early_stop_error_pct = _resolve_optional_config_value("early_stop_error_pct", None, config, default=None)
    dataset_name = _resolve_optional_config_value("dataset", None, config, default="digits")
    weight_gains = config.get("weight_gains")
    weight_min = config.get("weight_min")
    weight_max = config.get("weight_max")
    input_gain = _resolve_config_value("input_gain", None, config)
    seed = _resolve_optional_config_value("seed", None, config, default=None)
    voltage_amp = _resolve_config_value("voltage_amp", None, config)
    current_amp = _resolve_config_value("current_amp", None, config)

    double_diode_updater = config.get("double_diode_updater")
    double_diode_runtime = config.get("double_diode_runtime")
    adaptive_equilibrium = config.get("adaptive_equilibrium")
    if non_linearity == "double_diode_exponential":
        if not double_diode_updater:
            if double_diode_runtime:
                double_diode_updater, adaptive_equilibrium = _resolve_double_diode_runtime(double_diode_runtime)
            else:
                raise SystemExit("Config must define 'double_diode_updater'.")
    elif not double_diode_updater and double_diode_runtime:
        double_diode_updater, adaptive_equilibrium = _resolve_double_diode_runtime(double_diode_runtime)
    if adaptive_equilibrium is None:
        raise SystemExit("Config must define 'adaptive_equilibrium' (bool).")

    overrelaxation_factor = _resolve_optional_config_value(
        "overrelaxation_factor",
        None,
        config,
        default=1.1,
    )
    overrelaxation_reject_steps = _resolve_optional_config_value(
        "overrelaxation_reject_steps",
        None,
        config,
        default=False,
    )
    overrelaxation_reject_max_tries = _resolve_optional_config_value(
        "overrelaxation_reject_max_tries",
        None,
        config,
        default=3,
    )
    overrelaxation_reject_shrink = _resolve_optional_config_value(
        "overrelaxation_reject_shrink",
        None,
        config,
        default=0.5,
    )
    overrelaxation_reject_eps = _resolve_optional_config_value(
        "overrelaxation_reject_eps",
        None,
        config,
        default=0.0,
    )
    damping_value = _resolve_optional_config_value("damping", None, config, default=None)
    requires_damping = non_linearity in ("single_diode_exponential", "double_diode_exponential")
    if requires_damping and damping_value is None:
        raise SystemExit(
            "Expected 'damping' to be a numeric config value (for example, 0.5) "
            f"for non_linearity={non_linearity!r}. Provided value: {damping_value!r}."
        )
    try:
        # Non-exponential nonlinearities can omit damping; default to neutral damping.
        damping = 1.0 if damping_value is None else float(damping_value)
    except (TypeError, ValueError) as exc:
        raise SystemExit(
            "Expected 'damping' to be a numeric config value (for example, 0.5). "
            f"Provided value: {damping_value!r}."
        ) from exc
    single_diode_updater = config.get("single_diode_updater")
    if non_linearity == "single_diode_exponential" and not single_diode_updater:
        raise SystemExit("Config must define 'single_diode_updater'.")

    rel_tol = _resolve_optional_config_value("rel_tol", None, config, default=1e-5)
    vn_tol = _resolve_optional_config_value("vn_tol", None, config, default=1e-6)
    residual_current_tol = _resolve_optional_config_value("residual_current_tol", None, config, default=None)
    use_polish = _resolve_optional_config_value("use_polish", None, config, default=True)
    max_newton_iters = _resolve_optional_config_value("max_newton_iters", None, config, default=32)
    z_thresh = _resolve_optional_config_value("z_thresh", None, config, default=1e10)
    if non_linearity in ("single_diode_exponential", "double_diode_exponential"):
        exp_clip = _resolve_config_value("exp_clip", None, config)
    else:
        exp_clip = _resolve_optional_config_value("exp_clip", None, config, default=100000.0)
    dynamic_polish = _resolve_optional_config_value("dynamic_polish", None, config, default=False)

    minimizer_impl = _resolve_optional_config_value("minimizer_impl", None, config, default=None)
    if not minimizer_impl:
        raise SystemExit("Config must define 'minimizer_impl'.")
    if minimizer_impl not in ("custom", "andersson"):
        raise SystemExit(
            "minimizer_impl must be one of: custom, andersson; "
            f"got {minimizer_impl!r}"
        )
    training_algorithm = _resolve_optional_config_value("training_algorithm", None, config, default="EP")
    if training_algorithm is None:
        training_algorithm = "EP"
    training_algorithm = str(training_algorithm).upper()
    if training_algorithm not in ("EP", "BP"):
        raise SystemExit(
            "training_algorithm must be 'EP' or 'BP'. "
            f"Got {training_algorithm!r}."
        )
    anderson_m = _resolve_optional_config_value("anderson_m", None, config, default=8)
    anderson_omega = _resolve_optional_config_value("anderson_omega", None, config, default=1.0)
    anderson_tol_floor = _resolve_optional_config_value("anderson_tol_floor", None, config, default=5e-3)
    anderson_reg = _resolve_optional_config_value("anderson_reg", None, config, default=1e-8)
    save_epochs = _resolve_optional_config_value("save_epochs", None, config, default=None)
    if save_epochs is not None:
        if isinstance(save_epochs, (list, tuple)):
            save_epoch_values = save_epochs
        else:
            save_epoch_values = [save_epochs]
        try:
            save_epochs = sorted({int(epoch) for epoch in save_epoch_values if int(epoch) > 0})
        except (TypeError, ValueError) as exc:
            raise SystemExit("save_epochs must be an int or list of ints.") from exc
        if not save_epochs:
            save_epochs = None

    non_linearity_data = NonLinearityData(
        non_linearity=non_linearity,
        quadratic_diode_param=quadratic_params,
        exponential_diode_param=exponential_params,
        hard_sigmoid_param=hard_sigmoid_params,
    )
    minimizer_settings = MinimizerRuntimeSettings(
        exp_clip=exp_clip,
        double_diode_updater=double_diode_updater,
        adaptive_equilibrium=bool(adaptive_equilibrium),
        overrelaxation_factor=overrelaxation_factor,
        overrelaxation_reject_steps=bool(overrelaxation_reject_steps),
        overrelaxation_reject_max_tries=int(overrelaxation_reject_max_tries),
        overrelaxation_reject_shrink=float(overrelaxation_reject_shrink),
        overrelaxation_reject_eps=float(overrelaxation_reject_eps),
        single_diode_updater=single_diode_updater,
        rel_tol=rel_tol,
        vn_tol=vn_tol,
        residual_current_tol=residual_current_tol,
        use_polish=use_polish,
        max_newton_iters=max_newton_iters,
        z_thresh=z_thresh,
        dynamic_polish=dynamic_polish,
        minimizer_impl=minimizer_impl,
        anderson_m=anderson_m,
        anderson_omega=anderson_omega,
        anderson_tol_floor=anderson_tol_floor,
        anderson_reg=anderson_reg,
        damping=damping,
    )

    num_points = 2000
    return TrainRunSettings(
        dims=dims_config,
        num_points=num_points,
        batch_size=batch_size,
        num_iterations=num_iterations,
        num_epochs=num_epochs,
        seed=seed,
        voltage_amp=voltage_amp,
        current_amp=current_amp,
        input_gain=input_gain,
        weight_gains=weight_gains,
        weight_min=weight_min,
        weight_max=weight_max,
        dataset_name=dataset_name,
        training_algorithm=training_algorithm,
        learning_rate=learning_rate,
        nudging=nudging,
        non_linearity_data=non_linearity_data,
        minimizer_settings=minimizer_settings,
        early_stop_min_epochs=early_stop_min_epochs,
        early_stop_error_pct=early_stop_error_pct,
        save_epochs=save_epochs,
    )


def main() -> int:
    p = argparse.ArgumentParser(description="Optuna sweep for digits small_network training.")
    p.add_argument("--base-config", required=True, help="Path to base JSON config.")
    p.add_argument("--trials", type=int, default=50)
    p.add_argument(
        "--tune-input-gain",
        action="store_true",
        help="Include input_gain in the Optuna search space.",
    )
    p.add_argument(
        "--input-gain-min",
        type=float,
        default=None,
        help="Minimum input_gain when --tune-input-gain is enabled.",
    )
    p.add_argument(
        "--input-gain-max",
        type=float,
        default=None,
        help="Maximum input_gain when --tune-input-gain is enabled.",
    )
    p.add_argument(
        "--input-gain-log",
        action="store_true",
        help="Sample input_gain on a log scale when --tune-input-gain is enabled.",
    )
    p.add_argument(
        "--nudging-min",
        type=float,
        default=None,
        help="Override minimum nudging sampled by Optuna.",
    )
    p.add_argument(
        "--nudging-max",
        type=float,
        default=None,
        help="Override maximum nudging sampled by Optuna.",
    )
    p.add_argument(
        "--nudging-log",
        action="store_true",
        help="Sample nudging on a log scale (use with --nudging-min/--nudging-max).",
    )
    p.add_argument("--output-dir", default=None, help="Directory to write Optuna outputs.")
    p.add_argument(
        "--storage",
        default="auto",
        help="Optuna storage URI or path. Use 'none' for in-memory. "
        "Default: 'auto' -> <output_dir>/optuna.db.",
    )
    p.add_argument(
        "--study-name",
        default=None,
        help="Optuna study name (default: base config stem).",
    )
    p.add_argument(
        "--load-if-exists",
        action="store_true",
        help="Reuse existing study when storage is enabled.",
    )
    p.add_argument(
        "--prune-train-cost-threshold",
        type=float,
        default=None,
        help="Prune trial if train_cost is above this threshold after the epoch cutoff.",
    )
    p.add_argument(
        "--prune-train-cost-epoch",
        type=int,
        default=20,
        help="Epoch (inclusive) after which train_cost threshold pruning applies.",
    )
    p.add_argument(
        "--prune-test-error-threshold",
        type=float,
        default=None,
        help="Prune trial if test_error is not below this threshold after the epoch cutoff.",
    )
    p.add_argument(
        "--prune-test-error-epoch",
        type=int,
        default=50,
        help="Epoch (inclusive) after which test_error threshold pruning applies.",
    )
    p.add_argument(
        "--pruner",
        choices=("none", "median"),
        default="none",
        help="Enable Optuna pruning (median).",
    )
    p.add_argument(
        "--pruner-metric",
        choices=(
            "test_accuracy",
            "train_accuracy",
            "test_error",
            "train_error",
            "test_cost",
            "train_cost",
        ),
        default="test_accuracy",
        help="Metric used for pruning decisions.",
    )
    p.add_argument("--pruner-startup-trials", type=int, default=5)
    p.add_argument("--pruner-warmup-epochs", type=int, default=5)
    p.add_argument("--pruner-interval-epochs", type=int, default=1)
    args = p.parse_args()

    base_path = Path(args.base_config).expanduser().resolve()
    base_config = load_json_config(base_path)

    base_dir = base_path.parent
    output_root = Path(args.output_dir).expanduser().resolve() if args.output_dir else base_dir / "optuna_runs"
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

    base_learning_rate = base_config.get("learning_rate")
    base_nudging = base_config.get("nudging", 0.0)
    base_weight_gains = base_config.get("weight_gains")
    base_input_gain = base_config.get("input_gain", 10.0)
    if not isinstance(base_learning_rate, (list, tuple)):
        raise SystemExit("base config learning_rate must be a list")
    if not isinstance(base_weight_gains, (list, tuple)):
        raise SystemExit("base config weight_gains must be a list")
    try:
        base_input_gain = float(base_input_gain)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"base config input_gain must be numeric; got {base_input_gain!r}") from exc

    if not args.tune_input_gain and (
        args.input_gain_min is not None or args.input_gain_max is not None or args.input_gain_log
    ):
        raise SystemExit(
            "--input-gain-min/--input-gain-max/--input-gain-log require --tune-input-gain."
        )

    input_gain_min = None
    input_gain_max = None
    if args.tune_input_gain:
        if args.input_gain_min is not None:
            input_gain_min = float(args.input_gain_min)
        elif base_input_gain > 0.0:
            input_gain_min = base_input_gain * 0.1
        else:
            input_gain_min = 0.1

        if args.input_gain_max is not None:
            input_gain_max = float(args.input_gain_max)
        elif base_input_gain > 0.0:
            input_gain_max = base_input_gain * 10.0
        else:
            input_gain_max = 20.0

        if input_gain_min >= input_gain_max:
            raise SystemExit(
                f"input_gain bounds invalid: min={input_gain_min} must be < max={input_gain_max}."
            )
        if args.input_gain_log and input_gain_min <= 0.0:
            raise SystemExit(
                f"--input-gain-log requires a positive minimum, got {input_gain_min}."
            )

    nudging_min = None
    nudging_max = None
    if args.nudging_min is not None or args.nudging_max is not None:
        if args.nudging_min is None or args.nudging_max is None:
            raise SystemExit("Specify both --nudging-min and --nudging-max together.")
        nudging_min = float(args.nudging_min)
        nudging_max = float(args.nudging_max)
        if nudging_min >= nudging_max:
            raise SystemExit(
                f"nudging bounds invalid: min={nudging_min} must be < max={nudging_max}."
            )
        if args.nudging_log and nudging_min <= 0.0:
            raise SystemExit(
                f"--nudging-log requires a positive minimum, got {nudging_min}."
            )

    pruner = None
    if args.pruner == "median":
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=args.pruner_startup_trials,
            n_warmup_steps=args.pruner_warmup_epochs,
            interval_steps=args.pruner_interval_epochs,
        )

    def _pruner_score(metric_name: str, metrics: dict):
        if metric_name == "train_accuracy":
            err = metrics.get("train_error")
            return None if err is None else 100.0 - float(err)
        if metric_name == "test_accuracy":
            err = metrics.get("test_error")
            return None if err is None else 100.0 - float(err)
        if metric_name == "train_error":
            err = metrics.get("train_error")
            return None if err is None else 100.0 - float(err)
        if metric_name == "test_error":
            err = metrics.get("test_error")
            return None if err is None else 100.0 - float(err)
        if metric_name == "train_cost":
            cost = metrics.get("train_cost")
            return None if cost is None else -float(cost)
        if metric_name == "test_cost":
            cost = metrics.get("test_cost")
            return None if cost is None else -float(cost)
        return None

    def objective(trial: optuna.Trial) -> float:
        config = dict(base_config)
        lr_list = []
        for idx, lr in enumerate(base_learning_rate):
            lr_list.append(
                trial.suggest_float(
                    f"learning_rate_{idx}",
                    float(lr) * 0.1,
                    float(lr) * 10.0,
                    log=True,
                )
            )

        weight_gains = []
        for idx, gain in enumerate(base_weight_gains):
            weight_gains.append(
                trial.suggest_float(
                    f"weight_gain_{idx}",
                    float(gain) * 0.1,
                    float(gain) * 10.0,
                    log=True,
                )
            )

        if nudging_min is not None and nudging_max is not None:
            nudging = trial.suggest_float(
                "nudging",
                nudging_min,
                nudging_max,
                log=bool(args.nudging_log),
            )
        elif base_nudging is None:
            nudging = trial.suggest_float("nudging", 0.0, 5.0)
        elif float(base_nudging) > 0:
            nudging = trial.suggest_float(
                "nudging",
                float(base_nudging) * 0.1,
                float(base_nudging) * 10.0,
                log=True,
            )
        else:
            nudging = trial.suggest_float("nudging", 0.0, 5.0)

        if args.tune_input_gain:
            config["input_gain"] = trial.suggest_float(
                "input_gain",
                input_gain_min,
                input_gain_max,
                log=bool(args.input_gain_log),
            )
        else:
            config["input_gain"] = base_input_gain
        config["learning_rate"] = lr_list
        config["weight_gains"] = weight_gains
        config["nudging"] = nudging

        trial_dir = output_root / f"trial_{trial.number:03d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        (trial_dir / "config.json").write_text(json.dumps(config, indent=2))

        settings = _build_train_settings(config)

        epoch_log_path = trial_dir / "epoch_metrics.jsonl"

        def _epoch_callback(epoch: int, epoch_metrics: dict) -> None:
            record = {"epoch": epoch, **epoch_metrics}
            with epoch_log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            if args.prune_train_cost_threshold is not None and epoch >= args.prune_train_cost_epoch:
                train_cost = epoch_metrics.get("train_cost")
                if train_cost is not None and float(train_cost) > float(args.prune_train_cost_threshold):
                    raise optuna.TrialPruned(
                        f"Pruned at epoch {epoch}: train_cost={train_cost:.5f} "
                        f"> {args.prune_train_cost_threshold:.5f}"
                    )
            if args.prune_test_error_threshold is not None and epoch >= args.prune_test_error_epoch:
                test_error = epoch_metrics.get("test_error")
                if test_error is not None and float(test_error) >= float(args.prune_test_error_threshold):
                    raise optuna.TrialPruned(
                        f"Pruned at epoch {epoch}: test_error={test_error:.3f} "
                        f">= {args.prune_test_error_threshold:.3f}"
                    )
            if pruner is None:
                return
            score = _pruner_score(args.pruner_metric, epoch_metrics)
            if score is None:
                return
            trial.report(score, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned(
                    f"Pruned at epoch {epoch} with {args.pruner_metric}={score:.4f}"
                )

        run_dir, _model_path, _npz_path, metrics = train(
            settings=settings,
            output_root=str(trial_dir),
            run_subdir=None,
            weights_path=None,
            return_metrics=True,
            epoch_callback=_epoch_callback,
        )
        train_acc = metrics.get("train_accuracy")
        trial.set_user_attr("train_accuracy", train_acc)
        trial.set_user_attr("run_dir", str(run_dir))
        return float(train_acc) if train_acc is not None else 0.0

    study = optuna.create_study(
        direction="maximize",
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
