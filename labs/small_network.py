import argparse
import shlex
import sys
from pathlib import Path

LABS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = LABS_DIR.parent
for path in (LABS_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from small_network_config import (  # noqa: E402
    load_json_config,
    _parse_layer_shapes,
    _parse_non_linearity_params,
    _resolve_config_value,
    _resolve_optional_config_value,
)
from small_network_core import (  # noqa: E402
    LinspaceSettings,
    LinspaceRunSettings,
    MinimizerRuntimeSettings,
    NonLinearityData,
    TrainRunSettings,
    ValidateRunSettings,
    _resolve_double_diode_runtime,
    digits_validate,
    train,
    moons_linspace,
)
from training.tiki_taka import parse_update_pipeline  # noqa: E402

DEFAULT_NUM_POINTS = 2000


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None, help="Path to JSON config or run_metadata.json.")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--batch-size", type=int, default=None, help="Training batch size (default: 1).")
    p.add_argument(
        "--dataset",
        choices=("moons", "yinyang", "digits"),
        default=None,
        help="Training dataset to use.",
    )
    p.add_argument("--num-iterations", type=int, default=None)
    p.add_argument("--num-epochs", type=int, default=None)
    p.add_argument(
        "--mode",
        choices=("train", "linspace", "digits_validate"),
        default="train",
        help="Run mode: train, linspace, or digits_validate.",
    )
    p.add_argument(
        "--training-algorithm",
        default=None,
        help="Training algorithm: EP or BP (default: EP).",
    )
    p.add_argument(
        "--run-subdir",
        default=None,
        help="Optional subdirectory under output-dir for this run.",
    )
    p.add_argument(
        "--use-hidden-subdir",
        action="store_true",
        help="Place outputs under hidden_N based on config dims.",
    )
    p.add_argument("--weights", default=None)
    p.add_argument(
        "--device",
        default=None,
        help="Torch device string, for example 'cpu', 'cuda', or 'cuda:0'.",
    )
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--voltage-amp", type=float, default=None)
    p.add_argument("--current-amp", type=float, default=None)
    p.add_argument("--linspace-min", type=float, default=None)
    p.add_argument("--linspace-max", type=float, default=None)
    p.add_argument("--linspace-samples", type=int, default=None)
    p.add_argument(
        "--use-tensorboard",
        action="store_true",
        default=None,
        help="Write TensorBoard events (enabled if set or use_tensorboard=true in config).",
    )
    p.add_argument(
        "--tensorboard-dir",
        default=None,
        help="Optional TensorBoard log directory (defaults to the run directory).",
    )
    p.add_argument(
        "--overrelaxation-factor",
        type=float,
        default=None,
        help="Overrelaxation factor omega for float32_overrelaxed mode (default from config or 1.1).",
    )
    p.add_argument(
        "--save-test-accuracy-threshold",
        type=float,
        default=None,
        help=(
            "If set, save a dedicated model snapshot the first time test accuracy "
            "reaches or exceeds this value (0 to 1)."
        ),
    )
    p.add_argument(
        "--save-test-accuracy-thresholds",
        type=float,
        nargs="+",
        default=None,
        help=(
            "If set, save dedicated model snapshots the first time test accuracy "
            "reaches or exceeds each provided value (0 to 1)."
        ),
    )
    p.add_argument(
        "--save-every-epoch",
        action="store_true",
        default=None,
        help="Save a model_epoch_XXXX.pt checkpoint after every epoch.",
    )
    args = p.parse_args()
    cli_command = shlex.join([sys.executable] + sys.argv)

    config_path = None
    if args.config:
        config_path = Path(args.config).expanduser()
    else:
        default_cfg = LABS_DIR / "small_network_config.json"
        if default_cfg.exists():
            config_path = default_cfg
    # Config can also be a saved run_metadata.json; both use the same keys.
    config = load_json_config(config_path) if config_path else {}

    def resolve(name, value):
        return _resolve_config_value(name, value, config)

    def resolve_optional(name, value, default=None):
        return _resolve_optional_config_value(name, value, config, default=default)

    output_root = args.output_dir or config.get("output_dir") or config.get("output_root")
    if not output_root:
        raise SystemExit("Missing required output directory. Provide --output-dir or output_root in config.")

    dims_config = config.get("dims")
    if not dims_config:
        raise SystemExit("Config must define 'dims'.")
    input_dim, hidden_dims, output_dim, _layer_shapes = _parse_layer_shapes(dims_config)

    mode = args.mode
    weights_path = resolve_optional("weights", args.weights)
    non_linearity = config.get("non_linearity")
    if not non_linearity:
        raise SystemExit("Config must define 'non_linearity' (no default applied).")
    quadratic_params, exponential_params, hard_sigmoid_params = _parse_non_linearity_params(
        config,
        non_linearity,
    )
    iv_data_path = resolve_optional("iv_data_path", None, default=None)

    run_weights = weights_path
    if mode in ("linspace", "digits_validate") and not run_weights:
        raise SystemExit("--weights is required for linspace or digits_validate mode.")
    config_run_subdir = config.get("run_subdir")
    config_use_hidden_subdir = bool(config.get("use_hidden_subdir", False))
    if args.run_subdir is not None:
        run_subdir = args.run_subdir
    elif config_run_subdir is not None:
        run_subdir = config_run_subdir
    else:
        use_hidden_subdir = args.use_hidden_subdir or config_use_hidden_subdir
        run_subdir = f"hidden_{len(hidden_dims)}" if use_hidden_subdir else None

    num_iterations = resolve("num_iterations", args.num_iterations)
    num_epochs = resolve("num_epochs", args.num_epochs)
    batch_size = resolve_optional("batch_size", args.batch_size, default=1)
    learning_rate = resolve_optional("learning_rate", None, default=None)
    try:
        update_pipeline = parse_update_pipeline(config.get("update_pipeline"))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    nudging = resolve_optional("nudging", None, default=0.0)
    early_stop_min_epochs = resolve_optional("early_stop_min_epochs", None, default=None)
    early_stop_error_pct = resolve_optional("early_stop_error_pct", None, default=None)
    dataset_name = resolve_optional("dataset", args.dataset, default="moons")
    weight_gains = config.get("weight_gains")
    weight_min = config.get("weight_min")
    weight_max = config.get("weight_max")
    input_gain = resolve("input_gain", None)
    seed = resolve_optional("seed", args.seed)
    device = resolve_optional("device", args.device, default=None)
    voltage_amp = resolve("voltage_amp", args.voltage_amp)
    current_amp = resolve("current_amp", args.current_amp)
    double_diode_updater = config.get("double_diode_updater")
    double_diode_runtime = config.get("double_diode_runtime")
    adaptive_equilibrium = config.get("adaptive_equilibrium")
    if non_linearity == "double_diode_exponential":
        if not double_diode_updater:
            if double_diode_runtime:
                double_diode_updater, adaptive_equilibrium = _resolve_double_diode_runtime(double_diode_runtime)
                print(
                    "[config] 'double_diode_runtime' is deprecated; "
                    "please switch to 'double_diode_updater' + 'adaptive_equilibrium'."
                )
            else:
                raise SystemExit("Config must define 'double_diode_updater'.")
    elif not double_diode_updater and double_diode_runtime:
        double_diode_updater, adaptive_equilibrium = _resolve_double_diode_runtime(double_diode_runtime)
        print(
            "[config] 'double_diode_runtime' is deprecated; "
            "please switch to 'double_diode_updater' + 'adaptive_equilibrium'."
        )
    if adaptive_equilibrium is None:
        raise SystemExit("Config must define 'adaptive_equilibrium' (bool).")
    overrelaxation_factor = resolve_optional("overrelaxation_factor", args.overrelaxation_factor, default=1.1)
    overrelaxation_reject_steps = resolve_optional("overrelaxation_reject_steps", None, default=False)
    overrelaxation_reject_max_tries = resolve_optional("overrelaxation_reject_max_tries", None, default=3)
    overrelaxation_reject_shrink = resolve_optional("overrelaxation_reject_shrink", None, default=0.5)
    overrelaxation_reject_eps = resolve_optional("overrelaxation_reject_eps", None, default=0.0)
    damping_value = resolve_optional("damping", None, default=None)
    if non_linearity == "experimental" and damping_value is None:
        raise SystemExit(
            "Expected 'damping' to be a numeric config value (for example, 0.5). "
            f"Provided value: {damping_value!r}."
        )
    if damping_value is None:
        damping = 0.5
    else:
        try:
            damping = float(damping_value)
        except (TypeError, ValueError) as exc:
            raise SystemExit(
                "Expected 'damping' to be a numeric config value (for example, 0.5). "
                f"Provided value: {damping_value!r}."
            ) from exc
    experimental_newton_max_steps = resolve_optional("experimental_newton_max_steps", None, default=100)
    experimental_exponential_newton_tol_progressive = resolve_optional(
        "experimental_exponential_newton_tol_progressive",
        None,
        default=True,
    )
    experimental_exponential_newton_tol_start = resolve_optional(
        "experimental_exponential_newton_tol_start",
        None,
        default=1e-5,
    )
    experimental_exponential_newton_tol_end = resolve_optional(
        "experimental_exponential_newton_tol_end",
        None,
        default=1e-5,
    )
    experimental_exponential_newton_tol_switch_hi = resolve_optional(
        "experimental_exponential_newton_tol_switch_hi",
        None,
        default=1e-2,
    )
    experimental_exponential_newton_tol_switch_lo = resolve_optional(
        "experimental_exponential_newton_tol_switch_lo",
        None,
        default=5e-4,
    )
    single_diode_updater = config.get("single_diode_updater")
    if non_linearity == "single_diode_exponential" and not single_diode_updater:
        raise SystemExit("Config must define 'single_diode_updater'.")
    rel_tol = resolve_optional("rel_tol", None, default=1e-5)
    vn_tol = resolve_optional("vn_tol", None, default=1e-6)
    residual_current_tol = resolve_optional("residual_current_tol", None, default=None)
    use_polish = resolve_optional("use_polish", None, default=True)
    max_newton_iters = resolve_optional("max_newton_iters", None, default=32)
    z_thresh = resolve_optional("z_thresh", None, default=1e10)
    if non_linearity in ("single_diode_exponential", "double_diode_exponential"):
        exp_clip = resolve("exp_clip", None)
    else:
        exp_clip = resolve_optional("exp_clip", None, default=100000.0)
    dynamic_polish = resolve_optional("dynamic_polish", None, default=False)
    minimizer_impl = resolve_optional("minimizer_impl", None, default=None)
    if not minimizer_impl:
        raise SystemExit("Config must define 'minimizer_impl'.")
    if minimizer_impl not in ("custom", "andersson"):
        raise SystemExit(
            "minimizer_impl must be one of: custom, andersson; "
            f"got {minimizer_impl!r}"
        )
    training_algorithm = resolve_optional("training_algorithm", args.training_algorithm, default="EP")
    if training_algorithm is None:
        training_algorithm = "EP"
    training_algorithm = str(training_algorithm).upper()
    if training_algorithm not in ("EP", "BP"):
        raise SystemExit(
            "training_algorithm must be 'EP' or 'BP'. "
            f"Got {training_algorithm!r}."
        )
    anderson_m = resolve_optional("anderson_m", None, default=8)
    anderson_omega = resolve_optional("anderson_omega", None, default=1.0)
    anderson_tol_floor = resolve_optional("anderson_tol_floor", None, default=5e-3)
    anderson_reg = resolve_optional("anderson_reg", None, default=1e-8)
    use_tensorboard = resolve_optional("use_tensorboard", args.use_tensorboard, default=True)
    tensorboard_dir = resolve_optional("tensorboard_dir", args.tensorboard_dir, default=None)
    save_epochs = resolve_optional("save_epochs", None, default=None)
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
    save_every_epoch = bool(resolve_optional("save_every_epoch", args.save_every_epoch, default=False))
    save_test_accuracy_threshold = resolve_optional(
        "save_test_accuracy_threshold",
        args.save_test_accuracy_threshold,
        default=None,
    )
    if save_test_accuracy_threshold is not None:
        try:
            save_test_accuracy_threshold = float(save_test_accuracy_threshold)
        except (TypeError, ValueError) as exc:
            raise SystemExit(
                "Expected save_test_accuracy_threshold to be a float in [0, 1]. "
                f"Provided value: {save_test_accuracy_threshold!r}."
            ) from exc
        if not (0.0 <= save_test_accuracy_threshold <= 1.0):
            raise SystemExit(
                "Expected save_test_accuracy_threshold to be a float in [0, 1]. "
                f"Provided value: {save_test_accuracy_threshold!r}."
            )
    save_test_accuracy_thresholds = resolve_optional(
        "save_test_accuracy_thresholds",
        args.save_test_accuracy_thresholds,
        default=None,
    )
    if save_test_accuracy_thresholds is not None:
        if not isinstance(save_test_accuracy_thresholds, (list, tuple)):
            save_test_accuracy_thresholds = [save_test_accuracy_thresholds]
        parsed_thresholds = []
        for raw_threshold in save_test_accuracy_thresholds:
            try:
                parsed_threshold = float(raw_threshold)
            except (TypeError, ValueError) as exc:
                raise SystemExit(
                    "Expected save_test_accuracy_thresholds to be floats in [0, 1]. "
                    f"Provided value: {save_test_accuracy_thresholds!r}."
                ) from exc
            if not (0.0 <= parsed_threshold <= 1.0):
                raise SystemExit(
                    "Expected save_test_accuracy_thresholds to be floats in [0, 1]. "
                    f"Provided value: {save_test_accuracy_thresholds!r}."
                )
            parsed_thresholds.append(parsed_threshold)
        save_test_accuracy_thresholds = sorted(set(parsed_thresholds))
    elif save_test_accuracy_threshold is not None:
        save_test_accuracy_thresholds = [save_test_accuracy_threshold]

    non_linearity_data = NonLinearityData(
        non_linearity=non_linearity,
        quadratic_diode_param=quadratic_params,
        exponential_diode_param=exponential_params,
        hard_sigmoid_param=hard_sigmoid_params,
        iv_data_path=iv_data_path,
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
        experimental_newton_max_steps=int(experimental_newton_max_steps),
        experimental_exponential_newton_tol_progressive=bool(
            experimental_exponential_newton_tol_progressive
        ),
        experimental_exponential_newton_tol_start=float(experimental_exponential_newton_tol_start),
        experimental_exponential_newton_tol_end=float(experimental_exponential_newton_tol_end),
        experimental_exponential_newton_tol_switch_hi=float(experimental_exponential_newton_tol_switch_hi),
        experimental_exponential_newton_tol_switch_lo=float(experimental_exponential_newton_tol_switch_lo),
        double_diode_runtime=double_diode_runtime,
    )

    if mode == "linspace":
        linspace_min = resolve("linspace_min", args.linspace_min)
        linspace_max = resolve("linspace_max", args.linspace_max)
        linspace_samples = resolve("linspace_samples", args.linspace_samples)
        linspace_settings = LinspaceSettings(
            linspace_min=linspace_min,
            linspace_max=linspace_max,
            linspace_samples=linspace_samples,
        )
        linspace_run_settings = LinspaceRunSettings(
            dims=dims_config,
            batch_size=batch_size,
            num_iterations=num_iterations,
            num_epochs=num_epochs,
            seed=seed,
            device=device,
            voltage_amp=voltage_amp,
            current_amp=current_amp,
            input_gain=input_gain,
            weight_gains=weight_gains,
            weight_min=weight_min,
            weight_max=weight_max,
            dataset_name=dataset_name,
            linspace_settings=linspace_settings,
            non_linearity_data=non_linearity_data,
            minimizer_settings=minimizer_settings,
        )
        moons_linspace(
            settings=linspace_run_settings,
            output_root=output_root,
            run_subdir=run_subdir,
            weights_path=run_weights,
            cli_command=cli_command,
        )
        return

    if mode == "digits_validate":
        linspace_min_validate = resolve_optional("linspace_min", args.linspace_min, default=None)
        linspace_max_validate = resolve_optional("linspace_max", args.linspace_max, default=None)
        linspace_samples_validate = resolve_optional("linspace_samples", args.linspace_samples, default=None)
        validate_settings = ValidateRunSettings(
            dims=dims_config,
            batch_size=batch_size,
            num_iterations=num_iterations,
            num_epochs=num_epochs,
            num_points=DEFAULT_NUM_POINTS,
            seed=seed,
            device=device,
            voltage_amp=voltage_amp,
            current_amp=current_amp,
            linspace_min=linspace_min_validate,
            linspace_max=linspace_max_validate,
            linspace_samples=linspace_samples_validate,
            input_gain=input_gain,
            weight_gains=weight_gains,
            weight_min=weight_min,
            weight_max=weight_max,
            dataset_name=dataset_name,
            non_linearity_data=non_linearity_data,
            minimizer_settings=minimizer_settings,
        )
        digits_validate(
            settings=validate_settings,
            output_root=output_root,
            run_subdir=run_subdir,
            weights_path=run_weights,
            cli_command=cli_command,
        )
        return

    num_points = DEFAULT_NUM_POINTS
    train_settings = TrainRunSettings(
        dims=dims_config,
        num_points=num_points,
        batch_size=batch_size,
        num_iterations=num_iterations,
        num_epochs=num_epochs,
        seed=seed,
        device=device,
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
        use_tensorboard=use_tensorboard,
        tensorboard_dir=tensorboard_dir,
        save_epochs=save_epochs,
        save_every_epoch=save_every_epoch,
        save_test_accuracy_threshold=save_test_accuracy_threshold,
        save_test_accuracy_thresholds=save_test_accuracy_thresholds,
        update_pipeline=update_pipeline,
    )
    train(
        settings=train_settings,
        output_root=output_root,
        run_subdir=run_subdir,
        weights_path=run_weights,
    )


if __name__ == "__main__":
    main()
