import argparse
import json
import random
import shlex
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

LABS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = LABS_DIR.parent
for path in (LABS_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from labs.datasets import LinSpaceDataset, MoonsDataset  # noqa: E402
from labs.common import export_pt_to_npz  # noqa: E402
from model.resistive.network import DeepResistiveEnergy  # noqa: E402
from model.function.network import Network  # noqa: E402
from labs.custom_minimizer import CustomQuadraticMinimizer  # noqa: E402
from training.epoch import Trainer, Evaluator  # noqa: E402
from training.monitor import Optimizer  # noqa: E402
from training.sgd import AugmentedFunction, EquilibriumProp  # noqa: E402
from model.function.cost import SquaredError, SquaredErrorPairedOutputs  # noqa: E402
from model.variable.parameter import Bias  # noqa: E402
from training.statistics import Counter, EnergyStat, CostStat, ErrorStat, TopFiveErrorStat  # noqa: E402


def _set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def load_json_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return json.loads(config_path.read_text())


def _require_config_dict(config: dict, name: str) -> dict:
    value = config.get(name)
    if value is None:
        raise SystemExit(f"Config must define '{name}'.")
    if not isinstance(value, dict):
        raise SystemExit(f"Config '{name}' must be an object.")
    return value


def _require_keys(params: dict, keys: Sequence[str], label: str) -> None:
    missing = [key for key in keys if key not in params]
    if missing:
        raise SystemExit(f"Config '{label}' missing keys: {', '.join(missing)}")


_NON_LINEARITY_REQUIREMENTS = {
    "double_diode_quadratic": ("quadratic_diode_param", ["diode_conductance", "v_off"]),
    "double_diode_exponential": ("exponential_diode_param", ["I_s", "V_t", "V_off"]),
    "double_diode": ("hard_sigmoid_param", ["g_on", "g_off", "v_min", "v_max"]),
    "hard_sigmoid": ("hard_sigmoid_param", ["g_on", "g_off", "v_min", "v_max"]),
}


def _parse_non_linearity_params(config: dict, non_linearity: str) -> tuple[dict, dict, dict]:
    quadratic_params = config.get("quadratic_diode_param", {})
    exponential_params = config.get("exponential_diode_param", {})
    hard_sigmoid_params = config.get("hard_sigmoid_param", {})

    requirement = _NON_LINEARITY_REQUIREMENTS.get(non_linearity)
    if requirement:
        param_name, keys = requirement
        params = _require_config_dict(config, param_name)
        _require_keys(params, keys, param_name)
        if param_name == "quadratic_diode_param":
            quadratic_params = params
        elif param_name == "exponential_diode_param":
            exponential_params = params
        else:
            hard_sigmoid_params = params

    return quadratic_params, exponential_params, hard_sigmoid_params


def _resolve_config_value(name: str, cli_value, config: dict):
    if cli_value is not None:
        return cli_value
    if name in config:
        return config[name]
    cli_flag = f"--{name.replace('_', '-')}"
    raise SystemExit(f"Missing required value for '{name}'. Provide it in the config or via {cli_flag}.")


def _resolve_optional_config_value(name: str, cli_value, config: dict, default=None):
    if cli_value is not None:
        return cli_value
    if name in config:
        return config[name]
    return default


def _resolve_config_list(name: str, cli_value, config: dict) -> list:
    resolved = _resolve_config_value(name, cli_value, config)
    if resolved is None:
        return []
    if isinstance(resolved, (list, tuple)):
        return list(resolved)
    return [resolved]


def _parse_layer_shapes(layer_shapes) -> tuple[int, list[int], int, list[tuple[int, ...]]]:
    if not isinstance(layer_shapes, (list, tuple)) or len(layer_shapes) < 2:
        raise SystemExit("Config 'layer_shapes' must be a list with at least input and output shapes.")
    parsed = []
    for shape in layer_shapes:
        if isinstance(shape, (list, tuple)):
            if len(shape) != 1 or not isinstance(shape[0], int):
                raise SystemExit("Each entry in 'layer_shapes' must be a single integer list, e.g. [4].")
            parsed.append((shape[0],))
        elif isinstance(shape, int):
            parsed.append((shape,))
        else:
            raise SystemExit("Each entry in 'layer_shapes' must be an int or single-int list.")

    input_size = parsed[0][0]
    if input_size % 2 != 0:
        raise SystemExit("Input layer size must be divisible by 2 to infer input_dim.")
    input_dim = input_size // 2
    hidden_dims = [shape[0] for shape in parsed[1:-1]]
    output_dim = parsed[-1][0]
    return input_dim, hidden_dims, output_dim, parsed


def _build_hidden_runs(hidden_dims: list) -> list[tuple[int, list[int], Optional[int]]]:
    if not hidden_dims:
        raise SystemExit("Config 'layer_shapes' must include at least one hidden layer.")
    return [(len(hidden_dims), hidden_dims, None)]


def _create_run_dir(
    output_root: str,
    run_subdir: Optional[str],
    non_linearity: str,
    linspace_only: bool,
) -> Path:
    output_root_path = Path(output_root).expanduser()
    output_root_path.mkdir(parents=True, exist_ok=True)
    suffix = "_linspace" if linspace_only else ""
    run_name = datetime.now().strftime("%Y%m%d-%H%M%S") + f"_{non_linearity}{suffix}"
    run_dir = output_root_path / run_name if run_subdir is None else output_root_path / run_subdir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _find_latest_model(weights_root: Path, hidden_count: int) -> Path:
    candidates = sorted(
        (weights_root / f"hidden_{hidden_count}").glob("*/model.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No model.pt found under {weights_root}/hidden_{hidden_count}")
    return candidates[0]


def _build_energy_stack(
    *,
    device: torch.device,
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    non_linearity: str,
    voltage_amp: float,
    current_amp: float,
    quadratic_diode_param: Optional[dict],
    exponential_diode_param: Optional[dict],
    hard_sigmoid_param: Optional[dict],
    weights_path: Optional[str],
):
    if not non_linearity:
        raise SystemExit("non_linearity must be provided via config.")
    layer_shapes = [(input_dim * 2,)] + [(dim,) for dim in hidden_dims] + [(output_dim,)]
    weight_gains = [1] * (len(layer_shapes) - 1)
    input_gain = 10.0
    quadratic_params = dict(quadratic_diode_param or {})
    exponential_params = dict(exponential_diode_param or {})
    hard_sigmoid_params = dict(hard_sigmoid_param or {})
    weight_min, weight_max = 1e-4, 100.0

    energy_fn = DeepResistiveEnergy(
        layer_shapes=layer_shapes,
        weight_gains=weight_gains,
        input_gain=input_gain,
        non_linearity=non_linearity,
        exponential_diode_param=exponential_params,
        quadratic_diode_param=quadratic_params,
        hard_sigmoid_param=hard_sigmoid_params,
        voltage_amp=voltage_amp,
        current_amp=current_amp,
        weight_min=weight_min,
        weight_max=weight_max,
    )
    energy_fn.set_device(device)
    if weights_path:
        energy_fn.load(Path(weights_path).expanduser())

    network = Network(energy_fn)
    free_layers = network.free_layers()
    output_layer = energy_fn.layers()[-1]
    if output_layer.shape[0] == 2:
        cost_fn = SquaredError(output_layer)
    elif output_layer.shape[0] == 4:
        cost_fn = SquaredErrorPairedOutputs(output_layer, 2)
    else:
        raise ValueError(f"Unsupported output size: {output_layer.shape}")

    return (
        energy_fn,
        network,
        free_layers,
        cost_fn,
        output_layer,
        layer_shapes,
        input_gain,
        quadratic_params,
        exponential_params,
        hard_sigmoid_params,
    )


def _attach_stats(trainer, evaluator, energy_fn, cost_fn, train_loader, test_loader, output_dim):
    train_counter = Counter(energy_fn, len(train_loader.dataset))
    train_counter_post = Counter(energy_fn, len(train_loader.dataset))
    train_counter_post.display = False
    eval_counter = Counter(energy_fn, len(test_loader.dataset))

    core_stats = [EnergyStat(energy_fn), CostStat(cost_fn), ErrorStat(cost_fn)]
    if output_dim >= 5:
        core_stats.append(TopFiveErrorStat(cost_fn))

    trainer.add_statistic(train_counter, list_idx=0)
    trainer.add_statistic(train_counter_post, list_idx=1)
    for stat in core_stats:
        trainer.add_statistic(stat, list_idx=0)
        trainer.add_statistic(stat, list_idx=1)

    evaluator.add_statistic(eval_counter)
    for stat in core_stats:
        evaluator.add_statistic(stat)


def _report_non_finite_params(params, label: str) -> None:
    for param in params:
        state = param.state
        if torch.isfinite(state).all():
            continue
        nan_count = torch.isnan(state).sum().item()
        inf_count = torch.isinf(state).sum().item()
        name = getattr(param, "name", param.__class__.__name__)
        print(
            f"[nan-check] {label} {name}: nan={nan_count} inf={inf_count} "
            f"shape={tuple(state.shape)} dtype={state.dtype} device={state.device}"
        )
        break


def train(
    output_root: str,
    num_points: int,
    batch_size: int,
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    num_iterations: int,
    num_epochs: int,
    seed: Optional[int],
    voltage_amp: float,
    current_amp: float,
    run_subdir: Optional[str],
    weights_path: Optional[str],
    non_linearity: str,
    quadratic_diode_param: Optional[dict],
    exponential_diode_param: Optional[dict],
    hard_sigmoid_param: Optional[dict],
) -> tuple[Path, Path, Path]:
    _set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = MoonsDataset(name="moons", batch_size=batch_size, device=device, num_samples=num_points)
    train_loader, test_loader = dataset.build()

    (
        energy_fn,
        network,
        free_layers,
        cost_fn,
        output_layer,
        _layer_shapes,
        _input_gain,
        quadratic_params,
        exponential_params,
        hard_sigmoid_params,
    ) = _build_energy_stack(
        device=device,
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        non_linearity=non_linearity,
        voltage_amp=voltage_amp,
        current_amp=current_amp,
        quadratic_diode_param=quadratic_diode_param,
        exponential_diode_param=exponential_diode_param,
        hard_sigmoid_param=hard_sigmoid_param,
        weights_path=weights_path,
    )

    augmented_fn = AugmentedFunction(energy_fn, cost_fn)
    minimizer_mode = "asynchronous"

    energy_minimizer_training = CustomQuadraticMinimizer(
        fn=augmented_fn,
        free_layers=free_layers,
        num_iterations=num_iterations,
        mode=minimizer_mode,
        non_linearity=non_linearity,
        quadratic_diode_param=quadratic_params,
        exponential_diode_param=exponential_params,
        voltage_amp=voltage_amp,
        current_amp=current_amp,
        hard_sigmoid_param=hard_sigmoid_params,
    )

    energy_minimizer_inference = CustomQuadraticMinimizer(
        fn=energy_fn,
        free_layers=free_layers,
        num_iterations=num_iterations,
        mode=minimizer_mode,
        non_linearity=non_linearity,
        quadratic_diode_param=quadratic_params,
        exponential_diode_param=exponential_params,
        voltage_amp=voltage_amp,
        current_amp=current_amp,
        hard_sigmoid_param=hard_sigmoid_params,
    )

    params = energy_fn.params()
    estimator = EquilibriumProp(
        params,
        free_layers,
        augmented_fn,
        cost_fn,
        energy_minimizer_training,
        variant="centered",
        nudging=0,
    )

    learning_rate = 1e-3
    learning_rates = []
    for param in params:
        if isinstance(param, Bias):
            learning_rates.append(0.0)
        else:
            learning_rates.append(learning_rate)
    optimizer = Optimizer(
        energy_fn,
        cost_fn,
        learning_rates,
        momentum=0.0,
        weight_decay=0.0,
    )

    trainer = Trainer(
        network,
        cost_fn,
        params,
        train_loader,
        estimator,
        optimizer,
        energy_minimizer_inference,
    )
    evaluator = Evaluator(
        network,
        cost_fn,
        test_loader,
        energy_minimizer_inference,
    )
    _attach_stats(trainer, evaluator, energy_fn, cost_fn, train_loader, test_loader, output_layer.shape[0])

    run_dir = _create_run_dir(output_root, run_subdir, non_linearity, linspace_only=False)

    for epoch in range(1, num_epochs + 1):
        print(f"[train] epoch {epoch}/{num_epochs}")
        trainer.run(verbose=True)
        _report_non_finite_params(params, f"epoch {epoch} after train")
        evaluator.run(verbose=True)
        _report_non_finite_params(params, f"epoch {epoch} after eval")

    model_path = run_dir / "model.pt"
    energy_fn.save(model_path)
    model_npz_path = run_dir / "model.npz"
    export_pt_to_npz(model_path, model_npz_path)

    return run_dir, model_path, model_npz_path


def moons_linspace(
    *,
    output_root: str,
    batch_size: int,
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    num_iterations: int,
    num_epochs: int,
    seed: Optional[int],
    voltage_amp: float,
    current_amp: float,
    linspace_min: float,
    linspace_max: float,
    linspace_samples: int,
    run_subdir: Optional[str],
    weights_path: Optional[str],
    linspace_only: bool,
    non_linearity: str,
    quadratic_diode_param: Optional[dict],
    exponential_diode_param: Optional[dict],
    hard_sigmoid_param: Optional[dict],
    cli_command: Optional[str],
    run_dir: Optional[Path] = None,
    model_path: Optional[Path] = None,
    model_npz_path: Optional[Path] = None,
) -> Path:
    _set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    (
        energy_fn,
        network,
        free_layers,
        _cost_fn,
        _output_layer,
        layer_shapes,
        input_gain,
        quadratic_params,
        exponential_params,
        hard_sigmoid_params,
    ) = _build_energy_stack(
        device=device,
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        non_linearity=non_linearity,
        voltage_amp=voltage_amp,
        current_amp=current_amp,
        quadratic_diode_param=quadratic_diode_param,
        exponential_diode_param=exponential_diode_param,
        hard_sigmoid_param=hard_sigmoid_param,
        weights_path=weights_path,
    )

    energy_minimizer_inference = CustomQuadraticMinimizer(
        fn=energy_fn,
        free_layers=free_layers,
        num_iterations=num_iterations,
        mode="asynchronous",
        non_linearity=non_linearity,
        quadratic_diode_param=quadratic_params,
        exponential_diode_param=exponential_params,
        voltage_amp=voltage_amp,
        current_amp=current_amp,
        hard_sigmoid_param=hard_sigmoid_params,
    )

    if run_dir is None:
        run_dir = _create_run_dir(output_root, run_subdir, non_linearity, linspace_only)

    grid_dataset = LinSpaceDataset(
        name="linspace_eval",
        batch_size=batch_size,
        device=device,
        num_inputs=input_dim,
        min=linspace_min,
        max=linspace_max,
        num_samples=linspace_samples,
    )
    grid_loader = grid_dataset.build_mesh(
        xmin=linspace_min,
        xmax=linspace_max,
        ymin=linspace_min,
        ymax=linspace_max,
    )

    input_layer = network.layers()[0]
    original_input_gain = getattr(input_layer, "_gain", None)
    if original_input_gain is not None:
        input_layer._gain = 1.0
    state_history = {layer.name: [] for layer in network.layers()}
    residual_history = {layer.name: [] for layer in network.layers()}
    input_batches = []

    for batch in grid_loader:
        if isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch
        input_batches.append(x.detach().cpu())
        network.set_input(x, reset=True)
        input_layer.set_input(x, mode="test")
        energy_minimizer_inference.compute_equilibrium()
        for layer in network.layers():
            state_history[layer.name].append(layer.state.detach().cpu())
        fn = network._function
        for layer in fn.layers():
            grad = fn.grad_layer_fn(layer)()
            res_current = torch.norm(grad, p=2) / (grad.numel() ** 0.5)
            residual_history[layer.name].append(float(res_current.item()))

    if original_input_gain is not None:
        input_layer._gain = original_input_gain

    inputs_array = torch.cat(input_batches, dim=0).numpy()
    states_array = {name: torch.cat(history, dim=0).numpy() for name, history in state_history.items()}

    inputs_path = run_dir / "linspace_inputs.npz"
    states_path = run_dir / "linspace_states.npz"
    residuals_path = run_dir / "linspace_residual_currents.npz"
    np.savez(inputs_path, inputs=inputs_array)
    np.savez(states_path, **states_array)
    np.savez(residuals_path, **{name: np.asarray(values) for name, values in residual_history.items()})

    metadata = {
        "run_dir": str(run_dir),
        "non_linearity": non_linearity,
        "voltage_amp": voltage_amp,
        "current_amp": current_amp,
        "input_gain": input_gain,
        "quadratic_diode_param": quadratic_params,
        "exponential_diode_param": exponential_params,
        "hard_sigmoid_param": hard_sigmoid_params,
        "layer_shapes": layer_shapes,
        "num_iterations": num_iterations,
        "num_epochs": num_epochs,
        "linspace_min": linspace_min,
        "linspace_max": linspace_max,
        "linspace_samples": linspace_samples,
        "linspace_only": linspace_only,
        "weights_path": str(weights_path) if weights_path else None,
        "linspace_residual_currents": str(residuals_path),
    }
    if cli_command:
        metadata["cli_command"] = cli_command
    metadata_path = run_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    if not linspace_only:
        if model_path is not None:
            print(f"Saved model to {model_path}")
        if model_npz_path is not None:
            print(f"Saved weights to {model_npz_path}")
    print(f"Saved linspace inputs to {inputs_path}")
    print(f"Saved linspace states to {states_path}")
    print(f"Saved linspace residual currents to {residuals_path}")
    print(f"Saved metadata to {metadata_path}")

    return run_dir


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None, help="Path to JSON config (defaults to labs/small_network_config.json).")
    p.add_argument("--output-root", default=None)
    p.add_argument("--num-points", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--input-dim", type=int, default=None)
    p.add_argument("--hidden-dims", type=int, nargs="+", default=None)
    p.add_argument("--output-dim", type=int, default=None)
    p.add_argument("--num-iterations", type=int, default=None)
    p.add_argument("--num-epochs", type=int, default=None)
    p.add_argument("--linspace-only", action="store_true")
    p.add_argument("--weights-root", default=None)
    p.add_argument("--weights", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--voltage-amp", type=float, default=None)
    p.add_argument("--current-amp", type=float, default=None)
    p.add_argument("--linspace-min", type=float, default=None)
    p.add_argument("--linspace-max", type=float, default=None)
    p.add_argument("--linspace-samples", type=int, default=None)
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

    output_root = resolve("output_root", args.output_root)
    if args.input_dim is not None or args.hidden_dims is not None or args.output_dim is not None:
        raise SystemExit("Provide 'layer_shapes' in the config; input_dim/hidden_dims/output_dim are inferred.")
    if any(key in config for key in ("input_dim", "hidden_dims", "output_dim")):
        raise SystemExit("Config should define only 'layer_shapes' for dimensions.")
    input_dim, hidden_dims, output_dim, layer_shapes = _parse_layer_shapes(resolve("layer_shapes", None))
    linspace_only = bool(args.linspace_only) or bool(config.get("linspace_only", False))
    weights_root = resolve_optional("weights_root", args.weights_root)
    weights_path = resolve_optional("weights", args.weights)
    non_linearity = config.get("non_linearity")
    if not non_linearity:
        raise SystemExit("Config must define 'non_linearity' (no default applied).")
    quadratic_params, exponential_params, hard_sigmoid_params = _parse_non_linearity_params(
        config,
        non_linearity,
    )

    hidden_runs = _build_hidden_runs(hidden_dims)

    for count, dims, dim_value in hidden_runs:
        run_weights = None
        if linspace_only:
            if weights_path:
                run_weights = weights_path
            else:
                root = Path(weights_root or output_root).expanduser()
                run_weights = _find_latest_model(root, count)
        run_subdir = f"h{count}_d{dim_value}" if dim_value is not None else f"hidden_{count}"
        num_iterations = resolve("num_iterations", args.num_iterations)
        num_epochs = resolve("num_epochs", args.num_epochs)
        seed = resolve_optional("seed", args.seed)
        voltage_amp = resolve("voltage_amp", args.voltage_amp)
        current_amp = resolve("current_amp", args.current_amp)
        linspace_min = resolve("linspace_min", args.linspace_min)
        linspace_max = resolve("linspace_max", args.linspace_max)
        linspace_samples = resolve("linspace_samples", args.linspace_samples)

        if linspace_only:
            moons_linspace(
                output_root=output_root,
                batch_size=1,
                input_dim=input_dim,
                hidden_dims=dims,
                output_dim=output_dim,
                num_iterations=num_iterations,
                num_epochs=num_epochs,
                seed=seed,
                voltage_amp=voltage_amp,
                current_amp=current_amp,
                linspace_min=linspace_min,
                linspace_max=linspace_max,
                linspace_samples=linspace_samples,
                run_subdir=run_subdir,
                weights_path=run_weights,
                linspace_only=linspace_only,
                non_linearity=non_linearity,
                quadratic_diode_param=quadratic_params,
                exponential_diode_param=exponential_params,
                hard_sigmoid_param=hard_sigmoid_params,
                cli_command=cli_command,
            )
            continue

        num_points = resolve("num_points", args.num_points)
        run_dir, model_path, model_npz_path = train(
            output_root=output_root,
            num_points=num_points,
            batch_size=1,
            input_dim=input_dim,
            hidden_dims=dims,
            output_dim=output_dim,
            num_iterations=num_iterations,
            num_epochs=num_epochs,
            seed=seed,
            voltage_amp=voltage_amp,
            current_amp=current_amp,
            run_subdir=run_subdir,
            weights_path=run_weights,
            non_linearity=non_linearity,
            quadratic_diode_param=quadratic_params,
            exponential_diode_param=exponential_params,
            hard_sigmoid_param=hard_sigmoid_params,
        )
        moons_linspace(
            output_root=output_root,
            batch_size=1,
            input_dim=input_dim,
            hidden_dims=dims,
            output_dim=output_dim,
            num_iterations=num_iterations,
            num_epochs=num_epochs,
            seed=seed,
            voltage_amp=voltage_amp,
            current_amp=current_amp,
            linspace_min=linspace_min,
            linspace_max=linspace_max,
            linspace_samples=linspace_samples,
            run_subdir=run_subdir,
            weights_path=str(model_path),
            linspace_only=linspace_only,
            non_linearity=non_linearity,
            quadratic_diode_param=quadratic_params,
            exponential_diode_param=exponential_params,
            hard_sigmoid_param=hard_sigmoid_params,
            cli_command=cli_command,
            run_dir=run_dir,
            model_path=model_path,
            model_npz_path=model_npz_path,
        )


if __name__ == "__main__":
    main()
