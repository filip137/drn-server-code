import argparse
from dataclasses import dataclass
import importlib.util
from itertools import product
import json
import random
import sys
import time
from pathlib import Path
from typing import Optional
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

LABS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = LABS_DIR.parent
DATASETS_MODULE_PATH = PROJECT_ROOT / "datasets.py"
DEFAULT_OUTPUT_ROOT = Path("/home/filip/server_code/simulation_results/experiments_labs")

for path in (LABS_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.append(str(path))
from training.epoch import BetaSize
from model.resistive.network import DeepResistiveEnergy  # noqa: E402
from model.function.network import Network  # noqa: E402
from labs.custom_minimizer import CustomQuadraticMinimizer  # noqa: E402
from training.sgd import AugmentedFunction, EquilibriumProp  # noqa: E402
from model.function.cost import SquaredError, SquaredErrorPairedOutputs  # noqa: E402
from labs.common import (
    MnistParts,
    save_beta_summary,
    build_evaluator,
    LayerStateEvaluator,
    ResidualCurrentEvaluator,
)
from supporting.stats_snapshot import snapshot_stats
from supporting.pca_utils import prepare_pca_grid
from labs.compute_pca import (
    compute_pca,
    _resolve_pca_cache_path,
    _load_pca_cache,
    _save_pca_cache,
)
from labs.mnist_io_runs import _init_run_dir
from labs.mnist_monitor_training import track_training_statistics

TRACK_FULL_HISTORY = False
DEFAULT_PCA_STEPS = 30
DEFAULT_PCA_N_SIGMA = 3.0
DEFAULT_PCA_BATCH_SIZE = None
DEFAULT_PCA_INCLUDE_IDX = False


@dataclass
class PcaGridSpec:
    steps: int
    n_sigma: float
    batch_size: Optional[int] = None
    include_idx: bool = False
    as_train_loader: bool = False
    cache_path: Optional[str] = None


def load_json_config(config_path):
    path = Path(config_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    return json.loads(path.read_text())


def _load_project_datasets_module():
    module_name = "ebl_datasets"
    if module_name in sys.modules:
        return sys.modules[module_name]
    if not DATASETS_MODULE_PATH.exists():
        raise FileNotFoundError(f"Expected datasets.py at {DATASETS_MODULE_PATH}")
    spec = importlib.util.spec_from_file_location(module_name, DATASETS_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load datasets module from {DATASETS_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[module_name] = module
    return module

load_dataloaders = _load_project_datasets_module().load_dataloaders


def prepare_mnist(
    config_path: str,
    model_key: str,
    voltage_amp: float = 1.0,
    current_amp: float = 1.0,
    weights_path: str = None,
    seed: int = None,
    *,
    data_mode: str = "mnist",
    train_loader_override=None,
    test_loader_override=None,
    pca: Optional[PcaGridSpec] = None,
    verbose: bool = False,
):
    """Common MNIST setup returning a bundled MnistParts."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_json_config(config_path)
    model_cfg = dict(config["models"][model_key])  # shallow copy for local overrides
    training_cfg = config.get("training", {})

    dataset_name = training_cfg["dataset"]
    batch_size = training_cfg["batch_size"]
    normalize = training_cfg["normalize"]
    normalize_std = training_cfg["normalize_std"]
    augment_32 = training_cfg["augment_32x32"]
    if verbose:
        print(
            f"[prepare_mnist] model={model_key} data_mode={data_mode} "
            f"dataset={dataset_name} batch_size={batch_size} voltage_amp={voltage_amp} current_amp={current_amp}"
        )

    default_train_loader = None
    default_test_loader = None
    if train_loader_override is None or test_loader_override is None:
        default_train_loader, default_test_loader = load_dataloaders(
            dataset_name,
            batch_size,
            augment_32x32=augment_32,
            normalize=normalize,
            normalize_std=normalize_std,
        )
    train_loader = default_train_loader if train_loader_override is None else train_loader_override
    test_loader = default_test_loader if test_loader_override is None else test_loader_override

    layer_shapes = [tuple(shape) for shape in model_cfg["layer_shapes"]]
    energy_fn = DeepResistiveEnergy(
        layer_shapes=layer_shapes,
        weight_gains=model_cfg["weight_gains"],
        input_gain=model_cfg.get("input_gain"),
        non_linearity=model_cfg["non_linearity"],
        exponential_diode_param=dict(model_cfg.get("exponential_diode_param", {})),
        quadratic_diode_param=dict(model_cfg.get("quadratic_diode_param", {})),
        hard_sigmoid_param=dict(model_cfg.get("hard_sigmoid_param", {})),
        voltage_amp=voltage_amp,
        current_amp=current_amp,
        weight_min=model_cfg.get("weight_min"),
        weight_max=model_cfg.get("weight_max"),
        conv_pipeline=model_cfg.get("conv_pipeline"),
        pooling_mode=model_cfg.get("pooling_mode"),
    )
    energy_fn.set_device(device)
    if weights_path:
        energy_fn.load(weights_path)

    network = Network(energy_fn)
    free_layers = network.free_layers()
    output_layer = energy_fn.layers()[-1]
    if output_layer.shape[0] == 10:
        cost_fn = SquaredError(output_layer)
    elif output_layer.shape[0] == 20:
        cost_fn = SquaredErrorPairedOutputs(output_layer, 10)
    else:
        raise ValueError(f"Unsupported output size: {output_layer.shape}")

    minimizer_mode = config["energy_minimizer"]["mode"]

    # Learning rates are model-specific in this codebase and live under the model config.
    learning_rates = None
    if "learning_rates" in model_cfg:
        learning_rates = list(model_cfg["learning_rates"])
    elif "learning_rates_biases" in model_cfg and "learning_rates_weights" in model_cfg:
        learning_rates = list(model_cfg["learning_rates_biases"]) + list(model_cfg["learning_rates_weights"])

    if test_loader_override is None and data_mode == "pca_grid":
        if pca is None:
            raise ValueError("pca spec is required when data_mode='pca_grid'.")
        cache_path = _resolve_pca_cache_path(pca.cache_path, model_key, dataset_name)
        if cache_path and cache_path.exists():
            mu, eigvals_2, eigvecs_2, n_total = _load_pca_cache(cache_path)
            if verbose:
                print(f"[prepare_mnist] loaded PCA cache from {cache_path}")
        else:
            mu, eigvals_2, eigvecs_2, n_total = compute_pca(config_path, model_key)
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                _save_pca_cache(cache_path, mu, eigvals_2, eigvecs_2, n_total)
                if verbose:
                    print(f"[prepare_mnist] saved PCA cache to {cache_path}")
        side = 32 if augment_32 else 28
        pca_loader_batch_size = batch_size if pca.batch_size is None else pca.batch_size
        pca_loader = prepare_pca_grid(
            eigvals_2,
            eigvecs_2,
            mu,
            n_total,
            steps=pca.steps,
            n_sigma=pca.n_sigma,
            image_shape=(1, side, side),
            batch_size=pca_loader_batch_size,
        )
        if verbose:
            dataset_len = len(pca_loader.dataset) if hasattr(pca_loader, "dataset") else "?"
            print(
                f"[prepare_mnist] PCA grid prepared: steps={pca.steps} n_sigma={pca.n_sigma} "
                f"points={dataset_len} batch_size={pca_loader_batch_size} n_total={n_total}"
            )
        if pca.include_idx:
            x_tensor, y_tensor = pca_loader.dataset.tensors[:2]
            idx_tensor = torch.arange(x_tensor.shape[0], dtype=torch.long)
            pca_loader = DataLoader(
                TensorDataset(x_tensor, y_tensor, idx_tensor),
                batch_size=pca_loader.batch_size,
                shuffle=False,
            )
        test_loader = pca_loader
        if pca.as_train_loader:
            train_loader = pca_loader

    return MnistParts(
        energy_fn=energy_fn,
        network=network,
        cost_fn=cost_fn,
        train_loader=train_loader,
        test_loader=test_loader,
        free_layers=free_layers,
        minimizer_mode=minimizer_mode,
        model_cfg=model_cfg,
        training_cfg=training_cfg,
        learning_rates=learning_rates,
        scheduler=None,
        voltage_amp=voltage_amp,
        current_amp=current_amp,
        weights_path=weights_path,
    )


def _build_minimizer(parts: MnistParts, num_iterations: int, *, fn=None):
    energy_fn = parts.energy_fn if fn is None else fn
    return CustomQuadraticMinimizer(
        fn=energy_fn,
        free_layers=parts.free_layers,
        num_iterations=num_iterations,
        mode=parts.minimizer_mode,
        non_linearity=parts.model_cfg["non_linearity"],
        quadratic_diode_param=parts.model_cfg.get("quadratic_diode_param", {}),
        exponential_diode_param=parts.model_cfg.get("exponential_diode_param", {}),
        hard_sigmoid_param=parts.model_cfg.get("hard_sigmoid_param", {}),
        voltage_amp=energy_fn._voltage_amp,
        current_amp=energy_fn._current_amp,
    )


def _compute_beta_summary(
    parts: MnistParts,
    num_iterations: int,
    *,
    max_batches: Optional[int] = None,
    verbose: bool = False,
    nudging_override: Optional[float] = None,
) -> dict:
    """Measure free->nudged displacement (beta-size) on parts.test_loader."""
    energy_fn = parts.energy_fn
    network = parts.network
    cost_fn = parts.cost_fn
    model_cfg = parts.model_cfg
    training_cfg = parts.training_cfg
    params = energy_fn.params()

    energy_minimizer_inference = _build_minimizer(parts, num_iterations)
    augmented_fn = AugmentedFunction(energy_fn, cost_fn)
    energy_minimizer_training = _build_minimizer(parts, num_iterations, fn=augmented_fn)

    nudging = nudging_override if nudging_override is not None else model_cfg.get("nudging", training_cfg.get("nudging"))
    ep_variant = training_cfg.get("ep_variant")
    ep_kwargs = {}
    if nudging is not None:
        ep_kwargs["nudging"] = nudging
    if ep_variant is not None:
        ep_kwargs["variant"] = ep_variant
    estimator = EquilibriumProp(
        params, parts.free_layers, augmented_fn, cost_fn, energy_minimizer_training, **ep_kwargs
    )

    beta = BetaSize(
        network=network,
        cost_fn=cost_fn,
        params=params,
        dataloader=parts.test_loader,
        differentiator=estimator,
        energy_minimizer=energy_minimizer_inference,
        max_batches=max_batches,
    )
    beta.run(verbose=verbose)
    beta_summary = beta.summary()
    layer_names = beta_summary.get("layers", [])
    if layer_names:
        beta_summary["layers"] = [f"Layer_{i}" for i, _ in enumerate(layer_names)]
    return beta_summary


def prepare_mnist_parts_sweep(
    config_path: str,
    model_key: str,
    voltage_amps=(1.0,),
    current_amps=(1.0,),
    iteration_counts=(12,),
    weights_path: str = None,
    seed: int = None,
    *,
    data_mode: str = "pca_grid",
    pca: Optional[PcaGridSpec] = None,
    verbose: bool = True,
):
    """Build one MnistParts per sweep point (amps x iteration_counts)."""
    voltage_values = tuple(voltage_amps)
    current_values = tuple(current_amps)
    iteration_values = tuple(iteration_counts)
    if not voltage_values or not current_values or not iteration_values:
        raise ValueError("voltage_amps, current_amps and iteration_counts must be non-empty.")

    total_points = len(voltage_values) * len(current_values) * len(iteration_values)
    if verbose:
        print(
            f"[prepare_mnist_parts_sweep] building {total_points} sweep points "
            f"(voltage={voltage_values}, current={current_values}, iterations={iteration_values})"
        )

    # Build loaders once and reuse them across sweep points so the PCA grid is not recomputed.
    shared_train_loader = None
    shared_test_loader = None
    if data_mode == "pca_grid":
        if pca is None:
            raise ValueError("pca spec is required when data_mode='pca_grid'.")
        effective_pca = PcaGridSpec(
            steps=pca.steps,
            n_sigma=pca.n_sigma,
            batch_size=pca.batch_size,
            include_idx=True,
            as_train_loader=pca.as_train_loader,
            cache_path=pca.cache_path,
        )
        if verbose and not pca.include_idx:
            print("[prepare_mnist_parts_sweep] forcing pca.include_idx=True for evaluator compatibility.")
        shared_parts = prepare_mnist(
            config_path=config_path,
            model_key=model_key,
            voltage_amp=float(voltage_values[0]),
            current_amp=float(current_values[0]),
            weights_path=None,
            seed=seed,
            data_mode=data_mode,
            pca=effective_pca,
            verbose=False,
        )
        shared_train_loader = shared_parts.train_loader
        shared_test_loader = shared_parts.test_loader

    parts_list = []
    for idx, (voltage_amp, current_amp, num_iterations) in enumerate(
        product(voltage_values, current_values, iteration_values),
        start=1,
    ):
        if verbose:
            print(
                f"[prepare_mnist_parts_sweep] [{idx}/{total_points}] "
                f"v={float(voltage_amp):g} c={float(current_amp):g} iters={int(num_iterations)}"
            )
        parts = prepare_mnist(
            config_path=config_path,
            model_key=model_key,
            voltage_amp=float(voltage_amp),
            current_amp=float(current_amp),
            weights_path=weights_path,
            seed=seed,
            data_mode=data_mode,
            train_loader_override=shared_train_loader,
            test_loader_override=shared_test_loader,
            pca=effective_pca if data_mode == "pca_grid" else pca,
            verbose=False,
        )
        parts.num_iterations = int(num_iterations)
        parts_list.append(parts)
    return parts_list


def sweep_MnistParts(
    sweep_parts,
    weights_path=None,
    record_statistics=("calc_residual_current",),
    verbose=True,
    output_dir: Optional[Path] = None,
    track_beta: bool = False,
    beta_max_batches: Optional[int] = None,
    beta_only: bool = False,
    beta_nudging: Optional[float] = None,
    record: Optional[str] = None,
):
    """Run a single PCA sweep per MnistParts and collect requested outputs."""
    sweep_parts = list(sweep_parts)
    results = []
    total_points = len(sweep_parts)
    if beta_only and not track_beta:
        raise ValueError("beta_only=True requires track_beta=True.")
    if record is None:
        raise ValueError("record is required; choose one of: stats, residuals, states.")
    if record not in ("stats", "residuals", "states"):
        raise ValueError("record must be one of: stats, residuals, states.")
    for idx, parts in enumerate(sweep_parts, start=1):
        energy_fn = parts.energy_fn
        load_path = weights_path or parts.weights_path
        if load_path:
            energy_fn.load(load_path)

        if parts.num_iterations is None:
            raise ValueError("Each MnistParts in sweep_parts must define num_iterations.")
        if verbose:
            pca_batch = getattr(parts.test_loader, "batch_size", None)
            print(
                f"[sweep_MnistParts] [{idx}/{total_points}] "
                f"v={parts.voltage_amp:g} c={parts.current_amp:g} iters={parts.num_iterations} "
                f"test_batch_size={pca_batch}"
            )

        stats_snapshot = {}
        residual_current = {}
        layer_states = {}
        if not beta_only:
            sweep_result = sweep_pca(
                parts,
                parts.num_iterations,
                output=record,
                verbose=verbose,
                record_statistics=record_statistics,
            )
            if record == "stats":
                stats_snapshot = sweep_result or {}
            elif record == "residuals":
                residual_current = sweep_result or {}
            elif record == "states":
                layer_states = sweep_result or {}
        result = {
            "voltage_amp": parts.voltage_amp,
            "current_amp": parts.current_amp,
            "iterations": parts.num_iterations,
        }
        if stats_snapshot:
            result["stats_snapshot"] = stats_snapshot
        if residual_current:
            result["residual_current"] = residual_current
        if track_beta:
            result["beta_summary"] = _compute_beta_summary(
                parts,
                num_iterations=int(parts.num_iterations),
                max_batches=beta_max_batches,
                verbose=verbose,
                nudging_override=beta_nudging,
            )
        if record == "states":
            result["layer_states"] = layer_states

        if output_dir is not None:
            point_dir = output_dir / (
                f"v{parts.voltage_amp:g}_c{parts.current_amp:g}_iter{parts.num_iterations}"
            )
            point_dir.mkdir(parents=True, exist_ok=True)
            if stats_snapshot:
                (point_dir / "stats_snapshot.json").write_text(json.dumps(stats_snapshot, indent=2))
                result["stats_path"] = str(point_dir / "stats_snapshot.json")
            if residual_current:
                (point_dir / "residual_current.json").write_text(json.dumps(residual_current, indent=2))
                result["residual_current_path"] = str(point_dir / "residual_current.json")
            if track_beta and "beta_summary" in result:
                save_beta_summary(result["beta_summary"], point_dir)
                result["beta_summary_path"] = str(point_dir / "beta_summary.json")
            if verbose:
                print(f"[sweep_MnistParts] wrote outputs to {point_dir}")

            if record == "states" and layer_states:
                arrays = {}
                for name, values in layer_states.items():
                    if values:
                        arrays[name] = torch.cat(values, dim=0).detach().cpu().numpy()
                if arrays:
                    layer_states_path = point_dir / "layer_states.npz"
                    np.savez(layer_states_path, **arrays)
                    result["layer_states_path"] = str(layer_states_path)

        results.append(result)

    if verbose:
        print(f"[sweep_MnistParts] finished {len(results)} sweep points")

    return results


def sweep_pca(
    parts: MnistParts,
    num_of_iterations,
    *,
    output: str = "states",
    verbose: bool = False,
    record_statistics=("calc_residual_current",),
):
    """
    Run inference over a PCA grid loader and collect either states or residual currents.

    Args:
        output: "states" (per-layer settled states) or "residuals" (per-layer residual currents).
    """
    energy_minimizer = _build_minimizer(parts, int(num_of_iterations))

    if output == "stats":
        evaluator = build_evaluator(
            parts.network,
            parts.cost_fn,
            parts.test_loader,
            energy_minimizer,
            parts.model_cfg,
            record_statistics,
        )
        evaluator.run(verbose=verbose)
        return snapshot_stats(evaluator._stats)

    if output == "states":
        evaluator = LayerStateEvaluator(
            parts.network,
            parts.cost_fn,
            parts.test_loader,
            energy_minimizer,
        )
        evaluator.run(verbose=verbose)
        return {
            name: torch.cat(values, dim=0)
            for name, values in evaluator.layer_states.items()
            if values
        }

    if output == "residuals":
        evaluator = ResidualCurrentEvaluator(
            parts.network,
            parts.cost_fn,
            parts.test_loader,
            energy_minimizer,
        )
        evaluator.run(verbose=verbose)
        return evaluator.res_currents

    raise ValueError("output must be 'stats', 'states' or 'residuals'.")


def _build_arg_parser() -> argparse.ArgumentParser:
    """CLI for running small MNIST experiments from this module."""
    p = argparse.ArgumentParser(prog="mnist_tests.py")

    # Common experiment wiring (kept shared across subcommands).
    p.add_argument("--config", required=True, help="Path to JSON config (e.g. config.used.json).")
    p.add_argument("--model-key", required=True, help="Key under config['models'] (e.g. drn-conv).")
    p.add_argument("--weights", default=None, help="Optional path to saved model weights (.pt).")
    p.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root directory for run outputs.",
    )
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    p.add_argument("--num-iterations", type=int, required=True, help="Minimizer iterations.")
    p.add_argument("--voltage-amp", type=float, default=1.0, help="Voltage amplification factor.")
    p.add_argument("--current-amp", type=float, default=1.0, help="Current amplification factor.")
    p.add_argument("--verbose", action="store_true", help="Verbose per-batch logging.")
    p.add_argument(
        "--pca-cache",
        default=None,
        help=(
            "Optional .pt path to cache PCA components. "
            "If omitted and using PCA grid, defaults to labs/datasets/pca_<dataset>_<model>.pt."
        ),
    )

    sub = p.add_subparsers(dest="cmd")
    p.set_defaults(cmd="train-stats")

    train_stats = sub.add_parser("train-stats", help="Train for a few epochs while tracking residual currents.")
    train_stats.add_argument("--num-epochs", type=int, default=None, help="Number of training epochs (default: 1 if missing).")
    train_stats.add_argument(
        "--record-statistics",
        nargs="*",
        default=("calc_residual_current",),
        help="Optional recordings (e.g. calc_residual_current).",
    )
    train_stats.add_argument(
        "--out",
        default=None,
        help="Optional JSON path to write residual currents (if omitted, prints a short summary).",
    )

    sub.add_parser(
        "pca-sweep",
        help="Run inference over a PCA grid and save layer states/residual currents.",
    )

    pca_sweep_loop = sub.add_parser(
        "pca-sweep-loop",
        help="Run PCA sweeps over one selected variable and save each value to a folder.",
    )
    pca_sweep_loop.add_argument(
        "--swept_variable",
        choices=("voltage_amp", "current_amp", "num_iterations"),
        required=True,
        help="Variable to sweep.",
    )
    pca_sweep_loop.add_argument(
        "--sweep_values",
        nargs="+",
        required=True,
        help="Values for the swept variable (space-separated).",
    )

    return p


def _parse_sweep_values(raw_values, swept_variable):
    if swept_variable == "num_iterations":
        values = tuple(int(v) for v in raw_values)
        if any(v <= 0 for v in values):
            raise ValueError("num_iterations sweep values must be > 0.")
        return values
    return tuple(float(v) for v in raw_values)


def _format_swept_value(value) -> str:
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        value = value.item()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:g}"
    return str(value).replace("/", "_")


def main(argv: Optional[list[str]] = None) -> int:
    """
    Thin CLI dispatcher.

    Keep experiment logic in helpers (`prepare_mnist`, `track_training_statistics`, ...),
    and keep this function focused on: parse args → build parts → dispatch → save minimal outputs.
    """
    args = _build_arg_parser().parse_args(argv)
    run_dir = _init_run_dir(args)

    if args.cmd == "train-stats":
        parts = prepare_mnist(
            config_path=args.config,
            model_key=args.model_key,
            voltage_amp=args.voltage_amp,
            current_amp=args.current_amp,
            weights_path=args.weights,
            seed=args.seed,
            verbose=args.verbose,
        )
        res_currents = track_training_statistics(
            parts,
            num_iterations=args.num_iterations,
            weights_path=args.weights,
            build_minimizer_fn=_build_minimizer,
            record_statistics=tuple(args.record_statistics),
            verbose=args.verbose,
            num_epochs=args.num_epochs,
            output_dir=run_dir,
        )
        out_path = Path(args.out).expanduser() if args.out else run_dir / "residual_currents.json"
        if not out_path.is_absolute():
            out_path = run_dir / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(res_currents, indent=2))
        train_layers = len(res_currents.get("train", {}))
        test_layers = len(res_currents.get("test", {}))
        print(f"Wrote residual currents to {out_path} (train layers={train_layers}, test layers={test_layers}).")
        return 0

    if args.cmd == "pca-sweep":
        parts = prepare_mnist(
            config_path=args.config,
            model_key=args.model_key,
            voltage_amp=args.voltage_amp,
            current_amp=args.current_amp,
            weights_path=args.weights,
            seed=args.seed,
            data_mode="pca_grid",
            pca=PcaGridSpec(
                steps=DEFAULT_PCA_STEPS,
                n_sigma=DEFAULT_PCA_N_SIGMA,
                batch_size=DEFAULT_PCA_BATCH_SIZE,
                include_idx=DEFAULT_PCA_INCLUDE_IDX,
                cache_path=args.pca_cache,
            ),
            verbose=args.verbose,
        )
        pca_samples = None
        if hasattr(parts.test_loader, "dataset"):
            try:
                pca_samples = len(parts.test_loader.dataset)
            except TypeError:
                pca_samples = None
        start_time = time.perf_counter()
        layer_states = sweep_pca(parts, args.num_iterations, output="states", verbose=args.verbose)
        elapsed = time.perf_counter() - start_time
        states_path = run_dir / f"pca_sweep_iter{args.num_iterations}.npz"
        states_path.parent.mkdir(parents=True, exist_ok=True)
        state_arrays = {name: tensor.detach().cpu().numpy() for name, tensor in layer_states.items()}
        np.savez(states_path, **state_arrays)
        print(f"Wrote PCA sweep states to {states_path}")

        residuals = sweep_pca(parts, args.num_iterations, output="residuals", verbose=args.verbose)
        residuals_path = run_dir / f"residuals_iter{args.num_iterations}.npz"
        np.savez(residuals_path, **{k: np.asarray(v) for k, v in residuals.items()})
        print(f"Wrote PCA sweep residuals to {residuals_path}")
        if pca_samples:
            per_sample = elapsed / pca_samples
            print(
                f"[pca-sweep] elapsed={elapsed:.3f}s samples={pca_samples} "
                f"avg_per_sample={per_sample:.6f}s"
            )
            try:
                dataset_name = parts.training_cfg.get("dataset", "MNIST")
                info_path = run_dir / "run_info.json"
                info = json.loads(info_path.read_text()) if info_path.exists() else {}
                info.update(
                    {
                        "pca_sweep_seconds": elapsed,
                        "pca_samples": pca_samples,
                        "pca_seconds_per_sample": per_sample,
                        "pca_steps": DEFAULT_PCA_STEPS,
                        "pca_n_sigma": DEFAULT_PCA_N_SIGMA,
                        "pca_batch_size": DEFAULT_PCA_BATCH_SIZE,
                        "pca_num_iterations": args.num_iterations,
                        "pca_cache_path": str(
                            _resolve_pca_cache_path(args.pca_cache, args.model_key, dataset_name)
                        ),
                    }
                )
                info_path.write_text(json.dumps(info, indent=2))
            except Exception as exc:
                print(f"[pca-sweep] warning: failed to update run_info.json: {exc}")
        return 0

    if args.cmd == "pca-sweep-loop":
        sweep_values = _parse_sweep_values(args.sweep_values, args.swept_variable)
        voltage_amps = (args.voltage_amp,)
        current_amps = (args.current_amp,)
        iteration_counts = (args.num_iterations,)
        if args.swept_variable == "voltage_amp":
            voltage_amps = sweep_values
        elif args.swept_variable == "current_amp":
            current_amps = sweep_values
        else:
            iteration_counts = sweep_values

        sweep_parts = prepare_mnist_parts_sweep(
            config_path=args.config,
            model_key=args.model_key,
            voltage_amps=voltage_amps,
            current_amps=current_amps,
            iteration_counts=iteration_counts,
            weights_path=args.weights,
            seed=args.seed,
            data_mode="pca_grid",
            pca=PcaGridSpec(
                steps=DEFAULT_PCA_STEPS,
                n_sigma=DEFAULT_PCA_N_SIGMA,
                batch_size=DEFAULT_PCA_BATCH_SIZE,
                include_idx=DEFAULT_PCA_INCLUDE_IDX,
                cache_path=args.pca_cache,
            ),
            verbose=args.verbose,
        )

        base_dir = run_dir / "pca_sweep_loop"
        base_dir.mkdir(parents=True, exist_ok=True)
        for parts in sweep_parts:
            if not hasattr(parts, args.swept_variable):
                raise AttributeError(
                    f"MnistParts has no attribute '{args.swept_variable}' for folder naming."
                )
            swept_value = getattr(parts, args.swept_variable)
            value_tag = _format_swept_value(swept_value)

            layer_states = sweep_pca(parts, parts.num_iterations, output="states", verbose=args.verbose)
            iter_dir = base_dir / f"{args.swept_variable}_{value_tag}"
            iter_dir.mkdir(parents=True, exist_ok=True)
            out_path = iter_dir / f"pca_sweep_iter{parts.num_iterations}.npz"
            arrays = {name: tensor.detach().cpu().numpy() for name, tensor in layer_states.items()}
            np.savez(out_path, **arrays)
            print(f"Wrote PCA sweep states to {out_path}")
            residuals = sweep_pca(parts, parts.num_iterations, output="residuals", verbose=args.verbose)
            residuals_path = iter_dir / f"residuals_iter{parts.num_iterations}.npz"
            np.savez(residuals_path, **{k: np.asarray(v) for k, v in residuals.items()})
            print(f"Wrote PCA sweep residuals to {residuals_path}")
        return 0

    raise RuntimeError(f"Unhandled command: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
