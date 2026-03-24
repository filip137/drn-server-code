from dataclasses import dataclass
import json
import os
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from model.resistive.interaction import BasePoolResistive
from training.epoch import Trainer, Evaluator
from training.statistics import (
    Counter,
    CostStat,
    EnergyStat,
    ErrorStat,
    TopFiveErrorStat,
    NormStat,
    SaturationStat,
)


@dataclass
class MnistParts:
    energy_fn: object
    network: object
    cost_fn: torch.nn.Module
    train_loader: object
    test_loader: object
    free_layers: list
    minimizer_mode: str
    model_cfg: dict
    training_cfg: dict
    # Optional training wiring:
    # - `learning_rates` should align with `Optimizer`'s parameter list (after PoolWeight filtering).
    # - `scheduler` is created by training utilities (e.g. `track_training_statistics`), not during setup.
    learning_rates: Optional[list]
    scheduler: Optional[object]
    voltage_amp: float = 1.0
    current_amp: float = 1.0
    weights_path: Optional[str] = None
    num_iterations: Optional[int] = None


def build_evaluator(network, cost_fn, dataloader, energy_minimizer, model_cfg, record_statistics=()):
    """Construct a fresh Evaluator with core and per-layer stats."""
    evaluator = CustomEvaluator(network, cost_fn, dataloader, energy_minimizer, record_statistics)
    energy_source = getattr(network, "_function", None) or network
    if not hasattr(energy_source, "eval"):
        raise ValueError("energy_source for evaluator has no eval()")
    evaluator.add_statistic(Counter(energy_source, len(dataloader.dataset)))
    evaluator.add_statistic(EnergyStat(energy_source))
    evaluator.add_statistic(CostStat(cost_fn))
    evaluator.add_statistic(ErrorStat(cost_fn))
    evaluator.add_statistic(TopFiveErrorStat(cost_fn))

    quad_params = model_cfg.get("quadratic_diode_param", {})
    v_min = quad_params.get("v_min")
    v_max = quad_params.get("v_max")
    for layer in energy_source.layers():
        evaluator.add_statistic(NormStat(layer))
        evaluator.add_statistic(
            SaturationStat(
                layer,
                non_linearity=model_cfg["non_linearity"],
                v_min=v_min,
                v_max=v_max,
            )
        )
    evaluator._reset()
    return evaluator


class CustomTrainer(Trainer):
    """Trainer variant that tracks residual currents for debugging."""

    def __init__(
        self,
        network,
        cost_fn,
        params,
        dataloader,
        differentiator,
        optimizer,
        energy_minimizer,
        record_statistics=(),
    ):
        super().__init__(network, cost_fn, params, dataloader, differentiator, optimizer, energy_minimizer)
        self.record_statistics = tuple(record_statistics) if record_statistics is not None else ()
        self.res_currents = {}
        self.layer_states = {}





    
    def winner_hist_within_window(self, x, Kh, Kw, S, P=0, D=1):
        # x: (N,C,H,W)
        N, C, H, W = x.shape
        patches = F.unfold(x.abs(), (Kh, Kw), padding=P, stride=S, dilation=D)  # (N, C*Kh*Kw, L)
        patches = patches.view(N, C, Kh*Kw, -1)
        rel_idx = patches.argmax(dim=2)  # (N,C,L) index within window
        hist = torch.bincount(rel_idx.reshape(-1), minlength=Kh*Kw).float()
        hist = hist / hist.sum().clamp(min=1)
        return hist  # length Kh*Kw


        
    def absmax_margin_stats(self, x, Kh, Kw, stride, padding=0, dilation=1, tol=1e-6):
        # x: (N,C,H,W)
        N, C, H, W = x.shape
        patches = F.unfold(x.abs(), (Kh, Kw), padding=padding, stride=stride, dilation=dilation)
        # patches: (N, C*Kh*Kw, L) -> (N, C, Kh*Kw, L)
        patches = patches.view(N, C, Kh*Kw, -1)
        top2 = patches.topk(2, dim=2).values  # (N, C, 2, L)
        margin = top2[:, :, 0, :] - top2[:, :, 1, :]  # (N, C, L)

        frac_ties = (margin <= tol).float().mean().item()
        mean_margin = margin.mean().item()
        return mean_margin, frac_ties

    def run(self, verbose: bool = False):
        self._reset()
        self.res_currents = {}
        max_pooling_dict = {}
        winners_dict = {}
        debug = bool(os.environ.get("DRN_DEBUG_DIODE"))
        store_states = "store_states" in self.record_statistics
        if store_states:
            self.layer_states = {layer.name: [] for layer in self._network.layers()}
        else:
            self.layer_states = {}
        for x, y in self._dataloader:
            self._network.set_input(x, reset=False)
            self._energy_minimizer.compute_equilibrium()
            fn = self._network._function
            if debug:
                for layer in fn.layers():
                    contribs = []
                    for j, inter in enumerate(fn._interactions):
                        if layer not in inter.layers():
                            continue
                        grad = inter.grad_layer_fn(layer)()           # this interaction’s dE/dz for this layer
                        if isinstance(inter, BasePoolResistive):
                            analytical_grad = inter.analytical_grad_layer_fn(layer)
                            diff = analytical_grad() - grad
                        contribs.append((j, inter.__class__.__name__, grad))

                    # Example: print norms per interaction and the total
                    total = sum(g for *_, g in contribs)
                    print(f"[{layer.name}] total ||grad||={torch.norm(total):.4e}")
                    for j, name, g in contribs:
                        print(f"  {j}:{name} ||grad||={torch.norm(g):.4e}")
            if "calc_residual_current" in self.record_statistics:
                fn = self._network._function
                for layer in fn.layers():
                    grad = fn.grad_layer_fn(layer)()
                    # Residual current = ||dE/dz||_inf
                    res = grad.abs().max()
                    self.res_currents.setdefault(layer.name, []).append(float(res.item()))
                    if layer.name == "Layer_1" or layer.name == "Layer_3":
                        x = layer.state
                        mean_margin, frac_ties = self.absmax_margin_stats(x, Kh =2, Kw=2, stride=2, padding=0, dilation=1, tol=1e-6)
                        max_pooling_dict.setdefault(layer.name, []).append([float(mean_margin), float(frac_ties)])
                        winners = self.winner_hist_within_window(x, Kh = 2, Kw = 2, S = 2, P=0, D=1)
                        winners_dict.setdefault(layer.name, []).append(winners)
            if "store_states" in self.record_statistics:
                for i, layer in enumerate(self._network.layers()):
                    name = getattr(layer, "name", f"layer_{i}")
                    self.layer_states[name].append(layer.state.detach().cpu().clone())

            self._cost_fn.set_target(y)
            self._do_measurements(0)

            grads = self._differentiator.compute_gradient()
            for param, grad in zip(self._params, grads):
                if not torch.isfinite(grad).all():
                    if store_states and debug:
                        for i, layer in enumerate(self._network.layers()):
                            state = layer.state
                            if torch.isfinite(state).all():
                                continue
                            nan_count = torch.isnan(state).sum().item()
                            inf_count = torch.isinf(state).sum().item()
                            name = getattr(layer, "name", f"layer_{i}")
                            print(
                                f"[state-check] non-finite state in {name}: nan={nan_count} inf={inf_count} "
                                f"shape={tuple(state.shape)} dtype={state.dtype} device={state.device}"
                            )
                    nan_count = torch.isnan(grad).sum().item()
                    inf_count = torch.isinf(grad).sum().item()
                    name = getattr(param, "name", param.__class__.__name__)
                    import pdb
                    pdb.set_trace()
                    raise RuntimeError(
                        f"[grad-check] non-finite gradient for {name}: nan={nan_count} inf={inf_count} "
                        f"shape={tuple(grad.shape)} dtype={grad.dtype} device={grad.device}"
                    )
            for param, grad in zip(self._params, grads):
                param.state.grad = grad
            self._do_measurements(1)

            self.track_gradient_and_update_sizes(grads, verbose=debug)
            for param in self._params:
                param.clamp_()

            if verbose:
                print(f"\r{self}", end="", flush=True)

        if verbose:
            print(f"\r{self}", end="", flush=True)


class CustomEvaluator(Evaluator):
    """Evaluator variant that records per-layer states for each batch."""

    def __init__(self, network, cost_fn, dataloader, energy_minimizer, record_statistics):
        super().__init__(network, cost_fn, dataloader, energy_minimizer)
        self.layer_states = {}
        self.res_currents = {}
        self.record_statistics = record_statistics

    def run(self, verbose: bool = False):
        self._reset()
        store_states = "store_states" in self.record_statistics
        if store_states:
            self.layer_states = {layer.name: [] for layer in self._network.layers()}
        else:
            self.layer_states = {}

        for x, y, idx in self._dataloader:
            self._network.set_input(x, reset=True)
            self._energy_minimizer.compute_equilibrium()

            if store_states:
                for i, layer in enumerate(self._network.layers()):
                    name = getattr(layer, "name", f"layer_{i}")
                    self.layer_states[name].append(layer.state.detach().cpu().clone())

            self._idx = idx
            self._cost_fn.set_target(y)
            self._do_measurements()

            if verbose:
                sys.stdout.write("\r")
                sys.stdout.write(str(self))
                sys.stdout.flush()

        if verbose:
            sys.stdout.write("\r")
            sys.stdout.write(str(self))
            sys.stdout.flush()


def _extract_batch_input(batch):
    """Return the input tensor from dataloader batches with flexible tuple shapes."""
    if isinstance(batch, (list, tuple)):
        if len(batch) >= 1:
            return batch[0]
        return None
    return batch


class LayerStateEvaluator(Evaluator):
    """Evaluator that records per-layer settled states for each batch."""

    def __init__(self, network, cost_fn, dataloader, energy_minimizer):
        super().__init__(network, cost_fn, dataloader, energy_minimizer)
        self.layer_states = {}

    def run(self, verbose: bool = False):
        self._reset()
        self.layer_states = {layer.name: [] for layer in self._network._function.layers()}

        for batch in self._dataloader:
            x = _extract_batch_input(batch)
            if x is None:
                continue

            self._network.set_input(x, reset=True)
            self._energy_minimizer.compute_equilibrium()

            for i, layer in enumerate(self._network._function.layers()):
                name = getattr(layer, "name", f"layer_{i}")
                self.layer_states[name].append(layer.state.detach().cpu().clone())

            if verbose:
                print(f"\r{self}", end="", flush=True)

        if verbose:
            print(f"\r{self}", end="", flush=True)


class ResidualCurrentEvaluator(Evaluator):
    """Evaluator that records residual currents (||dE/dz||_inf) per layer."""

    def __init__(self, network, cost_fn, dataloader, energy_minimizer):
        super().__init__(network, cost_fn, dataloader, energy_minimizer)
        self.res_currents = {}

    def run(self, verbose: bool = False):
        self._reset()
        self.res_currents = {layer.name: [] for layer in self._network._function.layers()}

        for batch in self._dataloader:
            x = _extract_batch_input(batch)
            if x is None:
                continue

            self._network.set_input(x, reset=True)
            self._energy_minimizer.compute_equilibrium()

            # Record per-layer residual current (infinity norm)
            for layer in self._network._function.layers():
                grad_fn = self._network._function.grad_layer_fn(layer)
                grad = grad_fn()
                res_current = grad.abs().max()
                self.res_currents[layer.name].append(float(res_current.item()))

            if verbose:
                print(f"\r{self}", end="", flush=True)

        if verbose:
            print(f"\r{self}", end="", flush=True)


def export_pt_to_npz(pt_path, npz_path: Optional[Path] = None, param_names=None):
    """Convert a saved model.pt (list of tensors) into a NumPy .npz bundle."""
    import numpy as np

    pt_path = Path(pt_path)
    tensors = torch.load(pt_path, map_location="cpu")
    if not isinstance(tensors, (list, tuple)):
        raise ValueError(f"Expected list/tuple of tensors in {pt_path}, got {type(tensors)}")

    arrays = {}
    for i, tensor in enumerate(tensors):
        name = param_names[i] if param_names and i < len(param_names) else f"param_{i}"
        arrays[name] = tensor.detach().cpu().numpy()

    target = pt_path.with_suffix(".npz") if npz_path is None else Path(npz_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez(target, **arrays)
    return target


def build_metadata_from_config(config_path, model_key):
    """Build a metadata dict from a config JSON in the same shape as small_network.py."""
    config_path = Path(config_path)
    config = json.loads(config_path.read_text())
    if "models" not in config or model_key not in config["models"]:
        raise KeyError(f"Model '{model_key}' not found in config: {config_path}")
    model_cfg = config["models"][model_key]

    return {
        "voltage_amp": model_cfg.get("voltage_amp"),
        "current_amp": model_cfg.get("current_amp"),
        "layer_shapes": model_cfg.get("layer_shapes"),
        "non_linearity": {
            "type": model_cfg.get("non_linearity"),
            "quadratic_params": dict(model_cfg.get("quadratic_diode_param", {})),
            "exponential_params": dict(model_cfg.get("exponential_diode_param", {})),
        },
    }


def flatten_weights_and_inputs(
    weights_npz_path,
    inputs_npz_path,
    output_dir,
    *,
    input_layer: str = "Layer_0",
    shapes_path: Optional[Path] = None,
):
    """Flatten model weights and the first input layer, saving new .npz files."""
    import numpy as np

    weights_npz_path = Path(weights_npz_path)
    inputs_npz_path = Path(inputs_npz_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shapes = {"weights": {}, "inputs": {}}

    with np.load(weights_npz_path) as weights_data:
        flat_weights = {}
        for name, array in weights_data.items():
            if array.ndim > 2:
                flat_array = array.reshape(-1, array.shape[-1])
            else:
                flat_array = array
            flat_weights[name] = flat_array
            shapes["weights"][name] = {
                "original": list(array.shape),
                "flat": list(flat_array.shape),
            }
    with np.load(inputs_npz_path) as inputs_data:
        flat_inputs = {}
        if input_layer not in inputs_data.files:
            input_layer = inputs_data.files[0] if inputs_data.files else input_layer
        for name, array in inputs_data.items():
            if name == input_layer and array.ndim >= 2:
                flat_array = array.reshape(array.shape[0], -1)
            else:
                flat_array = array
            flat_inputs[name] = flat_array
            shapes["inputs"][name] = {
                "original": list(array.shape),
                "flat": list(flat_array.shape),
            }

    weights_out = output_dir / f"{weights_npz_path.stem}_flat.npz"
    inputs_out = output_dir / f"{inputs_npz_path.stem}_flat.npz"
    np.savez(weights_out, **flat_weights)
    np.savez(inputs_out, **flat_inputs)
    if shapes_path is None:
        shapes_path = output_dir / "flattened_shapes.json"
    shapes_path.write_text(json.dumps(shapes, indent=2))
    return weights_out, inputs_out


# ---------- Plotting helpers ----------

def plot_layer_series(x_axis, layer_data, output_dir, prefix, ylabel):
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    for layer_name, tensor in layer_data.items():
        plt.figure(figsize=(8, 4))
        max_units = min(tensor.shape[1], 8)
        for idx in range(max_units):
            plt.plot(x_axis, tensor[:, idx], label=f"{layer_name}[{idx}]")
        plt.xlabel("Input value")
        plt.ylabel(ylabel)
        plt.title(f"{prefix.capitalize()} for {layer_name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(Path(output_dir) / f"{prefix}_{layer_name}.png")
        plt.close()


def plot_layer_extrema(x_axis, max_data, min_data, output_dir):
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    for layer_name in max_data:
        plt.figure(figsize=(8, 4))
        plt.plot(x_axis, max_data[layer_name], label="max")
        plt.plot(x_axis, min_data[layer_name], label="min")
        plt.xlabel("Input value")
        plt.ylabel("State extrema")
        plt.title(f"State extrema for {layer_name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(Path(output_dir) / f"extrema_{layer_name}.png")
        plt.close()


def save_beta_summary(beta_summary, run_dir: Path):
    import matplotlib.pyplot as plt

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "beta_summary.json").write_text(json.dumps(beta_summary, indent=2))
    layers = beta_summary.get("layers", [])
    means = beta_summary.get("per_layer_mean", [])
    medians = beta_summary.get("per_layer_median", [])
    if not layers or not means:
        return
    x = range(len(layers))
    plt.figure(figsize=(10, 5))
    plt.bar(x, means, width=0.4, label="mean", align="center")
    if medians:
        plt.bar([i + 0.4 for i in x], medians, width=0.4, label="median", align="center")
    plt.xticks([i + 0.2 for i in x], layers, rotation=45, ha="right")
    plt.ylabel("ratio (avg displacement / free magnitude)")
    plt.title("Beta-size per layer")
    plt.legend()
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(run_dir / "beta_summary.png")
    plt.close()


def plot_beta_displacements(beta_results, output_path: Path):
    import matplotlib.pyplot as plt

    if not beta_results:
        return
    layer_sets = [set(entry["beta_summary"].get("layers", [])) for entry in beta_results]
    all_layers = sorted(set().union(*layer_sets))
    if not all_layers:
        return
    x = list(range(len(all_layers)))

    plt.figure(figsize=(10, 5))
    for entry in beta_results:
        summary = entry["beta_summary"]
        layers = summary.get("layers", [])
        means = summary.get("per_layer_mean", [])
        if not layers or not means:
            continue
        mean_map = dict(zip(layers, means))
        aligned = [mean_map.get(name, float("nan")) for name in all_layers]
        label = f"cur={entry['current_amp']}, volt={entry['voltage_amp']}, iters={entry['num_iterations']}"
        plt.plot(x, aligned, marker="o", label=label)

    plt.xticks(x, all_layers, rotation=45, ha="right")
    plt.yscale("log")
    plt.ylabel("Normalized displacement between free and nudged phases")
    plt.xlabel("Layer")
    plt.title("Normalized displacement between free and nudged phases")
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_saturation(sat_results, output_path: Path):
    import matplotlib.pyplot as plt

    if not sat_results:
        return
    layer_sets = [set(entry["layers"]) for entry in sat_results]
    all_layers = sorted(set().union(*layer_sets))
    if not all_layers:
        return
    x = list(range(len(all_layers)))

    plt.figure(figsize=(10, 5))
    for entry in sat_results:
        mean_map = dict(zip(entry["layers"], entry["means"]))
        aligned = [mean_map.get(name, float("nan")) for name in all_layers]
        label = f"cur={entry['current_amp']}, volt={entry['voltage_amp']}, iters={entry['num_iterations']}"
        plt.plot(x, aligned, marker="o", label=label)

    plt.ylabel("saturation (mean)")
    plt.xlabel("Layer")
    plt.title("Saturation per layer")
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_saturation_from_sweep_results(
    sweep_results_path: Path,
    output_path: Path,
    *,
    normalized_layer_count: int = 4,
    title: str = "Saturation vs normalized layer index",
    voltage_alias: str = "v",
    current_alias: str = "c",
    iterations_alias: str = "iters",
):
    """
    Plot saturation curves from `sweep_results.json` while normalizing layer ids per run.

    Each run may expose saturation keys like Saturation_Layer_4..7, 8..11, etc.
    This function remaps each run's layer ids to a local 0..N-1 index before plotting.
    """
    import matplotlib.pyplot as plt
    import re

    sweep_results_path = Path(sweep_results_path)
    output_path = Path(output_path)
    results = json.loads(sweep_results_path.read_text())
    if not isinstance(results, list) or not results:
        return

    x = list(range(normalized_layer_count))
    plt.figure(figsize=(10, 5))

    for entry in results:
        saturation = entry.get("saturation") or {}
        if not saturation:
            continue

        parsed = []
        for key, value in saturation.items():
            match = re.search(r"(\d+)$", str(key))
            if not match:
                continue
            parsed.append((int(match.group(1)), float(value)))
        if not parsed:
            continue

        parsed.sort(key=lambda item: item[0])
        base = parsed[0][0]
        normalized_map = {}
        for raw_idx, sat_value in parsed:
            norm_idx = raw_idx - base
            if 0 <= norm_idx < normalized_layer_count:
                normalized_map[norm_idx] = sat_value

        y = [normalized_map.get(i, float("nan")) for i in x]
        label = (
            f"{voltage_alias}={entry.get('voltage_amp')}, "
            f"{current_alias}={entry.get('current_amp')}, "
            f"{iterations_alias}={entry.get('iterations')}"
        )
        plt.plot(x, y, marker="o", label=label)

    plt.xticks(x, [str(i) for i in x])
    plt.xlabel("Normalized layer index")
    plt.ylabel("Saturation (%)")
    plt.title(title)
    plt.ylim(0, 100)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_max_gradients(x_axis, grad_data, output_dir):
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    for layer_name, tensor in grad_data.items():
        plt.figure(figsize=(8, 4))
        plt.plot(x_axis, tensor, label="max |grad|")
        plt.xlabel("Input value")
        plt.ylabel("Max |gradient|")
        plt.title(f"Max gradient for {layer_name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(Path(output_dir) / f"max_grad_{layer_name}.png")
        plt.close()


def plot_pca_sweep_grid(
    npz_path,
    output_dir=None,
    *,
    layers=None,
    reduce: str = "mean",
    feature_idx: Optional[int] = None,
    grid_size: Optional[int] = None,
    cmap: str = "viridis",
):
    """
    Plot PCA sweep states (saved by pca-sweep) as 2D grids.

    Each layer's state is reduced to a scalar per sample, then reshaped into a square grid.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    npz_path = Path(npz_path)
    data = np.load(npz_path)
    if layers is None:
        layer_names = list(data.files)
    elif isinstance(layers, (list, tuple, set)):
        layer_names = list(layers)
    else:
        layer_names = [layers]

    if output_dir is None:
        output_dir = npz_path.with_suffix("")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for layer_name in layer_names:
        if layer_name not in data:
            raise KeyError(f"Layer '{layer_name}' not found in {npz_path}")
        arr = data[layer_name]
        if arr.ndim < 1:
            raise ValueError(f"Layer '{layer_name}' has invalid shape {arr.shape}")
        flat = arr.reshape(arr.shape[0], -1)

        if feature_idx is not None:
            if feature_idx < 0 or feature_idx >= flat.shape[1]:
                raise IndexError(
                    f"feature_idx={feature_idx} out of range for layer '{layer_name}' (size={flat.shape[1]})"
                )
            values = flat[:, feature_idx]
            label = f"idx={feature_idx}"
        else:
            if reduce == "mean":
                values = flat.mean(axis=1)
            elif reduce == "max":
                values = flat.max(axis=1)
            elif reduce == "min":
                values = flat.min(axis=1)
            elif reduce == "norm":
                values = np.linalg.norm(flat, axis=1)
            elif reduce == "argmax":
                values = flat.argmax(axis=1)
            else:
                raise ValueError(f"Unsupported reduce='{reduce}'")
            label = reduce

        if grid_size is None:
            side = int(np.sqrt(values.shape[0]))
            if side * side != values.shape[0]:
                raise ValueError(
                    f"Sample count {values.shape[0]} is not a square; pass grid_size explicitly."
                )
        else:
            side = grid_size
            if side * side != values.shape[0]:
                raise ValueError(
                    f"grid_size={side} does not match sample count {values.shape[0]}."
                )

        grid = values.reshape(side, side)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(grid, origin="lower", cmap=cmap)
        fig.colorbar(im, ax=ax)
        ax.set_title(f"{layer_name} ({label})")
        ax.set_xlabel("PCA dim 1")
        ax.set_ylabel("PCA dim 2")
        safe_name = layer_name.replace("/", "_")
        fig.tight_layout()
        fig.savefig(output_dir / f"pca_grid_{safe_name}_{label}.png", dpi=150)
        plt.close(fig)


def flatten_history(history_list):
    tensor = torch.cat(history_list, dim=0)
    tensor = tensor.reshape(tensor.shape[0], -1)
    return tensor.numpy()


def flatten_vector_history(history_list):
    tensor = torch.cat(history_list, dim=0)
    return tensor.numpy()
