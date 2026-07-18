#!/usr/bin/env python3
"""Find Conv MNIST input gains that produce target initialized saturation fractions."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
LABS_DIR = REPO_ROOT / "labs"
for path in (REPO_ROOT, LABS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from custom_classes import FlexibleDeepResistiveEnergy  # noqa: E402
from custom_minimizer import CustomQuadraticMinimizer, MinimizerSettings  # noqa: E402
from labs.datasets import (  # noqa: E402
    AFFINE_PRESETS,
    AffineMnistDataset,
    MnistDataset,
    affine_config_from_preset,
)
from model.function.network import Network  # noqa: E402
from model.variable.layer import Layer  # noqa: E402
from model.variable.parameter import (  # noqa: E402
    Bias,
    ConvWeight,
    DenseWeight,
    HardSigmoidVOff,
    PoolWeight,
)


MINIMIZER_SETTINGS_FIELDS = (
    "rel_tol",
    "vn_tol",
    "use_polish",
    "max_newton_iters",
    "z_thresh",
    "exp_clip",
    "dynamic_polish",
    "overrelaxation_reject_steps",
    "overrelaxation_reject_max_tries",
    "overrelaxation_reject_shrink",
    "overrelaxation_reject_eps",
    "experimental_exponential_newton_tol_progressive",
    "experimental_exponential_newton_tol_start",
    "experimental_exponential_newton_tol_end",
    "experimental_exponential_newton_tol_switch_hi",
    "experimental_exponential_newton_tol_switch_lo",
)

AMPLIFICATION_GRID = [
    ("mnist_bp_amp_v1_c1", 1.0, 1.0),
    ("mnist_bp_amp_v2_c1", 2.0, 1.0),
    ("mnist_bp_amp_v4_c1", 4.0, 1.0),
    ("mnist_bp_amp_v1_c2", 1.0, 2.0),
    ("mnist_bp_amp_v1_c4", 1.0, 4.0),
    ("mnist_bp_amp_v4_c0p25", 4.0, 0.25),
    ("mnist_bp_amp_v2_c2", 2.0, 2.0),
]
AMPLIFICATION_BY_NAME = {name: (v_amp, c_amp) for name, v_amp, c_amp in AMPLIFICATION_GRID}


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _reset_name_counters() -> None:
    """Make every independently calibrated model use the same layer indices."""
    Layer._counter = 0
    Bias._counter = 0
    DenseWeight._counter = 0
    ConvWeight._counter = 0
    PoolWeight._counter = 0
    HardSigmoidVOff._counter = 0


def _conv_spatial(size: int, kernel: int, stride: int, padding: int) -> int:
    return (size + 2 * padding - (kernel - 1) - 1) // stride + 1


def _expanded_layer_ints(
    values: list[int] | None,
    *,
    fallback: int,
    depth: int,
    name: str,
) -> list[int]:
    raw = [int(fallback)] if values is None else [int(value) for value in values]
    if len(raw) == 1:
        return raw * depth
    if len(raw) != depth:
        raise ValueError(
            f"Expected --{name} to contain 1 value or {depth} values "
            f"for this conv depth, got {len(raw)}."
        )
    return raw


def _conv_geometry(args: argparse.Namespace) -> tuple[list[int], list[int]]:
    return (
        _expanded_layer_ints(
            args.strides, fallback=args.stride, depth=args.conv_depth, name="strides"
        ),
        _expanded_layer_ints(
            args.paddings, fallback=args.padding, depth=args.conv_depth, name="paddings"
        ),
    )


def _inference_iterations(args: argparse.Namespace) -> int:
    if args.num_iterations_inference is not None:
        return int(args.num_iterations_inference)
    return int(args.num_iterations)


def _architecture(args: argparse.Namespace) -> tuple[list[tuple[int, ...]], list[dict]]:
    channels = [int(value) for value in args.conv_channels]
    if args.conv_depth < 1:
        raise ValueError(f"Expected --conv-depth >= 1, got {args.conv_depth}.")
    if len(channels) < args.conv_depth:
        raise ValueError(
            f"Expected at least {args.conv_depth} --conv-channels values, got {channels}."
        )

    layer_shapes: list[tuple[int, ...]] = [(2, 28, 28)]
    conv_pipeline: list[dict] = []
    height = 28
    width = 28
    strides, paddings = _conv_geometry(args)
    for index in range(args.conv_depth):
        stride = strides[index]
        padding = paddings[index]
        height = _conv_spatial(height, args.kernel_size, stride, padding)
        width = _conv_spatial(width, args.kernel_size, stride, padding)
        if height <= 0 or width <= 0:
            raise ValueError(
                "Convolution geometry produced non-positive spatial dimensions: "
                f"depth={index + 1}, height={height}, width={width}."
            )
        layer_shapes.append((channels[index], height, width))
        conv_pipeline.append(
            {
                "kernel": [args.kernel_size, args.kernel_size],
                "stride": stride,
                "padding": padding,
                "mode": "convolution",
            }
        )
    layer_shapes.append((args.output_dim,))
    return layer_shapes, conv_pipeline


def _load_minimizer_config(path_value: str | None) -> dict:
    if path_value is None:
        raise SystemExit(
            "Expected --minimizer-config to point to an explicit simulator/minimizer JSON object. "
            "Provided value: None."
        )
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    if not path.exists():
        raise SystemExit(
            "Expected --minimizer-config to point to an existing JSON file. "
            f"Provided value: {path}."
        )
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise SystemExit(
            "Expected --minimizer-config JSON to contain an object. "
            f"Provided value: {data!r}."
        )
    if data.get("adaptive_equilibrium") is not False:
        raise SystemExit(
            "Expected --minimizer-config adaptive_equilibrium to be false for fixed-step calibration. "
            f"Provided value: {data.get('adaptive_equilibrium')!r}."
        )
    settings = data.get("settings")
    if not isinstance(settings, dict):
        raise SystemExit(
            "Expected --minimizer-config settings to contain an object. "
            f"Provided value: {settings!r}."
        )
    missing = [field for field in MINIMIZER_SETTINGS_FIELDS if field not in settings]
    if missing:
        raise SystemExit(
            "Expected --minimizer-config settings to define all solver fields. "
            f"Missing fields: {missing}. Provided value: {settings!r}."
        )
    return data


def _settings_from_config(minimizer_cfg: dict) -> MinimizerSettings:
    settings = minimizer_cfg["settings"]
    return MinimizerSettings(**{field: settings[field] for field in MINIMIZER_SETTINGS_FIELDS})


def _build_minimizer(energy_fn, free_layers, args):
    minimizer_cfg = args.minimizer_config_data
    return CustomQuadraticMinimizer(
        fn=energy_fn,
        free_layers=free_layers,
        num_iterations=_inference_iterations(args),
        mode="asynchronous",
        non_linearity="hard_sigmoid",
        quadratic_diode_param={"diode_conductance": 1.0, "v_min": -1.0e6, "v_max": 1.0e6},
        exponential_diode_param={"I_s": 1.0e-6, "V_t": 0.025, "V_off": 0.0},
        hard_sigmoid_param={
            "g_on": args.g_on,
            "g_off": args.g_off,
            "v_min": -args.v_off,
            "v_max": args.v_off,
        },
        voltage_amp=args.voltage_amp,
        current_amp=args.current_amp,
        iv_data=minimizer_cfg.get("iv_data"),
        iv_data_path=minimizer_cfg.get("iv_data_path"),
        double_diode_updater=minimizer_cfg["double_diode_updater"],
        adaptive_equilibrium=minimizer_cfg["adaptive_equilibrium"],
        overrelaxation_factor=minimizer_cfg["overrelaxation_factor"],
        single_diode_updater=minimizer_cfg["single_diode_updater"],
        minimizer_settings=_settings_from_config(minimizer_cfg),
        damping=minimizer_cfg["experimental_damping"],
        experimental_newton_max_steps=minimizer_cfg["experimental_newton_max_steps"],
    )


def _build_energy(args):
    layer_shapes, conv_pipeline = _architecture(args)
    energy_fn = FlexibleDeepResistiveEnergy(
        layer_shapes=layer_shapes,
        conv_pipeline=conv_pipeline,
        pooling_mode=None,
        weight_gains=[1.0] * (args.conv_depth + 1),
        input_gain=1.0,
        non_linearity="hard_sigmoid",
        exponential_diode_param={"I_s": 1.0e-6, "V_t": 0.025, "V_off": 0.0},
        quadratic_diode_param={"diode_conductance": 1.0, "v_min": -1.0e6, "v_max": 1.0e6},
        hard_sigmoid_param={
            "g_on": args.g_on,
            "g_off": args.g_off,
            "v_min": -args.v_off,
            "v_max": args.v_off,
        },
        voltage_amp=args.voltage_amp,
        current_amp=args.current_amp,
        weight_min=0.0,
        weight_max=100.0,
        weight_init_mode="kaiming_uniform",
        input_mode="train",
    )
    energy_fn.set_device(args.device)
    return energy_fn


def _load_samples(args):
    affine_config = _affine_config_from_args(args)
    dataset_cls = AffineMnistDataset if affine_config.get("enabled") else MnistDataset
    dataset_kwargs = {}
    if affine_config.get("enabled"):
        dataset_kwargs["affine_config"] = affine_config
    dataset = dataset_cls(
        name="mnist",
        batch_size=args.batch_size,
        device=args.device,
        root=args.dataset_root,
        train=True,
        download=False,
        normalize=True,
        normalize_mean=args.normalize_mean,
        normalize_std=args.normalize_std,
        normalize_scale=args.normalize_scale,
        shuffle_seed=args.seed,
        **dataset_kwargs,
    )
    train_loader, _ = dataset.build()
    images = []
    seen = 0
    for batch, _labels in train_loader:
        take = min(args.num_samples - seen, batch.shape[0])
        images.append(batch[:take].to(args.device))
        seen += take
        if seen >= args.num_samples:
            break
    if seen != args.num_samples:
        raise RuntimeError(f"Expected {args.num_samples} samples, loaded {seen}.")
    return images


def _affine_config_from_args(args: argparse.Namespace) -> dict:
    return affine_config_from_preset(
        args.affine_preset,
        degrees=args.affine_degrees,
        translate=args.affine_translate,
        scale=args.affine_scale,
        shear=args.affine_shear,
        seed=args.affine_seed,
    )


@torch.no_grad()
def saturation_fraction(
    gain: float,
    energy_fn,
    network,
    minimizer,
    batches,
    v_off: float,
    saturation_scope: str,
    record_all_layers: bool = False,
) -> tuple[float, list[float]]:
    input_layer = energy_fn.layers()[0]
    old_gain = input_layer._gain
    input_layer._gain = float(gain)
    layer_totals: list[int] = []
    layer_saturated: list[int] = []
    try:
        for batch in batches:
            network.set_input(batch, reset=True)
            minimizer.compute_equilibrium()
            if saturation_scope not in {"first_hidden", "all_hidden"}:
                raise ValueError(
                    "Expected saturation_scope to be 'first_hidden' or 'all_hidden', "
                    f"got {saturation_scope!r}."
                )
            if saturation_scope == "first_hidden" and not record_all_layers:
                hidden_layers = energy_fn.layers()[1:2]
            else:
                hidden_layers = energy_fn.layers()[1:-1]
            if not layer_totals:
                layer_totals = [0 for _ in hidden_layers]
                layer_saturated = [0 for _ in hidden_layers]
            for index, layer in enumerate(hidden_layers):
                state = layer.state.detach()
                layer_saturated[index] += int((state.abs() > v_off).sum().item())
                layer_totals[index] += state.numel()
    finally:
        input_layer._gain = old_gain

    layer_fracs = [
        sat / total_for_layer if total_for_layer else 0.0
        for sat, total_for_layer in zip(layer_saturated, layer_totals)
    ]
    if saturation_scope == "first_hidden":
        total = layer_totals[0]
        saturated = layer_saturated[0]
    else:
        total = sum(layer_totals)
        saturated = sum(layer_saturated)
    return saturated / total, layer_fracs


def find_gain(target: float, energy_fn, network, minimizer, batches, args) -> tuple[float, float, list[float]]:
    lo = 0.0
    hi = args.initial_hi
    sat_hi, layer_sat_hi = saturation_fraction(
        hi, energy_fn, network, minimizer, batches, args.v_off, args.saturation_scope
    )
    while sat_hi < target:
        hi *= 2.0
        if hi > args.max_gain:
            raise RuntimeError(
                f"Could not bracket target saturation {target}; "
                f"gain={hi} gives saturation={sat_hi}."
            )
        sat_hi, layer_sat_hi = saturation_fraction(
            hi, energy_fn, network, minimizer, batches, args.v_off, args.saturation_scope
        )

    best_gain = hi
    best_sat = sat_hi
    best_layer_sat = layer_sat_hi
    for _ in range(args.search_steps):
        mid = 0.5 * (lo + hi)
        sat_mid, layer_sat_mid = saturation_fraction(
            mid, energy_fn, network, minimizer, batches, args.v_off, args.saturation_scope
        )
        best_gain = mid
        best_sat = sat_mid
        best_layer_sat = layer_sat_mid
        if sat_mid < target:
            lo = mid
        else:
            hi = mid
    return best_gain, best_sat, best_layer_sat


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default=str(REPO_ROOT / "data"))
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-iterations", type=int, default=4)
    parser.add_argument("--num-iterations-inference", type=int, default=None)
    parser.add_argument(
        "--minimizer-config",
        help="Path to a JSON object for model_base.minimizer.",
    )
    parser.add_argument(
        "--run-name",
        nargs="+",
        default=["mnist_bp_amp_v1_c1"],
        choices=sorted(AMPLIFICATION_BY_NAME),
        help="Amplification run names to calibrate. Defaults to no amplification.",
    )
    parser.add_argument(
        "--all-amplifications",
        action="store_true",
        help="Calibrate all supported amplification settings, including legacy A=4,B=0.25.",
    )
    parser.add_argument("--conv-depth", type=int, default=1)
    parser.add_argument("--conv-channels", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--padding", type=int, default=0)
    parser.add_argument(
        "--strides",
        type=int,
        nargs="+",
        default=None,
        help="Per-convolution strides. One value broadcasts to all conv layers.",
    )
    parser.add_argument(
        "--paddings",
        type=int,
        nargs="+",
        default=None,
        help="Per-convolution paddings. One value broadcasts to all conv layers.",
    )
    parser.add_argument("--output-dim", type=int, default=20)
    parser.add_argument("--v-off", type=float, required=True)
    parser.add_argument(
        "--saturation-scope",
        choices=["first_hidden", "all_hidden"],
        default="first_hidden",
        help="Which hidden states define the target saturation fraction.",
    )
    parser.add_argument("--g-on", type=float, default=100.0)
    parser.add_argument("--g-off", type=float, default=0.0)
    parser.add_argument("--normalize-mean", type=float, default=0.1307)
    parser.add_argument("--normalize-std", type=float, default=0.3081)
    parser.add_argument("--normalize-scale", type=float, default=0.3)
    parser.add_argument(
        "--affine-preset",
        choices=sorted(AFFINE_PRESETS),
        default="none",
        help="Deterministic per-sample affine MNIST preset.",
    )
    parser.add_argument("--affine-degrees", type=float, default=None)
    parser.add_argument("--affine-translate", type=float, nargs="+", default=None)
    parser.add_argument("--affine-scale", type=float, nargs=2, default=None)
    parser.add_argument("--affine-shear", type=float, default=None)
    parser.add_argument("--affine-seed", type=int, default=1729)
    parser.add_argument("--targets", type=float, nargs="+", default=[0.10, 0.30, 0.50, 0.70, 0.90])
    parser.add_argument("--initial-hi", type=float, default=64.0)
    parser.add_argument("--max-gain", type=float, default=8192.0)
    parser.add_argument("--search-steps", type=int, default=24)
    parser.add_argument(
        "--lr-numerator",
        type=float,
        default=1.44,
        help="Learning-rate numerator for lr = numerator / input_gain.",
    )
    args = parser.parse_args()

    _architecture(args)
    affine_config = _affine_config_from_args(args)
    args.minimizer_config_data = _load_minimizer_config(args.minimizer_config)
    _set_seed(args.seed)
    batches = _load_samples(args)
    run_names = [name for name, _, _ in AMPLIFICATION_GRID] if args.all_amplifications else args.run_name

    rows = []
    for run_name in run_names:
        voltage_amp, current_amp = AMPLIFICATION_BY_NAME[run_name]
        args.voltage_amp = voltage_amp
        args.current_amp = current_amp
        _reset_name_counters()
        _set_seed(args.seed)
        energy_fn = _build_energy(args)
        network = Network(energy_fn)
        minimizer = _build_minimizer(energy_fn, network.free_layers(), args)
        for target in args.targets:
            gain, _measured, _layer_measured = find_gain(
                target, energy_fn, network, minimizer, batches, args
            )
            measured, layer_measured = saturation_fraction(
                gain,
                energy_fn,
                network,
                minimizer,
                batches,
                args.v_off,
                args.saturation_scope,
                record_all_layers=True,
            )
            lr = args.lr_numerator / gain
            row = {
                "dataset_name": (
                    "deterministic_medium_affine_mnist"
                    if affine_config.get("enabled")
                    else "ordinary_mnist"
                ),
                "non_linearity": "hard_sigmoid",
                "run_name": run_name,
                "voltage_amp": voltage_amp,
                "current_amp": current_amp,
                "target_saturation": target,
                "input_gain": gain,
                "measured_saturation": measured,
                "layer_saturations": json.dumps(layer_measured),
                "learning_rate": lr,
                "v_off": args.v_off,
                "lr_numerator": args.lr_numerator,
                "seed": args.seed,
                "shuffle_seed": args.seed,
                "num_samples": args.num_samples,
                "batch_size": args.batch_size,
                "num_iterations": _inference_iterations(args),
                "num_iterations_inference": _inference_iterations(args),
                "conv_depth": args.conv_depth,
                "strides": json.dumps(_conv_geometry(args)[0]),
                "paddings": json.dumps(_conv_geometry(args)[1]),
                "padding": ",".join(str(value) for value in _conv_geometry(args)[1]),
                "conv_pipeline": json.dumps(_architecture(args)[1]),
                "saturation_scope": args.saturation_scope,
                "affine_config": json.dumps(affine_config),
                "normalize_mean": args.normalize_mean,
                "normalize_std": args.normalize_std,
                "normalize_scale": args.normalize_scale,
                "adaptive_equilibrium": bool(
                    args.minimizer_config_data.get("adaptive_equilibrium", False)
                ),
                "search_initial_hi": args.initial_hi,
                "search_max_gain": args.max_gain,
                "search_steps": args.search_steps,
                "model_counter_reset_before_row": True,
                "calibration_status": "selected",
            }
            rows.append(row)
            print(
                f"run_name={run_name} A={voltage_amp:g} B={current_amp:g} "
                f"target={target:.2f} input_gain={gain:.6g} "
                f"measured={measured:.6f} layers={layer_measured} lr={lr:.8g}",
                flush=True,
            )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
