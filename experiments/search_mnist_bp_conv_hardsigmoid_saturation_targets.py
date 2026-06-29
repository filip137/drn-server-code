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
from labs.datasets import MnistDataset  # noqa: E402
from model.function.network import Network  # noqa: E402


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


def _conv_spatial(size: int, kernel: int, stride: int, padding: int) -> int:
    return (size + 2 * padding - (kernel - 1) - 1) // stride + 1


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
    for index in range(args.conv_depth):
        height = _conv_spatial(height, args.kernel_size, args.stride, args.padding)
        width = _conv_spatial(width, args.kernel_size, args.stride, args.padding)
        if height <= 0 or width <= 0:
            raise ValueError(
                "Convolution geometry produced non-positive spatial dimensions: "
                f"depth={index + 1}, height={height}, width={width}."
            )
        layer_shapes.append((channels[index], height, width))
        conv_pipeline.append(
            {
                "kernel": [args.kernel_size, args.kernel_size],
                "stride": args.stride,
                "padding": args.padding,
                "mode": "convolution",
            }
        )
    layer_shapes.append((args.output_dim,))
    return layer_shapes, conv_pipeline


def _build_minimizer(energy_fn, free_layers, args):
    return CustomQuadraticMinimizer(
        fn=energy_fn,
        free_layers=free_layers,
        num_iterations=args.num_iterations,
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
        iv_data=None,
        iv_data_path=None,
        double_diode_updater="CustomExponentialDoubleDiodeUpdater",
        adaptive_equilibrium=True,
        overrelaxation_factor=1.1,
        single_diode_updater="custom",
        minimizer_settings=MinimizerSettings(
            rel_tol=1.0e-5,
            vn_tol=1.0e-6,
            use_polish=True,
            max_newton_iters=32,
            z_thresh=1.0e10,
            exp_clip=80.0,
            dynamic_polish=True,
            overrelaxation_reject_steps=False,
            overrelaxation_reject_max_tries=3,
            overrelaxation_reject_shrink=0.5,
            overrelaxation_reject_eps=0.0,
        ),
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
    dataset = MnistDataset(
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


@torch.no_grad()
def saturation_fraction(
    gain: float,
    energy_fn,
    network,
    minimizer,
    batches,
    v_off: float,
    saturation_scope: str,
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
            if saturation_scope == "first_hidden":
                hidden_layers = energy_fn.layers()[1:2]
            elif saturation_scope == "all_hidden":
                hidden_layers = energy_fn.layers()[1:-1]
            else:
                raise ValueError(
                    "Expected saturation_scope to be 'first_hidden' or 'all_hidden', "
                    f"got {saturation_scope!r}."
                )
            if not layer_totals:
                layer_totals = [0 for _ in hidden_layers]
                layer_saturated = [0 for _ in hidden_layers]
            for index, layer in enumerate(hidden_layers):
                state = layer.state.detach()
                layer_saturated[index] += int((state.abs() > v_off).sum().item())
                layer_totals[index] += state.numel()
    finally:
        input_layer._gain = old_gain

    total = sum(layer_totals)
    saturated = sum(layer_saturated)
    layer_fracs = [
        sat / total_for_layer if total_for_layer else 0.0
        for sat, total_for_layer in zip(layer_saturated, layer_totals)
    ]
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
    _set_seed(args.seed)
    batches = _load_samples(args)
    run_names = [name for name, _, _ in AMPLIFICATION_GRID] if args.all_amplifications else args.run_name

    rows = []
    for run_name in run_names:
        voltage_amp, current_amp = AMPLIFICATION_BY_NAME[run_name]
        args.voltage_amp = voltage_amp
        args.current_amp = current_amp
        _set_seed(args.seed)
        energy_fn = _build_energy(args)
        network = Network(energy_fn)
        minimizer = _build_minimizer(energy_fn, network.free_layers(), args)
        for target in args.targets:
            gain, measured, layer_measured = find_gain(target, energy_fn, network, minimizer, batches, args)
            lr = args.lr_numerator / gain
            row = {
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
                "num_samples": args.num_samples,
                "num_iterations": args.num_iterations,
                "conv_depth": args.conv_depth,
                "padding": args.padding,
                "saturation_scope": args.saturation_scope,
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
