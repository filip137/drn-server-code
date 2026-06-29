#!/usr/bin/env python3
"""Evaluate a saved analog MNIST MLP under repeated drift realizations."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import statistics
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ANALOG_SCRIPT = Path("/home/filip/analog-foundation-models/mnist_analog_drift.py")
DEFAULT_MODEL = (
    REPO_ROOT
    / "results"
    / "analog_drn_neck_to_neck"
    / "analog_784_50_10_train_eval"
    / "model.pt"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "results"
    / "analog_drn_neck_to_neck"
    / "analog_784_50_10_512_dense_times"
)
DEFAULT_TIMES = [
    1_000.0,
    3_000.0,
    10_000.0,
    30_000.0,
    86_400.0,
    259_200.0,
    604_800.0,
    1_000_000.0,
    2_592_000.0,
    7_776_000.0,
    15_552_000.0,
    31_536_000.0,
]

RAW_COLUMNS = [
    "model_family",
    "architecture",
    "num_examples",
    "drift_seed",
    "time_s",
    "compensation",
    "loss",
    "accuracy",
    "kl_initial_to_current_nats",
    "drift_factor_mean",
    "drift_factor_std",
    "mean_relative_conductance_after",
]

SUMMARY_COLUMNS = [
    "model_family",
    "architecture",
    "num_examples",
    "time_s",
    "compensation",
    "num_drift_seeds",
    "accuracy_mean",
    "accuracy_std",
    "loss_mean",
    "kl_mean",
    "kl_std",
    "drift_factor_mean",
    "mean_relative_conductance_after",
]


def _load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("mnist_analog_drift", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _parse_csv_floats(value: str) -> list[float]:
    out = [float(item) for item in value.split(",") if item.strip()]
    if not out:
        raise argparse.ArgumentTypeError("expected at least one comma-separated float")
    return out


def _parse_csv_ints(value: str) -> list[int]:
    out = [int(item) for item in value.split(",") if item.strip()]
    if not out:
        raise argparse.ArgumentTypeError("expected at least one comma-separated integer")
    return out


def _parse_csv_strings(value: str) -> list[str]:
    out = [item.strip() for item in value.split(",") if item.strip()]
    if not out:
        raise argparse.ArgumentTypeError("expected at least one comma-separated string")
    return out


def _default_data_dir() -> str:
    data_dir = REPO_ROOT / "data"
    return str(data_dir) if data_dir.exists() else "data/torchvision"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analog-script", type=Path, default=DEFAULT_ANALOG_SCRIPT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--hidden-sizes", default="50")
    parser.add_argument("--times", type=_parse_csv_floats, default=DEFAULT_TIMES)
    parser.add_argument("--drift-seeds", type=_parse_csv_ints, default=[0, 1, 2, 3, 4])
    parser.add_argument("--compensations", type=_parse_csv_strings, default=["none", "per_layer"])
    parser.add_argument("--data-dir", default=_default_data_dir())
    parser.add_argument("--test-limit", type=int, default=512)
    parser.add_argument("--subset-mode", choices=("first", "random"), default="first")
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--g-min-us", type=float, default=0.0)
    parser.add_argument("--g-max-us", type=float, default=180.0)
    parser.add_argument("--t0-s", type=float, default=1000.0)
    parser.add_argument(
        "--nu-mode",
        choices=("constant", "aihwkit"),
        default="constant",
        help="Drift exponent model. 'constant' uses nu_mean +/- nu_std; 'aihwkit' uses IBM AIHWKIT-style conductance-dependent nu.",
    )
    parser.add_argument("--nu-mean", type=float, default=0.06)
    parser.add_argument("--nu-std", type=float, default=0.02)
    parser.add_argument("--write-rel-std", type=float, default=0.0)
    parser.add_argument("--write-abs-std-us", type=float, default=0.0)
    return parser.parse_args()


def _build_test_loader(analog: ModuleType, args: argparse.Namespace):
    transform = analog.transforms.Compose(
        [
            analog.transforms.ToTensor(),
            analog.transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    test_set = analog.datasets.MNIST(
        root=str(args.data_dir),
        train=False,
        download=False,
        transform=transform,
    )
    if args.test_limit > 0 and args.test_limit < len(test_set):
        if args.subset_mode == "first":
            indices = list(range(args.test_limit))
        else:
            generator = torch.Generator().manual_seed(int(args.seed) + 1)
            indices = torch.randperm(len(test_set), generator=generator)[: args.test_limit].tolist()
        test_set = analog.Subset(test_set, indices)
    return analog.DataLoader(
        test_set,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=str(args.device).startswith("cuda"),
    )


def _sample_aihwkit_nu(
    conductance: torch.Tensor,
    *,
    g_max_s: float,
    generator: torch.Generator,
) -> torch.Tensor:
    g_relative = torch.clamp(torch.abs(conductance / float(g_max_s)), min=1e-7)
    mu_nu = torch.clamp(-0.0155 * torch.log(g_relative) + 0.0244, min=0.049, max=0.100)
    sigma_nu = torch.clamp(-0.0125 * torch.log(g_relative) - 0.0059, min=0.008, max=0.045)
    xi = torch.randn(
        conductance.shape,
        generator=generator,
        device=conductance.device,
        dtype=conductance.dtype,
    )
    return torch.clamp(torch.abs(mu_nu + sigma_nu * xi), min=0.0)


def _override_aihwkit_nu(model, drift, seed: int) -> None:
    first_layer = next(iter(model.layers))
    generator = torch.Generator(device=first_layer.g_pos_programmed.device)
    generator.manual_seed(int(seed))
    for layer in model.layers:
        layer.nu_pos = _sample_aihwkit_nu(
            layer.g_pos_programmed,
            g_max_s=drift.g_max_s,
            generator=generator,
        )
        layer.nu_neg = _sample_aihwkit_nu(
            layer.g_neg_programmed,
            g_max_s=drift.g_max_s,
            generator=generator,
        )


def _drift_factor_stats(model, drift, time_s: float) -> dict[str, float]:
    ratio = max(float(time_s), float(drift.t0_s)) / float(drift.t0_s)
    factors = []
    before_sum = 0.0
    after_sum = 0.0
    for layer in model.layers:
        factor_pos = ratio ** (-layer.nu_pos.detach().cpu())
        factor_neg = ratio ** (-layer.nu_neg.detach().cpu())
        factors.append(factor_pos.reshape(-1))
        factors.append(factor_neg.reshape(-1))
        g_pos = layer.g_pos_programmed.detach().cpu()
        g_neg = layer.g_neg_programmed.detach().cpu()
        before_sum += float((g_pos.abs() + g_neg.abs()).sum().item())
        after_sum += float(((g_pos * factor_pos).abs() + (g_neg * factor_neg).abs()).sum().item())
    all_factors = torch.cat(factors).to(dtype=torch.float64)
    return {
        "drift_factor_mean": float(all_factors.mean().item()),
        "drift_factor_std": float(all_factors.std(unbiased=all_factors.numel() > 1).item()),
        "mean_relative_conductance_after": after_sum / before_sum if before_sum > 0.0 else float("nan"),
    }


def _write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _mean(values: list[float]) -> float:
    return statistics.fmean(values)


def _stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _summarize(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[float, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((float(row["time_s"]), str(row["compensation"])), []).append(row)

    summary_rows = []
    for (time_s, compensation), group in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1])):
        summary_rows.append(
            {
                "model_family": group[0]["model_family"],
                "architecture": group[0]["architecture"],
                "num_examples": group[0]["num_examples"],
                "time_s": time_s,
                "compensation": compensation,
                "num_drift_seeds": len({row["drift_seed"] for row in group}),
                "accuracy_mean": _mean([float(row["accuracy"]) for row in group]),
                "accuracy_std": _stdev([float(row["accuracy"]) for row in group]),
                "loss_mean": _mean([float(row["loss"]) for row in group]),
                "kl_mean": _mean([float(row["kl_initial_to_current_nats"]) for row in group]),
                "kl_std": _stdev([float(row["kl_initial_to_current_nats"]) for row in group]),
                "drift_factor_mean": _mean([float(row["drift_factor_mean"]) for row in group]),
                "mean_relative_conductance_after": _mean(
                    [float(row["mean_relative_conductance_after"]) for row in group]
                ),
            }
        )
    return summary_rows


def main() -> None:
    args = parse_args()
    analog = _load_module(args.analog_script)
    device = torch.device(args.device)
    hidden_sizes = [int(value) for value in args.hidden_sizes.split(",") if value.strip()]
    architecture = "784-" + "-".join(str(size) for size in hidden_sizes) + "-10"

    for mode in args.compensations:
        if mode not in {"none", "per_layer", "per_row"}:
            raise ValueError(f"Unknown compensation mode: {mode}")

    drift = analog.DriftConfig(
        g_min_s=args.g_min_us * 1e-6,
        g_max_s=args.g_max_us * 1e-6,
        t0_s=args.t0_s,
        nu_mean=args.nu_mean,
        nu_std=args.nu_std,
        write_rel_std=args.write_rel_std,
        write_abs_std_s=args.write_abs_std_us * 1e-6,
    )

    test_loader = _build_test_loader(analog, args)
    model = analog.AnalogMLP(hidden_sizes).to(device)
    state = torch.load(args.model, map_location=device)
    model.load_state_dict(state)
    model.eval()

    raw_rows: list[dict] = []
    for drift_seed in args.drift_seeds:
        analog.set_seed(int(drift_seed))
        model.program_crossbars(drift)
        if args.nu_mode == "aihwkit":
            _override_aihwkit_nu(model, drift, int(drift_seed))
        for time_s in args.times:
            stats = _drift_factor_stats(model, drift, float(time_s))
            for mode in args.compensations:
                loss, acc, kl = analog.evaluate_against_initial(
                    model,
                    test_loader,
                    device,
                    t_s=float(time_s),
                    compensation=mode,
                    drift=drift,
                )
                row = {
                    "model_family": "analog_crossbar_relu_mlp",
                    "architecture": architecture,
                    "num_examples": args.test_limit,
                    "drift_seed": int(drift_seed),
                    "time_s": float(time_s),
                    "compensation": mode,
                    "loss": float(loss),
                    "accuracy": float(acc),
                    "kl_initial_to_current_nats": float(kl),
                    **stats,
                }
                raw_rows.append(row)
                print(
                    f"[analog] seed={drift_seed} t={float(time_s):.6g}s mode={mode} "
                    f"acc={100.0 * acc:.2f}% kl={kl:.6g} "
                    f"factor_mean={stats['drift_factor_mean']:.4g}"
                )

    summary_rows = _summarize(raw_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "raw_results.csv", RAW_COLUMNS, raw_rows)
    _write_csv(args.output_dir / "summary.csv", SUMMARY_COLUMNS, summary_rows)
    metadata = {
        "source_script": str(args.analog_script),
        "source_model": str(args.model),
        "hidden_sizes": hidden_sizes,
        "architecture": architecture,
        "times_s": [float(value) for value in args.times],
        "drift_seeds": [int(value) for value in args.drift_seeds],
        "compensations": args.compensations,
        "num_examples": args.test_limit,
        "subset_mode": args.subset_mode,
        "drift": {
            "g_min_s": drift.g_min_s,
            "g_max_s": drift.g_max_s,
            "t0_s": drift.t0_s,
            "nu_mode": args.nu_mode,
            "nu_mean": drift.nu_mean,
            "nu_std": drift.nu_std,
            "write_rel_std": drift.write_rel_std,
            "write_abs_std_s": drift.write_abs_std_s,
        },
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"[done] wrote {args.output_dir}")


if __name__ == "__main__":
    main()
