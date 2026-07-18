#!/usr/bin/env python3
"""Compare initial and trained hidden saturation for selected Conv MNIST runs."""

from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
LABS_DIR = REPO_ROOT / "labs"
for path in (REPO_ROOT, LABS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import model  # noqa: F401,E402 - anchor repo-local package before labs imports.
from custom_classes import FlexibleDeepResistiveEnergy  # noqa: E402
from labs.mnist_train import (  # noqa: E402
    _build_tracking_minimizer,
    _reset_name_counters,
    _resolve_callable,
    _resolve_dataset_config,
    _set_seed,
    load_config,
)
from model.function.cost import SquaredError, SquaredErrorPairedOutputs  # noqa: E402
from model.function.network import Network  # noqa: E402


OUTPUT_COLUMNS = [
    "conv_depth",
    "non_linearity",
    "run_name",
    "voltage_amp",
    "current_amp",
    "input_gain",
    "v_off",
    "lr",
    "epochs",
    "seed",
    "run_dir",
    "best_test_accuracy",
    "final_test_accuracy",
    "best_epoch",
    "target_label",
    "target_saturation",
    "split",
    "num_samples",
    "init_first_hidden_saturation",
    "init_all_hidden_saturation",
    "init_layer_saturations",
    "final_first_hidden_saturation",
    "final_all_hidden_saturation",
    "final_layer_saturations",
    "delta_first_hidden_saturation",
    "delta_all_hidden_saturation",
]

HISTORICAL_MINIMIZER_DEFAULTS = {
    "double_diode_updater": "CustomExponentialDoubleDiodeUpdater",
    "adaptive_equilibrium": True,
    "overrelaxation_factor": 1.1,
    "single_diode_updater": "custom",
    "iv_data_path": None,
    "experimental_damping": 0.5,
    "experimental_newton_max_steps": 100,
    "settings": {
        "rel_tol": 1e-5,
        "vn_tol": 1e-6,
        "use_polish": True,
        "max_newton_iters": 32,
        "z_thresh": 1e10,
        "exp_clip": 100000.0,
        "dynamic_polish": True,
        "overrelaxation_reject_steps": False,
        "overrelaxation_reject_max_tries": 3,
        "overrelaxation_reject_shrink": 0.5,
        "overrelaxation_reject_eps": 0.0,
        "experimental_exponential_newton_tol_progressive": True,
        "experimental_exponential_newton_tol_start": 1e-5,
        "experimental_exponential_newton_tol_end": 1e-5,
        "experimental_exponential_newton_tol_switch_hi": 1e-2,
        "experimental_exponential_newton_tol_switch_lo": 5e-4,
    },
}


@dataclass(frozen=True)
class RunSpec:
    run_dir: Path
    metrics: dict
    source_config: dict
    model_cfg: dict


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _model_cfg(config: dict) -> dict:
    model_key = config.get("lab", {}).get("model_key", "mnist_bp_conv_amp")
    model_cfg = {
        **config.get("model_base", {}),
        **config.get("model_overrides", {}).get(model_key, {}),
    }
    if "minimizer" not in model_cfg:
        model_cfg["minimizer"] = deepcopy(HISTORICAL_MINIMIZER_DEFAULTS)
    return model_cfg


def _path_part(run_dir: Path, prefix: str) -> str:
    for part in reversed(run_dir.parts):
        if part.startswith(prefix):
            return part[len(prefix) :]
    return ""


def _target_from_label(label: str) -> float:
    if label.startswith("sat"):
        try:
            return int(label[3:]) / 100.0
        except ValueError:
            return math.nan
    return math.nan


def _normalize_run_dir(path: str) -> Path:
    run_dir = Path(path).expanduser().resolve()
    if run_dir.name == "metrics.json":
        run_dir = run_dir.parent
    if not (run_dir / "source_config.json").exists():
        raise FileNotFoundError(f"Expected source_config.json under run dir: {run_dir}")
    return run_dir


def _discover_run_dirs(input_roots: list[str]) -> list[Path]:
    run_dirs: list[Path] = []
    seen: set[Path] = set()
    for raw in input_roots:
        path = Path(raw).expanduser().resolve()
        if (path / "source_config.json").exists():
            candidates = [path]
        else:
            candidates = [p.parent for p in path.glob("**/source_config.json")]
        for candidate in candidates:
            candidate = candidate.resolve()
            if candidate not in seen:
                seen.add(candidate)
                run_dirs.append(candidate)
    return sorted(run_dirs)


def _load_specs(run_dirs: list[Path]) -> list[RunSpec]:
    specs = []
    for run_dir in run_dirs:
        source_config = _load_json(run_dir / "source_config.json")
        specs.append(
            RunSpec(
                run_dir=run_dir,
                metrics=_load_json(run_dir / "metrics.json"),
                source_config=source_config,
                model_cfg=_model_cfg(source_config),
            )
        )
    return specs


def _best_specs_per_group(specs: list[RunSpec]) -> list[RunSpec]:
    best: dict[tuple[int, str, str], RunSpec] = {}
    for spec in specs:
        model_cfg = spec.model_cfg
        conv_depth = len(model_cfg.get("conv_pipeline") or [])
        non_linearity = str(model_cfg.get("non_linearity", ""))
        run_name = spec.run_dir.parent.name
        if conv_depth not in (1, 2):
            continue
        if non_linearity not in {"hard_sigmoid", "perfect_diode"}:
            continue
        if not (spec.run_dir / "final_model.pt").exists():
            continue
        key = (conv_depth, non_linearity, run_name)
        score = float(spec.metrics.get("best_test_accuracy", float("nan")))
        if not math.isfinite(score):
            continue
        prev = best.get(key)
        if prev is None or score > float(prev.metrics.get("best_test_accuracy", float("-inf"))):
            best[key] = spec
    return [best[key] for key in sorted(best)]


def _build_context(
    spec: RunSpec,
    *,
    device: torch.device,
    load_final: bool,
    batch_size: int,
    no_download: bool,
    dataset_root: str | None,
    inference_iterations_override: int | None,
    adaptive_equilibrium: bool | None = None,
) -> dict:
    config = load_config(spec.run_dir / "source_config.json")
    seed = int(config.get("seed", spec.metrics.get("seed", 0)))
    _reset_name_counters()
    _set_seed(seed)

    model_cfg = _model_cfg(config)
    dataset_key, dataset_cfg = _resolve_dataset_config(
        config,
        config.get("lab", {}).get("dataset_key", "mnist"),
    )
    del dataset_key

    layer_shapes = [
        tuple(shape) if isinstance(shape, (list, tuple)) else (shape,)
        for shape in model_cfg["layer_shapes"]
    ]
    energy_fn = FlexibleDeepResistiveEnergy(
        layer_shapes=layer_shapes,
        conv_pipeline=model_cfg.get("conv_pipeline") or [],
        pooling_mode=model_cfg.get("pooling_mode"),
        weight_gains=model_cfg["weight_gains"],
        input_gain=model_cfg["input_gain"],
        non_linearity=model_cfg["non_linearity"],
        exponential_diode_param=model_cfg["exponential_diode_param"],
        quadratic_diode_param=model_cfg["quadratic_diode_param"],
        hard_sigmoid_param=model_cfg.get("hard_sigmoid_param", {}),
        voltage_amp=model_cfg["voltage_amp"],
        current_amp=model_cfg["current_amp"],
        weight_min=model_cfg["weight_min"],
        weight_max=model_cfg["weight_max"],
        weight_init_mode=model_cfg.get("weight_init_mode", "kaiming_uniform"),
        input_mode=config.get("input_mode", "train"),
    )
    energy_fn.set_device(device)
    if load_final:
        checkpoint_path = spec.run_dir / "final_model.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Expected final checkpoint at {checkpoint_path}")
        energy_fn.load(checkpoint_path)

    network = Network(energy_fn)
    free_layers = network.free_layers()
    output_layer = energy_fn.layers()[-1]
    output_dim = output_layer.shape[0]
    num_classes = 10
    if output_dim == num_classes:
        cost_fn = SquaredError(output_layer)
    elif output_dim == 2 * num_classes:
        cost_fn = SquaredErrorPairedOutputs(output_layer, num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported output dimension {output_dim}.")
    source_iterations = int(model_cfg["num_iterations_inference"])
    inference_iterations = (
        int(inference_iterations_override)
        if inference_iterations_override is not None
        else source_iterations
    )
    minimizer = _build_tracking_minimizer(
        energy_fn,
        free_layers,
        model_cfg,
        config["energy_minimizer"]["mode"],
        num_iterations=inference_iterations,
        voltage_amp=energy_fn._voltage_amp,
        current_amp=energy_fn._current_amp,
        adaptive_equilibrium=adaptive_equilibrium,
    )

    dataset_factory = _resolve_callable(dataset_cfg["factory"])
    dataset_params = dict(dataset_cfg["params"])
    dataset_params["device"] = device
    dataset_params["batch_size"] = int(batch_size)
    if dataset_root is not None:
        dataset_params["root"] = os.path.expanduser(str(dataset_root))
    if "root" in dataset_params:
        dataset_params["root"] = os.path.expanduser(str(dataset_params["root"]))
    if no_download:
        dataset_params["download"] = False
    loader_result = dataset_factory(**dataset_params).build()
    if not isinstance(loader_result, tuple):
        raise ValueError("Expected dataset factory to return train/test loaders.")
    train_loader, test_loader = loader_result

    return {
        "energy_fn": energy_fn,
        "network": network,
        "free_layers": free_layers,
        "output_layer": output_layer,
        "cost_fn": cost_fn,
        "minimizer": minimizer,
        "train_loader": train_loader,
        "test_loader": test_loader,
        "model_cfg": model_cfg,
    }


def _hard_sigmoid_mask(state: torch.Tensor, params: dict) -> torch.Tensor:
    if "v_off" in params and "v_min" not in params and "v_max" not in params:
        v_off = float(params["v_off"])
        v_min = -v_off
        v_max = v_off
    else:
        v_min = float(params.get("v_min", -math.inf))
        v_max = float(params.get("v_max", math.inf))
    return (state < v_min) | (state > v_max)


def _perfect_diode_fraction(state: torch.Tensor, *, eps: float) -> tuple[int, int]:
    if state.ndim < 2 or state.shape[1] % 2 != 0:
        return 0, 0
    half = state.shape[1] // 2
    excitatory = state[:, :half]
    inhibitory = state[:, half:]
    saturated = int((excitatory <= eps).sum().item()) + int((inhibitory >= -eps).sum().item())
    total = int(excitatory.numel() + inhibitory.numel())
    return saturated, total


def _layer_saturation(
    state: torch.Tensor,
    *,
    non_linearity: str,
    hard_sigmoid_param: dict,
    perfect_eps: float,
) -> tuple[float, int]:
    if non_linearity == "hard_sigmoid":
        saturated = int(_hard_sigmoid_mask(state, hard_sigmoid_param).sum().item())
        total = int(state.numel())
        return (saturated / total if total else math.nan), total
    if non_linearity == "perfect_diode":
        saturated, total = _perfect_diode_fraction(state, eps=perfect_eps)
        return (saturated / total if total else math.nan), total
    return math.nan, 0


def _measure_context(
    context: dict,
    *,
    split: str,
    max_samples: int,
    perfect_eps: float,
) -> dict:
    loader = context["train_loader"] if split == "train" else context["test_loader"]
    layers = context["energy_fn"].layers()
    hidden_layers = layers[1:-1]
    model_cfg = context["model_cfg"]
    non_linearity = str(model_cfg.get("non_linearity", ""))
    hard_sigmoid_param = dict(model_cfg.get("hard_sigmoid_param", {}))

    layer_saturated = [0.0 for _ in hidden_layers]
    layer_weight = [0 for _ in hidden_layers]
    total_saturated = 0.0
    total_weight = 0
    seen = 0

    with torch.no_grad():
        for images, _labels in loader:
            if seen >= max_samples:
                break
            if seen + int(images.shape[0]) > max_samples:
                images = images[: max_samples - seen]
            images = images.to(next(iter(context["energy_fn"].params())).state.device)
            context["network"].set_input(images, reset=True)
            context["minimizer"].compute_equilibrium()
            seen += int(images.shape[0])

            for index, layer in enumerate(hidden_layers):
                state = layer.state.detach()
                fraction, count = _layer_saturation(
                    state,
                    non_linearity=non_linearity,
                    hard_sigmoid_param=hard_sigmoid_param,
                    perfect_eps=perfect_eps,
                )
                if not math.isfinite(fraction) or count == 0:
                    continue
                layer_saturated[index] += fraction * count
                layer_weight[index] += count
                total_saturated += fraction * count
                total_weight += count

    layer_values = [
        layer_saturated[index] / layer_weight[index] if layer_weight[index] else math.nan
        for index in range(len(hidden_layers))
    ]
    return {
        "num_samples": seen,
        "first_hidden_saturation": layer_values[0] if layer_values else math.nan,
        "all_hidden_saturation": total_saturated / total_weight if total_weight else math.nan,
        "layer_saturations": layer_values,
    }


def _measure_spec(spec: RunSpec, args: argparse.Namespace, device: torch.device) -> dict:
    init_context = _build_context(
        spec,
        device=device,
        load_final=False,
        batch_size=args.batch_size,
        no_download=args.no_download,
        dataset_root=args.dataset_root,
        inference_iterations_override=args.inference_iterations,
    )
    init_result = _measure_context(
        init_context,
        split=args.split,
        max_samples=args.max_samples,
        perfect_eps=args.perfect_eps,
    )
    del init_context

    if args.init_only:
        final_result = {
            "num_samples": init_result["num_samples"],
            "first_hidden_saturation": math.nan,
            "all_hidden_saturation": math.nan,
            "layer_saturations": [],
        }
    else:
        final_context = _build_context(
            spec,
            device=device,
            load_final=True,
            batch_size=args.batch_size,
            no_download=args.no_download,
            dataset_root=args.dataset_root,
            inference_iterations_override=args.inference_iterations,
        )
        final_result = _measure_context(
            final_context,
            split=args.split,
            max_samples=args.max_samples,
            perfect_eps=args.perfect_eps,
        )

    model_cfg = spec.model_cfg
    target_label = _path_part(spec.run_dir, "target_")
    hard_sigmoid_param = model_cfg.get("hard_sigmoid_param", {})
    v_off = float(
        hard_sigmoid_param.get("v_off", hard_sigmoid_param.get("v_max", math.nan))
    )
    init_first = init_result["first_hidden_saturation"]
    init_all = init_result["all_hidden_saturation"]
    final_first = final_result["first_hidden_saturation"]
    final_all = final_result["all_hidden_saturation"]
    return {
        "conv_depth": len(model_cfg.get("conv_pipeline") or []),
        "non_linearity": model_cfg.get("non_linearity"),
        "run_name": spec.run_dir.parent.name,
        "voltage_amp": model_cfg.get("voltage_amp", math.nan),
        "current_amp": model_cfg.get("current_amp", math.nan),
        "input_gain": model_cfg.get("input_gain", math.nan),
        "v_off": v_off,
        "lr": json.dumps(spec.source_config.get("lr", "")),
        "epochs": spec.source_config.get("lab", {}).get("epochs", ""),
        "seed": spec.source_config.get("seed", spec.metrics.get("seed", "")),
        "run_dir": str(spec.run_dir),
        "best_test_accuracy": spec.metrics.get("best_test_accuracy", math.nan),
        "final_test_accuracy": spec.metrics.get("final_test_accuracy", math.nan),
        "best_epoch": spec.metrics.get("best_epoch", ""),
        "target_label": target_label,
        "target_saturation": _target_from_label(target_label),
        "split": args.split,
        "num_samples": min(init_result["num_samples"], final_result["num_samples"]),
        "init_first_hidden_saturation": init_first,
        "init_all_hidden_saturation": init_all,
        "init_layer_saturations": json.dumps(init_result["layer_saturations"]),
        "final_first_hidden_saturation": final_first,
        "final_all_hidden_saturation": final_all,
        "final_layer_saturations": json.dumps(final_result["layer_saturations"]),
        "delta_first_hidden_saturation": final_first - init_first,
        "delta_all_hidden_saturation": final_all - init_all,
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", action="append", default=[])
    parser.add_argument("--run-dir", action="append", default=[])
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--best-per-group", action="store_true")
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--dataset-root", default=str(REPO_ROOT / "data"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--inference-iterations", type=int, default=None)
    parser.add_argument(
        "--init-only",
        action="store_true",
        help="Measure initialization only; final-checkpoint columns are written as NaN.",
    )
    parser.add_argument("--perfect-eps", type=float, default=1e-8)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_root and not args.run_dir:
        raise ValueError("Provide at least one --input-root or --run-dir.")

    run_dirs = [_normalize_run_dir(path) for path in args.run_dir]
    run_dirs.extend(_discover_run_dirs(args.input_root))
    specs = _load_specs(run_dirs)
    if args.best_per_group:
        specs = _best_specs_per_group(specs)

    specs = sorted(
        specs,
        key=lambda spec: (
            len(spec.model_cfg.get("conv_pipeline") or []),
            str(spec.model_cfg.get("non_linearity", "")),
            spec.run_dir.parent.name,
            str(spec.run_dir),
        ),
    )
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(
        f"[discover] runs={len(specs)} device={device} split={args.split} "
        f"max_samples={args.max_samples}"
    )
    if args.dry_run:
        for spec in specs:
            score = spec.metrics.get("best_test_accuracy", math.nan)
            print(
                f"[dry-run] conv{len(spec.model_cfg.get('conv_pipeline') or [])} "
                f"{spec.model_cfg.get('non_linearity')} {spec.run_dir.parent.name} "
                f"best={score} {spec.run_dir}"
            )
        return

    rows = []
    for index, spec in enumerate(specs, start=1):
        print(
            f"[measure] {index}/{len(specs)} conv{len(spec.model_cfg.get('conv_pipeline') or [])} "
            f"{spec.model_cfg.get('non_linearity')} {spec.run_dir.parent.name}",
            flush=True,
        )
        rows.append(_measure_spec(spec, args, device))
    _write_csv(Path(args.output_csv), rows)
    print(f"[done] wrote {args.output_csv}")


if __name__ == "__main__":
    main()
