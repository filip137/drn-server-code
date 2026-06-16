#!/usr/bin/env python3
"""Evaluate clean-trained MNIST DRNs under persistent test-time write noise."""

from __future__ import annotations

import argparse
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

from custom_classes import FlexibleDeepResistiveEnergy  # noqa: E402
from labs.mnist_train import (  # noqa: E402
    _build_tracking_minimizer,
    _json_sanitize,
    _reset_name_counters,
    _resolve_callable,
    _resolve_dataset_config,
    _set_seed,
    load_config,
)
from model.function.cost import SquaredError, SquaredErrorPairedOutputs  # noqa: E402
from model.function.network import Network  # noqa: E402


DEFAULT_INPUT_ROOT = (
    REPO_ROOT / "results" / "mnist_bp_amplification_sweep_drn_xs_10epoch_legacy_preproc"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "mnist_bp_write_noise_sweep"
DEFAULT_SIGMAS = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2]
RESULT_COLUMNS = [
    "run_name",
    "voltage_amp",
    "current_amp",
    "training_seed",
    "checkpoint_kind",
    "checkpoint_path",
    "sigma",
    "noise_seed",
    "clean_accuracy",
    "noisy_accuracy",
    "accuracy_drop",
    "test_loss",
    "mean_logit_margin",
    "solver_converged_fraction",
    "mean_solver_iterations",
    "mean_kcl_residual",
    "fraction_saturated_nodes",
    "noised_param_names",
]
SUMMARY_COLUMNS = [
    "run_name",
    "voltage_amp",
    "current_amp",
    "training_seed",
    "sigma",
    "num_noise_seeds",
    "clean_accuracy",
    "mean_noisy_accuracy",
    "std_noisy_accuracy",
    "mean_accuracy_drop",
    "mean_test_loss",
    "mean_logit_margin",
    "mean_fraction_saturated_nodes",
]
RAUC_COLUMNS = [
    "run_name",
    "voltage_amp",
    "current_amp",
    "num_training_seeds",
    "num_noise_seeds_per_sigma",
    "rauc_mean_accuracy",
    "sigma_1pct",
    "clean_accuracy_mean",
]


@dataclass(frozen=True)
class RunRow:
    non_linearity: str
    run_name: str
    seed: int
    voltage_amp: float
    current_amp: float
    checkpoint_path: Path
    clean_accuracy: float
    run_dir: Path


def _read_summary_rows(
    input_root: Path,
    *,
    checkpoint_kind: str,
    run_names: set[str] | None,
    training_seeds: set[int] | None,
) -> list[RunRow]:
    summary_path = input_root / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Expected summary.csv at {summary_path}.")

    rows: list[RunRow] = []
    with summary_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            run_name = row["run_name"]
            seed = int(row["seed"])
            if run_names is not None and run_name not in run_names:
                continue
            if training_seeds is not None and seed not in training_seeds:
                continue
            if checkpoint_kind == "best":
                ckpt = Path(row["weights_best_path"]).with_name("best_model.pt")
                clean_acc = float(row["best_test_accuracy"])
            elif checkpoint_kind == "final":
                ckpt = Path(row["checkpoint_path"])
                clean_acc = float(row["final_test_accuracy"])
            else:
                raise ValueError(f"Unknown checkpoint_kind {checkpoint_kind!r}.")
            rows.append(
                RunRow(
                    non_linearity=row.get("non_linearity", ""),
                    run_name=run_name,
                    seed=seed,
                    voltage_amp=float(row["voltage_amp"]),
                    current_amp=float(row["current_amp"]),
                    checkpoint_path=ckpt,
                    clean_accuracy=clean_acc,
                    run_dir=ckpt.parent,
                )
            )
    if not rows:
        raise ValueError("No runs matched the requested filters.")
    return rows


def _select_shard(rows: list[RunRow], num_shards: int, shard_index: int) -> list[RunRow]:
    if num_shards < 1:
        raise ValueError("--num-shards must be >= 1.")
    if not 0 <= shard_index < num_shards:
        raise ValueError("Expected 0 <= --shard-index < --num-shards.")
    return [row for index, row in enumerate(rows) if index % num_shards == shard_index]


def _build_eval_context(
    run: RunRow,
    *,
    device: torch.device,
    eval_batch_size: int | None,
    no_download: bool,
    inference_iterations_override: int | None,
) -> dict:
    source_config_path = run.run_dir / "source_config.json"
    if not source_config_path.exists():
        raise FileNotFoundError(f"Expected source_config.json at {source_config_path}.")
    if not run.checkpoint_path.exists():
        raise FileNotFoundError(f"Expected checkpoint at {run.checkpoint_path}.")

    config = load_config(source_config_path)
    _reset_name_counters()
    _set_seed(run.seed)

    model_key = config.get("lab", {}).get("model_key", "mnist_bp_amp")
    model_overrides = config["model_overrides"]
    model_cfg = {**config["model_base"], **model_overrides[model_key]}
    dataset_key, dataset_cfg = _resolve_dataset_config(config, config.get("lab", {}).get("dataset_key", "mnist"))

    layer_shapes = [
        tuple(shape) if isinstance(shape, (list, tuple)) else (shape,)
        for shape in model_cfg["layer_shapes"]
    ]
    conv_pipeline = model_cfg.get("conv_pipeline") or []
    energy_fn = FlexibleDeepResistiveEnergy(
        layer_shapes=layer_shapes,
        conv_pipeline=conv_pipeline,
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
    energy_fn.load(run.checkpoint_path)

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

    source_inference_iterations = int(model_cfg["num_iterations_inference"])
    inference_iterations = (
        int(inference_iterations_override)
        if inference_iterations_override is not None
        else source_inference_iterations
    )
    minimizer = _build_tracking_minimizer(
        energy_fn,
        free_layers,
        model_cfg,
        config["energy_minimizer"]["mode"],
        num_iterations=inference_iterations,
        voltage_amp=energy_fn._voltage_amp,
        current_amp=energy_fn._current_amp,
    )

    dataset_factory = _resolve_callable(dataset_cfg["factory"])
    dataset_params = dict(dataset_cfg["params"])
    dataset_params["device"] = device
    if eval_batch_size is not None:
        dataset_params["batch_size"] = int(eval_batch_size)
    if "root" in dataset_params:
        dataset_params["root"] = os.path.expanduser(str(dataset_params["root"]))
    if no_download:
        dataset_params["download"] = False
    loader_result = dataset_factory(**dataset_params).build()
    if not isinstance(loader_result, tuple):
        raise ValueError("Expected MNIST dataset factory to return train and test loaders.")
    train_loader, test_loader = loader_result

    params = energy_fn.params()
    base_states = [param.state.detach().clone() for param in params]
    noisable_indices = [
        index
        for index, param in enumerate(params)
        if "Weight" in param.__class__.__name__
    ]

    return {
        "config": config,
        "model_cfg": model_cfg,
        "energy_fn": energy_fn,
        "network": network,
        "free_layers": free_layers,
        "output_layer": output_layer,
        "cost_fn": cost_fn,
        "minimizer": minimizer,
        "train_loader": train_loader,
        "test_loader": test_loader,
        "params": params,
        "base_states": base_states,
        "noisable_indices": noisable_indices,
        "inference_iterations": inference_iterations,
        "source_inference_iterations": source_inference_iterations,
        "dataset_params": dataset_params,
        "num_classes": num_classes,
    }


def _reset_params(context: dict) -> None:
    for param, base in zip(context["params"], context["base_states"]):
        param.state = base.detach().clone()


def _apply_lognormal_write_noise(
    context: dict,
    *,
    sigma: float,
    noise_seed: int,
    include_biases: bool,
) -> list[str]:
    params = context["params"]
    base_states = context["base_states"]
    if include_biases:
        indices = [
            index
            for index, param in enumerate(params)
            if "Weight" in param.__class__.__name__ or "Bias" in param.__class__.__name__
        ]
    else:
        indices = context["noisable_indices"]

    generator = torch.Generator(device=str(base_states[0].device))
    generator.manual_seed(int(noise_seed))
    noised_names = []
    for index, param in enumerate(params):
        base = base_states[index]
        if index not in indices or sigma == 0.0:
            param.state = base.detach().clone()
            continue
        xi = torch.randn(
            base.shape,
            generator=generator,
            device=base.device,
            dtype=base.dtype,
        )
        param.state = base * torch.exp(float(sigma) * xi)
        param.clamp_()
        noised_names.append(getattr(param, "name", f"param_{index}"))
    return noised_names


def _scores(output: torch.Tensor, num_classes: int) -> torch.Tensor:
    if output.shape[1] == num_classes:
        return output
    if output.shape[1] == 2 * num_classes:
        return output.view(output.shape[0], num_classes, 2)[..., 0] - output.view(
            output.shape[0], num_classes, 2
        )[..., 1]
    raise ValueError(f"Unsupported output shape {tuple(output.shape)}.")


def _mean_margin(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    true_scores = scores.gather(1, labels.view(-1, 1)).squeeze(1)
    mask = torch.ones_like(scores, dtype=torch.bool)
    mask.scatter_(1, labels.view(-1, 1), False)
    other_scores = scores.masked_fill(~mask, float("-inf")).max(dim=1).values
    return true_scores - other_scores


def _saturation_fraction(free_layers: list, *, eps: float = 1e-8) -> tuple[float, int]:
    saturated = 0
    total = 0
    for layer in free_layers[:-1]:
        state = layer.state.detach()
        if state.ndim < 2 or state.shape[1] % 2 != 0:
            continue
        half = state.shape[1] // 2
        excitatory = state[:, :half]
        inhibitory = state[:, half:]
        saturated += int((excitatory <= eps).sum().item())
        saturated += int((inhibitory >= -eps).sum().item())
        total += int(excitatory.numel() + inhibitory.numel())
    if total == 0:
        return float("nan"), 0
    return saturated / total, total


def _evaluate(context: dict, *, max_eval_batches: int | None = None) -> dict:
    network = context["network"]
    minimizer = context["minimizer"]
    cost_fn = context["cost_fn"]
    output_layer = context["output_layer"]
    test_loader = context["test_loader"]
    device = context["base_states"][0].device
    num_classes = context["num_classes"]

    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    margin_sum = 0.0
    saturated_sum = 0.0
    saturated_weight = 0

    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(test_loader):
            if max_eval_batches is not None and batch_index >= max_eval_batches:
                break
            images = images.to(device)
            labels = labels.to(device)
            network.set_input(images, reset=True)
            minimizer.compute_equilibrium()
            cost_fn.set_target(labels)

            batch_loss = cost_fn.eval().mean().item()
            errors = cost_fn.error_fn()
            batch_correct = int((~errors).sum().item())
            output_scores = _scores(output_layer.state.detach(), num_classes)
            margins = _mean_margin(output_scores, labels)
            saturation, saturation_count = _saturation_fraction(network.free_layers())

            batch_size = int(images.size(0))
            total_loss += batch_loss * batch_size
            total_correct += batch_correct
            total_seen += batch_size
            margin_sum += float(margins.sum().item())
            if saturation_count:
                saturated_sum += saturation * saturation_count
                saturated_weight += saturation_count

    if total_seen == 0:
        raise ValueError("Evaluation saw zero examples.")
    return {
        "accuracy": total_correct / total_seen,
        "loss": total_loss / total_seen,
        "mean_logit_margin": margin_sum / total_seen,
        "fraction_saturated_nodes": (
            saturated_sum / saturated_weight if saturated_weight else float("nan")
        ),
        "solver_converged_fraction": float("nan"),
        "mean_solver_iterations": float(context["inference_iterations"]),
        "mean_kcl_residual": float("nan"),
        "num_examples": total_seen,
    }


def _write_rows(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({column: _json_sanitize(row.get(column, "")) for column in columns})


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_sanitize(payload), indent=2, sort_keys=True))


def write_summaries(output_root: Path) -> None:
    raw_paths = sorted(output_root.glob("raw_results*.csv"))
    if not raw_paths:
        return
    rows = []
    for raw_path in raw_paths:
        rows.extend(csv.DictReader(raw_path.open()))
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (
            row["run_name"],
            row["voltage_amp"],
            row["current_amp"],
            row["training_seed"],
            row["sigma"],
        )
        grouped.setdefault(key, []).append(row)

    clean_by_model: dict[tuple[str, str], float] = {}
    for key, values in grouped.items():
        run_name, _voltage_amp, _current_amp, training_seed, sigma = key
        if float(sigma) != 0.0:
            continue
        acc = np.asarray([float(v["noisy_accuracy"]) for v in values], dtype=float)
        clean_by_model[(run_name, training_seed)] = float(np.mean(acc))

    summary_rows = []
    for key, values in sorted(grouped.items()):
        run_name, voltage_amp, current_amp, training_seed, sigma = key
        acc = np.asarray([float(v["noisy_accuracy"]) for v in values], dtype=float)
        loss = np.asarray([float(v["test_loss"]) for v in values], dtype=float)
        margin = np.asarray([float(v["mean_logit_margin"]) for v in values], dtype=float)
        saturation = np.asarray([float(v["fraction_saturated_nodes"]) for v in values], dtype=float)
        clean = clean_by_model.get((run_name, training_seed), float(values[0]["clean_accuracy"]))
        summary_rows.append(
            {
                "run_name": run_name,
                "voltage_amp": voltage_amp,
                "current_amp": current_amp,
                "training_seed": training_seed,
                "sigma": sigma,
                "num_noise_seeds": len(values),
                "clean_accuracy": clean,
                "mean_noisy_accuracy": float(np.mean(acc)),
                "std_noisy_accuracy": float(np.std(acc, ddof=1)) if len(acc) > 1 else 0.0,
                "mean_accuracy_drop": clean - float(np.mean(acc)),
                "mean_test_loss": float(np.mean(loss)),
                "mean_logit_margin": float(np.mean(margin)),
                "mean_fraction_saturated_nodes": float(np.nanmean(saturation)),
            }
        )

    summary_path = output_root / "summary_by_model_sigma.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary_rows)

    by_run: dict[str, list[dict]] = {}
    for row in summary_rows:
        by_run.setdefault(row["run_name"], []).append(row)
    rauc_rows = []
    for run_name, values in sorted(by_run.items()):
        by_sigma: dict[float, list[dict]] = {}
        for row in values:
            by_sigma.setdefault(float(row["sigma"]), []).append(row)
        sigmas = np.asarray(sorted(by_sigma), dtype=float)
        mean_acc = np.asarray(
            [
                np.mean([float(row["mean_noisy_accuracy"]) for row in by_sigma[sigma]])
                for sigma in sigmas
            ],
            dtype=float,
        )
        clean_mean = float(np.mean([float(row["clean_accuracy"]) for row in by_sigma[sigmas[0]]]))
        target = clean_mean - 0.01
        sigma_1pct = float("nan")
        below = np.where(mean_acc <= target)[0]
        if below.size:
            idx = int(below[0])
            if idx == 0:
                sigma_1pct = float(sigmas[0])
            else:
                x0, x1 = sigmas[idx - 1], sigmas[idx]
                y0, y1 = mean_acc[idx - 1], mean_acc[idx]
                if y1 != y0:
                    sigma_1pct = float(x0 + (target - y0) * (x1 - x0) / (y1 - y0))
                else:
                    sigma_1pct = float(x1)
        first = values[0]
        rauc_rows.append(
            {
                "run_name": run_name,
                "voltage_amp": first["voltage_amp"],
                "current_amp": first["current_amp"],
                "num_training_seeds": len({row["training_seed"] for row in values}),
                "num_noise_seeds_per_sigma": max(int(row["num_noise_seeds"]) for row in values),
                "rauc_mean_accuracy": float(np.trapz(mean_acc, sigmas)),
                "sigma_1pct": sigma_1pct,
                "clean_accuracy_mean": clean_mean,
            }
        )
    with (output_root / "rauc_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAUC_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rauc_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default=str(DEFAULT_INPUT_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--device", default=None)
    parser.add_argument("--checkpoint-kind", choices=("best", "final"), default="best")
    parser.add_argument("--sigmas", type=float, nargs="+", default=DEFAULT_SIGMAS)
    parser.add_argument("--noise-seeds", type=int, nargs="+", default=list(range(30)))
    parser.add_argument("--run-name", action="append", help="Restrict to one or more run names.")
    parser.add_argument("--training-seeds", type=int, nargs="+")
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument(
        "--inference-iterations",
        type=int,
        default=None,
        help="Override num_iterations_inference from the checkpoint config during evaluation.",
    )
    parser.add_argument("--max-eval-batches", type=int)
    parser.add_argument("--include-bias-noise", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--raw-results-name", default=None)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    raw_results_name = args.raw_results_name or f"raw_results_shard_{args.shard_index}.csv"
    raw_results_path = output_root / raw_results_name

    if args.summary_only:
        write_summaries(output_root)
        print(f"[summary] wrote summaries under {output_root}")
        return

    requested_device = args.device
    if requested_device is not None:
        device = torch.device(requested_device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"Requested CUDA device {requested_device!r}, but CUDA is unavailable.")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = _read_summary_rows(
        input_root,
        checkpoint_kind=args.checkpoint_kind,
        run_names=set(args.run_name) if args.run_name else None,
        training_seeds=set(args.training_seeds) if args.training_seeds else None,
    )
    selected = _select_shard(rows, args.num_shards, args.shard_index)
    config_payload = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "checkpoint_kind": args.checkpoint_kind,
        "sigmas": [float(value) for value in args.sigmas],
        "noise_seeds": [int(value) for value in args.noise_seeds],
        "eval_batch_size": args.eval_batch_size,
        "inference_iterations_override": args.inference_iterations,
        "include_bias_noise": bool(args.include_bias_noise),
        "device": str(device),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "selected_runs": [f"{row.run_name}/seed_{row.seed}" for row in selected],
        "noise_model": "persistent lognormal conductance noise on Weight parameters",
    }
    _write_json(output_root / f"config_shard_{args.shard_index}.json", config_payload)

    print(f"[write-noise] input_root={input_root}")
    print(f"[write-noise] output_root={output_root}")
    print(f"[write-noise] selected_runs={len(selected)}/{len(rows)} shard={args.shard_index}/{args.num_shards}")
    if args.dry_run:
        for row in selected:
            print(f"[dry-run] {row.run_name} seed={row.seed} checkpoint={row.checkpoint_path}")
        return

    raw_rows = []
    for run in selected:
        print(f"[run] {run.run_name} seed={run.seed} checkpoint={run.checkpoint_path}")
        context = _build_eval_context(
            run,
            device=device,
            eval_batch_size=args.eval_batch_size,
            no_download=args.no_download,
            inference_iterations_override=args.inference_iterations,
        )
        for sigma in args.sigmas:
            for noise_seed in args.noise_seeds:
                _reset_params(context)
                noised_param_names = _apply_lognormal_write_noise(
                    context,
                    sigma=float(sigma),
                    noise_seed=int(noise_seed),
                    include_biases=args.include_bias_noise,
                )
                metrics = _evaluate(context, max_eval_batches=args.max_eval_batches)
                row = {
                    "run_name": run.run_name,
                    "voltage_amp": run.voltage_amp,
                    "current_amp": run.current_amp,
                    "training_seed": run.seed,
                    "checkpoint_kind": args.checkpoint_kind,
                    "checkpoint_path": str(run.checkpoint_path),
                    "sigma": float(sigma),
                    "noise_seed": int(noise_seed),
                    "clean_accuracy": run.clean_accuracy,
                    "noisy_accuracy": metrics["accuracy"],
                    "accuracy_drop": run.clean_accuracy - metrics["accuracy"],
                    "test_loss": metrics["loss"],
                    "mean_logit_margin": metrics["mean_logit_margin"],
                    "solver_converged_fraction": metrics["solver_converged_fraction"],
                    "mean_solver_iterations": metrics["mean_solver_iterations"],
                    "mean_kcl_residual": metrics["mean_kcl_residual"],
                    "fraction_saturated_nodes": metrics["fraction_saturated_nodes"],
                    "noised_param_names": ";".join(noised_param_names),
                }
                raw_rows.append(row)
                print(
                    f"[result] {run.run_name} seed={run.seed} sigma={float(sigma):.4g} "
                    f"noise_seed={int(noise_seed)} acc={metrics['accuracy']*100:.2f}% "
                    f"drop={(run.clean_accuracy - metrics['accuracy'])*100:.2f}%"
                )
                if len(raw_rows) >= 16:
                    _write_rows(raw_results_path, RESULT_COLUMNS, raw_rows)
                    raw_rows = []
        if raw_rows:
            _write_rows(raw_results_path, RESULT_COLUMNS, raw_rows)
        raw_rows = []
        write_summaries(output_root)

    if raw_rows:
        _write_rows(raw_results_path, RESULT_COLUMNS, raw_rows)
    write_summaries(output_root)
    print(f"[done] output={output_root}")


if __name__ == "__main__":
    main()
