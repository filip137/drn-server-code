#!/usr/bin/env python3
"""Replay a frozen Conv checkpoint on the ordinary-MNIST validation split.

This is a read-only implementation-change diagnostic.  It loads the exact
stored parameter tensors, recomputes free equilibria with the current source
tree, and never constructs or reads the official MNIST test split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from experiments.reporting import (
    append_metric,
    atomic_write_json,
    complete_run,
    fail_run,
    runtime_context,
    sha256_file,
    start_run,
    update_status_progress,
)
from labs import mnist_train as trainer
from model.function.cost import SquaredError, SquaredErrorPairedOutputs
from model.function.network import Network


REPO_ROOT = Path(__file__).resolve().parents[1]
ENERGY_CONTRACT = "physical-kcl-diagonally-symmetrized-energy/v1"


def _git_state() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"commit": commit, "dirty": dirty}


def _resolved_model_config(config: Mapping[str, Any]) -> dict[str, Any]:
    model_overrides = config.get("model_overrides")
    if not isinstance(model_overrides, Mapping) or not model_overrides:
        raise ValueError("Expected a non-empty model_overrides object.")
    requested = config.get("lab", {}).get("model_key")
    if requested in model_overrides:
        selected = model_overrides[requested]
    elif len(model_overrides) == 1:
        selected = next(iter(model_overrides.values()))
    else:
        raise KeyError(
            f"Cannot resolve model override {requested!r}; available={sorted(model_overrides)}."
        )
    model_base = config.get("model_base")
    if not isinstance(model_base, Mapping) or not isinstance(selected, Mapping):
        raise ValueError("Expected model_base and the selected model override to be objects.")
    model_config = {**model_base, **selected}
    trainer._validate_diode_param_config(model_config)
    trainer._require_minimizer_config(model_config)
    return model_config


def _build_runtime(
    *,
    config_path: Path,
    checkpoint_path: Path,
    dataset_root: Path,
    validation_batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    config = trainer.load_config(config_path)
    evaluation = config.get("evaluation", {})
    official_test = evaluation.get("official_test", {}) if isinstance(evaluation, Mapping) else {}
    if not isinstance(official_test, Mapping) or official_test.get("policy") != "disabled":
        raise ValueError(
            "Checkpoint replay is validation-only and requires official_test.policy='disabled'."
        )

    trainer._reset_name_counters()
    trainer._set_seed(config.get("seed", 0))
    model_config = _resolved_model_config(config)
    layer_shapes = [
        tuple(shape) if isinstance(shape, (list, tuple)) else (shape,)
        for shape in model_config["layer_shapes"]
    ]
    energy_fn = trainer.FlexibleDeepResistiveEnergy(
        layer_shapes=layer_shapes,
        conv_pipeline=model_config.get("conv_pipeline") or [],
        pooling_mode=model_config.get("pooling_mode"),
        weight_gains=model_config["weight_gains"],
        input_gain=model_config["input_gain"],
        non_linearity=model_config["non_linearity"],
        exponential_diode_param=model_config["exponential_diode_param"],
        quadratic_diode_param=model_config["quadratic_diode_param"],
        hard_sigmoid_param=model_config["hard_sigmoid_param"],
        voltage_amp=model_config["voltage_amp"],
        current_amp=model_config["current_amp"],
        weight_min=model_config["weight_min"],
        weight_max=model_config["weight_max"],
        weight_init_mode=model_config.get("weight_init_mode", "kaiming_uniform"),
        input_mode=config.get("input_mode", "train"),
        trainable_amplification=bool(model_config.get("trainable_amplification", False)),
        amplification_min=model_config.get("amplification_min", 1e-6),
        amplification_max=model_config.get("amplification_max"),
    )
    energy_fn.set_device(device)
    energy_fn.load(checkpoint_path)
    runtime_dtype_name, runtime_dtype = trainer._runtime_dtype_from_config(config)
    trainer._convert_energy_runtime_dtype(energy_fn, runtime_dtype)

    network = Network(energy_fn)
    free_layers = network.free_layers()
    output_layer = energy_fn.layers()[-1]
    if output_layer.shape == (10,):
        cost_fn = SquaredError(output_layer)
    elif output_layer.shape == (20,):
        cost_fn = SquaredErrorPairedOutputs(output_layer, num_classes=10)
    else:
        raise ValueError(f"Expected output shape (10,) or (20,), got {output_layer.shape!r}.")

    inference_iterations = int(model_config["num_iterations_inference"])
    minimizer = trainer._build_tracking_minimizer(
        energy_fn,
        free_layers,
        model_config,
        config["energy_minimizer"]["mode"],
        num_iterations=inference_iterations,
        voltage_amp=energy_fn._voltage_amp,
        current_amp=energy_fn._current_amp,
    )

    dataset_key, dataset_config = trainer._resolve_dataset_config(
        config, config.get("lab", {}).get("dataset_key", "mnist")
    )
    dataset_factory = trainer._resolve_callable(dataset_config["factory"])
    dataset_params = dict(dataset_config["params"])
    dataset_params.update(
        {
            "device": device,
            "root": str(dataset_root),
            "download": False,
            "validation_batch_size": int(validation_batch_size),
        }
    )
    loaders = dataset_factory(**dataset_params).build()
    if not hasattr(loaders, "validation_loader"):
        raise ValueError(
            "Expected a deterministic train/validation loader; refusing a loader that may use test."
        )

    return {
        "config": config,
        "model_config": model_config,
        "dataset_key": dataset_key,
        "dataset_params": dataset_params,
        "dataset_provenance": {
            "split_seed": int(loaders.split_seed),
            "shuffle_seed": int(loaders.shuffle_seed),
            "train_indices_sha256": loaders.train_indices_hash,
            "validation_indices_sha256": loaders.validation_indices_hash,
            "first_epoch_batch_order_sha256": loaders.first_epoch_batch_order_hash,
        },
        "validation_loader": loaders.validation_loader,
        "energy_fn": energy_fn,
        "network": network,
        "free_layers": free_layers,
        "output_layer": output_layer,
        "cost_fn": cost_fn,
        "minimizer": minimizer,
        "runtime_dtype": runtime_dtype,
        "runtime_dtype_name": runtime_dtype_name,
        "inference_iterations": inference_iterations,
    }


def _empty_state_stat(layer: Any) -> dict[str, Any]:
    return {
        "layer": str(layer.name),
        "count": 0,
        "sum": 0.0,
        "sum_abs": 0.0,
        "sum_sq": 0.0,
        "min": math.inf,
        "max": -math.inf,
    }


def _update_state_stat(stat: dict[str, Any], value: torch.Tensor) -> None:
    detached = value.detach()
    stat["count"] += int(detached.numel())
    stat["sum"] += float(detached.sum().item())
    stat["sum_abs"] += float(detached.abs().sum().item())
    stat["sum_sq"] += float((detached * detached).sum().item())
    stat["min"] = min(float(stat["min"]), float(detached.min().item()))
    stat["max"] = max(float(stat["max"]), float(detached.max().item()))


def _finish_state_stat(stat: Mapping[str, Any]) -> dict[str, Any]:
    count = int(stat["count"])
    if count <= 0:
        raise RuntimeError(f"No values accumulated for {stat['layer']}.")
    mean = float(stat["sum"]) / count
    return {
        "layer": stat["layer"],
        "count": count,
        "mean": mean,
        "mean_abs": float(stat["sum_abs"]) / count,
        "rms": math.sqrt(float(stat["sum_sq"]) / count),
        "std": math.sqrt(max(0.0, float(stat["sum_sq"]) / count - mean * mean)),
        "min": float(stat["min"]),
        "max": float(stat["max"]),
    }


def _score_tensor(cost_fn: Any, output: torch.Tensor) -> torch.Tensor:
    if isinstance(cost_fn, SquaredErrorPairedOutputs):
        return cost_fn._scores(output)
    return output


def _evaluate(
    runtime: Mapping[str, Any],
    *,
    max_validation_batches: int | None,
    run_dir: Path,
) -> dict[str, Any]:
    loader = runtime["validation_loader"]
    state_stats = [_empty_state_stat(layer) for layer in runtime["free_layers"]]
    score_stat = {
        "layer": "paired_scores",
        "count": 0,
        "sum": 0.0,
        "sum_abs": 0.0,
        "sum_sq": 0.0,
        "min": math.inf,
        "max": -math.inf,
    }
    prediction_hash = hashlib.sha256()
    label_hash = hashlib.sha256()
    total_loss = 0.0
    total_correct = 0
    seen = 0
    evaluated_batches = 0
    total_batches = len(loader)

    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(loader):
            if max_validation_batches is not None and batch_index >= max_validation_batches:
                break
            images = images.to(
                device=runtime["energy_fn"]._device,
                dtype=runtime["runtime_dtype"],
            )
            labels = labels.to(runtime["energy_fn"]._device)
            runtime["network"].set_input(images, reset=True)
            runtime["minimizer"].compute_equilibrium()
            if not all(
                bool(torch.isfinite(layer.state).all())
                for layer in runtime["free_layers"]
            ):
                raise FloatingPointError(
                    f"Non-finite free state at validation batch {batch_index}."
                )

            runtime["cost_fn"].set_target(labels)
            costs = runtime["cost_fn"].eval()
            if not bool(torch.isfinite(costs).all()):
                raise FloatingPointError(
                    f"Non-finite validation cost at batch {batch_index}."
                )
            errors = runtime["cost_fn"].error_fn()
            scores = _score_tensor(runtime["cost_fn"], runtime["output_layer"].state)
            predictions = scores.argmax(dim=1)
            batch_size = int(images.shape[0])
            total_loss += float(costs.sum().item())
            total_correct += int((~errors).sum().item())
            seen += batch_size
            evaluated_batches += 1
            prediction_hash.update(
                predictions.detach().cpu().to(torch.int64).contiguous().numpy().tobytes()
            )
            label_hash.update(
                labels.detach().cpu().to(torch.int64).contiguous().numpy().tobytes()
            )
            for stat, layer in zip(state_stats, runtime["free_layers"]):
                _update_state_stat(stat, layer.state)
            _update_state_stat(score_stat, scores)

            append_metric(
                run_dir / "metrics.jsonl",
                {
                    "stage": "validation",
                    "batch": batch_index + 1,
                    "total_batches": total_batches,
                    "examples_seen": seen,
                    "running_loss": total_loss / seen,
                    "running_accuracy": total_correct / seen,
                    "official_test_read": False,
                },
            )
            if (batch_index + 1) % 10 == 0 or batch_index + 1 == total_batches:
                update_status_progress(
                    run_dir,
                    {
                        "stage": "validation",
                        "batch": batch_index + 1,
                        "total_batches": total_batches,
                        "examples_seen": seen,
                    },
                )

    if seen <= 0:
        raise RuntimeError("Validation loader yielded no examples.")
    return {
        "examples": seen,
        "batches": evaluated_batches,
        "loss": total_loss / seen,
        "accuracy": total_correct / seen,
        "correct": total_correct,
        "prediction_sha256": prediction_hash.hexdigest(),
        "label_sha256": label_hash.hexdigest(),
        "state_statistics": [_finish_state_stat(stat) for stat in state_stats],
        "score_statistics": _finish_state_stat(score_stat),
        "official_test_read": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-role", choices=("best", "final"), required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--arm-id", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--validation-batch-size", type=int, default=64)
    parser.add_argument("--max-validation-batches", type=int)
    parser.add_argument("--source-validation-accuracy", type=float, required=True)
    parser.add_argument("--materiality-threshold-pp", type=float, default=0.5)
    parser.add_argument("--target", default="main")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = args.config.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    run_dir = args.run_dir.expanduser().resolve()
    for path, label in (
        (config_path, "config"),
        (checkpoint_path, "checkpoint"),
        (dataset_root, "dataset root"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Expected {label} to exist: {path}")
    if args.validation_batch_size <= 0:
        raise ValueError("validation-batch-size must be positive.")
    if args.max_validation_batches is not None and args.max_validation_batches <= 0:
        raise ValueError("max-validation-batches must be positive when supplied.")
    if not math.isfinite(args.source_validation_accuracy):
        raise ValueError("source-validation-accuracy must be finite.")
    if args.materiality_threshold_pp < 0.0:
        raise ValueError("materiality-threshold-pp must be non-negative.")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {args.device!r}, but CUDA is unavailable.")
    config = trainer.load_config(config_path)
    dataset_config = config["datasets"][config["lab"]["dataset_key"]]
    manifest = {
        "study_id": args.study_id,
        "run_id": args.run_id,
        "arm_id": args.arm_id,
        "evidence_class": "ordinary_mnist_checkpoint_replay_diagnostic",
        "smoke": args.max_validation_batches is not None,
        "scientific_question": (
            "Does an unchanged legacy Conv checkpoint retain validation accuracy "
            "when its equilibrium is recomputed with the physical KCL energy?"
        ),
        "energy_contract": ENERGY_CONTRACT,
        "source": {
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "checkpoint_role": args.checkpoint_role,
            "source_validation_accuracy": float(args.source_validation_accuracy),
        },
        "dataset": {
            "key": config["lab"]["dataset_key"],
            "factory": dataset_config["factory"],
            "runtime_root": str(dataset_root),
            "split": "validation",
            "official_test_read": False,
        },
        "contract": {
            "parameters_mutated": False,
            "optimizer_steps": 0,
            "batch_state_policy": "reset_each_batch",
            "inference_iterations": config["model_base"]["num_iterations_inference"],
            "materiality_threshold_pp": float(args.materiality_threshold_pp),
        },
        "exact_command": shlex.join([sys.executable, *sys.argv]),
        "git": _git_state(),
        "runtime": runtime_context(target=args.target),
    }
    start_run(run_dir, manifest)
    try:
        runtime = _build_runtime(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            dataset_root=dataset_root,
            validation_batch_size=args.validation_batch_size,
            device=device,
        )
        replay = _evaluate(
            runtime,
            max_validation_batches=args.max_validation_batches,
            run_dir=run_dir,
        )
        delta_pp = 100.0 * (
            float(replay["accuracy"]) - float(args.source_validation_accuracy)
        )
        summary = {
            "schema_version": "conv-checkpoint-physical-kcl-replay/v1",
            "energy_contract": ENERGY_CONTRACT,
            "checkpoint_role": args.checkpoint_role,
            "source_validation_accuracy": float(args.source_validation_accuracy),
            "replay": replay,
            "delta_accuracy_pp": delta_pp,
            "absolute_delta_accuracy_pp": abs(delta_pp),
            "materiality_threshold_pp": float(args.materiality_threshold_pp),
            "material_change": abs(delta_pp) >= float(args.materiality_threshold_pp),
            "runtime_dtype": runtime["runtime_dtype_name"],
            "inference_iterations": runtime["inference_iterations"],
            "dataset_provenance": runtime["dataset_provenance"],
            "parameters_mutated": False,
            "optimizer_steps": 0,
            "official_test_read": False,
        }
        atomic_write_json(run_dir / "artifacts" / "replay_summary.json", summary)
        complete_run(
            run_dir,
            terminal_metrics={
                "validation": {
                    "accuracy": replay["accuracy"],
                    "loss": replay["loss"],
                    "examples": replay["examples"],
                },
                "source_validation_accuracy": float(args.source_validation_accuracy),
                "delta_accuracy_pp": delta_pp,
                "material_change": summary["material_change"],
            },
            completion={
                "criteria_met": args.max_validation_batches is not None
                or replay["examples"] == 5000,
                "full_validation_split": args.max_validation_batches is None,
                "parameters_mutated": False,
                "optimizer_steps": 0,
                "official_test_read": False,
            },
        )
    except BaseException as error:
        fail_run(run_dir, error=error)
        raise
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
