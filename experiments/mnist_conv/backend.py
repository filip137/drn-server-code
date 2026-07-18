"""Adapter from canonical RunSpec to the existing Conv training engine."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .protocol import ensure_run_executable
from .specs import RunSpec


@dataclass(frozen=True)
class ExecutionContext:
    dataset_root: Path
    device: str | None = None
    download: bool = False
    log_interval: int = 500
    initialization_checkpoint_path: Path | None = None

    def __init__(
        self,
        dataset_root: str | Path,
        device: str | None = None,
        download: bool = False,
        log_interval: int = 500,
        initialization_checkpoint_path: str | Path | None = None,
    ):
        if not isinstance(dataset_root, (str, Path)) or not str(dataset_root):
            raise ValueError(f"Expected dataset_root to be a non-empty path. Provided value: {dataset_root!r}.")
        if device is not None and (not isinstance(device, str) or not device):
            raise ValueError(f"Expected device to be null or a non-empty string. Provided value: {device!r}.")
        if not isinstance(download, bool):
            raise ValueError(f"Expected download to be a boolean. Provided value: {download!r}.")
        if isinstance(log_interval, bool) or not isinstance(log_interval, int) or log_interval < 0:
            raise ValueError(f"Expected log_interval to be an integer >= 0. Provided value: {log_interval!r}.")
        object.__setattr__(self, "dataset_root", Path(dataset_root).expanduser())
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "download", download)
        object.__setattr__(self, "log_interval", log_interval)
        object.__setattr__(
            self,
            "initialization_checkpoint_path",
            None if initialization_checkpoint_path is None else Path(initialization_checkpoint_path),
        )

    def with_initialization_checkpoint(self, path: str | Path | None) -> "ExecutionContext":
        return ExecutionContext(
            dataset_root=self.dataset_root,
            device=self.device,
            download=self.download,
            log_interval=self.log_interval,
            initialization_checkpoint_path=path,
        )


class TrainingBackend(Protocol):
    def __call__(
        self,
        *,
        spec: RunSpec,
        context: ExecutionContext,
        engine_config_path: Path,
        output_dir: Path,
    ) -> Any: ...


def _conv_spatial(size: int, kernel: int, stride: int, padding: int) -> int:
    return (size + 2 * padding - (kernel - 1) - 1) // stride + 1


def build_engine_config(spec: RunSpec, context: ExecutionContext) -> dict[str, Any]:
    value = spec.data
    run = value["run"]
    dataset = run["dataset"]
    architecture = run["architecture"]
    model = run["model"]
    solver = run["solver"]
    training = run["training"]
    optimizer = training["optimizer"]

    height = width = 28
    layer_shapes: list[list[int]] = [[2, height, width]]
    conv_pipeline: list[dict[str, Any]] = []
    for channels, kernel, stride, padding in zip(
        architecture["channels"],
        architecture["kernel_sizes"],
        architecture["strides"],
        architecture["paddings"],
    ):
        height = _conv_spatial(height, kernel, stride, padding)
        width = _conv_spatial(width, kernel, stride, padding)
        if height <= 0 or width <= 0:
            raise ValueError(
                "Expected convolution geometry to retain positive spatial dimensions. "
                f"Provided height={height}, width={width}."
            )
        layer_shapes.append([channels, height, width])
        conv_pipeline.append(
            {
                "kernel": [kernel, kernel],
                "stride": stride,
                "padding": padding,
                "mode": "convolution",
            }
        )
    layer_shapes.append([architecture["output_dim"]])

    affine = dataset["affine"]
    dataset_params: dict[str, Any] = {
        "name": "mnist",
        "batch_size": dataset["batch_size"],
        "root": str(context.dataset_root),
        "train": True,
        "download": context.download,
        "normalize": True,
        "normalize_mean": dataset["normalization"]["mean"],
        "normalize_std": dataset["normalization"]["std"],
        "normalize_scale": dataset["normalization"]["scale"],
    }
    if affine["enabled"]:
        dataset_params["affine_config"] = dict(affine)
        dataset_factory = "labs.datasets.AffineMnistDataset"
    else:
        dataset_factory = "labs.datasets.MnistDataset"

    model_key = "mnist_resistive_conv_v1"
    return {
        "lab": {
            "model_key": model_key,
            "dataset_key": "mnist",
            "epochs": training["epochs"],
            "plots_dir": "plots/mnist_conv",
        },
        "input_mode": "train",
        "training_algorithm": "BP",
        "seed": value["seed"],
        "batch_state_policy": training["batch_state_policy"],
        "lr": list(training["learning_rate"]),
        "beta": training["beta"],
        "lr_decay": training["lr_decay"],
        "optimizer": {
            "name": optimizer["name"],
            "learning_rate": list(training["learning_rate"]),
            "lr_decay": training["lr_decay"],
            "momentum": optimizer["momentum"],
            "weight_decay": optimizer["weight_decay"],
        },
        "log_interval": context.log_interval,
        "max_batches": dataset["max_batches"],
        "max_test_batches": dataset["max_test_batches"],
        "datasets": {
            "mnist": {
                "factory": dataset_factory,
                "params": dataset_params,
            }
        },
        "model_base": {
            "weight_min": model["weight_min"],
            "weight_max": model["weight_max"],
            "weight_init_mode": model["weight_init_mode"],
            "input_gain": model["input_gain"],
            "voltage_amp": model["voltage_amp"],
            "current_amp": model["current_amp"],
            "trainable_amplification": model["trainable_parameters"]["amplification"],
            "amplification_min": model["amplification_min"],
            "amplification_max": model["amplification_max"],
            "non_linearity": model["non_linearity"],
            "quadratic_diode_param": dict(model["quadratic_diode_param"]),
            "exponential_diode_param": dict(model["exponential_diode_param"]),
            "hard_sigmoid_param": dict(model["hard_sigmoid_param"]),
            "num_iterations_inference": solver["inference_iterations"],
            "num_iterations_training": solver["training_iterations"],
            "minimizer": json.loads(json.dumps(solver["minimizer"])),
        },
        "model_overrides": {
            model_key: {
                "layer_shapes": layer_shapes,
                "conv_pipeline": conv_pipeline,
                "weight_gains": list(model["weight_gains"]),
            }
        },
        "energy_minimizer": {"mode": solver["energy_mode"]},
    }


def train_with_mnist_backend(
    *,
    spec: RunSpec,
    context: ExecutionContext,
    engine_config_path: Path,
    output_dir: Path,
) -> Any:
    """Run ``labs.mnist_train.train_mnist_conv`` lazily.

    Keeping this import inside the backend lets planners, collectors, and unit
    tests run without importing torch/torchvision or initializing CUDA.
    """

    ensure_run_executable(spec)
    from labs.mnist_train import train_mnist_conv

    run = spec.data["run"]
    dataset = run["dataset"]
    training = run["training"]
    checkpoint = run["initialization"]["checkpoint"]
    if checkpoint is not None and context.initialization_checkpoint_path is None:
        raise ValueError(
            "Expected the runner to resolve initialization checkpoint relative to results_root. "
            f"Provided checkpoint record: {checkpoint!r}."
        )
    pruning = training["pruning"]
    prune_state = {"pruned": False}

    def epoch_callback(epoch_info: dict[str, Any], history: dict[str, Any]) -> bool:
        if not pruning["enabled"] or int(epoch_info["epoch"]) < pruning["after_epoch"]:
            return False
        finite = [
            float(item)
            for item in history.get("test_accuracy", [])
            if item is not None
        ]
        if finite and max(finite) < pruning["min_best_test_accuracy"]:
            prune_state.update({"pruned": True, "epoch": int(epoch_info["epoch"]), "best_test_accuracy": max(finite)})
            return True
        return False

    result = train_mnist_conv(
        config_path=engine_config_path,
        epochs=training["epochs"],
        lr=list(training["learning_rate"]),
        beta=training["beta"],
        log_interval=context.log_interval,
        max_batches=dataset["max_batches"],
        max_test_batches=dataset["max_test_batches"],
        device=context.device,
        sanity_check=False,
        dataset_key="mnist",
        output_dir=output_dir,
        model_key="mnist_resistive_conv_v1",
        training_algorithm="BP",
        seed=spec.data["seed"],
        lr_decay=training["lr_decay"],
        optimizer_name=training["optimizer"]["name"],
        momentum=training["optimizer"]["momentum"],
        weight_decay=training["optimizer"]["weight_decay"],
        init_checkpoint_path=(str(context.initialization_checkpoint_path) if checkpoint else None),
        batch_state_policy=training["batch_state_policy"],
        epoch_callback=epoch_callback if pruning["enabled"] else None,
    )
    if prune_state["pruned"]:
        return {"status": "pruned", **prune_state}
    return result
