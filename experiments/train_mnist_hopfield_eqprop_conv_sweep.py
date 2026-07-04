#!/usr/bin/env python3
"""Train no-pooling Conv MNIST Hopfield networks with EqProp."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.function.cost import SquaredError  # noqa: E402
from model.function.network import Network  # noqa: E402
from model.hopfield.minimizer import FixedPointMinimizer  # noqa: E402
from model.hopfield.network import FlexibleConvHopfieldEnergy  # noqa: E402
from model.variable.layer import Layer  # noqa: E402
from model.variable.parameter import Bias, ConvWeight, DenseWeight  # noqa: E402
from training.sgd import AugmentedFunction, EquilibriumProp  # noqa: E402


DEFAULT_CHANNELS = [64, 128, 256]
DEFAULT_BASE_LRS = [5.0e-5, 1.0e-5, 8.0e-6]
DATASETS = {
    "mnist": datasets.MNIST,
    "fashion_mnist": datasets.FashionMNIST,
}
INPUT_PREPROCESSING = {
    "identity": ((0.0,), (1.0,)),
    "centered": ((0.5,), (0.5,)),
}
AFFINE_PRESETS = {
    "none": {
        "degrees": 0.0,
        "translate": [0.0, 0.0],
        "scale": [1.0, 1.0],
        "shear": 0.0,
    },
    "medium": {
        "degrees": 25.0,
        "translate": [0.20, 0.20],
        "scale": [0.80, 1.20],
        "shear": 0.0,
    },
    "mnist_affine": {
        "degrees": 60.0,
        "translate": [0.15, 0.15],
        "scale": [0.80, 1.20],
        "shear": 15.0,
    },
}
SUMMARY_COLUMNS = [
    "conv_depth",
    "seed",
    "lr_multiplier",
    "best_test_accuracy",
    "final_test_accuracy",
    "best_epoch",
    "run_dir",
]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _reset_name_counters() -> None:
    Layer._counter = 0
    Bias._counter = 0
    ConvWeight._counter = 0
    DenseWeight._counter = 0


def _label_float(value: float) -> str:
    text = f"{float(value):.12g}"
    return text.replace("-", "m").replace(".", "p")


def _conv_spatial(size: int, kernel: int, stride: int, padding: int) -> int:
    return (size + 2 * padding - (kernel - 1) - 1) // stride + 1


def _default_padding(conv_depth: int) -> int:
    return 1 if int(conv_depth) == 3 else 0


def _strides_from_args(args: argparse.Namespace) -> list[int]:
    if args.strides:
        requested = [int(value) for value in args.strides]
        if len(requested) < int(args.conv_depth):
            raise ValueError(
                f"Expected at least {args.conv_depth} --strides values, got {requested}."
            )
        effective = []
        for value in requested[: int(args.conv_depth)]:
            if value < 0:
                raise ValueError(f"Expected non-negative --strides values, got {requested}.")
            # In experiment labels, 0 means no downsampling in that layer.
            effective.append(1 if value == 0 else value)
        return effective
    if int(args.stride) <= 0:
        raise ValueError(f"Expected --stride > 0, got {args.stride}.")
    return [int(args.stride)] * int(args.conv_depth)


def architecture_from_args(args: argparse.Namespace) -> tuple[list[tuple[int, ...]], list[dict]]:
    channels = [int(value) for value in args.conv_channels]
    if args.conv_depth < 1:
        raise ValueError(f"Expected --conv-depth >= 1, got {args.conv_depth}.")
    if len(channels) < args.conv_depth:
        raise ValueError(
            f"Expected at least {args.conv_depth} --conv-channels values, got {channels}."
        )
    padding = _default_padding(args.conv_depth) if args.padding is None else int(args.padding)
    strides = _strides_from_args(args)
    height = 28
    width = 28
    layer_shapes: list[tuple[int, ...]] = [(1, height, width)]
    conv_pipeline: list[dict] = []
    for index in range(args.conv_depth):
        stride = strides[index]
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
                "mode": "convolution",
                "kernel": [args.kernel_size, args.kernel_size],
                "stride": stride,
                "padding": padding,
            }
        )
    layer_shapes.append((10,))
    return layer_shapes, conv_pipeline


def default_weight_gains(conv_depth: int) -> list[float]:
    return [0.6] * int(conv_depth) + [1.5]


def learning_rates_from_args(args: argparse.Namespace) -> list[float]:
    expected = 2 * (args.conv_depth + 1)
    if args.learning_rates:
        values = [float(value) for value in args.learning_rates]
        if len(values) == 1:
            return values * expected
        if len(values) != expected:
            raise ValueError(
                f"Expected --learning-rates to have 1 or {expected} values, got {len(values)}."
            )
        return values

    edge_lrs = []
    for index in range(args.conv_depth):
        source_index = min(index, len(DEFAULT_BASE_LRS) - 1)
        edge_lrs.append(DEFAULT_BASE_LRS[source_index])
    edge_lrs.append(DEFAULT_BASE_LRS[-1])
    edge_lrs = [float(args.lr_multiplier) * value for value in edge_lrs]
    return edge_lrs + edge_lrs


def run_dir_from_args(args: argparse.Namespace) -> Path:
    lr_label = _label_float(args.lr_multiplier)
    return (
        Path(args.output_root).expanduser().resolve()
        / args.run_group
        / f"conv{args.conv_depth}"
        / f"lr_mult_{lr_label}"
        / f"seed_{args.seed}"
    )


def _make_loader(dataset, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=0,
        generator=generator if shuffle else None,
    )


class DeterministicAffineDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        *,
        transform,
        affine_config: dict,
        split: str,
    ) -> None:
        self.dataset = dataset
        self.transform = transform
        self.affine_config = affine_config
        self.split_offset = 0 if split == "train" else 10_000_000

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, label = self.dataset[index]
        if bool(self.affine_config.get("enabled", False)):
            image = self._apply_affine(image, index)
        return self.transform(image), label

    def _apply_affine(self, image, index: int):
        seed = int(self.affine_config.get("seed", 0))
        rng = random.Random(seed + self.split_offset + 104_729 * int(index))
        degrees = float(self.affine_config["degrees"])
        translate_frac = [float(value) for value in self.affine_config["translate"]]
        scale_range = [float(value) for value in self.affine_config["scale"]]
        shear = float(self.affine_config.get("shear", 0.0))

        angle = rng.uniform(-degrees, degrees) if degrees else 0.0
        scale = rng.uniform(scale_range[0], scale_range[1])
        shear_x = rng.uniform(-shear, shear) if shear else 0.0
        width, height = image.size
        max_dx = translate_frac[0] * width
        max_dy = translate_frac[1] * height
        translate = (
            int(round(rng.uniform(-max_dx, max_dx))),
            int(round(rng.uniform(-max_dy, max_dy))),
        )
        return TF.affine(
            image,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=[shear_x, 0.0],
            interpolation=InterpolationMode.BILINEAR,
            fill=0,
        )


def affine_config_from_args(args: argparse.Namespace) -> dict:
    config = dict(AFFINE_PRESETS[args.affine_preset])
    if args.affine_degrees is not None:
        config["degrees"] = float(args.affine_degrees)
    if args.affine_translate is not None:
        translate = [float(value) for value in args.affine_translate]
        if len(translate) == 1:
            translate = [translate[0], translate[0]]
        if len(translate) != 2:
            raise ValueError(
                f"Expected --affine-translate to have 1 or 2 values, got {translate}."
            )
        config["translate"] = translate
    if args.affine_scale is not None:
        scale = [float(value) for value in args.affine_scale]
        if len(scale) != 2:
            raise ValueError(f"Expected --affine-scale to have 2 values, got {scale}.")
        config["scale"] = scale
    if args.affine_shear is not None:
        config["shear"] = float(args.affine_shear)

    scale_min, scale_max = [float(value) for value in config["scale"]]
    if scale_min <= 0.0 or scale_max <= 0.0 or scale_min > scale_max:
        raise ValueError(f"Expected positive ordered --affine-scale, got {config['scale']}.")
    translate = [float(value) for value in config["translate"]]
    if any(value < 0.0 for value in translate):
        raise ValueError(f"Expected non-negative --affine-translate, got {translate}.")

    enabled = (
        abs(float(config["degrees"])) > 0.0
        or any(abs(value) > 0.0 for value in translate)
        or abs(scale_min - 1.0) > 0.0
        or abs(scale_max - 1.0) > 0.0
        or abs(float(config.get("shear", 0.0))) > 0.0
    )
    config["enabled"] = enabled
    config["preset"] = args.affine_preset
    config["seed"] = int(args.affine_seed)
    return config


def build_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    if args.synthetic_samples:
        generator = torch.Generator().manual_seed(int(args.seed))
        train_count = int(args.synthetic_samples)
        test_count = max(int(args.synthetic_test_samples), int(args.batch_size))
        train_x = torch.rand(train_count, 1, 28, 28, generator=generator)
        train_y = torch.randint(0, 10, (train_count,), generator=generator)
        test_x = torch.rand(test_count, 1, 28, 28, generator=generator)
        test_y = torch.randint(0, 10, (test_count,), generator=generator)
        if args.input_preprocessing == "centered":
            train_x = 2.0 * train_x - 1.0
            test_x = 2.0 * test_x - 1.0
        return (
            _make_loader(TensorDataset(train_x, train_y), args.batch_size, True, args.seed),
            _make_loader(TensorDataset(test_x, test_y), args.test_batch_size, False, args.seed),
        )

    mean, std = INPUT_PREPROCESSING[args.input_preprocessing]
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    affine_config = affine_config_from_args(args)
    dataset_root = os.path.expanduser(str(args.dataset_root))
    download = not args.no_download
    dataset_cls = DATASETS[args.dataset]
    if affine_config["enabled"]:
        train_base = dataset_cls(dataset_root, train=True, transform=None, download=download)
        test_base = dataset_cls(dataset_root, train=False, transform=None, download=download)
        train_ds = DeterministicAffineDataset(
            train_base,
            transform=transform,
            affine_config=affine_config,
            split="train",
        )
        test_ds = DeterministicAffineDataset(
            test_base,
            transform=transform,
            affine_config=affine_config,
            split="test",
        )
    else:
        train_ds = dataset_cls(dataset_root, train=True, transform=transform, download=download)
        test_ds = dataset_cls(dataset_root, train=False, transform=transform, download=download)
    return (
        _make_loader(train_ds, args.batch_size, True, args.seed),
        _make_loader(test_ds, args.test_batch_size, False, args.seed),
    )


def build_model(args: argparse.Namespace, device: torch.device) -> tuple[FlexibleConvHopfieldEnergy, dict]:
    _reset_name_counters()
    layer_shapes, conv_pipeline = architecture_from_args(args)
    weight_gains = (
        [float(value) for value in args.weight_gains]
        if args.weight_gains
        else default_weight_gains(args.conv_depth)
    )
    if len(weight_gains) == 1:
        weight_gains = weight_gains * (args.conv_depth + 1)
    if len(weight_gains) != args.conv_depth + 1:
        raise ValueError(
            f"Expected --weight-gains to have 1 or {args.conv_depth + 1} values, "
            f"got {len(weight_gains)}."
        )
    energy_fn = FlexibleConvHopfieldEnergy(
        layer_shapes=layer_shapes,
        weight_gains=weight_gains,
        conv_pipeline=conv_pipeline,
        activation=args.activation,
        weight_init_mode=args.weight_init_mode,
    )
    energy_fn.set_device(device)
    model_config = {
        "layer_shapes": [list(shape) for shape in layer_shapes],
        "conv_pipeline": conv_pipeline,
        "weight_gains": weight_gains,
        "activation": args.activation,
        "weight_init_mode": args.weight_init_mode,
    }
    return energy_fn, model_config


def load_init_checkpoint(energy_fn: FlexibleConvHopfieldEnergy, checkpoint_path: str) -> Path:
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Expected --init-checkpoint to exist, got {path}.")
    try:
        checkpoint = torch.load(
            path,
            map_location=torch.device(energy_fn._device),
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(path, map_location=torch.device(energy_fn._device))
    params = energy_fn.params()
    if len(checkpoint) != len(params):
        raise ValueError(
            f"Expected checkpoint with {len(params)} tensors, got {len(checkpoint)} at {path}."
        )
    for index, (param, state) in enumerate(zip(params, checkpoint)):
        if tuple(state.shape) != tuple(param.state.shape):
            raise ValueError(
                "Checkpoint tensor shape mismatch at parameter "
                f"{index}: expected {tuple(param.state.shape)}, got {tuple(state.shape)}."
            )
        param.state = state.detach().clone().to(device=param.state.device, dtype=param.state.dtype)
    return path


@torch.no_grad()
def evaluate(
    network: Network,
    cost_fn: SquaredError,
    minimizer: FixedPointMinimizer,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None,
) -> dict:
    running_loss = 0.0
    running_correct = 0
    seen = 0
    for batch_index, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_index >= int(max_batches):
            break
        images = images.to(device)
        labels = labels.to(device)
        network.set_input(images, reset=True)
        minimizer.compute_equilibrium()
        cost_fn.set_target(labels)
        loss = float(cost_fn.eval().mean().item())
        errors = cost_fn.error_fn()
        correct = int((~errors).sum().item())
        running_loss += loss * images.size(0)
        running_correct += correct
        seen += int(images.size(0))
    return {
        "loss": running_loss / seen if seen else math.nan,
        "accuracy": running_correct / seen if seen else math.nan,
        "num_samples": seen,
    }


def train(args: argparse.Namespace) -> dict:
    _set_seed(int(args.seed))
    device = torch.device(args.device)
    run_dir = run_dir_from_args(args)
    run_dir.mkdir(parents=True, exist_ok=True)

    energy_fn, model_config = build_model(args, device)
    init_checkpoint_path = (
        load_init_checkpoint(energy_fn, args.init_checkpoint)
        if args.init_checkpoint
        else None
    )
    network = Network(energy_fn)
    output_layer = energy_fn.layers()[-1]
    cost_fn = SquaredError(output_layer)
    free_layers = network.free_layers()
    augmented_fn = AugmentedFunction(energy_fn, cost_fn)
    minimizer_inference = FixedPointMinimizer(
        energy_fn,
        free_layers,
        num_iterations=args.t1,
        mode=args.mode,
    )
    minimizer_training = FixedPointMinimizer(
        augmented_fn,
        free_layers,
        num_iterations=args.t2,
        mode=args.mode,
    )
    estimator = EquilibriumProp(
        energy_fn.params(),
        free_layers,
        augmented_fn,
        cost_fn,
        minimizer_training,
        variant="centered",
        nudging=args.beta,
    )

    learning_rates = learning_rates_from_args(args)
    params = energy_fn.params()
    if len(learning_rates) != len(params):
        raise ValueError(
            f"Expected {len(params)} learning rates for Hopfield params, got {len(learning_rates)}."
        )
    optimizer = torch.optim.Adam(
        [{"params": param.state, "lr": lr} for param, lr in zip(params, learning_rates)]
    )
    train_loader, test_loader = build_loaders(args)

    affine_config = affine_config_from_args(args)
    source_config = {
        "script": str(Path(__file__).resolve()),
        "run_group": args.run_group,
        "conv_depth": args.conv_depth,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "test_batch_size": args.test_batch_size,
        "t1": args.t1,
        "t2": args.t2,
        "beta": args.beta,
        "mode": args.mode,
        "lr_multiplier": args.lr_multiplier,
        "learning_rates": learning_rates,
        "optimizer": "Adam",
        "dataset": args.dataset,
        "dataset_root": args.dataset_root,
        "input_preprocessing": args.input_preprocessing,
        "affine_transform": affine_config,
        "synthetic_samples": args.synthetic_samples,
        "init_checkpoint": str(init_checkpoint_path) if init_checkpoint_path else None,
        "requested_strides": [int(value) for value in args.strides] if args.strides else None,
        "effective_strides": [int(conf["stride"]) for conf in model_config["conv_pipeline"]],
        "model": model_config,
    }
    (run_dir / "source_config.json").write_text(json.dumps(source_config, indent=2))
    (run_dir / "config.json").write_text(json.dumps(source_config, indent=2))

    history = {
        "loss_train": [],
        "accuracy_train": [],
        "loss_test": [],
        "accuracy_test": [],
        "learning_rate": [],
    }
    best_test_accuracy = float("-inf")
    best_epoch = None
    best_model_path = run_dir / "best_model.pt"

    for epoch in range(int(args.epochs)):
        running_loss = 0.0
        running_correct = 0
        seen = 0
        total_batches = len(train_loader)
        for batch_index, (images, labels) in enumerate(train_loader):
            if args.max_batches is not None and batch_index >= int(args.max_batches):
                break
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            network.set_input(images, reset=True)
            minimizer_inference.compute_equilibrium()
            cost_fn.set_target(labels)
            loss = float(cost_fn.eval().mean().item())
            errors = cost_fn.error_fn()
            correct = int((~errors).sum().item())
            running_loss += loss * images.size(0)
            running_correct += correct
            seen += int(images.size(0))

            grads = estimator.compute_gradient()
            for param, grad in zip(params, grads):
                param.state.grad = grad
            optimizer.step()
            for param in params:
                param.state.grad = None

            if _should_log(batch_index, total_batches, args.log_interval):
                train_acc = running_correct / seen if seen else math.nan
                train_loss = running_loss / seen if seen else math.nan
                print(
                    f"Epoch {epoch + 1}/{args.epochs} | Batch {batch_index + 1}/{total_batches} "
                    f"| train loss={train_loss:.4f} acc={train_acc * 100:.2f}%",
                    flush=True,
                )

        train_loss = running_loss / seen if seen else math.nan
        train_acc = running_correct / seen if seen else math.nan
        test_result = evaluate(
            network,
            cost_fn,
            minimizer_inference,
            test_loader,
            device,
            args.max_test_batches,
        )
        history["loss_train"].append(train_loss)
        history["accuracy_train"].append(train_acc)
        history["loss_test"].append(test_result["loss"])
        history["accuracy_test"].append(test_result["accuracy"])
        history["learning_rate"].append([float(group["lr"]) for group in optimizer.param_groups])
        print(
            f"[epoch] {epoch + 1}/{args.epochs} train_acc={train_acc * 100:.2f}% "
            f"test_acc={test_result['accuracy'] * 100:.2f}%",
            flush=True,
        )
        if test_result["accuracy"] > best_test_accuracy:
            best_test_accuracy = float(test_result["accuracy"])
            best_epoch = epoch + 1
            energy_fn.save(best_model_path)

    final_model_path = run_dir / "final_model.pt"
    energy_fn.save(final_model_path)
    if not best_model_path.exists():
        energy_fn.save(best_model_path)
        best_test_accuracy = history["accuracy_test"][-1] if history["accuracy_test"] else math.nan
        best_epoch = len(history["accuracy_test"]) or None

    np.save(run_dir / "loss_train.npy", np.asarray(history["loss_train"], dtype=np.float64))
    np.save(run_dir / "accuracy_train.npy", np.asarray(history["accuracy_train"], dtype=np.float64))
    np.save(run_dir / "loss_test.npy", np.asarray(history["loss_test"], dtype=np.float64))
    np.save(run_dir / "accuracy_test.npy", np.asarray(history["accuracy_test"], dtype=np.float64))

    metrics = {
        "run_dir": str(run_dir),
        "conv_depth": args.conv_depth,
        "seed": args.seed,
        "run_group": args.run_group,
        "optimizer": "Adam",
        "lr_multiplier": args.lr_multiplier,
        "learning_rates": learning_rates,
        "activation": args.activation,
        "input_preprocessing": args.input_preprocessing,
        "affine_transform": source_config["affine_transform"],
        "init_checkpoint": source_config["init_checkpoint"],
        "requested_strides": source_config["requested_strides"],
        "effective_strides": source_config["effective_strides"],
        "best_epoch": best_epoch,
        "best_test_accuracy": best_test_accuracy,
        "final_test_accuracy": history["accuracy_test"][-1] if history["accuracy_test"] else math.nan,
        "final_train_accuracy": history["accuracy_train"][-1] if history["accuracy_train"] else math.nan,
        "final_test_loss": history["loss_test"][-1] if history["loss_test"] else math.nan,
        "final_train_loss": history["loss_train"][-1] if history["loss_train"] else math.nan,
        "checkpoint_path": str(final_model_path),
        "best_checkpoint_path": str(best_model_path),
        "source_config_path": str(run_dir / "source_config.json"),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (run_dir / "run_metadata.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def _should_log(batch_index: int, total_batches: int, log_interval: int) -> bool:
    batch_num = batch_index + 1
    if batch_num == 1 or batch_num == total_batches:
        return True
    return int(log_interval) > 0 and batch_num % int(log_interval) == 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="mnist")
    parser.add_argument("--dataset-root", default=str(REPO_ROOT / "data"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--run-group", default="lr_screen")
    parser.add_argument("--conv-depth", type=int, required=True)
    parser.add_argument("--conv-channels", type=int, nargs="+", default=DEFAULT_CHANNELS)
    parser.add_argument(
        "--activation",
        choices=("hard-sigmoid", "sigmoid", "tanh", "silu"),
        default="hard-sigmoid",
        help="Hidden-layer activation. Output layer remains linear.",
    )
    parser.add_argument(
        "--input-preprocessing",
        choices=sorted(INPUT_PREPROCESSING),
        default="identity",
        help="identity keeps MNIST pixels in [0,1]; centered maps pixels to [-1,1].",
    )
    parser.add_argument(
        "--affine-preset",
        choices=sorted(AFFINE_PRESETS),
        default="none",
        help="Deterministic per-sample affine transform preset.",
    )
    parser.add_argument("--affine-degrees", type=float, default=None)
    parser.add_argument("--affine-translate", type=float, nargs="+", default=None)
    parser.add_argument("--affine-scale", type=float, nargs=2, default=None)
    parser.add_argument("--affine-shear", type=float, default=None)
    parser.add_argument(
        "--affine-seed",
        type=int,
        default=1729,
        help="Dataset-transform seed shared across model seeds.",
    )
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument(
        "--strides",
        type=int,
        nargs="+",
        default=None,
        help="Optional per-convolution strides. Value 0 means no downsampling and is run as stride 1.",
    )
    parser.add_argument("--padding", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-batch-size", type=int, default=200)
    parser.add_argument("--t1", type=int, default=200)
    parser.add_argument("--t2", type=int, default=10)
    parser.add_argument("--beta", type=float, default=0.4)
    parser.add_argument("--mode", choices=("forward", "backward", "synchronous", "asynchronous"), default="asynchronous")
    parser.add_argument("--lr-multiplier", type=float, default=1.0)
    parser.add_argument("--learning-rates", type=float, nargs="+", default=None)
    parser.add_argument("--weight-gains", type=float, nargs="+", default=None)
    parser.add_argument("--weight-init-mode", default="kaiming_uniform")
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help="Optional Hopfield final_model.pt/best_model.pt to initialize parameters before training.",
    )
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--max-test-batches", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--synthetic-samples", type=int, default=0)
    parser.add_argument("--synthetic-test-samples", type=int, default=0)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()
    architecture_from_args(args)
    learning_rates_from_args(args)
    affine_config_from_args(args)
    return args


def main() -> None:
    args = parse_args()
    metrics = train(args)
    print("[done] " + json.dumps(metrics, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
