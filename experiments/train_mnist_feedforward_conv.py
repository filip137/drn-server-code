#!/usr/bin/env python3
"""Train ordinary feed-forward ConvNets on MNIST-style datasets with BP."""

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
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import datasets, transforms


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "mnist_feedforward_conv"
DEFAULT_CHANNELS = [32, 64, 128]
DATASET_SPECS = {
    "mnist": {
        "cls": datasets.MNIST,
        "input_channels": 1,
        "image_size": 28,
        "standard": ((0.1307,), (0.3081,)),
    },
    "mnist_rotated": {
        "cls": datasets.MNIST,
        "input_channels": 1,
        "image_size": 28,
        "standard": ((0.1307,), (0.3081,)),
    },
    "mnist_affine": {
        "cls": datasets.MNIST,
        "input_channels": 1,
        "image_size": 28,
        "standard": ((0.1307,), (0.3081,)),
    },
    "fashion_mnist": {
        "cls": datasets.FashionMNIST,
        "input_channels": 1,
        "image_size": 28,
        "standard": ((0.2860,), (0.3530,)),
    },
    "cifar10_gray": {
        "cls": datasets.CIFAR10,
        "input_channels": 1,
        "image_size": 32,
        "standard": ((0.4734,), (0.2516,)),
        "grayscale": True,
    },
    "cifar10_rgb": {
        "cls": datasets.CIFAR10,
        "input_channels": 3,
        "image_size": 32,
        "standard": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    },
}
DATASETS = {name: spec["cls"] for name, spec in DATASET_SPECS.items()}
STANDARD_NORMALIZATION = {
    name: spec["standard"] for name, spec in DATASET_SPECS.items()
}
INPUT_PREPROCESSING = {
    "identity": ((0.0,), (1.0,)),
    "centered": ((0.5,), (0.5,)),
}


class Clamp01(nn.Module):
    """Hard-sigmoid used in the DRN/Hopfield code: clamp pre-activations to [0, 1]."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.clamp(0.0, 1.0)


def _splitmix64(value: int) -> int:
    value = (int(value) + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
    value = (value ^ (value >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
    return value ^ (value >> 31)


def _deterministic_unit(*, index: int, seed: int, split_offset: int, salt: int) -> float:
    hashed = _splitmix64(
        int(seed)
        + int(split_offset)
        + int(index) * 0x9E3779B97F4A7C15
        + int(salt) * 0xBF58476D1CE4E5B9
    )
    return (hashed >> 11) * (1.0 / (1 << 53))


def _symmetric_deterministic_value(
    *,
    index: int,
    magnitude: float,
    seed: int,
    split_offset: int,
    salt: int,
) -> float:
    if float(magnitude) <= 0.0:
        return 0.0
    unit = _deterministic_unit(
        index=index,
        seed=seed,
        split_offset=split_offset,
        salt=salt,
    )
    return (2.0 * unit - 1.0) * float(magnitude)


def _deterministic_scale(
    *,
    index: int,
    minimum: float,
    maximum: float,
    seed: int,
    split_offset: int,
) -> float:
    minimum = float(minimum)
    maximum = float(maximum)
    if minimum == maximum:
        return minimum
    unit = _deterministic_unit(
        index=index,
        seed=seed,
        split_offset=split_offset,
        salt=3,
    )
    return minimum + unit * (maximum - minimum)


class DeterministicAffineDataset(Dataset):
    """Dataset wrapper with one fixed affine transform per sample index."""

    def __init__(
        self,
        base_dataset: Dataset,
        *,
        transform,
        rotation_degrees: float,
        translation_fraction: float,
        scale_min: float,
        scale_max: float,
        shear_degrees: float,
        seed: int,
        split_offset: int,
    ) -> None:
        self.base_dataset = base_dataset
        self.transform = transform
        self.rotation_degrees = float(rotation_degrees)
        self.translation_fraction = float(translation_fraction)
        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)
        self.shear_degrees = float(shear_degrees)
        self.seed = int(seed)
        self.split_offset = int(split_offset)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        image, label = self.base_dataset[index]
        width, height = image.size
        angle = _symmetric_deterministic_value(
            index=int(index),
            magnitude=self.rotation_degrees,
            seed=self.seed,
            split_offset=self.split_offset,
            salt=0,
        )
        translate_x = int(
            round(
                _symmetric_deterministic_value(
                    index=int(index),
                    magnitude=self.translation_fraction * float(width),
                    seed=self.seed,
                    split_offset=self.split_offset,
                    salt=1,
                )
            )
        )
        translate_y = int(
            round(
                _symmetric_deterministic_value(
                    index=int(index),
                    magnitude=self.translation_fraction * float(height),
                    seed=self.seed,
                    split_offset=self.split_offset,
                    salt=2,
                )
            )
        )
        scale = _deterministic_scale(
            index=int(index),
            minimum=self.scale_min,
            maximum=self.scale_max,
            seed=self.seed,
            split_offset=self.split_offset,
        )
        shear_x = _symmetric_deterministic_value(
            index=int(index),
            magnitude=self.shear_degrees,
            seed=self.seed,
            split_offset=self.split_offset,
            salt=4,
        )
        image = transforms.functional.affine(
            image,
            angle=angle,
            translate=[translate_x, translate_y],
            scale=scale,
            shear=[shear_x, 0.0],
            interpolation=transforms.InterpolationMode.BILINEAR,
            fill=0,
        )
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def _deterministic_rotation_angle(
    *,
    index: int,
    degrees: float,
    seed: int,
    split_offset: int,
) -> float:
    if float(degrees) <= 0.0:
        return 0.0
    hashed = _splitmix64(int(seed) + int(split_offset) + int(index))
    unit = (hashed >> 11) * (1.0 / (1 << 53))
    return (2.0 * unit - 1.0) * float(degrees)


class DeterministicRotatedDataset(Dataset):
    """MNIST with one fixed random rotation per sample index."""

    def __init__(
        self,
        base_dataset: Dataset,
        *,
        transform,
        degrees: float,
        seed: int,
        split_offset: int,
    ) -> None:
        self.base_dataset = base_dataset
        self.transform = transform
        self.degrees = float(degrees)
        self.seed = int(seed)
        self.split_offset = int(split_offset)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        image, label = self.base_dataset[index]
        angle = _deterministic_rotation_angle(
            index=int(index),
            degrees=self.degrees,
            seed=self.seed,
            split_offset=self.split_offset,
        )
        image = transforms.functional.rotate(
            image,
            angle=angle,
            interpolation=transforms.InterpolationMode.BILINEAR,
            fill=0,
        )
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _label_float(value: float) -> str:
    text = f"{float(value):.12g}"
    return text.replace("-", "m").replace(".", "p")


def _activation_label(value: str) -> str:
    return str(value).replace("-", "_")


def _conv_spatial(size: int, kernel: int, stride: int, padding: int) -> int:
    return (size + 2 * padding - (kernel - 1) - 1) // stride + 1


def _expanded_int_sequence(
    values: list[int] | None,
    scalar: int,
    depth: int,
    name: str,
) -> list[int]:
    if values is None:
        return [int(scalar)] * int(depth)
    if len(values) == 1:
        return [int(values[0])] * int(depth)
    if len(values) < int(depth):
        raise ValueError(
            f"Expected {name} to contain 1 value or at least {depth} values, got {values}."
        )
    return [int(value) for value in values[: int(depth)]]


def architecture_from_args(args: argparse.Namespace) -> tuple[list[tuple[int, ...]], list[dict]]:
    dataset_spec = DATASET_SPECS[args.dataset]
    channels = [int(value) for value in args.conv_channels]
    if int(args.conv_depth) < 1:
        raise ValueError(f"Expected --conv-depth >= 1, got {args.conv_depth}.")
    if len(channels) < int(args.conv_depth):
        raise ValueError(
            f"Expected at least {args.conv_depth} --conv-channels values, got {channels}."
        )

    strides = _expanded_int_sequence(args.strides, args.stride, args.conv_depth, "--strides")
    paddings = _expanded_int_sequence(args.paddings, args.padding, args.conv_depth, "--paddings")
    if any(value <= 0 for value in strides):
        raise ValueError(f"Expected all strides to be positive, got {strides}.")
    if any(value < 0 for value in paddings):
        raise ValueError(f"Expected all paddings to be non-negative, got {paddings}.")

    height = int(dataset_spec["image_size"])
    width = int(dataset_spec["image_size"])
    layer_shapes: list[tuple[int, ...]] = [
        (int(dataset_spec["input_channels"]), height, width)
    ]
    conv_pipeline: list[dict] = []
    for index in range(int(args.conv_depth)):
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
                "kernel": [int(args.kernel_size), int(args.kernel_size)],
                "stride": int(stride),
                "padding": int(padding),
                "out_channels": int(channels[index]),
            }
        )
    layer_shapes.append((int(args.num_classes),))
    return layer_shapes, conv_pipeline


def activation_module(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "hard-sigmoid":
        return Clamp01()
    raise ValueError(f"Expected activation 'relu' or 'hard-sigmoid', got {name!r}.")


class FeedForwardConvNet(nn.Module):
    def __init__(
        self,
        *,
        layer_shapes: list[tuple[int, ...]],
        conv_pipeline: list[dict],
        activation: str,
        num_classes: int,
    ) -> None:
        super().__init__()
        modules: list[nn.Module] = []
        in_channels = int(layer_shapes[0][0])
        for conf in conv_pipeline:
            out_channels = int(conf["out_channels"])
            kernel = tuple(int(value) for value in conf["kernel"])
            modules.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel,
                    stride=int(conf["stride"]),
                    padding=int(conf["padding"]),
                )
            )
            modules.append(activation_module(activation))
            in_channels = out_channels
        self.features = nn.Sequential(*modules)
        flatten_dim = int(np.prod(layer_shapes[-2]))
        self.classifier = nn.Linear(flatten_dim, int(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(torch.flatten(x, start_dim=1))


def run_dir_from_args(args: argparse.Namespace) -> Path:
    loss_label = None if args.loss in {"cross_entropy", "cse"} else str(args.loss)
    base = (
        Path(args.output_root).expanduser().resolve()
        / args.run_group
        / args.dataset
        / _activation_label(args.activation)
    )
    if loss_label is not None:
        base = base / loss_label
    return (
        base
        / f"conv{args.conv_depth}"
        / f"lr_{_label_float(args.learning_rate)}"
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
        pin_memory=torch.cuda.is_available(),
        generator=generator if shuffle else None,
    )


def build_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    dataset_spec = DATASET_SPECS[args.dataset]
    input_channels = int(dataset_spec["input_channels"])
    image_size = int(dataset_spec["image_size"])
    if args.synthetic_samples:
        generator = torch.Generator().manual_seed(int(args.seed))
        train_count = int(args.synthetic_samples)
        test_count = max(int(args.synthetic_test_samples), int(args.batch_size))
        train_x = torch.rand(
            train_count,
            input_channels,
            image_size,
            image_size,
            generator=generator,
        )
        train_y = torch.randint(0, args.num_classes, (train_count,), generator=generator)
        test_x = torch.rand(
            test_count,
            input_channels,
            image_size,
            image_size,
            generator=generator,
        )
        test_y = torch.randint(0, args.num_classes, (test_count,), generator=generator)
        if args.input_preprocessing == "centered":
            train_x = 2.0 * train_x - 1.0
            test_x = 2.0 * test_x - 1.0
        return (
            _make_loader(TensorDataset(train_x, train_y), args.batch_size, True, args.seed),
            _make_loader(TensorDataset(test_x, test_y), args.test_batch_size, False, args.seed),
        )

    if args.input_preprocessing == "standard":
        mean, std = STANDARD_NORMALIZATION[args.dataset]
    else:
        mean, std = INPUT_PREPROCESSING[args.input_preprocessing]
    if len(mean) == 1 and input_channels > 1:
        mean = tuple(float(mean[0]) for _ in range(input_channels))
        std = tuple(float(std[0]) for _ in range(input_channels))
    transform_steps = []
    if dataset_spec.get("grayscale", False):
        transform_steps.append(transforms.Grayscale(num_output_channels=1))
    transform_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    transform = transforms.Compose(transform_steps)
    dataset_root = os.path.expanduser(str(args.dataset_root))
    dataset_cls = DATASETS[args.dataset]
    download = not args.no_download
    if args.dataset == "mnist_rotated":
        train_base = dataset_cls(dataset_root, train=True, transform=None, download=download)
        test_base = dataset_cls(dataset_root, train=False, transform=None, download=download)
        train_ds = DeterministicRotatedDataset(
            train_base,
            transform=transform,
            degrees=args.rotation_degrees,
            seed=args.rotation_seed,
            split_offset=0,
        )
        test_ds = DeterministicRotatedDataset(
            test_base,
            transform=transform,
            degrees=args.rotation_degrees,
            seed=args.rotation_seed,
            split_offset=1_000_000,
        )
    elif args.dataset == "mnist_affine":
        train_base = dataset_cls(dataset_root, train=True, transform=None, download=download)
        test_base = dataset_cls(dataset_root, train=False, transform=None, download=download)
        train_ds = DeterministicAffineDataset(
            train_base,
            transform=transform,
            rotation_degrees=args.rotation_degrees,
            translation_fraction=args.translation_fraction,
            scale_min=args.scale_min,
            scale_max=args.scale_max,
            shear_degrees=args.shear_degrees,
            seed=args.rotation_seed,
            split_offset=0,
        )
        test_ds = DeterministicAffineDataset(
            test_base,
            transform=transform,
            rotation_degrees=args.rotation_degrees,
            translation_fraction=args.translation_fraction,
            scale_min=args.scale_min,
            scale_max=args.scale_max,
            shear_degrees=args.shear_degrees,
            seed=args.rotation_seed,
            split_offset=1_000_000,
        )
    else:
        train_ds = dataset_cls(dataset_root, train=True, transform=transform, download=download)
        test_ds = dataset_cls(dataset_root, train=False, transform=transform, download=download)
    return (
        _make_loader(train_ds, args.batch_size, True, args.seed),
        _make_loader(test_ds, args.test_batch_size, False, args.seed),
    )


def build_criterion(loss_name: str) -> nn.Module:
    if loss_name in {"cross_entropy", "cse"}:
        return nn.CrossEntropyLoss()
    if loss_name == "mse":
        return nn.MSELoss()
    raise ValueError(f"Expected loss 'cross_entropy', 'cse', or 'mse', got {loss_name!r}.")


def effective_transform_config(args: argparse.Namespace) -> dict:
    if args.dataset == "mnist_affine":
        return {
            "rotation_degrees": args.rotation_degrees,
            "rotation_seed": args.rotation_seed,
            "translation_fraction": args.translation_fraction,
            "scale_min": args.scale_min,
            "scale_max": args.scale_max,
            "shear_degrees": args.shear_degrees,
        }
    if args.dataset == "mnist_rotated":
        return {
            "rotation_degrees": args.rotation_degrees,
            "rotation_seed": args.rotation_seed,
            "translation_fraction": 0.0,
            "scale_min": 1.0,
            "scale_max": 1.0,
            "shear_degrees": 0.0,
        }
    return {
        "rotation_degrees": 0.0,
        "rotation_seed": "",
        "translation_fraction": 0.0,
        "scale_min": 1.0,
        "scale_max": 1.0,
        "shear_degrees": 0.0,
    }


def compute_loss(
    criterion: nn.Module,
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_name: str,
    num_classes: int,
) -> torch.Tensor:
    if loss_name in {"cross_entropy", "cse"}:
        return criterion(logits, labels)
    targets = torch.nn.functional.one_hot(labels, num_classes=int(num_classes)).to(
        device=logits.device,
        dtype=logits.dtype,
    )
    return criterion(logits, targets)


def evaluate(
    model: nn.Module,
    criterion: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None,
    loss_name: str,
    num_classes: int,
) -> dict:
    model.eval()
    running_loss = 0.0
    running_correct = 0
    seen = 0
    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(loader):
            if max_batches is not None and batch_index >= int(max_batches):
                break
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = compute_loss(criterion, logits, labels, loss_name, num_classes)
            running_loss += float(loss.item()) * int(images.size(0))
            running_correct += int((logits.argmax(dim=1) == labels).sum().item())
            seen += int(images.size(0))
    return {
        "loss": running_loss / seen if seen else math.nan,
        "accuracy": running_correct / seen if seen else math.nan,
        "num_samples": seen,
    }


def _should_log(batch_index: int, total_batches: int, log_interval: int) -> bool:
    batch_num = batch_index + 1
    if batch_num == 1 or batch_num == total_batches:
        return True
    return int(log_interval) > 0 and batch_num % int(log_interval) == 0


def train(args: argparse.Namespace) -> dict:
    _set_seed(int(args.seed))
    device = torch.device(args.device)
    run_dir = run_dir_from_args(args)
    if args.skip_complete and (run_dir / "metrics.json").exists():
        metrics = json.loads((run_dir / "metrics.json").read_text())
        print(f"[skip] complete run at {run_dir}", flush=True)
        return metrics
    run_dir.mkdir(parents=True, exist_ok=True)

    layer_shapes, conv_pipeline = architecture_from_args(args)
    model = FeedForwardConvNet(
        layer_shapes=layer_shapes,
        conv_pipeline=conv_pipeline,
        activation=args.activation,
        num_classes=args.num_classes,
    ).to(device)
    criterion = build_criterion(args.loss)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.learning_rate))
    train_loader, test_loader = build_loaders(args)
    transform_config = effective_transform_config(args)

    source_config = {
        "script": str(Path(__file__).resolve()),
        "run_group": args.run_group,
        "dataset": args.dataset,
        "dataset_root": args.dataset_root,
        "input_preprocessing": args.input_preprocessing,
        "activation": args.activation,
        "conv_depth": args.conv_depth,
        "conv_channels": [int(value) for value in args.conv_channels],
        "kernel_size": args.kernel_size,
        "strides": [int(conf["stride"]) for conf in conv_pipeline],
        "paddings": [int(conf["padding"]) for conf in conv_pipeline],
        "layer_shapes": [list(shape) for shape in layer_shapes],
        "input_channels": int(layer_shapes[0][0]),
        "image_size": int(layer_shapes[0][1]),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "test_batch_size": args.test_batch_size,
        "learning_rate": args.learning_rate,
        "optimizer": "Adam",
        "loss": args.loss,
        "seed": args.seed,
        "synthetic_samples": args.synthetic_samples,
        **transform_config,
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
        model.train()
        running_loss = 0.0
        running_correct = 0
        seen = 0
        total_batches = len(train_loader)
        for batch_index, (images, labels) in enumerate(train_loader):
            if args.max_batches is not None and batch_index >= int(args.max_batches):
                break
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = compute_loss(criterion, logits, labels, args.loss, args.num_classes)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * int(images.size(0))
            running_correct += int((logits.argmax(dim=1) == labels).sum().item())
            seen += int(images.size(0))

            if _should_log(batch_index, total_batches, args.log_interval):
                train_loss = running_loss / seen if seen else math.nan
                train_acc = running_correct / seen if seen else math.nan
                print(
                    f"Epoch {epoch + 1}/{args.epochs} | Batch {batch_index + 1}/{total_batches} "
                    f"| train loss={train_loss:.4f} acc={train_acc * 100:.2f}%",
                    flush=True,
                )

        train_loss = running_loss / seen if seen else math.nan
        train_acc = running_correct / seen if seen else math.nan
        test_result = evaluate(
            model,
            criterion,
            test_loader,
            device,
            args.max_test_batches,
            args.loss,
            args.num_classes,
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
            torch.save(model.state_dict(), best_model_path)

    final_model_path = run_dir / "final_model.pt"
    torch.save(model.state_dict(), final_model_path)
    if not best_model_path.exists():
        torch.save(model.state_dict(), best_model_path)
        best_test_accuracy = history["accuracy_test"][-1] if history["accuracy_test"] else math.nan
        best_epoch = len(history["accuracy_test"]) or None

    np.save(run_dir / "loss_train.npy", np.asarray(history["loss_train"], dtype=np.float64))
    np.save(run_dir / "accuracy_train.npy", np.asarray(history["accuracy_train"], dtype=np.float64))
    np.save(run_dir / "loss_test.npy", np.asarray(history["loss_test"], dtype=np.float64))
    np.save(run_dir / "accuracy_test.npy", np.asarray(history["accuracy_test"], dtype=np.float64))

    metrics = {
        "run_dir": str(run_dir),
        "run_group": args.run_group,
        "dataset": args.dataset,
        "activation": args.activation,
        "conv_depth": args.conv_depth,
        "conv_channels": [int(value) for value in args.conv_channels],
        "layer_shapes": [list(shape) for shape in layer_shapes],
        "strides": source_config["strides"],
        "paddings": source_config["paddings"],
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "optimizer": "Adam",
        "loss": args.loss,
        "input_preprocessing": args.input_preprocessing,
        **transform_config,
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
    print("[done] " + json.dumps(metrics, sort_keys=True), flush=True)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="mnist")
    parser.add_argument("--dataset-root", default=str(REPO_ROOT / "data"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--run-group", default="final")
    parser.add_argument("--conv-depth", type=int, required=True)
    parser.add_argument("--conv-channels", type=int, nargs="+", default=DEFAULT_CHANNELS)
    parser.add_argument("--activation", choices=("relu", "hard-sigmoid"), default="relu")
    parser.add_argument(
        "--input-preprocessing",
        choices=("identity", "centered", "standard"),
        default="identity",
    )
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--strides", type=int, nargs="+", default=None)
    parser.add_argument("--padding", type=int, default=1)
    parser.add_argument("--paddings", type=int, nargs="+", default=None)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--test-batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument(
        "--loss",
        choices=("cross_entropy", "cse", "mse"),
        default="cross_entropy",
        help="Training loss. 'cse' is accepted as an alias for cross entropy.",
    )
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--max-test-batches", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--synthetic-samples", type=int, default=0)
    parser.add_argument("--synthetic-test-samples", type=int, default=0)
    parser.add_argument(
        "--rotation-degrees",
        type=float,
        default=45.0,
        help="Maximum absolute per-sample rotation for mnist_rotated.",
    )
    parser.add_argument(
        "--rotation-seed",
        type=int,
        default=1729,
        help="Seed for deterministic per-index rotations/affine transforms.",
    )
    parser.add_argument(
        "--translation-fraction",
        type=float,
        default=0.0,
        help="Maximum absolute x/y translation as a fraction of image size for mnist_affine.",
    )
    parser.add_argument(
        "--scale-min",
        type=float,
        default=1.0,
        help="Minimum deterministic scale factor for mnist_affine.",
    )
    parser.add_argument(
        "--scale-max",
        type=float,
        default=1.0,
        help="Maximum deterministic scale factor for mnist_affine.",
    )
    parser.add_argument(
        "--shear-degrees",
        type=float,
        default=0.0,
        help="Maximum absolute x-shear angle for mnist_affine.",
    )
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--skip-complete", action="store_true")
    args = parser.parse_args()
    if args.translation_fraction < 0.0:
        raise ValueError(
            f"Expected --translation-fraction >= 0, got {args.translation_fraction}."
        )
    if args.scale_min <= 0.0 or args.scale_max <= 0.0 or args.scale_min > args.scale_max:
        raise ValueError(
            "Expected 0 < --scale-min <= --scale-max, "
            f"got scale_min={args.scale_min}, scale_max={args.scale_max}."
        )
    architecture_from_args(args)
    return args


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
