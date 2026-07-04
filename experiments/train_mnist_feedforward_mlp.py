#!/usr/bin/env python3
"""Train ordinary feed-forward MLPs on MNIST-style datasets with BP."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.train_mnist_feedforward_conv import (  # noqa: E402
    DATASET_SPECS,
    build_criterion,
    build_loaders,
    compute_loss,
    effective_transform_config,
)


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "mnist_feedforward_mlp"


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


def activation_module(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    raise ValueError(f"Expected activation 'relu', got {name!r}.")


class FeedForwardMLP(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_depth: int,
        hidden_size: int,
        activation: str,
        num_classes: int,
    ) -> None:
        super().__init__()
        if int(hidden_depth) < 1:
            raise ValueError(f"Expected --hidden-depth >= 1, got {hidden_depth}.")
        modules: list[nn.Module] = [nn.Flatten()]
        in_features = int(input_dim)
        for _ in range(int(hidden_depth)):
            modules.append(nn.Linear(in_features, int(hidden_size)))
            modules.append(activation_module(activation))
            in_features = int(hidden_size)
        modules.append(nn.Linear(in_features, int(num_classes)))
        self.network = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def architecture_from_args(args: argparse.Namespace) -> dict:
    spec = DATASET_SPECS[args.dataset]
    input_channels = int(spec["input_channels"])
    image_size = int(spec["image_size"])
    input_dim = input_channels * image_size * image_size
    return {
        "input_shape": [input_channels, image_size, image_size],
        "input_dim": input_dim,
        "hidden_depth": int(args.hidden_depth),
        "hidden_size": int(args.hidden_size),
        "num_classes": int(args.num_classes),
    }


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
        / f"mlp_depth{args.hidden_depth}_width{args.hidden_size}"
        / f"lr_{_label_float(args.learning_rate)}"
        / f"seed_{args.seed}"
    )


def evaluate(
    model: nn.Module,
    criterion: nn.Module,
    loader,
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

    architecture = architecture_from_args(args)
    model = FeedForwardMLP(
        input_dim=architecture["input_dim"],
        hidden_depth=args.hidden_depth,
        hidden_size=args.hidden_size,
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
        "model": "mlp",
        **architecture,
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
        "model": "mlp",
        **architecture,
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
    parser.add_argument("--dataset", choices=sorted(DATASET_SPECS), default="mnist")
    parser.add_argument("--dataset-root", default=str(REPO_ROOT / "data"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--run-group", default="final")
    parser.add_argument("--hidden-depth", type=int, required=True)
    parser.add_argument("--hidden-size", type=int, default=100)
    parser.add_argument("--activation", choices=("relu",), default="relu")
    parser.add_argument(
        "--input-preprocessing",
        choices=("identity", "centered", "standard"),
        default="identity",
    )
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
    parser.add_argument("--rotation-degrees", type=float, default=45.0)
    parser.add_argument("--rotation-seed", type=int, default=1729)
    parser.add_argument("--translation-fraction", type=float, default=0.0)
    parser.add_argument("--scale-min", type=float, default=1.0)
    parser.add_argument("--scale-max", type=float, default=1.0)
    parser.add_argument("--shear-degrees", type=float, default=0.0)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--skip-complete", action="store_true")
    args = parser.parse_args()
    if args.hidden_depth < 1:
        raise ValueError(f"Expected --hidden-depth >= 1, got {args.hidden_depth}.")
    if args.hidden_size < 1:
        raise ValueError(f"Expected --hidden-size >= 1, got {args.hidden_size}.")
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
