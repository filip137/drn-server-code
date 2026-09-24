#!/usr/bin/env python3
"""Fixed digital ReLU reference on the DRN train/validation split (no test read)."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from experiments import reporting
from experiments.train_mnist_feedforward_conv import FeedForwardConvNet, architecture_from_args
from labs.datasets import (
    _TrainOnlyMNIST, _batch_original_indices, stable_batch_order_hash,
    stable_index_sequence_hash, stratified_mnist_train_validation_indices,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    "experiments/train_digital_relu_reference.py",
    "experiments/train_mnist_feedforward_conv.py",
    "experiments/reporting.py", "labs/datasets.py",
)


def build_model(config, depth):
    args = argparse.Namespace(
        dataset="mnist", conv_channels=config["channels"], conv_depth=depth,
        strides=config["strides"], stride=2, paddings=None,
        padding=config["padding"], kernel_size=config["kernel_size"],
        num_classes=config["num_classes"],
    )
    shapes, pipeline = architecture_from_args(args)
    model = FeedForwardConvNet(layer_shapes=shapes, conv_pipeline=pipeline,
                              activation=config["activation"], num_classes=10)
    for name, parameter in model.named_parameters():
        if name.endswith("bias"):
            nn.init.constant_(parameter, config["bias_initialization"])
    return model


def build_optimizer(model, config):
    return torch.optim.Adam([
        {"params": [p], "lr": config["bias_learning_rate"] if n.endswith("bias")
         else config["weight_learning_rate"], "name": n}
        for n, p in model.named_parameters()
    ], betas=tuple(config["adam_betas"]), eps=config["adam_eps"],
        weight_decay=config["weight_decay"], foreach=False, fused=False)


def loss_value(logits, labels, loss):
    if loss == "mse":
        return F.mse_loss(logits, F.one_hot(labels, 10).to(logits.dtype))
    if loss == "cross_entropy":
        return F.cross_entropy(logits, labels)
    raise ValueError(loss)


def state_hash(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def check_state(model):
    for name, value in model.state_dict().items():
        if not torch.isfinite(value).all():
            raise RuntimeError(f"Nonfinite parameter {name}")
        if name.endswith("bias") and torch.count_nonzero(value):
            raise RuntimeError(f"Bias changed from zero: {name}")


def cached_data(config, dataset_root, device):
    # This reader checks and loads only the two training resources.
    dataset = _TrainOnlyMNIST(dataset_root, download=False)
    train, validation = stratified_mnist_train_validation_indices(
        dataset.targets, split_seed=config["split_seed"])
    norm = config["normalization"]
    images = dataset.data[:, None].to(dtype=torch.float32).div_(255)
    images.sub_(norm["mean"]).div_(norm["std"]).mul_(norm["scale"])
    labels = dataset.targets.to(dtype=torch.long)
    return images.to(device), labels.to(device), train, validation


@torch.no_grad()
def evaluate(model, images, labels, batches, loss):
    model.eval()
    totals = torch.zeros(2, device=images.device, dtype=torch.float64)
    count = 0
    for indices in batches:
        logits = model(images[indices])
        batch_labels = labels[indices]
        totals[0] += loss_value(logits, batch_labels, loss).double() * len(indices)
        totals[1] += (logits.argmax(1) == batch_labels).sum()
        count += len(indices)
    values = (totals / count).cpu().tolist()
    if not all(math.isfinite(v) for v in values):
        raise RuntimeError("Nonfinite validation metric")
    return {"loss": values[0], "accuracy": values[1], "examples": count}


def run_case(config, args, data, depth, seed, loss, deadline):
    images, labels, train, validation = data
    run_id = f"conv{depth}_{loss}_seed{seed}"
    run_dir = args.output_root / ("smoke" if args.smoke else "training") / run_id
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {run_dir}")
    torch.manual_seed(seed)
    model = build_model(config, depth)
    initial_hash = state_hash(model)
    model.to(args.device)
    optimizer = build_optimizer(model, config)
    epochs = 1 if args.smoke else config["epochs_by_depth"][str(depth)]
    order = _batch_original_indices(train, batch_size=config["batch_size"],
                                   shuffle_seed=seed, num_epochs=epochs)
    if args.smoke:
        order = tuple(tuple(epoch[:1]) for epoch in order)
        validation = validation[:config["validation_batch_size"]]
    val_batches = torch.tensor(validation, device=args.device).split(config["validation_batch_size"])
    provenance = {
        "name": "ordinary_mnist", "evaluation_split": "validation",
        "official_test_read": False, "split_seed": config["split_seed"],
        "shuffle_seed": seed, "train_size": len(train), "validation_size": len(validation),
        "train_indices_sha256": stable_index_sequence_hash(train),
        "validation_indices_sha256": stable_index_sequence_hash(validation),
        "epoch_batch_order_sha256": [stable_batch_order_hash(e) for e in order],
        "raw_training_files": {name: reporting.sha256_file(args.dataset_root / "MNIST/raw" / name)
                               for name in ("train-images-idx3-ubyte", "train-labels-idx1-ubyte")},
    }
    source = {name: reporting.sha256_file(ROOT / name) for name in SOURCE_FILES}
    git_identity = ROOT / "source_git.json"
    git = json.loads(git_identity.read_text()) if git_identity.exists() else {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "status": subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True),
    }
    resolved = dict(config, depth=depth, channels=config["channels"][:depth],
                    strides=config["strides"][:depth], seed=seed, loss=loss, epochs=epochs)
    manifest = {
        "study_id": config["study_id"], "run_id": run_id,
        "arm_id": f"conv{depth}_{loss}", "seed": seed, "smoke": args.smoke,
        "evidence_class": "ordinary_mnist_selection", "dataset": provenance,
        "config": resolved, "command": shlex.join([sys.executable, *sys.argv]),
        "config_sha256": reporting.sha256_file(args.config), "source_sha256": source,
        "git": git, "initial_state_sha256": initial_hash,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "learning_rates_by_parameter": {g["name"]: g["lr"] for g in optimizer.param_groups},
        "runtime": dict(reporting.runtime_context(target=args.target), pid=os.getpid()),
        "environment": {"torch": torch.__version__, "numpy": np.__version__,
                        "cuda": torch.version.cuda, "device": str(args.device),
                        "gpu": torch.cuda.get_device_name() if args.device.startswith("cuda") else None,
                        "deterministic_algorithms": True, "tf32": False},
    }
    reporting.start_run(run_dir, manifest)
    start = time.monotonic()
    try:
        reporting.atomic_write_json(run_dir / "artifacts/config.json", resolved)
        for name in SOURCE_FILES:
            dest = run_dir / "artifacts/source" / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, dest)
        np.savez(run_dir / "artifacts/split_indices.npz", train=train, validation=validation)
        torch.save(model.state_dict(), run_dir / "checkpoints/initial_model.pt")
        best, best_epoch, steps = -1.0, None, 0
        log = (run_dir / "artifacts/train.log").open("w")
        for epoch, batches in enumerate(order, 1):
            if time.monotonic() > deadline:
                raise TimeoutError("Declared wall-time budget exhausted")
            model.train()
            totals = torch.zeros(2, device=args.device, dtype=torch.float64)
            count = 0
            indices = torch.tensor([i for batch in batches for i in batch], device=args.device)
            epoch_start = time.monotonic()
            for step, batch in enumerate(indices.split(config["batch_size"]), 1):
                optimizer.zero_grad(set_to_none=True)
                logits = model(images[batch])
                value = loss_value(logits, labels[batch], loss)
                if not torch.isfinite(value):
                    raise RuntimeError("Nonfinite training loss")
                value.backward()
                optimizer.step()
                totals[0] += value.detach().double() * len(batch)
                totals[1] += (logits.detach().argmax(1) == labels[batch]).sum()
                count += len(batch)
                steps += 1
                if step % 500 == 0:
                    if time.monotonic() > deadline:
                        raise TimeoutError("Declared wall-time budget exhausted")
                    reporting.update_status_progress(run_dir, {
                        "stage": "training", "epoch": epoch, "epochs": epochs,
                        "step": step, "steps": steps})
            check_state(model)
            measured = evaluate(model, images, labels, val_batches, loss)
            train_loss, train_accuracy = (totals / count).cpu().tolist()
            metric = {"stage": "training", "epoch": epoch, "split": "validation",
                      "validation_loss": measured["loss"], "validation_accuracy": measured["accuracy"],
                      "train_loss": train_loss, "train_accuracy": train_accuracy,
                      "train_examples": count, "validation_examples": measured["examples"],
                      "steps": steps, "epoch_seconds": time.monotonic() - epoch_start}
            reporting.append_metric(run_dir / "metrics.jsonl", metric)
            if measured["accuracy"] > best:
                best, best_epoch = measured["accuracy"], epoch
                torch.save(model.state_dict(), run_dir / "checkpoints/best_model.pt")
            message = f"{run_id} epoch {epoch}/{epochs}: validation={100*measured['accuracy']:.2f}% best={100*best:.2f}% seconds={metric['epoch_seconds']:.2f}"
            print(message, flush=True)
            log.write(message + "\n")
            log.flush()
            reporting.update_status_progress(run_dir, {"stage": "training", "epoch": epoch, "epochs": epochs})
        torch.save(model.state_dict(), run_dir / "checkpoints/final_model.pt")
        torch.save(optimizer.state_dict(), run_dir / "checkpoints/final_optimizer.pt")
        log.close()
        reporting.complete_run(run_dir, terminal_metrics={
            "best_validation_accuracy": best, "best_epoch": best_epoch,
            "final_validation_accuracy": measured["accuracy"],
            "final_validation_loss": measured["loss"], "epochs": epochs,
            "optimizer_steps": steps, "elapsed_seconds": time.monotonic() - start,
        }, completion={"criteria_met": True, "biases_exact_zero": True,
                       "official_test_read": False, "protocol_deviations": [],
                       "expected_optimizer_steps": sum(len(b) for b in order)})
        errors = reporting.validate_run(run_dir)
        if errors:
            raise RuntimeError(errors)
    except BaseException as error:
        if not (run_dir / "result.json").exists():
            reporting.fail_run(run_dir, error=error)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3], choices=[1, 2, 3])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--target", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=14400)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config["official_test_read"] or config["weight_bounds"] is not None:
        raise ValueError("This runner supports validation-only unconstrained digital references")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    data = cached_data(config, args.dataset_root, args.device)
    deadline = time.monotonic() + args.max_seconds
    for depth in args.depths:
        for seed in ([0] if args.smoke else config["seeds"]):
            for loss in config["losses"]:
                run_case(config, args, data, depth, seed, loss, deadline)


if __name__ == "__main__":
    main()
