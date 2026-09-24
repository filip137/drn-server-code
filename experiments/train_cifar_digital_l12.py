#!/usr/bin/env python3
"""Exploratory digital CIFAR-10 block networks (L12 and L8 logical graphs)."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import shlex
import shutil
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
import torchvision
from torchvision import datasets, transforms

from experiments import reporting


class DigitalL12(nn.Module):
    def __init__(self, config):
        super().__init__()
        layers = []
        channels = config["input_shape"][0]
        for index, block in enumerate(config["blocks"]):
            for output_channels in block:
                layers.extend([
                    nn.Conv2d(channels, output_channels, config["kernel_size"],
                              stride=config["stride"], padding=config["padding"],
                              bias=config["conv_bias"]),
                    nn.ReLU(inplace=False),
                ])
                channels = output_channels
            if index in config["pool_after_blocks"]:
                layers.append(nn.MaxPool2d(2, 2))
            if index in config["batchnorm_after_blocks"]:
                layers.append(nn.BatchNorm2d(channels, affine=config["batchnorm_affine"]))
        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(config["readout_features"], config["num_classes"])

    def forward(self, inputs):
        return self.head(self.features(inputs).flatten(1))


def seed_worker(_worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def build_loaders(config, dataset_root, smoke, device):
    norm = transforms.Normalize(config["dataset"]["mean"], config["dataset"]["std"])
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomCrop(32, padding=4, padding_mode="edge"),
        transforms.ToTensor(), norm,
    ])
    test_transform = transforms.Compose([transforms.ToTensor(), norm])
    train = datasets.CIFAR10(dataset_root, train=True, transform=train_transform, download=False)
    test = datasets.CIFAR10(dataset_root, train=False, transform=test_transform, download=False)
    if len(train) != config["dataset"]["train_size"] or len(test) != config["dataset"]["test_size"]:
        raise ValueError("Unexpected CIFAR-10 split sizes")
    if smoke:
        train = Subset(train, range(3 * config["batch_size"]))
        test = Subset(test, range(config["evaluation_batch_size"]))
    common = dict(num_workers=config["num_workers"], pin_memory=device.type == "cuda",
                  worker_init_fn=seed_worker, persistent_workers=config["num_workers"] > 0)
    return (
        DataLoader(train, batch_size=config["batch_size"], shuffle=True,
                   generator=torch.Generator().manual_seed(config["seed"]), **common),
        DataLoader(test, batch_size=config["evaluation_batch_size"], shuffle=False,
                   generator=torch.Generator().manual_seed(config["seed"] + 1), **common),
    )


def evaluate(model, loader, device, deadline):
    model.eval()
    totals = torch.zeros(2, device=device, dtype=torch.float64)
    count = 0
    with torch.no_grad():
        for images, labels in loader:
            if time.monotonic() > deadline:
                raise TimeoutError("Declared runtime budget exhausted")
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, labels)
            totals[0] += loss.double() * len(labels)
            totals[1] += (logits.argmax(1) == labels).sum()
            count += len(labels)
    loss, accuracy = (totals / count).cpu().tolist()
    if not math.isfinite(loss):
        raise FloatingPointError("Nonfinite test loss")
    return {"loss": loss, "accuracy": accuracy, "examples": count}


def save_checkpoint(path, model, optimizer, scheduler, epoch, config):
    temporary = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "epoch": epoch, "config": config,
                "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
                "python_rng": random.getstate()}, temporary)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--source-identity", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--target", default="akibscomputer")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    run_dir = args.output_root / ("smoke" if args.smoke else "production-seed0")
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {run_dir}")
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(config["deterministic_algorithms"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = config["tf32"]
    torch.backends.cudnn.allow_tf32 = config["tf32"]
    device = torch.device(args.device)
    model = DigitalL12(config).to(device)
    if sum(isinstance(m, nn.Conv2d) for m in model.modules()) != sum(map(len, config["blocks"])):
        raise ValueError("Convolution count differs from configured blocks")
    parameters = sum(p.numel() for p in model.parameters())
    opt = config["optimizer"]
    optimizer_parameters = model.parameters()
    if opt.get("exclude_batchnorm_from_weight_decay", False):
        bn_parameters = [p for m in model.modules() if isinstance(m, nn.BatchNorm2d)
                         for p in m.parameters(recurse=False)]
        bn_ids = {id(p) for p in bn_parameters}
        optimizer_parameters = [
            {"params": [p for p in model.parameters() if id(p) not in bn_ids]},
            {"params": bn_parameters, "weight_decay": 0.0},
        ]
    optimizer = torch.optim.Adam(optimizer_parameters, lr=opt["lr"], betas=tuple(opt["betas"]),
                                 eps=opt["eps"], weight_decay=opt["weight_decay"],
                                 foreach=False, fused=False)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["scheduler"]["horizon_epochs"],
        eta_min=opt["lr"] * config["scheduler"]["final_lr_ratio"])
    source_identity = json.loads(args.source_identity.read_text())
    root = Path(__file__).resolve().parents[1]
    for name, expected in source_identity["source_sha256"].items():
        if reporting.sha256_file(root / name) != expected:
            raise ValueError(f"Staged source mismatch: {name}")
    epochs = 1 if args.smoke else config["epochs"]
    reporting.start_run(run_dir, {
        "study_id": config["study_id"], "run_id": run_dir.name,
        "arm_id": config.get("arm_id", "digital-l12-logical-width"),
        "seed": seed, "smoke": args.smoke, "evidence_class": config["evidence_class"],
        "resolved_config": config, "effective_epochs": epochs,
        "dataset": dict(config["dataset"], root=str(args.dataset_root), official_test_read=True),
        "command": shlex.join([sys.executable, *sys.argv]), "source_identity": source_identity,
        "config_sha256": reporting.sha256_file(args.config), "parameter_count": parameters,
        "runtime": dict(reporting.runtime_context(target=args.target), pid=os.getpid()),
        "environment": {"torch": torch.__version__, "torchvision": torchvision.__version__,
                        "numpy": np.__version__, "cuda": torch.version.cuda,
                        "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
    })
    start = time.monotonic()
    deadline = start + config["maximum_runtime_hours"] * 3600
    try:
        reporting.atomic_write_json(run_dir / "artifacts/resolved_config.json", config)
        shutil.copyfile(args.source_identity, run_dir / "artifacts/source_identity.json")
        for name in source_identity["source_sha256"]:
            destination = run_dir / "artifacts/source" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / name, destination)
        train, test = build_loaders(config, args.dataset_root, args.smoke, device)
        print(f"START parameters={parameters} train={len(train.dataset)} test={len(test.dataset)} batch={config['batch_size']} epochs={epochs}", flush=True)
        save_checkpoint(run_dir / "checkpoints/initial_model.pt", model, optimizer, scheduler, 0, config)
        best_key = (-1.0, -float("inf"))
        best_epoch, steps, epoch_seconds = 0, 0, []
        for epoch in range(1, epochs + 1):
            epoch_start = time.monotonic()
            model.train()
            totals = torch.zeros(2, device=device, dtype=torch.float64)
            count = 0
            lr = optimizer.param_groups[0]["lr"]
            for batch, (images, labels) in enumerate(train, 1):
                if time.monotonic() > deadline:
                    raise TimeoutError("Declared runtime budget exhausted")
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = nn.functional.cross_entropy(logits, labels)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite train loss")
                loss.backward()
                if batch == 1:
                    norms = {n: float(p.grad.norm()) for n, p in model.named_parameters() if p.grad is not None}
                    if len(norms) != len(list(model.parameters())) or not all(math.isfinite(v) and v > 0 for v in norms.values()):
                        raise FloatingPointError("Missing, nonfinite or zero parameter gradient")
                optimizer.step()
                totals[0] += loss.detach().double() * len(labels)
                totals[1] += (logits.detach().argmax(1) == labels).sum()
                count += len(labels)
                steps += 1
                if batch == 1 or batch % 100 == 0:
                    reporting.update_status_progress(run_dir, {"stage": "training", "epoch": epoch,
                        "epochs": epochs, "batch": batch, "batches": len(train), "optimizer_steps": steps})
                    print(f"epoch={epoch}/{epochs} batch={batch}/{len(train)} loss={float(loss):.5f}", flush=True)
            reporting.update_status_progress(run_dir, {"stage": "test", "epoch": epoch, "epochs": epochs})
            measured = evaluate(model, test, device, deadline)
            train_loss, train_accuracy = (totals / count).cpu().tolist()
            elapsed = time.monotonic() - epoch_start
            epoch_seconds.append(elapsed)
            metric = {"stage": "training", "epoch": epoch, "evaluation_split": "official_test",
                      "train_loss": train_loss, "train_accuracy": train_accuracy, "train_examples": count,
                      "test_loss": measured["loss"], "test_accuracy": measured["accuracy"],
                      "test_examples": measured["examples"], "lr": lr, "optimizer_steps": steps,
                      "epoch_seconds": elapsed, "first_batch_gradient_norms": norms,
                      "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0}
            reporting.append_metric(run_dir / "metrics.jsonl", metric)
            scheduler.step()
            key = (measured["accuracy"], -measured["loss"])
            if key > best_key:
                best_key, best_epoch = key, epoch
                save_checkpoint(run_dir / "checkpoints/best_model.pt", model, optimizer, scheduler, epoch, config)
            save_checkpoint(run_dir / "checkpoints/last_model.pt", model, optimizer, scheduler, epoch, config)
            print(f"EPOCH {epoch}/{epochs} train={train_accuracy:.6f} test={measured['accuracy']:.6f} best={best_key[0]:.6f} seconds={elapsed:.2f}", flush=True)
        os.replace(run_dir / "checkpoints/last_model.pt", run_dir / "checkpoints/final_model.pt")
        reporting.complete_run(run_dir, terminal_metrics={
            "epochs_completed": epochs, "best_test_accuracy": best_key[0], "best_epoch": best_epoch,
            "final_test_accuracy": measured["accuracy"], "final_test_loss": measured["loss"],
            "final_train_accuracy": train_accuracy, "optimizer_steps": steps,
            "elapsed_seconds": time.monotonic() - start, "mean_epoch_seconds": float(np.mean(epoch_seconds)),
            "parameter_count": parameters,
        }, completion={"criteria_met": epochs == (1 if args.smoke else config["epochs"]),
                       "official_test_read": True, "paper_facing": False})
        errors = reporting.validate_run(run_dir)
        if errors:
            raise RuntimeError(errors)
        print(f"COMPLETE {run_dir}", flush=True)
    except BaseException as error:
        if not (run_dir / "result.json").exists():
            reporting.fail_run(run_dir, error=error)
        raise


if __name__ == "__main__":
    main()
