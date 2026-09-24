"""Exact epoch-boundary continuation for reset-each-batch MNIST training.

This supports planned pauses only. A damaged or advanced artifact tail is an
error, never silently discarded. No numerical update or epoch budget changes.
"""
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from experiments.reporting import atomic_write_json, sha256_file

PAUSED = 75


def file_identity(path):
    path = Path(path)
    return dict(bytes=path.stat().st_size, sha256=sha256_file(path)) if path.exists() else None


class EpochContinuation:
    def __init__(self, *, run_dir, config, epochs, chunk_epochs, parameters,
                 optimizer, estimator, loaders, recorder, contract):
        self.root = Path(run_dir)
        self.path = self.root / "continuation.pt"
        self.chunk = int(chunk_epochs)
        if self.chunk <= 0:
            raise ValueError("Continuation chunk must be positive")
        if config.get("batch_state_policy") != "reset_each_batch":
            raise ValueError("Continuation requires reset-each-batch states")
        if config.get("evaluation", {}).get("official_test", {}).get("policy") != "disabled":
            raise ValueError("Continuation currently supports validation-only training")
        if not hasattr(loaders, "_train_sampler") or not hasattr(loaders, "_worker_generator"):
            raise ValueError("Continuation requires deterministic MNIST loaders")
        self.epochs = int(epochs)
        self.parameters = tuple(parameters)
        self.optimizer, self.estimator, self.loaders, self.recorder = optimizer, estimator, loaders, recorder
        self.contract = contract
        self.start = 0

    def restore(self):
        if not self.path.exists():
            return None
        # This is a self-produced checkpoint, hash-checked before pickle decoding.
        receipt = json.loads((self.root / "continuation.json").read_text())
        if receipt["checkpoint_sha256"] != sha256_file(self.path):
            raise ValueError("Continuation checkpoint hash mismatch")
        state = torch.load(self.path, map_location="cpu", weights_only=False)
        if state["contract"] != self.contract:
            raise ValueError("Continuation scientific/runtime contract mismatch")
        if not 0 < state["epoch"] < self.epochs:
            raise ValueError("Continuation must precede the final epoch")
        for name, identity in state["artifacts"].items():
            if file_identity(self.root / name) != identity:
                raise ValueError(f"Continuation artifact changed: {name}")
        names = [str(p.name).strip() for p in self.parameters]
        if names != state["parameter_names"]:
            raise ValueError("Continuation parameter order mismatch")
        for parameter, saved in zip(self.parameters, state["parameters"], strict=True):
            if parameter.state.shape != saved.shape or parameter.state.dtype != saved.dtype:
                raise ValueError("Continuation tensor schema mismatch")
            with torch.no_grad():
                parameter.state.copy_(saved.to(parameter.state.device))
        self.optimizer.load_state_dict(state["optimizer"])
        self.loaders._train_sampler.generator.set_state(state["sampler_rng"])
        self.loaders._worker_generator.set_state(state["worker_rng"])
        noise = state["endpoint_noise"]
        if noise is not None:
            device = self.parameters[0].state.device
            generators = {}
            for key, rng in noise["generators"].items():
                if key != str(device):
                    raise ValueError("Continuation noise-generator device changed")
                generator = torch.Generator(device=device)
                generator.set_state(rng)
                generators[key] = generator
            self.estimator._endpoint_read_noise_generators = generators
            self.estimator._endpoint_read_noise_draw_count = noise["draw_count"]
        if self.recorder is not None:
            self.recorder.recorded_steps = state["trace_counts"][0]
            self.recorder.recorded_rows = state["trace_counts"][1]
        random.setstate(state["rng"]["python"])
        np.random.set_state(state["rng"]["numpy"])
        torch.set_rng_state(state["rng"]["cpu"])
        if state["rng"]["cuda"]:
            torch.cuda.set_rng_state_all(state["rng"]["cuda"])
        self.start = state["epoch"]
        return state

    def pause_after(self, epoch, *, history, best_accuracy, best_epoch):
        if epoch >= self.epochs or epoch - self.start < self.chunk:
            return False
        noise = None
        if hasattr(self.estimator, "_endpoint_read_noise_generators"):
            noise = dict(generators={k: g.get_state() for k, g in self.estimator._endpoint_read_noise_generators.items()},
                         draw_count=self.estimator.endpoint_read_noise_draw_count)
        state = dict(
            schema="mnist-epoch-continuation/v1", contract=self.contract, epoch=epoch,
            parameter_names=[str(p.name).strip() for p in self.parameters],
            parameters=[p.state.detach().cpu().clone() for p in self.parameters],
            optimizer=self.optimizer.state_dict(),
            sampler_rng=self.loaders._train_sampler.generator.get_state(),
            worker_rng=self.loaders._worker_generator.get_state(), endpoint_noise=noise,
            rng=dict(python=random.getstate(), numpy=np.random.get_state(), cpu=torch.get_rng_state(),
                     cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []),
            history=history, best_accuracy=best_accuracy, best_epoch=best_epoch,
            trace_counts=(self.recorder.recorded_steps, self.recorder.recorded_rows) if self.recorder is not None else None,
            artifacts={name: file_identity(self.root / name) for name in
                       ("best_model.pt", "metrics.jsonl", "gradient_trace.jsonl")},
        )
        temporary = self.path.with_suffix(".tmp")
        torch.save(state, temporary)
        os.replace(temporary, self.path)
        atomic_write_json(self.root / "continuation.json", dict(
            schema=state["schema"], completed_epochs=epoch, total_epochs=self.epochs,
            checkpoint_sha256=sha256_file(self.path), contract=self.contract))
        return True


def continuation_contract(config, *, epochs, max_batches, max_test_batches, device,
                          gradient_trace_samples_per_epoch):
    return dict(config_sha256=hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
                epochs=epochs, max_batches=max_batches, max_test_batches=max_test_batches,
                device=str(device), torch=torch.__version__, cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(device) if str(device).startswith("cuda") else "cpu",
                source=os.environ.get("EXPERIMENT_SOURCE_ARCHIVE_SHA256"),
                gradient_trace_samples_per_epoch=gradient_trace_samples_per_epoch)
