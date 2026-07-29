#!/usr/bin/env python3
"""Probe optimizer update units and run a layerwise Conv two-rho grid."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import itertools
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


PARAMETER_RE = re.compile(r"^(ConvWeight|DenseWeight|Bias)_(\d+)$")
DEFAULT_PROBE_COUNTS = (32, 64, 128)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _positive(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Expected {label} to be positive and finite. Provided value: {value!r}.") from error
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"Expected {label} to be positive and finite. Provided value: {value!r}.")
    return number


def rms(tensor: torch.Tensor) -> float:
    value = tensor.detach().to(dtype=torch.float64)
    return float(torch.sqrt(torch.mean(value.square())).item())


def parameter_sha256(states: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in states.items():
        value = tensor.detach().contiguous().cpu()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(value.shape)).encode("ascii") + b"\0")
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def linear_quantile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError(f"Expected a non-empty sample. Provided value: {values!r}.")
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"Expected q to be in [0, 1]. Provided value: {q!r}.")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def parameter_topology(names: Sequence[str]) -> dict[str, Any]:
    normalized = [str(name).strip() for name in names]
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"Expected unique parameter names. Provided value: {normalized!r}.")

    kinds: dict[str, str] = {}
    conv_by_index: dict[int, str] = {}
    biases: dict[str, int] = {}
    for name in normalized:
        match = PARAMETER_RE.fullmatch(name)
        if match is None:
            raise ValueError(
                "Expected only ConvWeight_i, DenseWeight_i, and Bias_i parameters. "
                f"Provided value: {name!r}."
            )
        kind, raw_index = match.groups()
        kinds[name] = kind
        index = int(raw_index)
        if kind == "ConvWeight":
            conv_by_index[index] = name
        elif kind == "Bias":
            biases[name] = index

    mapping = {}
    for bias, index in biases.items():
        if index not in conv_by_index:
            raise ValueError(
                f"Expected {bias!r} to have an attached ConvWeight_{index}. "
                f"Provided parameter names: {normalized!r}."
            )
        mapping[bias] = conv_by_index[index]
    if not conv_by_index or not any(kind == "DenseWeight" for kind in kinds.values()):
        raise ValueError(
            "Expected at least one ConvWeight and one DenseWeight. "
            f"Provided parameter names: {normalized!r}."
        )
    return {
        "parameter_names": normalized,
        "kinds": kinds,
        "bias_to_weight": mapping,
        "weight_names": [
            name for name in normalized if kinds[name] in {"ConvWeight", "DenseWeight"}
        ],
    }


def nominal_proposal(gradient: torch.Tensor, optimizer_name: str, eps: float = 1e-8) -> torch.Tensor:
    if optimizer_name == "SGD":
        return -gradient
    if optimizer_name == "Adam":
        return -gradient / (gradient.abs() + eps)
    raise ValueError(
        "Expected optimizer_name to be one of ('SGD', 'Adam'). "
        f"Provided value: {optimizer_name!r}."
    )


def _relative_difference(left: float, right: float) -> float:
    return abs(left - right) / max(abs(left), abs(right), 1e-30)


class OptimizerProbe:
    """Collect nominal-LR-one proposal units from the trainer's gradients."""

    def __init__(
        self,
        optimizer_name: str,
        batch_counts: Sequence[int] = DEFAULT_PROBE_COUNTS,
        stability_tolerance: float = 0.10,
        adam_eps: float = 1e-8,
    ):
        counts = tuple(int(value) for value in batch_counts)
        if not counts or any(value < 2 or value % 2 for value in counts):
            raise ValueError(
                "Expected probe batch counts to be positive even integers >= 2. "
                f"Provided value: {counts!r}."
            )
        if tuple(sorted(set(counts))) != counts:
            raise ValueError(
                "Expected probe batch counts to be strictly increasing. "
                f"Provided value: {counts!r}."
            )
        self.optimizer_name = optimizer_name
        self.batch_counts = counts
        self.stability_tolerance = float(stability_tolerance)
        if not math.isfinite(self.stability_tolerance) or self.stability_tolerance < 0.0:
            raise ValueError(
                "Expected stability_tolerance to be non-negative and finite. "
                f"Provided value: {stability_tolerance!r}."
            )
        self.adam_eps = _positive(adam_eps, "Adam epsilon")
        self.topology: dict[str, Any] | None = None
        self.parameter_refs: dict[str, Any] = {}
        self.initial_states: dict[str, torch.Tensor] = {}
        self.initial_weight_rms: dict[str, float] = {}
        self.samples: dict[str, list[float]] = {}
        self.split_statistics: dict[str, dict[str, float]] = {}
        self.used_batches = 0
        self.stable = False

    @staticmethod
    def _statistic(name: str, values: Sequence[float]) -> float:
        return linear_quantile(values, 0.9 if name.startswith("Bias_") else 0.5)

    def __call__(self, batch: Mapping[str, Any]) -> bool:
        parameters = tuple(batch["parameters"])
        gradients = tuple(batch["gradients"])
        names = [str(getattr(param, "name", "")).strip() for param in parameters]
        if len(parameters) != len(gradients):
            raise RuntimeError(
                f"Expected one gradient per parameter. Provided values: "
                f"parameters={len(parameters)}, gradients={len(gradients)}."
            )

        if self.topology is None:
            self.topology = parameter_topology(names)
            self.parameter_refs = dict(zip(names, parameters))
            self.initial_states = {
                name: param.state.detach().clone()
                for name, param in zip(names, parameters)
            }
            self.initial_weight_rms = {
                name: rms(self.initial_states[name])
                for name in self.topology["weight_names"]
            }
            invalid = {
                name: value
                for name, value in self.initial_weight_rms.items()
                if not math.isfinite(value) or value <= 0.0
            }
            if invalid:
                raise RuntimeError(
                    "Expected every initial weight RMS to be positive and finite. "
                    f"Provided value: {invalid!r}."
                )
            self.samples = {name: [] for name in names}
        elif names != self.topology["parameter_names"]:
            raise RuntimeError(
                "Expected stable parameter names during the probe. "
                f"Provided value: {names!r}."
            )

        for name, gradient in zip(names, gradients):
            attached = self.topology["bias_to_weight"].get(name, name)
            denominator = self.initial_weight_rms[attached]
            delta = nominal_proposal(
                gradient.detach(),
                self.optimizer_name,
                eps=self.adam_eps,
            )
            unit = rms(delta) / denominator
            if not math.isfinite(unit) or unit < 0.0:
                raise RuntimeError(
                    f"Expected a finite non-negative proposal unit for {name!r}. "
                    f"Provided value: {unit!r}."
                )
            self.samples[name].append(unit)

        self.used_batches += 1
        if self.used_batches not in self.batch_counts:
            return False

        half = self.used_batches // 2
        self.split_statistics = {}
        unstable = []
        for name, values in self.samples.items():
            first = self._statistic(name, values[:half])
            second = self._statistic(name, values[half:self.used_batches])
            difference = _relative_difference(first, second)
            self.split_statistics[name] = {
                "first_half": first,
                "second_half": second,
                "relative_difference": difference,
            }
            if difference > self.stability_tolerance:
                unstable.append(name)
        self.stable = not unstable
        return self.stable or self.used_batches == self.batch_counts[-1]

    def result(self) -> dict[str, Any]:
        if self.topology is None or self.used_batches not in self.batch_counts:
            raise RuntimeError(
                "Expected the probe to reach a configured batch count. "
                f"Provided value: {self.used_batches!r}."
            )
        mutated = [
            name
            for name, initial in self.initial_states.items()
            if not torch.equal(initial, self.parameter_refs[name].state.detach())
        ]
        if mutated:
            raise RuntimeError(
                "Expected probing not to mutate parameters. "
                f"Provided value: {mutated!r}."
            )
        weight_units = {
            name: self._statistic(name, self.samples[name][:self.used_batches])
            for name in self.topology["weight_names"]
        }
        bias_units = {
            name: self._statistic(name, self.samples[name][:self.used_batches])
            for name in self.topology["bias_to_weight"]
        }
        invalid = {
            name: value
            for name, value in weight_units.items()
            if not math.isfinite(value) or value <= 0.0
        }
        if invalid:
            raise RuntimeError(
                "Expected every weight proposal unit to be positive and finite. "
                f"Provided value: {invalid!r}."
            )
        unstable = [
            name
            for name, values in self.split_statistics.items()
            if values["relative_difference"] > self.stability_tolerance
        ]
        return {
            "schema_version": "conv-rho-probe/v1",
            "status": "complete" if self.stable else "unresolved_probe",
            "probe_stable": self.stable,
            "optimizer_name": self.optimizer_name,
            "parameter_names": self.topology["parameter_names"],
            "weight_names": self.topology["weight_names"],
            "bias_to_weight": self.topology["bias_to_weight"],
            "used_batches": self.used_batches,
            "batch_counts": list(self.batch_counts),
            "stability_tolerance": self.stability_tolerance,
            "unstable_parameters": unstable,
            "split_half_statistics": self.split_statistics,
            "initial_parameter_sha256": parameter_sha256(self.initial_states),
            "initial_weight_rms_by_parameter": self.initial_weight_rms,
            "normalization_unit_by_weight": weight_units,
            "bias_q90_unit_by_parameter": bias_units,
            "proposal_unit_samples_by_parameter": {
                name: values[:self.used_batches] for name, values in self.samples.items()
            },
            "official_test_read": False,
        }


def derive_learning_rates(
    probe: Mapping[str, Any],
    rho_conv: float,
    rho_dense: float,
    bias_policy: str = "q90_cap",
) -> dict[str, float]:
    if probe.get("probe_stable") is not True:
        raise ValueError(
            "Expected a stable optimizer probe before deriving learning rates. "
            f"Provided value: {probe.get('status')!r}."
        )
    rho_conv = _positive(rho_conv, "rho_conv")
    rho_dense = _positive(rho_dense, "rho_dense")
    if bias_policy not in {"q90_cap", "tied"}:
        raise ValueError(
            "Expected bias_policy to be one of ('q90_cap', 'tied'). "
            f"Provided value: {bias_policy!r}."
        )

    names = [str(name) for name in probe["parameter_names"]]
    topology = parameter_topology(names)
    units = probe["normalization_unit_by_weight"]
    rates: dict[str, float] = {}
    for name in topology["weight_names"]:
        unit = _positive(units.get(name), f"proposal unit for {name!r}")
        target = rho_conv if name.startswith("ConvWeight_") else rho_dense
        rates[name] = target / unit

    bias_units = probe["bias_q90_unit_by_parameter"]
    for bias, attached in topology["bias_to_weight"].items():
        attached_rate = rates[attached]
        unit = float(bias_units[bias])
        if not math.isfinite(unit) or unit < 0.0:
            raise ValueError(
                f"Expected bias unit for {bias!r} to be non-negative and finite. "
                f"Provided value: {unit!r}."
            )
        rates[bias] = (
            attached_rate
            if bias_policy == "tied" or unit == 0.0
            else min(attached_rate, rho_conv / unit)
        )
    return {name: float(rates[name]) for name in names}


def _optimizer_config(name: str, rates: Sequence[float]) -> dict[str, Any]:
    config: dict[str, Any] = {
        "name": name,
        "learning_rate": list(rates),
        "lr_decay": 1.0,
        "momentum": 0.0,
        "weight_decay": 0.0,
    }
    if name == "Adam":
        config.update(betas=[0.9, 0.999], eps=1e-8)
    return config


def prepare_training_config(
    base: Mapping[str, Any],
    optimizer_name: str,
    rates: Sequence[float],
    *,
    epochs: int,
    max_batches: int | None,
    max_validation_batches: int | None,
    split_seed: int,
    shuffle_seed: int,
    validation_batch_size: int,
) -> dict[str, Any]:
    config = copy.deepcopy(dict(base))
    dataset_key = str(config.get("lab", {}).get("dataset_key", "mnist"))
    if dataset_key != "mnist":
        raise ValueError(
            "Expected the two-rho search source config to use dataset_key 'mnist'. "
            f"Provided value: {dataset_key!r}."
        )
    dataset = config["datasets"][dataset_key]
    params = dataset["params"]
    dataset["factory"] = "labs.datasets.MnistTrainValidationDataset"
    params.setdefault("name", dataset_key)
    params["train"] = True
    params["split_seed"] = int(split_seed)
    params["shuffle_seed"] = int(shuffle_seed)
    params["validation_batch_size"] = int(validation_batch_size)

    config.setdefault("lab", {})["epochs"] = int(epochs)
    config["training_algorithm"] = "BP"
    config["lr"] = list(rates)
    config["lr_decay"] = 1.0
    config["optimizer"] = _optimizer_config(optimizer_name, rates)
    config["max_batches"] = max_batches
    config["max_test_batches"] = max_validation_batches
    return config


def _configured_rate_count(config: Mapping[str, Any]) -> int:
    optimizer = config.get("optimizer", {})
    values = optimizer.get("learning_rate") if isinstance(optimizer, Mapping) else None
    if not isinstance(values, (list, tuple)):
        values = config.get("lr")
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(
            "Expected the source config to contain an explicit optimizer learning-rate vector. "
            f"Provided value: {values!r}."
        )
    return len(values)


def _run_trainer(
    config_path: Path,
    output_dir: Path,
    *,
    device: str | None,
    gradient_callback=None,
    apply_optimizer_steps: bool = True,
) -> dict[str, Any]:
    from labs.mnist_train import train_mnist_conv

    config = json.loads(config_path.read_text(encoding="utf-8"))
    return train_mnist_conv(
        config_path=config_path,
        epochs=int(config["lab"]["epochs"]),
        lr=config["lr"],
        beta=config.get("beta"),
        log_interval=int(config.get("log_interval", 100)),
        max_batches=config.get("max_batches"),
        max_test_batches=config.get("max_test_batches"),
        device=device,
        dataset_key=str(config["lab"].get("dataset_key", "mnist")),
        output_dir=output_dir,
        model_key=str(config["lab"]["model_key"]),
        training_algorithm="BP",
        seed=config.get("seed"),
        lr_decay=1.0,
        gradient_callback=gradient_callback,
        apply_optimizer_steps=apply_optimizer_steps,
    )


def _float_token(value: float) -> str:
    return format(value, ".12g").replace("-", "m").replace(".", "p").replace("+", "")


def _summary_row(index: int, rho_conv: float, rho_dense: float, cell_dir: Path) -> dict[str, Any]:
    cell = json.loads((cell_dir / "cell.json").read_text(encoding="utf-8"))
    metrics_path = cell_dir / "metrics.json"
    metrics = (
        json.loads(metrics_path.read_text(encoding="utf-8"))
        if cell["status"] == "complete" and metrics_path.exists()
        else {}
    )
    return {
        "index": index,
        "optimizer": cell["optimizer"],
        "rho_conv": rho_conv,
        "rho_dense": rho_dense,
        "status": cell["status"],
        "final_validation_loss": metrics.get("final_test_loss"),
        "final_validation_accuracy": metrics.get("final_test_accuracy"),
        "best_validation_accuracy": metrics.get("best_test_accuracy"),
        "learning_rates": json.dumps(cell["learning_rate_vector"]),
        "path": str(cell_dir),
    }


def _write_summary(output_root: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _write_json(output_root / "summary.json", list(rows))
    fieldnames = list(rows[0]) if rows else [
        "index", "optimizer", "rho_conv", "rho_dense", "status",
        "final_validation_loss", "final_validation_accuracy",
        "best_validation_accuracy", "learning_rates", "path",
    ]
    with (output_root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _git_state() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD", "--"],
        cwd=root,
        check=False,
        capture_output=True,
    )
    fingerprint = hashlib.sha256()
    fingerprint.update(status.stdout.encode("utf-8"))
    fingerprint.update(diff.stdout)
    fingerprint.update(Path(__file__).read_bytes())
    return {
        "commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
        "working_tree_sha256": fingerprint.hexdigest(),
    }


def _cell_signature(
    resolved: Mapping[str, Any],
    probe: Mapping[str, Any],
    rho_conv: float,
    rho_dense: float,
) -> dict[str, Any]:
    probe_digest = hashlib.sha256(
        json.dumps(probe, sort_keys=True, allow_nan=False).encode("utf-8")
    ).hexdigest()
    return {
        "source_config_sha256": resolved["source_config_sha256"],
        "code": resolved["code"],
        "optimizer": resolved["optimizer"],
        "bias_policy": resolved["bias_policy"],
        "probe_sha256": probe_digest,
        "rho_conv": rho_conv,
        "rho_dense": rho_dense,
        "epochs": resolved["epochs"],
        "max_batches": resolved["max_batches"],
        "max_validation_batches": resolved["max_validation_batches"],
        "split_seed": resolved["split_seed"],
        "shuffle_seed": resolved["shuffle_seed"],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    base_path = Path(args.config).expanduser().resolve()
    base = json.loads(base_path.read_text(encoding="utf-8"))
    optimizer_name = args.optimizer or str(base.get("optimizer", {}).get("name", ""))
    if optimizer_name not in {"SGD", "Adam"}:
        raise ValueError(
            "Expected --optimizer or config optimizer name to be one of ('SGD', 'Adam'). "
            f"Provided value: {optimizer_name!r}."
        )
    epochs = int(args.epochs if args.epochs is not None else base["lab"]["epochs"])
    if epochs <= 0:
        raise ValueError(f"Expected epochs to be positive. Provided value: {epochs!r}.")
    rhos_conv = [_positive(value, "rho_conv") for value in args.rho_conv]
    rhos_dense = [_positive(value, "rho_dense") for value in args.rho_dense]
    counts = tuple(args.probe_batches)
    rate_count = _configured_rate_count(base)
    output_root = Path(args.output_root).expanduser().resolve()
    resolved = {
        "schema_version": "conv-rho-search/v1",
        "source_config": str(base_path),
        "source_config_sha256": hashlib.sha256(base_path.read_bytes()).hexdigest(),
        "code": _git_state(),
        "optimizer": optimizer_name,
        "rho_conv": rhos_conv,
        "rho_dense": rhos_dense,
        "bias_policy": args.bias_policy,
        "probe_batches": list(counts),
        "stability_tolerance": args.stability_tolerance,
        "epochs": epochs,
        "max_batches": args.max_batches,
        "max_validation_batches": args.max_validation_batches,
        "split_seed": args.split_seed,
        "shuffle_seed": args.shuffle_seed,
        "validation_batch_size": args.validation_batch_size,
        "device": args.device,
    }
    if args.dry_run:
        resolved["cells"] = len(rhos_conv) * len(rhos_dense)
        return resolved

    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "resolved.json", resolved)
    probe_path = output_root / "probe.json"
    probe = None
    if probe_path.exists() and not args.force:
        candidate = json.loads(probe_path.read_text(encoding="utf-8"))
        if candidate.get("search_signature") == resolved:
            probe = candidate
    if probe is None:
        probe_dir = output_root / "probe"
        probe_config = prepare_training_config(
            base,
            optimizer_name,
            [0.0] * rate_count,
            epochs=1,
            max_batches=max(counts),
            max_validation_batches=1,
            split_seed=args.split_seed,
            shuffle_seed=args.shuffle_seed,
            validation_batch_size=args.validation_batch_size,
        )
        probe_config_path = probe_dir / "source_config.json"
        _write_json(probe_config_path, probe_config)
        callback = OptimizerProbe(
            optimizer_name,
            counts,
            args.stability_tolerance,
            adam_eps=float(probe_config["optimizer"].get("eps", 1e-8)),
        )
        _run_trainer(
            probe_config_path,
            probe_dir,
            device=args.device,
            gradient_callback=callback,
            apply_optimizer_steps=False,
        )
        probe = callback.result()
        probe_metrics = json.loads(
            (probe_dir / "metrics.json").read_text(encoding="utf-8")
        )
        probe["dataset_provenance"] = probe_metrics["dataset_provenance"]
        probe["search_signature"] = resolved
        _write_json(probe_path, probe)

    if probe.get("probe_stable") is not True:
        raise RuntimeError(
            "Rho probe remained unstable at the largest requested batch count. "
            f"Unstable parameters: {probe.get('unstable_parameters')!r}."
        )
    if args.probe_only:
        return {"probe": str(probe_path), "status": "complete"}

    rows = []
    for index, (rho_conv, rho_dense) in enumerate(
        itertools.product(rhos_conv, rhos_dense)
    ):
        name = (
            f"{index:03d}_rc_{_float_token(rho_conv)}_"
            f"rd_{_float_token(rho_dense)}"
        )
        cell_dir = output_root / "cells" / name
        cell_path = cell_dir / "cell.json"
        signature = _cell_signature(resolved, probe, rho_conv, rho_dense)
        if cell_path.exists() and not args.force:
            existing = json.loads(cell_path.read_text(encoding="utf-8"))
            reusable = (
                existing.get("signature") == signature
                and (
                    (
                        existing.get("status") == "complete"
                        and (cell_dir / "metrics.json").exists()
                    )
                    or existing.get("status") == "failed_nonfinite"
                )
            )
            if reusable:
                rows.append(_summary_row(index, rho_conv, rho_dense, cell_dir))
                continue

        rates_by_name = derive_learning_rates(
            probe,
            rho_conv,
            rho_dense,
            args.bias_policy,
        )
        rate_vector = [rates_by_name[name] for name in probe["parameter_names"]]
        cell = {
            "index": index,
            "optimizer": optimizer_name,
            "rho_conv": rho_conv,
            "rho_dense": rho_dense,
            "bias_policy": args.bias_policy,
            "learning_rates_by_parameter": rates_by_name,
            "learning_rate_vector": rate_vector,
            "signature": signature,
            "status": "running",
        }
        _write_json(cell_path, cell)
        candidate_config = prepare_training_config(
            base,
            optimizer_name,
            rate_vector,
            epochs=epochs,
            max_batches=args.max_batches,
            max_validation_batches=args.max_validation_batches,
            split_seed=args.split_seed,
            shuffle_seed=args.shuffle_seed,
            validation_batch_size=args.validation_batch_size,
        )
        config_path = cell_dir / "source_config.json"
        _write_json(config_path, candidate_config)
        from labs.mnist_train import NonFiniteTrainingError

        try:
            _run_trainer(config_path, cell_dir, device=args.device)
        except NonFiniteTrainingError as error:
            cell["status"] = "failed_nonfinite"
            cell["error"] = str(error)
        else:
            cell["status"] = "complete"
        _write_json(cell_path, cell)
        rows.append(_summary_row(index, rho_conv, rho_dense, cell_dir))

    _write_summary(output_root, rows)
    return {
        "status": "complete",
        "probe": str(probe_path),
        "summary": str(output_root / "summary.json"),
        "cells": len(rows),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Normal Conv source_config.json")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--rho-conv", type=float, nargs="+", required=True)
    parser.add_argument("--rho-dense", type=float, nargs="+", required=True)
    parser.add_argument("--optimizer", choices=("SGD", "Adam"))
    parser.add_argument("--bias-policy", choices=("q90_cap", "tied"), default="q90_cap")
    parser.add_argument("--probe-batches", type=int, nargs="+", default=list(DEFAULT_PROBE_COUNTS))
    parser.add_argument("--stability-tolerance", type=float, default=0.10)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--max-validation-batches", type=int)
    parser.add_argument("--validation-batch-size", type=int, default=128)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--shuffle-seed", type=int, default=0)
    parser.add_argument("--device")
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    result = run(parse_args(argv))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
