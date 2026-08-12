#!/usr/bin/env python3
"""Replay epoch checkpoints on one fixed medium-affine validation cohort.

This analyzer is deliberately read-only with respect to the input study.  It
loads the real epoch-0 through epoch-10 model checkpoints, replays identical
validation tensors through the frozen BPTT contract, and never constructs or
steps an optimizer.  Checkpoint bytes and parameter tensors are hashed before
and after every measurement.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LABS_ROOT = REPOSITORY_ROOT / "labs"
for import_root in (REPOSITORY_ROOT, LABS_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from custom_classes import FlexibleDeepResistiveEnergy  # noqa: E402
from labs.datasets import (  # noqa: E402
    stable_batch_order_hash,
    stable_index_sequence_hash,
)
from labs.mnist_train import (  # noqa: E402
    _batch_reset_flag,
    _build_tracking_minimizer,
    _require_minimizer_config,
    _reset_name_counters,
    _resolve_callable,
    _resolve_dataset_config,
    _set_seed,
    _validate_diode_param_config,
    load_config,
)
from model.function.cost import (  # noqa: E402
    SquaredError,
    SquaredErrorPairedOutputs,
)
from model.function.network import Network  # noqa: E402
from training.sgd import Backprop  # noqa: E402
from experiments.paper_gradient_trace_contract import (  # noqa: E402
    REPOSITORY_ROOT as FROZEN_RUNTIME_SOURCE_ROOT,
    validate_frozen_manifest,
)


SCHEMA = "paper-gradient-trace-epoch-checkpoint-replay/v1"
DETAIL_SCHEMA = "paper-gradient-trace-epoch-gradient-detail/v1"
SUMMARY_SCHEMA = "paper-gradient-trace-epoch-gradient-summary/v1"
ZERO_EPSILON = 1.0e-12
ARM_PATTERN = re.compile(
    r"^(conv[123])_(baseline|ours|legacy)_(sgd|adam)_seed(\d+)$"
)


def _expected_arm_ids() -> tuple[str, ...]:
    arms = [
        f"conv1_{scheme}_{optimizer}_seed0"
        for scheme in ("baseline", "ours", "legacy")
        for optimizer in ("sgd", "adam")
    ]
    arms.extend(
        f"{architecture}_{scheme}_sgd_seed0"
        for architecture in ("conv2", "conv3")
        for scheme in ("baseline", "ours", "legacy")
    )
    return tuple(arms)


EXPECTED_ARM_IDS = _expected_arm_ids()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(b"\0")
    digest.update(json.dumps(list(tensor.shape)).encode("utf-8"))
    digest.update(b"\0")
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _parameter_state_sha256(parameters: Sequence[Any]) -> str:
    digest = hashlib.sha256()
    digest.update(b"drn-ordered-parameter-state/v1\0")
    for parameter in parameters:
        name = str(getattr(parameter, "name", "")).strip()
        state = parameter.state.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(state.dtype).encode("utf-8"))
        digest.update(b"\0")
        digest.update(json.dumps(list(state.shape)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(state.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _batch_payload_sha256(
    images: torch.Tensor,
    labels: torch.Tensor,
    source_indices: Sequence[int],
) -> str:
    digest = hashlib.sha256()
    digest.update(b"paper-gradient-replay-batch/v1\0")
    digest.update(_tensor_sha256(images).encode("ascii"))
    digest.update(_tensor_sha256(labels).encode("ascii"))
    digest.update(stable_index_sequence_hash(source_indices).encode("ascii"))
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Expected at least one row for {path}.")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected {path} to contain a JSON object.")
    return value


def _arm_identity(config: dict[str, Any]) -> dict[str, Any]:
    arm_id = config.get("arm_id")
    if not arm_id:
        reporting = config.get("reporting", {})
        arm_id = reporting.get("arm_id") if isinstance(reporting, dict) else None
    match = ARM_PATTERN.fullmatch(str(arm_id or ""))
    if match is None:
        raise ValueError(f"Expected a canonical paper arm_id; found {arm_id!r}.")
    architecture, scheme, optimizer, seed = match.groups()
    canonical_optimizer = {"sgd": "SGD", "adam": "Adam"}[optimizer]
    configured_optimizer = str(config.get("optimizer", {}).get("name", ""))
    if configured_optimizer and configured_optimizer != canonical_optimizer:
        raise ValueError(
            f"arm_id optimizer {optimizer!r} differs from config optimizer "
            f"{configured_optimizer!r}."
        )
    return {
        "arm_id": str(arm_id),
        "architecture": architecture,
        "scheme": scheme,
        "optimizer": canonical_optimizer,
        "seed": int(seed),
    }


def _source_config_provenance(
    run_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    manifest_path = run_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    configuration = manifest.get("configuration")
    if not isinstance(configuration, dict):
        raise ValueError(f"Missing manifest configuration in {manifest_path}.")
    manifest_config = json.loads(json.dumps(configuration.get("resolved")))
    used_config = json.loads(json.dumps(config))
    dataset_key = str(config["lab"]["dataset_key"])
    try:
        manifest_root = manifest_config["datasets"][dataset_key]["params"].pop(
            "root"
        )
        used_root = used_config["datasets"][dataset_key]["params"].pop("root")
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"Missing source dataset root while validating {run_dir}."
        ) from error
    if manifest_config != used_config:
        raise ValueError(
            "config.used.json differs scientifically from the immutable manifest "
            f"config in {run_dir}."
        )
    transport = manifest.get("transport_overrides", {})
    recorded_override = (
        transport.get("dataset_root") if isinstance(transport, dict) else None
    )
    if manifest_root != used_root and recorded_override != used_root:
        raise ValueError(
            f"Unrecorded dataset-root override in {run_dir}: {manifest_root!r} -> "
            f"{used_root!r}."
        )
    return {
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": _sha256_file(manifest_path),
        "source_config_path": configuration.get("path"),
        "source_config_sha256": configuration.get("sha256"),
        "config_used_path": str((run_dir / "config.used.json").resolve()),
        "config_used_sha256": _sha256_file(run_dir / "config.used.json"),
        "resolved_config_science_matches_manifest": True,
        "manifest_dataset_root": manifest_root,
        "runtime_dataset_root": used_root,
        "dataset_root_transport_override": recorded_override,
    }


def _discover_runs(input_root: Path) -> list[Path]:
    root = input_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Input study root does not exist: {root}")
    by_arm: dict[str, Path] = {}
    for result_path in sorted(root.rglob("result.json")):
        run_dir = result_path.parent
        if "smoke" in run_dir.relative_to(root).parts:
            continue
        config_path = run_dir / "config.used.json"
        checkpoint_index = run_dir / "epoch_checkpoint_index.jsonl"
        if not config_path.is_file() or not checkpoint_index.is_file():
            continue
        identity = _arm_identity(_read_json(config_path))
        arm_id = identity["arm_id"]
        if arm_id not in EXPECTED_ARM_IDS:
            raise ValueError(
                f"Unexpected canonical production arm under the study root: {arm_id}."
            )
        if arm_id in by_arm:
            raise ValueError(
                f"Found duplicate production run directories for {arm_id}: "
                f"{by_arm[arm_id]} and {run_dir}."
            )
        result = _read_json(result_path)
        completion = result.get("completion", {})
        if not isinstance(completion, dict) or completion.get("criteria_met") is not True:
            raise ValueError(f"Run completion criteria were not met: {run_dir}")
        by_arm[arm_id] = run_dir.resolve()
    missing = sorted(set(EXPECTED_ARM_IDS) - set(by_arm))
    extra = sorted(set(by_arm) - set(EXPECTED_ARM_IDS))
    if missing or extra:
        raise ValueError(
            "Expected exactly the twelve frozen gradient-trace arms. "
            f"Missing={missing}, extra={extra}."
        )
    return [by_arm[arm_id] for arm_id in EXPECTED_ARM_IDS]


def _safe_run_path(run_dir: Path, relative: str) -> Path:
    candidate = (run_dir / relative).resolve()
    if not candidate.is_relative_to(run_dir.resolve()):
        raise ValueError(f"Checkpoint path escapes run directory: {relative!r}.")
    return candidate


def _load_checkpoint_inventory(
    run_dir: Path,
    *,
    expected_epochs: Sequence[int] = tuple(range(11)),
) -> list[dict[str, Any]]:
    index_path = run_dir / "epoch_checkpoint_index.jsonl"
    records = [
        json.loads(line)
        for line in index_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    epochs = [int(record["epoch"]) for record in records]
    if epochs != list(expected_epochs):
        raise ValueError(
            f"Expected checkpoint epochs {list(expected_epochs)} in {run_dir}; "
            f"found {epochs}."
        )
    inventory = []
    for record in records:
        if record.get("schema_version") != "mnist-conv-diagnostic-epoch-checkpoint/v1":
            raise ValueError(f"Unexpected checkpoint index schema in {run_dir}: {record!r}")
        model_path = _safe_run_path(run_dir, str(record["model_path"]))
        optimizer_path = _safe_run_path(run_dir, str(record["optimizer_path"]))
        for path, size_key, hash_key in (
            (model_path, "model_size_bytes", "model_sha256"),
            (optimizer_path, "optimizer_size_bytes", "optimizer_sha256"),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"Missing checkpoint artifact: {path}")
            if path.stat().st_size != int(record[size_key]):
                raise ValueError(f"Checkpoint size mismatch: {path}")
            if _sha256_file(path) != str(record[hash_key]):
                raise ValueError(f"Checkpoint SHA-256 mismatch: {path}")
        inventory.append(
            {
                "epoch": int(record["epoch"]),
                "model_path": model_path,
                "model_sha256": str(record["model_sha256"]),
                "model_size_bytes": int(record["model_size_bytes"]),
                "optimizer_path": optimizer_path,
                "optimizer_sha256": str(record["optimizer_sha256"]),
                "optimizer_size_bytes": int(record["optimizer_size_bytes"]),
            }
        )
    return inventory


def _model_config(config: dict[str, Any]) -> dict[str, Any]:
    model_key = config["lab"]["model_key"]
    model = {
        **config["model_base"],
        **config["model_overrides"][model_key],
    }
    _validate_diode_param_config(model)
    _require_minimizer_config(model)
    return model


def _dataset_signature(
    config: dict[str, Any],
    *,
    data_root: Path,
    batch_size: int,
) -> dict[str, Any]:
    dataset_key = str(config["lab"]["dataset_key"])
    _, dataset_config = _resolve_dataset_config(config, dataset_key)
    params = json.loads(json.dumps(dataset_config["params"]))
    params["root"] = str(data_root.expanduser().resolve())
    params["validation_batch_size"] = int(batch_size)
    return {
        "dataset_key": dataset_key,
        "factory": dataset_config["factory"],
        "params": params,
        "input_mode": config.get("input_mode", "train"),
    }


def _build_validation_cohort(
    config: dict[str, Any],
    *,
    data_root: Path,
    batch_size: int,
    max_batches: int,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    signature = _dataset_signature(
        config,
        data_root=data_root,
        batch_size=batch_size,
    )
    params = dict(signature["params"])
    params["device"] = device
    dataset_factory = _resolve_callable(signature["factory"])
    loader_result = dataset_factory(**params).build()
    if not hasattr(loader_result, "validation_loader"):
        raise ValueError("Gradient replay requires the deterministic validation split.")
    validation_indices = tuple(int(value) for value in loader_result.validation_indices)
    expected_batches = min(
        int(max_batches),
        math.ceil(len(validation_indices) / int(batch_size)),
    )
    batches: list[dict[str, Any]] = []
    for batch_index, (images, labels) in enumerate(loader_result.validation_loader):
        if batch_index >= expected_batches:
            break
        start = batch_index * int(batch_size)
        stop = start + int(images.shape[0])
        source_indices = validation_indices[start:stop]
        images = images.detach().cpu().contiguous()
        labels = labels.detach().cpu().contiguous()
        batches.append(
            {
                "batch_index": batch_index,
                "images": images,
                "labels": labels,
                "source_indices": source_indices,
                "source_indices_sha256": stable_index_sequence_hash(source_indices),
                "payload_sha256": _batch_payload_sha256(
                    images,
                    labels,
                    source_indices,
                ),
            }
        )
    if len(batches) != expected_batches:
        raise RuntimeError(
            f"Expected {expected_batches} validation batches; found {len(batches)}."
        )
    cohort_hash = stable_batch_order_hash(
        batch["source_indices"] for batch in batches
    )
    cohort = {
        "schema": "paper-gradient-replay-validation-cohort/v1",
        "source_split": "MNIST train partitioned validation",
        "split_seed": int(loader_result.split_seed),
        "validation_indices_sha256": loader_result.validation_indices_hash,
        "validation_size": len(validation_indices),
        "batch_size": int(batch_size),
        "batch_count": len(batches),
        "example_count": sum(len(batch["source_indices"]) for batch in batches),
        "cohort_sha256": cohort_hash,
        "batches": [
            {
                "batch_index": int(batch["batch_index"]),
                "source_indices": list(batch["source_indices"]),
                "source_indices_sha256": batch["source_indices_sha256"],
                "payload_sha256": batch["payload_sha256"],
            }
            for batch in batches
        ],
        "dataset_signature": signature,
    }
    return batches, cohort


def _verify_source_dataset_provenance(
    run_dir: Path,
    config: dict[str, Any],
    cohort: dict[str, Any],
    *,
    data_root: Path,
    batch_size: int,
) -> None:
    if _dataset_signature(config, data_root=data_root, batch_size=batch_size) != cohort[
        "dataset_signature"
    ]:
        raise ValueError(f"Dataset/preprocessing contract differs for {run_dir}.")
    metrics = _read_json(run_dir / "metrics.json")
    provenance = metrics.get("dataset_provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"Missing deterministic dataset provenance in {run_dir}.")
    for source_key, cohort_key in (
        ("validation_indices_sha256", "validation_indices_sha256"),
        ("split_seed", "split_seed"),
    ):
        if provenance.get(source_key) != cohort[cohort_key]:
            raise ValueError(
                f"Source/replay dataset mismatch for {run_dir}: {source_key}."
            )


def _build_replay_runtime(
    config: dict[str, Any],
    *,
    device: torch.device,
) -> dict[str, Any]:
    if str(config.get("training_algorithm", "")).upper() != "BP":
        raise ValueError("Epoch replay is defined only for the frozen BPTT paper arms.")
    if config.get("batch_state_policy") != "reset_each_batch":
        raise ValueError("Expected the paper replay batch_state_policy reset_each_batch.")
    _set_seed(int(config["seed"]))
    _reset_name_counters()
    model = _model_config(config)
    energy_fn = FlexibleDeepResistiveEnergy(
        layer_shapes=[tuple(shape) for shape in model["layer_shapes"]],
        conv_pipeline=model.get("conv_pipeline") or [],
        pooling_mode=model.get("pooling_mode"),
        weight_gains=model["weight_gains"],
        input_gain=model["input_gain"],
        non_linearity=model["non_linearity"],
        exponential_diode_param=model["exponential_diode_param"],
        quadratic_diode_param=model["quadratic_diode_param"],
        hard_sigmoid_param=model["hard_sigmoid_param"],
        voltage_amp=model["voltage_amp"],
        current_amp=model["current_amp"],
        weight_min=model["weight_min"],
        weight_max=model["weight_max"],
        weight_init_mode=model.get("weight_init_mode", "kaiming_uniform"),
        input_mode=config.get("input_mode", "train"),
        trainable_amplification=bool(model.get("trainable_amplification", False)),
        amplification_min=model.get("amplification_min", 1.0e-6),
        amplification_max=model.get("amplification_max"),
    )
    energy_fn.set_device(device)
    network = Network(energy_fn)
    free_layers = network.free_layers()
    output_layer = energy_fn.layers()[-1]
    output_dim = int(output_layer.shape[0])
    if output_dim == 10:
        cost_fn = SquaredError(output_layer)
    elif output_dim == 20:
        cost_fn = SquaredErrorPairedOutputs(output_layer, num_classes=10)
    else:
        raise ValueError(f"Unsupported replay output dimension: {output_dim}.")
    inference_iterations = int(model["num_iterations_inference"])
    gradient_iterations = int(
        model.get("num_iterations_training", model["num_iterations_inference"])
    )
    mode = config["energy_minimizer"]["mode"]
    minimizer_inference = _build_tracking_minimizer(
        energy_fn,
        free_layers,
        model,
        mode,
        num_iterations=inference_iterations,
        voltage_amp=energy_fn._voltage_amp,
        current_amp=energy_fn._current_amp,
    )
    minimizer_gradient = _build_tracking_minimizer(
        energy_fn,
        free_layers,
        model,
        mode,
        num_iterations=gradient_iterations,
        voltage_amp=energy_fn._voltage_amp,
        current_amp=energy_fn._current_amp,
    )
    parameters = list(energy_fn.params())
    configured_order = config.get("parameter_order")
    actual_order = [str(parameter.name).strip() for parameter in parameters]
    if configured_order is not None and list(configured_order) != actual_order:
        raise ValueError(
            "Configured and runtime parameter orders differ: "
            f"configured={configured_order!r}, runtime={actual_order!r}."
        )
    return {
        "energy_fn": energy_fn,
        "network": network,
        "free_layers": free_layers,
        "cost_fn": cost_fn,
        "minimizer_inference": minimizer_inference,
        "estimator": Backprop(
            parameters,
            free_layers,
            cost_fn,
            minimizer_gradient,
        ),
        "parameters": parameters,
        "inference_iterations": inference_iterations,
        "gradient_iterations": gradient_iterations,
        "minimizer_mode": mode,
        "output_dim": output_dim,
        "model_config": model,
    }


def _finite_tensor(value: torch.Tensor, label: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"Non-finite tensor encountered in {label}.")


def _gradient_metrics(
    gradient: torch.Tensor,
    initial_gradient: torch.Tensor,
) -> dict[str, Any]:
    current = gradient.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    initial = initial_gradient.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    if current.shape != initial.shape:
        raise ValueError(
            f"Gradient shape changed across epochs: {current.shape} vs {initial.shape}."
        )
    _finite_tensor(current, "current gradient")
    _finite_tensor(initial, "initial gradient")
    count = int(current.numel())
    current_l2 = float(torch.linalg.vector_norm(current))
    initial_l2 = float(torch.linalg.vector_norm(initial))
    current_rms = float(torch.sqrt(torch.mean(current.square())))
    initial_rms = float(torch.sqrt(torch.mean(initial.square())))
    denominator = current_l2 * initial_l2
    cosine = (
        float(torch.dot(current, initial) / denominator)
        if denominator > 1.0e-30
        else None
    )
    return {
        "element_count": count,
        "gradient_l2": current_l2,
        "gradient_rms": current_rms,
        "gradient_zero_fraction": float(
            (current.abs() <= ZERO_EPSILON).to(torch.float64).mean()
        ),
        "gradient_exact_zero_fraction": float(
            (current == 0.0).to(torch.float64).mean()
        ),
        "initial_gradient_l2_same_batch": initial_l2,
        "initial_gradient_rms_same_batch": initial_rms,
        "gradient_rms_relative_to_epoch0": (
            current_rms / initial_rms if initial_rms > 1.0e-30 else None
        ),
        "gradient_cosine_vs_epoch0_same_batch": cosine,
        "gradient_relative_vector_change_vs_epoch0_same_batch": (
            float(torch.linalg.vector_norm(current - initial) / initial_l2)
            if initial_l2 > 1.0e-30
            else None
        ),
    }


def _quantile(values: Iterable[float], q: float) -> float:
    array = np.asarray(tuple(values), dtype=np.float64)
    if not len(array):
        raise ValueError("Expected a non-empty quantile input.")
    return float(np.quantile(array, q))


def _aggregate_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["arm_id"]), int(row["epoch"]), str(row["parameter_name"]))
        groups.setdefault(key, []).append(row)
    metrics = (
        "loss",
        "accuracy",
        "parameter_rms",
        "gradient_l2",
        "gradient_rms",
        "gradient_zero_fraction",
        "gradient_exact_zero_fraction",
        "gradient_rms_relative_to_epoch0",
        "gradient_cosine_vs_epoch0_same_batch",
        "gradient_relative_vector_change_vs_epoch0_same_batch",
    )
    output = []
    for key in sorted(groups):
        group = groups[key]
        first = group[0]
        summary = {"schema": SUMMARY_SCHEMA}
        summary.update(
            {
                name: first[name]
                for name in (
                    "architecture",
                    "scheme",
                    "optimizer",
                    "seed",
                    "arm_id",
                    "run_dir",
                    "config_sha256",
                    "epoch",
                    "checkpoint_role",
                    "checkpoint_path",
                    "checkpoint_sha256",
                "parameter_name",
                "parameter_type",
                "element_count",
                "learning_rate",
                "cohort_sha256",
                    "inference_iterations",
                    "gradient_iterations",
                )
            }
        )
        summary["batch_count"] = len(group)
        for metric in metrics:
            values = [
                float(row[metric])
                for row in group
                if row.get(metric) is not None
            ]
            summary[f"{metric}_median"] = _quantile(values, 0.5) if values else None
            summary[f"{metric}_q10"] = _quantile(values, 0.1) if values else None
            summary[f"{metric}_q90"] = _quantile(values, 0.9) if values else None
        output.append(summary)
    return output


def _replay_run(
    run_dir: Path,
    *,
    config: dict[str, Any],
    identity: dict[str, Any],
    inventory: Sequence[dict[str, Any]],
    batches: Sequence[dict[str, Any]],
    cohort: dict[str, Any],
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    runtime = _build_replay_runtime(config, device=device)
    parameters = runtime["parameters"]
    learning_rates = config.get("learning_rates_by_parameter")
    if not isinstance(learning_rates, dict):
        raise ValueError(f"Missing named learning rates in {run_dir}.")
    parameter_names = [str(parameter.name).strip() for parameter in parameters]
    if set(learning_rates) != set(parameter_names):
        raise ValueError(
            f"Named learning rates and runtime parameters differ in {run_dir}."
        )
    initial_gradients: dict[tuple[int, str], torch.Tensor] = {}
    rows: list[dict[str, Any]] = []
    guards = []
    config_sha256 = _sha256_file(run_dir / "config.used.json")
    frozen_provenance = validate_frozen_manifest(
        _read_json(run_dir / "manifest.json"),
        arm_id=identity["arm_id"],
    )
    source_config = _source_config_provenance(run_dir, config)

    for checkpoint in inventory:
        epoch = int(checkpoint["epoch"])
        checkpoint_path = checkpoint["model_path"]
        checkpoint_hash_before = _sha256_file(checkpoint_path)
        if checkpoint_hash_before != checkpoint["model_sha256"]:
            raise RuntimeError(f"Checkpoint changed after inventory: {checkpoint_path}")
        runtime["energy_fn"].load(checkpoint_path)
        parameter_hash_before = _parameter_state_sha256(parameters)
        for parameter in parameters:
            _finite_tensor(parameter.state, f"{identity['arm_id']} {parameter.name}")

        for batch in batches:
            batch_index = int(batch["batch_index"])
            images = batch["images"].to(device)
            labels = batch["labels"].to(device)
            runtime["network"].set_input(
                images,
                reset=_batch_reset_flag(config["batch_state_policy"], batch_index),
            )
            runtime["minimizer_inference"].compute_equilibrium()
            for layer in runtime["free_layers"]:
                _finite_tensor(
                    layer.state,
                    f"{identity['arm_id']} epoch {epoch} batch {batch_index} state",
                )
            runtime["cost_fn"].set_target(labels)
            batch_cost = runtime["cost_fn"].eval()
            _finite_tensor(batch_cost, "validation loss")
            loss = float(batch_cost.mean().item())
            accuracy = float(
                (~runtime["cost_fn"].error_fn()).to(torch.float64).mean().item()
            )
            gradients = runtime["estimator"].compute_gradient()
            if len(gradients) != len(parameters):
                raise RuntimeError(
                    f"Gradient/parameter count mismatch in {identity['arm_id']}."
                )
            for parameter, gradient in zip(parameters, gradients, strict=True):
                name = str(parameter.name).strip()
                gradient_cpu = gradient.detach().cpu().contiguous()
                initial_key = (batch_index, name)
                if epoch == 0:
                    initial_gradients[initial_key] = gradient_cpu.clone()
                initial = initial_gradients.get(initial_key)
                if initial is None:
                    raise RuntimeError(
                        f"Missing epoch-0 gradient for batch={batch_index}, parameter={name}."
                    )
                parameter_state = parameter.state.detach().to(
                    device="cpu", dtype=torch.float64
                )
                rows.append(
                    {
                        "schema": DETAIL_SCHEMA,
                        **identity,
                        "run_dir": str(run_dir.resolve()),
                        "config_sha256": config_sha256,
                        "epoch": epoch,
                        "checkpoint_role": "initialization" if epoch == 0 else "epoch",
                        "checkpoint_path": str(checkpoint_path.resolve()),
                        "checkpoint_sha256": checkpoint_hash_before,
                        "batch_index": batch_index,
                        "batch_size": int(images.shape[0]),
                        "batch_source_indices_sha256": batch[
                            "source_indices_sha256"
                        ],
                        "batch_payload_sha256": batch["payload_sha256"],
                        "cohort_sha256": cohort["cohort_sha256"],
                        "loss": loss,
                        "accuracy": accuracy,
                        "parameter_name": name,
                        "parameter_type": type(parameter).__name__,
                        "learning_rate": float(learning_rates[name]),
                        "parameter_rms": float(
                            torch.sqrt(torch.mean(parameter_state.square()))
                        ),
                        "parameter_sha256": _tensor_sha256(parameter.state),
                        "gradient_sha256": _tensor_sha256(gradient_cpu),
                        "inference_iterations": runtime["inference_iterations"],
                        "gradient_iterations": runtime["gradient_iterations"],
                        "minimizer_mode": runtime["minimizer_mode"],
                        "gradient_estimator": "BPTT",
                        "optimizer_steps_applied": False,
                        **_gradient_metrics(gradient_cpu, initial),
                    }
                )

        parameter_hash_after = _parameter_state_sha256(parameters)
        checkpoint_hash_after = _sha256_file(checkpoint_path)
        if parameter_hash_before != parameter_hash_after:
            raise RuntimeError(
                f"Replay modified parameter tensors for {identity['arm_id']} epoch {epoch}."
            )
        if checkpoint_hash_before != checkpoint_hash_after:
            raise RuntimeError(f"Replay modified source checkpoint: {checkpoint_path}")
        guards.append(
            {
                "epoch": epoch,
                "checkpoint_path": str(checkpoint_path.resolve()),
                "checkpoint_sha256_before": checkpoint_hash_before,
                "checkpoint_sha256_after": checkpoint_hash_after,
                "parameter_state_sha256_before": parameter_hash_before,
                "parameter_state_sha256_after": parameter_hash_after,
                "checkpoint_bytes_unchanged": True,
                "parameter_tensors_unchanged": True,
                "optimizer_steps_applied": False,
            }
        )

    return rows, {
        **identity,
        "run_dir": str(run_dir.resolve()),
        "config_path": str((run_dir / "config.used.json").resolve()),
        "config_sha256": config_sha256,
        "source_config": source_config,
        "frozen_provenance": frozen_provenance,
        "model_contract": {
            "output_dim": runtime["output_dim"],
            "inference_iterations": runtime["inference_iterations"],
            "gradient_iterations": runtime["gradient_iterations"],
            "minimizer_mode": runtime["minimizer_mode"],
            "gradient_estimator": "BPTT",
            "loss": (
                "paired_squared_error" if runtime["output_dim"] == 20 else "squared_error"
            ),
            "batch_state_policy": config["batch_state_policy"],
            "non_linearity": runtime["model_config"]["non_linearity"],
            "voltage_amp": float(runtime["model_config"]["voltage_amp"]),
            "current_amp": float(runtime["model_config"]["current_amp"]),
            "weight_min": runtime["model_config"]["weight_min"],
            "weight_max": runtime["model_config"]["weight_max"],
            "layer_shapes": runtime["model_config"]["layer_shapes"],
            "conv_pipeline": runtime["model_config"].get("conv_pipeline") or [],
            "parameter_order": parameter_names,
            "training_optimizer": config["optimizer"],
            "learning_rates_by_parameter": learning_rates,
        },
        "checkpoint_inventory": [
            {
                key: str(value) if isinstance(value, Path) else value
                for key, value in item.items()
            }
            for item in inventory
        ],
        "read_only_guards": guards,
    }


def _plot_run(
    rows: Sequence[dict[str, Any]],
    *,
    arm_id: str,
    output_dir: Path,
) -> Path:
    selected = [row for row in rows if row["arm_id"] == arm_id]
    if not selected:
        raise ValueError(f"No summary rows for {arm_id}.")
    parameters = sorted(
        {str(row["parameter_name"]) for row in selected},
        key=lambda name: (
            0 if name.startswith("ConvWeight_") else
            1 if name.startswith("DenseWeight_") else
            2,
            name,
        ),
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    panels = (
        (axes[0, 0], "gradient_rms_median", "Gradient RMS", True),
        (
            axes[0, 1],
            "gradient_rms_relative_to_epoch0_median",
            "Gradient RMS / epoch-0 RMS",
            True,
        ),
        (
            axes[1, 0],
            "gradient_cosine_vs_epoch0_same_batch_median",
            "Cosine vs epoch 0",
            False,
        ),
        (
            axes[1, 1],
            "gradient_zero_fraction_median",
            f"Fraction |g| <= {ZERO_EPSILON:.0e}",
            False,
        ),
    )
    for parameter in parameters:
        series = sorted(
            (row for row in selected if row["parameter_name"] == parameter),
            key=lambda row: int(row["epoch"]),
        )
        epochs = [int(row["epoch"]) for row in series]
        for axis, metric, ylabel, log_scale in panels:
            values = [row.get(metric) for row in series]
            if any(value is not None for value in values):
                axis.plot(epochs, values, marker="o", markersize=3, label=parameter)
            axis.set_ylabel(ylabel)
            axis.grid(True, alpha=0.25)
            if log_scale:
                axis.set_yscale("log")
    for axis in axes[1, :]:
        axis.set_xlabel("Checkpoint epoch")
    axes[1, 0].set_ylim(-1.05, 1.05)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="center right", fontsize=8)
        fig.subplots_adjust(right=0.82)
    fig.suptitle(f"Fixed-validation gradient replay: {arm_id}")
    output_path = output_dir / f"{arm_id}_epoch_gradients.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def analyze(
    *,
    input_root: Path,
    output_dir: Path,
    data_root: Path,
    device_name: str,
    max_batches: int,
    batch_size: int,
) -> dict[str, Any]:
    if int(max_batches) <= 0:
        raise ValueError(f"Expected max_batches > 0; found {max_batches!r}.")
    if int(batch_size) <= 0:
        raise ValueError(f"Expected batch_size > 0; found {batch_size!r}.")
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Expected a new or empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested CUDA device {device_name!r}, but CUDA is unavailable.")

    runs = _discover_runs(input_root)
    first_config = load_config(runs[0] / "config.used.json")
    batches, cohort = _build_validation_cohort(
        first_config,
        data_root=data_root,
        batch_size=batch_size,
        max_batches=max_batches,
        device=device,
    )

    detailed_rows: list[dict[str, Any]] = []
    source_runs = []
    for run_index, run_dir in enumerate(runs, start=1):
        config = load_config(run_dir / "config.used.json")
        identity = _arm_identity(config)
        _verify_source_dataset_provenance(
            run_dir,
            config,
            cohort,
            data_root=data_root,
            batch_size=batch_size,
        )
        inventory = _load_checkpoint_inventory(run_dir)
        print(
            f"[{run_index:02d}/{len(runs):02d}] replaying {identity['arm_id']} "
            f"({len(inventory)} checkpoints x {len(batches)} batches)",
            flush=True,
        )
        rows, run_summary = _replay_run(
            run_dir,
            config=config,
            identity=identity,
            inventory=inventory,
            batches=batches,
            cohort=cohort,
            device=device,
        )
        detailed_rows.extend(rows)
        source_runs.append(run_summary)

    summary_rows = _aggregate_rows(detailed_rows)
    detail_path = output_dir / "parameter_gradient_details.csv"
    summary_path = output_dir / "parameter_gradient_summary.csv"
    _write_csv(detail_path, detailed_rows)
    _write_csv(summary_path, summary_rows)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plots = [
        _plot_run(summary_rows, arm_id=arm_id, output_dir=plot_dir)
        for arm_id in EXPECTED_ARM_IDS
    ]
    result = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "input_root": str(input_root.expanduser().resolve()),
        "output_dir": str(output_dir),
        "analysis_settings": {
            "device": str(device),
            "cohort_split": "validation",
            "batch_size": int(batch_size),
            "max_batches": int(max_batches),
            "gradient_zero_epsilon": ZERO_EPSILON,
            "gradient_estimator": "BPTT",
            "checkpoint_epochs": list(range(11)),
            "optimizer_constructed": False,
            "optimizer_steps_applied": False,
            "official_test_read": False,
            "raw_gradients_not_lr_scaled": True,
            "runtime_source_root": str(FROZEN_RUNTIME_SOURCE_ROOT.resolve()),
        },
        "cohort": cohort,
        "source_runs": source_runs,
        "outputs": {
            "parameter_gradient_details_csv": str(detail_path),
            "parameter_gradient_summary_csv": str(summary_path),
            "plots": [str(path) for path in plots],
        },
        "counts": {
            "runs": len(runs),
            "checkpoints": sum(
                len(run["checkpoint_inventory"]) for run in source_runs
            ),
            "detail_rows": len(detailed_rows),
            "summary_rows": len(summary_rows),
        },
    }
    result_path = output_dir / "analysis_summary.json"
    result["outputs"]["analysis_summary_json"] = str(result_path)
    _write_json(result_path, result)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--max-batches",
        type=int,
        default=4,
        help="Number of leading deterministic validation batches to cache and replay.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Fixed validation replay batch size.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = analyze(
        input_root=args.input_root,
        output_dir=args.output_dir,
        data_root=args.data_root,
        device_name=args.device,
        max_batches=args.max_batches,
        batch_size=args.batch_size,
    )
    print(json.dumps(result["outputs"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
