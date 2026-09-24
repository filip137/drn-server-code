#!/usr/bin/env python3
"""Compare EqProp and BPTT weight gradients at Conv checkpoint endpoints.

The selected checkpoints were trained with signed, amplification-scaled biases.
This analyzer therefore bootstraps their exact compatible source worktree before
importing any project modules.  It is a read-only replay: no optimizer is ever
constructed, and source files plus in-memory parameter tensors are hashed before
and after every checkpoint case.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence


ANALYZER_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ANALYZER_REPOSITORY_ROOT
    / "configs/conv/perfectdiode_conv123_legacy_adam_eqprop_bptt_beta_cosine_seed0_20260810_v1.json"
)


def _preparse_path(flag: str, default: Path | None = None) -> Path | None:
    try:
        index = sys.argv.index(flag)
    except ValueError:
        return default
    if index + 1 >= len(sys.argv):
        raise SystemExit(f"{flag} requires a path")
    return Path(sys.argv[index + 1]).expanduser().resolve()


def _bootstrap_runtime_source() -> tuple[Path, Path]:
    config_path = _preparse_path("--config", DEFAULT_CONFIG)
    assert config_path is not None
    if not config_path.is_file():
        raise SystemExit(f"Missing analyzer config: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    configured_root = Path(config["source_contract"]["runtime_source_root"])
    source_root = _preparse_path("--runtime-source-root", configured_root)
    assert source_root is not None
    source_root = source_root.expanduser().resolve()
    if not (source_root / "model").is_dir() or not (source_root / "labs").is_dir():
        raise SystemExit(f"Invalid exact runtime source root: {source_root}")
    for import_root in (source_root / "labs", source_root):
        value = str(import_root)
        if value in sys.path:
            sys.path.remove(value)
        sys.path.insert(0, value)
    return config_path, source_root


BOOTSTRAP_CONFIG_PATH, RUNTIME_SOURCE_ROOT = _bootstrap_runtime_source()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from custom_classes import FlexibleDeepResistiveEnergy  # noqa: E402
from experiments.reporting import (  # noqa: E402
    append_metric,
    complete_run,
    fail_run,
    runtime_context,
    sha256_file,
    start_run,
    update_status_progress,
    validate_run,
)
from labs.datasets import (  # noqa: E402
    stable_batch_order_hash,
    stable_index_sequence_hash,
)
from labs.mnist_train import (  # noqa: E402
    _build_tracking_minimizer,
    _require_minimizer_config,
    _reset_name_counters,
    _resolve_callable,
    _resolve_dataset_config,
    _set_seed,
    _validate_diode_param_config,
)
from model.function.cost import SquaredError, SquaredErrorPairedOutputs  # noqa: E402
from model.function.network import Network  # noqa: E402
from model.minimizer.minimizer import ParamUpdater  # noqa: E402
from training.sgd import AugmentedFunction, Backprop  # noqa: E402


SCHEMA = "perfectdiode-conv-eqprop-bptt-checkpoint-gradient-analysis/v1"
COHORT_SCHEMA = "ordinary-mnist-fixed-validation-gradient-cohort/v1"
WEIGHT_TYPES = frozenset({"ConvWeight", "DenseWeight"})
NORM_EPSILON = 1.0e-30
STATE_BOUND_TOLERANCE = 1.0e-8


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Expected rows for {path}.")
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


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(b"\0")
    digest.update(json.dumps(list(tensor.shape)).encode("utf-8"))
    digest.update(b"\0")
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _tensor_sequence_sha256(
    named_tensors: Iterable[tuple[str, torch.Tensor]],
    *,
    schema: bytes,
) -> str:
    digest = hashlib.sha256()
    digest.update(schema + b"\0")
    for name, value in named_tensors:
        tensor = value.detach().cpu().contiguous()
        digest.update(str(name).strip().encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(b"\0")
        digest.update(json.dumps(list(tensor.shape)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _parameter_state_sha256(parameters: Sequence[Any]) -> str:
    return _tensor_sequence_sha256(
        ((str(parameter.name), parameter.state) for parameter in parameters),
        schema=b"drn-ordered-parameter-state/v1",
    )


def _layer_state_sha256(layers: Sequence[Any]) -> str:
    return _tensor_sequence_sha256(
        ((str(layer.name), layer.state) for layer in layers),
        schema=b"drn-ordered-free-layer-state/v1",
    )


def _historical_initialization_sha256(parameters: Sequence[Any]) -> str:
    """Reproduce the mechanism-study digest (name bytes + float32 bytes)."""

    digest = hashlib.sha256()
    for parameter in parameters:
        digest.update(str(parameter.name).strip().encode("utf-8"))
        array = (
            parameter.state.detach().cpu().numpy().astype(np.float32, copy=False)
        )
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _batch_payload_sha256(
    images: torch.Tensor,
    labels: torch.Tensor,
    source_indices: Sequence[int],
) -> str:
    digest = hashlib.sha256()
    digest.update(b"eqprop-bptt-gradient-replay-batch/v1\0")
    digest.update(_tensor_sha256(images).encode("ascii"))
    digest.update(_tensor_sha256(labels).encode("ascii"))
    digest.update(stable_index_sequence_hash(source_indices).encode("ascii"))
    return digest.hexdigest()


def _git_output(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _validate_runtime_source(config: Mapping[str, Any]) -> dict[str, Any]:
    contract = config["source_contract"]
    expected = Path(contract["runtime_source_root"]).expanduser().resolve()
    if RUNTIME_SOURCE_ROOT != expected:
        raise ValueError(
            f"Runtime override {RUNTIME_SOURCE_ROOT} differs from configured exact "
            f"source {expected}; a different source is not evidence-compatible."
        )
    head = _git_output(RUNTIME_SOURCE_ROOT, "rev-parse", "HEAD")
    dirty = _git_output(RUNTIME_SOURCE_ROOT, "status", "--porcelain")
    if head != contract["runtime_source_expected_head"]:
        raise ValueError(f"Exact runtime source HEAD mismatch: {head}.")
    if contract.get("runtime_source_requires_clean_worktree") and dirty:
        raise ValueError("Exact runtime source worktree is not clean.")
    mechanism_path = Path(
        contract["mechanism_initialization_hash_artifact"]
    ).expanduser().resolve()
    if sha256_file(mechanism_path) != contract[
        "mechanism_initialization_hash_artifact_sha256"
    ]:
        raise ValueError("Initialization-hash provenance artifact changed.")
    return {
        "root": str(RUNTIME_SOURCE_ROOT),
        "head": head,
        "clean": not bool(dirty),
        "production_source_commit": contract["production_source_commit"],
        "production_source_archive_sha256": contract[
            "production_source_archive_sha256"
        ],
        "mechanism_initialization_hash_artifact": str(mechanism_path),
        "mechanism_initialization_hash_artifact_sha256": sha256_file(
            mechanism_path
        ),
    }


def _model_config(config: Mapping[str, Any]) -> dict[str, Any]:
    model_key = config["lab"]["model_key"]
    model = {**config["model_base"], **config["model_overrides"][model_key]}
    _validate_diode_param_config(model)
    _require_minimizer_config(model)
    return model


def _dataset_signature(
    source_config: Mapping[str, Any],
    *,
    data_root: Path,
    batch_size: int,
) -> dict[str, Any]:
    dataset_key = str(source_config["lab"]["dataset_key"])
    _, dataset_config = _resolve_dataset_config(source_config, dataset_key)
    params = json.loads(json.dumps(dataset_config["params"]))
    params["root"] = str(data_root.expanduser().resolve())
    params["validation_batch_size"] = int(batch_size)
    params["download"] = False
    return {
        "dataset_key": dataset_key,
        "factory": dataset_config["factory"],
        "params": params,
        "input_mode": source_config.get("input_mode", "train"),
    }


def _checked_replay_indices(validation_indices, source_indices, example_count, excluded_source_indices=()):
    """Resolve an ordered cohort without admitting training/test examples or reuse."""
    selected = tuple(validation_indices[:example_count] if source_indices is None else source_indices)
    if len(selected) != example_count or example_count <= 0:
        raise ValueError("Explicit replay indices must match the declared example count.")
    if any(isinstance(v, bool) or not isinstance(v, int) for v in selected):
        raise ValueError("Replay source indices must be integers.")
    if len(set(selected)) != len(selected):
        raise ValueError("Duplicate replay source indices.")
    if not set(selected).issubset(set(validation_indices)):
        raise ValueError("Replay source indices must belong to the validation partition.")
    if set(selected).intersection(excluded_source_indices):
        raise ValueError("Replay cohort overlaps the explicitly excluded selection cohort.")
    return selected


def _build_validation_cohort(
    source_config: Mapping[str, Any],
    *,
    data_root: Path,
    batch_size: int,
    example_count: int,
    source_indices: Sequence[int] | None = None,
    excluded_source_indices: Sequence[int] = (),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if example_count <= 0 or example_count % batch_size:
        raise ValueError("Cohort size must be a positive multiple of batch size.")
    signature = _dataset_signature(
        source_config, data_root=data_root, batch_size=batch_size
    )
    params = dict(signature["params"])
    params["device"] = torch.device("cpu")
    loader_result = _resolve_callable(signature["factory"])(**params).build()
    validation_indices = tuple(int(v) for v in loader_result.validation_indices)
    if example_count > len(validation_indices):
        raise ValueError("Requested cohort exceeds deterministic validation split.")
    selected_indices = _checked_replay_indices(
        validation_indices, source_indices, example_count, excluded_source_indices
    )
    replay_loader = loader_result.validation_loader
    if source_indices is not None:
        if len(replay_loader.dataset) != len(validation_indices):
            raise ValueError("Validation dataset and source-index map differ in length.")
        positions = {index: position for position, index in enumerate(validation_indices)}
        replay_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(replay_loader.dataset, [positions[i] for i in selected_indices]),
            batch_size=batch_size, shuffle=False, num_workers=0,
            collate_fn=replay_loader.collate_fn,
        )
    batches: list[dict[str, Any]] = []
    needed_batches = example_count // batch_size
    class_counts = np.zeros(10, dtype=np.int64)
    for batch_index, (images, labels) in enumerate(replay_loader):
        if batch_index >= needed_batches:
            break
        start = batch_index * batch_size
        stop = start + int(images.shape[0])
        batch_source_indices = selected_indices[start:stop]
        images = images.detach().cpu().contiguous()
        labels = labels.detach().cpu().contiguous()
        if len(batch_source_indices) != batch_size or images.shape[0] != batch_size:
            raise RuntimeError("Unexpected partial batch in fixed replay cohort.")
        class_counts += np.bincount(labels.numpy(), minlength=10)
        batches.append(
            {
                "batch_index": batch_index,
                "images": images,
                "labels": labels,
                "source_indices": batch_source_indices,
                "source_indices_sha256": stable_index_sequence_hash(batch_source_indices),
                "payload_sha256": _batch_payload_sha256(
                    images, labels, batch_source_indices
                ),
            }
        )
    if len(batches) != needed_batches:
        raise RuntimeError(
            f"Expected {needed_batches} cohort batches; found {len(batches)}."
        )
    cohort = {
        "schema": COHORT_SCHEMA,
        "source_split": "MNIST training split, deterministic validation partition",
        "official_test_read": False,
        "split_seed": int(loader_result.split_seed),
        "validation_size": len(validation_indices),
        "validation_indices_sha256": loader_result.validation_indices_hash,
        "cohort_policy": ("explicit ordered validation source indices" if source_indices is not None
                          else "first examples in stored validation order"),
        "cohort_sha256": stable_batch_order_hash(
            batch["source_indices"] for batch in batches
        ),
        "example_count": int(example_count),
        "batch_size": int(batch_size),
        "batch_count": len(batches),
        "class_counts": class_counts.tolist(),
        "dataset_signature": signature,
        "batches": [
            {
                "batch_index": batch["batch_index"],
                "source_indices": list(batch["source_indices"]),
                "source_indices_sha256": batch["source_indices_sha256"],
                "payload_sha256": batch["payload_sha256"],
            }
            for batch in batches
        ],
    }
    return batches, cohort


def _source_run_inventory(
    analysis_config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]]]:
    source_root = Path(
        analysis_config["source_contract"]["source_study_root"]
    ).expanduser().resolve()
    contract = analysis_config["source_contract"]
    inventory: list[dict[str, Any]] = []
    file_hashes: dict[str, dict[str, str]] = {}
    for case in analysis_config["cases"]:
        run_dir = source_root / "final_runs" / case["run_id"]
        errors = validate_run(run_dir)
        if errors:
            raise ValueError(
                f"Invalid source reporting bundle {run_dir}: {'; '.join(errors)}"
            )
        paths = {
            "config.used.json": run_dir / "config.used.json",
            "result.json": run_dir / "result.json",
            "best_model.pt": run_dir / "best_model.pt",
            "weights_best.npz": run_dir / "weights_best.npz",
            "manifest.json": run_dir / "manifest.json",
            "metrics.json": run_dir / "metrics.json",
        }
        observed = {name: sha256_file(path) for name, path in paths.items()}
        for name, expected_key in (
            ("config.used.json", "config_used_sha256"),
            ("result.json", "result_sha256"),
            ("best_model.pt", "best_checkpoint_sha256"),
        ):
            if observed[name] != case[expected_key]:
                raise ValueError(
                    f"Source hash mismatch for {case['architecture']}/{name}."
                )
        manifest = _read_json(paths["manifest.json"])
        if observed["manifest.json"] != case["source_manifest_sha256"]:
            raise ValueError(f"Source manifest hash mismatch for {case['architecture']}.")
        source_git = manifest.get("git", {})
        if source_git.get("commit") != contract["production_source_commit"]:
            raise ValueError("Frozen production source commit mismatch.")
        if source_git.get("source_archive_sha256") != contract[
            "production_source_archive_sha256"
        ]:
            raise ValueError("Frozen production source archive mismatch.")
        result = _read_json(paths["result.json"])
        terminal = result["terminal_metrics"]
        if int(terminal["best_epoch"]) != int(case["best_epoch"]):
            raise ValueError(f"Best epoch mismatch for {case['architecture']}.")
        observed_accuracy = float(terminal["validation"]["best_accuracy"])
        if observed_accuracy != float(case["best_validation_accuracy"]):
            raise ValueError(
                f"Best validation accuracy mismatch for {case['architecture']}."
            )
        source_config = _read_json(paths["config.used.json"])
        dataset_provenance = _read_json(paths["metrics.json"]).get(
            "dataset_provenance", {}
        )
        for config_key, provenance_key in (
            ("train_indices_sha256", "train_indices_sha256"),
            ("validation_indices_sha256", "validation_indices_sha256"),
            ("first_epoch_batch_order_sha256", "first_epoch_batch_order_sha256"),
        ):
            if dataset_provenance.get(provenance_key) != analysis_config[
                "dataset"
            ][config_key]:
                raise ValueError(
                    f"Dataset provenance mismatch for {case['architecture']}/{provenance_key}."
                )
        if source_config.get("training_algorithm") != "BP":
            raise ValueError("Expected BPTT-trained source checkpoint.")
        if source_config.get("batch_state_policy") != "reset_each_batch":
            raise ValueError("Expected reset_each_batch source state policy.")
        model = _model_config(source_config)
        if int(model["num_iterations_inference"]) != int(
            case["inference_iterations"]
        ) or int(model["num_iterations_training"]) != int(
            case["gradient_iterations"]
        ):
            raise ValueError(f"T/K mismatch for {case['architecture']}.")
        file_hashes[case["architecture"]] = observed
        inventory.append(
            {
                **case,
                "run_dir": run_dir,
                "source_config": source_config,
                "source_hashes": observed,
                "weights_best_path": paths["weights_best.npz"],
                "best_checkpoint_path": paths["best_model.pt"],
            }
        )
    return inventory, file_hashes


def _verify_source_hashes(
    inventory: Sequence[Mapping[str, Any]],
    expected: Mapping[str, Mapping[str, str]],
) -> dict[str, dict[str, str]]:
    observed: dict[str, dict[str, str]] = {}
    for case in inventory:
        run_dir = Path(case["run_dir"])
        architecture = str(case["architecture"])
        observed[architecture] = {
            name: sha256_file(run_dir / name)
            for name in expected[architecture]
        }
        if observed[architecture] != expected[architecture]:
            raise RuntimeError(f"Source files changed during replay: {architecture}.")
    return observed


def _build_runtime(
    source_config: Mapping[str, Any],
    *,
    device: torch.device,
    gradient_iterations: int,
    nudging_mode: str = "cost",
    current_scale: str | float | None = "auto",
) -> dict[str, Any]:
    _set_seed(int(source_config["seed"]))
    _reset_name_counters()
    model = _model_config(source_config)
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
        input_mode=source_config.get("input_mode", "train"),
        trainable_amplification=bool(model.get("trainable_amplification", False)),
        amplification_min=model.get("amplification_min", 1.0e-6),
        amplification_max=model.get("amplification_max"),
    )
    energy_fn.set_device(device)
    # Match mnist_train: explicit saved initialization supersedes the model's
    # random draw, including when training ceilings differ from its support.
    if source_config.get("init_checkpoint_path"):
        initializer = Path(source_config["init_checkpoint_path"]).expanduser().resolve()
        expected = source_config.get("initialization", {}).get("checkpoint_sha256")
        sweep_expected = source_config.get("weight_ceiling_sweep", {}).get(
            "initializer_checkpoint_sha256"
        )
        for digest in (expected, sweep_expected):
            if digest is not None and sha256_file(initializer) != digest:
                raise ValueError(f"Initializer checkpoint hash mismatch: {initializer}.")
        energy_fn.load(initializer)
    network = Network(energy_fn)
    free_layers = network.free_layers()
    output_layer = energy_fn.layers()[-1]
    output_dim = int(output_layer.shape[0])
    if output_dim == 10:
        cost_fn = SquaredError(output_layer)
    elif output_dim == 20:
        cost_fn = SquaredErrorPairedOutputs(output_layer, num_classes=10)
    else:
        raise ValueError(f"Unsupported output dimension {output_dim}.")
    mode = source_config["energy_minimizer"]["mode"]
    inference_iterations = int(model["num_iterations_inference"])
    minimizer_inference = _build_tracking_minimizer(
        energy_fn,
        free_layers,
        model,
        mode,
        num_iterations=inference_iterations,
        voltage_amp=energy_fn._voltage_amp,
        current_amp=energy_fn._current_amp,
        adaptive_equilibrium=False,
    )
    minimizer_gradient = _build_tracking_minimizer(
        energy_fn,
        free_layers,
        model,
        mode,
        num_iterations=int(gradient_iterations),
        voltage_amp=energy_fn._voltage_amp,
        current_amp=energy_fn._current_amp,
        adaptive_equilibrium=False,
    )
    if nudging_mode not in {"cost", "current"}:
        raise ValueError(f"Unsupported nudging mode {nudging_mode!r}.")
    augmented_fn = AugmentedFunction(
        energy_fn,
        cost_fn,
        nudging_mode=nudging_mode,
        current_scale=current_scale,
    )
    minimizer_augmented = _build_tracking_minimizer(
        augmented_fn,
        free_layers,
        model,
        mode,
        num_iterations=int(gradient_iterations),
        voltage_amp=energy_fn._voltage_amp,
        current_amp=energy_fn._current_amp,
        adaptive_equilibrium=False,
    )
    parameters = list(energy_fn.params())
    parameter_names = [str(parameter.name).strip() for parameter in parameters]
    configured_order = source_config.get("parameter_order")
    if configured_order is not None and list(configured_order) != parameter_names:
        raise ValueError("Runtime parameter order differs from source config.")
    weight_indices = [
        index
        for index, parameter in enumerate(parameters)
        if parameter.__class__.__name__ in WEIGHT_TYPES
    ]
    if not weight_indices or any(
        parameters[index].__class__.__name__ not in WEIGHT_TYPES
        for index in weight_indices
    ):
        raise RuntimeError("Weight-only inclusion contract failed.")
    if any(
        parameters[index].__class__.__name__ == "Bias" for index in weight_indices
    ):
        raise RuntimeError("Bias leaked into weight-only gradient selection.")
    return {
        "energy_fn": energy_fn,
        "network": network,
        "free_layers": free_layers,
        "cost_fn": cost_fn,
        "parameters": parameters,
        "weight_indices": weight_indices,
        "weight_names": [parameter_names[index] for index in weight_indices],
        "weight_types": [
            parameters[index].__class__.__name__ for index in weight_indices
        ],
        "inference_iterations": inference_iterations,
        "gradient_iterations": int(gradient_iterations),
        "minimizer_inference": minimizer_inference,
        "minimizer_gradient": minimizer_gradient,
        "minimizer_augmented": minimizer_augmented,
        "augmented_fn": augmented_fn,
        "nudging_mode": nudging_mode,
        "nudging_current_scale": float(augmented_fn._current_scale),
        "backprop": Backprop(
            parameters, free_layers, cost_fn, minimizer_gradient
        ),
        "energy_param_updaters": [
            ParamUpdater(parameter, energy_fn) for parameter in parameters
        ],
        "model_config": model,
    }


def _clone_states(layers: Sequence[Any]) -> list[torch.Tensor]:
    return [layer.state.detach().clone() for layer in layers]


def _restore_states(layers: Sequence[Any], states: Sequence[torch.Tensor]) -> None:
    if len(layers) != len(states):
        raise ValueError("Layer/state count mismatch.")
    for layer, state in zip(layers, states, strict=True):
        layer.state = state.detach().clone()


def _finite(value: torch.Tensor, label: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"Non-finite tensor in {label}.")


def _energy_gradients(runtime: Mapping[str, Any]) -> list[torch.Tensor]:
    gradients = []
    for parameter, updater in zip(
        runtime["parameters"], runtime["energy_param_updaters"], strict=True
    ):
        gradient = updater.grad().detach().clone()
        _finite(gradient, f"energy gradient {str(parameter.name).strip()}")
        gradients.append(gradient)
    return gradients


def _weight_gradient_dict(
    runtime: Mapping[str, Any], gradients: Sequence[torch.Tensor]
) -> dict[str, torch.Tensor]:
    if len(gradients) != len(runtime["parameters"]):
        raise ValueError("Gradient/parameter count mismatch.")
    result = {}
    for index in runtime["weight_indices"]:
        parameter = runtime["parameters"][index]
        name = str(parameter.name).strip()
        gradient = gradients[index].detach().cpu().to(torch.float64).contiguous()
        _finite(gradient, f"selected weight gradient {name}")
        result[name] = gradient
    return result


def _phase_residual(runtime: Mapping[str, Any]) -> dict[str, Any]:
    squared_sum = 0.0
    element_count = 0
    max_abs = 0.0
    active_count = 0
    constrained_count = 0
    for layer in runtime["free_layers"]:
        gradient = runtime["augmented_fn"].grad_layer_fn(layer)().detach()
        state = layer.state.detach()
        _finite(gradient, f"phase residual {layer.name}")
        projected = gradient.clone()
        if getattr(layer, "non_linearity", None) == "perfect_diode":
            split = int(state.shape[1]) // 2
            excitatory_state = state[:, :split]
            inhibitory_state = state[:, split:]
            excitatory_grad = projected[:, :split]
            inhibitory_grad = projected[:, split:]
            excitatory_active = excitatory_state <= STATE_BOUND_TOLERANCE
            inhibitory_active = inhibitory_state >= -STATE_BOUND_TOLERANCE
            excitatory_grad[excitatory_active & (excitatory_grad >= 0.0)] = 0.0
            inhibitory_grad[inhibitory_active & (inhibitory_grad <= 0.0)] = 0.0
            active_count += int(excitatory_active.sum() + inhibitory_active.sum())
            constrained_count += int(state.numel())
        squared_sum += float(projected.to(torch.float64).square().sum().cpu())
        element_count += int(projected.numel())
        max_abs = max(max_abs, float(projected.abs().max().cpu()))
    return {
        "projected_kkt_rms": math.sqrt(squared_sum / max(element_count, 1)),
        "projected_kkt_max_abs": max_abs,
        "active_fraction": (
            active_count / constrained_count if constrained_count else None
        ),
    }


def _state_delta_metrics(
    layers: Sequence[Any], reference: Sequence[torch.Tensor]
) -> dict[str, float]:
    delta_sq = 0.0
    reference_sq = 0.0
    active_transitions = 0
    constrained_count = 0
    for layer, initial in zip(layers, reference, strict=True):
        current = layer.state.detach()
        delta_sq += float(
            (current.to(torch.float64) - initial.to(torch.float64))
            .square()
            .sum()
            .cpu()
        )
        reference_sq += float(initial.to(torch.float64).square().sum().cpu())
        if getattr(layer, "non_linearity", None) == "perfect_diode":
            split = int(current.shape[1]) // 2
            current_active = torch.cat(
                (
                    current[:, :split] <= STATE_BOUND_TOLERANCE,
                    current[:, split:] >= -STATE_BOUND_TOLERANCE,
                ),
                dim=1,
            )
            initial_active = torch.cat(
                (
                    initial[:, :split] <= STATE_BOUND_TOLERANCE,
                    initial[:, split:] >= -STATE_BOUND_TOLERANCE,
                ),
                dim=1,
            )
            active_transitions += int((current_active != initial_active).sum())
            constrained_count += int(current.numel())
    delta_l2 = math.sqrt(delta_sq)
    return {
        "state_displacement_l2": delta_l2,
        "state_relative_displacement": (
            delta_l2 / math.sqrt(reference_sq)
            if reference_sq > NORM_EPSILON
            else None
        ),
        "active_set_transition_fraction": (
            active_transitions / constrained_count if constrained_count else 0.0
        ),
    }


def _symmetry_defect(
    free: Sequence[torch.Tensor],
    negative: Sequence[torch.Tensor],
    positive: Sequence[torch.Tensor],
) -> float | None:
    numerator_sq = 0.0
    negative_sq = 0.0
    positive_sq = 0.0
    for zero, minus, plus in zip(free, negative, positive, strict=True):
        zero64 = zero.detach().to(device="cpu", dtype=torch.float64)
        dminus = minus.detach().to(device="cpu", dtype=torch.float64) - zero64
        dplus = plus.detach().to(device="cpu", dtype=torch.float64) - zero64
        numerator_sq += float((dplus + dminus).square().sum())
        negative_sq += float(dminus.square().sum())
        positive_sq += float(dplus.square().sum())
    denominator = math.sqrt(negative_sq) + math.sqrt(positive_sq)
    return math.sqrt(numerator_sq) / denominator if denominator > NORM_EPSILON else None


def _vector_metrics(
    reference: torch.Tensor,
    estimate: torch.Tensor,
    *,
    zero_epsilon: float,
) -> dict[str, Any]:
    ref = reference.detach().cpu().to(torch.float64).reshape(-1)
    est = estimate.detach().cpu().to(torch.float64).reshape(-1)
    if ref.shape != est.shape:
        raise ValueError(f"Gradient shape mismatch: {ref.shape} vs {est.shape}.")
    _finite(ref, "reference gradient")
    _finite(est, "estimated gradient")
    ref_norm = float(torch.linalg.vector_norm(ref))
    est_norm = float(torch.linalg.vector_norm(est))
    dot = float(torch.dot(ref, est))
    denominator = ref_norm * est_norm
    cosine = dot / denominator if denominator > NORM_EPSILON else None
    difference = float(torch.linalg.vector_norm(est - ref))
    return {
        "element_count": int(ref.numel()),
        "dot_product": dot,
        "bptt_l2": ref_norm,
        "eqprop_l2": est_norm,
        "cosine": cosine,
        "relative_l2_error": (
            difference / ref_norm if ref_norm > NORM_EPSILON else None
        ),
        "norm_ratio_eqprop_over_bptt": (
            est_norm / ref_norm if ref_norm > NORM_EPSILON else None
        ),
        "bptt_rms": (
            ref_norm / math.sqrt(ref.numel()) if ref.numel() else 0.0
        ),
        "eqprop_rms": (
            est_norm / math.sqrt(est.numel()) if est.numel() else 0.0
        ),
        "bptt_zero_fraction": float((ref.abs() <= zero_epsilon).double().mean()),
        "eqprop_zero_fraction": float((est.abs() <= zero_epsilon).double().mean()),
        "bptt_exact_zero_fraction": float((ref == 0.0).double().mean()),
        "eqprop_exact_zero_fraction": float((est == 0.0).double().mean()),
    }


def _run_gradient_phases(
    runtime: Mapping[str, Any],
    *,
    free_states: Sequence[torch.Tensor],
    betas: Sequence[float],
    zero_epsilon: float,
    zero_endpoint_relative_tolerance: float,
) -> tuple[
    dict[str, torch.Tensor],
    dict[float, dict[str, dict[str, torch.Tensor]]],
    list[dict[str, Any]],
    dict[float, dict[str, str]],
]:
    free_hash = _tensor_sequence_sha256(
        ((str(layer.name), state) for layer, state in zip(runtime["free_layers"], free_states, strict=True)),
        schema=b"drn-ordered-free-layer-state/v1",
    )
    _restore_states(runtime["free_layers"], free_states)
    if _layer_state_sha256(runtime["free_layers"]) != free_hash:
        raise RuntimeError("BPTT did not start from the common free state.")
    bptt_all = runtime["backprop"].compute_gradient()
    if len(bptt_all) != len(runtime["parameters"]):
        raise RuntimeError("BPTT gradient/parameter count mismatch.")
    for gradient in bptt_all:
        _finite(gradient, "BPTT gradient")
    bptt = _weight_gradient_dict(runtime, bptt_all)
    zero_energy_bptt = _energy_gradients(runtime)
    bptt_zero_states = _clone_states(runtime["free_layers"])
    bptt_zero_endpoint_hash = _layer_state_sha256(runtime["free_layers"])

    # Canonical one-sided EqProp includes an explicit beta=0 K-step phase.  It
    # must be identical to BPTT's forward K-step endpoint, but we execute and
    # compare it once per batch so that an implementation drift fails closed.
    _restore_states(runtime["free_layers"], free_states)
    zero_start_hash = _layer_state_sha256(runtime["free_layers"])
    runtime["augmented_fn"].prepare_nudging()
    runtime["augmented_fn"].nudging = 0.0
    runtime["minimizer_augmented"].compute_equilibrium()
    zero_energy = _energy_gradients(runtime)
    zero_endpoint_hash = _layer_state_sha256(runtime["free_layers"])
    zero_gradient_max_abs_delta = max(
        float((left - right).abs().max().cpu())
        for left, right in zip(zero_energy_bptt, zero_energy, strict=True)
    )
    gradient_difference_sq = sum(
        float(
            (left.detach().to(torch.float64) - right.detach().to(torch.float64))
            .square()
            .sum()
            .cpu()
        )
        for left, right in zip(zero_energy_bptt, zero_energy, strict=True)
    )
    gradient_reference_sq = sum(
        float(left.detach().to(torch.float64).square().sum().cpu())
        for left in zero_energy_bptt
    )
    zero_gradient_relative_l2_delta = math.sqrt(gradient_difference_sq) / max(
        math.sqrt(gradient_reference_sq), NORM_EPSILON
    )
    state_difference_sq = sum(
        float(
            (left.detach().to(torch.float64) - right.detach().to(torch.float64))
            .square()
            .sum()
            .cpu()
        )
        for left, right in zip(
            bptt_zero_states,
            _clone_states(runtime["free_layers"]),
            strict=True,
        )
    )
    state_reference_sq = sum(
        float(left.detach().to(torch.float64).square().sum().cpu())
        for left in bptt_zero_states
    )
    zero_state_relative_l2_delta = math.sqrt(state_difference_sq) / max(
        math.sqrt(state_reference_sq), NORM_EPSILON
    )
    zero_endpoint_equivalent = bool(
        zero_gradient_relative_l2_delta <= zero_endpoint_relative_tolerance
        and zero_state_relative_l2_delta <= zero_endpoint_relative_tolerance
    )
    if not zero_endpoint_equivalent:
        raise RuntimeError(
            "The explicit EqProp beta=0 endpoint exceeds the BPTT K-step "
            "equivalence gate: "
            f"gradient relative L2={zero_gradient_relative_l2_delta}, "
            f"state relative L2={zero_state_relative_l2_delta}, "
            f"tolerance={zero_endpoint_relative_tolerance}."
        )

    estimates: dict[float, dict[str, dict[str, torch.Tensor]]] = {}
    phase_rows: list[dict[str, Any]] = [
        {
            "beta": 0.0,
            "phase": "zero",
            "nudging": 0.0,
            "common_free_state_sha256": free_hash,
            "phase_start_state_sha256": zero_start_hash,
            "phase_end_state_sha256": zero_endpoint_hash,
            "bptt_zero_endpoint_sha256": bptt_zero_endpoint_hash,
            "bptt_zero_gradient_max_abs_delta": zero_gradient_max_abs_delta,
            "bptt_zero_gradient_relative_l2_delta": zero_gradient_relative_l2_delta,
            "bptt_zero_state_relative_l2_delta": zero_state_relative_l2_delta,
            "bptt_zero_endpoint_relative_l2_tolerance": zero_endpoint_relative_tolerance,
            "bptt_zero_endpoint_exact_match": (
                bptt_zero_endpoint_hash == zero_endpoint_hash
                and zero_gradient_max_abs_delta == 0.0
            ),
            "bptt_zero_endpoint_equivalent": zero_endpoint_equivalent,
            "outcome": "ok",
            **_state_delta_metrics(runtime["free_layers"], free_states),
            **_phase_residual(runtime),
            "state_symmetry_defect": None,
            "zero_epsilon": float(zero_epsilon),
        }
    ]
    variant_outcomes: dict[float, dict[str, str]] = {}
    for beta in betas:
        endpoints: dict[str, list[torch.Tensor]] = {}
        endpoint_states: dict[str, list[torch.Tensor]] = {}
        phase_row_indices: list[int] = []
        for label, nudging in (("negative", -float(beta)), ("positive", float(beta))):
            _restore_states(runtime["free_layers"], free_states)
            start_hash = _layer_state_sha256(runtime["free_layers"])
            if start_hash != free_hash:
                raise RuntimeError(
                    f"EqProp {label} phase did not start from common free state."
                )
            runtime["augmented_fn"].prepare_nudging()
            runtime["augmented_fn"].nudging = nudging
            row: dict[str, Any] = {
                "beta": float(beta),
                "phase": label,
                "nudging": nudging,
                "common_free_state_sha256": free_hash,
                "phase_start_state_sha256": start_hash,
            }
            try:
                runtime["minimizer_augmented"].compute_equilibrium()
                candidate_gradients = _energy_gradients(runtime)
                candidate_states = _clone_states(runtime["free_layers"])
                state_finite = all(
                    bool(torch.isfinite(state).all()) for state in candidate_states
                )
                if not state_finite:
                    raise FloatingPointError(
                        f"Non-finite layer state in EqProp {label} phase."
                    )
                row.update(
                    {
                        "phase_end_state_sha256": _layer_state_sha256(
                            runtime["free_layers"]
                        ),
                        "phase_state_all_finite": True,
                        "outcome": "ok",
                        **_state_delta_metrics(
                            runtime["free_layers"], free_states
                        ),
                        **_phase_residual(runtime),
                    }
                )
                endpoints[label] = candidate_gradients
                endpoint_states[label] = candidate_states
            except FloatingPointError as error:
                row.update(
                    {
                        "phase_end_state_sha256": _layer_state_sha256(
                            runtime["free_layers"]
                        ),
                        "phase_state_all_finite": all(
                            bool(torch.isfinite(layer.state).all())
                            for layer in runtime["free_layers"]
                        ),
                        "outcome": f"{label}_phase_unstable",
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                    }
                )
            phase_rows.append(row)
            phase_row_indices.append(len(phase_rows) - 1)

        available_variants = []
        if "negative" in endpoints and "positive" in endpoints:
            available_variants.append("centered")
        if "positive" in endpoints:
            available_variants.append("positive")
        if "negative" in endpoints:
            available_variants.append("negative")
        estimates[float(beta)] = {variant: {} for variant in available_variants}
        for index in runtime["weight_indices"]:
            name = str(runtime["parameters"][index].name).strip()
            gzero = zero_energy[index].detach().cpu().to(torch.float64)
            if "positive" in endpoints:
                gplus = endpoints["positive"][index].detach().cpu().to(torch.float64)
                estimates[float(beta)]["positive"][name] = (
                    gplus - gzero
                ) / float(beta)
            if "negative" in endpoints:
                gminus = endpoints["negative"][index].detach().cpu().to(torch.float64)
                estimates[float(beta)]["negative"][name] = (
                    gzero - gminus
                ) / float(beta)
            if "negative" in endpoints and "positive" in endpoints:
                estimates[float(beta)]["centered"][name] = (
                    gplus - gminus
                ) / (2.0 * float(beta))
            for variant in available_variants:
                _finite(
                    estimates[float(beta)][variant][name],
                    f"EqProp {variant} {name} beta={beta}",
                )
        symmetry = (
            _symmetry_defect(
                free_states,
                endpoint_states["negative"],
                endpoint_states["positive"],
            )
            if "negative" in endpoint_states and "positive" in endpoint_states
            else None
        )
        for row_index in phase_row_indices:
            row = phase_rows[row_index]
            row["state_symmetry_defect"] = symmetry
            row["zero_epsilon"] = float(zero_epsilon)
        negative_ok = "negative" in endpoints
        positive_ok = "positive" in endpoints
        centered_outcome = (
            "ok"
            if negative_ok and positive_ok
            else (
                "both_nudged_phases_unstable"
                if not negative_ok and not positive_ok
                else (
                    "negative_phase_unstable"
                    if not negative_ok
                    else "positive_phase_unstable"
                )
            )
        )
        variant_outcomes[float(beta)] = {
            "centered": centered_outcome,
            "positive": "ok" if positive_ok else "positive_phase_unstable",
            "negative": "ok" if negative_ok else "negative_phase_unstable",
        }
    return bptt, estimates, phase_rows, variant_outcomes


class _GradientAccumulator:
    def __init__(self) -> None:
        self._sums: dict[tuple[Any, ...], torch.Tensor] = {}
        self._examples: defaultdict[tuple[Any, ...], int] = defaultdict(int)
        self._batch_cosines: defaultdict[tuple[Any, ...], list[float]] = defaultdict(list)
        self._batch_count: defaultdict[tuple[Any, ...], int] = defaultdict(int)

    def add(
        self,
        key: tuple[Any, ...],
        gradient: torch.Tensor,
        *,
        batch_size: int,
        batch_cosine: float | None = None,
    ) -> None:
        value = gradient.detach().cpu().to(torch.float64).contiguous()
        weighted = value * int(batch_size)
        if key not in self._sums:
            self._sums[key] = weighted.clone()
        else:
            self._sums[key] += weighted
        self._examples[key] += int(batch_size)
        self._batch_count[key] += 1
        if batch_cosine is not None:
            self._batch_cosines[key].append(float(batch_cosine))

    def mean(self, key: tuple[Any, ...]) -> torch.Tensor:
        return self._sums[key] / self._examples[key]

    def examples(self, key: tuple[Any, ...]) -> int:
        return self._examples[key]

    def batch_count(self, key: tuple[Any, ...]) -> int:
        return self._batch_count[key]

    def batch_cosines(self, key: tuple[Any, ...]) -> list[float]:
        return self._batch_cosines[key]

    def keys(self) -> Iterable[tuple[Any, ...]]:
        return self._sums.keys()


def _quantile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def _verify_npz_checkpoint(runtime: Mapping[str, Any], path: Path) -> None:
    with np.load(path, allow_pickle=False) as archive:
        names = [str(value) for value in archive["param_names"].tolist()]
        types = [str(value) for value in archive["param_types"].tolist()]
        runtime_names = [str(parameter.name).strip() for parameter in runtime["parameters"]]
        runtime_types = [
            parameter.__class__.__name__ for parameter in runtime["parameters"]
        ]
        if names != runtime_names or types != runtime_types:
            raise ValueError("PT and NPZ checkpoint schemas differ.")
        for parameter, name in zip(runtime["parameters"], names, strict=True):
            value = parameter.state.detach().cpu().numpy()
            if not np.array_equal(value, archive[name]):
                raise ValueError(f"PT and NPZ tensors differ for {name}.")


def _parameter_summary(
    accumulator: _GradientAccumulator,
    *,
    architectures: Sequence[str],
    roles: Sequence[str],
    phases: Sequence[str],
    betas: Sequence[float],
    variants: Sequence[str],
    weight_names_by_architecture: Mapping[str, Sequence[str]],
    weight_types_by_architecture: Mapping[str, Mapping[str, str]],
    zero_epsilon: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for architecture in architectures:
        for role in roles:
            for phase in phases:
                for beta in betas:
                    for variant in variants:
                        for name in weight_names_by_architecture[architecture]:
                            bptt_key = (architecture, role, phase, "bptt", None, None, name)
                            ep_key = (
                                architecture,
                                role,
                                phase,
                                "eqprop",
                                float(beta),
                                variant,
                                name,
                            )
                            if bptt_key not in accumulator._sums or ep_key not in accumulator._sums:
                                continue
                            metrics = _vector_metrics(
                                accumulator.mean(bptt_key),
                                accumulator.mean(ep_key),
                                zero_epsilon=zero_epsilon,
                            )
                            cosines = accumulator.batch_cosines(ep_key)
                            expected_examples = accumulator.examples(bptt_key)
                            observed_examples = accumulator.examples(ep_key)
                            coverage_complete = observed_examples == expected_examples
                            rows.append(
                                {
                                    "schema": SCHEMA,
                                    "architecture": architecture,
                                    "checkpoint_role": role,
                                    "phase_length": phase,
                                    "gradient_iterations": (
                                        64 if phase == "k64_sentinel" else None
                                    ),
                                    "beta": float(beta),
                                    "eqprop_variant": variant,
                                    "parameter_name": name,
                                    "parameter_type": weight_types_by_architecture[
                                        architecture
                                    ][name],
                                    "bias_excluded": True,
                                    "example_count": observed_examples,
                                    "expected_example_count": expected_examples,
                                    "coverage_complete": coverage_complete,
                                    "batch_count": accumulator.batch_count(ep_key),
                                    "outcome": (
                                        "incomplete_phase_coverage"
                                        if not coverage_complete
                                        else (
                                            "ok"
                                            if metrics["cosine"] is not None
                                            else "dead_gradient"
                                        )
                                    ),
                                    **metrics,
                                    "batch_cosine_median": _quantile(cosines, 0.5),
                                    "batch_cosine_q10": _quantile(cosines, 0.1),
                                    "batch_cosine_q90": _quantile(cosines, 0.9),
                                }
                            )
    return rows


def _select_betas(
    parameter_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: defaultdict[tuple[str, str, float], list[Mapping[str, Any]]] = defaultdict(list)
    for row in parameter_rows:
        if row["phase_length"] == "operating" and row["eqprop_variant"] == "centered":
            grouped[
                (
                    str(row["architecture"]),
                    str(row["checkpoint_role"]),
                    float(row["beta"]),
                )
            ].append(row)
    candidates: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for (architecture, role, beta), rows in grouped.items():
        cosines = [row["cosine"] for row in rows]
        coverage_complete = all(bool(row["coverage_complete"]) for row in rows)
        if not coverage_complete or any(value is None for value in cosines):
            worst = None
            mean = None
        else:
            numeric = [float(value) for value in cosines]
            worst = min(numeric)
            mean = float(np.mean(numeric))
        candidates[(architecture, role)].append(
            {
                "architecture": architecture,
                "checkpoint_role": role,
                "beta": beta,
                "worst_layer_centered_cosine": worst,
                "mean_layer_centered_cosine": mean,
                "worst_layer": (
                    min(rows, key=lambda row: float(row["cosine"]))[
                        "parameter_name"
                    ]
                    if worst is not None
                    else None
                ),
                "layer_count": len(rows),
                "selectable": bool(coverage_complete and worst is not None),
                "eligibility_outcome": (
                    "ok"
                    if coverage_complete and worst is not None
                    else (
                        "incomplete_phase_coverage"
                        if not coverage_complete
                        else "dead_gradient"
                    )
                ),
            }
        )
    output = []
    for key in sorted(candidates):
        valid = [
            row
            for row in candidates[key]
            if row["selectable"]
        ]
        selected = (
            sorted(
                valid,
                key=lambda row: (
                    -float(row["worst_layer_centered_cosine"]),
                    float(row["beta"]),
                ),
            )[0]
            if valid
            else None
        )
        for row in sorted(candidates[key], key=lambda value: float(value["beta"])):
            output.append(
                {
                    **row,
                    "selected": bool(
                        selected is not None
                        and float(row["beta"]) == float(selected["beta"])
                    ),
                    "selection_status": (
                        "selected" if selected is not None else "no_selectable_beta"
                    ),
                    "selection_rule": "maximize worst layer centered cohort-mean cosine; exact ties choose smaller beta",
                }
            )
    return output


def _sentinel_summary(
    accumulator: _GradientAccumulator,
    *,
    architectures: Sequence[str],
    roles: Sequence[str],
    betas: Sequence[float],
    variants: Sequence[str],
    weight_names_by_architecture: Mapping[str, Sequence[str]],
    zero_epsilon: float,
    gates: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    estimators: list[tuple[str, float | None, str | None]] = [("bptt", None, None)]
    estimators.extend(
        ("eqprop", float(beta), variant)
        for beta in betas
        for variant in variants
    )
    for architecture in architectures:
        for role in roles:
            for estimator, beta, variant in estimators:
                for name in weight_names_by_architecture[architecture]:
                    op_key = (
                        architecture,
                        role,
                        "operating_sentinel_cohort",
                        estimator,
                        beta,
                        variant,
                        name,
                    )
                    ref_key = (
                        architecture,
                        role,
                        "k64_sentinel",
                        estimator,
                        beta,
                        variant,
                        name,
                    )
                    bptt_op_key = (
                        architecture,
                        role,
                        "operating_sentinel_cohort",
                        "bptt",
                        None,
                        None,
                        name,
                    )
                    expected_batches = accumulator.batch_count(bptt_op_key)
                    op_complete = (
                        op_key in accumulator._sums
                        and accumulator.batch_count(op_key) == expected_batches
                    )
                    ref_complete = (
                        ref_key in accumulator._sums
                        and accumulator.batch_count(ref_key) == expected_batches
                    )
                    if not op_complete or not ref_complete:
                        rows.append(
                            {
                                "architecture": architecture,
                                "checkpoint_role": role,
                                "estimator": estimator,
                                "beta": beta,
                                "eqprop_variant": variant,
                                "parameter_name": name,
                                "operating_vs_k64_cosine": None,
                                "operating_vs_k64_relative_l2_error": None,
                                "relative_l2_norm_delta": None,
                                "absolute_zero_fraction_delta": None,
                                "expected_batch_count": expected_batches,
                                "operating_batch_count": (
                                    accumulator.batch_count(op_key)
                                    if op_key in accumulator._sums
                                    else 0
                                ),
                                "k64_batch_count": (
                                    accumulator.batch_count(ref_key)
                                    if ref_key in accumulator._sums
                                    else 0
                                ),
                                "cosine_minimum": gates["cosine_minimum"],
                                "relative_l2_norm_delta_maximum": gates[
                                    "relative_l2_norm_delta_maximum"
                                ],
                                "absolute_zero_fraction_delta_maximum": gates[
                                    "absolute_zero_fraction_delta_maximum"
                                ],
                                "sentinel_pass": False,
                                "interpretation": (
                                    "reference_unresolved"
                                    if not ref_complete
                                    else "phase_iteration_mismatch"
                                ),
                            }
                        )
                        continue
                    op = accumulator.mean(op_key)
                    ref = accumulator.mean(ref_key)
                    comparison = _vector_metrics(
                        ref, op, zero_epsilon=zero_epsilon
                    )
                    op_norm = float(torch.linalg.vector_norm(op.reshape(-1)))
                    ref_norm = float(torch.linalg.vector_norm(ref.reshape(-1)))
                    norm_scale = max(op_norm, ref_norm, NORM_EPSILON)
                    relative_norm_delta = abs(op_norm - ref_norm) / norm_scale
                    op_zero = float((op.abs() <= zero_epsilon).double().mean())
                    ref_zero = float((ref.abs() <= zero_epsilon).double().mean())
                    zero_delta = abs(op_zero - ref_zero)
                    passed = bool(
                        comparison["cosine"] is not None
                        and float(comparison["cosine"])
                        >= float(gates["cosine_minimum"])
                        and relative_norm_delta is not None
                        and relative_norm_delta
                        <= float(gates["relative_l2_norm_delta_maximum"])
                        and zero_delta
                        <= float(gates["absolute_zero_fraction_delta_maximum"])
                    )
                    rows.append(
                        {
                            "architecture": architecture,
                            "checkpoint_role": role,
                            "estimator": estimator,
                            "beta": beta,
                            "eqprop_variant": variant,
                            "parameter_name": name,
                            "expected_batch_count": expected_batches,
                            "operating_batch_count": accumulator.batch_count(op_key),
                            "k64_batch_count": accumulator.batch_count(ref_key),
                            "operating_vs_k64_cosine": comparison["cosine"],
                            "operating_vs_k64_relative_l2_error": comparison[
                                "relative_l2_error"
                            ],
                            "relative_l2_norm_delta": relative_norm_delta,
                            "absolute_zero_fraction_delta": zero_delta,
                            "cosine_minimum": gates["cosine_minimum"],
                            "relative_l2_norm_delta_maximum": gates[
                                "relative_l2_norm_delta_maximum"
                            ],
                            "absolute_zero_fraction_delta_maximum": gates[
                                "absolute_zero_fraction_delta_maximum"
                            ],
                            "sentinel_pass": passed,
                            "interpretation": (
                                "phase_length_stable"
                                if passed
                                else "finite_iteration_limited"
                            ),
                        }
                    )
    return rows


def _checkpoint_evolution(
    accumulator: _GradientAccumulator,
    *,
    architectures: Sequence[str],
    betas: Sequence[float],
    variants: Sequence[str],
    weight_names_by_architecture: Mapping[str, Sequence[str]],
    zero_epsilon: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    estimators: list[tuple[str, float | None, str | None]] = [("bptt", None, None)]
    estimators.extend(
        ("eqprop", float(beta), variant)
        for beta in betas
        for variant in variants
    )
    for architecture in architectures:
        for estimator, beta, variant in estimators:
            for name in weight_names_by_architecture[architecture]:
                init_key = (
                    architecture,
                    "reconstructed_initialization",
                    "operating",
                    estimator,
                    beta,
                    variant,
                    name,
                )
                best_key = (
                    architecture,
                    "best_validation",
                    "operating",
                    estimator,
                    beta,
                    variant,
                    name,
                )
                if init_key not in accumulator._sums or best_key not in accumulator._sums:
                    continue
                metrics = _vector_metrics(
                    accumulator.mean(init_key),
                    accumulator.mean(best_key),
                    zero_epsilon=zero_epsilon,
                )
                rows.append(
                    {
                        "architecture": architecture,
                        "estimator": estimator,
                        "beta": beta,
                        "eqprop_variant": variant,
                        "parameter_name": name,
                        "best_vs_initial_cosine": metrics["cosine"],
                        "best_vs_initial_relative_l2_change": metrics[
                            "relative_l2_error"
                        ],
                        "best_over_initial_gradient_norm": metrics[
                            "norm_ratio_eqprop_over_bptt"
                        ],
                        "initial_gradient_l2": metrics["bptt_l2"],
                        "best_gradient_l2": metrics["eqprop_l2"],
                        "initial_zero_fraction": metrics["bptt_zero_fraction"],
                        "best_zero_fraction": metrics["eqprop_zero_fraction"],
                    }
                )
    return rows


def _save_mean_gradients(
    path: Path, accumulator: _GradientAccumulator
) -> dict[str, str]:
    arrays: dict[str, np.ndarray] = {}
    key_map: dict[str, str] = {}
    for index, key in enumerate(sorted(accumulator.keys(), key=str)):
        array_key = f"gradient_{index:05d}"
        arrays[array_key] = accumulator.mean(key).numpy()
        key_map[array_key] = json.dumps(key, separators=(",", ":"))
    arrays["key_map_json"] = np.asarray(
        json.dumps(key_map, sort_keys=True, allow_nan=False)
    )
    np.savez_compressed(path, **arrays)
    return key_map


def _plot_cosines(
    path: Path,
    parameter_rows: Sequence[Mapping[str, Any]],
    architectures: Sequence[str],
    roles: Sequence[str],
) -> None:
    fig, axes = plt.subplots(
        len(architectures),
        len(roles),
        figsize=(12, 11),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for row_index, architecture in enumerate(architectures):
        for column_index, role in enumerate(roles):
            axis = axes[row_index][column_index]
            subset = [
                row
                for row in parameter_rows
                if row["architecture"] == architecture
                and row["checkpoint_role"] == role
                and row["phase_length"] == "operating"
                and row["eqprop_variant"] == "centered"
                and row["cosine"] is not None
            ]
            names = sorted({str(row["parameter_name"]) for row in subset})
            for name in names:
                values = sorted(
                    (row for row in subset if row["parameter_name"] == name),
                    key=lambda row: float(row["beta"]),
                )
                axis.plot(
                    [float(row["beta"]) for row in values],
                    [float(row["cosine"]) for row in values],
                    marker="o",
                    linewidth=1.6,
                    label=name,
                )
            axis.axhline(0.0, color="0.65", linewidth=0.8)
            axis.set_xscale("log")
            axis.set_ylim(-1.02, 1.02)
            axis.grid(True, alpha=0.25)
            axis.set_title(f"{architecture.upper()} — {role.replace('_', ' ')}")
            if column_index == 0:
                axis.set_ylabel("EqProp–BPTT cosine")
            if row_index == len(architectures) - 1:
                axis.set_xlabel(r"EqProp nudging $\beta$")
            axis.legend(fontsize=8)
    fig.suptitle("Centered EqProp vs BPTT weight gradients (biases excluded)")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def _write_report(
    path: Path,
    *,
    analysis_config: Mapping[str, Any],
    beta_rows: Sequence[Mapping[str, Any]],
    sentinel_rows: Sequence[Mapping[str, Any]],
    parameter_rows: Sequence[Mapping[str, Any]],
    smoke: bool,
) -> None:
    selected = [row for row in beta_rows if row["selected"]]
    selection_table = []
    selection_keys = sorted(
        {
            (str(row["architecture"]), str(row["checkpoint_role"]))
            for row in beta_rows
        }
    )
    for architecture, role in selection_keys:
        matches = [
            row
            for row in selected
            if row["architecture"] == architecture
            and row["checkpoint_role"] == role
        ]
        if matches:
            row = matches[0]
            selection_table.append(
                (
                    architecture.upper(),
                    role.replace("_", " "),
                    f"{float(row['beta']):g}",
                    f"{float(row['worst_layer_centered_cosine']):.6f}",
                    row["worst_layer"],
                )
            )
        else:
            selection_table.append(
                (
                    architecture.upper(),
                    role.replace("_", " "),
                    "--",
                    "--",
                    "no centered beta with complete viable layer coverage",
                )
            )
    selected_lookup = {
        (str(row["architecture"]), str(row["checkpoint_role"])): float(row["beta"])
        for row in selected
    }
    relevant_sentinels = [
        row
        for row in sentinel_rows
        if row["estimator"] == "bptt"
        or (
            row["eqprop_variant"] == "centered"
            and (str(row["architecture"]), str(row["checkpoint_role"]))
            in selected_lookup
            and float(row["beta"])
            == selected_lookup[
                (str(row["architecture"]), str(row["checkpoint_role"]))
            ]
        )
    ]
    sentinel_table = []
    for architecture in sorted({str(row["architecture"]) for row in relevant_sentinels}):
        for role in ("reconstructed_initialization", "best_validation"):
            rows = [
                row
                for row in relevant_sentinels
                if row["architecture"] == architecture and row["checkpoint_role"] == role
            ]
            sentinel_table.append(
                (
                    architecture.upper(),
                    role.replace("_", " "),
                    "pass" if all(bool(row["sentinel_pass"]) for row in rows) else "finite-iteration-limited",
                    sum(not bool(row["sentinel_pass"]) for row in rows),
                    len(rows),
                )
            )
    lines = [
        "# EqProp vs BPTT checkpoint-gradient analysis",
        "",
        ("Smoke result." if smoke else "Production diagnostic result."),
        " Bias tensors participated in network dynamics but were excluded from every gradient comparison.",
        " The official MNIST test split was not read.",
        "",
        "## Selected beta by worst weight layer",
        "",
        _markdown_table(
            ("Architecture", "Checkpoint", "beta", "Worst cosine", "Worst layer"),
            selection_table,
        ),
        "",
        "Selection maximizes the minimum centered cohort-mean cosine across weight layers independently for each architecture and checkpoint; exact ties choose the smaller beta.",
        "",
        "## Phase-length sentinel",
        "",
        _markdown_table(
            ("Architecture", "Checkpoint", "Status", "Failed rows", "Checked rows"),
            sentinel_table,
        ),
        "",
        "The sentinel compares the accepted training K with K=64 on the first fixed minibatches while holding the accepted-T common free state unchanged. A failed row is labeled finite-iteration-limited; it is not silently retuned.",
        "",
        "## Scope and artifacts",
        "",
        f"- Beta grid: `{analysis_config['gradient_contract']['betas']}`.",
        f"- Parameter rows: `{len(parameter_rows)}`.",
        "- `parameter_summary.csv` contains cohort-gradient cosines and batch robustness quantiles.",
        "- `raw_batch_metrics.csv` contains layer-wise batch comparisons.",
        "- `phase_diagnostics.csv` contains common-state hashes, projected residuals, displacements, symmetry defects, and active-set transitions.",
        "- `checkpoint_evolution.csv` compares initialization with the best-validation checkpoint.",
        "- `read_only_guards.json` records source and in-memory immutability checks.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _resolved_run_dir(
    output_root: Path, *, smoke: bool, run_id: str | None
) -> tuple[str, Path]:
    if run_id is None:
        run_id = "smoke-attempt-01" if smoke else "analysis"
    if not run_id or "/" in run_id or run_id in {".", ".."}:
        raise ValueError(f"Invalid run id {run_id!r}.")
    return run_id, output_root.expanduser().resolve() / run_id


def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.expanduser().resolve()
    analysis_config = _read_json(config_path)
    if analysis_config.get("schema_version") != "perfectdiode-conv-eqprop-bptt-beta-gradient-replay/v1":
        raise ValueError("Unexpected analysis config schema.")
    runtime_source = _validate_runtime_source(analysis_config)
    inventory, source_hashes_before = _source_run_inventory(analysis_config)
    architectures = [str(case["architecture"]) for case in inventory]
    if len(set(architectures)) != len(architectures):
        raise ValueError("Architectures must be unique in the selected source inventory.")
    data_root = args.dataset_root.expanduser().resolve()
    dataset_cfg = analysis_config["dataset"]
    batch_size = int(dataset_cfg["batch_size"])
    requested_examples = int(dataset_cfg["cohort_examples"])
    sentinel_cfg = analysis_config["gradient_contract"][
        "eqprop_phase_length_sentinel"
    ]
    sentinel_batches = int(sentinel_cfg["minibatches"])
    if args.smoke:
        requested_examples = batch_size
        sentinel_batches = 1
    first_source_config = inventory[0]["source_config"]
    batches, cohort = _build_validation_cohort(
        first_source_config,
        data_root=data_root,
        batch_size=batch_size,
        example_count=requested_examples,
    )
    if cohort["validation_indices_sha256"] != dataset_cfg[
        "validation_indices_sha256"
    ]:
        raise ValueError("Materialized validation split hash differs from config.")
    for case in inventory[1:]:
        if _dataset_signature(
            case["source_config"], data_root=data_root, batch_size=batch_size
        ) != cohort["dataset_signature"]:
            raise ValueError("Selected architectures do not share one dataset contract.")
    if sentinel_batches > len(batches):
        raise ValueError("Sentinel minibatches exceed replay cohort.")
    betas = [float(value) for value in analysis_config["gradient_contract"]["betas"]]
    if not betas or any(not math.isfinite(beta) or beta <= 0.0 for beta in betas):
        raise ValueError("All beta values must be positive and finite.")
    if len(set(betas)) != len(betas):
        raise ValueError("Beta values must be unique.")
    variants = [
        str(value)
        for value in analysis_config["gradient_contract"][
            "eqprop_variants_reported"
        ]
    ]
    if variants != ["centered", "positive", "negative"]:
        raise ValueError("Expected centered, positive, and negative EqProp variants.")
    zero_epsilon = float(analysis_config["gradient_contract"]["zero_epsilon"])
    zero_endpoint_relative_tolerance = float(
        analysis_config["gradient_contract"][
            "zero_endpoint_equivalence_relative_l2_tolerance"
        ]
    )
    if (
        not math.isfinite(zero_endpoint_relative_tolerance)
        or zero_endpoint_relative_tolerance <= 0.0
    ):
        raise ValueError("Zero-endpoint equivalence tolerance must be positive.")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    output_root = args.output_root.expanduser().resolve()
    run_id, run_dir = _resolved_run_dir(
        output_root, smoke=args.smoke, run_id=args.run_id
    )
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
    manifest = {
        "study_id": analysis_config["study_id"],
        "run_id": run_id,
        "arm_id": "conv123_legacy_adam_eqprop_bptt_checkpoint_gradients",
        "evidence_class": analysis_config["evidence_class"],
        "dataset": analysis_config["dataset"]["name"],
        "smoke": bool(args.smoke),
        "command": command,
        "configuration": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
            "resolved": analysis_config,
        },
        "runtime": {
            **runtime_context(target=args.target),
            "device": str(device),
            "hostname": socket.gethostname(),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "runtime_source": runtime_source,
        },
        "replay": {
            "cohort_examples": requested_examples,
            "batch_size": batch_size,
            "sentinel_minibatches": sentinel_batches,
            "official_test_read": False,
            "optimizer_constructed": False,
            "optimizer_steps_applied": False,
        },
        "analyzer": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
            "repository_root": str(ANALYZER_REPOSITORY_ROOT),
            "repository_head": _git_output(ANALYZER_REPOSITORY_ROOT, "rev-parse", "HEAD"),
        },
    }
    start_run(run_dir, manifest)
    _write_json(run_dir / "config.resolved.json", analysis_config)
    _write_json(run_dir / "cohort.json", cohort)
    _write_json(
        run_dir / "source_inventory.json",
        {
            "runtime_source": runtime_source,
            "cases": [
                {
                    key: (str(value) if isinstance(value, Path) else value)
                    for key, value in case.items()
                    if key != "source_config"
                }
                for case in inventory
            ],
        },
    )

    accumulator = _GradientAccumulator()
    raw_rows: list[dict[str, Any]] = []
    phase_rows: list[dict[str, Any]] = []
    parameter_guards: list[dict[str, Any]] = []
    common_state_guards: list[dict[str, Any]] = []
    variant_outcome_guards: list[dict[str, Any]] = []
    weight_names_by_architecture: dict[str, list[str]] = {}
    weight_types_by_architecture: dict[str, dict[str, str]] = {}
    roles = ["reconstructed_initialization", "best_validation"]
    total_cases = len(inventory) * len(roles)
    completed_cases = 0

    try:
        for case in inventory:
            architecture = str(case["architecture"])
            source_config = case["source_config"]
            for role in roles:
                operating_k = int(case["gradient_iterations"])
                runtimes = {
                    "operating": _build_runtime(
                        source_config,
                        device=device,
                        gradient_iterations=operating_k,
                    ),
                    "k64_sentinel": _build_runtime(
                        source_config,
                        device=device,
                        gradient_iterations=int(sentinel_cfg["gradient_iterations"]),
                    ),
                }
                init_hashes = {
                    phase: _historical_initialization_sha256(runtime["parameters"])
                    for phase, runtime in runtimes.items()
                }
                if len(set(init_hashes.values())) != 1 or next(iter(init_hashes.values())) != case[
                    "reconstructed_initialization_tensor_sha256"
                ]:
                    raise ValueError(
                        f"Initialization reconstruction mismatch for {architecture}."
                    )
                for runtime in runtimes.values():
                    if role == "best_validation":
                        runtime["energy_fn"].load(case["best_checkpoint_path"])
                        _verify_npz_checkpoint(runtime, case["weights_best_path"])
                first_runtime = runtimes["operating"]
                names = list(first_runtime["weight_names"])
                types = dict(zip(names, first_runtime["weight_types"], strict=True))
                if architecture in weight_names_by_architecture and weight_names_by_architecture[architecture] != names:
                    raise ValueError("Weight schema changed between checkpoint roles.")
                weight_names_by_architecture[architecture] = names
                weight_types_by_architecture[architecture] = types
                if any(value not in WEIGHT_TYPES for value in types.values()):
                    raise RuntimeError("Non-weight parameter passed the inclusion gate.")

                hashes_before = {
                    phase: _parameter_state_sha256(runtime["parameters"])
                    for phase, runtime in runtimes.items()
                }
                if len(set(hashes_before.values())) != 1:
                    raise RuntimeError(
                        f"Operating and K64 runtime parameters differ for {architecture}/{role}."
                    )
                for batch in batches:
                    batch_index = int(batch["batch_index"])
                    phases = ["operating"]
                    if batch_index < sentinel_batches:
                        phases.append("k64_sentinel")
                    phase_outputs: dict[
                        str,
                        tuple[
                            dict[str, torch.Tensor],
                            dict[float, dict[str, dict[str, torch.Tensor]]],
                            dict[float, dict[str, str]],
                        ],
                    ] = {}
                    for phase in phases:
                        runtime = runtimes[phase]
                        images = batch["images"].to(device)
                        labels = batch["labels"].to(device)
                        runtime["network"].set_input(images, reset=True)
                        runtime["minimizer_inference"].compute_equilibrium()
                        runtime["cost_fn"].set_target(labels)
                        common_states = _clone_states(runtime["free_layers"])
                        common_hash = _layer_state_sha256(runtime["free_layers"])
                        bptt, estimates, diagnostics, variant_outcomes = _run_gradient_phases(
                            runtime,
                            free_states=common_states,
                            betas=betas,
                            zero_epsilon=zero_epsilon,
                            zero_endpoint_relative_tolerance=zero_endpoint_relative_tolerance,
                        )
                        phase_outputs[phase] = (
                            bptt,
                            estimates,
                            variant_outcomes,
                        )
                        for beta in betas:
                            for variant in variants:
                                variant_outcome_guards.append(
                                    {
                                        "architecture": architecture,
                                        "checkpoint_role": role,
                                        "phase_length": phase,
                                        "batch_index": batch_index,
                                        "beta": beta,
                                        "eqprop_variant": variant,
                                        "outcome": variant_outcomes[beta][variant],
                                    }
                                )
                        for diagnostic in diagnostics:
                            phase_rows.append(
                                {
                                    "architecture": architecture,
                                    "checkpoint_role": role,
                                    "phase_length": phase,
                                    "gradient_iterations": runtime[
                                        "gradient_iterations"
                                    ],
                                    "batch_index": batch_index,
                                    "batch_payload_sha256": batch[
                                        "payload_sha256"
                                    ],
                                    **diagnostic,
                                }
                            )
                        common_state_guards.append(
                            {
                                "architecture": architecture,
                                "checkpoint_role": role,
                                "phase_length": phase,
                                "batch_index": batch_index,
                                "common_free_state_sha256": common_hash,
                                "phase_starts_match_common_state": all(
                                    diagnostic["phase_start_state_sha256"]
                                    == common_hash
                                    for diagnostic in diagnostics
                                ),
                            }
                        )
                        aggregate_phases = [phase]
                        if phase == "operating" and batch_index < sentinel_batches:
                            aggregate_phases.append("operating_sentinel_cohort")
                        for aggregate_phase in aggregate_phases:
                            for name in names:
                                bptt_key = (
                                    architecture,
                                    role,
                                    aggregate_phase,
                                    "bptt",
                                    None,
                                    None,
                                    name,
                                )
                                accumulator.add(
                                    bptt_key,
                                    bptt[name],
                                    batch_size=int(images.shape[0]),
                                )
                                for beta in betas:
                                    for variant in variants:
                                        outcome = variant_outcomes[beta][variant]
                                        if outcome != "ok":
                                            if aggregate_phase == phase:
                                                raw_rows.append(
                                                    {
                                                        "architecture": architecture,
                                                        "checkpoint_role": role,
                                                        "phase_length": phase,
                                                        "gradient_iterations": runtime[
                                                            "gradient_iterations"
                                                        ],
                                                        "batch_index": batch_index,
                                                        "batch_size": int(images.shape[0]),
                                                        "batch_payload_sha256": batch[
                                                            "payload_sha256"
                                                        ],
                                                        "beta": beta,
                                                        "eqprop_variant": variant,
                                                        "parameter_name": name,
                                                        "parameter_type": types[name],
                                                        "bias_excluded": True,
                                                        "outcome": outcome,
                                                    }
                                                )
                                            continue
                                        metrics = _vector_metrics(
                                            bptt[name],
                                            estimates[beta][variant][name],
                                            zero_epsilon=zero_epsilon,
                                        )
                                        ep_key = (
                                            architecture,
                                            role,
                                            aggregate_phase,
                                            "eqprop",
                                            beta,
                                            variant,
                                            name,
                                        )
                                        accumulator.add(
                                            ep_key,
                                            estimates[beta][variant][name],
                                            batch_size=int(images.shape[0]),
                                            batch_cosine=metrics["cosine"],
                                        )
                                        if aggregate_phase == phase:
                                            raw_rows.append(
                                                {
                                                    "architecture": architecture,
                                                    "checkpoint_role": role,
                                                    "phase_length": phase,
                                                    "gradient_iterations": runtime[
                                                        "gradient_iterations"
                                                    ],
                                                    "batch_index": batch_index,
                                                    "batch_size": int(images.shape[0]),
                                                    "batch_payload_sha256": batch[
                                                        "payload_sha256"
                                                    ],
                                                    "beta": beta,
                                                    "eqprop_variant": variant,
                                                    "parameter_name": name,
                                                    "parameter_type": types[name],
                                                    "bias_excluded": True,
                                                    "outcome": (
                                                        "ok"
                                                        if metrics["cosine"] is not None
                                                        else "dead_gradient"
                                                    ),
                                                    **metrics,
                                                }
                                            )
                    if "k64_sentinel" in phase_outputs:
                        op_bptt, op_ep, _ = phase_outputs["operating"]
                        ref_bptt, ref_ep, _ = phase_outputs["k64_sentinel"]
                        for name in names:
                            _finite(op_bptt[name] - ref_bptt[name], "BPTT sentinel delta")
                            for beta in betas:
                                for variant in variants:
                                    if (
                                        variant not in op_ep[beta]
                                        or variant not in ref_ep[beta]
                                    ):
                                        continue
                                    _finite(
                                        op_ep[beta][variant][name]
                                        - ref_ep[beta][variant][name],
                                        "EqProp sentinel delta",
                                    )
                hashes_after = {
                    phase: _parameter_state_sha256(runtime["parameters"])
                    for phase, runtime in runtimes.items()
                }
                parameter_guard_pass = hashes_before == hashes_after
                parameter_guards.append(
                    {
                        "architecture": architecture,
                        "checkpoint_role": role,
                        "parameter_state_sha256_before": hashes_before,
                        "parameter_state_sha256_after": hashes_after,
                        "unchanged": parameter_guard_pass,
                        "optimizer_constructed": False,
                        "optimizer_steps_applied": False,
                    }
                )
                if not parameter_guard_pass:
                    raise RuntimeError(
                        f"Parameter mutation detected for {architecture}/{role}."
                    )
                completed_cases += 1
                update_status_progress(
                    run_dir,
                    {
                        "stage": "checkpoint-gradient-replay",
                        "completed_cases": completed_cases,
                        "total_cases": total_cases,
                        "architecture": architecture,
                        "checkpoint_role": role,
                    },
                )
                append_metric(
                    run_dir / "metrics.jsonl",
                    {
                        "kind": "checkpoint_case_complete",
                        "architecture": architecture,
                        "checkpoint_role": role,
                        "cohort_examples": requested_examples,
                        "beta_count": len(betas),
                        "weight_layers": names,
                        "parameter_state_unchanged": True,
                    },
                )
                del runtimes
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        source_hashes_after = _verify_source_hashes(
            inventory, source_hashes_before
        )
        if not all(
            guard["phase_starts_match_common_state"]
            for guard in common_state_guards
        ):
            raise RuntimeError("At least one EqProp phase start-state guard failed.")
        phases = ["operating", "operating_sentinel_cohort", "k64_sentinel"]
        parameter_rows = _parameter_summary(
            accumulator,
            architectures=architectures,
            roles=roles,
            phases=phases,
            betas=betas,
            variants=variants,
            weight_names_by_architecture=weight_names_by_architecture,
            weight_types_by_architecture=weight_types_by_architecture,
            zero_epsilon=zero_epsilon,
        )
        beta_rows = _select_betas(parameter_rows)
        sentinel_rows = _sentinel_summary(
            accumulator,
            architectures=architectures,
            roles=roles,
            betas=betas,
            variants=variants,
            weight_names_by_architecture=weight_names_by_architecture,
            zero_epsilon=zero_epsilon,
            gates=sentinel_cfg,
        )
        evolution_rows = _checkpoint_evolution(
            accumulator,
            architectures=architectures,
            betas=betas,
            variants=variants,
            weight_names_by_architecture=weight_names_by_architecture,
            zero_epsilon=zero_epsilon,
        )
        _write_csv(run_dir / "raw_batch_metrics.csv", raw_rows)
        _write_csv(run_dir / "phase_diagnostics.csv", phase_rows)
        _write_csv(run_dir / "variant_outcomes.csv", variant_outcome_guards)
        _write_csv(run_dir / "parameter_summary.csv", parameter_rows)
        _write_csv(run_dir / "joint_beta_summary.csv", beta_rows)
        _write_csv(run_dir / "phase_sentinel_summary.csv", sentinel_rows)
        _write_csv(run_dir / "checkpoint_evolution.csv", evolution_rows)
        _save_mean_gradients(run_dir / "mean_gradients.npz", accumulator)
        _plot_cosines(
            run_dir / "cosine_by_beta_layer.png",
            parameter_rows,
            architectures,
            roles,
        )
        read_only_guards = {
            "schema": "eqprop-bptt-read-only-guards/v1",
            "source_hashes_before": source_hashes_before,
            "source_hashes_after": source_hashes_after,
            "source_files_unchanged": source_hashes_before == source_hashes_after,
            "parameter_guards": parameter_guards,
            "all_parameter_tensors_unchanged": all(
                guard["unchanged"] for guard in parameter_guards
            ),
            "common_state_guards": common_state_guards,
            "all_phase_starts_match_common_state": all(
                guard["phase_starts_match_common_state"]
                for guard in common_state_guards
            ),
            "biases_present_in_dynamics": True,
            "bias_gradients_excluded": True,
            "included_parameter_types": sorted(WEIGHT_TYPES),
            "optimizer_constructed": False,
            "optimizer_steps_applied": False,
            "official_test_read": False,
            "structured_variant_outcomes": variant_outcome_guards,
            "unstable_variant_outcome_count": sum(
                guard["outcome"] != "ok" for guard in variant_outcome_guards
            ),
        }
        _write_json(run_dir / "read_only_guards.json", read_only_guards)
        _write_report(
            run_dir / "report.md",
            analysis_config=analysis_config,
            beta_rows=beta_rows,
            sentinel_rows=sentinel_rows,
            parameter_rows=parameter_rows,
            smoke=args.smoke,
        )
        selected = [row for row in beta_rows if row["selected"]]
        sentinel_failures = sum(
            not bool(row["sentinel_pass"]) for row in sentinel_rows
        )
        unstable_variant_outcomes = sum(
            guard["outcome"] != "ok" for guard in variant_outcome_guards
        )
        append_metric(
            run_dir / "metrics.jsonl",
            {
                "kind": "analysis_summary",
                "selected_betas": selected,
                "sentinel_failure_rows": sentinel_failures,
                "unstable_variant_outcomes": unstable_variant_outcomes,
                "source_files_unchanged": True,
                "parameter_tensors_unchanged": True,
                "bias_gradients_excluded": True,
            },
        )
        result = complete_run(
            run_dir,
            terminal_metrics={
                "checkpoint_cases": total_cases,
                "cohort_examples": requested_examples,
                "beta_count": len(betas),
                "selected_betas": selected,
                "sentinel_failure_rows": sentinel_failures,
                "unstable_variant_outcomes": unstable_variant_outcomes,
                "weight_parameter_rows": len(parameter_rows),
            },
            completion={
                "criteria_met": True,
                "source_files_unchanged": True,
                "parameter_tensors_unchanged": True,
                "all_phase_starts_match_common_state": True,
                "bias_gradients_excluded": True,
                "official_test_read": False,
                "optimizer_steps_applied": False,
                "unstable_beta_phases_retained_as_structured_outcomes": True,
                "smoke": bool(args.smoke),
            },
        )
        errors = validate_run(run_dir)
        if errors:
            raise RuntimeError(
                f"Completed reporting bundle failed validation: {'; '.join(errors)}"
            )
        return result
    except BaseException as error:
        status_path = run_dir / "status.json"
        if status_path.is_file():
            status = _read_json(status_path)
            if status.get("state") == "running" and not (run_dir / "result.json").exists():
                fail_run(run_dir, error=error)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=BOOTSTRAP_CONFIG_PATH)
    parser.add_argument("--runtime-source-root", type=Path, default=RUNTIME_SOURCE_ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            ANALYZER_REPOSITORY_ROOT
            / "results/perfectdiode-conv123-legacy-adam-eqprop-bptt-beta-cosine-seed0-20260810-v1"
        ),
    )
    parser.add_argument(
        "--dataset-root", type=Path, default=Path("/home/filip/datasets/mnist")
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--target", default="main")
    parser.add_argument("--run-id")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.runtime_source_root.expanduser().resolve() != RUNTIME_SOURCE_ROOT:
        raise ValueError(
            "--runtime-source-root must be supplied during process bootstrap and "
            "must match the configured exact source contract."
        )
    result = run_analysis(args)
    print(
        json.dumps(
            {
                "state": "complete",
                "result_sha256": sha256_file(
                    args.output_root.expanduser().resolve()
                    / (args.run_id or ("smoke-attempt-01" if args.smoke else "analysis"))
                    / "result.json"
                ),
                "terminal_metrics": result["terminal_metrics"],
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
