#!/usr/bin/env python3
"""Analyze final weights and projected-gradient pressure in Conv1/Conv2 paper runs.

The analysis is read-only with respect to source checkpoints.  It validates the
checkpoint artifacts, measures final and best-checkpoint conductance
distributions, reconstructs the seed-0 initialization, and replays a fixed
training cohort from every final checkpoint without applying optimizer steps.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

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

from custom_classes import FlexibleDeepResistiveEnergy
from labs.datasets import stable_index_sequence_hash
from labs.mnist_train import (
    _reset_name_counters,
    _resolve_callable,
    _resolve_dataset_config,
    _set_seed,
    load_config,
    train_mnist_conv,
)


SCHEMA = "paper-conv12-final-weight-projection-analysis/v1"
CONDUCTANCE_PREFIXES = ("ConvWeight_", "DenseWeight_")
BOUND_ATOL = 1.0e-12


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Expected at least one row for {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _finite(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Expected finite {label}; found {value!r}.")
    return result


def _quantile(values: Iterable[float], q: float) -> float:
    array = np.asarray(tuple(values), dtype=np.float64)
    if not len(array):
        raise ValueError("Expected a non-empty quantile input.")
    return float(np.quantile(array, q))


def _discover_runs(input_root: Path) -> list[Path]:
    runs = sorted(
        path
        for architecture in ("conv1", "conv2")
        for path in (input_root / architecture).iterdir()
        if path.is_dir() and path.name[:1].isdigit()
    )
    if len(runs) != 12:
        raise ValueError(f"Expected 12 production runs; found {len(runs)}.")
    return runs


def _verify_artifacts(runs: list[Path]) -> list[dict[str, Any]]:
    verified: list[dict[str, Any]] = []
    for run in runs:
        result = json.loads((run / "result.json").read_text(encoding="utf-8"))
        checkpoints = [
            item for item in result["artifacts"] if item["kind"] == "checkpoint"
        ]
        if len(checkpoints) != 4:
            raise ValueError(f"Expected four checkpoint artifacts in {run}.")
        for item in checkpoints:
            path = run / Path(item["path"]).name
            digest = _sha256_file(path)
            size = path.stat().st_size
            if digest != item["sha256"] or size != int(item["size_bytes"]):
                raise ValueError(f"Checkpoint validation failed for {path}.")
            verified.append(
                {
                    "run_dir": str(run.resolve()),
                    "path": str(path.resolve()),
                    "sha256": digest,
                    "size_bytes": size,
                }
            )
    return verified


def _load_npz_weights(path: Path) -> dict[str, torch.Tensor]:
    with np.load(path, allow_pickle=False) as payload:
        names = [str(value) for value in payload["param_names"].tolist()]
        return {
            name: torch.from_numpy(np.asarray(payload[name])).to(dtype=torch.float64)
            for name in names
        }


def _model_config(config: dict[str, Any]) -> dict[str, Any]:
    model_key = config["lab"]["model_key"]
    return {
        **config["model_base"],
        **config["model_overrides"][model_key],
    }


def _build_initial_weights(config: dict[str, Any]) -> dict[str, torch.Tensor]:
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
    return {
        str(parameter.name).strip(): parameter.state.detach().cpu().to(
            dtype=torch.float64
        )
        for parameter in energy_fn.params()
    }


def _tensor_hash(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(str(tuple(tensor.shape)).encode("utf-8"))
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _correlation(left: torch.Tensor, right: torch.Tensor) -> float | None:
    x = left.reshape(-1) - left.mean()
    y = right.reshape(-1) - right.mean()
    denominator = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    if float(denominator) <= 0.0:
        return None
    return float(torch.dot(x, y) / denominator)


def _distribution_stats(
    final: torch.Tensor,
    initial: torch.Tensor,
    *,
    lower: float,
    upper: float,
) -> dict[str, Any]:
    if final.shape != initial.shape:
        raise ValueError(f"Mismatched tensor shapes: {final.shape} and {initial.shape}.")
    vector = final.reshape(-1)
    start = initial.reshape(-1)
    lower_tensor = torch.as_tensor(lower, dtype=vector.dtype)
    upper_tensor = torch.as_tensor(upper, dtype=vector.dtype)
    at_lower = torch.isclose(vector, lower_tensor, rtol=0.0, atol=BOUND_ATOL)
    at_upper = torch.isclose(vector, upper_tensor, rtol=0.0, atol=BOUND_ATOL)
    delta = vector - start
    result: dict[str, Any] = {
        "count": int(vector.numel()),
        "minimum": float(vector.min()),
        "maximum": float(vector.max()),
        "mean": float(vector.mean()),
        "std": float(vector.std(unbiased=False)),
        "rms": float(torch.sqrt(torch.mean(vector.square()))),
        "exact_lower_fraction": float(at_lower.to(torch.float64).mean()),
        "exact_upper_fraction": float(at_upper.to(torch.float64).mean()),
        "combined_exact_bound_fraction": float(
            (at_lower | at_upper).to(torch.float64).mean()
        ),
        "mean_abs_delta_from_initial": float(delta.abs().mean()),
        "rms_delta_from_initial": float(torch.sqrt(torch.mean(delta.square()))),
        "unchanged_from_initial_fraction": float(
            torch.isclose(vector, start, rtol=0.0, atol=0.0)
            .to(torch.float64)
            .mean()
        ),
        "initial_final_correlation": _correlation(start, vector),
        "maximum_fraction_of_upper_bound": float(vector.max()) / upper,
    }
    for threshold in (1.0e-8, 1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2):
        result[f"le_{threshold:.0e}_fraction"] = float(
            (vector <= lower + threshold).to(torch.float64).mean()
        )
    for q in (0.001, 0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 0.999):
        result[f"q{100.0 * q:g}"] = float(torch.quantile(vector, q))
    return result


def _aggregate_named(
    values: dict[str, torch.Tensor], prefixes: tuple[str, ...]
) -> torch.Tensor:
    selected = [
        value.reshape(-1)
        for name, value in sorted(values.items())
        if name.startswith(prefixes)
    ]
    if not selected:
        raise ValueError(f"No tensors matched prefixes {prefixes!r}.")
    return torch.cat(selected)


def _run_identity(run: Path, config: dict[str, Any]) -> dict[str, Any]:
    arm_id = str(config["arm_id"])
    architecture = "conv1" if arm_id.startswith("conv1_") else "conv2"
    scheme = next(
        scheme for scheme in ("baseline", "ours", "legacy") if f"_{scheme}_" in arm_id
    )
    optimizer = str(config["optimizer"]["name"])
    return {
        "architecture": architecture,
        "scheme": scheme,
        "optimizer": optimizer,
        "arm_id": arm_id,
        "run_id": run.name,
    }


def _distribution_analysis(
    runs: list[Path],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, torch.Tensor]]]:
    rows: list[dict[str, Any]] = []
    initial_by_architecture: dict[str, dict[str, torch.Tensor]] = {}
    initial_hashes: dict[str, dict[str, str]] = {}
    bounds_by_architecture: dict[str, tuple[float, float]] = {}
    for run in runs:
        config = json.loads((run / "config.used.json").read_text(encoding="utf-8"))
        identity = _run_identity(run, config)
        architecture = identity["architecture"]
        initial = _build_initial_weights(config)
        hashes = {
            name: _tensor_hash(value)
            for name, value in initial.items()
            if name.startswith(CONDUCTANCE_PREFIXES)
        }
        if architecture in initial_hashes and hashes != initial_hashes[architecture]:
            raise ValueError(
                f"Reconstructed initialization differs within {architecture}."
            )
        initial_hashes[architecture] = hashes
        initial_by_architecture[architecture] = initial

        final = _load_npz_weights(run / "weights_final.npz")
        best = _load_npz_weights(run / "weights_best.npz")
        model = _model_config(config)
        lower = float(model["weight_min"])
        upper = float(model["weight_max"])
        bounds_by_architecture[architecture] = (lower, upper)
        result = json.loads((run / "result.json").read_text(encoding="utf-8"))
        metrics = result["terminal_metrics"]
        common = {
            **identity,
            "weight_lower": lower,
            "weight_upper": upper,
            "final_validation_accuracy": float(metrics["validation"]["final_accuracy"]),
            "best_validation_accuracy": float(metrics["validation"]["best_accuracy"]),
            "official_test_accuracy_best_checkpoint": float(metrics["test"]["accuracy"]),
            "best_epoch": int(metrics["best_epoch"]),
        }
        for role, weights in (("final", final), ("best", best)):
            weight_names = sorted(
                name for name in weights if name.startswith(CONDUCTANCE_PREFIXES)
            )
            for name in weight_names:
                rows.append(
                    {
                        **common,
                        "checkpoint_role": role,
                        "parameter_name": name,
                        **_distribution_stats(
                            weights[name], initial[name], lower=lower, upper=upper
                        ),
                    }
                )
            rows.append(
                {
                    **common,
                    "checkpoint_role": role,
                    "parameter_name": "ALL_CONDUCTANCE_WEIGHTS",
                    **_distribution_stats(
                        _aggregate_named(weights, CONDUCTANCE_PREFIXES),
                        _aggregate_named(initial, CONDUCTANCE_PREFIXES),
                        lower=lower,
                        upper=upper,
                    ),
                }
            )
    for architecture, initial in sorted(initial_by_architecture.items()):
        lower, upper = bounds_by_architecture[architecture]
        common = {
            "architecture": architecture,
            "scheme": "shared",
            "optimizer": "shared",
            "arm_id": f"{architecture}_shared_seed0_initialization",
            "run_id": "reconstructed_from_exact_production_source_and_seed",
            "weight_lower": lower,
            "weight_upper": upper,
            "final_validation_accuracy": None,
            "best_validation_accuracy": None,
            "official_test_accuracy_best_checkpoint": None,
            "best_epoch": None,
            "checkpoint_role": "initial_reconstructed",
        }
        weight_names = sorted(
            name for name in initial if name.startswith(CONDUCTANCE_PREFIXES)
        )
        for name in weight_names:
            rows.append(
                {
                    **common,
                    "parameter_name": name,
                    **_distribution_stats(
                        initial[name], initial[name], lower=lower, upper=upper
                    ),
                }
            )
        rows.append(
            {
                **common,
                "parameter_name": "ALL_CONDUCTANCE_WEIGHTS",
                **_distribution_stats(
                    _aggregate_named(initial, CONDUCTANCE_PREFIXES),
                    _aggregate_named(initial, CONDUCTANCE_PREFIXES),
                    lower=lower,
                    upper=upper,
                ),
            }
        )
    return rows, initial_by_architecture


def _unit_norm_rows(runs: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        config = json.loads((run / "config.used.json").read_text(encoding="utf-8"))
        identity = _run_identity(run, config)
        weights = _load_npz_weights(run / "weights_final.npz")
        for name, value in sorted(weights.items()):
            if not name.startswith(CONDUCTANCE_PREFIXES):
                continue
            array = value.detach().cpu().numpy().astype(np.float64, copy=False)
            if name.startswith("ConvWeight_"):
                unit_axis = "output_channel"
                reduction_axes = tuple(range(1, array.ndim))
            else:
                unit_axis = "paired_output"
                reduction_axes = tuple(range(array.ndim - 1))
            norms = np.sqrt(np.sum(array * array, axis=reduction_axes))
            energy = np.sort(norms * norms)[::-1]
            top_count = max(1, int(math.ceil(0.10 * len(norms))))
            median = float(np.median(norms))
            mean = float(np.mean(norms))
            rows.append(
                {
                    **identity,
                    "parameter_name": name,
                    "unit_axis": unit_axis,
                    "unit_count": int(len(norms)),
                    "zero_unit_count": int(np.sum(norms == 0.0)),
                    "unit_norm_mean": mean,
                    "unit_norm_std": float(np.std(norms)),
                    "unit_norm_cv": float(np.std(norms) / mean) if mean > 0.0 else None,
                    "unit_norm_median": median,
                    "unit_norm_maximum": float(np.max(norms)),
                    "maximum_over_median": (
                        float(np.max(norms) / median) if median > 0.0 else None
                    ),
                    "top_10pct_unit_energy_fraction": float(
                        np.sum(energy[:top_count]) / np.sum(energy)
                    ),
                }
            )
    return rows


def _cohort_from_config(
    config_path: Path, *, replay_batches: int
) -> dict[str, Any]:
    config = load_config(config_path)
    dataset_key, dataset_config = _resolve_dataset_config(
        config, config["lab"]["dataset_key"]
    )
    factory = _resolve_callable(dataset_config["factory"])
    dataset_params = dict(dataset_config["params"])
    dataset_params.setdefault("device", torch.device("cpu"))
    builder = factory(**dataset_params)
    loaders = builder.build()
    batches = loaders.train_batch_indices(num_epochs=1)[0][:replay_batches]
    if len(batches) != replay_batches:
        raise ValueError(
            f"Expected {replay_batches} replay batches; found {len(batches)}."
        )
    flattened = tuple(index for batch in batches for index in batch)
    return {
        "dataset_key": dataset_key,
        "split_seed": int(loaders.split_seed),
        "shuffle_seed": int(loaders.shuffle_seed),
        "train_indices_sha256": loaders.train_indices_hash,
        "validation_indices_sha256": loaders.validation_indices_hash,
        "first_epoch_batch_order_sha256": loaders.first_epoch_batch_order_hash,
        "batch_count": replay_batches,
        "example_count": len(flattened),
        "source_indices": [list(map(int, batch)) for batch in batches],
        "source_indices_sha256": stable_index_sequence_hash(flattened),
        "batch_source_indices_sha256": [
            stable_index_sequence_hash(tuple(map(int, batch))) for batch in batches
        ],
    }


def _norm(value: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(value.reshape(-1).to(torch.float64)))


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0.0:
        return None
    result = numerator / denominator
    return result if math.isfinite(result) else None


def _gradient_row(
    *,
    identity: dict[str, Any],
    batch: int,
    loss: float,
    parameter: Any,
    gradient: torch.Tensor,
    learning_rate: float,
    optimizer: str,
    adam_epsilon: float,
    lower: float,
    upper: float,
) -> dict[str, Any]:
    state = parameter.state.detach()
    grad = gradient.detach()
    vector = state.reshape(-1)
    grad_vector = grad.reshape(-1)
    lower_tensor = torch.as_tensor(lower, dtype=vector.dtype, device=vector.device)
    upper_tensor = torch.as_tensor(upper, dtype=vector.dtype, device=vector.device)
    at_lower = torch.isclose(vector, lower_tensor, rtol=0.0, atol=BOUND_ATOL)
    at_upper = torch.isclose(vector, upper_tensor, rtol=0.0, atol=BOUND_ATOL)
    descent = -grad_vector
    outward = (at_lower & (descent < 0.0)) | (at_upper & (descent > 0.0))
    tangent_descent = descent.clone()
    tangent_descent[outward] = 0.0
    descent_norm = _norm(descent)
    tangent_norm = _norm(tangent_descent)
    outward_norm = _norm(descent[outward]) if bool(outward.any()) else 0.0

    if optimizer == "SGD":
        proposal = -learning_rate * grad_vector
        proposal_kind = "exact_sgd_momentum0_weight_decay0"
    elif optimizer == "Adam":
        proposal = -learning_rate * grad_vector / (grad_vector.abs() + adam_epsilon)
        proposal_kind = "fresh_adam_shadow_without_saved_moments"
    else:
        raise ValueError(f"Unsupported optimizer {optimizer!r}.")
    unprojected = vector + proposal
    projected = torch.clamp(unprojected, min=lower, max=upper)
    applied = projected - vector
    projection_change = projected - unprojected
    proposal_norm = _norm(proposal)
    applied_norm = _norm(applied)
    changed = ~torch.isclose(projected, unprojected, rtol=0.0, atol=0.0)
    bound_count = int((at_lower | at_upper).sum())
    outward_count = int(outward.sum())

    return {
        **identity,
        "batch": batch,
        "loss": _finite(loss, "replay loss"),
        "parameter_name": str(parameter.name).strip(),
        "element_count": int(vector.numel()),
        "learning_rate": learning_rate,
        "proposal_kind": proposal_kind,
        "gradient_rms": float(torch.sqrt(torch.mean(grad_vector.to(torch.float64).square()))),
        "gradient_zero_fraction": float(
            (grad_vector.abs() <= 1.0e-12).to(torch.float64).mean()
        ),
        "exact_lower_fraction": float(at_lower.to(torch.float64).mean()),
        "exact_upper_fraction": float(at_upper.to(torch.float64).mean()),
        "outward_component_fraction_all": outward_count / int(vector.numel()),
        "outward_component_fraction_among_bound": (
            outward_count / bound_count if bound_count else None
        ),
        "outward_gradient_energy_fraction": (
            (outward_norm / descent_norm) ** 2 if descent_norm > 0.0 else None
        ),
        "tangent_gradient_efficiency_l2": _safe_ratio(tangent_norm, descent_norm),
        "proposal_rms": float(torch.sqrt(torch.mean(proposal.to(torch.float64).square()))),
        "applied_update_rms": float(
            torch.sqrt(torch.mean(applied.to(torch.float64).square()))
        ),
        "projection_change_rms": float(
            torch.sqrt(torch.mean(projection_change.to(torch.float64).square()))
        ),
        "projection_efficiency_l2": _safe_ratio(applied_norm, proposal_norm),
        "projection_changed_fraction": float(changed.to(torch.float64).mean()),
        "proposal_out_of_bounds_fraction": float(
            ((unprojected < lower) | (unprojected > upper))
            .to(torch.float64)
            .mean()
        ),
    }


def _aggregate_gradient_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["arm_id"], row["parameter_name"]), []).append(row)
    metrics = (
        "loss",
        "gradient_rms",
        "gradient_zero_fraction",
        "exact_lower_fraction",
        "exact_upper_fraction",
        "outward_component_fraction_all",
        "outward_component_fraction_among_bound",
        "outward_gradient_energy_fraction",
        "tangent_gradient_efficiency_l2",
        "proposal_rms",
        "applied_update_rms",
        "projection_change_rms",
        "projection_efficiency_l2",
        "projection_changed_fraction",
        "proposal_out_of_bounds_fraction",
    )
    output = []
    for (_arm_id, _parameter_name), group in sorted(grouped.items()):
        summary: dict[str, Any] = {
            key: group[0][key]
            for key in (
                "architecture",
                "scheme",
                "optimizer",
                "arm_id",
                "run_id",
                "parameter_name",
                "element_count",
                "learning_rate",
                "proposal_kind",
            )
        }
        summary["batch_count"] = len(group)
        for metric in metrics:
            values = [float(row[metric]) for row in group if row[metric] is not None]
            summary[f"{metric}_median"] = _quantile(values, 0.5) if values else None
            summary[f"{metric}_q10"] = _quantile(values, 0.1) if values else None
            summary[f"{metric}_q90"] = _quantile(values, 0.9) if values else None
        output.append(summary)
    return output


def _replay_gradients(
    runs: list[Path],
    *,
    output_dir: Path,
    dataset_root: Path,
    replay_batches: int,
    device: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    replay_root = output_dir / "replay_runs"
    replay_root.mkdir(parents=True, exist_ok=True)
    cohort: dict[str, Any] | None = None
    batch_rows: list[dict[str, Any]] = []
    checkpoint_guards: list[dict[str, Any]] = []
    for run in runs:
        config = json.loads((run / "config.used.json").read_text(encoding="utf-8"))
        identity = _run_identity(run, config)
        local_config = json.loads(json.dumps(config))
        local_config["datasets"][config["lab"]["dataset_key"]]["params"]["root"] = str(
            dataset_root.resolve()
        )
        replay_dir = replay_root / identity["arm_id"]
        replay_dir.mkdir(parents=True, exist_ok=True)
        replay_config_path = replay_dir / "replay_config.json"
        _write_json(replay_config_path, local_config)
        current_cohort = _cohort_from_config(
            replay_config_path, replay_batches=replay_batches
        )
        if cohort is None:
            cohort = current_cohort
        elif current_cohort != cohort:
            raise ValueError("Replay cohorts differ across matched runs.")
        source_provenance = json.loads(
            (run / "metrics.json").read_text(encoding="utf-8")
        )["dataset_provenance"]
        for key in (
            "train_indices_sha256",
            "validation_indices_sha256",
            "first_epoch_batch_order_sha256",
        ):
            if cohort[key] != source_provenance[key]:
                raise ValueError(
                    f"Replay cohort provenance mismatch for {identity['arm_id']} {key}."
                )

        model = _model_config(config)
        lower = float(model["weight_min"])
        upper = float(model["weight_max"])
        learning_rates = {
            str(name): float(value)
            for name, value in config["learning_rates_by_parameter"].items()
        }
        optimizer = str(config["optimizer"]["name"])
        adam_epsilon = float(config["optimizer"].get("eps", 1.0e-8))
        checkpoint = run / "final_model.pt"
        before_hash = _sha256_file(checkpoint)

        def callback(payload: dict[str, Any]) -> bool:
            for parameter, gradient in zip(
                payload["parameters"], payload["gradients"], strict=True
            ):
                name = str(parameter.name).strip()
                if not name.startswith(CONDUCTANCE_PREFIXES):
                    continue
                batch_rows.append(
                    _gradient_row(
                        identity=identity,
                        batch=int(payload["batch"]),
                        loss=float(payload["loss"]),
                        parameter=parameter,
                        gradient=gradient,
                        learning_rate=learning_rates[name],
                        optimizer=optimizer,
                        adam_epsilon=adam_epsilon,
                        lower=lower,
                        upper=upper,
                    )
                )
            return False

        train_mnist_conv(
            config_path=replay_config_path,
            epochs=1,
            lr=config["lr"],
            beta=config["beta"],
            log_interval=0,
            max_batches=replay_batches,
            max_test_batches=1,
            device=device,
            dataset_key=config["lab"]["dataset_key"],
            model_key=config["lab"]["model_key"],
            output_dir=replay_dir,
            training_algorithm=config["training_algorithm"],
            seed=config["seed"],
            lr_decay=config["optimizer"]["lr_decay"],
            init_checkpoint_path=checkpoint,
            batch_state_policy=config["batch_state_policy"],
            optimizer_name=optimizer,
            momentum=config["optimizer"].get("momentum"),
            weight_decay=config["optimizer"].get("weight_decay"),
            gradient_callback=callback,
            apply_optimizer_steps=False,
            skip_terminal_official_test=True,
        )
        after_hash = _sha256_file(checkpoint)
        if before_hash != after_hash:
            raise RuntimeError(f"Source checkpoint changed during replay: {checkpoint}.")
        source_weights = _load_npz_weights(run / "weights_final.npz")
        replay_weights = _load_npz_weights(replay_dir / "weights_final.npz")
        tensor_match = all(
            torch.equal(source_weights[name], replay_weights[name])
            for name in source_weights
        )
        if not tensor_match:
            raise RuntimeError(f"Replay modified parameters for {identity['arm_id']}.")
        checkpoint_guards.append(
            {
                **identity,
                "checkpoint": str(checkpoint.resolve()),
                "sha256_before": before_hash,
                "sha256_after": after_hash,
                "source_replay_tensor_match": tensor_match,
                "optimizer_steps_applied": False,
            }
        )
    if cohort is None:
        raise RuntimeError("No replay cohort was constructed.")
    return batch_rows, _aggregate_gradient_rows(batch_rows), {
        **cohort,
        "checkpoint_guards": checkpoint_guards,
    }


def _plot_boundary_occupancy(rows: list[dict[str, Any]], path: Path) -> None:
    final = [
        row
        for row in rows
        if row["checkpoint_role"] == "final"
        and row["parameter_name"] != "ALL_CONDUCTANCE_WEIGHTS"
    ]
    arms = sorted(
        {row["arm_id"] for row in final},
        key=lambda value: (
            0 if value.startswith("conv1") else 1,
            ("baseline", "ours", "legacy").index(value.split("_")[1]),
            0 if value.endswith("sgd_seed0") else 1,
        ),
    )
    parameters = ("ConvWeight_0", "ConvWeight_1", "DenseWeight_0")
    matrix = np.full((len(arms), len(parameters)), np.nan)
    lookup = {(row["arm_id"], row["parameter_name"]): row for row in final}
    for row_index, arm in enumerate(arms):
        for column, parameter in enumerate(parameters):
            row = lookup.get((arm, parameter))
            if row is not None:
                matrix[row_index, column] = row["exact_lower_fraction"]
    figure, axis = plt.subplots(figsize=(8.5, 7.2), constrained_layout=True)
    image = axis.imshow(matrix, aspect="auto", vmin=0.0, vmax=max(0.3, np.nanmax(matrix)), cmap="magma")
    axis.set_xticks(range(len(parameters)), parameters, rotation=15)
    axis.set_yticks(range(len(arms)), [arm.removesuffix("_seed0") for arm in arms])
    axis.set_title("Final exact lower-bound occupancy ([0,100] weights)")
    for row_index in range(len(arms)):
        for column in range(len(parameters)):
            value = matrix[row_index, column]
            if math.isfinite(value):
                axis.text(column, row_index, f"{100.0 * value:.1f}%", ha="center", va="center", color="white" if value > 0.12 else "black", fontsize=8)
            else:
                axis.text(column, row_index, "—", ha="center", va="center", color="0.4")
    figure.colorbar(image, ax=axis, label="fraction exactly at 0")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_weight_histograms(runs: list[Path], path: Path) -> None:
    figure, axes = plt.subplots(4, 3, figsize=(12.0, 12.0), constrained_layout=True)
    architecture_rows = {"conv1": (0, 1), "conv2": (2, 3)}
    colors = {"baseline": "#4c78a8", "ours": "#e45756", "legacy": "#54a24b"}
    for run in runs:
        config = json.loads((run / "config.used.json").read_text(encoding="utf-8"))
        identity = _run_identity(run, config)
        weights = _load_npz_weights(run / "weights_final.npz")
        optimizer_offset = 0 if identity["optimizer"] == "SGD" else 1
        row = architecture_rows[identity["architecture"]][optimizer_offset]
        for column, parameter in enumerate(("ConvWeight_0", "ConvWeight_1", "DenseWeight_0")):
            axis = axes[row, column]
            if parameter not in weights:
                axis.set_axis_off()
                continue
            values = weights[parameter].reshape(-1).numpy()
            positive = values[values > 0.0]
            axis.hist(
                np.log10(np.maximum(positive, 1.0e-12)),
                bins=np.linspace(-8.0, 2.0, 81),
                density=True,
                histtype="step",
                linewidth=1.25,
                color=colors[identity["scheme"]],
                label=f"{identity['scheme']} (zero={100.0*np.mean(values == 0.0):.1f}%)",
            )
            axis.grid(alpha=0.2)
            axis.set_xlim(-8.0, 2.0)
            if row == 0:
                axis.set_title(parameter)
            if column == 0:
                axis.set_ylabel(f"{identity['architecture']} {identity['optimizer']}\ndensity")
            if row == 3:
                axis.set_xlabel("log10(weight), positive weights only")
            axis.legend(fontsize=7)
    figure.suptitle("Final conductance-weight distributions; zero mass shown in legends")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_projection(summary: list[dict[str, Any]], path: Path) -> None:
    arms = sorted(
        {row["arm_id"] for row in summary},
        key=lambda value: (
            0 if value.startswith("conv1") else 1,
            ("baseline", "ours", "legacy").index(value.split("_")[1]),
            0 if value.endswith("sgd_seed0") else 1,
        ),
    )
    parameters = ("ConvWeight_0", "ConvWeight_1", "DenseWeight_0")
    lookup = {(row["arm_id"], row["parameter_name"]): row for row in summary}
    metrics = (
        ("tangent_gradient_efficiency_l2_median", "feasible raw-gradient L2 fraction"),
        ("projection_efficiency_l2_median", "one-step proposal projection efficiency"),
        ("projection_changed_fraction_median", "proposal components clipped"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(15.0, 7.2), constrained_layout=True)
    for axis, (metric, title) in zip(axes, metrics, strict=True):
        matrix = np.full((len(arms), len(parameters)), np.nan)
        for row_index, arm in enumerate(arms):
            for column, parameter in enumerate(parameters):
                row = lookup.get((arm, parameter))
                if row is not None and row[metric] is not None:
                    matrix[row_index, column] = row[metric]
        image = axis.imshow(matrix, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
        axis.set_xticks(range(len(parameters)), parameters, rotation=20)
        axis.set_yticks(range(len(arms)), [arm.removesuffix("_seed0") for arm in arms])
        axis.set_title(title)
        for row_index in range(len(arms)):
            for column in range(len(parameters)):
                value = matrix[row_index, column]
                axis.text(column, row_index, "—" if not math.isfinite(value) else f"{value:.2f}", ha="center", va="center", color="white" if math.isfinite(value) and value < 0.45 else "black", fontsize=8)
        figure.colorbar(image, ax=axis, shrink=0.75)
    figure.suptitle("Final-checkpoint read-only replay (median over 16 minibatches)")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--replay-batches", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--production-source-commit", required=True)
    args = parser.parse_args()
    if args.replay_batches <= 0:
        raise ValueError("--replay-batches must be positive.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = _discover_runs(args.input_root.resolve())
    verified = _verify_artifacts(runs)
    distribution_rows, initial = _distribution_analysis(runs)
    unit_norm_rows = _unit_norm_rows(runs)
    gradient_batches, gradient_summary, cohort = _replay_gradients(
        runs,
        output_dir=args.output_dir,
        dataset_root=args.dataset_root,
        replay_batches=args.replay_batches,
        device=args.device,
    )
    _write_csv(args.output_dir / "weight_distribution_by_layer.csv", distribution_rows)
    _write_csv(args.output_dir / "weight_unit_norm_summary.csv", unit_norm_rows)
    _write_csv(args.output_dir / "gradient_projection_by_batch.csv", gradient_batches)
    _write_csv(args.output_dir / "gradient_projection_summary.csv", gradient_summary)
    _write_json(args.output_dir / "replay_cohort.json", cohort)
    _write_json(args.output_dir / "verified_checkpoints.json", verified)
    _write_json(
        args.output_dir / "analysis_manifest.json",
        {
            "schema_version": SCHEMA,
            "production_source_commit": args.production_source_commit,
            "input_root": str(args.input_root.resolve()),
            "output_dir": str(args.output_dir.resolve()),
            "run_count": len(runs),
            "verified_checkpoint_count": len(verified),
            "replay_batches_per_run": args.replay_batches,
            "replay_examples_per_run": cohort["example_count"],
            "replay_device": args.device,
            "optimizer_steps_applied": False,
            "initialization_tensor_hashes": {
                architecture: {
                    name: _tensor_hash(value)
                    for name, value in values.items()
                    if name.startswith(CONDUCTANCE_PREFIXES)
                }
                for architecture, values in initial.items()
            },
            "limitations": [
                "The production runs did not save optimizer state or per-step projection traces.",
                "SGD one-step proposal projection is exact for momentum=weight_decay=0.",
                "Adam one-step proposal projection uses a fresh-moment shadow proposal; raw tangent-gradient geometry is optimizer-state independent.",
                "Official test accuracy belongs to the best-validation checkpoint, whereas replay and headline distributions here use final checkpoints.",
            ],
        },
    )
    _plot_boundary_occupancy(
        distribution_rows, args.output_dir / "final_lower_bound_occupancy.png"
    )
    _plot_weight_histograms(runs, args.output_dir / "final_weight_histograms.png")
    _plot_projection(
        gradient_summary, args.output_dir / "final_gradient_projection.png"
    )
    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()
