"""Deterministic gradient replay for trained Conv LR-study checkpoints.

The replay uses a sparse, fixed subset of the original 32 probe minibatches.
It measures the production Backprop gradient, a same-free-state-settling
``K=64`` reference, and the update that the frozen per-parameter learning rate
would propose.  No optimizer step is taken and the official MNIST test split
is never instantiated.
"""

from __future__ import annotations

import copy
import itertools
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .lr_engine import (
    _unpack_batch,
    build_loader_bundle,
    build_model_runtime,
    canonical_parameter_name,
    parameter_tensor_digest,
)
from .saturation_replay import (
    _checkpoint_contract,
    _load_json,
    _row_by_id,
    _run_spec_path,
    _validate_run_spec_contract,
    _validate_study,
    _write_csv,
    infer_rho_targets,
    sha256_file,
)


GRADIENT_REPLAY_SCHEMA = "mnist-conv-lr-gradient-replay/v1"
DEFAULT_PROBE_BATCH_POSITIONS = tuple(range(8))
CHECKPOINT_ROLES = ("initialization", "best_validation", "final")
REFERENCE_CHECKPOINT_ROLES = ("initialization", "best_validation")


def _finite(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(
            f"Expected {label} to be finite. Provided value: {value!r}."
        )
    return result


def validate_probe_batch_positions(
    positions: Sequence[int],
    *,
    probe_batch_count: int = 32,
) -> tuple[int, ...]:
    """Validate and canonicalize selected positions in the frozen probe stream."""

    values = tuple(int(value) for value in positions)
    if not values:
        raise ValueError(
            f"Expected at least one probe batch position. Provided value: {values!r}."
        )
    if len(set(values)) != len(values):
        raise ValueError(
            "Expected unique probe batch positions. "
            f"Provided value: {values!r}."
        )
    if values != tuple(sorted(values)):
        raise ValueError(
            "Expected probe batch positions in increasing order. "
            f"Provided value: {values!r}."
        )
    invalid = [
        value for value in values if value < 0 or value >= int(probe_batch_count)
    ]
    if invalid:
        raise ValueError(
            f"Expected probe batch positions in [0,{int(probe_batch_count) - 1}]. "
            f"Provided value: {values!r}."
        )
    return values


def gradient_vector_comparison(vector: Any, reference: Any) -> dict[str, float]:
    """Compare two same-shaped gradient tensors in float64 on CPU."""

    import torch

    current = vector.detach().reshape(-1).to(dtype=torch.float64, device="cpu")
    target = reference.detach().reshape(-1).to(dtype=torch.float64, device="cpu")
    if tuple(current.shape) != tuple(target.shape):
        raise ValueError(
            "Expected gradient vectors with identical shapes. "
            f"Provided value: current={tuple(current.shape)!r}, "
            f"reference={tuple(target.shape)!r}."
        )
    norm = float(torch.linalg.vector_norm(current).item())
    reference_norm = float(torch.linalg.vector_norm(target).item())
    difference_norm = float(torch.linalg.vector_norm(current - target).item())
    cosine = (
        float(torch.dot(current, target).item() / (norm * reference_norm))
        if norm > 0.0 and reference_norm > 0.0
        else math.nan
    )
    return {
        "gradient_l2": norm,
        "reference_gradient_l2": reference_norm,
        "difference_l2": difference_norm,
        "reference_cosine": cosine,
        "reference_relative_error": (
            difference_norm / reference_norm
            if reference_norm > 0.0
            else math.nan
        ),
        "reference_norm_ratio": (
            norm / reference_norm if reference_norm > 0.0 else math.nan
        ),
    }


def summarize_gradient_vectors(vectors: Sequence[Any]) -> dict[str, float]:
    """Summarize scale, sparsity, tails, and cross-minibatch coherence."""

    import torch

    if not vectors:
        raise ValueError(
            f"Expected at least one gradient vector. Provided value: {len(vectors)}."
        )
    flattened = [
        value.detach().reshape(-1).to(dtype=torch.float64, device="cpu")
        for value in vectors
    ]
    shape = tuple(flattened[0].shape)
    if any(tuple(value.shape) != shape for value in flattened):
        raise ValueError(
            "Expected all gradient vectors to have identical shapes. "
            f"Provided value: {[tuple(value.shape) for value in flattened]!r}."
        )
    stack = torch.stack(flattened)
    rms = torch.sqrt(torch.mean(stack * stack, dim=1))
    norms = torch.linalg.vector_norm(stack, dim=1)
    zero_fraction = (stack.abs() <= 1.0e-12).to(torch.float64).mean(dim=1)
    absolute_p99 = torch.quantile(stack.abs(), 0.99, dim=1)
    tail_ratio = torch.where(
        rms > 0.0,
        absolute_p99 / rms,
        torch.full_like(rms, math.nan),
    )
    mean_vector = stack.mean(dim=0)
    rms_batch_norm = float(torch.sqrt(torch.mean(norms * norms)).item())
    coherence = (
        float(torch.linalg.vector_norm(mean_vector).item() / rms_batch_norm)
        if rms_batch_norm > 0.0
        else math.nan
    )
    pairwise_cosines: list[float] = []
    for left, right in itertools.combinations(range(len(flattened)), 2):
        denominator = float(norms[left].item() * norms[right].item())
        if denominator > 0.0:
            pairwise_cosines.append(
                float(torch.dot(stack[left], stack[right]).item() / denominator)
            )
    q = torch.quantile(
        rms, torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64)
    )
    pairwise = (
        torch.as_tensor(pairwise_cosines, dtype=torch.float64)
        if pairwise_cosines
        else torch.empty(0, dtype=torch.float64)
    )
    return {
        "gradient_rms_p10": float(q[0].item()),
        "gradient_rms_median": float(q[1].item()),
        "gradient_rms_p90": float(q[2].item()),
        "gradient_rms_mean": float(rms.mean().item()),
        "gradient_rms_batch_cv": (
            float(rms.std(unbiased=False).item() / rms.mean().item())
            if float(rms.mean().item()) > 0.0
            else math.nan
        ),
        "gradient_zero_fraction_median": float(zero_fraction.median().item()),
        "gradient_abs_p99_over_rms_median": float(tail_ratio.median().item()),
        "gradient_batch_coherence": coherence,
        "gradient_pairwise_cosine_median": (
            float(pairwise.median().item()) if pairwise.numel() else math.nan
        ),
    }


def _average_ranks(values: Any) -> Any:
    """Return average ranks for a one-dimensional float64 CPU tensor."""

    import torch

    vector = values.detach().reshape(-1).to(dtype=torch.float64, device="cpu")
    order = torch.argsort(vector, stable=True)
    ranks = torch.empty_like(vector)
    start = 0
    while start < int(vector.numel()):
        end = start + 1
        value = vector[order[start]]
        while end < int(vector.numel()) and bool(vector[order[end]] == value):
            end += 1
        average = 0.5 * float(start + end - 1)
        ranks[order[start:end]] = average
        start = end
    return ranks


def summarize_channel_activity_gradients(
    activity_by_batch: Sequence[Any],
    gradient_by_batch: Sequence[Any],
) -> tuple[dict[str, float | int], list[dict[str, float | int]]]:
    """Relate hidden-channel activity to parameter-gradient energy.

    The parameter's first axis must be the corresponding hidden-channel axis.
    This holds for the frozen Conv weights, spatial Conv biases, and the
    channel-first Dense readout tensors used in this study.
    """

    import torch

    if not activity_by_batch or len(activity_by_batch) != len(gradient_by_batch):
        raise ValueError(
            "Expected equal non-empty activity and gradient batch sequences. "
            f"Provided value: activity={len(activity_by_batch)}, "
            f"gradients={len(gradient_by_batch)}."
        )
    activity = torch.stack(
        [
            value.detach().reshape(-1).to(dtype=torch.float64, device="cpu")
            for value in activity_by_batch
        ]
    )
    gradients = [
        value.detach().to(dtype=torch.float64, device="cpu")
        for value in gradient_by_batch
    ]
    channels = int(activity.shape[1])
    if any(value.ndim < 1 or int(value.shape[0]) != channels for value in gradients):
        raise ValueError(
            "Expected each parameter gradient's first axis to match hidden channels. "
            f"Provided value: channels={channels}, "
            f"gradient_shapes={[tuple(value.shape) for value in gradients]!r}."
        )
    active_mean = activity.mean(dim=0)
    energy = torch.zeros(channels, dtype=torch.float64)
    element_count = 0
    zero_elements = torch.zeros(channels, dtype=torch.float64)
    for value in gradients:
        flattened = value.reshape(channels, -1)
        energy += torch.sum(flattened * flattened, dim=1)
        zero_elements += torch.sum(
            flattened.abs() <= 1.0e-12, dim=1, dtype=torch.float64
        )
        element_count += int(flattened.shape[1])
    gradient_rms = torch.sqrt(energy / float(element_count))
    zero_fraction = zero_elements / float(element_count)
    active_ranks = _average_ranks(active_mean)
    gradient_ranks = _average_ranks(gradient_rms)
    active_centered = active_ranks - active_ranks.mean()
    gradient_centered = gradient_ranks - gradient_ranks.mean()
    correlation_denominator = float(
        torch.linalg.vector_norm(active_centered).item()
        * torch.linalg.vector_norm(gradient_centered).item()
    )
    spearman = (
        float(
            torch.dot(active_centered, gradient_centered).item()
            / correlation_denominator
        )
        if correlation_denominator > 0.0
        else math.nan
    )
    quartile_count = max(1, int(math.ceil(channels / 4.0)))
    least_active = torch.argsort(active_mean, stable=True)[:quartile_count]
    total_energy = float(energy.sum().item())
    least_active_share = (
        float(energy[least_active].sum().item() / total_energy)
        if total_energy > 0.0
        else math.nan
    )
    channel_rows = [
        {
            "channel_index": channel,
            "active_fraction_mean": float(active_mean[channel].item()),
            "saturation_fraction_mean": float(1.0 - active_mean[channel].item()),
            "gradient_rms": float(gradient_rms[channel].item()),
            "gradient_energy": float(energy[channel].item()),
            "gradient_energy_fraction": (
                float(energy[channel].item() / total_energy)
                if total_energy > 0.0
                else math.nan
            ),
            "gradient_zero_fraction": float(zero_fraction[channel].item()),
            "in_least_active_quartile": bool(
                (least_active == channel).any().item()
            ),
        }
        for channel in range(channels)
    ]
    aggregate: dict[str, float | int] = {
        "channel_count": channels,
        "channel_active_gradient_spearman": spearman,
        "least_active_quartile_channel_count": quartile_count,
        "least_active_quartile_gradient_energy_share": least_active_share,
        "effectively_zero_gradient_channel_count": int(
            (gradient_rms <= 1.0e-12).sum().item()
        ),
        "channel_active_fraction_mean": float(active_mean.mean().item()),
        "channel_active_fraction_min": float(active_mean.min().item()),
        "channel_active_fraction_max": float(active_mean.max().item()),
    }
    return aggregate, channel_rows


def attached_weight_name(
    parameter_name: str,
    bias_weight_mapping: Mapping[str, str],
) -> str | None:
    """Return the normalization weight for a weight or attached Conv bias."""

    name = str(parameter_name)
    if name.startswith(("ConvWeight_", "DenseWeight_")):
        return name
    if name.startswith("Bias_"):
        if name not in bias_weight_mapping:
            raise ValueError(
                "Expected every bias to have an explicit attached-weight mapping. "
                f"Provided value: parameter={name!r}, "
                f"mapping={dict(bias_weight_mapping)!r}."
            )
        return str(bias_weight_mapping[name])
    return None


def _median(values: Sequence[float]) -> float:
    import torch

    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return math.nan
    return float(
        torch.quantile(
            torch.as_tensor(finite, dtype=torch.float64),
            torch.as_tensor(0.5, dtype=torch.float64),
        ).item()
    )


def _quantile(values: Sequence[float], q: float) -> float:
    import torch

    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return math.nan
    return float(
        torch.quantile(
            torch.as_tensor(finite, dtype=torch.float64),
            torch.as_tensor(float(q), dtype=torch.float64),
        ).item()
    )


def _probe_contract(
    *,
    study_root: Path,
    row_id: str,
    run_spec: Mapping[str, Any],
    generated_probe_batches: Sequence[Sequence[int]],
) -> dict[str, Any]:
    """Verify the exact 32 minibatches used to estimate the initial medians."""

    from labs.datasets import stable_batch_order_hash

    probe_dir = study_root / "stages" / "probe" / "entries" / row_id
    summary_path = probe_dir / "summary.json"
    minibatches_path = probe_dir / "minibatches.json"
    summary = _load_json(summary_path)
    minibatches = _load_json(minibatches_path)
    provenance = run_spec["run"]["lr_provenance"]
    expected_summary_hash = str(provenance["probe_sha256"])
    actual_summary_hash = sha256_file(summary_path)
    summary_hash_matches_candidate = actual_summary_hash == expected_summary_hash
    if not summary_hash_matches_candidate:
        # The v6 Conv2 bundles were converted from the original v5 probes and
        # therefore bind a converted probe-summary hash that is not part of the
        # local v5 study tree.  In that case validate every scientific field
        # used here instead of pretending the byte hashes are interchangeable.
        candidate_medians = {
            str(name): float(value)
            for name, value in provenance["median_unit_by_weight"].items()
        }
        local_medians = {
            str(name): float(value)
            for name, value in summary[
                "rho_unit_relative_q50_by_parameter"
            ].items()
        }
        local_contract = {
            "row_id": str(summary["row"]["row_id"]) == str(row_id),
            "initialization_tensor_sha256": str(
                summary["initial_parameter_tensor_sha256"]
            )
            == str(provenance["initialization_tensor_sha256"]),
            "median_unit_by_weight": candidate_medians == local_medians,
        }
        if not all(local_contract.values()):
            raise ValueError(
                "Expected a converted probe to match the local v5 scientific "
                f"contract. Provided value for {summary_path}: "
                f"candidate_hash={expected_summary_hash!r}, "
                f"local_hash={actual_summary_hash!r}, checks={local_contract!r}."
            )
    batches = tuple(
        tuple(int(index) for index in batch) for batch in minibatches["batches"]
    )
    expected_batches = tuple(
        tuple(int(index) for index in batch) for batch in generated_probe_batches
    )
    if len(batches) != 32 or batches != expected_batches:
        raise ValueError(
            "Expected the stored 32 probe minibatches to match the reconstructed "
            f"training stream. Provided value for {minibatches_path}: "
            f"stored_count={len(batches)}, expected_count={len(expected_batches)}."
        )
    actual_order_hash = stable_batch_order_hash(batches)
    recorded_hashes = {
        str(minibatches["batch_order_sha256"]),
        str(summary["minibatch_order_sha256"]),
    }
    if recorded_hashes != {actual_order_hash}:
        raise ValueError(
            "Expected probe minibatch hashes to agree with the reconstructed order. "
            f"Provided value: recorded={sorted(recorded_hashes)!r}, "
            f"actual={actual_order_hash!r}."
        )
    return {
        "probe_dir": str(probe_dir),
        "probe_summary_path": str(summary_path),
        "probe_summary_sha256": actual_summary_hash,
        "candidate_probe_summary_sha256": expected_summary_hash,
        "candidate_probe_summary_sha256_matches_local": (
            summary_hash_matches_candidate
        ),
        "probe_minibatches_path": str(minibatches_path),
        "probe_minibatches_sha256": sha256_file(minibatches_path),
        "probe_batch_order_sha256": actual_order_hash,
        "probe_batch_count": len(batches),
    }


def _selected_training_batches(
    bundle: Any,
    *,
    positions: Sequence[int],
    expected_probe_batches: Sequence[Sequence[int]],
) -> list[dict[str, Any]]:
    """Load selected original probe batches and verify their source indices."""

    selected = set(int(value) for value in positions)
    maximum = max(selected)
    result: list[dict[str, Any]] = []
    bundle.reset_train_shuffle()
    for batch_position, batch in enumerate(bundle.train_loader):
        if batch_position > maximum:
            break
        if batch_position not in selected:
            continue
        images, labels, source_indices = _unpack_batch(batch)
        expected_indices = tuple(
            int(value) for value in expected_probe_batches[batch_position]
        )
        if source_indices != expected_indices:
            raise RuntimeError(
                "Expected loaded training data to follow the frozen probe order. "
                f"Provided value at position {batch_position}: "
                f"loaded={source_indices!r}, expected={expected_indices!r}."
            )
        result.append(
            {
                "probe_batch_position": batch_position,
                "images": images.detach().cpu().clone(),
                "labels": labels.detach().cpu().clone(),
                "source_indices": source_indices,
            }
        )
    if tuple(row["probe_batch_position"] for row in result) != tuple(positions):
        raise RuntimeError(
            "Expected to load every requested probe batch position. "
            f"Provided value: requested={tuple(positions)!r}, "
            f"loaded={tuple(row['probe_batch_position'] for row in result)!r}."
        )
    return result


def _hidden_activity(runtime: Any, *, v_off: float) -> tuple[list[float], list[Any]]:
    import torch

    saturation: list[float] = []
    active_by_channel: list[Any] = []
    for layer in runtime.energy_fn.layers()[1:-1]:
        state = layer.state.detach()
        outside = (state < -float(v_off)) | (state > float(v_off))
        active = ~outside
        saturation.append(float(outside.to(torch.float64).mean().item()))
        active_by_channel.append(
            active.to(torch.float64).mean(dim=(0, 2, 3)).cpu()
        )
    return saturation, active_by_channel


def _compute_checkpoint_gradients(
    *,
    study: Mapping[str, Any],
    row: Mapping[str, Any],
    checkpoint_path: Path,
    batches: Sequence[Mapping[str, Any]],
    training_iterations: int,
    inference_iterations: int | None,
    device: str,
) -> dict[str, Any]:
    """Compute read-only gradients with one fresh runtime for one K value."""

    import torch

    runtime_row = dict(row)
    runtime_row["training_iterations"] = int(training_iterations)
    if inference_iterations is not None:
        runtime_row["inference_iterations"] = int(inference_iterations)
    runtime = build_model_runtime(
        study,
        runtime_row,
        device=device,
        initialization_checkpoint=checkpoint_path,
        learning_rate=0.0,
    )
    names = [canonical_parameter_name(value) for value in runtime.parameters]
    if len(set(names)) != len(names):
        raise RuntimeError(
            f"Expected unique canonical parameter names. Provided value: {names!r}."
        )
    parameter_types = {
        name: type(parameter).__name__
        for name, parameter in zip(names, runtime.parameters)
    }
    parameter_shapes = {
        name: [int(value) for value in parameter.state.shape]
        for name, parameter in zip(names, runtime.parameters)
    }
    parameter_states = {
        name: parameter.state.detach().to(device="cpu").clone()
        for name, parameter in zip(names, runtime.parameters)
    }
    digest_before = parameter_tensor_digest(runtime.parameters)
    records: list[dict[str, Any]] = []
    v_off = float(study["model"]["hard_sigmoid"]["v_off"])
    for batch in batches:
        images = batch["images"].to(runtime.device)
        labels = batch["labels"].to(runtime.device)
        runtime.network.set_input(images, reset=True)
        with torch.no_grad():
            runtime.minimizer_inference.compute_equilibrium()
        saturation, active_by_channel = _hidden_activity(runtime, v_off=v_off)
        runtime.cost_fn.set_target(labels)
        with torch.no_grad():
            cost = float(runtime.cost_fn.eval().mean().item())
            accuracy = float((~runtime.cost_fn.error_fn()).to(torch.float64).mean().item())
        gradients = runtime.estimator.compute_gradient()
        if len(gradients) < len(runtime.parameters):
            raise RuntimeError(
                f"Expected at least {len(runtime.parameters)} gradients. "
                f"Provided value: {len(gradients)}."
            )
        cloned: dict[str, Any] = {}
        for name, parameter, gradient in zip(
            names, runtime.parameters, gradients[: len(runtime.parameters)]
        ):
            if tuple(gradient.shape) != tuple(parameter.state.shape):
                raise RuntimeError(
                    f"Expected gradient {name!r} to have shape "
                    f"{tuple(parameter.state.shape)!r}. "
                    f"Provided value: {tuple(gradient.shape)!r}."
                )
            if not bool(torch.isfinite(gradient).all().item()):
                count = int((~torch.isfinite(gradient)).sum().item())
                raise ValueError(
                    f"Expected gradient {name!r} to contain only finite values. "
                    f"Provided value: {count} non-finite element(s)."
                )
            cloned[name] = gradient.detach().to(dtype=torch.float64, device="cpu").clone()
        records.append(
            {
                "probe_batch_position": int(batch["probe_batch_position"]),
                "source_indices": tuple(int(value) for value in batch["source_indices"]),
                "sample_count": int(images.shape[0]),
                "loss": cost,
                "accuracy": accuracy,
                "hidden_saturation_by_layer": saturation,
                "hidden_active_by_channel": active_by_channel,
                "gradients": cloned,
            }
        )
        del gradients
    digest_after = parameter_tensor_digest(runtime.parameters)
    if digest_after != digest_before:
        raise RuntimeError(
            "Expected gradient replay to leave parameter tensors unchanged. "
            f"Provided value: before={digest_before!r}, after={digest_after!r}."
        )
    for parameter in runtime.parameters:
        if bool(parameter.state.requires_grad):
            raise RuntimeError(
                "Expected gradient replay to restore requires_grad=False. "
                f"Provided value for {canonical_parameter_name(parameter)!r}: True."
            )
    result = {
        "parameter_names": names,
        "parameter_types": parameter_types,
        "parameter_shapes": parameter_shapes,
        "parameter_states": parameter_states,
        "parameter_tensor_sha256": digest_before,
        "records": records,
    }
    del runtime
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def _source_validation_metric(
    summary: Mapping[str, Any],
    *,
    checkpoint_role: str,
    checkpoint_epoch: int,
) -> Mapping[str, Any] | None:
    if checkpoint_role == "initialization":
        return None
    matches = [
        value
        for value in summary["validation_metrics"]
        if int(value["epoch"]) == int(checkpoint_epoch)
    ]
    if len(matches) != 1:
        raise ValueError(
            "Expected one full-validation record for the checkpoint epoch. "
            f"Provided value: role={checkpoint_role!r}, "
            f"epoch={checkpoint_epoch!r}, matches={len(matches)}."
        )
    return matches[0]


def _plot_gradient_summary(
    rows: Sequence[Mapping[str, Any]],
    scheme_comparisons: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    weight_rows = [row for row in rows if bool(row["is_weight"])]
    scheme_order = {"baseline": 0, "ours": 1, "legacy": 2}
    role_order = {role: index for index, role in enumerate(CHECKPOINT_ROLES)}
    colors = {
        "initialization": "#A5A5A5",
        "best_validation": "#4472C4",
        "final": "#ED7D31",
    }
    for architecture in sorted({str(row["architecture"]) for row in weight_rows}):
        architecture_rows = [
            row for row in weight_rows if row["architecture"] == architecture
        ]
        parameters = sorted(
            {str(row["parameter_name"]) for row in architecture_rows},
            key=lambda value: (
                1 if value.startswith("Dense") else 0,
                value,
            ),
        )
        fig, axes = plt.subplots(
            len(parameters),
            2,
            figsize=(12.5, 3.4 * len(parameters)),
            squeeze=False,
        )
        for parameter_index, parameter_name in enumerate(parameters):
            selected = sorted(
                [
                    row
                    for row in architecture_rows
                    if row["parameter_name"] == parameter_name
                ],
                key=lambda row: (
                    scheme_order[str(row["scheme"])],
                    role_order[str(row["checkpoint_role"])],
                ),
            )
            labels = [
                f"{row['scheme']}\n{str(row['checkpoint_role']).replace('_', ' ')}"
                for row in selected
            ]
            positions = np.arange(len(selected))
            raw = np.asarray(
                [float(row["relative_gradient_median"]) for row in selected]
            )
            scaled = np.asarray(
                [float(row["lr_scaled_relative_update_median"]) for row in selected]
            )
            bar_colors = [colors[str(row["checkpoint_role"])] for row in selected]
            axes[parameter_index, 0].bar(positions, raw, color=bar_colors)
            axes[parameter_index, 1].bar(positions, scaled, color=bar_colors)
            for axis, values in zip(axes[parameter_index], (raw, scaled)):
                axis.set_xticks(positions, labels)
                if np.any(np.isfinite(values) & (values > 0.0)):
                    axis.set_yscale("log")
                axis.grid(axis="y", alpha=0.25)
            axes[parameter_index, 0].set_ylabel("RMS(g) / RMS(W₀)")
            axes[parameter_index, 1].set_ylabel("LR × RMS(g) / RMS(W₀)")
            axes[parameter_index, 0].set_title(f"{parameter_name}: raw")
            axes[parameter_index, 1].set_title(f"{parameter_name}: LR-scaled")
            target_values = {
                float(row["target_rho"])
                for row in selected
                if math.isfinite(float(row["target_rho"]))
            }
            for target in sorted(target_values):
                axes[parameter_index, 1].axhline(
                    target, color="black", linestyle="--", linewidth=0.8
                )
        fig.suptitle(
            f"{architecture.upper()} checkpoint gradients on 8 frozen probe batches"
        )
        fig.tight_layout()
        fig.savefig(output_dir / f"{architecture}_gradient_scales.png", dpi=180)
        plt.close(fig)

        fig, axes = plt.subplots(
            len(parameters),
            2,
            figsize=(12.5, 3.4 * len(parameters)),
            squeeze=False,
        )
        for parameter_index, parameter_name in enumerate(parameters):
            selected = sorted(
                [
                    row
                    for row in architecture_rows
                    if row["parameter_name"] == parameter_name
                    and bool(row["reference_gradient_computed"])
                ],
                key=lambda row: (
                    scheme_order[str(row["scheme"])],
                    role_order[str(row["checkpoint_role"])],
                ),
            )
            labels = [
                f"{row['scheme']}\n{str(row['checkpoint_role']).replace('_', ' ')}"
                for row in selected
            ]
            positions = np.arange(len(selected))
            cosine = np.asarray(
                [float(row["reference_cosine_median"]) for row in selected]
            )
            error = np.asarray(
                [float(row["reference_relative_error_median"]) for row in selected]
            )
            bar_colors = [colors[str(row["checkpoint_role"])] for row in selected]
            axes[parameter_index, 0].bar(positions, cosine, color=bar_colors)
            axes[parameter_index, 1].bar(positions, error, color=bar_colors)
            for axis in axes[parameter_index]:
                axis.set_xticks(positions, labels)
                axis.tick_params(axis="x", labelrotation=25)
                axis.grid(axis="y", alpha=0.25)
            axes[parameter_index, 0].set_ylim(-1.0, 1.0)
            axes[parameter_index, 0].set_ylabel(
                "cos(g operational, g reference)"
            )
            axes[parameter_index, 1].set_ylabel(
                "‖g operational-g reference‖ / ‖g reference‖"
            )
            if np.any(np.isfinite(error) & (error > 0.0)):
                axes[parameter_index, 1].set_yscale("log")
            axes[parameter_index, 0].set_title(f"{parameter_name}: direction")
            axes[parameter_index, 1].set_title(f"{parameter_name}: relative error")
        reference_rows = [
            row
            for row in architecture_rows
            if bool(row["reference_gradient_computed"])
        ]
        reference_t_values = {
            int(row["reference_inference_iterations"])
            for row in reference_rows
        }
        changes_t = any(
            int(row["reference_inference_iterations"])
            != int(row["inference_iterations"])
            for row in reference_rows
        )
        if changes_t and len(reference_t_values) == 1:
            reference_title = (
                f"operational row T/K vs common "
                f"T={next(iter(reference_t_values))},"
                f"K={int(reference_rows[0]['reference_training_iterations'])}"
            )
        else:
            reference_title = (
                "operational K vs "
                f"K={int(reference_rows[0]['reference_training_iterations'])} "
                "at fixed row T"
            )
        fig.suptitle(
            f"{architecture.upper()} gradient reference: {reference_title}"
        )
        fig.tight_layout()
        fig.savefig(output_dir / f"{architecture}_gradient_k64_fidelity.png", dpi=180)
        plt.close(fig)

        geometry = [
            row
            for row in scheme_comparisons
            if row["architecture"] == architecture
            and row["checkpoint_role"] == "initialization"
            and row["gradient_kind"] == "reference"
            and bool(row["is_weight"])
        ]
        if geometry:
            geometry_parameters = sorted(
                {str(row["parameter_name"]) for row in geometry},
                key=lambda value: (
                    1 if value.startswith("Dense") else 0,
                    value,
                ),
            )
            amplified_schemes = ("ours", "legacy")
            width = 0.34
            fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
            x = np.arange(len(geometry_parameters))
            for offset_index, amplified_scheme in enumerate(amplified_schemes):
                values = {
                    str(row["parameter_name"]): row
                    for row in geometry
                    if row["scheme"] == amplified_scheme
                }
                positions = x + (offset_index - 0.5) * width
                axes[0].bar(
                    positions,
                    [
                        float(values[name]["gradient_cosine_median"])
                        for name in geometry_parameters
                    ],
                    width,
                    label=amplified_scheme,
                )
                axes[1].bar(
                    positions,
                    [
                        float(values[name]["raw_gradient_norm_ratio_median"])
                        for name in geometry_parameters
                    ],
                    width,
                    label=amplified_scheme,
                )
            for axis in axes:
                axis.set_xticks(x, geometry_parameters)
                axis.grid(axis="y", alpha=0.25)
                axis.legend()
            axes[0].set_ylim(-1.0, 1.0)
            axes[0].set_ylabel("cos(g amplified, g baseline)")
            axes[1].set_yscale("log")
            axes[1].set_ylabel("‖g amplified‖ / ‖g baseline‖")
            if (
                geometry[0]["reference_definition"]
                == "common_reference_T_and_training_K"
            ):
                geometry_reference = (
                    f"common T={int(geometry[0]['reference_inference_iterations'])},"
                    f"K={int(geometry[0]['training_iterations'])}"
                )
            else:
                geometry_reference = (
                    f"fixed row T, K={int(geometry[0]['training_iterations'])}"
                )
            fig.suptitle(
                f"{architecture.upper()} initialization: amplification gradient "
                f"geometry ({geometry_reference})"
            )
            fig.tight_layout()
            fig.savefig(
                output_dir / f"{architecture}_amplification_gradient_geometry.png",
                dpi=180,
            )
            plt.close(fig)


def _scheme_comparison_rows(
    payloads: Mapping[tuple[str, str, str, str], Mapping[str, Any]],
    metadata: Mapping[tuple[str, str, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Compare amplified gradients with baseline in the same parameter space."""

    rows: list[dict[str, Any]] = []
    architectures = sorted({key[0] for key in payloads})
    checkpoint_roles = sorted(
        {key[1] for key in payloads},
        key=lambda value: CHECKPOINT_ROLES.index(value),
    )
    for architecture in architectures:
        for checkpoint_role in checkpoint_roles:
            for gradient_kind in ("operational", "reference"):
                baseline_key = (
                    architecture,
                    checkpoint_role,
                    "baseline",
                    gradient_kind,
                )
                if baseline_key not in payloads:
                    continue
                baseline = payloads[baseline_key]
                baseline_metadata = metadata[
                    (architecture, checkpoint_role, "baseline")
                ]
                for scheme in ("ours", "legacy"):
                    amplified_key = (
                        architecture,
                        checkpoint_role,
                        scheme,
                        gradient_kind,
                    )
                    if amplified_key not in payloads:
                        continue
                    amplified = payloads[amplified_key]
                    amplified_metadata = metadata[
                        (architecture, checkpoint_role, scheme)
                    ]
                    if (
                        baseline["parameter_names"]
                        != amplified["parameter_names"]
                    ):
                        raise RuntimeError(
                            "Expected schemes in one architecture to expose the same "
                            "parameter order. "
                            f"Provided value: baseline={baseline['parameter_names']!r}, "
                            f"{scheme}={amplified['parameter_names']!r}."
                        )
                    for parameter_name in baseline["parameter_names"]:
                        comparisons: list[dict[str, float]] = []
                        norm_ratios: list[float] = []
                        update_ratios: list[float] = []
                        for baseline_record, amplified_record in zip(
                            baseline["records"], amplified["records"]
                        ):
                            if baseline_record["probe_batch_position"] != amplified_record[
                                "probe_batch_position"
                            ]:
                                raise RuntimeError(
                                    "Expected cross-scheme comparison to use identical "
                                    "probe batch positions."
                                )
                            comparison = gradient_vector_comparison(
                                amplified_record["gradients"][parameter_name],
                                baseline_record["gradients"][parameter_name],
                            )
                            comparisons.append(comparison)
                            ratio = comparison["reference_norm_ratio"]
                            norm_ratios.append(ratio)
                            update_ratios.append(
                                ratio
                                * float(
                                    amplified_metadata["learning_rates"][
                                        parameter_name
                                    ]
                                )
                                / float(
                                    baseline_metadata["learning_rates"][
                                        parameter_name
                                    ]
                                )
                            )
                        finite_logs = [
                            math.log(value)
                            for value in norm_ratios
                            if math.isfinite(value) and value > 0.0
                        ]
                        log_median = _median(finite_logs)
                        log_mad = _median(
                            [abs(value - log_median) for value in finite_logs]
                        )
                        rows.append(
                            {
                                "architecture": architecture,
                                "checkpoint_role": checkpoint_role,
                                "comparison_scope": (
                                    "amplification_scheme_shared_initial_weights"
                                    if checkpoint_role == "initialization"
                                    else "trained_trajectory_endpoint"
                                ),
                                "gradient_kind": gradient_kind,
                                "training_iterations": (
                                    amplified_metadata[
                                        "reference_training_iterations"
                                    ]
                                    if gradient_kind == "reference"
                                    else amplified_metadata[
                                        "operational_training_iterations"
                                    ]
                                ),
                                "baseline_training_iterations": (
                                    baseline_metadata[
                                        "reference_training_iterations"
                                    ]
                                    if gradient_kind == "reference"
                                    else baseline_metadata[
                                        "operational_training_iterations"
                                    ]
                                ),
                                "inference_iterations": amplified_metadata[
                                    "operational_inference_iterations"
                                ],
                                "baseline_inference_iterations": baseline_metadata[
                                    "operational_inference_iterations"
                                ],
                                "reference_inference_iterations": amplified_metadata[
                                    "reference_inference_iterations"
                                ],
                                "baseline_reference_inference_iterations": (
                                    baseline_metadata[
                                        "reference_inference_iterations"
                                    ]
                                ),
                                "reference_definition": amplified_metadata[
                                    "reference_definition"
                                ],
                                "scheme": scheme,
                                "baseline_scheme": "baseline",
                                "parameter_name": parameter_name,
                                "is_weight": parameter_name.startswith(
                                    ("ConvWeight_", "DenseWeight_")
                                ),
                                "is_bias": parameter_name.startswith("Bias_"),
                                "batch_count": len(comparisons),
                                "gradient_cosine_median": _median(
                                    [
                                        value["reference_cosine"]
                                        for value in comparisons
                                    ]
                                ),
                                "gradient_cosine_p10": _quantile(
                                    [
                                        value["reference_cosine"]
                                        for value in comparisons
                                    ],
                                    0.1,
                                ),
                                "raw_gradient_norm_ratio_median": _median(
                                    norm_ratios
                                ),
                                "raw_gradient_log_norm_ratio_mad": log_mad,
                                "lr_scaled_update_norm_ratio_median": _median(
                                    update_ratios
                                ),
                                "amplified_learning_rate": amplified_metadata[
                                    "learning_rates"
                                ][parameter_name],
                                "baseline_learning_rate": baseline_metadata[
                                    "learning_rates"
                                ][parameter_name],
                            }
                        )
    return rows


def replay_gradients(
    *,
    study_path: str | Path,
    entry_dirs: Iterable[str | Path],
    initialization_dir: str | Path,
    data_root: str | Path,
    output_dir: str | Path,
    device: str,
    probe_batch_positions: Sequence[int] = DEFAULT_PROBE_BATCH_POSITIONS,
    reference_training_iterations: int = 64,
    reference_inference_iterations: int | None = None,
) -> dict[str, Any]:
    """Replay gradients for initial, best, and final checkpoints."""

    import torch
    from labs.datasets import (
        stable_batch_order_hash,
        stable_index_sequence_hash,
    )

    positions = validate_probe_batch_positions(probe_batch_positions)
    if int(reference_training_iterations) <= 0:
        raise ValueError(
            "Expected reference_training_iterations to be positive. "
            f"Provided value: {reference_training_iterations!r}."
        )
    if (
        reference_inference_iterations is not None
        and int(reference_inference_iterations) <= 0
    ):
        raise ValueError(
            "Expected reference_inference_iterations to be positive when provided. "
            f"Provided value: {reference_inference_iterations!r}."
        )
    study_source = Path(study_path).expanduser().resolve()
    study = _load_json(study_source)
    _validate_study(study)
    runtime_study = copy.deepcopy(study)
    runtime_study["dataset"]["train"]["batch_size"] = 16
    runtime_study["dataset"]["validation"]["batch_size"] = 16
    bundle = build_loader_bundle(
        runtime_study,
        data_root=data_root,
        download=False,
        return_source_indices=True,
    )
    generated_epochs = bundle.train_batch_indices(num_epochs=5)
    generated_all_batches = tuple(
        batch for epoch in generated_epochs for batch in epoch
    )
    generated_candidate_hash = stable_batch_order_hash(generated_all_batches)
    generated_probe_batches = tuple(generated_epochs[0][:32])
    generated_probe_hash = stable_batch_order_hash(generated_probe_batches)
    batches = _selected_training_batches(
        bundle,
        positions=positions,
        expected_probe_batches=generated_probe_batches,
    )
    selected_indices = tuple(
        index for batch in batches for index in batch["source_indices"]
    )

    entries = sorted({Path(value).expanduser().resolve() for value in entry_dirs})
    if not entries:
        raise ValueError(
            f"Expected at least one candidate entry. Provided value: {entries!r}."
        )
    initialization_root = Path(initialization_dir).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    cases: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    for entry_dir in entries:
        summary_path = entry_dir / "summary.json"
        run_spec_path = _run_spec_path(entry_dir)
        summary = _load_json(summary_path)
        run_spec = _load_json(run_spec_path)
        run_spec_hash = sha256_file(run_spec_path)
        recorded_run_spec_hashes = [
            str(value)
            for key, value in summary.items()
            if key.startswith("run_spec_v") and key.endswith("_sha256")
        ]
        if recorded_run_spec_hashes != [run_spec_hash]:
            raise ValueError(
                "Expected exactly one matching run-spec SHA-256 in the candidate "
                f"summary. Provided value for {entry_dir}: "
                f"recorded={recorded_run_spec_hashes!r}, actual={run_spec_hash!r}."
            )
        row_id = str(summary["row"]["row_id"])
        row = _row_by_id(runtime_study, row_id)
        if dict(row) != dict(summary["row"]):
            raise ValueError(
                "Expected candidate row metadata to match the runtime study. "
                f"Provided value for {entry_dir}: study={dict(row)!r}, "
                f"candidate={summary['row']!r}."
            )
        _validate_run_spec_contract(runtime_study, row, run_spec)
        rho_conv, rho_dense = infer_rho_targets(run_spec)
        if not math.isclose(rho_conv, 1e-2, rel_tol=0.0, abs_tol=1e-15) or not math.isclose(
            rho_dense, 1e-2, rel_tol=0.0, abs_tol=1e-15
        ):
            raise ValueError(
                "Expected the matched rho_conv=rho_dense=1e-2 comparison. "
                f"Provided value for {entry_dir}: "
                f"rho_conv={rho_conv!r}, rho_dense={rho_dense!r}."
            )
        if str(summary["validation_indices_sha256"]) != bundle.validation_indices_hash:
            raise RuntimeError(
                "Expected candidate validation indices to match the reconstructed "
                f"split. Provided value for {entry_dir}: "
                f"candidate={summary['validation_indices_sha256']!r}, "
                f"actual={bundle.validation_indices_hash!r}."
            )
        provenance = run_spec["run"]["lr_provenance"]
        if str(provenance["batch_order_sha256"]) != generated_candidate_hash:
            raise RuntimeError(
                "Expected candidate minibatches to match the reconstructed five-epoch "
                f"stream. Provided value for {entry_dir}: "
                f"candidate={provenance['batch_order_sha256']!r}, "
                f"actual={generated_candidate_hash!r}."
            )
        probe = _probe_contract(
            study_root=study_source.parent,
            row_id=row_id,
            run_spec=run_spec,
            generated_probe_batches=generated_probe_batches,
        )
        learning_rates = {
            str(name): _finite(value, f"learning rate for {name!r}")
            for name, value in run_spec["run"]["training"][
                "learning_rates_by_parameter"
            ].items()
        }
        initial_medians = {
            str(name): _finite(value, f"initial median unit for {name!r}")
            for name, value in provenance["median_unit_by_weight"].items()
        }
        bias_mapping = {
            str(name): str(weight)
            for name, weight in provenance["bias_weight_mapping"].items()
        }
        checkpoints = _checkpoint_contract(
            entry_dir, summary, run_spec, initialization_root
        )
        case = {
            "entry_dir": entry_dir,
            "summary": summary,
            "summary_path": summary_path,
            "run_spec": run_spec,
            "run_spec_path": run_spec_path,
            "run_spec_hash": run_spec_hash,
            "row": row,
            "row_id": row_id,
            "rho_conv": rho_conv,
            "rho_dense": rho_dense,
            "learning_rates": learning_rates,
            "initial_medians": initial_medians,
            "bias_mapping": bias_mapping,
            "checkpoints": checkpoints,
            "probe": probe,
        }
        cases.append(case)
        source_rows.append(
            {
                "entry_id": entry_dir.name,
                "entry_dir": str(entry_dir),
                "row_id": row_id,
                "architecture": str(row["architecture"]),
                "scheme": str(row["scheme"]),
                "summary_path": str(summary_path),
                "summary_sha256": sha256_file(summary_path),
                "run_spec_path": str(run_spec_path),
                "run_spec_sha256": run_spec_hash,
                **probe,
            }
        )

    batch_rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    saturation_rows: list[dict[str, Any]] = []
    channel_rows: list[dict[str, Any]] = []
    comparison_payloads: dict[
        tuple[str, str, str, str], Mapping[str, Any]
    ] = {}
    comparison_metadata: dict[
        tuple[str, str, str], Mapping[str, Any]
    ] = {}

    for case in cases:
        row = case["row"]
        summary = case["summary"]
        initial_states: dict[str, Any] | None = None
        initial_scales: dict[str, float] | None = None
        for checkpoint in case["checkpoints"]:
            checkpoint_path = Path(checkpoint["path"])
            if not checkpoint_path.is_file():
                raise FileNotFoundError(
                    "Expected checkpoint file to exist. "
                    f"Provided value: {checkpoint_path}."
                )
            file_hash_before = sha256_file(checkpoint_path)
            if file_hash_before != checkpoint["expected_file_sha256"]:
                raise ValueError(
                    "Expected checkpoint SHA-256 to match its immutable manifest. "
                    f"Provided value for {checkpoint_path}: "
                    f"expected={checkpoint['expected_file_sha256']!r}, "
                    f"actual={file_hash_before!r}."
                )
            operational_k = int(row["training_iterations"])
            operational = _compute_checkpoint_gradients(
                study=runtime_study,
                row=row,
                checkpoint_path=checkpoint_path,
                batches=batches,
                training_iterations=operational_k,
                inference_iterations=None,
                device=device,
            )
            reference = None
            if checkpoint["checkpoint_role"] in REFERENCE_CHECKPOINT_ROLES:
                reference = _compute_checkpoint_gradients(
                    study=runtime_study,
                    row=row,
                    checkpoint_path=checkpoint_path,
                    batches=batches,
                    training_iterations=int(reference_training_iterations),
                    inference_iterations=reference_inference_iterations,
                    device=device,
                )
                if operational["parameter_names"] != reference["parameter_names"]:
                    raise RuntimeError(
                        "Expected operational and reference parameter ordering to "
                        f"match. Provided value: "
                        f"operational={operational['parameter_names']!r}, "
                        f"reference={reference['parameter_names']!r}."
                    )
            expected_parameter_hash = checkpoint["expected_parameter_sha256"]
            actual_parameter_hash = operational["parameter_tensor_sha256"]
            if (
                expected_parameter_hash is not None
                and actual_parameter_hash != expected_parameter_hash
            ):
                raise ValueError(
                    "Expected loaded parameter tensors to match checkpoint provenance. "
                    f"Provided value for {checkpoint_path}: "
                    f"expected={expected_parameter_hash!r}, "
                    f"actual={actual_parameter_hash!r}."
                )
            if (
                reference is not None
                and reference["parameter_tensor_sha256"] != actual_parameter_hash
            ):
                raise RuntimeError(
                    "Expected operational and reference runtimes to load identical "
                    f"parameters. Provided value: operational={actual_parameter_hash!r}, "
                    f"reference={reference['parameter_tensor_sha256']!r}."
                )
            if checkpoint["checkpoint_role"] == "initialization":
                initial_states = {
                    name: value.clone()
                    for name, value in operational["parameter_states"].items()
                }
                initial_scales = {
                    name: float(
                        torch.sqrt(
                            torch.mean(value.to(dtype=torch.float64) ** 2)
                        ).item()
                    )
                    for name, value in initial_states.items()
                    if name.startswith(("ConvWeight_", "DenseWeight_"))
                }
                for name, recorded in summary[
                    "initial_bounded_parameter_rms"
                ].items():
                    actual = initial_scales[str(name)]
                    if not math.isclose(
                        actual, float(recorded), rel_tol=1e-11, abs_tol=1e-13
                    ):
                        raise ValueError(
                            "Expected initialization RMS to match candidate summary. "
                            f"Provided value for {name!r}: "
                            f"summary={recorded!r}, actual={actual!r}."
                        )
            if initial_states is None or initial_scales is None:
                raise RuntimeError(
                    "Expected initialization checkpoint to be processed before "
                    f"{checkpoint['checkpoint_role']!r}."
                )
            source_validation = _source_validation_metric(
                summary,
                checkpoint_role=str(checkpoint["checkpoint_role"]),
                checkpoint_epoch=int(checkpoint["checkpoint_epoch"]),
            )
            common = {
                "entry_id": case["entry_dir"].name,
                "row_id": case["row_id"],
                "architecture": str(row["architecture"]),
                "scheme": str(row["scheme"]),
                "rho_conv": case["rho_conv"],
                "rho_dense": case["rho_dense"],
                "checkpoint_role": checkpoint["checkpoint_role"],
                "checkpoint_epoch": checkpoint["checkpoint_epoch"],
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_file_sha256": file_hash_before,
                "parameter_tensor_sha256": actual_parameter_hash,
                "inference_iterations": int(row["inference_iterations"]),
                "operational_training_iterations": operational_k,
                "reference_training_iterations": int(
                    reference_training_iterations
                ),
                "reference_inference_iterations": (
                    int(reference_inference_iterations)
                    if reference_inference_iterations is not None
                    else int(row["inference_iterations"])
                ),
                "reference_gradient_computed": reference is not None,
                "batch_size": 16,
                "probe_batch_count": len(batches),
                "sample_count": sum(int(value["sample_count"]) for value in operational["records"]),
                "selected_source_indices_sha256": stable_index_sequence_hash(
                    selected_indices
                ),
                "source_operational_full_validation_loss": (
                    float(source_validation["loss"])
                    if source_validation is not None
                    else None
                ),
                "source_operational_full_validation_accuracy": (
                    float(source_validation["accuracy"])
                    if source_validation is not None
                    else None
                ),
            }
            comparison_key = (
                str(row["architecture"]),
                str(checkpoint["checkpoint_role"]),
                str(row["scheme"]),
            )
            comparison_payloads[(*comparison_key, "operational")] = operational
            if reference is not None:
                comparison_payloads[(*comparison_key, "reference")] = reference
            comparison_metadata[comparison_key] = {
                "learning_rates": dict(case["learning_rates"]),
                "operational_training_iterations": operational_k,
                "operational_inference_iterations": int(
                    row["inference_iterations"]
                ),
                "reference_training_iterations": int(
                    reference_training_iterations
                ),
                "reference_inference_iterations": (
                    int(reference_inference_iterations)
                    if reference_inference_iterations is not None
                    else int(row["inference_iterations"])
                ),
                "reference_definition": (
                    "common_reference_T_and_training_K"
                    if reference_inference_iterations is not None
                    else "same_operational_T_reference_training_K"
                ),
            }
            losses = [float(value["loss"]) for value in operational["records"]]
            accuracies = [
                float(value["accuracy"]) for value in operational["records"]
            ]
            checkpoint_rows.append(
                {
                    **common,
                    "probe_loss_mean": sum(losses) / len(losses),
                    "probe_accuracy_mean": sum(accuracies) / len(accuracies),
                }
            )
            layer_count = len(
                operational["records"][0]["hidden_saturation_by_layer"]
            )
            for layer_index in range(layer_count):
                values = [
                    float(value["hidden_saturation_by_layer"][layer_index])
                    for value in operational["records"]
                ]
                saturation_rows.append(
                    {
                        **common,
                        "layer_index": layer_index,
                        "layer_name": f"Hidden_{layer_index}",
                        "saturation_fraction_mean": sum(values) / len(values),
                        "saturation_fraction_median": _median(values),
                        "saturation_fraction_p90": _quantile(values, 0.9),
                    }
                )

            for parameter_name in operational["parameter_names"]:
                parameter_vectors = [
                    value["gradients"][parameter_name]
                    for value in operational["records"]
                ]
                reference_vectors = (
                    [
                        value["gradients"][parameter_name]
                        for value in reference["records"]
                    ]
                    if reference is not None
                    else []
                )
                vector_summary = summarize_gradient_vectors(parameter_vectors)
                comparisons = (
                    [
                        gradient_vector_comparison(value, target)
                        for value, target in zip(
                            parameter_vectors, reference_vectors
                        )
                    ]
                    if reference is not None
                    else [
                        {
                            "gradient_l2": math.nan,
                            "reference_gradient_l2": math.nan,
                            "difference_l2": math.nan,
                            "reference_cosine": math.nan,
                            "reference_relative_error": math.nan,
                            "reference_norm_ratio": math.nan,
                        }
                        for _ in parameter_vectors
                    ]
                )
                normalization_weight = attached_weight_name(
                    parameter_name, case["bias_mapping"]
                )
                is_weight = parameter_name.startswith(
                    ("ConvWeight_", "DenseWeight_")
                )
                is_bias = parameter_name.startswith("Bias_")
                learning_rate = case["learning_rates"][parameter_name]
                normalization_rms = (
                    initial_scales[normalization_weight]
                    if normalization_weight is not None
                    else math.nan
                )
                relative_gradient = (
                    vector_summary["gradient_rms_median"] / normalization_rms
                    if normalization_rms > 0.0
                    else math.nan
                )
                initial_median = (
                    case["initial_medians"][parameter_name]
                    if is_weight
                    else math.nan
                )
                target_rho = (
                    case["rho_conv"]
                    if parameter_name.startswith("ConvWeight_")
                    else (
                        case["rho_dense"]
                        if parameter_name.startswith("DenseWeight_")
                        else math.nan
                    )
                )
                current_state = operational["parameter_states"][parameter_name]
                initial_state = initial_states[parameter_name]
                displacement_rms = float(
                    torch.sqrt(
                        torch.mean(
                            (
                                current_state.to(dtype=torch.float64)
                                - initial_state.to(dtype=torch.float64)
                            )
                            ** 2
                        )
                    ).item()
                )
                displacement_relative = (
                    displacement_rms / normalization_rms
                    if normalization_rms > 0.0
                    else math.nan
                )
                if parameter_name.startswith("ConvWeight_"):
                    hidden_layer_index = int(parameter_name.rsplit("_", 1)[1])
                elif parameter_name.startswith("Bias_"):
                    hidden_layer_index = int(parameter_name.rsplit("_", 1)[1])
                elif parameter_name.startswith("DenseWeight_"):
                    hidden_layer_index = layer_count - 1
                else:
                    hidden_layer_index = None
                channel_summary: dict[str, Any] = {}
                parameter_channel_rows: list[dict[str, Any]] = []
                if hidden_layer_index is not None:
                    channel_summary, parameter_channel_rows = (
                        summarize_channel_activity_gradients(
                            [
                                value["hidden_active_by_channel"][
                                    hidden_layer_index
                                ]
                                for value in operational["records"]
                            ],
                            parameter_vectors,
                        )
                    )
                parameter_common = {
                    **common,
                    "parameter_index": operational["parameter_names"].index(
                        parameter_name
                    ),
                    "parameter_name": parameter_name,
                    "parameter_type": operational["parameter_types"][
                        parameter_name
                    ],
                    "parameter_shape": "x".join(
                        str(value)
                        for value in operational["parameter_shapes"][
                            parameter_name
                        ]
                    ),
                    "is_weight": is_weight,
                    "is_bias": is_bias,
                    "attached_weight_name": (
                        normalization_weight if is_bias else None
                    ),
                    "learning_rate": learning_rate,
                    "normalization_initial_weight_rms": normalization_rms,
                    "initial_probe_median_unit": initial_median,
                    "target_rho": target_rho,
                    "relative_gradient_median": relative_gradient,
                    "lr_scaled_relative_update_median": (
                        learning_rate * relative_gradient
                    ),
                    "gradient_drift_from_initial_probe": (
                        relative_gradient / initial_median
                        if is_weight and initial_median > 0.0
                        else math.nan
                    ),
                    "parameter_displacement_rms": displacement_rms,
                    "parameter_displacement_relative_to_initial_weight": (
                        displacement_relative
                    ),
                    "hidden_layer_index": hidden_layer_index,
                    **channel_summary,
                    **vector_summary,
                    "reference_cosine_median": _median(
                        [value["reference_cosine"] for value in comparisons]
                    ),
                    "reference_cosine_p10": _quantile(
                        [value["reference_cosine"] for value in comparisons], 0.1
                    ),
                    "reference_relative_error_median": _median(
                        [
                            value["reference_relative_error"]
                            for value in comparisons
                        ]
                    ),
                    "reference_relative_error_p90": _quantile(
                        [
                            value["reference_relative_error"]
                            for value in comparisons
                        ],
                        0.9,
                    ),
                    "reference_norm_ratio_median": _median(
                        [value["reference_norm_ratio"] for value in comparisons]
                    ),
                }
                parameter_rows.append(parameter_common)
                channel_rows.extend(
                    {
                        **common,
                        "parameter_name": parameter_name,
                        "hidden_layer_index": hidden_layer_index,
                        **value,
                    }
                    for value in parameter_channel_rows
                )
                for record, reference_record, comparison in zip(
                    operational["records"],
                    (
                        reference["records"]
                        if reference is not None
                        else operational["records"]
                    ),
                    comparisons,
                ):
                    vector = record["gradients"][parameter_name]
                    gradient_rms = float(
                        torch.sqrt(torch.mean(vector * vector)).item()
                    )
                    batch_rows.append(
                        {
                            **common,
                            "probe_batch_position": record[
                                "probe_batch_position"
                            ],
                            "batch_source_indices_sha256": stable_index_sequence_hash(
                                record["source_indices"]
                            ),
                            "parameter_name": parameter_name,
                            "learning_rate": learning_rate,
                            "gradient_rms": gradient_rms,
                            "relative_gradient": (
                                gradient_rms / normalization_rms
                                if normalization_rms > 0.0
                                else math.nan
                            ),
                            "lr_scaled_relative_update": (
                                learning_rate * gradient_rms / normalization_rms
                                if normalization_rms > 0.0
                                else math.nan
                            ),
                            "gradient_zero_fraction": float(
                                (vector.abs() <= 1.0e-12)
                                .to(torch.float64)
                                .mean()
                                .item()
                            ),
                            **comparison,
                            "reference_probe_batch_position": reference_record[
                                "probe_batch_position"
                            ],
                        }
                    )
            file_hash_after = sha256_file(checkpoint_path)
            if file_hash_after != file_hash_before:
                raise RuntimeError(
                    "Expected gradient replay to leave checkpoint bytes unchanged. "
                    f"Provided value: before={file_hash_before!r}, "
                    f"after={file_hash_after!r}."
                )

    scheme_comparison_rows = _scheme_comparison_rows(
        comparison_payloads, comparison_metadata
    )
    cohort = {
        "source": "mnist_train_training_split",
        "selection": "first_8_original_32_probe_batches/v1",
        "official_test_read": False,
        "batch_size": 16,
        "probe_batch_positions": list(positions),
        "probe_batch_count": len(batches),
        "sample_count": len(selected_indices),
        "source_indices": list(selected_indices),
        "source_indices_sha256": stable_index_sequence_hash(selected_indices),
        "complete_probe_batch_order_sha256": generated_probe_hash,
        "complete_candidate_batch_order_sha256": generated_candidate_hash,
        "train_indices_sha256": bundle.train_indices_hash,
        "validation_indices_sha256": bundle.validation_indices_hash,
    }
    result = {
        "schema_version": GRADIENT_REPLAY_SCHEMA,
        "study_path": str(study_source),
        "study_sha256": sha256_file(study_source),
        "device": str(device),
        "gradient_estimator": "Backprop",
        "reference_definition": (
            "common_reference_T_and_training_K"
            if reference_inference_iterations is not None
            else "same_operational_T_reference_training_K"
        ),
        "reference_training_iterations": int(reference_training_iterations),
        "reference_inference_iterations": (
            int(reference_inference_iterations)
            if reference_inference_iterations is not None
            else None
        ),
        "reference_checkpoint_roles": list(REFERENCE_CHECKPOINT_ROLES),
        "cohort": cohort,
        "sources": source_rows,
        "checkpoints": checkpoint_rows,
        "hidden_saturation": saturation_rows,
        "parameters": parameter_rows,
        "channels": channel_rows,
        "scheme_comparisons": scheme_comparison_rows,
    }
    _write_csv(destination / "checkpoint_gradient_summary.csv", checkpoint_rows)
    _write_csv(destination / "parameter_gradient_summary.csv", parameter_rows)
    _write_csv(destination / "batch_parameter_gradients.csv", batch_rows)
    _write_csv(destination / "training_hidden_saturation.csv", saturation_rows)
    _write_csv(destination / "channel_activity_gradients.csv", channel_rows)
    if scheme_comparison_rows:
        _write_csv(
            destination / "amplification_gradient_comparisons.csv",
            scheme_comparison_rows,
        )
    (destination / "cohort.json").write_text(
        json.dumps(cohort, sort_keys=True, separators=(",", ":")) + "\n"
    )
    (destination / "summary.json").write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n"
    )
    _plot_gradient_summary(parameter_rows, scheme_comparison_rows, destination)
    return result


__all__ = [
    "DEFAULT_PROBE_BATCH_POSITIONS",
    "GRADIENT_REPLAY_SCHEMA",
    "attached_weight_name",
    "gradient_vector_comparison",
    "replay_gradients",
    "summarize_channel_activity_gradients",
    "summarize_gradient_vectors",
    "validate_probe_batch_positions",
]
