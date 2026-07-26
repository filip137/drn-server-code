"""Deterministic saturation replay for trained Conv LR-study checkpoints.

This module deliberately reuses the v5/v6 numerical runtime and the
MNIST-train-only validation loader.  It never instantiates the official MNIST
test split.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .lr_engine import (
    _unpack_batch,
    build_loader_bundle,
    build_model_runtime,
    parameter_tensor_digest,
)


SATURATION_REPLAY_SCHEMA = "mnist-conv-lr-saturation-replay/v1"
CHECKPOINT_ROLES = ("initialization", "best_validation", "final")


def sha256_file(path: str | Path) -> str:
    """Return the exact SHA-256 of one file."""

    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(
            f"Expected {label} to be finite. Provided value: {value!r}."
        )
    return result


def _fraction(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else math.nan


def _quantiles(values: Sequence[float]) -> dict[str, float]:
    import torch

    if not values:
        return {
            "sample_saturation_mean": math.nan,
            "sample_saturation_p50": math.nan,
            "sample_saturation_p90": math.nan,
            "sample_saturation_p99": math.nan,
        }
    tensor = torch.as_tensor(values, dtype=torch.float64)
    q = torch.quantile(tensor, torch.tensor([0.5, 0.9, 0.99], dtype=torch.float64))
    return {
        "sample_saturation_mean": float(tensor.mean().item()),
        "sample_saturation_p50": float(q[0].item()),
        "sample_saturation_p90": float(q[1].item()),
        "sample_saturation_p99": float(q[2].item()),
    }


def balanced_validation_positions(
    labels: Sequence[int],
    *,
    sample_count: int,
    class_count: int = 10,
) -> tuple[int, ...]:
    """Select a deterministic near-equal class cohort in round-robin order."""

    if int(class_count) <= 0:
        raise ValueError(
            f"Expected class_count to be positive. Provided value: {class_count!r}."
        )
    if int(sample_count) < int(class_count):
        raise ValueError(
            "Expected sample_count to include at least one example per class. "
            f"Provided value: sample_count={sample_count!r}, "
            f"class_count={class_count!r}."
        )
    if int(sample_count) > len(labels):
        raise ValueError(
            "Expected sample_count not to exceed the validation cohort. "
            f"Provided value: sample_count={sample_count!r}, labels={len(labels)}."
        )
    buckets: list[list[int]] = [[] for _ in range(int(class_count))]
    for position, raw_label in enumerate(labels):
        label = int(raw_label)
        if label < 0 or label >= int(class_count):
            raise ValueError(
                f"Expected labels in [0,{int(class_count) - 1}]. "
                f"Provided value at position {position}: {label!r}."
            )
        buckets[label].append(position)
    base, remainder = divmod(int(sample_count), int(class_count))
    quotas = [base + (label < remainder) for label in range(int(class_count))]
    for label, (bucket, quota) in enumerate(zip(buckets, quotas)):
        if len(bucket) < quota:
            raise ValueError(
                "Expected enough validation examples in every class. "
                f"Provided value for class {label}: available={len(bucket)}, "
                f"required={quota}."
            )
    selected: list[int] = []
    for offset in range(max(quotas)):
        for label in range(int(class_count)):
            if offset < quotas[label]:
                selected.append(buckets[label][offset])
    if len(selected) != int(sample_count):
        raise RuntimeError(
            "Expected balanced selection to produce the requested sample count. "
            f"Provided value: expected={sample_count}, actual={len(selected)}."
        )
    return tuple(selected)


def _limited_validation_loader(bundle: Any, *, batch_size: int, max_batches: int):
    """Build a deterministic class-balanced subset of MNIST-train validation."""

    import torch

    validation_dataset = bundle.validation_loader.dataset
    source_dataset = getattr(validation_dataset, "dataset", None)
    targets = getattr(source_dataset, "targets", None)
    if targets is None:
        raise TypeError(
            "Expected validation dataset to expose MNIST training targets. "
            f"Provided value: {type(source_dataset).__name__}."
        )
    labels = [int(targets[int(index)]) for index in bundle.validation_indices]
    sample_count = int(batch_size) * int(max_batches)
    positions = balanced_validation_positions(labels, sample_count=sample_count)
    subset = torch.utils.data.Subset(validation_dataset, positions)
    return (
        torch.utils.data.DataLoader(
            subset,
            batch_size=int(batch_size),
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        ),
        tuple(labels[position] for position in positions),
    )


@dataclass
class HardSigmoidLayerAccumulator:
    """Count-weighted hard-sigmoid state statistics for one Conv hidden layer."""

    v_off: float
    layer_name: str
    layer_index: int
    low_count: int = 0
    active_count: int = 0
    high_count: int = 0
    state_sum: float = 0.0
    state_sum_squares: float = 0.0
    state_min: float = math.inf
    state_max: float = -math.inf
    sample_saturation: list[float] = field(default_factory=list)
    channel_low: Any = None
    channel_active: Any = None
    channel_high: Any = None
    channel_sum: Any = None
    channel_sum_squares: Any = None
    channel_min: Any = None
    channel_max: Any = None
    channel_elements: int = 0

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.v_off)) or float(self.v_off) <= 0.0:
            raise ValueError(
                "Expected v_off to be a finite positive scalar. "
                f"Provided value: {self.v_off!r}."
            )

    def update(self, state: Any) -> None:
        """Merge one ``[batch, channel, height, width]`` hidden-state tensor."""

        import torch

        if not torch.is_tensor(state) or state.ndim != 4:
            shape = tuple(state.shape) if torch.is_tensor(state) else type(state).__name__
            raise ValueError(
                "Expected a Conv hidden state with shape [B,C,H,W]. "
                f"Provided value: {shape!r}."
            )
        detached = state.detach()
        if not bool(torch.isfinite(detached).all().item()):
            count = int((~torch.isfinite(detached)).sum().item())
            raise ValueError(
                "Expected hidden state to contain only finite values. "
                f"Provided value: {count} non-finite element(s)."
            )
        low = detached < -float(self.v_off)
        high = detached > float(self.v_off)
        active = ~(low | high)
        if not bool((low | active | high).all().item()):
            raise RuntimeError("Expected low, active, and high masks to be exhaustive.")
        if bool(((low & active) | (low & high) | (active & high)).any().item()):
            raise RuntimeError("Expected low, active, and high masks to be disjoint.")

        batch, channels, height, width = (int(value) for value in detached.shape)
        if batch <= 0 or channels <= 0 or height <= 0 or width <= 0:
            raise ValueError(
                "Expected every Conv hidden-state dimension to be positive. "
                f"Provided value: {tuple(detached.shape)!r}."
            )
        spatial_elements = batch * height * width
        if self.channel_elements and int(self.channel_low.numel()) != channels:
            raise ValueError(
                "Expected a fixed hidden-channel count across batches. "
                f"Provided value: previous={int(self.channel_low.numel())}, "
                f"current={channels}."
            )

        low_cpu = low.sum(dim=(0, 2, 3), dtype=torch.int64).cpu()
        active_cpu = active.sum(dim=(0, 2, 3), dtype=torch.int64).cpu()
        high_cpu = high.sum(dim=(0, 2, 3), dtype=torch.int64).cpu()
        state64 = detached.to(dtype=torch.float64)
        sum_cpu = state64.sum(dim=(0, 2, 3)).cpu()
        sum_squares_cpu = (state64 * state64).sum(dim=(0, 2, 3)).cpu()
        min_cpu = state64.amin(dim=(0, 2, 3)).cpu()
        max_cpu = state64.amax(dim=(0, 2, 3)).cpu()

        if self.channel_low is None:
            self.channel_low = low_cpu
            self.channel_active = active_cpu
            self.channel_high = high_cpu
            self.channel_sum = sum_cpu
            self.channel_sum_squares = sum_squares_cpu
            self.channel_min = min_cpu
            self.channel_max = max_cpu
        else:
            self.channel_low += low_cpu
            self.channel_active += active_cpu
            self.channel_high += high_cpu
            self.channel_sum += sum_cpu
            self.channel_sum_squares += sum_squares_cpu
            self.channel_min = torch.minimum(self.channel_min, min_cpu)
            self.channel_max = torch.maximum(self.channel_max, max_cpu)
        self.channel_elements += spatial_elements

        low_total = int(low.sum().item())
        active_total = int(active.sum().item())
        high_total = int(high.sum().item())
        self.low_count += low_total
        self.active_count += active_total
        self.high_count += high_total
        self.state_sum += float(state64.sum().item())
        self.state_sum_squares += float((state64 * state64).sum().item())
        self.state_min = min(self.state_min, float(state64.min().item()))
        self.state_max = max(self.state_max, float(state64.max().item()))
        sample_fraction = (low | high).flatten(1).to(torch.float64).mean(dim=1)
        self.sample_saturation.extend(float(value) for value in sample_fraction.cpu())

    @property
    def total_count(self) -> int:
        return self.low_count + self.active_count + self.high_count

    def layer_row(self) -> dict[str, Any]:
        total = self.total_count
        if total <= 0:
            raise RuntimeError(
                f"Expected at least one state element for {self.layer_name!r}."
            )
        mean = self.state_sum / total
        variance = max(0.0, self.state_sum_squares / total - mean * mean)
        low_fraction = _fraction(self.low_count, total)
        high_fraction = _fraction(self.high_count, total)
        return {
            "layer_index": int(self.layer_index),
            "layer_name": self.layer_name,
            "sample_count": len(self.sample_saturation),
            "channel_count": int(self.channel_low.numel()),
            "element_count": total,
            "low_count": self.low_count,
            "active_count": self.active_count,
            "high_count": self.high_count,
            "low_fraction": low_fraction,
            "active_fraction": _fraction(self.active_count, total),
            "high_fraction": high_fraction,
            "saturation_fraction": low_fraction + high_fraction,
            "state_mean": mean,
            "state_std": math.sqrt(variance),
            "state_rms": math.sqrt(self.state_sum_squares / total),
            "state_min": self.state_min,
            "state_max": self.state_max,
            **_quantiles(self.sample_saturation),
        }

    def channel_rows(self) -> list[dict[str, Any]]:
        import torch

        if self.channel_low is None or self.channel_elements <= 0:
            raise RuntimeError(
                f"Expected channel statistics for {self.layer_name!r}."
            )
        rows: list[dict[str, Any]] = []
        for channel in range(int(self.channel_low.numel())):
            low = int(self.channel_low[channel].item())
            active = int(self.channel_active[channel].item())
            high = int(self.channel_high[channel].item())
            total = low + active + high
            if total != self.channel_elements:
                raise RuntimeError(
                    "Expected equal element counts for every channel. "
                    f"Provided value for channel {channel}: {total}, "
                    f"expected {self.channel_elements}."
                )
            mean = float(self.channel_sum[channel].item()) / total
            mean_square = float(self.channel_sum_squares[channel].item()) / total
            rows.append(
                {
                    "layer_index": int(self.layer_index),
                    "layer_name": self.layer_name,
                    "channel_index": channel,
                    "element_count": total,
                    "low_count": low,
                    "active_count": active,
                    "high_count": high,
                    "low_fraction": _fraction(low, total),
                    "active_fraction": _fraction(active, total),
                    "high_fraction": _fraction(high, total),
                    "saturation_fraction": _fraction(low + high, total),
                    "state_mean": mean,
                    "state_std": math.sqrt(max(0.0, mean_square - mean * mean)),
                    "state_rms": math.sqrt(mean_square),
                    "state_min": float(self.channel_min[channel].item()),
                    "state_max": float(self.channel_max[channel].item()),
                }
            )
        saturation = torch.as_tensor(
            [row["saturation_fraction"] for row in rows], dtype=torch.float64
        )
        layer = self.layer_row()
        layer["channels_saturation_gt_0p5_fraction"] = float(
            (saturation > 0.5).to(torch.float64).mean().item()
        )
        layer["channels_saturation_gt_0p9_fraction"] = float(
            (saturation > 0.9).to(torch.float64).mean().item()
        )
        return rows

    def finalize(self) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        channel_rows = self.channel_rows()
        layer_row = self.layer_row()
        saturation = [float(row["saturation_fraction"]) for row in channel_rows]
        layer_row["channels_saturation_gt_0p5_fraction"] = _fraction(
            sum(value > 0.5 for value in saturation), len(saturation)
        )
        layer_row["channels_saturation_gt_0p9_fraction"] = _fraction(
            sum(value > 0.9 for value in saturation), len(saturation)
        )
        return layer_row, channel_rows


def infer_rho_targets(run_spec: Mapping[str, Any]) -> tuple[float, float]:
    """Read direct two-rho targets or derive v5 strict/profile targets."""

    provenance = run_spec["run"]["lr_provenance"]
    if "rho_conv" in provenance and "rho_dense" in provenance:
        return (
            _finite(provenance["rho_conv"], "rho_conv"),
            _finite(provenance["rho_dense"], "rho_dense"),
        )
    alpha = provenance.get("alpha_arch")
    multipliers = provenance.get("target_multipliers_by_weight")
    if alpha is None or not isinstance(multipliers, Mapping):
        raise ValueError(
            "Expected run LR provenance to contain direct rho targets or "
            "alpha_arch with target_multipliers_by_weight. "
            f"Provided value: {provenance!r}."
        )
    conv_values = {
        float(value)
        for name, value in multipliers.items()
        if str(name).startswith("ConvWeight_")
    }
    dense_values = {
        float(value)
        for name, value in multipliers.items()
        if str(name).startswith("DenseWeight_")
    }
    if len(conv_values) != 1 or len(dense_values) != 1:
        raise ValueError(
            "Expected one common Conv multiplier and one Dense multiplier. "
            f"Provided value: conv={sorted(conv_values)!r}, "
            f"dense={sorted(dense_values)!r}."
        )
    return float(alpha) * conv_values.pop(), float(alpha) * dense_values.pop()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(
            f"Expected a JSON object in {path}. Provided value: {type(value).__name__}."
        )
    return value


def _run_spec_path(entry_dir: Path) -> Path:
    matches = sorted(entry_dir.glob("run_spec.v*.json"))
    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one run_spec.v*.json in a candidate entry. "
            f"Provided value for {entry_dir}: {[path.name for path in matches]!r}."
        )
    return matches[0]


def _validate_study(study: Mapping[str, Any]) -> None:
    dataset = study.get("dataset", {})
    official = dataset.get("official_test")
    if official != {"enabled": False, "read_allowed": False}:
        raise ValueError(
            "Expected official_test to be exactly "
            "{'enabled': False, 'read_allowed': False}. "
            f"Provided value: {official!r}."
        )
    if dataset.get("variant") != "ordinary":
        raise ValueError(
            "Expected the ordinary-MNIST optimization diagnostic. "
            f"Provided value: {dataset.get('variant')!r}."
        )
    if dataset.get("validation", {}).get("source") != "mnist_train":
        raise ValueError(
            "Expected validation source to be 'mnist_train'. "
            f"Provided value: {dataset.get('validation', {}).get('source')!r}."
        )


def _row_by_id(study: Mapping[str, Any], row_id: str) -> Mapping[str, Any]:
    rows = [row for row in study["rows"] if row.get("row_id") == row_id]
    if len(rows) != 1:
        raise ValueError(
            f"Expected exactly one study row for {row_id!r}. "
            f"Provided value: {len(rows)}."
        )
    return rows[0]


def _validate_run_spec_contract(
    study: Mapping[str, Any],
    row: Mapping[str, Any],
    run_spec: Mapping[str, Any],
) -> None:
    """Reject replay under a runtime different from the checkpoint contract."""

    run = run_spec.get("run")
    if not isinstance(run, Mapping):
        raise ValueError(
            "Expected run spec to contain a run object. "
            f"Provided value: {run!r}."
        )
    architecture = study["model"]["architectures"][row["architecture"]]
    run_architecture = run["architecture"]
    architecture_fields = (
        "channels",
        "kernel_sizes",
        "strides",
        "paddings",
        "output_dim",
    )
    for field_name in architecture_fields:
        if run_architecture.get(field_name) != architecture.get(field_name):
            raise ValueError(
                f"Expected run-spec architecture {field_name!r} to match the "
                "runtime study. "
                f"Provided value: run={run_architecture.get(field_name)!r}, "
                f"study={architecture.get(field_name)!r}."
            )

    model = study["model"]
    run_model = run["model"]
    expected_model = {
        "non_linearity": model["non_linearity"],
        "quadratic_diode_param": model["quadratic_diode_param"],
        "exponential_diode_param": model["exponential_diode_param"],
        "hard_sigmoid_param": model["hard_sigmoid"],
        "input_gain": row["input_gain"],
        "voltage_amp": row["voltage_amp"],
        "current_amp": row["current_amp"],
        "weight_min": model["conductance_bounds"][0],
        "weight_max": model["conductance_bounds"][1],
        "weight_init_mode": model["weight_initialization"],
        "weight_gains": [float(model["weight_gains"])]
        * (len(architecture["channels"]) + 1),
    }
    for field_name, expected in expected_model.items():
        if run_model.get(field_name) != expected:
            raise ValueError(
                f"Expected run-spec model {field_name!r} to match the runtime "
                "study row. "
                f"Provided value: run={run_model.get(field_name)!r}, "
                f"study={expected!r}."
            )

    solver = study["solver"]
    run_solver = run["solver"]
    expected_solver = {
        "energy_mode": solver["energy_mode"],
        "minimizer": solver["minimizer"],
        "inference_iterations": row["inference_iterations"],
        "training_iterations": row["training_iterations"],
    }
    for field_name, expected in expected_solver.items():
        if run_solver.get(field_name) != expected:
            raise ValueError(
                f"Expected run-spec solver {field_name!r} to match the runtime "
                "study row. "
                f"Provided value: run={run_solver.get(field_name)!r}, "
                f"study={expected!r}."
            )

    dataset = study["dataset"]
    run_dataset = run["dataset"]
    for field_name in ("name", "normalization", "affine"):
        expected = dataset[field_name]
        actual = run_dataset.get(field_name)
        if field_name == "affine":
            actual = {
                key: actual[key]
                for key in (
                    "degrees",
                    "enabled",
                    "fill",
                    "interpolation",
                    "preset",
                    "scale",
                    "seed",
                    "shear",
                    "translate",
                )
            }
            expected = {key: expected[key] for key in actual}
        if actual != expected:
            raise ValueError(
                f"Expected run-spec dataset {field_name!r} to match the runtime "
                "study. "
                f"Provided value: run={actual!r}, study={expected!r}."
            )
    validation = run_dataset["validation"]
    expected_validation = dataset["validation"]
    dataset_checks = {
        "batch_size": (
            run_dataset.get("batch_size"),
            dataset["train"]["batch_size"],
        ),
        "validation.source": (
            validation.get("source"),
            expected_validation["source"],
        ),
        "validation.split_seed": (
            validation.get("split_seed"),
            expected_validation["split_seed"],
        ),
        "validation.size": (
            validation.get("size"),
            expected_validation["size"],
        ),
        "validation.official_test_enabled": (
            validation.get("official_test_enabled"),
            False,
        ),
        "seed": (run_spec.get("seed"), model["model_seed"]),
    }
    for label, (actual, expected) in dataset_checks.items():
        if actual != expected:
            raise ValueError(
                f"Expected run-spec {label} to match the runtime study. "
                f"Provided value: run={actual!r}, study={expected!r}."
            )


def _checkpoint_contract(
    entry_dir: Path,
    summary: Mapping[str, Any],
    run_spec: Mapping[str, Any],
    initialization_dir: Path,
) -> list[dict[str, Any]]:
    architecture = str(summary["row"]["architecture"])
    initialization_path = initialization_dir / f"{architecture}.pt"
    initialization = run_spec["run"]["initialization"]["checkpoint"]
    best_epoch = summary.get("best_validation_epoch")
    if best_epoch is None:
        raise ValueError(
            "Expected a real best-validation checkpoint with a recorded epoch. "
            f"Provided value for {entry_dir}: {best_epoch!r}."
        )
    return [
        {
            "checkpoint_role": "initialization",
            "path": initialization_path,
            "expected_file_sha256": str(initialization["sha256"]),
            "expected_parameter_sha256": summary.get(
                "initial_parameter_tensor_sha256"
            ),
            "checkpoint_epoch": 0,
        },
        {
            "checkpoint_role": "best_validation",
            "path": entry_dir / "best_validation.pt",
            "expected_file_sha256": str(summary["best_checkpoint_sha256"]),
            "expected_parameter_sha256": None,
            "checkpoint_epoch": int(best_epoch),
        },
        {
            "checkpoint_role": "final",
            "path": entry_dir / "final.pt",
            "expected_file_sha256": str(summary["final_checkpoint_sha256"]),
            "expected_parameter_sha256": summary.get(
                "final_parameter_tensor_sha256"
            ),
            "checkpoint_epoch": 5,
        },
    ]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(
            f"Expected at least one row for CSV output. Provided value: {len(rows)}."
        )
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _plot_saturation(layer_rows: Sequence[Mapping[str, Any]], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    architectures = sorted({str(row["architecture"]) for row in layer_rows})
    role_order = {role: index for index, role in enumerate(CHECKPOINT_ROLES)}
    scheme_order = {"baseline": 0, "ours": 1, "legacy": 2}
    for architecture in architectures:
        selected = [row for row in layer_rows if row["architecture"] == architecture]
        layer_indices = sorted({int(row["layer_index"]) for row in selected})
        fig, axes = plt.subplots(
            len(layer_indices),
            1,
            figsize=(10.5, 3.6 * len(layer_indices)),
            squeeze=False,
        )
        for axis, layer_index in zip(axes[:, 0], layer_indices):
            rows = sorted(
                [row for row in selected if int(row["layer_index"]) == layer_index],
                key=lambda row: (
                    scheme_order[str(row["scheme"])],
                    role_order[str(row["checkpoint_role"])],
                ),
            )
            labels = [
                f"{row['scheme']}\n{str(row['checkpoint_role']).replace('_', ' ')}"
                for row in rows
            ]
            positions = np.arange(len(rows))
            low = np.asarray([100.0 * float(row["low_fraction"]) for row in rows])
            high = np.asarray([100.0 * float(row["high_fraction"]) for row in rows])
            axis.bar(positions, low, label="state < -4", color="#4472C4")
            axis.bar(
                positions,
                high,
                bottom=low,
                label="state > 4",
                color="#ED7D31",
            )
            axis.set_xticks(positions, labels)
            axis.set_ylabel("Outside-window states (%)")
            axis.set_ylim(0.0, 100.0)
            axis.set_title(
                f"{architecture.upper()} hidden layer {layer_index + 1}"
            )
            axis.grid(axis="y", alpha=0.25)
            axis.legend(loc="upper left")
        fig.suptitle(
            "Ordinary-MNIST checkpoint saturation replay "
            "(matched rho_conv=rho_dense=1e-2)"
        )
        fig.tight_layout()
        fig.savefig(output_dir / f"{architecture}_saturation.png", dpi=180)
        plt.close(fig)


def replay_saturation(
    *,
    study_path: str | Path,
    entry_dirs: Iterable[str | Path],
    initialization_dir: str | Path,
    data_root: str | Path,
    output_dir: str | Path,
    device: str,
    batch_size: int = 16,
    max_batches: int = 8,
    reference_inference_iterations: int | None = None,
) -> dict[str, Any]:
    """Replay selected checkpoints on a fixed prefix of validation minibatches."""

    import torch
    from labs.datasets import stable_index_sequence_hash

    if int(batch_size) <= 0:
        raise ValueError(
            f"Expected batch_size to be positive. Provided value: {batch_size!r}."
        )
    if int(max_batches) <= 0:
        raise ValueError(
            f"Expected max_batches to be positive. Provided value: {max_batches!r}."
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
    runtime_study["dataset"]["validation"]["batch_size"] = int(batch_size)
    bundle = build_loader_bundle(
        runtime_study,
        data_root=data_root,
        download=False,
        return_source_indices=True,
    )
    validation_loader, cohort_labels = _limited_validation_loader(
        bundle,
        batch_size=int(batch_size),
        max_batches=int(max_batches),
    )
    entries = sorted({Path(path).expanduser().resolve() for path in entry_dirs})
    if not entries:
        raise ValueError(
            f"Expected at least one candidate entry. Provided value: {entries!r}."
        )
    initialization_root = Path(initialization_dir).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    layer_rows: list[dict[str, Any]] = []
    channel_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    source_contracts: list[dict[str, Any]] = []
    cohort_indices: tuple[int, ...] | None = None

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
        if len(recorded_run_spec_hashes) != 1:
            raise ValueError(
                "Expected exactly one recorded run-spec SHA-256 in the candidate "
                f"summary. Provided value for {entry_dir}: "
                f"{recorded_run_spec_hashes!r}."
            )
        if run_spec_hash != recorded_run_spec_hashes[0]:
            raise ValueError(
                "Expected run-spec bytes to match the candidate summary. "
                f"Provided value for {run_spec_path}: "
                f"expected={recorded_run_spec_hashes[0]!r}, "
                f"actual={run_spec_hash!r}."
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
        expected_split = str(summary["validation_indices_sha256"])
        if expected_split != bundle.validation_indices_hash:
            raise RuntimeError(
                "Expected candidate validation indices to match the reconstructed "
                f"split. Provided value for {entry_dir}: "
                f"candidate={expected_split!r}, actual={bundle.validation_indices_hash!r}."
            )
        checkpoints = _checkpoint_contract(
            entry_dir, summary, run_spec, initialization_root
        )
        runtime_row = dict(row)
        solver_mode = "operational"
        if reference_inference_iterations is not None:
            runtime_row["inference_iterations"] = int(
                reference_inference_iterations
            )
            solver_mode = "reference"
        case_base = {
            "entry_id": entry_dir.name,
            "row_id": row_id,
            "architecture": str(row["architecture"]),
            "scheme": str(row["scheme"]),
            "rho_conv": rho_conv,
            "rho_dense": rho_dense,
        }
        source_contracts.append(
            {
                **case_base,
                "entry_dir": str(entry_dir),
                "summary_path": str(summary_path),
                "summary_sha256": sha256_file(summary_path),
                "run_spec_path": str(run_spec_path),
                "run_spec_sha256": run_spec_hash,
                "checkpoints": [
                    {
                        **{
                            key: value
                            for key, value in checkpoint.items()
                            if key != "path"
                        },
                        "path": str(checkpoint["path"]),
                    }
                    for checkpoint in checkpoints
                ],
            }
        )

        for checkpoint in checkpoints:
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
            runtime = build_model_runtime(
                runtime_study,
                runtime_row,
                device=device,
                initialization_checkpoint=checkpoint_path,
                learning_rate=1.0,
            )
            parameter_hash = parameter_tensor_digest(runtime.parameters)
            expected_parameter_hash = checkpoint["expected_parameter_sha256"]
            if (
                expected_parameter_hash is not None
                and parameter_hash != expected_parameter_hash
            ):
                raise ValueError(
                    "Expected loaded parameter tensors to match candidate provenance. "
                    f"Provided value for {checkpoint_path}: "
                    f"expected={expected_parameter_hash!r}, actual={parameter_hash!r}."
                )

            accumulators: list[HardSigmoidLayerAccumulator] | None = None
            running_loss = 0.0
            running_correct = 0
            running_samples = 0
            indices: list[int] = []
            v_off = float(runtime_study["model"]["hard_sigmoid"]["v_off"])
            with torch.no_grad():
                for batch_index, batch in enumerate(validation_loader):
                    if batch_index >= int(max_batches):
                        break
                    images, labels, source_indices = _unpack_batch(batch)
                    images = images.to(runtime.device)
                    labels = labels.to(runtime.device)
                    runtime.network.set_input(images, reset=True)
                    runtime.minimizer_inference.compute_equilibrium()
                    hidden_layers = runtime.energy_fn.layers()[1:-1]
                    expected_hidden = 1 if row["architecture"] == "conv1" else 2
                    if len(hidden_layers) != expected_hidden:
                        raise RuntimeError(
                            "Expected the frozen architecture hidden-layer count. "
                            f"Provided value for {row_id}: {len(hidden_layers)}."
                        )
                    if accumulators is None:
                        accumulators = [
                            HardSigmoidLayerAccumulator(
                                v_off=v_off,
                                layer_name=f"Hidden_{index}",
                                layer_index=index,
                            )
                            for index in range(len(hidden_layers))
                        ]
                    for accumulator, layer in zip(accumulators, hidden_layers):
                        accumulator.update(layer.state)
                    runtime.cost_fn.set_target(labels)
                    cost = runtime.cost_fn.eval()
                    batch_count = int(images.shape[0])
                    running_loss += float(cost.mean().item()) * batch_count
                    running_correct += int((~runtime.cost_fn.error_fn()).sum().item())
                    running_samples += batch_count
                    indices.extend(source_indices)
            if accumulators is None or running_samples <= 0:
                raise RuntimeError(
                    f"Expected replay batches for checkpoint {checkpoint_path}."
                )
            current_indices = tuple(indices)
            if cohort_indices is None:
                cohort_indices = current_indices
            elif current_indices != cohort_indices:
                raise RuntimeError(
                    "Expected every checkpoint to use identical validation indices "
                    "in identical order."
                )
            source_validation = None
            if checkpoint["checkpoint_role"] != "initialization":
                source_matches = [
                    value
                    for value in summary["validation_metrics"]
                    if int(value["epoch"]) == int(checkpoint["checkpoint_epoch"])
                ]
                if len(source_matches) != 1:
                    raise ValueError(
                        "Expected one full-validation record for the checkpoint "
                        f"epoch. Provided value for {entry_dir}: "
                        f"role={checkpoint['checkpoint_role']!r}, "
                        f"epoch={checkpoint['checkpoint_epoch']!r}, "
                        f"matches={len(source_matches)}."
                    )
                source_validation = source_matches[0]
            common = {
                **case_base,
                "checkpoint_role": checkpoint["checkpoint_role"],
                "checkpoint_epoch": checkpoint["checkpoint_epoch"],
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_file_sha256": file_hash_before,
                "parameter_tensor_sha256": parameter_hash,
                "solver_mode": solver_mode,
                "operational_inference_iterations": int(
                    row["inference_iterations"]
                ),
                "inference_iterations": int(
                    runtime_row["inference_iterations"]
                ),
                "batch_size": int(batch_size),
                "max_batches": int(max_batches),
                "sample_count": running_samples,
                "cohort_indices_sha256": stable_index_sequence_hash(current_indices),
                "source_operational_validation_inference_iterations": (
                    int(row["inference_iterations"])
                    if source_validation is not None
                    else None
                ),
                "source_operational_full_validation_sample_count": (
                    int(source_validation["sample_count"])
                    if source_validation is not None
                    else None
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
            checkpoint_rows.append(
                {
                    **common,
                    "cohort_loss": running_loss / running_samples,
                    "cohort_accuracy": running_correct / running_samples,
                }
            )
            for accumulator in accumulators:
                layer_row, channels = accumulator.finalize()
                layer_rows.append({**common, **layer_row})
                channel_rows.extend({**common, **row_value} for row_value in channels)
            file_hash_after = sha256_file(checkpoint_path)
            if file_hash_after != file_hash_before:
                raise RuntimeError(
                    "Expected checkpoint replay to leave source bytes unchanged. "
                    f"Provided value: before={file_hash_before!r}, "
                    f"after={file_hash_after!r}."
                )
            del runtime
            if str(device).startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()

    assert cohort_indices is not None
    cohort = {
        "source": "mnist_train_validation",
        "selection": "deterministic_class_balanced_round_robin/v1",
        "official_test_read": False,
        "batch_size": int(batch_size),
        "max_batches": int(max_batches),
        "sample_count": len(cohort_indices),
        "source_indices": list(cohort_indices),
        "class_counts": {
            str(label): sum(value == label for value in cohort_labels)
            for label in range(10)
        },
        "source_indices_sha256": stable_index_sequence_hash(cohort_indices),
        "complete_validation_indices_sha256": bundle.validation_indices_hash,
    }
    result = {
        "schema_version": SATURATION_REPLAY_SCHEMA,
        "study_path": str(study_source),
        "study_sha256": sha256_file(study_source),
        "device": str(device),
        "solver_mode": (
            "reference" if reference_inference_iterations is not None else "operational"
        ),
        "cohort": cohort,
        "sources": source_contracts,
        "checkpoints": checkpoint_rows,
        "layers": layer_rows,
    }
    _write_csv(destination / "checkpoint_summary.csv", checkpoint_rows)
    _write_csv(destination / "layer_saturation.csv", layer_rows)
    _write_csv(destination / "channel_saturation.csv", channel_rows)
    (destination / "cohort.json").write_text(
        json.dumps(cohort, sort_keys=True, separators=(",", ":")) + "\n"
    )
    (destination / "summary.json").write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n"
    )
    _plot_saturation(layer_rows, destination)
    return result
