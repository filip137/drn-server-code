"""Analyze selected bounded Conv2 conductance checkpoints.

This is a read-only, ordinary-MNIST diagnostic for the baseline/ours
three-epoch rho studies.  It compares every selected checkpoint with the
shared initializer from which that initializer family was trained.  The
official MNIST test split is never constructed or read.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


LOWER_CONDUCTANCE = 1e-5
UPPER_CONDUCTANCE = 1e-4
CONDUCTANCE_RANGE = UPPER_CONDUCTANCE - LOWER_CONDUCTANCE


@dataclass(frozen=True)
class Selection:
    surface_id: str
    initializer: str
    scheme: str
    optimizer: str
    accuracy: float
    rho_conv: float
    rho_dense: float
    range_classification: str
    checkpoint: str


SELECTIONS = (
    Selection(
        "bounded_uniform__conv2__baseline__sgd",
        "bounded_uniform",
        "baseline",
        "SGD",
        0.7728,
        0.001,
        0.00037037037037037035,
        "unbounded",
        "perfectdiode-conv2-bounded-rho-continuation-wave2-seed0-v1/"
        "surfaces/bounded_uniform__conv2__baseline__sgd/rho/cells/"
        "005_rc_0p001_rd_0p00037037037037/final_model.pt",
    ),
    Selection(
        "bounded_uniform__conv2__baseline__adam",
        "bounded_uniform",
        "baseline",
        "Adam",
        0.8530,
        0.081,
        0.003333333333333333,
        "unbounded",
        "perfectdiode-conv2-bounded-rho-continuation-wave2-seed0-v1/"
        "surfaces/bounded_uniform__conv2__baseline__adam/rho/cells/"
        "017_rc_0p081_rd_0p00333333333333/final_model.pt",
    ),
    Selection(
        "bounded_uniform__conv2__ours__sgd",
        "bounded_uniform",
        "ours",
        "SGD",
        0.8580,
        0.0003333333333333333,
        0.0001234567901234568,
        "bounded_dense_lower",
        "perfectdiode-conv2-bounded-rho-continuation-wave3-seed0-v1/"
        "surfaces/bounded_uniform__conv2__ours__sgd/rho/cells/"
        "006_rc_0p000333333333333_rd_0p000123456790123/final_model.pt",
    ),
    Selection(
        "bounded_uniform__conv2__ours__adam",
        "bounded_uniform",
        "ours",
        "Adam",
        0.8780,
        0.027,
        0.01,
        "unbounded",
        "perfectdiode-conv12-bounded-rho-baseline-ours-tk46-seed0-v1/"
        "surfaces/bounded_uniform__conv2__ours__adam/rho/cells/"
        "097_rc_0p027_rd_0p01/final_model.pt",
    ),
    Selection(
        "bounded_kaiming_uniform__conv2__baseline__sgd",
        "bounded_kaiming_uniform",
        "baseline",
        "SGD",
        0.6366,
        0.0001111111111111111,
        0.0011111111111111111,
        "bounded_dense_lower",
        "perfectdiode-conv2-bounded-rho-continuation-wave3-seed0-v1/"
        "surfaces/bounded_kaiming_uniform__conv2__baseline__sgd/rho/cells/"
        "004_rc_0p000111111111111_rd_0p00111111111111/final_model.pt",
    ),
    Selection(
        "bounded_kaiming_uniform__conv2__baseline__adam",
        "bounded_kaiming_uniform",
        "baseline",
        "Adam",
        0.8526,
        0.081,
        0.003333333333333333,
        "unbounded",
        "perfectdiode-conv2-bounded-rho-continuation-wave2-seed0-v1/"
        "surfaces/bounded_kaiming_uniform__conv2__baseline__adam/rho/cells/"
        "017_rc_0p081_rd_0p00333333333333/final_model.pt",
    ),
    Selection(
        "bounded_kaiming_uniform__conv2__ours__sgd",
        "bounded_kaiming_uniform",
        "ours",
        "SGD",
        0.1008,
        0.003,
        0.01,
        "unbounded",
        "perfectdiode-conv12-bounded-rho-baseline-ours-tk46-seed0-v1/"
        "surfaces/bounded_kaiming_uniform__conv2__ours__sgd/rho/cells/"
        "077_rc_0p003_rd_0p01/final_model.pt",
    ),
    Selection(
        "bounded_kaiming_uniform__conv2__ours__adam",
        "bounded_kaiming_uniform",
        "ours",
        "Adam",
        0.8810,
        0.081,
        0.03,
        "unbounded",
        "perfectdiode-conv2-bounded-rho-continuation-wave2-seed0-v1/"
        "surfaces/bounded_kaiming_uniform__conv2__ours__adam/rho/cells/"
        "018_rc_0p081_rd_0p03/final_model.pt",
    ),
)


INITIAL_CHECKPOINTS = {
    "bounded_uniform": (
        "perfectdiode-conv12-bounded-rho-baseline-ours-tk46-seed0-v1/"
        "assets/bounded_uniform/conv2/final_model.pt"
    ),
    "bounded_kaiming_uniform": (
        "perfectdiode-conv12-bounded-rho-baseline-ours-tk46-seed0-v1/"
        "assets/bounded_kaiming_uniform/conv2/final_model.pt"
    ),
}


def _load_named_states(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "drn.function.parameters":
        raise ValueError(f"Unexpected checkpoint format in {path}.")
    schema = payload["schema"]
    states = payload["states"]
    if len(schema) != len(states):
        raise ValueError(f"Mismatched checkpoint schema and states in {path}.")
    result: dict[str, torch.Tensor] = {}
    for specification, state in zip(schema, states, strict=True):
        name = str(specification["name"]).strip()
        if name in result:
            raise ValueError(f"Duplicate parameter {name!r} in {path}.")
        result[name] = state.detach().to(dtype=torch.float64).reshape(-1)
    return result


def _quantile(values: torch.Tensor, q: float) -> float:
    return float(torch.quantile(values, q).item())


def _distribution_stats(
    final: torch.Tensor,
    initial: torch.Tensor,
) -> dict[str, float | int]:
    if final.shape != initial.shape:
        raise ValueError(
            f"Final and initial shapes differ: {final.shape} versus {initial.shape}."
        )
    lower = float(
        torch.tensor(LOWER_CONDUCTANCE, dtype=torch.float32)
        .to(dtype=torch.float64)
        .item()
    )
    upper = float(
        torch.tensor(UPPER_CONDUCTANCE, dtype=torch.float32)
        .to(dtype=torch.float64)
        .item()
    )
    width = upper - lower
    delta = final - initial
    centered_initial = initial - initial.mean()
    centered_final = final - final.mean()
    correlation_denominator = torch.linalg.vector_norm(
        centered_initial
    ) * torch.linalg.vector_norm(centered_final)
    correlation = (
        float(torch.dot(centered_initial, centered_final) / correlation_denominator)
        if float(correlation_denominator) > 0.0
        else math.nan
    )
    result: dict[str, float | int] = {
        "count": int(final.numel()),
        "minimum": float(final.min().item()),
        "maximum": float(final.max().item()),
        "mean": float(final.mean().item()),
        "std": float(final.std(unbiased=False).item()),
        "at_lower_fraction": float((final <= lower).to(torch.float64).mean().item()),
        "at_upper_fraction": float((final >= upper).to(torch.float64).mean().item()),
        "combined_bound_fraction": float(
            ((final <= lower) | (final >= upper)).to(torch.float64).mean().item()
        ),
        "near_lower_1pct_fraction": float(
            (final <= lower + 0.01 * width).to(torch.float64).mean().item()
        ),
        "near_upper_1pct_fraction": float(
            (final >= upper - 0.01 * width).to(torch.float64).mean().item()
        ),
        "mean_delta": float(delta.mean().item()),
        "mean_abs_delta": float(delta.abs().mean().item()),
        "rms_delta": float(torch.sqrt(torch.mean(delta.square())).item()),
        "normalized_rms_delta": float(
            torch.sqrt(torch.mean(delta.square())).item() / width
        ),
        "unchanged_fraction": float((delta == 0.0).to(torch.float64).mean().item()),
        "moved_at_least_1pct_range_fraction": float(
            (delta.abs() >= 0.01 * width).to(torch.float64).mean().item()
        ),
        "moved_at_least_10pct_range_fraction": float(
            (delta.abs() >= 0.10 * width).to(torch.float64).mean().item()
        ),
        "initial_final_correlation": correlation,
    }
    for q in (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99):
        result[f"q{int(q * 100):02d}"] = _quantile(final, q)
    return result


def _aggregate(tensors: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([tensors[name] for name in sorted(tensors)])


def _weight_states(states: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        name: state
        for name, state in states.items()
        if name.startswith("ConvWeight_") or name.startswith("DenseWeight_")
    }


def _bias_states(states: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: state for name, state in states.items() if name.startswith("Bias_")}


def _bias_stats(final: torch.Tensor, initial: torch.Tensor) -> dict[str, float | int]:
    delta = final - initial
    return {
        "count": int(final.numel()),
        "minimum": float(final.min().item()),
        "maximum": float(final.max().item()),
        "mean": float(final.mean().item()),
        "std": float(final.std(unbiased=False).item()),
        "mean_abs_delta": float(delta.abs().mean().item()),
        "rms_delta": float(torch.sqrt(torch.mean(delta.square())).item()),
        "unchanged_fraction": float((delta == 0.0).to(torch.float64).mean().item()),
    }


def analyze(results_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    initial_states = {
        initializer: _load_named_states(results_root / relative)
        for initializer, relative in INITIAL_CHECKPOINTS.items()
    }
    report: dict[str, Any] = {
        "schema_version": "conv2-bounded-weight-distribution/v1",
        "dataset_role": "ordinary_mnist_diagnostic",
        "official_test_read": False,
        "bounds": {
            "lower": LOWER_CONDUCTANCE,
            "upper": UPPER_CONDUCTANCE,
        },
        "initializers": {
            initializer: {
                name: _distribution_stats(state, state)
                for name, state in sorted(_weight_states(states).items())
            }
            for initializer, states in initial_states.items()
        },
        "surfaces": [],
    }
    plot_payload: dict[str, Any] = {}

    for selection in SELECTIONS:
        checkpoint = results_root / selection.checkpoint
        cell_dir = checkpoint.parent
        final = _load_named_states(checkpoint)
        initial = initial_states[selection.initializer]
        if set(final) != set(initial):
            raise ValueError(
                f"Parameter mismatch between {checkpoint} and its initializer."
            )
        final_weights = _weight_states(final)
        initial_weights = _weight_states(initial)
        layers = {
            name: _distribution_stats(final_weights[name], initial_weights[name])
            for name in sorted(final_weights)
        }
        aggregate = _distribution_stats(
            _aggregate(final_weights), _aggregate(initial_weights)
        )
        bias_layers = {
            name: _bias_stats(final[name], initial[name])
            for name in sorted(_bias_states(final))
        }
        cell = json.loads(
            (cell_dir / "artifacts" / "cell.json").read_text(encoding="utf-8")
        )
        result = json.loads(
            (cell_dir / "result.json").read_text(encoding="utf-8")
        )
        epoch_metrics = [
            json.loads(line)
            for line in (cell_dir / "metrics.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
        ]
        epoch_metrics = [
            record["metrics"]
            for record in epoch_metrics
            if record.get("kind") == "epoch"
        ]
        entry = {
            "surface_id": selection.surface_id,
            "initializer": selection.initializer,
            "scheme": selection.scheme,
            "optimizer": selection.optimizer,
            "final_validation_accuracy": selection.accuracy,
            "rho_conv": selection.rho_conv,
            "rho_dense": selection.rho_dense,
            "rho_range_classification": selection.range_classification,
            "checkpoint": str(checkpoint.resolve()),
            "weights": {
                "aggregate": aggregate,
                "layers": layers,
            },
            "biases": bias_layers,
            "training_diagnostics": {
                "best_epoch": result["terminal_metrics"]["best_epoch"],
                "validation_accuracy_by_epoch": [
                    metrics["validation_accuracy"] for metrics in epoch_metrics
                ],
                "validation_loss_by_epoch": [
                    metrics["validation_loss"] for metrics in epoch_metrics
                ],
                "learning_rates_by_parameter": cell[
                    "learning_rates_by_parameter"
                ],
            },
        }
        report["surfaces"].append(entry)
        plot_payload[selection.surface_id] = {
            "selection": selection,
            "final": final_weights,
            "initial": initial_weights,
        }
    return report, plot_payload


def write_csv(report: dict[str, Any], path: Path) -> None:
    rows = []
    for surface in report["surfaces"]:
        for layer, stats in surface["weights"]["layers"].items():
            rows.append(
                {
                    "surface_id": surface["surface_id"],
                    "initializer": surface["initializer"],
                    "scheme": surface["scheme"],
                    "optimizer": surface["optimizer"],
                    "final_validation_accuracy": surface[
                        "final_validation_accuracy"
                    ],
                    "rho_conv": surface["rho_conv"],
                    "rho_dense": surface["rho_dense"],
                    "rho_range_classification": surface[
                        "rho_range_classification"
                    ],
                    "layer": layer,
                    **stats,
                }
            )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_histograms(plot_payload: dict[str, Any], path: Path) -> None:
    layer_names = ("ConvWeight_0", "ConvWeight_1", "DenseWeight_0")
    figure, axes = plt.subplots(
        len(SELECTIONS),
        len(layer_names),
        figsize=(12.0, 18.0),
        sharex=True,
        sharey="col",
        constrained_layout=True,
    )
    bins = np.linspace(0.0, 1.0, 41)
    for row, selection in enumerate(SELECTIONS):
        payload = plot_payload[selection.surface_id]
        for column, layer_name in enumerate(layer_names):
            axis = axes[row, column]
            initial = (
                payload["initial"][layer_name].numpy() - LOWER_CONDUCTANCE
            ) / CONDUCTANCE_RANGE
            final = (
                payload["final"][layer_name].numpy() - LOWER_CONDUCTANCE
            ) / CONDUCTANCE_RANGE
            axis.hist(
                initial,
                bins=bins,
                density=True,
                histtype="step",
                linewidth=1.0,
                color="0.35",
                label="initial",
            )
            axis.hist(
                final,
                bins=bins,
                density=True,
                alpha=0.58,
                color="#2474b5" if selection.optimizer == "Adam" else "#d97620",
                label="final",
            )
            axis.grid(alpha=0.2)
            if row == 0:
                axis.set_title(layer_name)
            if column == 0:
                initializer = (
                    "uniform"
                    if selection.initializer == "bounded_uniform"
                    else "kaiming"
                )
                axis.set_ylabel(
                    f"{initializer} {selection.scheme} {selection.optimizer}\n"
                    f"{100.0 * selection.accuracy:.2f}%\ndensity"
                )
            if row == len(SELECTIONS) - 1:
                axis.set_xlabel("normalized conductance position")
            if row == 0 and column == 2:
                axis.legend(loc="upper right", fontsize=8)
    figure.suptitle(
        "Selected Conv2 bounded-weight distributions after three epochs",
        fontsize=14,
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_diagnostics(report: dict[str, Any], path: Path) -> None:
    layer_names = ("ConvWeight_0", "ConvWeight_1", "DenseWeight_0")
    labels = []
    occupancy = []
    movement = []
    accuracy = []
    for surface in report["surfaces"]:
        initializer = (
            "U" if surface["initializer"] == "bounded_uniform" else "K"
        )
        labels.append(
            f"{initializer}-{surface['scheme'][0].upper()}-{surface['optimizer']}"
        )
        occupancy.append(
            [
                surface["weights"]["layers"][layer]["combined_bound_fraction"]
                for layer in layer_names
            ]
        )
        movement.append(
            [
                surface["weights"]["layers"][layer]["normalized_rms_delta"]
                for layer in layer_names
            ]
        )
        accuracy.append(surface["final_validation_accuracy"])

    figure, axes = plt.subplots(1, 3, figsize=(14, 5.8), constrained_layout=True)
    for axis, values, title, format_string in (
        (
            axes[0],
            np.asarray(occupancy),
            "exact bound occupancy",
            ".1%",
        ),
        (
            axes[1],
            np.asarray(movement),
            "RMS movement / conductance range",
            ".2f",
        ),
    ):
        image = axis.imshow(values, aspect="auto", cmap="magma")
        axis.set_xticks(range(len(layer_names)), layer_names, rotation=20)
        axis.set_yticks(range(len(labels)), labels)
        axis.set_title(title)
        for row in range(len(labels)):
            for column in range(len(layer_names)):
                axis.text(
                    column,
                    row,
                    format(values[row, column], format_string),
                    ha="center",
                    va="center",
                    color="white" if values[row, column] > values.max() * 0.45 else "black",
                    fontsize=8,
                )
        figure.colorbar(image, ax=axis, shrink=0.78)

    axes[2].barh(range(len(labels)), np.asarray(accuracy) * 100.0, color="#3975a9")
    axes[2].axvline(90.0, color="#bd2d28", linestyle="--", linewidth=1.2)
    axes[2].axvline(80.0, color="0.35", linestyle=":", linewidth=1.2)
    axes[2].set_yticks(range(len(labels)), labels)
    axes[2].invert_yaxis()
    axes[2].set_xlim(0.0, 100.0)
    axes[2].set_xlabel("validation accuracy (%)")
    axes[2].set_title("three-epoch selection accuracy")
    for row, value in enumerate(accuracy):
        axes[2].text(100.0 * value + 1.0, row, f"{100.0 * value:.2f}%", va="center")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    repository_root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--results-root",
        type=Path,
        default=repository_root / "results",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            repository_root
            / "results"
            / "perfectdiode-conv2-bounded-rho-continuation-wave3-seed0-v1"
            / "analysis"
            / "weight_distribution"
        ),
    )
    arguments = parser.parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    report, plot_payload = analyze(arguments.results_root)
    (arguments.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_csv(report, arguments.output_dir / "layer_statistics.csv")
    plot_histograms(
        plot_payload,
        arguments.output_dir / "selected_weight_histograms.png",
    )
    plot_diagnostics(
        report,
        arguments.output_dir / "weight_diagnostics.png",
    )
    print(arguments.output_dir)


if __name__ == "__main__":
    main()
