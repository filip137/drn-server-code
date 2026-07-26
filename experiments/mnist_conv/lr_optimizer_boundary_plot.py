"""Matplotlib reporting for the Conv2 optimizer-boundary diagnostic."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .lr_optimizer_boundary import (
    OPTIMIZERS,
    RHO_DENSE_BY_SCHEME,
    SCHEMES,
)


def _matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _finite_or_nan(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def _rho_values(
    records: Sequence[Mapping[str, Any]], scheme: str, optimizer: str
) -> tuple[list[float], list[float]]:
    subset = [
        record
        for record in records
        if record.get("scheme") == scheme and record.get("optimizer") == optimizer
        and record.get("stage") in {"main_grid", "upper_sentinel"}
        and record.get("bias_policy", "attached") == "attached"
    ]
    conv = sorted({float(record["rho_conv"]) for record in subset})
    dense = sorted({float(record["rho_dense"]) for record in subset})
    if not conv:
        conv = []
    if not dense:
        dense = list(RHO_DENSE_BY_SCHEME[scheme])
    return conv, dense


def plot_metric_heatmaps(
    records: Sequence[Mapping[str, Any]],
    output_path: str | Path,
    *,
    field: str,
    title: str,
) -> Path:
    """Draw the same rho surface for all six scheme x optimizer arms."""

    plt = _matplotlib()
    figure, axes = plt.subplots(
        len(SCHEMES),
        len(OPTIMIZERS),
        figsize=(11.0, 12.0),
        squeeze=False,
    )
    plotted = False
    for row_index, scheme in enumerate(SCHEMES):
        for column_index, optimizer in enumerate(OPTIMIZERS):
            axis = axes[row_index][column_index]
            rho_conv, rho_dense = _rho_values(records, scheme, optimizer)
            by_pair = {
                (float(record["rho_conv"]), float(record["rho_dense"])): record
                for record in records
                if record.get("scheme") == scheme
                and record.get("optimizer") == optimizer
                and record.get("stage") in {"main_grid", "upper_sentinel"}
                and record.get("bias_policy", "attached") == "attached"
            }
            matrix = [
                [
                    _finite_or_nan(
                        by_pair.get((conv_target, dense_target), {}).get(field)
                    )
                    for dense_target in rho_dense
                ]
                for conv_target in rho_conv
            ]
            if matrix and rho_dense:
                image = axis.imshow(matrix, origin="lower", aspect="auto")
                figure.colorbar(image, ax=axis, shrink=0.8)
                plotted = True
            else:
                axis.text(
                    0.5,
                    0.5,
                    "no completed cells",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                )
            axis.set_xticks(range(len(rho_dense)))
            axis.set_xticklabels([f"{value:g}" for value in rho_dense])
            axis.set_yticks(range(len(rho_conv)))
            axis.set_yticklabels([f"{value:g}" for value in rho_conv])
            axis.set_xlabel(r"$\rho_{\rm dense}$")
            axis.set_ylabel(r"$\rho_{\rm conv}$")
            axis.set_title(f"{scheme} / {optimizer.upper()}")
    if not plotted:
        raise ValueError(
            f"Expected at least one finite {field!r} value. Provided value: records."
        )
    figure.suptitle(title)
    figure.tight_layout()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    return destination


def _selected_replays(
    records: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], dict[str, Mapping[str, Any]]]:
    """Retain both the selected attached run and its capped confirmation."""

    selected: dict[
        tuple[str, str], dict[str, Mapping[str, Any]]
    ] = {}
    for record in records:
        key = (str(record.get("scheme")), str(record.get("optimizer")))
        policy = str(record.get("bias_policy", "attached"))
        if record.get("diagnostic_best") and policy == "attached":
            selected.setdefault(key, {})["attached"] = record
        elif (
            policy == "capped"
            and record.get("stage") == "bias_capped_confirmation"
        ):
            selected.setdefault(key, {})["capped"] = record
    return selected


def plot_epochwise_effective_rho(
    records: Sequence[Mapping[str, Any]],
    output_path: str | Path,
) -> Path:
    """Plot achieved rho at initialization and each epoch for selected runs."""

    plt = _matplotlib()
    selected = _selected_replays(records)
    if not selected:
        raise ValueError(
            "Expected selected or bias-capped records with epoch diagnostics. "
            "Provided value: no selected records."
        )
    figure, axes = plt.subplots(
        len(SCHEMES),
        len(OPTIMIZERS),
        figsize=(12.0, 12.0),
        sharex=True,
        squeeze=False,
    )
    for row_index, scheme in enumerate(SCHEMES):
        for column_index, optimizer in enumerate(OPTIMIZERS):
            axis = axes[row_index][column_index]
            policies = selected.get((scheme, optimizer), {})
            plotted_names = []
            for policy in ("attached", "capped"):
                record = policies.get(policy)
                diagnostics = (
                    []
                    if record is None
                    else record.get("epoch_replay_records")
                    or record.get("epoch_diagnostics", [])
                )
                parameter_names = sorted(
                    {
                        name
                        for diagnostic in diagnostics
                        for name in (
                            diagnostic.get(
                                "achieved_weight_rho_median_by_parameter"
                            )
                            or {}
                        )
                    }
                )
                for name in parameter_names:
                    epochs = [int(item["epoch"]) for item in diagnostics]
                    values = [
                        _finite_or_nan(
                            item[
                                "achieved_weight_rho_median_by_parameter"
                            ].get(name)
                        )
                        for item in diagnostics
                    ]
                    axis.plot(
                        epochs,
                        values,
                        marker="o",
                        linestyle="-" if policy == "attached" else "--",
                        label=f"{policy}/{name}",
                    )
                    plotted_names.append(f"{policy}/{name}")
            if plotted_names:
                target_record = policies.get("attached") or policies["capped"]
                axis.axhline(
                    float(target_record["rho_conv"]),
                    color="black",
                    linestyle="--",
                    linewidth=1,
                    label="conv target",
                )
                axis.axhline(
                    float(target_record["rho_dense"]),
                    color="black",
                    linestyle=":",
                    linewidth=1,
                    label="dense target",
                )
                axis.set_yscale("log")
                axis.legend(fontsize=7)
            else:
                axis.text(
                    0.5,
                    0.5,
                    "replay pending",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                )
            axis.set_title(f"{scheme} / {optimizer.upper()}")
            axis.set_xlabel("epoch (0 = initialization)")
            axis.set_ylabel("achieved weight rho")
            axis.grid(alpha=0.2)
    figure.suptitle("Conv2 achieved effective update rho")
    figure.tight_layout()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    return destination


def plot_epochwise_bias_ratios(
    records: Sequence[Mapping[str, Any]],
    output_path: str | Path,
) -> Path:
    """Plot bias-step/attached-weight-step ratios from frozen replay batches."""

    plt = _matplotlib()
    selected = _selected_replays(records)
    if not selected:
        raise ValueError(
            "Expected selected or bias-capped records with epoch diagnostics. "
            "Provided value: no selected records."
        )
    figure, axes = plt.subplots(
        len(SCHEMES),
        len(OPTIMIZERS),
        figsize=(12.0, 12.0),
        sharex=True,
        squeeze=False,
    )
    for row_index, scheme in enumerate(SCHEMES):
        for column_index, optimizer in enumerate(OPTIMIZERS):
            axis = axes[row_index][column_index]
            policies = selected.get((scheme, optimizer), {})
            plotted_names = []
            for policy in ("attached", "capped"):
                record = policies.get(policy)
                diagnostics = (
                    []
                    if record is None
                    else record.get("epoch_replay_records")
                    or record.get("epoch_diagnostics", [])
                )
                bias_names = sorted(
                    {
                        name
                        for diagnostic in diagnostics
                        for name in (
                            diagnostic.get(
                                "bias_weight_step_ratio_median_by_parameter"
                            )
                            or {}
                        )
                    }
                )
                for name in bias_names:
                    epochs = [int(item["epoch"]) for item in diagnostics]
                    values = [
                        _finite_or_nan(
                            item[
                                "bias_weight_step_ratio_median_by_parameter"
                            ].get(name)
                        )
                        for item in diagnostics
                    ]
                    axis.plot(
                        epochs,
                        values,
                        marker="o",
                        linestyle="-" if policy == "attached" else "--",
                        label=f"{policy}/{name}",
                    )
                    plotted_names.append(f"{policy}/{name}")
            if plotted_names:
                axis.axhline(1.0, color="black", linestyle=":", linewidth=1)
                axis.set_yscale("log")
                axis.legend(fontsize=8)
            else:
                axis.text(
                    0.5,
                    0.5,
                    "replay pending",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                )
            axis.set_title(f"{scheme} / {optimizer.upper()}")
            axis.set_xlabel("epoch (0 = initialization)")
            axis.set_ylabel("bias / attached-weight step")
            axis.grid(alpha=0.2)
    figure.suptitle("Conv2 bias-to-weight optimizer-step ratio")
    figure.tight_layout()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    return destination


def render_boundary_report(
    records: Sequence[Mapping[str, Any]], output_dir: str | Path
) -> list[Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    outputs = [
        plot_metric_heatmaps(
            records,
            root / "final_validation_loss_heatmap.png",
            field="final_validation_loss",
            title="Conv2 final validation loss: local high-rho diagnostic",
        ),
        plot_metric_heatmaps(
            records,
            root / "final_validation_accuracy_heatmap.png",
            field="final_validation_accuracy",
            title="Conv2 final validation accuracy: local high-rho diagnostic",
        ),
        plot_epochwise_effective_rho(
            records, root / "epochwise_effective_rho.png"
        ),
        plot_epochwise_bias_ratios(
            records, root / "epochwise_bias_weight_ratio.png"
        ),
    ]
    return outputs


__all__ = [
    "plot_epochwise_bias_ratios",
    "plot_epochwise_effective_rho",
    "plot_metric_heatmaps",
    "render_boundary_report",
]
