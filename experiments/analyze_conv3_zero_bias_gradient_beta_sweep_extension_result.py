#!/usr/bin/env python3
"""Join the Conv3 beta sweep with its one-decade-above-maximum extension."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import mean
import sys
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import analyze_conv3_zero_bias_gradient_beta_sweep_result as core  # noqa: E402
from experiments.reporting import sha256_file  # noqa: E402


DEFAULT_EXTENSION_CONFIG = (
    ROOT
    / "configs/conv/perfectdiode_conv3_zero_bias_adam_eqprop_bptt_gradient_beta_sweep_above_max_to_10xmax_sigma_5em4_seed0_20260817_v1.json"
)
DEFAULT_EXTENSION_ROOT = (
    ROOT
    / "results/perfectdiode-conv3-zero-bias-adam-centered-float64-eqprop-bptt-gradient-beta-sweep-above-max-to-10xmax-sigma-5em4-seed0-20260817-v1"
)


def _extension_points(
    parent_config: Mapping[str, Any], extension_config: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    contract = extension_config["sweep_contract"]
    parent_path = ROOT / str(contract["parent_config"])
    if sha256_file(parent_path) != str(contract["parent_config_sha256"]):
        raise ValueError("The reviewed parent sweep config changed.")
    if parent_config != core._read_json(parent_path):
        raise ValueError("Loaded parent config differs from the extension lineage.")
    for key in (
        "schema_version",
        "source_contract",
        "dataset",
        "gradient_contract",
        "read_noise_contract",
        "checkpoint_roles",
        "cases",
        "completion",
    ):
        if extension_config[key] != parent_config[key]:
            raise ValueError(f"Extension changed the parent scientific key {key}.")
    parent_points = core._sweep_points(parent_config)
    extension_points = [dict(row) for row in contract["beta_scale_points"]]
    if len(extension_points) != 8:
        raise ValueError("The above-maximum extension must contain eight points.")
    if [int(row["index"]) for row in extension_points] != list(range(9, 17)):
        raise ValueError("Extension point indices must continue from 9 through 16.")
    for row in extension_points:
        expected = 10.0 ** (int(row["index"]) / 8.0)
        if not math.isclose(
            float(row["beta_scale"]), expected, rel_tol=1.0e-14, abs_tol=0.0
        ):
            raise ValueError(f"Extension beta scale is off-grid: {row}.")
    combined = parent_points + extension_points
    factors = [float(row["beta_scale"]) for row in combined]
    if len(combined) != 17 or not math.isclose(factors[0], 1.0):
        raise ValueError("Combined beta sweep must contain 17 points from scale one.")
    if not math.isclose(factors[-1], 100.0):
        raise ValueError("Combined beta sweep must end at scale 100.")
    log_steps = [
        math.log10(right) - math.log10(left)
        for left, right in zip(factors, factors[1:])
    ]
    if max(log_steps) - min(log_steps) > 1.0e-12:
        raise ValueError("Combined beta sweep is not uniformly log-spaced.")
    if len({str(row["run_id"]) for row in combined}) != len(combined):
        raise ValueError("Combined beta sweep run IDs are not unique.")
    return parent_points, extension_points


def _average_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                int(row["point_index"]),
                str(row["scheme"]),
                str(row["checkpoint_role"]),
                str(row["parameter_name"]),
            )
        ].append(row)
    output: list[dict[str, Any]] = []
    for point_index in sorted({key[0] for key in grouped}):
        for scheme in core.SCHEMES:
            for role in core.ROLES:
                for parameter in core.PARAMETERS:
                    values = grouped[(point_index, scheme, role, parameter)]
                    if len(values) != 4:
                        raise ValueError(
                            f"Expected four batches for {point_index}/{scheme}/{role}/{parameter}."
                        )
                    output.append(
                        {
                            "point_index": point_index,
                            "source_study_id": values[0]["source_study_id"],
                            "run_id": values[0]["run_id"],
                            "beta_scale": float(values[0]["beta_scale"]),
                            "scheme": scheme,
                            "checkpoint_role": role,
                            "parameter_name": parameter,
                            "layer": core.LAYER_LABELS[parameter],
                            "base_beta": float(values[0]["base_beta"]),
                            "injected_beta": float(values[0]["injected_beta"]),
                            "batch_count": len(values),
                            "average_noisy_eqprop_bptt_cosine": mean(
                                float(row["cosine"]) for row in values
                            ),
                            "minimum_noisy_eqprop_bptt_cosine": min(
                                float(row["cosine"]) for row in values
                            ),
                            "maximum_noisy_eqprop_bptt_cosine": max(
                                float(row["cosine"]) for row in values
                            ),
                            "average_clean_eqprop_bptt_cosine": mean(
                                float(row["clean_eqprop_bptt_cosine"])
                                for row in values
                            ),
                        }
                    )
    return output


def _plot_best_checkpoint_average(
    rows: Sequence[Mapping[str, Any]], path: Path
) -> None:
    selected = [
        row for row in rows if row["checkpoint_role"] == "best_validation"
    ]
    all_values = [float(row["average_noisy_eqprop_bptt_cosine"]) for row in selected]
    lower = max(-1.0, min(-0.05, math.floor(min(all_values) * 10.0) / 10.0))
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 4.8), sharey=True)
    for axis, scheme in zip(axes, core.SCHEMES, strict=True):
        for parameter in core.PARAMETERS:
            values = sorted(
                (
                    row
                    for row in selected
                    if row["scheme"] == scheme
                    and row["parameter_name"] == parameter
                ),
                key=lambda row: float(row["injected_beta"]),
            )
            x = [float(row["injected_beta"]) for row in values]
            color = core.LAYER_COLORS[parameter]
            axis.plot(
                x,
                [float(row["average_noisy_eqprop_bptt_cosine"]) for row in values],
                color=color,
                marker="o",
                linewidth=1.8,
            )
            axis.plot(
                x,
                [float(row["average_clean_eqprop_bptt_cosine"]) for row in values],
                color=color,
                linestyle="--",
                linewidth=1.0,
                alpha=0.75,
            )
        axis.axhline(0.99, color="black", linestyle=":", linewidth=1.0)
        axis.set_xscale("log")
        axis.set_ylim(lower, 1.015)
        axis.grid(True, which="both", alpha=0.22)
        axis.set_title(f"{scheme} — best checkpoint")
        axis.set_xlabel("injected beta magnitude B")
    axes[0].set_ylabel("mean EqProp vs BPTT cosine over four batches")
    layer_handles = [
        Line2D(
            [0],
            [0],
            color=core.LAYER_COLORS[name],
            linewidth=2,
            label=core.LAYER_LABELS[name],
        )
        for name in core.PARAMETERS
    ]
    style_handles = [
        Line2D([0], [0], color="black", marker="o", label="noisy EqProp"),
        Line2D([0], [0], color="black", linestyle="--", label="clean EqProp"),
        Line2D([0], [0], color="black", linestyle=":", label="cosine = .99"),
    ]
    fig.legend(
        handles=layer_handles + style_handles,
        loc="lower center",
        ncol=7,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.suptitle(
        "Conv3 best-checkpoint average cosine vs beta\n"
        "17 points; endpoint read noise sigma = 5e-4; T=K=8",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 0.90))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_report(
    *,
    path: Path,
    context_rows: Sequence[Mapping[str, Any]],
    threshold_rows: Sequence[Mapping[str, Any]],
    receipts: Sequence[Mapping[str, Any]],
    bptt_guard: Mapping[str, Any],
    noise_guard: Mapping[str, Any],
) -> None:
    lines = [
        "# Conv3 broadened all-layer cosine versus beta under endpoint read noise",
        "",
        "## Outcome",
        "",
        "The reviewed nine-point sweep is joined with eight new points one decade above the original maximum, giving 17 uniformly log-spaced beta points. Both reconstructed initialization and best-validation checkpoints use the same four validation minibatches and identical endpoint-noise tensors at every point.",
        "",
        "### All-layer worst-batch result",
        "",
        "| Scheme | Checkpoint | First beta with all layers/batches cosine >= .99 | Best observed beta | Best observed worst noisy cosine | Clean worst cosine there |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in context_rows:
        lines.append(
            "| {scheme} | {role} | {first} | {best_beta} | {best_cos:.6f} | {clean:.6f} |".format(
                scheme=row["scheme"],
                role=core.ROLE_LABELS[str(row["checkpoint_role"])],
                first=core._format_beta(
                    row["first_all_layer_all_batch_cosine_ge_0p99_beta"]
                ),
                best_beta=core._format_beta(row["best_observed_beta"]),
                best_cos=float(row["best_observed_worst_noisy_cosine"]),
                clean=float(row["clean_worst_cosine_at_best_observed_beta"]),
            )
        )
    lines.extend(
        [
            "",
            "### Per-layer best observed point",
            "",
            "| Scheme | Checkpoint | Layer | First all-batch beta >= .99 | Best beta | Best minimum noisy cosine |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for row in threshold_rows:
        lines.append(
            "| {scheme} | {role} | {layer} | {first} | {best_beta} | {best_cos:.6f} |".format(
                scheme=row["scheme"],
                role=core.ROLE_LABELS[str(row["checkpoint_role"])],
                layer=row["layer"],
                first=core._format_beta(row["first_all_batch_cosine_ge_0p99_beta"]),
                best_beta=core._format_beta(row["best_observed_beta"]),
                best_cos=float(row["best_observed_minimum_noisy_cosine"]),
            )
        )
    lines.extend(
        [
            "",
            "## Contract and integrity",
            "",
            f"- Coverage: `{len(receipts)}/17` valid production bundles, `{len(receipts) * 24}` checkpoint-batch replays, `{len(receipts) * 96}` layer-batch comparisons, and `{len(receipts) * 192}` endpoint-noise records.",
            "- Combined injected beta ranges: baseline `100--10000`, ours `3--300`, and legacy `.001--.1`.",
            "- Shared contract: Conv3, exact-zero biases, `T=K=8`, centered frozen-current true-float64 EqProp, exact same-T/K BPTT reference, four fixed 16-example validation batches, endpoint read-noise sigma `5e-4`, and no optimizer step or official-test read.",
            f"- BPTT invariance guard: passed over `{bptt_guard['coordinate_count']}` coordinates; maximum L2-norm delta `{bptt_guard['maximum_absolute_bptt_l2_delta']:.3g}`.",
            f"- Cross-beta noise-draw guard: `{noise_guard['matched_signature_count']}/17` signatures match exactly.",
            "- The trained Conv3-baseline free-state residual caveat at `T=8` remains separate from these gradient curves.",
            "",
            "## Artifacts",
            "",
            "- `best_checkpoint_average_cosine_vs_beta.csv`: arithmetic mean over all four batches for every layer and beta.",
            "- `plots/best_checkpoint_average_cosine_vs_injected_beta_all_layers.png`: requested best-checkpoint average curves.",
            "- `layerwise_cosine_vs_beta.csv`: median, minimum, and maximum over batches for both checkpoint roles.",
            "- `per_layer_thresholds.csv` and `context_thresholds.csv`: conservative `.99` crossings and best observed points.",
            "- `summary.json`: parent/extension result hashes and integrity guards.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    extension_config_path = args.extension_config.expanduser().resolve()
    extension_root = args.extension_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    extension_config = core._read_json(extension_config_path)
    if extension_config.get("study_id") != extension_root.name:
        raise ValueError("Extension config study ID and result directory differ.")
    parent_config_path = ROOT / str(
        extension_config["sweep_contract"]["parent_config"]
    )
    parent_root = ROOT / str(
        extension_config["sweep_contract"]["parent_result_root"]
    )
    parent_config = core._read_json(parent_config_path)
    parent_points, extension_points = _extension_points(
        parent_config, extension_config
    )
    sources = (
        (parent_config, parent_config_path, parent_root, parent_points),
        (
            extension_config,
            extension_config_path,
            extension_root,
            extension_points,
        ),
    )
    all_rows: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    signatures: list[list[tuple[Any, ...]]] = []
    for config, config_path, study_root, points in sources:
        for point in points:
            rows, receipt, signature = core._validate_point(
                config=config,
                config_path=config_path,
                study_root=study_root,
                point=point,
            )
            for row in rows:
                row["source_study_id"] = config["study_id"]
            receipt["source_study_id"] = config["study_id"]
            all_rows.extend(rows)
            receipts.append(receipt)
            signatures.append(signature)
    matched_signatures = sum(signature == signatures[0] for signature in signatures)
    if matched_signatures != 17:
        raise RuntimeError("Endpoint-noise draws changed across the combined sweep.")
    noise_guard = {
        "point_count": len(signatures),
        "matched_signature_count": matched_signatures,
        "signature_row_count": len(signatures[0]),
        "passed": True,
    }
    bptt_guard = core._bptt_invariance_guard(all_rows)
    layer_rows = core._aggregate(all_rows)
    threshold_rows = core._threshold_rows(layer_rows)
    context_rows = core._context_rows(layer_rows)
    average_rows = _average_rows(all_rows)

    core._write_csv(output_dir / "layer_metrics_all_points.csv", all_rows)
    core._write_csv(output_dir / "layerwise_cosine_vs_beta.csv", layer_rows)
    core._write_csv(
        output_dir / "best_checkpoint_average_cosine_vs_beta.csv",
        [row for row in average_rows if row["checkpoint_role"] == "best_validation"],
    )
    core._write_csv(output_dir / "per_layer_thresholds.csv", threshold_rows)
    core._write_csv(output_dir / "context_thresholds.csv", context_rows)
    core._plot_layerwise(
        layer_rows,
        path=output_dir / "plots/median_cosine_vs_injected_beta_all_layers.png",
        worst_only=False,
    )
    core._plot_layerwise(
        layer_rows,
        path=output_dir / "plots/worst_batch_cosine_vs_injected_beta_all_layers.png",
        worst_only=True,
    )
    _plot_best_checkpoint_average(
        average_rows,
        output_dir
        / "plots/best_checkpoint_average_cosine_vs_injected_beta_all_layers.png",
    )
    summary = {
        "schema_version": "conv3-zero-bias-gradient-beta-sweep-extension-analysis/v1",
        "study_id": extension_config["study_id"],
        "parent_study_id": parent_config["study_id"],
        "parent_config_path": str(parent_config_path),
        "parent_config_sha256": sha256_file(parent_config_path),
        "extension_config_path": str(extension_config_path),
        "extension_config_sha256": sha256_file(extension_config_path),
        "point_count": 17,
        "parent_point_count": 9,
        "extension_point_count": 8,
        "replay_count": 17 * 24,
        "layer_comparison_count": len(all_rows),
        "endpoint_noise_record_count": 17 * 192,
        "run_receipts": receipts,
        "bptt_invariance_guard": bptt_guard,
        "cross_beta_noise_draw_guard": noise_guard,
        "context_thresholds": context_rows,
        "per_layer_thresholds": threshold_rows,
        "official_test_read": False,
        "optimizer_steps_applied": False,
    }
    core._write_json(output_dir / "summary.json", summary)
    _write_report(
        path=output_dir / "report.md",
        context_rows=context_rows,
        threshold_rows=threshold_rows,
        receipts=receipts,
        bptt_guard=bptt_guard,
        noise_guard=noise_guard,
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--extension-config", type=Path, default=DEFAULT_EXTENSION_CONFIG
    )
    parser.add_argument("--extension-root", type=Path, default=DEFAULT_EXTENSION_ROOT)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if args.output_dir is None:
        args.output_dir = args.extension_root / "analysis"
    return args


def main(argv: Sequence[str] | None = None) -> int:
    summary = run(parse_args(argv))
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
