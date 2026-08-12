#!/usr/bin/env python3
"""Analyze the trained-Conv3 ours EqProp tail above injected beta 10.

The analyzer joins the beta 3/5/10 one- versus two-sided anchors with the
dedicated beta 30/100/300/1000 extension.  It reuses the canonical-bundle
validator and metadata normalization from the parent comparison, but keeps
the parent analyzer's required baseline+ours coverage gate unchanged.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from experiments import analyze_conv3_eqprop_one_vs_two_sided_higher_beta as common
from experiments import reporting
DEFAULT_STUDY_ROOT = (
    REPOSITORY_ROOT
    / "results/perfectdiode-conv3-ours-eqprop-beta-above10-bug-audit-seed0-20260811-v1"
)
DEFAULT_ANCHOR_ROOT = (
    REPOSITORY_ROOT
    / "results/perfectdiode-conv3-baseline-ours-eqprop-one-vs-two-sided-higher-beta-seed0-20260811-v1"
)
SCHEMA = "conv3-ours-eqprop-beta-above10-analysis/v1"
EXPECTED_BETAS = (3.0, 5.0, 10.0, 30.0, 100.0, 300.0, 1000.0)
EXTENSION_BETAS = (30.0, 100.0, 300.0, 1000.0)
AMPLIFICATION_FACTOR = 64.0


def _close(left: float, right: float, *, rel_tol: float = 1.0e-9) -> bool:
    return math.isclose(float(left), float(right), rel_tol=rel_tol, abs_tol=1.0e-15)


def _rows_by_context(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[float, str, str], list[Mapping[str, Any]]]:
    grouped: dict[tuple[float, str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (
            float(row["injected_beta"]),
            str(row["eqprop_variant"]),
            str(row["precision"]),
        )
        grouped.setdefault(key, []).append(row)
    return grouped


def _validate_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("No normalized ours rows were found.")
    if {row["scheme"] for row in rows} != {"ours"}:
        raise ValueError("The beta-above-10 analysis is scoped to ours only.")
    if {(int(row["T"]), int(row["K"])) for row in rows} != {(64, 64)}:
        raise ValueError("Every row must use T=K=64.")
    if {float(row["injected_beta"]) for row in rows} != set(EXPECTED_BETAS):
        raise ValueError("The joined beta grid is incomplete.")
    if len({row["batch_payload_sha256"] for row in rows}) != 1:
        raise ValueError("The joined rows do not share one fixed minibatch.")
    if len({row["batch_source_indices_sha256"] for row in rows}) != 1:
        raise ValueError("The joined rows do not share fixed source indices.")
    if len({row["best_checkpoint_sha256"] for row in rows}) != 1:
        raise ValueError("The joined rows do not share one checkpoint.")
    if len({row["source_selection_row_sha256"] for row in rows}) != 1:
        raise ValueError("The joined rows do not share one source selection.")

    grouped = _rows_by_context(rows)
    expected_keys = {
        (beta, variant, precision)
        for beta in EXPECTED_BETAS
        for variant in common.VARIANTS
        for precision in common.PRECISIONS
    }
    if set(grouped) != expected_keys:
        missing = sorted(expected_keys.difference(grouped))
        extra = sorted(set(grouped).difference(expected_keys))
        raise ValueError(f"Incomplete beta/variant/precision coverage: missing={missing}, extra={extra}.")
    for key, context in grouped.items():
        names = {row["parameter_name"] for row in context}
        if names != set(common.PARAMETERS) or len(context) != len(common.PARAMETERS):
            raise ValueError(f"Incomplete or duplicate layer coverage for {key}.")
        for row in context:
            if row["gradient_status"] != "numeric":
                raise ValueError(f"Undefined gradient in {key}/{row['parameter_name']}.")
            if row["bias_excluded"] is not True:
                raise ValueError("A scored row includes a bias gradient.")
            if row["residual_limited"] is not False:
                raise ValueError("An included ours endpoint is residual-limited.")
            if not _close(
                float(row["actual_base_beta"]) * AMPLIFICATION_FACTOR,
                float(row["injected_beta"]),
            ):
                raise ValueError("The guarded Conv3 beta scaling is inconsistent.")

    for precision in common.PRECISIONS:
        for parameter in common.PARAMETERS:
            references = [
                float(row["bptt_l2"])
                for row in rows
                if row["precision"] == precision
                and row["parameter_name"] == parameter
            ]
            if any(
                not math.isclose(value, references[0], rel_tol=1.0e-6, abs_tol=1.0e-12)
                for value in references[1:]
            ):
                raise ValueError(f"BPTT reference changed for {precision}/{parameter}.")

    for beta in EXPECTED_BETAS:
        for precision in common.PRECISIONS:
            for parameter in common.PARAMETERS:
                paired = [
                    row
                    for row in rows
                    if _close(float(row["injected_beta"]), beta)
                    and row["precision"] == precision
                    and row["parameter_name"] == parameter
                ]
                if len(paired) != 2:
                    raise ValueError("Missing paired estimator displacement rows.")
                positives = [
                    float(row["matched_zero_positive_relative_displacement"])
                    for row in paired
                ]
                if not _close(positives[0], positives[1], rel_tol=1.0e-12):
                    raise ValueError(
                        f"Positive endpoint displacement changed across estimators for "
                        f"beta={beta}/{precision}/{parameter}."
                    )


def _read_phase_hashes(
    bundles: Sequence[Mapping[str, Any]],
) -> dict[tuple[float, str, str, str], str]:
    hashes: dict[tuple[float, str, str, str], str] = {}
    for bundle in bundles:
        if "ours" not in bundle["schemes"]:
            continue
        parameter_rows = [
            row for row in bundle["parameter_rows"] if row["scheme"] == "ours"
        ]
        beta = float(parameter_rows[0]["effective_beta"])
        if beta not in EXPECTED_BETAS:
            continue
        phase_path = Path(bundle["run_dir"]) / "phase_diagnostics.csv"
        with phase_path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            if row["scheme"] != "ours" or row["phase"] not in {"zero", "positive"}:
                continue
            key = (beta, str(bundle["variant"]), row["precision"], row["phase"])
            if key in hashes:
                raise ValueError(f"Duplicate phase hash context {key}.")
            hashes[key] = row["state_sha256"]
    return hashes


def _validate_paired_endpoint_hashes(
    hashes: Mapping[tuple[float, str, str, str], str]
) -> None:
    for beta in EXPECTED_BETAS:
        for precision in common.PRECISIONS:
            for phase in ("zero", "positive"):
                values = {
                    hashes.get((beta, variant, precision, phase))
                    for variant in common.VARIANTS
                }
                if None in values or len(values) != 1:
                    raise ValueError(
                        f"Paired {phase} state hashes differ for beta={beta}/{precision}."
                    )


def _load_joined_rows(
    study_root: Path, anchor_root: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    extension_bundles, extension_excluded = common.discover_completed_bundles(
        study_root, source_kind="extension"
    )
    anchor_bundles, anchor_excluded = common.discover_completed_bundles(
        anchor_root, source_kind="anchor"
    )
    extension_rows = [
        row
        for bundle in extension_bundles
        for row in common.normalize_bundle(bundle)
        if row["scheme"] == "ours"
        and float(row["injected_beta"]) in EXTENSION_BETAS
        and (int(row["T"]), int(row["K"])) == (64, 64)
    ]
    anchor_rows = [
        row
        for bundle in anchor_bundles
        for row in common.normalize_bundle(bundle)
        if row["scheme"] == "ours"
        and float(row["injected_beta"]) in {3.0, 5.0, 10.0}
        and (int(row["T"]), int(row["K"])) == (64, 64)
    ]
    rows = sorted(
        [*anchor_rows, *extension_rows],
        key=lambda row: (
            float(row["injected_beta"]),
            common.PRECISIONS.index(row["precision"]),
            common.VARIANTS.index(row["eqprop_variant"]),
            common.PARAMETERS.index(row["parameter_name"]),
        ),
    )
    _validate_rows(rows)
    all_bundles = [*anchor_bundles, *extension_bundles]
    phase_hashes = _read_phase_hashes(all_bundles)
    _validate_paired_endpoint_hashes(phase_hashes)
    return rows, all_bundles, [*anchor_excluded, *extension_excluded]


def _context(
    rows: Sequence[Mapping[str, Any]], *, beta: float, variant: str, precision: str
) -> dict[str, Mapping[str, Any]]:
    selected = {
        row["parameter_name"]: row
        for row in rows
        if _close(float(row["injected_beta"]), beta)
        and row["eqprop_variant"] == variant
        and row["precision"] == precision
    }
    if set(selected) != set(common.PARAMETERS):
        raise ValueError("Incomplete report context.")
    return selected


def _format_vector(values: Sequence[float]) -> str:
    return " / ".join(f"{value:.6f}" for value in values)


def _write_markdown(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    lines = [
        "# Conv3 ours EqProp above injected beta 10",
        "",
        "All rows replay the trained BPTT Conv3 ours best-validation checkpoint on the same 16-example ordinary-MNIST validation batch at `T=K=64`. Biases are active in the equilibria and excluded from gradient scoring.",
        "",
        "The reported beta is the injected-current magnitude `B`. The scalar assigned to `augmented_fn.nudging` is `B/64`, and both the physical current term and EqProp denominator use `B=(B/64)*64`.",
        "",
        "## Layerwise EqProp/BPTT cosine",
        "",
        "Vectors are `C0 / C1 / C2 / Dense`.",
        "",
        "| injected B | base beta | estimator | float32 cosine | float64 cosine |",
        "|---:|---:|---|---|---|",
    ]
    for beta in EXPECTED_BETAS:
        for variant in common.VARIANTS:
            values = []
            for precision in common.PRECISIONS:
                selected = _context(rows, beta=beta, variant=variant, precision=precision)
                values.append(
                    _format_vector(
                        [float(selected[name]["eqprop_vs_bptt_cosine"]) for name in common.PARAMETERS]
                    )
                )
            lines.append(
                f"| {beta:g} | {beta / AMPLIFICATION_FACTOR:.9g} | "
                f"{'one-sided' if variant == 'positive_one_sided' else 'centered'} | "
                f"{values[0]} | {values[1]} |"
            )

    lines.extend(
        [
            "",
            "## Centered matched-zero displacement",
            "",
            "Float64 cells are `negative / positive` relative L2 displacement. Float32 has the same scientific scale and is retained in the CSV.",
            "",
            "| injected B | H0 | H1 | H2 | output |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for beta in EXPECTED_BETAS:
        selected = _context(rows, beta=beta, variant="centered", precision="float64")
        cells = []
        for name in common.PARAMETERS:
            row = selected[name]
            cells.append(
                f"{float(row['matched_zero_negative_relative_displacement']):.6g} / "
                f"{float(row['matched_zero_positive_relative_displacement']):.6g}"
            )
        lines.append(f"| {beta:g} | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Integrity",
            "",
            "Every included canonical bundle validates. All rows share the checkpoint, cohort, batch bytes, BPTT reference, and `T/K`; all ours endpoint residual gates pass. For every beta and precision, the zero and positive endpoint state hashes are identical between the one-sided and centered runs. No optimizer step or official-test read occurred.",
            "",
            "The source grid's `0.01` injected-current cap is bypassed deliberately for this diagnostic. These very large betas are not candidate training settings.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _render_plot(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    figure, axes = plt.subplots(2, 4, figsize=(15.5, 7.2), sharex="col")
    styles = {
        ("float32", "positive_one_sided"): ("#D55E00", "-", "o", "float32 one-sided"),
        ("float32", "centered"): ("#D55E00", "--", "s", "float32 centered"),
        ("float64", "positive_one_sided"): ("#0072B2", "-", "o", "float64 one-sided"),
        ("float64", "centered"): ("#0072B2", "--", "s", "float64 centered"),
    }
    for index, parameter in enumerate(common.PARAMETERS):
        cosine_axis = axes[0, index]
        displacement_axis = axes[1, index]
        for (precision, variant), (color, linestyle, marker, label) in styles.items():
            series = sorted(
                [
                    row
                    for row in rows
                    if row["parameter_name"] == parameter
                    and row["precision"] == precision
                    and row["eqprop_variant"] == variant
                ],
                key=lambda row: float(row["injected_beta"]),
            )
            cosine_axis.plot(
                [row["injected_beta"] for row in series],
                [row["eqprop_vs_bptt_cosine"] for row in series],
                color=color,
                linestyle=linestyle,
                marker=marker,
                linewidth=1.5,
                markersize=4.5,
                label=label,
            )
        for precision, color, marker in (
            ("float32", "#D55E00", "o"),
            ("float64", "#0072B2", "s"),
        ):
            series = sorted(
                [
                    row
                    for row in rows
                    if row["parameter_name"] == parameter
                    and row["precision"] == precision
                    and row["eqprop_variant"] == "centered"
                ],
                key=lambda row: float(row["injected_beta"]),
            )
            displacement_axis.plot(
                [row["injected_beta"] for row in series],
                [row["matched_zero_positive_relative_displacement"] for row in series],
                color=color,
                marker=marker,
                linewidth=1.5,
                markersize=4.5,
                label=precision,
            )
        cosine_axis.axhline(0.99, color="#555555", linestyle=":", linewidth=1.0)
        cosine_axis.set_ylim(-1.05, 1.03)
        cosine_axis.set_title(common.LAYER_LABELS[parameter])
        displacement_axis.set_yscale("log")
        for axis in (cosine_axis, displacement_axis):
            axis.set_xscale("log")
            axis.grid(True, which="both", alpha=0.2)
        displacement_axis.set_xlabel("injected beta B")
        if index == 0:
            cosine_axis.set_ylabel("EqProp–BPTT cosine")
            displacement_axis.set_ylabel("positive relative displacement")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.90),
        ncol=4,
        frameon=False,
    )
    figure.suptitle(
        "Trained Conv3 ours: increasing EqProp beta beyond 10\n"
        "true float32/float64, one-sided versus centered, T=K=64",
        y=0.985,
    )
    figure.tight_layout(rect=(0.02, 0.02, 1.0, 0.84))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run_analysis(study_root: Path, anchor_root: Path, output_dir: Path) -> dict[str, Any]:
    rows, bundles, excluded = _load_joined_rows(study_root, anchor_root)
    output_dir = output_dir.expanduser().resolve()
    output_csv = output_dir / "conv3_ours_eqprop_beta_above10.csv"
    output_markdown = output_dir / "conv3_ours_eqprop_beta_above10.md"
    output_plot = output_dir / "conv3_ours_eqprop_beta_above10.png"
    output_json = output_dir / "conv3_ours_eqprop_beta_above10.json"
    common.write_csv(rows, output_csv)
    _write_markdown(rows, output_markdown)
    _render_plot(rows, output_plot)
    included_ids = sorted(
        {
            row["run_id"]
            for row in rows
        }
    )
    included_bundles = [bundle for bundle in bundles if bundle["run_id"] in included_ids]
    summary = {
        "schema": SCHEMA,
        "architecture": "conv3",
        "scheme": "ours",
        "checkpoint_role": "best_validation",
        "T": 64,
        "K": 64,
        "injected_betas": list(EXPECTED_BETAS),
        "base_betas": [value / AMPLIFICATION_FACTOR for value in EXPECTED_BETAS],
        "amplification_factor": AMPLIFICATION_FACTOR,
        "row_count": len(rows),
        "included_bundles": [
            {
                "run_id": bundle["run_id"],
                "path": str(bundle["run_dir"]),
                "result_sha256": bundle["result_sha256"],
            }
            for bundle in included_bundles
        ],
        "excluded": excluded,
        "guards": {
            "canonical_bundles_valid": True,
            "fixed_checkpoint_cohort_and_tk": True,
            "base_times_64_equals_injected_beta": True,
            "paired_zero_and_positive_state_hashes_identical": True,
            "bias_gradients_excluded": True,
            "optimizer_steps_applied": False,
            "official_test_read": False,
        },
        "outputs": {
            "csv": {"path": str(output_csv), "sha256": reporting.sha256_file(output_csv)},
            "markdown": {
                "path": str(output_markdown),
                "sha256": reporting.sha256_file(output_markdown),
            },
            "plot": {"path": str(output_plot), "sha256": reporting.sha256_file(output_plot)},
        },
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "row_count": len(rows),
        "bundle_count": len(included_bundles),
        "output_csv": str(output_csv),
        "output_markdown": str(output_markdown),
        "output_plot": str(output_plot),
        "output_json": str(output_json),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=DEFAULT_STUDY_ROOT)
    parser.add_argument("--anchor-root", type=Path, default=DEFAULT_ANCHOR_ROOT)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    study_root = args.study_root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else study_root / "analysis"
    )
    result = run_analysis(study_root, args.anchor_root.expanduser().resolve(), output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
