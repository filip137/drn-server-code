#!/usr/bin/env python3
"""Assemble and plot trained Conv3 true-dtype EqProp/BPTT beta curves.

Inputs are canonical literal-current float32/float64 shadow CSVs.  The output
uses the common requested pre-amplification beta-hat on the x axis, while
retaining the actual base beta, injected beta, and cap status in the table.
An exactly-zero EqProp gradient is categorical evidence: it is marked on a
dedicated ``ZERO`` rail and is never converted to cosine zero.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt


SCHEMES = ("baseline", "ours", "legacy")
DTYPES = ("float32", "float64")
PARAMETERS = ("ConvWeight_0", "ConvWeight_1", "ConvWeight_2", "DenseWeight_0")
LAYER_LABELS = {
    "ConvWeight_0": "C0",
    "ConvWeight_1": "C1",
    "ConvWeight_2": "C2",
    "DenseWeight_0": "D",
}
LAYER_COLORS = {
    "ConvWeight_0": "#0072B2",
    "ConvWeight_1": "#E69F00",
    "ConvWeight_2": "#009E73",
    "DenseWeight_0": "#CC79A7",
}
EXPECTED_BETAS = {
    "baseline": (
        1e-10, 3e-10, 1e-9, 3e-9, 1e-8, 3e-8, 1e-7,
        3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4,
    ),
    "ours": (
        1e-10, 3e-10, 1e-9, 3e-9, 1e-8, 3e-8, 1e-7,
        3e-7, 1e-6, 3e-6, 1e-5,
    ),
    "legacy": (1e-10, 3e-10, 1e-9, 3e-9, 1e-8, 3e-8, 1e-7),
}
REQUIRED_COLUMNS = {
    "architecture",
    "scheme",
    "checkpoint_role",
    "T",
    "K",
    "batch_index",
    "batch_payload_sha256",
    "parameter_name",
    "bias_excluded",
    "actual_beta",
    "effective_beta",
    "beta_hat_requested",
    "beta_capped",
    "residual_limited",
    *(f"{dtype}_eqprop_l2" for dtype in DTYPES),
    *(f"{dtype}_bptt_l2" for dtype in DTYPES),
    *(f"{dtype}_exact_zero_fraction" for dtype in DTYPES),
    *(f"{dtype}_eqprop_vs_bptt_cosine" for dtype in DTYPES),
    *(f"{dtype}_eqprop_over_bptt_norm_ratio" for dtype in DTYPES),
}
OUTPUT_FIELDS = (
    "scheme",
    "beta_hat_requested",
    "actual_base_beta",
    "injected_beta",
    "beta_capped",
    "T",
    "K",
    "batch_index",
    "batch_payload_sha256",
    "parameter_name",
    "layer",
    "bias_excluded",
    "residual_limited",
    "float32_cosine",
    "float32_status",
    "float32_eqprop_l2",
    "float32_bptt_l2",
    "float32_eqprop_over_bptt_norm_ratio",
    "float64_cosine",
    "float64_status",
    "float64_eqprop_l2",
    "float64_bptt_l2",
    "float64_eqprop_over_bptt_norm_ratio",
    "source_csv",
)


def _parse_bool(value: object) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no", ""}:
        return False
    raise ValueError(f"Expected a boolean, got {value!r}.")


def _optional_float(value: object) -> float | None:
    normalized = str(value).strip().lower()
    if normalized in {"", "none", "null", "nan"}:
        return None
    result = float(normalized)
    if not math.isfinite(result):
        raise ValueError(f"Expected a finite float or blank, got {value!r}.")
    return result


def _status(row: Mapping[str, str], dtype: str) -> str:
    cosine = _optional_float(row[f"{dtype}_eqprop_vs_bptt_cosine"])
    if cosine is not None:
        if not -1.000001 <= cosine <= 1.000001:
            raise ValueError(f"Cosine outside [-1,1]: {cosine}.")
        return "numeric"
    eqprop_l2 = float(row[f"{dtype}_eqprop_l2"])
    zero_fraction = float(row[f"{dtype}_exact_zero_fraction"])
    if eqprop_l2 == 0.0 and math.isclose(zero_fraction, 1.0, abs_tol=1e-12):
        return "exact_zero"
    return "undefined"


def _same_row(left: Mapping[str, str], right: Mapping[str, str]) -> bool:
    fields = REQUIRED_COLUMNS.difference({"batch_payload_sha256"})
    return all(left[field] == right[field] for field in fields) and (
        left["batch_payload_sha256"] == right["batch_payload_sha256"]
    )


def load_curve_rows(input_csvs: Sequence[Path]) -> list[dict[str, str]]:
    selected: dict[tuple[str, float, str], dict[str, str]] = {}
    for supplied_path in input_csvs:
        path = supplied_path.expanduser().resolve()
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            missing = REQUIRED_COLUMNS.difference(reader.fieldnames or ())
            if missing:
                raise ValueError(f"{path} is missing columns: {sorted(missing)}")
            for raw in reader:
                if raw["architecture"] != "conv3" or raw["checkpoint_role"] != "best_validation":
                    continue
                scheme = raw["scheme"]
                parameter = raw["parameter_name"]
                if scheme not in SCHEMES or parameter not in PARAMETERS:
                    raise ValueError(f"Unexpected Conv3 row in {path}: {scheme}/{parameter}.")
                if not _parse_bool(raw["bias_excluded"]):
                    raise ValueError(f"Bias-exclusion guard failed in {path}.")
                row = dict(raw)
                row["source_csv"] = str(path)
                beta = float(row["beta_hat_requested"])
                key = (scheme, beta, parameter)
                if key in selected:
                    if not _same_row(selected[key], row):
                        raise ValueError(f"Conflicting duplicate curve row for {key}.")
                    continue
                selected[key] = row

    for scheme, expected_betas in EXPECTED_BETAS.items():
        observed = sorted({beta for item_scheme, beta, _ in selected if item_scheme == scheme})
        if len(observed) != len(expected_betas) or any(
            not math.isclose(left, right, rel_tol=1e-12, abs_tol=0.0)
            for left, right in zip(observed, expected_betas, strict=True)
        ):
            raise ValueError(
                f"Incomplete beta coverage for {scheme}: observed={observed}, "
                f"expected={list(expected_betas)}."
            )
        for beta in expected_betas:
            present = {
                parameter
                for item_scheme, item_beta, parameter in selected
                if item_scheme == scheme and math.isclose(item_beta, beta, rel_tol=1e-12)
            }
            if present != set(PARAMETERS):
                raise ValueError(f"Incomplete layer coverage for {scheme}/{beta}: {present}.")

    batch_hashes = {row["batch_payload_sha256"] for row in selected.values()}
    batch_indices = {row["batch_index"] for row in selected.values()}
    if len(batch_hashes) != 1 or batch_indices != {"0"}:
        raise ValueError("Curve inputs do not share fixed batch zero.")
    return [
        selected[key]
        for key in sorted(
            selected,
            key=lambda item: (SCHEMES.index(item[0]), item[1], PARAMETERS.index(item[2])),
        )
    ]


def normalized_rows(rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        item: dict[str, Any] = {
            "scheme": row["scheme"],
            "beta_hat_requested": float(row["beta_hat_requested"]),
            "actual_base_beta": float(row["actual_beta"]),
            "injected_beta": float(row["effective_beta"]),
            "beta_capped": _parse_bool(row["beta_capped"]),
            "T": int(row["T"]),
            "K": int(row["K"]),
            "batch_index": int(row["batch_index"]),
            "batch_payload_sha256": row["batch_payload_sha256"],
            "parameter_name": row["parameter_name"],
            "layer": LAYER_LABELS[row["parameter_name"]],
            "bias_excluded": True,
            "residual_limited": _parse_bool(row["residual_limited"]),
            "source_csv": row["source_csv"],
        }
        for dtype in DTYPES:
            item[f"{dtype}_cosine"] = _optional_float(
                row[f"{dtype}_eqprop_vs_bptt_cosine"]
            )
            item[f"{dtype}_status"] = _status(row, dtype)
            item[f"{dtype}_eqprop_l2"] = float(row[f"{dtype}_eqprop_l2"])
            item[f"{dtype}_bptt_l2"] = float(row[f"{dtype}_bptt_l2"])
            item[f"{dtype}_eqprop_over_bptt_norm_ratio"] = _optional_float(
                row[f"{dtype}_eqprop_over_bptt_norm_ratio"]
            )
        output.append(item)
    return output


def write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _display_cosine(row: Mapping[str, Any], dtype: str) -> str:
    status = row[f"{dtype}_status"]
    if status == "exact_zero":
        return "ZERO"
    if status == "undefined":
        return "NaN"
    return f"{float(row[f'{dtype}_cosine']):.6f}"


def write_markdown(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    lines = [
        "# Conv3 EqProp--BPTT cosine versus beta",
        "",
        "Best-validation checkpoints; fixed batch 0 (16 examples); native T/K; biases excluded.",
        "The x variable is common requested pre-amplification beta-hat. `ZERO` means the EqProp gradient is exactly zero, so its cosine is undefined. Capped endpoints are not matched-base-beta comparisons.",
        "",
    ]
    for scheme in SCHEMES:
        lines.extend(
            [
                f"## {scheme}",
                "",
                "| requested beta-hat | actual base beta | injected beta | cap | f32 C0 | f32 C1 | f32 C2 | f32 D | f64 C0 | f64 C1 | f64 C2 | f64 D |",
                "|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        scheme_rows = [row for row in rows if row["scheme"] == scheme]
        for beta in EXPECTED_BETAS[scheme]:
            layer_rows = {
                row["parameter_name"]: row
                for row in scheme_rows
                if math.isclose(float(row["beta_hat_requested"]), beta, rel_tol=1e-12)
            }
            anchor = layer_rows[PARAMETERS[0]]
            values = [
                _display_cosine(layer_rows[parameter], dtype)
                for dtype in DTYPES
                for parameter in PARAMETERS
            ]
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"{beta:.0e}",
                        f"{float(anchor['actual_base_beta']):.9g}",
                        f"{float(anchor['injected_beta']):.9g}",
                        "yes" if anchor["beta_capped"] else "no",
                        *values,
                    ]
                )
                + " |"
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def render_plot(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    figure, axes = plt.subplots(3, 2, figsize=(15.5, 13.5), sharex=True, sharey=True)
    zero_rail = -0.075
    for row_index, scheme in enumerate(SCHEMES):
        scheme_rows = [row for row in rows if row["scheme"] == scheme]
        for column_index, dtype in enumerate(DTYPES):
            axis = axes[row_index, column_index]
            for parameter in PARAMETERS:
                parameter_rows = [
                    row for row in scheme_rows if row["parameter_name"] == parameter
                ]
                numeric = [row for row in parameter_rows if row[f"{dtype}_status"] == "numeric"]
                axis.plot(
                    [row["beta_hat_requested"] for row in numeric],
                    [row[f"{dtype}_cosine"] for row in numeric],
                    color=LAYER_COLORS[parameter],
                    marker="o",
                    markersize=4.5,
                    linewidth=1.7,
                    label=LAYER_LABELS[parameter],
                )
                exact_zero = [
                    row for row in parameter_rows if row[f"{dtype}_status"] == "exact_zero"
                ]
                if exact_zero:
                    axis.scatter(
                        [row["beta_hat_requested"] for row in exact_zero],
                        [zero_rail] * len(exact_zero),
                        color=LAYER_COLORS[parameter],
                        marker="X",
                        s=45,
                        zorder=5,
                    )
                undefined = [
                    row for row in parameter_rows if row[f"{dtype}_status"] == "undefined"
                ]
                if undefined:
                    axis.scatter(
                        [row["beta_hat_requested"] for row in undefined],
                        [zero_rail] * len(undefined),
                        facecolors="none",
                        edgecolors=LAYER_COLORS[parameter],
                        marker="D",
                        s=38,
                        zorder=5,
                    )
            capped = sorted(
                {float(row["beta_hat_requested"]) for row in scheme_rows if row["beta_capped"]}
            )
            for beta in capped:
                axis.axvline(beta, color="#666666", linestyle="--", linewidth=1.0)
                axis.text(beta, 0.16, "cap", rotation=90, va="bottom", ha="right", fontsize=8)
            axis.axhline(0.99, color="#444444", linestyle=":", linewidth=1.0)
            axis.axhline(zero_rail, color="#777777", linestyle=":", linewidth=0.8)
            axis.set_xscale("log")
            axis.set_xlim(7e-11, 5e-4)
            axis.set_ylim(-0.115, 1.035)
            axis.set_yticks((zero_rail, 0.0, 0.5, 0.9, 0.99, 1.0))
            axis.set_yticklabels(("ZERO", "0", "0.5", "0.9", "0.99", "1"))
            axis.grid(True, which="both", alpha=0.16)
            axis.set_title(f"{scheme} — {dtype}")
            if column_index == 0:
                axis.set_ylabel("EqProp–BPTT cosine")
            if row_index == len(SCHEMES) - 1:
                axis.set_xlabel(r"requested common pre-amplification $\hat{\beta}$")
    axes[0, 0].legend(ncol=4, loc="lower right", fontsize=9)
    figure.suptitle(
        "Conv3 best-checkpoint one-sided frozen-current EqProp vs BPTT\n"
        "true float32 and float64; weights only; native T/K; fixed 16-example batch",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.955))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    input_csvs: Sequence[Path],
    output_csv: Path,
    output_plot: Path,
    output_markdown: Path,
    path: Path,
) -> None:
    all_layer_windows: dict[str, dict[str, list[float]]] = {}
    for dtype in DTYPES:
        all_layer_windows[dtype] = {}
        for scheme in SCHEMES:
            passing: list[float] = []
            for beta in EXPECTED_BETAS[scheme]:
                cells = [
                    row
                    for row in rows
                    if row["scheme"] == scheme
                    and math.isclose(
                        float(row["beta_hat_requested"]), beta, rel_tol=1e-12
                    )
                ]
                if len(cells) == len(PARAMETERS) and all(
                    row[f"{dtype}_status"] == "numeric"
                    and float(row[f"{dtype}_cosine"]) >= 0.99
                    for row in cells
                ):
                    passing.append(beta)
            all_layer_windows[dtype][scheme] = passing
    summary = {
        "schema": "conv3-eqprop-true-dtype-beta-curve/v1",
        "architecture": "conv3",
        "checkpoint_role": "best_validation",
        "cohort": {"batch_index": 0, "example_count": 16},
        "bias_gradients_excluded": True,
        "optimizer_steps_applied": False,
        "official_test_read": False,
        "common_uncapped_beta_hat_values": [1e-10, 3e-10, 1e-9, 3e-9, 1e-8, 3e-8],
        "unique_beta_counts": {scheme: len(EXPECTED_BETAS[scheme]) for scheme in SCHEMES},
        "all_layer_cosine_at_least_0_99_beta_hat_values": all_layer_windows,
        "float32_c0_exact_zero_at_every_tested_point": {
            scheme: all(
                row["float32_status"] == "exact_zero"
                for row in rows
                if row["scheme"] == scheme and row["parameter_name"] == "ConvWeight_0"
            )
            for scheme in SCHEMES
        },
        "exact_zero_cells": sum(
            row[f"{dtype}_status"] == "exact_zero" for row in rows for dtype in DTYPES
        ),
        "undefined_nonzero_cells": sum(
            row[f"{dtype}_status"] == "undefined" for row in rows for dtype in DTYPES
        ),
        "input_csvs": [
            {
                "path": str(input_path.expanduser().resolve()),
                "sha256": _sha256(input_path.expanduser().resolve()),
            }
            for input_path in input_csvs
        ],
        "outputs": {
            "csv": {"path": str(output_csv.resolve()), "sha256": _sha256(output_csv)},
            "plot": {"path": str(output_plot.resolve()), "sha256": _sha256(output_plot)},
            "markdown": {
                "path": str(output_markdown.resolve()),
                "sha256": _sha256(output_markdown),
            },
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, action="append", required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-plot", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rows = normalized_rows(load_curve_rows(args.input_csv))
    write_csv(rows, args.output_csv)
    write_markdown(rows, args.output_markdown)
    render_plot(rows, args.output_plot)
    write_summary(
        rows,
        input_csvs=args.input_csv,
        output_csv=args.output_csv,
        output_plot=args.output_plot,
        output_markdown=args.output_markdown,
        path=args.output_summary,
    )
    print(
        json.dumps(
            {
                "row_count": len(rows),
                "output_csv": str(args.output_csv),
                "output_plot": str(args.output_plot),
                "output_markdown": str(args.output_markdown),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
