#!/usr/bin/env python3
"""Plot true-dtype EqProp--BPTT cosine evidence from shadow audits.

The figure compares the float32 and float64 learning-rule directions for the
same fixed batch, checkpoint, beta-hat, and runtime.  Blank cosine fields are
not coerced to zero: an exactly-zero EqProp gradient is marked ``ZERO``, while
any other undefined cosine is marked ``NaN``.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
from matplotlib import colors
from matplotlib import pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TOP_STUDY = (
    REPOSITORY_ROOT
    / "results/perfectdiode-conv123-fc123-positive-eqprop-small-beta-seed0-20260810-v1"
)
DEFAULT_BETA_1E5_CSV = (
    TOP_STUDY
    / "conv-float64-shadow-biasdepth-bptt-betahat1em5"
    / "parameter_precision_comparison.csv"
)
DEFAULT_BETA_1E3_CSV = (
    TOP_STUDY
    / "conv-float64-shadow-biasdepth-bptt-betahat1em3"
    / "parameter_precision_comparison.csv"
)
DEFAULT_OUTPUT = TOP_STUDY / "analysis/eqprop_true_dtype_bptt_cosine.png"

ARCHITECTURES = ("conv2", "conv3")
SCHEMES = ("baseline", "ours", "legacy")
DTYPES = ("float32", "float64")
PARAMETER_ORDER = {
    "conv2": ("ConvWeight_0", "ConvWeight_1", "DenseWeight_0"),
    "conv3": (
        "ConvWeight_0",
        "ConvWeight_1",
        "ConvWeight_2",
        "DenseWeight_0",
    ),
}
LAYER_LABELS = {
    "conv2": {
        "ConvWeight_0": "input → conv 1",
        "ConvWeight_1": "conv 1 → conv 2",
        "DenseWeight_0": "conv 2 → output",
    },
    "conv3": {
        "ConvWeight_0": "input → conv 1",
        "ConvWeight_1": "conv 1 → conv 2",
        "ConvWeight_2": "conv 2 → conv 3",
        "DenseWeight_0": "conv 3 → output",
    },
}
REQUIRED_COLUMNS = {
    "architecture",
    "scheme",
    "checkpoint_role",
    "parameter_name",
    "beta_hat_requested",
    "residual_limited",
    *(f"{dtype}_eqprop_l2" for dtype in DTYPES),
    *(f"{dtype}_exact_zero_fraction" for dtype in DTYPES),
    *(f"{dtype}_eqprop_vs_bptt_cosine" for dtype in DTYPES),
    *(f"{dtype}_eqprop_over_bptt_norm_ratio" for dtype in DTYPES),
}


@dataclass(frozen=True)
class CellValue:
    cosine: float | None
    norm_ratio: float | None
    exact_zero_fraction: float
    status: str


@dataclass(frozen=True)
class EvidenceTable:
    beta_hat: float
    rows: Mapping[tuple[str, str, str], Mapping[str, str]]


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
        return None
    return result


def _expected_keys() -> set[tuple[str, str, str]]:
    return {
        (architecture, scheme, parameter_name)
        for architecture in ARCHITECTURES
        for scheme in SCHEMES
        for parameter_name in PARAMETER_ORDER[architecture]
    }


def load_evidence_csv(path: Path) -> EvidenceTable:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing_columns = REQUIRED_COLUMNS.difference(reader.fieldnames or ())
        if missing_columns:
            raise ValueError(
                f"{path} is missing required columns: {sorted(missing_columns)}"
            )
        raw_rows = [dict(row) for row in reader]

    rows: dict[tuple[str, str, str], Mapping[str, str]] = {}
    beta_values: set[float] = set()
    for row in raw_rows:
        if row["checkpoint_role"] != "best_validation":
            raise ValueError(
                f"Expected only best_validation rows in {path}, got "
                f"{row['checkpoint_role']!r}."
            )
        key = (row["architecture"], row["scheme"], row["parameter_name"])
        if key in rows:
            raise ValueError(f"Duplicate evidence row for {key} in {path}.")
        rows[key] = row
        beta_values.add(float(row["beta_hat_requested"]))

    expected = _expected_keys()
    observed = set(rows)
    if observed != expected:
        missing = sorted(expected.difference(observed))
        unexpected = sorted(observed.difference(expected))
        raise ValueError(
            f"Unexpected case/layer coverage in {path}; missing={missing}, "
            f"unexpected={unexpected}."
        )
    if len(beta_values) != 1:
        raise ValueError(f"Expected one beta-hat value in {path}, got {beta_values}.")
    return EvidenceTable(beta_hat=next(iter(beta_values)), rows=rows)


def cell_value(row: Mapping[str, str], dtype: str) -> CellValue:
    if dtype not in DTYPES:
        raise ValueError(f"Unknown dtype {dtype!r}.")
    cosine = _optional_float(row[f"{dtype}_eqprop_vs_bptt_cosine"])
    norm_ratio = _optional_float(row[f"{dtype}_eqprop_over_bptt_norm_ratio"])
    eqprop_l2 = float(row[f"{dtype}_eqprop_l2"])
    exact_zero_fraction = float(row[f"{dtype}_exact_zero_fraction"])
    if cosine is not None:
        if not -1.000001 <= cosine <= 1.000001:
            raise ValueError(f"Cosine outside [-1, 1]: {cosine}.")
        status = "numeric"
    elif eqprop_l2 == 0.0 and math.isclose(
        exact_zero_fraction, 1.0, rel_tol=0.0, abs_tol=1.0e-12
    ):
        status = "exact_zero"
    else:
        status = "undefined"
    return CellValue(
        cosine=cosine,
        norm_ratio=norm_ratio,
        exact_zero_fraction=exact_zero_fraction,
        status=status,
    )


def _beta_label(value: float) -> str:
    exponent = round(math.log10(value))
    if math.isclose(value, 10.0**exponent, rel_tol=1.0e-10):
        return rf"$\hat{{\beta}}=10^{{{exponent}}}$"
    return rf"$\hat{{\beta}}={value:.2g}$"


def _annotation_color(cmap, norm, value: float) -> str:
    red, green, blue, _ = cmap(norm(value))
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "black" if luminance > 0.52 else "white"


def _format_cosine(value: float) -> str:
    if value != 0.0 and abs(value) < 1.0e-3:
        return f"{value:.1e}"
    return f"{value:.3f}"


def _case_is_residual_limited(
    tables: Sequence[EvidenceTable], architecture: str, scheme: str
) -> bool:
    flags = {
        _parse_bool(
            table.rows[(architecture, scheme, parameter_name)]["residual_limited"]
        )
        for table in tables
        for parameter_name in PARAMETER_ORDER[architecture]
    }
    if len(flags) != 1:
        raise ValueError(
            f"Inconsistent residual-limited flags for {architecture}/{scheme}: {flags}."
        )
    return next(iter(flags))


def render_true_dtype_figure(
    input_csvs: Sequence[Path], output_path: Path
) -> dict[str, int]:
    if len(input_csvs) != 2:
        raise ValueError("Exactly two beta-hat CSVs are required.")
    tables = sorted(
        (load_evidence_csv(path) for path in input_csvs),
        key=lambda table: table.beta_hat,
    )
    if math.isclose(tables[0].beta_hat, tables[1].beta_hat):
        raise ValueError("The two inputs must contain distinct beta-hat values.")

    column_specs = [
        (table, dtype) for table in tables for dtype in DTYPES
    ]
    cmap = plt.get_cmap("RdYlBu").copy()
    norm = colors.TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    figure, axes = plt.subplots(1, 2, figsize=(15.8, 9.2))
    counts = {"numeric": 0, "exact_zero": 0, "undefined": 0}
    residual_limited_cases = 0

    for axis, architecture in zip(axes, ARCHITECTURES, strict=True):
        row_specs = [
            (scheme, parameter_name)
            for scheme in SCHEMES
            for parameter_name in PARAMETER_ORDER[architecture]
        ]
        matrix = np.full((len(row_specs), len(column_specs)), np.nan, dtype=float)
        statuses: dict[tuple[int, int], CellValue] = {}
        for row_index, (scheme, parameter_name) in enumerate(row_specs):
            for column_index, (table, dtype) in enumerate(column_specs):
                cell = cell_value(
                    table.rows[(architecture, scheme, parameter_name)], dtype
                )
                statuses[(row_index, column_index)] = cell
                counts[cell.status] += 1
                if cell.cosine is not None:
                    matrix[row_index, column_index] = cell.cosine

        image = axis.imshow(
            np.ma.masked_invalid(matrix),
            aspect="auto",
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
        )
        for (row_index, column_index), cell in statuses.items():
            if cell.status == "numeric":
                assert cell.cosine is not None
                axis.text(
                    column_index,
                    row_index,
                    _format_cosine(cell.cosine),
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    fontweight="bold" if cell.cosine >= 0.99 else "normal",
                    color=_annotation_color(cmap, norm, cell.cosine),
                )
            elif cell.status == "exact_zero":
                axis.add_patch(
                    Rectangle(
                        (column_index - 0.5, row_index - 0.5),
                        1.0,
                        1.0,
                        facecolor="#4b5563",
                        edgecolor="white",
                        linewidth=0.8,
                    )
                )
                axis.text(
                    column_index,
                    row_index,
                    "ZERO",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    fontweight="bold",
                    color="white",
                )
            else:
                axis.add_patch(
                    Rectangle(
                        (column_index - 0.5, row_index - 0.5),
                        1.0,
                        1.0,
                        facecolor="#f3f4f6",
                        edgecolor="#6b7280",
                        hatch="///",
                        linewidth=0.8,
                    )
                )
                axis.text(
                    column_index,
                    row_index,
                    "NaN",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="#111827",
                )

        y_labels: list[str] = []
        layers_per_scheme = len(PARAMETER_ORDER[architecture])
        for scheme, parameter_name in row_specs:
            dagger = "†" if _case_is_residual_limited(
                tables, architecture, scheme
            ) else ""
            y_labels.append(
                f"{scheme}{dagger}  ·  {LAYER_LABELS[architecture][parameter_name]}"
            )

        for scheme_index, scheme in enumerate(SCHEMES):
            if _case_is_residual_limited(tables, architecture, scheme):
                residual_limited_cases += 1
                start = scheme_index * layers_per_scheme - 0.5
                axis.add_patch(
                    Rectangle(
                        (-0.5, start),
                        len(column_specs),
                        layers_per_scheme,
                        fill=False,
                        edgecolor="#b45309",
                        linewidth=2.2,
                        clip_on=False,
                    )
                )
            if scheme_index:
                boundary = scheme_index * layers_per_scheme - 0.5
                axis.axhline(boundary, color="#111827", linewidth=1.6)

        axis.set_title(architecture.replace("conv", "Conv "), fontsize=13, pad=12)
        axis.set_yticks(range(len(row_specs)), labels=y_labels, fontsize=9)
        axis.set_xticks(
            range(len(column_specs)),
            labels=[
                f"{_beta_label(table.beta_hat)}\n{dtype}"
                for table, dtype in column_specs
            ],
            fontsize=9,
        )
        axis.tick_params(axis="x", length=0, pad=8)
        axis.tick_params(axis="y", length=0, pad=6)
        axis.set_xticks(
            np.arange(-0.5, len(column_specs), 1.0), minor=True
        )
        axis.set_yticks(np.arange(-0.5, len(row_specs), 1.0), minor=True)
        axis.grid(which="minor", color="white", linewidth=0.8)
        axis.tick_params(which="minor", bottom=False, left=False)
        axis.axvline(1.5, color="#111827", linewidth=2.0)
        axis.set_xlabel("runtime dtype at the requested normalized nudge", labelpad=12)
        for spine in axis.spines.values():
            spine.set_visible(False)

    figure.suptitle(
        "True-dtype EqProp–BPTT direction by case and layer",
        fontsize=16,
        fontweight="bold",
        y=0.965,
    )
    figure.text(
        0.5,
        0.925,
        "Best-validation checkpoints · positive one-sided EqProp · "
        "fixed batch · biases excluded from scored gradients",
        ha="center",
        va="center",
        fontsize=10,
        color="#374151",
    )
    colorbar_axis = figure.add_axes((0.925, 0.20, 0.014, 0.62))
    colorbar = figure.colorbar(
        image,
        cax=colorbar_axis,
        orientation="vertical",
        ticks=(-1.0, -0.5, 0.0, 0.5, 1.0),
    )
    colorbar.set_label("EqProp–BPTT cosine", fontsize=10)
    legend_handles = [
        Patch(facecolor="#4b5563", edgecolor="white", label="ZERO: EqProp gradient exactly zero"),
        Patch(
            facecolor="#f3f4f6",
            edgecolor="#6b7280",
            hatch="///",
            label="NaN: undefined cosine with nonzero EqProp gradient",
        ),
        Patch(
            facecolor="none",
            edgecolor="#b45309",
            linewidth=2.0,
            label="† residual-limited: interpret direction diagnostically",
        ),
    ]
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.025),
        fontsize=9,
    )
    figure.subplots_adjust(
        left=0.18, right=0.885, top=0.875, bottom=0.16, wspace=0.58
    )

    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return {
        "row_count": len(_expected_keys()),
        "numeric_cell_count": counts["numeric"],
        "exact_zero_cell_count": counts["exact_zero"],
        "undefined_cell_count": counts["undefined"],
        "residual_limited_case_count": residual_limited_cases,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--beta-1e-5-csv", type=Path, default=DEFAULT_BETA_1E5_CSV
    )
    parser.add_argument(
        "--beta-1e-3-csv", type=Path, default=DEFAULT_BETA_1E3_CSV
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = render_true_dtype_figure(
        (args.beta_1e_5_csv, args.beta_1e_3_csv), args.output
    )
    print(
        f"Wrote {args.output}: {summary['numeric_cell_count']} numeric, "
        f"{summary['exact_zero_cell_count']} exact-zero, "
        f"{summary['undefined_cell_count']} other undefined cosine cells."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
