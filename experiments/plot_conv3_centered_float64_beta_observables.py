#!/usr/bin/env python3
"""Plot centered-float64 Conv3 gradient and state observables versus beta.

The input is the validated ``summary.csv`` produced by
``analyze_conv3_eqprop_complete_beta_sweep``.  This script fail-closes on the
expected 3 schemes x 2 checkpoints x 4 layers x 11 injected-beta grid, writes
a compact filtered CSV, and renders one shared-scale 3 x 3 matplotlib figure.

Displacement curves show both nudged endpoints matched to the centered run's
zero phase:

* relative displacement is the runner's state-relative RMS displacement;
* absolute displacement is the per-node ``RMS(state(+-beta) - state(0))``, not
  a total L2 norm.
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
from matplotlib.lines import Line2D


SCHEMES = ("baseline", "ours", "legacy")
CHECKPOINT_ROLES = ("reconstructed_initialization", "best_validation")
PARAMETERS = ("ConvWeight_0", "ConvWeight_1", "ConvWeight_2", "DenseWeight_0")
EXPECTED_BETAS = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
EXPECTED_ROWS = len(SCHEMES) * len(CHECKPOINT_ROLES) * len(PARAMETERS) * len(EXPECTED_BETAS)

GRADIENT_LABELS = {
    "ConvWeight_0": "C0",
    "ConvWeight_1": "C1",
    "ConvWeight_2": "C2",
    "DenseWeight_0": "Dense",
}
STATE_LABELS = {
    "ConvWeight_0": "H0",
    "ConvWeight_1": "H1",
    "ConvWeight_2": "H2",
    "DenseWeight_0": "output",
}
EXPECTED_STATE_NAMES = {
    "ConvWeight_0": "Layer_1",
    "ConvWeight_1": "Layer_2",
    "ConvWeight_2": "Layer_3",
    "DenseWeight_0": "Layer_4",
}
ROLE_LABELS = {
    "reconstructed_initialization": "initialization",
    "best_validation": "best checkpoint",
}
ROLE_STYLES = {
    "reconstructed_initialization": {"marker": "o", "filled": False},
    "best_validation": {"marker": "s", "filled": True},
}
ENDPOINT_STYLES = {"positive": "-", "negative": "--"}
LAYER_COLORS = {
    "ConvWeight_0": "#0072B2",
    "ConvWeight_1": "#E69F00",
    "ConvWeight_2": "#009E73",
    "DenseWeight_0": "#CC79A7",
}

REQUIRED_COLUMNS = {
    "schema",
    "eqprop_variant",
    "scheme",
    "checkpoint_role",
    "precision",
    "T",
    "K",
    "batch_index",
    "batch_payload_sha256",
    "batch_source_indices_sha256",
    "beta_index",
    "actual_base_beta",
    "injected_beta",
    "amplification_factor",
    "parameter_name",
    "gradient_layer",
    "state_layer_name",
    "state_layer",
    "bias_excluded",
    "cosine_defined",
    "cosine_undefined_reason",
    "eqprop_vs_bptt_cosine",
    "matched_zero_positive_relative_displacement",
    "matched_zero_positive_delta_rms",
    "matched_zero_negative_relative_displacement",
    "matched_zero_negative_delta_rms",
}

OUTPUT_FIELDS = (
    "scheme",
    "checkpoint_role",
    "checkpoint_label",
    "beta_index",
    "injected_beta",
    "actual_base_beta",
    "amplification_factor",
    "parameter_name",
    "gradient_layer",
    "state_layer_name",
    "state_layer",
    "eqprop_vs_bptt_cosine",
    "matched_zero_positive_relative_displacement",
    "matched_zero_positive_delta_rms",
    "matched_zero_negative_relative_displacement",
    "matched_zero_negative_delta_rms",
    "T",
    "K",
    "batch_index",
    "batch_payload_sha256",
    "batch_source_indices_sha256",
    "bias_excluded",
    "source_schema",
)

DEFAULT_STUDY = Path(
    "results/perfectdiode-conv3-eqprop-beta-1em3-to-1e2-init-best-true-dtype-seed0-20260811-v1"
)


def _parse_bool(value: object, *, label: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"{label} must be boolean, got {value!r}.")


def _finite_float(value: object, *, label: str) -> float:
    try:
        result = float(str(value).strip())
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric, got {value!r}.") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite, got {value!r}.")
    return result


def _same_beta(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1.0e-12, abs_tol=0.0)


def load_centered_float64_rows(path: Path) -> list[dict[str, Any]]:
    """Load, filter, normalize, and validate the complete plotting surface."""

    source = path.expanduser().resolve()
    with source.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{source} is missing required columns: {sorted(missing)}.")
        raw_rows = [
            row
            for row in reader
            if row["eqprop_variant"] == "centered" and row["precision"] == "float64"
        ]

    if len(raw_rows) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} centered-float64 rows, got {len(raw_rows)}.")

    rows: list[dict[str, Any]] = []
    observed: set[tuple[str, str, str, int]] = set()
    for raw in raw_rows:
        scheme = raw["scheme"]
        role = raw["checkpoint_role"]
        parameter = raw["parameter_name"]
        if scheme not in SCHEMES or role not in CHECKPOINT_ROLES or parameter not in PARAMETERS:
            raise ValueError(f"Unexpected plotting coordinate: {scheme}/{role}/{parameter}.")
        beta_index = int(raw["beta_index"])
        if not 0 <= beta_index < len(EXPECTED_BETAS):
            raise ValueError(f"Unexpected beta index {beta_index}.")
        injected_beta = _finite_float(raw["injected_beta"], label="injected beta")
        if not _same_beta(injected_beta, EXPECTED_BETAS[beta_index]):
            raise ValueError(
                f"Injected beta/index mismatch at {scheme}/{role}/{parameter}: "
                f"index={beta_index}, beta={injected_beta}."
            )
        coordinate = (scheme, role, parameter, beta_index)
        if coordinate in observed:
            raise ValueError(f"Duplicate plotting coordinate: {coordinate}.")
        observed.add(coordinate)

        if int(raw["T"]) != 64 or int(raw["K"]) != 64 or int(raw["batch_index"]) != 0:
            raise ValueError(f"Replay contract differs at {coordinate}.")
        if not _parse_bool(raw["bias_excluded"], label="bias_excluded"):
            raise ValueError(f"Bias gradients are present at {coordinate}.")
        if not _parse_bool(raw["cosine_defined"], label="cosine_defined"):
            raise ValueError(
                f"Centered float64 cosine is undefined at {coordinate}: "
                f"{raw['cosine_undefined_reason']!r}."
            )
        if raw["gradient_layer"] != GRADIENT_LABELS[parameter]:
            raise ValueError(f"Gradient-layer mapping differs at {coordinate}.")
        if raw["state_layer"] != STATE_LABELS[parameter]:
            raise ValueError(f"State-layer mapping differs at {coordinate}.")
        if raw["state_layer_name"] != EXPECTED_STATE_NAMES[parameter]:
            raise ValueError(f"State-name mapping differs at {coordinate}.")

        cosine = _finite_float(raw["eqprop_vs_bptt_cosine"], label="cosine")
        if not -1.000001 <= cosine <= 1.000001:
            raise ValueError(f"Cosine outside numerical [-1,1] tolerance at {coordinate}: {cosine}.")
        positive_relative = _finite_float(
            raw["matched_zero_positive_relative_displacement"],
            label="positive relative displacement",
        )
        positive_absolute = _finite_float(
            raw["matched_zero_positive_delta_rms"], label="positive delta RMS"
        )
        negative_relative = _finite_float(
            raw["matched_zero_negative_relative_displacement"],
            label="negative relative displacement",
        )
        negative_absolute = _finite_float(
            raw["matched_zero_negative_delta_rms"], label="negative delta RMS"
        )
        if min(positive_relative, positive_absolute, negative_relative, negative_absolute) <= 0.0:
            raise ValueError(f"Log-scale displacement is non-positive at {coordinate}.")

        rows.append(
            {
                "scheme": scheme,
                "checkpoint_role": role,
                "checkpoint_label": ROLE_LABELS[role],
                "beta_index": beta_index,
                "injected_beta": injected_beta,
                "actual_base_beta": _finite_float(raw["actual_base_beta"], label="actual base beta"),
                "amplification_factor": _finite_float(
                    raw["amplification_factor"], label="amplification factor"
                ),
                "parameter_name": parameter,
                "gradient_layer": raw["gradient_layer"],
                "state_layer_name": raw["state_layer_name"],
                "state_layer": raw["state_layer"],
                "eqprop_vs_bptt_cosine": cosine,
                "matched_zero_positive_relative_displacement": positive_relative,
                "matched_zero_positive_delta_rms": positive_absolute,
                "matched_zero_negative_relative_displacement": negative_relative,
                "matched_zero_negative_delta_rms": negative_absolute,
                "T": 64,
                "K": 64,
                "batch_index": 0,
                "batch_payload_sha256": raw["batch_payload_sha256"],
                "batch_source_indices_sha256": raw["batch_source_indices_sha256"],
                "bias_excluded": True,
                "source_schema": raw["schema"],
            }
        )

    expected = {
        (scheme, role, parameter, beta_index)
        for scheme in SCHEMES
        for role in CHECKPOINT_ROLES
        for parameter in PARAMETERS
        for beta_index in range(len(EXPECTED_BETAS))
    }
    if observed != expected:
        raise ValueError("Centered-float64 plotting coverage is incomplete.")
    if len({row["batch_payload_sha256"] for row in rows}) != 1:
        raise ValueError("Plotting rows do not share one fixed batch payload.")
    if len({row["batch_source_indices_sha256"] for row in rows}) != 1:
        raise ValueError("Plotting rows do not share one fixed source-index cohort.")

    for scheme in SCHEMES:
        factors = {row["amplification_factor"] for row in rows if row["scheme"] == scheme}
        expected_factor = {"baseline": 1.0, "ours": 64.0, "legacy": 4096.0}[scheme]
        if factors != {expected_factor}:
            raise ValueError(f"Unexpected amplification factor for {scheme}: {factors}.")
        for row in (item for item in rows if item["scheme"] == scheme):
            expected_base = row["injected_beta"] / expected_factor
            if not math.isclose(row["actual_base_beta"], expected_base, rel_tol=2.0e-7, abs_tol=0.0):
                raise ValueError(f"Base-beta scaling differs for {scheme}: {row}.")

    return sorted(
        rows,
        key=lambda row: (
            SCHEMES.index(str(row["scheme"])),
            CHECKPOINT_ROLES.index(str(row["checkpoint_role"])),
            PARAMETERS.index(str(row["parameter_name"])),
            int(row["beta_index"]),
        ),
    )


def write_filtered_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows({field: row[field] for field in OUTPUT_FIELDS} for row in rows)


def _curve_rows(
    rows: Sequence[Mapping[str, Any]], *, scheme: str, role: str, parameter: str
) -> list[Mapping[str, Any]]:
    selected = [
        row
        for row in rows
        if row["scheme"] == scheme
        and row["checkpoint_role"] == role
        and row["parameter_name"] == parameter
    ]
    return sorted(selected, key=lambda row: int(row["beta_index"]))


def render_plot(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    """Render one 3-scheme x 3-observable shared-scale figure."""

    specifications = (
        ("eqprop_vs_bptt_cosine", "EqProp–BPTT gradient cosine", False),
        (
            "matched_zero_positive_relative_displacement",
            r"Relative state displacement  $\|V_{\pm}-V_0\|_{\rm RMS}/\|V_0\|_{\rm RMS}$",
            True,
        ),
        (
            "matched_zero_positive_delta_rms",
            r"Absolute per-node RMS displacement  $\|V_{\pm}-V_0\|_{\rm RMS}$",
            True,
        ),
    )
    figure, axes = plt.subplots(3, 3, figsize=(16.4, 12.4), sharex="col")
    displacement_limits: dict[str, tuple[float, float]] = {}
    for field, _, logarithmic in specifications:
        if logarithmic:
            fields = (field,)
            if field == "matched_zero_positive_relative_displacement":
                fields = (
                    "matched_zero_positive_relative_displacement",
                    "matched_zero_negative_relative_displacement",
                )
            elif field == "matched_zero_positive_delta_rms":
                fields = (
                    "matched_zero_positive_delta_rms",
                    "matched_zero_negative_delta_rms",
                )
            values = [float(row[item]) for row in rows for item in fields]
            low = 10.0 ** math.floor(math.log10(min(values)))
            high = 10.0 ** math.ceil(math.log10(max(values)))
            displacement_limits[field] = (low, high)

    for row_index, scheme in enumerate(SCHEMES):
        for column_index, (field, title, logarithmic) in enumerate(specifications):
            axis = axes[row_index, column_index]
            for role in CHECKPOINT_ROLES:
                style = ROLE_STYLES[role]
                for parameter in PARAMETERS:
                    curve = _curve_rows(rows, scheme=scheme, role=role, parameter=parameter)
                    endpoint_fields = (("centered", field),)
                    if field == "matched_zero_positive_relative_displacement":
                        endpoint_fields = (
                            ("positive", "matched_zero_positive_relative_displacement"),
                            ("negative", "matched_zero_negative_relative_displacement"),
                        )
                    elif field == "matched_zero_positive_delta_rms":
                        endpoint_fields = (
                            ("positive", "matched_zero_positive_delta_rms"),
                            ("negative", "matched_zero_negative_delta_rms"),
                        )
                    for endpoint, endpoint_field in endpoint_fields:
                        y = [float(item[endpoint_field]) for item in curve]
                        if field == "eqprop_vs_bptt_cosine":
                            y = [min(1.0, max(-1.0, value)) for value in y]
                        axis.plot(
                            [float(item["injected_beta"]) for item in curve],
                            y,
                            color=LAYER_COLORS[parameter],
                            linestyle=(
                                "-" if endpoint == "centered" else ENDPOINT_STYLES[endpoint]
                            ),
                            marker=str(style["marker"]),
                            markerfacecolor=(
                                LAYER_COLORS[parameter]
                                if bool(style["filled"])
                                else "none"
                            ),
                            markersize=3.8,
                            linewidth=1.4,
                            alpha=0.88,
                        )
            axis.set_xscale("log")
            axis.grid(True, which="major", alpha=0.23, linewidth=0.75)
            axis.grid(True, which="minor", alpha=0.09, linewidth=0.55)
            if logarithmic:
                axis.set_yscale("log")
                axis.set_ylim(*displacement_limits[field])
            else:
                axis.set_ylim(-0.05, 1.02)
                axis.axhline(0.99, color="#555555", linestyle=":", linewidth=1.0)
                axis.set_yticks((-0.0, 0.5, 0.9, 1.0))
            if field == "matched_zero_positive_delta_rms":
                axis.axhline(1.0e-6, color="#555555", linestyle=":", linewidth=1.0)
            if row_index == 0:
                axis.set_title(title, fontsize=11.5)
            if row_index == len(SCHEMES) - 1:
                axis.set_xlabel(r"Injected output-current nudge $B$")
            if column_index == 0:
                axis.text(
                    -0.20,
                    0.5,
                    scheme,
                    transform=axis.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                )

    axes[0, 0].text(
        0.02,
        0.04,
        "dotted: cosine = 0.99",
        transform=axes[0, 0].transAxes,
        fontsize=8,
        color="#444444",
    )
    axes[0, 2].text(
        0.02,
        0.04,
        r"dotted: $\Delta V_{\rm RMS}=10^{-6}$",
        transform=axes[0, 2].transAxes,
        fontsize=8,
        color="#444444",
    )

    layer_handles = [
        Line2D([0], [0], color=LAYER_COLORS[parameter], linewidth=2.0, label=GRADIENT_LABELS[parameter])
        for parameter in PARAMETERS
    ]
    role_handles = [
        Line2D(
            [0],
            [0],
            color="#333333",
            linestyle="-",
            marker=str(ROLE_STYLES[role]["marker"]),
            markerfacecolor="#333333" if bool(ROLE_STYLES[role]["filled"]) else "none",
            linewidth=1.6,
            label=ROLE_LABELS[role],
        )
        for role in CHECKPOINT_ROLES
    ]
    endpoint_handles = [
        Line2D(
            [0],
            [0],
            color="#333333",
            linestyle=ENDPOINT_STYLES[endpoint],
            linewidth=1.6,
            label=f"{endpoint} endpoint",
        )
        for endpoint in ("positive", "negative")
    ]
    figure.legend(
        handles=[*layer_handles, *role_handles, *endpoint_handles],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        ncol=8,
        frameon=False,
    )
    figure.suptitle(
        "Conv3 centered EqProp: gradient fidelity and matched-zero state motion\n"
        "native float64, T=K=64, fixed 16-example cohort, weights only",
        fontsize=14,
    )
    figure.text(
        0.5,
        0.006,
        r"Both signed endpoints are matched to $V_0$.  Base $\beta$: baseline $=B$, ours $=B/64$, legacy $=B/4096$.  Displacement columns share scales across schemes.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0.045, 0.075, 1.0, 0.94), h_pad=1.05, w_pad=1.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metric_range(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, float]:
    values = [float(row[field]) for row in rows]
    return {"minimum": min(values), "maximum": max(values)}


def _first_beta_at_or_above(
    curve: Sequence[Mapping[str, Any]], field: str, threshold: float
) -> float | None:
    matches = [float(row["injected_beta"]) for row in curve if float(row[field]) >= threshold]
    return min(matches) if matches else None


def write_validation_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    input_csv: Path,
    output_csv: Path,
    output_plot: Path,
    path: Path,
) -> None:
    curves: list[dict[str, Any]] = []
    for scheme in SCHEMES:
        for role in CHECKPOINT_ROLES:
            for parameter in PARAMETERS:
                curve = _curve_rows(rows, scheme=scheme, role=role, parameter=parameter)
                cosines = [float(row["eqprop_vs_bptt_cosine"]) for row in curve]
                positive = [float(row["matched_zero_positive_delta_rms"]) for row in curve]
                negative = [float(row["matched_zero_negative_delta_rms"]) for row in curve]
                curves.append(
                    {
                        "scheme": scheme,
                        "checkpoint_role": role,
                        "gradient_layer": GRADIENT_LABELS[parameter],
                        "state_layer": STATE_LABELS[parameter],
                        "beta_count": len(curve),
                        "minimum_cosine": min(cosines),
                        "maximum_cosine": max(cosines),
                        "injected_betas_with_cosine_at_least_0_99": [
                            float(row["injected_beta"])
                            for row in curve
                            if float(row["eqprop_vs_bptt_cosine"]) >= 0.99
                        ],
                        "first_injected_beta_with_positive_delta_rms_at_least_1e_6": (
                            _first_beta_at_or_above(
                                curve, "matched_zero_positive_delta_rms", 1.0e-6
                            )
                        ),
                        "maximum_positive_to_negative_delta_rms_ratio": max(
                            max(left / right, right / left)
                            for left, right in zip(positive, negative, strict=True)
                        ),
                    }
                )

    source = input_csv.expanduser().resolve()
    output_csv = output_csv.resolve()
    output_plot = output_plot.resolve()
    payload = {
        "schema": "conv3-centered-float64-beta-observables/v1",
        "evidence_class": "ordinary_mnist_read_only_checkpoint_diagnostic",
        "filters": {
            "architecture": "conv3",
            "eqprop_variant": "centered",
            "precision": "float64",
            "T": 64,
            "K": 64,
            "batch_index": 0,
            "bias_gradients_excluded": True,
            "plotted_endpoints": [
                "positive_nudged_state_matched_to_zero_phase",
                "negative_nudged_state_matched_to_zero_phase",
            ],
        },
        "definitions": {
            "relative_displacement": "runner state-relative RMS displacement between each signed endpoint and its matched zero phase",
            "absolute_displacement": "per-node RMS(state(+-beta) - state(0)); not a total L2 norm",
        },
        "coverage": {
            "complete": True,
            "row_count": len(rows),
            "expected_row_count": EXPECTED_ROWS,
            "schemes": list(SCHEMES),
            "checkpoint_roles": list(CHECKPOINT_ROLES),
            "gradient_layers": [GRADIENT_LABELS[item] for item in PARAMETERS],
            "state_layers": [STATE_LABELS[item] for item in PARAMETERS],
            "injected_betas": list(EXPECTED_BETAS),
            "curve_count": len(curves),
            "points_per_curve": len(EXPECTED_BETAS),
        },
        "metric_ranges": {
            "eqprop_vs_bptt_cosine": _metric_range(rows, "eqprop_vs_bptt_cosine"),
            "positive_relative_displacement": _metric_range(
                rows, "matched_zero_positive_relative_displacement"
            ),
            "positive_absolute_displacement_delta_rms": _metric_range(
                rows, "matched_zero_positive_delta_rms"
            ),
            "negative_relative_displacement": _metric_range(
                rows, "matched_zero_negative_relative_displacement"
            ),
            "negative_absolute_displacement_delta_rms": _metric_range(
                rows, "matched_zero_negative_delta_rms"
            ),
        },
        "curve_checks": curves,
        "input": {"path": str(source), "sha256": _sha256(source)},
        "outputs": {
            "filtered_csv": {"path": str(output_csv), "sha256": _sha256(output_csv)},
            "plot_png": {"path": str(output_plot), "sha256": _sha256(output_plot)},
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_STUDY / "analysis" / "summary.csv",
        help="Validated complete-sweep summary.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_STUDY / "analysis",
        help="Existing study analysis directory.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir.expanduser().resolve()
    output_csv = output_dir / "centered_float64_beta_observables.csv"
    output_plot = output_dir / "plots" / "centered_float64_cosine_relative_absolute_displacement_vs_beta.png"
    output_summary = output_dir / "centered_float64_beta_observables.json"

    rows = load_centered_float64_rows(args.input_csv)
    write_filtered_csv(rows, output_csv)
    render_plot(rows, output_plot)
    write_validation_summary(
        rows,
        input_csv=args.input_csv,
        output_csv=output_csv,
        output_plot=output_plot,
        path=output_summary,
    )
    print(
        json.dumps(
            {
                "row_count": len(rows),
                "filtered_csv": str(output_csv),
                "plot_png": str(output_plot),
                "validation_summary": str(output_summary),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
