#!/usr/bin/env python3
"""Build paper timing assets from the current selected timing bundle."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path("/home/filip/server_code")
TIMINGS_ROOT = PROJECT_ROOT / "labs" / "figures_for_paper_digits" / "timings"
DEFAULT_BUNDLE = TIMINGS_ROOT / "selected_for_paper"
OLD_SELECTED = TIMINGS_ROOT / "old_run" / "selected_for_paper_20260429"
PLOT_SCRIPT = PROJECT_ROOT / "labs" / "tools" / "plot_spice_vs_coordinate_descent_loglog.py"

FAMILY_LABELS = {
    "single_diode_exponential": "Single Shockley",
    "double_diode_exponential": "Double Shockley",
    "experimental": "PWL i(v)",
}
FAMILY_ORDER = {
    "single_diode_exponential": 0,
    "double_diode_exponential": 1,
    "experimental": 2,
}
OLD_RUNTIME_OVERRIDES = {
    ("double_diode_exponential", 3, 256),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate current paper timing CSV/Markdown/TeX/PNG assets."
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        default=DEFAULT_BUNDLE,
        help=f"Selected timing bundle. Default: {DEFAULT_BUNDLE}",
    )
    parser.add_argument(
        "--old-selected",
        type=Path,
        default=OLD_SELECTED,
        help="Archived selected bundle used only as fallback for unchanged SPICE timings.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Write CSV/Markdown/TeX assets without regenerating PNGs.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def parse_case(case: str) -> tuple[str, int, int]:
    match = re.fullmatch(r"([^/]+)/hidden_(\d+)/hidden_(\d+)", case)
    if not match:
        raise ValueError(f"Expected case format <family>/hidden_X/hidden_Y. Provided value: {case}")
    return match.group(1), int(match.group(2)), int(match.group(3))


def case_sort_key(row: dict[str, Any]) -> tuple[int, int, int]:
    family = row["nonlinearity"]
    return FAMILY_ORDER.get(family, 99), int(row["hidden_layers"]), int(row["hidden_size"])


def as_bool(value: str | None) -> bool:
    return str(value or "").strip().lower() == "true"


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def fmt_float(value: Any, digits: int = 3) -> str:
    parsed = as_float(value)
    if parsed is None:
        return "--"
    return f"{parsed:.{digits}f}"


def fmt_speedup(value: Any) -> str:
    parsed = as_float(value)
    if parsed is None:
        return "--"
    return f"{parsed:.1f}"


def fmt_sci(value: Any) -> str:
    parsed = as_float(value)
    if parsed is None:
        return "--"
    return f"{parsed:.2e}"


def fmt_pct(value: Any) -> str:
    parsed = as_float(value)
    if parsed is None:
        return "--"
    return f"{parsed:.2f}"


def fmt_int(value: Any) -> str:
    parsed = as_float(value)
    if parsed is None:
        return "--"
    return f"{int(round(parsed)):,}"


def latex_escape(text: Any) -> str:
    raw = str(text)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in raw)


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    lines = [
        "| " + " | ".join(label for _, label in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(key, "--")) for key, _ in columns) + " |")
    return "\n".join(lines) + "\n"


def load_old_spice_rows(old_selected: Path) -> dict[tuple[str, int, int], dict[str, str]]:
    out: dict[tuple[str, int, int], dict[str, str]] = {}
    for family in FAMILY_LABELS:
        path = old_selected / "figure_inputs" / family / "combined_latest_by_hidden.csv"
        if not path.exists():
            continue
        for row in read_csv(path):
            out[(family, int(row["depth"]), int(row["width"]))] = row
    return out


def load_old_runtime_metadata(old_selected: Path, family: str, depth: int, width: int) -> dict[str, Any]:
    path = (
        old_selected
        / "runs"
        / family
        / f"hidden_{depth}"
        / f"hidden_{width}"
        / "coordinate_descent"
        / "validation_metadata.json"
    )
    if not path.exists():
        return {}
    return read_json(path)


def apply_old_runtime_override(
    row: dict[str, Any],
    old_selected: Path,
    old_spice_rows: dict[tuple[str, int, int], dict[str, str]],
) -> None:
    key = (row["nonlinearity"], row["hidden_layers"], row["hidden_size"])
    if key not in OLD_RUNTIME_OVERRIDES:
        return

    old_row = old_spice_rows.get(key)
    if old_row is None:
        return

    old_meta = load_old_runtime_metadata(old_selected, *key)
    old_cd_time = as_float(old_row.get("coord_validation_total_seconds")) or as_float(
        old_meta.get("validation_total_seconds")
    )
    old_outer = as_float(old_row.get("coord_avg_iterations")) or as_float(
        old_meta.get("equilibrium_iterations", {}).get("avg_iterations")
    )
    old_inner = old_meta.get("newton_iteration_stats", {})

    row["coord_user_time_seconds"] = as_float(old_row.get("coord_user_time_seconds"))
    row["coord_validation_total_seconds"] = old_cd_time
    row["coord_accuracy"] = as_float(old_row.get("coord_accuracy"))
    row["coord_avg_iterations"] = old_outer
    row["coord_source_rel_path"] = old_row.get("coord_source_rel_path", "")
    row["coord_metadata_rel_path"] = old_row.get("coord_metadata_rel_path", "")
    row["spice_netlist_seconds"] = as_float(old_row.get("spice_netlist_seconds"))
    row["spice_simulation_seconds"] = as_float(old_row.get("spice_simulation_seconds"))
    row["spice_total_seconds"] = as_float(old_row.get("spice_total_seconds"))
    row["spice_timestamp"] = old_row.get("spice_timestamp", "")
    row["spice_source_rel_path"] = old_row.get("spice_source_rel_path", "")
    row["has_spice_timing"] = row["spice_total_seconds"] is not None
    row["speedup_vs_spice"] = (
        row["spice_total_seconds"] / old_cd_time
        if old_cd_time and row["spice_total_seconds"]
        else None
    )
    row["average_outer_iterations"] = old_outer
    row["average_inner_iterations_per_call"] = as_float(old_inner.get("avg_iters_per_call"))
    row["total_inner_iterations"] = as_float(old_inner.get("total_iters"))
    row["note"] = "April29 timing CD"


def load_new_spice_timings(bundle: Path) -> dict[str, dict[str, str]]:
    path = bundle / "tables" / "run_breakdown_20260520.csv"
    by_npz: dict[str, dict[str, str]] = {}
    if not path.exists():
        return by_npz
    for row in read_csv(path):
        if not row.get("spice_total_seconds"):
            continue
        spice_npz = row.get("spice_npz", "").strip()
        if spice_npz:
            by_npz[spice_npz] = row
    return by_npz


def should_use_old_spice_fallback(source: str, notes: str) -> bool:
    text = f"{source} {notes}".lower()
    if "single-process spice timing still failed" in text:
        return False
    if "replacement" in text or "accepted ep" in text:
        return False
    return True


def load_metadata(bundle: Path, rel_path: str) -> dict[str, Any]:
    if not rel_path:
        return {}
    path = bundle / rel_path
    if not path.exists():
        return {}
    return read_json(path)


def compute_mean_mae(bundle: Path, manifest_row: dict[str, str], error_payload: dict[str, Any]) -> float | None:
    cd_rel = manifest_row.get("bundle_cd_npz_files", "").strip()
    spice_rel = manifest_row.get("bundle_spice_npz", "").strip()
    if not cd_rel or not spice_rel:
        return None
    cd_path = bundle / cd_rel
    spice_path = bundle / spice_rel
    if not cd_path.exists() or not spice_path.exists():
        return None

    cd_npz = np.load(cd_path)
    spice_npz = np.load(spice_path)
    layers = error_payload.get("layers") or sorted(set(cd_npz.files) & set(spice_npz.files))
    total_nodes = 0
    weighted = 0.0
    for layer in layers:
        if layer not in cd_npz.files or layer not in spice_npz.files:
            continue
        cd_vals = np.asarray(cd_npz[layer]).reshape(np.asarray(cd_npz[layer]).shape[0], -1)
        spice_vals = np.asarray(spice_npz[layer]).reshape(np.asarray(spice_npz[layer]).shape[0], -1)
        if cd_vals.shape != spice_vals.shape:
            return None
        nodes = cd_vals.shape[1]
        total_nodes += nodes
        weighted += float(np.mean(np.abs(cd_vals - spice_vals))) * nodes
    if total_nodes == 0:
        return None
    return weighted / total_nodes


def error_metrics(bundle: Path, row: dict[str, str]) -> dict[str, float | None]:
    rel = row.get("bundle_error_json", "").strip()
    if not rel:
        return {"mean_mae": None, "p50": None, "p90": None, "p99": None}
    path = bundle / rel
    if not path.exists():
        return {"mean_mae": None, "p50": None, "p90": None, "p99": None}
    payload = read_json(path)
    pct = payload.get("node_weighted_rel_l1_percentiles", {})
    return {
        "mean_mae": compute_mean_mae(bundle, row, payload),
        "p50": as_float(pct.get("p50")),
        "p90": as_float(pct.get("p90")),
        "p99": as_float(pct.get("p99")),
    }


def build_rows(bundle: Path, old_selected: Path) -> list[dict[str, Any]]:
    manifest = read_csv(bundle / "tables" / "bundle_manifest.csv")
    old_spice = load_old_spice_rows(old_selected)
    new_spice_by_npz = load_new_spice_timings(bundle)
    rows: list[dict[str, Any]] = []

    for manifest_row in manifest:
        family, depth, width = parse_case(manifest_row["case"])
        meta = load_metadata(bundle, manifest_row.get("bundle_cd_metadata", ""))
        source = manifest_row.get("source", "")
        notes = manifest_row.get("notes", "")

        spice = {
            "spice_netlist_seconds": None,
            "spice_simulation_seconds": None,
            "spice_total_seconds": None,
            "spice_source_rel_path": "",
            "spice_timestamp": "",
        }
        original_spice_npz = manifest_row.get("original_spice_npz_path", "").strip()
        new_spice = new_spice_by_npz.get(original_spice_npz)
        if new_spice:
            spice.update(
                {
                    "spice_netlist_seconds": as_float(new_spice.get("spice_netlist_seconds")),
                    "spice_simulation_seconds": as_float(new_spice.get("spice_simulation_seconds")),
                    "spice_total_seconds": as_float(new_spice.get("spice_total_seconds")),
                    "spice_source_rel_path": new_spice.get("spice_timing_source", ""),
                    "spice_timestamp": "2026-05-18/19",
                }
            )
        elif should_use_old_spice_fallback(source, notes):
            old_row = old_spice.get((family, depth, width))
            if old_row:
                spice.update(
                    {
                        "spice_netlist_seconds": as_float(old_row.get("spice_netlist_seconds")),
                        "spice_simulation_seconds": as_float(old_row.get("spice_simulation_seconds")),
                        "spice_total_seconds": as_float(old_row.get("spice_total_seconds")),
                        "spice_source_rel_path": old_row.get("spice_source_rel_path", ""),
                        "spice_timestamp": old_row.get("spice_timestamp", ""),
                    }
                )

        cd_time = as_float(meta.get("validation_total_seconds"))
        speedup = None
        if cd_time and spice["spice_total_seconds"]:
            speedup = spice["spice_total_seconds"] / cd_time

        errors = error_metrics(bundle, manifest_row)
        row = {
                "case": manifest_row["case"],
                "nonlinearity": family,
                "nonlinearity_label": FAMILY_LABELS.get(family, family),
                "hidden_layers": depth,
                "hidden_size": width,
                "accuracy_pct": as_float(manifest_row.get("accuracy_pct")),
                "correct": manifest_row.get("correct", ""),
                "total": manifest_row.get("total", ""),
                "source": source,
                "notes": notes,
                "input_gain": as_float(manifest_row.get("input_gain")),
                "voltage_amp": as_float(manifest_row.get("voltage_amp")),
                "current_amp": as_float(manifest_row.get("current_amp")),
                "batch_size": manifest_row.get("batch_size", ""),
                "exp_clip": manifest_row.get("exp_clip", ""),
                "max_newton_iters": manifest_row.get("max_newton_iters", ""),
                "rel_tol": manifest_row.get("rel_tol", ""),
                "vn_tol": manifest_row.get("vn_tol", ""),
                "num_iterations": manifest_row.get("num_iterations", ""),
                "overrelaxation_factor": manifest_row.get("overrelaxation_factor", ""),
                "has_spice_npz": as_bool(manifest_row.get("has_spice_npz")),
                "has_error_json": as_bool(manifest_row.get("has_error_json")),
                "has_spice_timing": spice["spice_total_seconds"] is not None,
                "coord_user_time_seconds": None,
                "coord_validation_total_seconds": cd_time,
                "coord_accuracy": as_float(manifest_row.get("accuracy_pct")) / 100.0
                if as_float(manifest_row.get("accuracy_pct")) is not None
                else None,
                "coord_avg_iterations": as_float(meta.get("equilibrium_iterations", {}).get("avg_iterations")),
                "coord_metadata_rel_path": manifest_row.get("bundle_cd_metadata", ""),
                "coord_source_rel_path": manifest_row.get("bundle_cd_timing_log", ""),
                "spice_netlist_seconds": spice["spice_netlist_seconds"],
                "spice_simulation_seconds": spice["spice_simulation_seconds"],
                "spice_total_seconds": spice["spice_total_seconds"],
                "spice_timestamp": spice["spice_timestamp"],
                "spice_source_rel_path": spice["spice_source_rel_path"],
                "speedup_vs_spice": speedup,
                "average_outer_iterations": as_float(meta.get("equilibrium_iterations", {}).get("avg_iterations")),
                "average_inner_iterations_per_call": as_float(
                    meta.get("newton_iteration_stats", {}).get("avg_iters_per_call")
                ),
                "total_inner_iterations": as_float(meta.get("newton_iteration_stats", {}).get("total_iters")),
                "mean_mae_node_weighted": errors["mean_mae"],
                "p50_rel_l1_node_weighted": errors["p50"],
                "p90_rel_l1_node_weighted": errors["p90"],
                "p99_rel_l1_node_weighted": errors["p99"],
                "error_summary_rel_path": manifest_row.get("bundle_error_json", ""),
                "note": source,
            }
        apply_old_runtime_override(row, old_selected, old_spice)
        rows.append(row)

    rows.sort(key=case_sort_key)
    return rows


def write_figure_inputs(bundle: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "depth",
        "width",
        "coord_user_time_seconds",
        "coord_validation_total_seconds",
        "coord_accuracy",
        "coord_avg_iterations",
        "coord_source_rel_path",
        "coord_metadata_rel_path",
        "spice_netlist_seconds",
        "spice_simulation_seconds",
        "spice_total_seconds",
        "spice_timestamp",
        "spice_source_rel_path",
        "p50_rel_l1_node_weighted",
        "p90_rel_l1_node_weighted",
        "p99_rel_l1_node_weighted",
        "mean_mae_node_weighted",
        "error_summary_rel_path",
        "note",
    ]
    for family in FAMILY_LABELS:
        family_rows: list[dict[str, Any]] = []
        for row in rows:
            if row["nonlinearity"] != family:
                continue
            family_rows.append(
                {
                    "depth": row["hidden_layers"],
                    "width": row["hidden_size"],
                    "coord_user_time_seconds": row["coord_validation_total_seconds"],
                    "coord_validation_total_seconds": row["coord_validation_total_seconds"],
                    "coord_accuracy": row["coord_accuracy"],
                    "coord_avg_iterations": row["coord_avg_iterations"],
                    "coord_source_rel_path": row["coord_source_rel_path"],
                    "coord_metadata_rel_path": row["coord_metadata_rel_path"],
                    "spice_netlist_seconds": row["spice_netlist_seconds"],
                    "spice_simulation_seconds": row["spice_simulation_seconds"],
                    "spice_total_seconds": row["spice_total_seconds"],
                    "spice_timestamp": row["spice_timestamp"],
                    "spice_source_rel_path": row["spice_source_rel_path"],
                    "p50_rel_l1_node_weighted": row["p50_rel_l1_node_weighted"],
                    "p90_rel_l1_node_weighted": row["p90_rel_l1_node_weighted"],
                    "p99_rel_l1_node_weighted": row["p99_rel_l1_node_weighted"],
                    "mean_mae_node_weighted": row["mean_mae_node_weighted"],
                    "error_summary_rel_path": row["error_summary_rel_path"],
                    "note": row["note"],
                }
            )
        write_csv(bundle / "figure_inputs" / family / "combined_latest_by_hidden.csv", family_rows, fields)


def write_tables(bundle: Path, rows: list[dict[str, Any]]) -> None:
    tables = bundle / "tables"
    table1_fields = [
        "nonlinearity",
        "hidden_layers",
        "hidden_size",
        "mean_absolute_error",
        "p50_relative_error",
        "p90_relative_error",
        "p99_relative_error",
        "error_summary_rel_path",
    ]
    table1_rows = [
        {
            "nonlinearity": r["nonlinearity"],
            "hidden_layers": r["hidden_layers"],
            "hidden_size": r["hidden_size"],
            "mean_absolute_error": r["mean_mae_node_weighted"],
            "p50_relative_error": r["p50_rel_l1_node_weighted"],
            "p90_relative_error": r["p90_rel_l1_node_weighted"],
            "p99_relative_error": r["p99_rel_l1_node_weighted"],
            "error_summary_rel_path": r["error_summary_rel_path"],
        }
        for r in rows
    ]
    write_csv(tables / "table1_accuracy.csv", table1_rows, table1_fields)

    table2_fields = [
        "nonlinearity",
        "hidden_layers",
        "hidden_size",
        "cd_runtime_seconds",
        "spice_runtime_seconds",
        "speedup_vs_spice",
        "average_outer_iterations",
        "average_inner_iterations_per_call",
        "total_inner_iterations",
        "coord_metadata_rel_path",
        "spice_source_rel_path",
    ]
    table2_rows = [
        {
            "nonlinearity": r["nonlinearity"],
            "hidden_layers": r["hidden_layers"],
            "hidden_size": r["hidden_size"],
            "cd_runtime_seconds": r["coord_validation_total_seconds"],
            "spice_runtime_seconds": r["spice_total_seconds"],
            "speedup_vs_spice": r["speedup_vs_spice"],
            "average_outer_iterations": r["average_outer_iterations"],
            "average_inner_iterations_per_call": r["average_inner_iterations_per_call"],
            "total_inner_iterations": r["total_inner_iterations"],
            "coord_metadata_rel_path": r["coord_metadata_rel_path"],
            "spice_source_rel_path": r["spice_source_rel_path"],
        }
        for r in rows
    ]
    write_csv(tables / "table2_runtime_decomposition.csv", table2_rows, table2_fields)

    table_acc_fields = [
        "nonlinearity",
        "hidden_layers",
        "hidden_size",
        "accuracy_pct",
        "correct",
        "total",
        "input_gain",
        "voltage_amp",
        "current_amp",
    ]
    table_acc_rows = [
        {
            "nonlinearity": r["nonlinearity"],
            "hidden_layers": r["hidden_layers"],
            "hidden_size": r["hidden_size"],
            "accuracy_pct": r["accuracy_pct"],
            "correct": r["correct"],
            "total": r["total"],
            "input_gain": r["input_gain"],
            "voltage_amp": r["voltage_amp"],
            "current_amp": r["current_amp"],
        }
        for r in rows
    ]
    write_csv(tables / "table_timing_run_accuracies_input_gain.csv", table_acc_rows, table_acc_fields)

    write_markdown_tables(tables, rows)
    write_latex_tables(tables, rows)
    write_summary(tables, rows)


def write_markdown_tables(tables: Path, rows: list[dict[str, Any]]) -> None:
    table1 = []
    table2 = []
    acc = []
    for r in rows:
        table1.append(
            {
                "Nonlinearity": r["nonlinearity_label"],
                "H": r["hidden_layers"],
                "Width": r["hidden_size"],
                "MAE": fmt_sci(r["mean_mae_node_weighted"]),
                "p50 rel. err.": fmt_sci(r["p50_rel_l1_node_weighted"]),
                "p90 rel. err.": fmt_sci(r["p90_rel_l1_node_weighted"]),
                "p99 rel. err.": fmt_sci(r["p99_rel_l1_node_weighted"]),
            }
        )
        table2.append(
            {
                "Nonlinearity": r["nonlinearity_label"],
                "H": r["hidden_layers"],
                "Width": r["hidden_size"],
                "CD [s]": fmt_float(r["coord_validation_total_seconds"]),
                "SPICE [s]": fmt_float(r["spice_total_seconds"]),
                "Speedup": fmt_speedup(r["speedup_vs_spice"]),
                "Outer avg.": fmt_float(r["average_outer_iterations"]),
                "Inner avg./call": fmt_float(r["average_inner_iterations_per_call"]),
                "Total inner": fmt_int(r["total_inner_iterations"]),
            }
        )
        acc.append(
            {
                "Nonlinearity": r["nonlinearity_label"],
                "H": r["hidden_layers"],
                "Width": r["hidden_size"],
                "Accuracy": fmt_pct(r["accuracy_pct"]),
                "Correct/total": f"{r['correct']}/{r['total']}",
                "input_gain": fmt_float(r["input_gain"], 1),
                "voltage_amp": fmt_float(r["voltage_amp"], 1),
                "current_amp": fmt_float(r["current_amp"], 1),
            }
        )

    tables.joinpath("table1_accuracy.md").write_text(
        markdown_table(table1, [(k, k) for k in table1[0].keys()])
    )
    tables.joinpath("table2_runtime_decomposition.md").write_text(
        markdown_table(table2, [(k, k) for k in table2[0].keys()])
    )
    tables.joinpath("table_timing_run_accuracies_input_gain.md").write_text(
        markdown_table(acc, [(k, k) for k in acc[0].keys()])
    )


def latex_rows(rows: list[dict[str, Any]], renderer) -> str:
    lines: list[str] = []
    last_family = None
    for row in rows:
        if last_family is not None and row["nonlinearity"] != last_family:
            lines.append(r"\addlinespace")
        lines.append(renderer(row))
        last_family = row["nonlinearity"]
    return "\n".join(lines)


def write_latex_tables(tables: Path, rows: list[dict[str, Any]]) -> None:
    table1_body = latex_rows(
        rows,
        lambda r: (
            f"{latex_escape(r['nonlinearity_label'])} & {r['hidden_layers']} & {r['hidden_size']} & "
            f"{fmt_sci(r['mean_mae_node_weighted'])} & {fmt_sci(r['p50_rel_l1_node_weighted'])} & "
            f"{fmt_sci(r['p90_rel_l1_node_weighted'])} & {fmt_sci(r['p99_rel_l1_node_weighted'])}\\\\"
        ),
    )
    table1_tex = (
        "{\\small\n"
        "\\setlength{\\tabcolsep}{5pt}\n"
        "\\renewcommand{\\arraystretch}{1.05}\n"
        "\\begin{longtable}{lccrrrr}\n"
        "\\caption{Accuracy and error summary for the CPU timing runs included in the solver comparison. Errors are computed against matched SPICE steady states when available; unavailable entries are marked with dashes.}\\label{tab:timing_accuracy_summary}\\\\\n"
        "\\toprule\n"
        "Nonlinearity & H & Width & MAE & p50 rel. err. & p90 rel. err. & p99 rel. err.\\\\\n"
        "\\midrule\n"
        "\\endfirsthead\n"
        "\\multicolumn{7}{l}{\\tablename~\\thetable\\ (continued)}\\\\\n"
        "\\toprule\n"
        "Nonlinearity & H & Width & MAE & p50 rel. err. & p90 rel. err. & p99 rel. err.\\\\\n"
        "\\midrule\n"
        "\\endhead\n"
        "\\midrule\n"
        "\\multicolumn{7}{r}{Continued on next page}\\\\\n"
        "\\endfoot\n"
        "\\bottomrule\n"
        "\\endlastfoot\n"
        f"{table1_body}\n"
        "\\end{longtable}\n"
        "}\n"
    )
    tables.joinpath("table1_accuracy_longtable.tex").write_text(table1_tex)
    tables.joinpath("table1_accuracy.tex").write_text(table1_tex)

    table2_body = latex_rows(
        rows,
        lambda r: (
            f"{latex_escape(r['nonlinearity_label'])} & {r['hidden_layers']} & {r['hidden_size']} & "
            f"{fmt_float(r['coord_validation_total_seconds'])} & {fmt_float(r['spice_total_seconds'])} & "
            f"{fmt_speedup(r['speedup_vs_spice'])} & {fmt_float(r['average_outer_iterations'])} & "
            f"{fmt_float(r['average_inner_iterations_per_call'])} & {fmt_int(r['total_inner_iterations'])}\\\\"
        ),
    )
    table2_tex = (
        "\\begin{landscape}\n"
        "{\\scriptsize\n"
        "\\setlength{\\tabcolsep}{5pt}\n"
        "\\renewcommand{\\arraystretch}{1.05}\n"
        "\\begin{longtable}{lccrrrrrr}\n"
        "\\caption{Runtime decomposition for the CPU timing runs included in the solver comparison. Outer iterations denote the average number of equilibrium sweeps per sample. Inner iterations denote the average number of local nonlinear-solver iterations per call. Unavailable SPICE timings and speedups are marked with dashes.}\\label{tab:timing_runtime_decomposition}\\\\\n"
        "\\toprule\n"
        "Nonlinearity & H & Width & CD [s] & SPICE [s] & Speedup & Outer avg. & Inner avg./call & Total inner\\\\\n"
        "\\midrule\n"
        "\\endfirsthead\n"
        "\\multicolumn{9}{l}{\\tablename~\\thetable\\ (continued)}\\\\\n"
        "\\toprule\n"
        "Nonlinearity & H & Width & CD [s] & SPICE [s] & Speedup & Outer avg. & Inner avg./call & Total inner\\\\\n"
        "\\midrule\n"
        "\\endhead\n"
        "\\midrule\n"
        "\\multicolumn{9}{r}{Continued on next page}\\\\\n"
        "\\endfoot\n"
        "\\bottomrule\n"
        "\\endlastfoot\n"
        f"{table2_body}\n"
        "\\end{longtable}\n"
        "}\n"
        "\\end{landscape}\n"
    )
    tables.joinpath("table2_runtime_decomposition_longtable.tex").write_text(table2_tex)
    tables.joinpath("table2_runtime_decomposition.tex").write_text(table2_tex)

    acc_body = latex_rows(
        rows,
        lambda r: (
            f"{latex_escape(r['nonlinearity_label'])} & {r['hidden_layers']} & {r['hidden_size']} & "
            f"{fmt_pct(r['accuracy_pct'])} & {latex_escape(str(r['correct']))}/{latex_escape(str(r['total']))} & "
            f"{fmt_float(r['input_gain'], 1)} & {fmt_float(r['voltage_amp'], 1)} & "
            f"{fmt_float(r['current_amp'], 1)}\\\\"
        ),
    )
    acc_tex = (
        "\\begin{landscape}\n"
        "{\\scriptsize\n"
        "\\setlength{\\tabcolsep}{5pt}\n"
        "\\renewcommand{\\arraystretch}{1.05}\n"
        "\\begin{longtable}{lccrrrcr}\n"
        "\\caption{Validation accuracy and amplification parameters for all CPU timing-grid runs.}\\label{tab:timing_accuracy_input_gain}\\\\\n"
        "\\toprule\n"
        "Nonlinearity & H & Width & Acc. [\\%] & Correct/total & input\\_gain & V amp. & I amp.\\\\\n"
        "\\midrule\n"
        "\\endfirsthead\n"
        "\\multicolumn{8}{l}{\\tablename~\\thetable\\ (continued)}\\\\\n"
        "\\toprule\n"
        "Nonlinearity & H & Width & Acc. [\\%] & Correct/total & input\\_gain & V amp. & I amp.\\\\\n"
        "\\midrule\n"
        "\\endhead\n"
        "\\midrule\n"
        "\\multicolumn{8}{r}{Continued on next page}\\\\\n"
        "\\endfoot\n"
        "\\bottomrule\n"
        "\\endlastfoot\n"
        f"{acc_body}\n"
        "\\end{longtable}\n"
        "}\n"
        "\\end{landscape}\n"
    )
    tables.joinpath("table_timing_run_accuracies_input_gain_longtable.tex").write_text(acc_tex)
    tables.joinpath("table_timing_run_accuracies_input_gain.tex").write_text(acc_tex)


def write_summary(tables: Path, rows: list[dict[str, Any]]) -> None:
    matched_speedups = [r["speedup_vs_spice"] for r in rows if r["speedup_vs_spice"] is not None]
    p90_values = [r["p90_rel_l1_node_weighted"] for r in rows if r["p90_rel_l1_node_weighted"] is not None]
    payload = {
        "total_rows": len(rows),
        "rows_with_spice_timing": sum(1 for r in rows if r["spice_total_seconds"] is not None),
        "rows_with_error": sum(1 for r in rows if r["p90_rel_l1_node_weighted"] is not None),
        "min_speedup": min(matched_speedups) if matched_speedups else None,
        "max_speedup": max(matched_speedups) if matched_speedups else None,
        "max_p90_relative_error": max(p90_values) if p90_values else None,
    }
    tables.joinpath("paper_timing_table_summary.json").write_text(json.dumps(payload, indent=2) + "\n")


def run_plots(bundle: Path) -> None:
    for family in FAMILY_LABELS:
        input_csv = bundle / "figure_inputs" / family / "combined_latest_by_hidden.csv"
        output_png = bundle / "figure_inputs" / family / f"{family}_cpu_spice_vs_coordinate_descent_loglog.png"
        subprocess.run(
            [
                sys.executable,
                str(PLOT_SCRIPT),
                "--combined-csv",
                str(input_csv),
                "--output",
                str(output_png),
                "--paper",
                "--split-legend",
            ],
            check=True,
        )


def main() -> int:
    args = parse_args()
    rows = build_rows(args.bundle, args.old_selected)
    write_figure_inputs(args.bundle, rows)
    write_tables(args.bundle, rows)
    if not args.skip_plots:
        run_plots(args.bundle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
