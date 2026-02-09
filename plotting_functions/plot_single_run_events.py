#!/usr/bin/env python3
import os
import sys
import glob
import re
import argparse
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# New helpers for comparing gradient series across runs
def _run_label(events_file: str) -> str:
    """Derive a short label (hostname or dirname) from a tfevents path."""
    base = os.path.basename(events_file)
    parts = base.split(".")
    if len(parts) >= 5 and parts[0] == "events" and parts[1] == "out" and parts[2] == "tfevents":
        host = parts[4]
    else:
        host = None
    if host:
        return f"{host}"
    # Fallback to parent directory name
    return os.path.basename(os.path.dirname(events_file)) or base


def _normalize_series(values, mode: str = "max", scale: float = None):
    """Normalize a numeric series; supports 'max', 'globalmax', 'first', 'l2', or 'none'.

    For 'globalmax', supply `scale` with the precomputed max across all gradients in a run.
    """
    arr = np.array(values, dtype=float)
    if arr.size == 0 or mode is None or mode.lower() == "none":
        return arr
    mode = mode.lower()
    if mode == "globalmax":
        denom = scale if scale is not None else np.nanmax(np.abs(arr))
    elif mode == "max":
        denom = np.nanmax(np.abs(arr))
    elif mode == "first":
        denom = np.abs(arr[0])
    elif mode == "l2":
        denom = np.linalg.norm(arr)
    else:
        raise ValueError(f"Unknown normalize mode: {mode}")
    if denom == 0 or np.isnan(denom):
        return arr
    return arr / denom


def _extract_amp_label(events_file: str) -> str:
    """Extract amplification label (voltage/current) from surrounding folders."""
    segments = os.path.normpath(events_file).split(os.sep)
    amp_type = None
    for seg in reversed(segments):
        lower = seg.lower()
        if "voltage_amp" in lower:
            amp_type = "voltage"
            break
        if "current_amp" in lower:
            amp_type = "current"
            break

    parent = os.path.basename(os.path.dirname(events_file))
    label_val = None
    if parent:
        parts = parent.split("_")
        if parts:
            label_val = parts[-1]
    if label_val:
        return f"{amp_type}_{label_val}" if amp_type else label_val
    return _run_label(events_file)


def _ordered_grad_mapping(data: dict, kind: str = "all") -> dict:
    """Build canonical -> actual tag map, preserving layer order per type."""
    tags = [t for t in data if "gradient" in t.lower() and "train" in t.lower()]
    if kind == "dense":
        tags = [t for t in tags if "denseweight" in t.lower()]
    elif kind == "conv":
        tags = [t for t in tags if "denseweight" not in t.lower()]

    conv = sorted([t for t in tags if "convweight" in t.lower()])
    dense = sorted([t for t in tags if "denseweight" in t.lower()])
    bias = sorted([t for t in tags if "bias" in t.lower()])

    mapping = {}
    for idx, t in enumerate(conv):
        mapping[f"ConvWeight_{idx}"] = t
    for idx, t in enumerate(dense):
        mapping[f"DenseWeight_{idx}"] = t
    for idx, t in enumerate(bias):
        mapping[f"Bias_{idx}"] = t

    used = set(conv + dense + bias)
    for t in tags:
        if t in used:
            continue
        # Fallback: keep as-is
        mapping[t] = t
    return mapping


def _find_events_file(path: str) -> str:
    """Return a single events.out.tfevents* file from a path.

    - If path is already a file matching events.out.tfevents*, return it.
    - If path is a directory, search recursively for the newest matching file.
    """
    if os.path.isfile(path) and os.path.basename(path).startswith("events.out.tfevents"):
        return path

    candidates = []
    if os.path.isdir(path):
        for root, _dirs, files in os.walk(path):
            for f in files:
                if f.startswith("events.out.tfevents"):
                    full = os.path.join(root, f)
                    candidates.append((os.path.getmtime(full), full))

    if not candidates:
        raise FileNotFoundError(f"No events.out.tfevents* file found under: {path}")

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _collect_metrics(events_file: str, max_steps: int = None) -> dict:
    """Read a TensorBoard events file and return a dict tag -> {steps:[], values:[]}.

    If max_steps is provided, stop after reaching that many global steps (best-effort).
    """
    data = defaultdict(lambda: {"steps": [], "values": []})

    try:
        for event in tf.compat.v1.train.summary_iterator(events_file):
            step = getattr(event, "step", None)
            if step is None:
                continue
            for value in event.summary.value:
                tag = value.tag
                val = value.simple_value
                d = data[tag]
                d["steps"].append(step)
                d["values"].append(val)
            if max_steps is not None and step >= max_steps:
                break
    except Exception as e:
        raise RuntimeError(f"Failed reading events '{events_file}': {e}")

    # sort by step per tag
    for tag, d in data.items():
        steps = np.array(d["steps"]) if d["steps"] else np.array([])
        values = np.array(d["values"]) if d["values"] else np.array([])
        if steps.size:
            # Use stable sort so duplicate steps (e.g., interleaved WeightDist stats)
            # keep their original write order.
            order = np.argsort(steps, kind="stable")
            data[tag]["steps"] = steps[order]
            data[tag]["values"] = values[order]

    # derive accuracy if Error/* present
    for split in ("train", "test"):
        err_tag = f"Error/{split}"
        if err_tag in data and len(data[err_tag]["values"]) > 0:
            err = np.array(data[err_tag]["values"])  # could be fraction or percent
            err_frac = np.where(err > 1.0, err / 100.0, err)
            acc = 1.0 - err_frac
            data[f"Accuracy/{split}"] = {
                "steps": np.array(data[err_tag]["steps"]),
                "values": acc,
            }

    return data


WEIGHT_DIST_STATS = ["mean", "std", "min", "max", "abs_mean", "abs_std"]


def _categorize(tags: list) -> dict:
    """Group tags into categories to organize output plots."""
    all_tags = set(tags)
    cats = {
        "performance": [t for t in all_tags if any(k in t for k in ["Error/", "Top5Error/", "Accuracy/"])],
        "energy": [t for t in all_tags if t.startswith("Energy/")],
        "cost": [t for t in all_tags if t.startswith("Cost/")],
        "layer_norms": [t for t in all_tags if "Norm" in t],
        "layer_saturation": [t for t in all_tags if "Saturation" in t],
        # handle WeightDist separately; include other weight stats here
        "weight_stats": [t for t in all_tags if any(k in t for k in ["WeightRowSum", "WeightColumnSum", "Weight"]) and not t.startswith("WeightDist/")],
        "gradients": [t for t in all_tags if "Gradient" in t],
    }
    # Add leftovers
    used = set(sum(cats.values(), []))
    cats["other"] = [t for t in all_tags if t not in used]
    return cats


def _save_line_plot(out_path: str, title: str, x, y, ylabel: str, ylim=None):
    plt.figure(figsize=(9, 5))
    plt.plot(x, y, "b-", linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.grid(True, alpha=0.3)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def _split_weightdist_series(series: dict, stat_types=WEIGHT_DIST_STATS):
    """Split interleaved WeightDist values (mean/std/min/max/abs_mean/abs_std) into separate series."""
    steps = np.array(series["steps"])
    values = np.array(series["values"])
    if steps.size == 0:
        return {}
    split = {stat: {"steps": [], "values": []} for stat in stat_types}
    for idx, (s, v) in enumerate(zip(steps, values)):
        stat = stat_types[idx % len(stat_types)]
        split[stat]["steps"].append(s)
        split[stat]["values"].append(v)
    for stat in split:
        split[stat]["steps"] = np.array(split[stat]["steps"])
        split[stat]["values"] = np.array(split[stat]["values"])
    return split


def _plot_weightdist(data: dict, out_dir: str, separate_subplots: bool = False, logy: bool = False):
    """Plot mean/std/min/max/abs_mean/abs_std separately for each WeightDist tag.

    If separate_subplots is True, stats are drawn on individual subplots to reduce overplotting.
    """
    weight_tags = [t for t in data if t.startswith("WeightDist/")]
    if not weight_tags:
        return
    ws_dir = os.path.join(out_dir, "weight_stats")
    os.makedirs(ws_dir, exist_ok=True)
    for tag in weight_tags:
        series = data.get(tag, {})
        if not series or len(series.get("steps", [])) == 0:
            continue
        split = _split_weightdist_series(series)
        # Use tag minus prefix for file naming
        suffix = tag.split("/", 1)[1] if "/" in tag else tag
        safe = suffix.replace("/", "_")
        if separate_subplots:
            available = [stat for stat in WEIGHT_DIST_STATS if len(split.get(stat, {}).get("steps", [])) > 0]
            if not available:
                continue
            cols = 2
            rows = int(np.ceil(len(available) / cols))
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3.5))
            axes = np.atleast_1d(axes).ravel()
            for ax, stat in zip(axes, available):
                s = split.get(stat)
                ax.plot(s["steps"], s["values"], label=stat)
                ax.set_xlabel("Epoch")
                ax.set_ylabel(stat)
                if logy:
                    ax.set_yscale("log")
                ax.set_title(f"{tag} [{stat}]")
                ax.grid(True, alpha=0.3)
            for ax in axes[len(available):]:
                ax.axis("off")
            fig.tight_layout()
            plt.savefig(os.path.join(ws_dir, f"{safe}_grid.png"), dpi=300, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.figure(figsize=(9, 5))
            for stat in WEIGHT_DIST_STATS:
                s = split.get(stat)
                if s is None or len(s["steps"]) == 0:
                    continue
                plt.plot(s["steps"], s["values"], label=stat)
            plt.xlabel("Epoch")
            plt.ylabel("Value")
            if logy:
                plt.yscale("log")
            plt.title(f"{tag} (per-stat)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(ws_dir, f"{safe}_split.png"), dpi=300, bbox_inches="tight")
            plt.close()

def _extract_layer_num(tag: str):
    """Extract integer layer index from tags like '...Layer_3'."""
    m = re.search(r"layer[_/ ]*(\d+)", tag, flags=re.IGNORECASE)
    return int(m.group(1)) if m else None


def _extract_split(tag: str):
    """Infer split name from tag (train/test)."""
    lower = tag.lower()
    if "train" in lower:
        return "train"
    if "test" in lower:
        return "test"
    return "all"

def _excluded_saturation_layers(data: dict) -> set:
    """Default saturation exclusions: input layer (0) and the highest-index layer."""
    layer_nums = []
    for tag in data:
        if "saturation" not in tag.lower():
            continue
        layer_num = _extract_layer_num(tag)
        if layer_num is not None:
            layer_nums.append(layer_num)
    if not layer_nums:
        return set()
    excluded = {0}
    excluded.add(max(layer_nums))
    return excluded


def _plot_saturation_all_layers(data: dict, out_dir: str, saturation_ylim=None):
    """Overlay layer saturation curves in one figure per split."""
    sat_tags = [t for t in data if "saturation" in t.lower()]
    if not sat_tags:
        return

    excluded_layers = _excluded_saturation_layers(data)
    sat_dir = os.path.join(out_dir, "layer_saturation")
    os.makedirs(sat_dir, exist_ok=True)

    grouped = defaultdict(list)
    for tag in sat_tags:
        if _extract_layer_num(tag) in excluded_layers:
            continue
        series = data.get(tag, {})
        steps = np.array(series.get("steps", []))
        values = np.array(series.get("values", []))
        if steps.size == 0 or values.size == 0:
            continue
        split = _extract_split(tag)
        layer_num = _extract_layer_num(tag)
        grouped[split].append((layer_num, tag, steps, values))

    for split, entries in grouped.items():
        if not entries:
            continue
        entries.sort(key=lambda x: (x[0] is None, x[0] if x[0] is not None else x[1]))
        plt.figure(figsize=(10, 6))
        for layer_num, tag, steps, values in entries:
            label = f"Layer {layer_num}" if layer_num is not None else tag
            plt.plot(steps, values, linewidth=2, label=label)
        plt.xlabel("Epoch")
        plt.ylabel("Saturation")
        plt.title(f"Saturation ({split}) - all layers")
        if saturation_ylim is not None:
            plt.ylim(*saturation_ylim)
        plt.legend(ncol=2)
        plt.grid(True, alpha=0.3)
        out_name = f"saturation_{split}_all_layers.png"
        plt.savefig(os.path.join(sat_dir, out_name), dpi=300, bbox_inches="tight")
        plt.close()


def compare_train_gradients(
    events_a: str,
    events_b: str,
    out_dir: str = None,
    normalize: str = "globalmax",
    max_steps: int = None,
    label_a: str = None,
    label_b: str = None,
    logy: bool = False,
    kind: str = "all",
) -> None:
    """Overlay Gradient/train series for two runs (one plot per layer).

    Parameters
    ----------
    events_a : str
        Path to first run directory or events.out.tfevents* file.
    events_b : str
        Path to second run directory or events.out.tfevents* file.
    out_dir : str, optional
        Output directory for comparison plots. Defaults to <common_dir>/plots/gradient_compare.
    normalize : str, optional
        Normalization per run: 'max' (default), 'first', 'l2', or 'none'.
    max_steps : int, optional
        Truncate plotting to steps <= max_steps.
    """
    file_a = _find_events_file(events_a)
    file_b = _find_events_file(events_b)

    data_a = _collect_metrics(file_a, max_steps=max_steps)
    data_b = _collect_metrics(file_b, max_steps=max_steps)

    def _grad_filter(tag: str) -> bool:
        if "gradient" not in tag.lower() or "train" not in tag.lower():
            return False
        if kind == "dense":
            return "denseweight" in tag.lower()
        if kind == "conv":
            return "denseweight" not in tag.lower()
        return True

    grad_tags_a = {t for t in data_a if _grad_filter(t)}
    grad_tags_b = {t for t in data_b if _grad_filter(t)}
    common_tags = sorted(grad_tags_a & grad_tags_b)

    if not common_tags:
        print(f"No common Gradient/train tags between:\n  {file_a}\n  {file_b}")
        return

    if out_dir is None:
        shared_root = os.path.commonpath([os.path.dirname(file_a), os.path.dirname(file_b)])
        out_dir = os.path.join(shared_root, "plots", "gradient_compare")
    os.makedirs(out_dir, exist_ok=True)

    label_a = label_a or _run_label(file_a)
    label_b = label_b or _run_label(file_b)

    def _global_grad_scale(data: dict) -> float:
        max_val = 0.0
        for t, series in data.items():
            if not _grad_filter(t):
                continue
            vals = np.abs(np.array(series.get("values", []), dtype=float))
            if vals.size:
                max_val = max(max_val, np.nanmax(vals))
        return max_val if max_val > 0 else None

    scale_a = scale_b = None
    if normalize and normalize.lower() == "globalmax":
        scale_a = _global_grad_scale(data_a)
        scale_b = _global_grad_scale(data_b)

    for tag in common_tags:
        series_a = data_a.get(tag, {})
        series_b = data_b.get(tag, {})
        if not series_a or not series_b:
            continue
        if len(series_a.get("steps", [])) == 0 or len(series_b.get("steps", [])) == 0:
            continue

        steps_a = np.array(series_a["steps"])
        steps_b = np.array(series_b["steps"])
        values_a = _normalize_series(series_a["values"], mode=normalize, scale=scale_a)
        values_b = _normalize_series(series_b["values"], mode=normalize, scale=scale_b)

        plt.figure(figsize=(9, 5))
        plt.plot(steps_a, values_a, label=label_a, color="tab:blue")
        plt.plot(steps_b, values_b, label=label_b, color="tab:orange")
        plt.xlabel("Epoch")
        ylabel = f"{tag} (normalized: {normalize})" if normalize and normalize.lower() != "none" else tag
        plt.ylabel(ylabel)
        plt.title(f"{tag} comparison")
        plt.legend()
        if logy:
            plt.yscale("symlog", linthresh=1e-6)
        plt.grid(True, alpha=0.3)

        safe = tag.replace("/", "_")
        plt.savefig(os.path.join(out_dir, f"{safe}.png"), dpi=300, bbox_inches="tight")
        plt.close()

    print(f"Wrote gradient comparison plots to: {out_dir}")


def compare_train_gradients_multi(
    events_root: str,
    out_dir: str = None,
    normalize: str = "globalmax",
    max_steps: int = None,
    logy: bool = False,
    kind: str = "all",
) -> None:
    """Overlay Gradient/train series across all events files under a root folder.

    Labels are derived from the parent folder name (voltage amplification).
    """
    events_files = find_all_events_files(events_root)
    if not events_files:
        print(f"No events.out.tfevents* files found under: {events_root}")
        return

    events_files = sorted(events_files)
    if out_dir is None:
        out_dir = os.path.join(events_root, "plots", "gradient_compare")
    os.makedirs(out_dir, exist_ok=True)

    series_per_run = []
    labels = []
    tag_maps = []  # per run: canonical_tag -> actual_tag
    scales = []    # per run: global max across gradient tags (for globalmax)
    for f in events_files:
        data = _collect_metrics(f, max_steps=max_steps)
        labels.append(_extract_amp_label(f))
        series_per_run.append(data)
        mapping = _ordered_grad_mapping(data, kind=kind)
        tag_maps.append(mapping)
        if normalize and normalize.lower() == "globalmax":
            max_val = 0.0
            for t, s in data.items():
                if "gradient" not in t.lower() or "train" not in t.lower():
                    continue
                if kind == "dense" and "denseweight" not in t.lower():
                    continue
                vals = np.abs(np.array(s.get("values", []), dtype=float))
                if vals.size:
                    max_val = max(max_val, np.nanmax(vals))
            scales.append(max_val if max_val > 0 else None)
        else:
            scales.append(None)

    # Find common canonical gradient/train tags across all runs
    common_tags = None
    any_tags = set()
    for mapping in tag_maps:
        grad_tags = set(mapping.keys())
        any_tags |= grad_tags
        common_tags = grad_tags if common_tags is None else (common_tags & grad_tags)
    common_tags = sorted(common_tags) if common_tags else []
    if not common_tags and any_tags:
        print("No Gradient/train tag present in all runs; falling back to plotting any available gradient tags.")
        common_tags = sorted(any_tags)

    if not common_tags:
        print(f"No Gradient/train tags found under: {events_root}")
        return

    colors = plt.cm.get_cmap("tab20", len(series_per_run))

    for tag in common_tags:
        plt.figure(figsize=(9, 5))
        for idx, (data, mapping) in enumerate(zip(series_per_run, tag_maps)):
            actual_tag = mapping.get(tag)
            if not actual_tag:
                continue
            series = data.get(actual_tag, {})
            if not series or len(series.get("steps", [])) == 0:
                continue
            steps = np.array(series["steps"])
            values = _normalize_series(series["values"], mode=normalize, scale=scales[idx])
            plt.plot(steps, values, label=labels[idx], color=colors(idx))

        if not plt.gca().has_data():
            plt.close()
            continue

        plt.xlabel("Epoch")
        ylabel = f"{tag} (normalized: {normalize})" if normalize and normalize.lower() != "none" else tag
        plt.ylabel(ylabel)
        plt.title(f"{tag} comparison")
        plt.legend()
        if logy:
            plt.yscale("symlog", linthresh=1e-6)
        plt.grid(True, alpha=0.3)

        safe = tag.replace("/", "_")
        plt.savefig(os.path.join(out_dir, f"{safe}.png"), dpi=300, bbox_inches="tight")
        plt.close()

    print(f"Wrote gradient comparison plots to: {out_dir}")


def plot_single_run(
    events_path: str,
    out_dir: str,
    max_steps: int = None,
    weightdist_subplots: bool = False,
    logy_weightdist: bool = False,
    saturation_only: bool = False,
    saturation_ylim=None,
) -> None:
    """Plot all available metrics for a single run/events file.

    Parameters
    ----------
    events_path : str
        Either a direct path to an events.out.tfevents* file or a directory containing it.
    out_dir : str
        Directory where plots will be written (created if missing).
    max_steps : int, optional
        If provided, truncate plotting to steps <= max_steps.
    """
    events_file = _find_events_file(events_path)
    os.makedirs(out_dir, exist_ok=True)

    data = _collect_metrics(events_file, max_steps=max_steps)
    if not data:
        print(f"No metrics found in {events_file}")
        return

    # Always provide combined saturation overlays when saturation tags exist.
    _plot_saturation_all_layers(data, out_dir, saturation_ylim=saturation_ylim)

    if saturation_only:
        print(f"Wrote saturation-only plots to: {os.path.join(out_dir, 'layer_saturation')}")
        return

    # Plot weight distribution stats with per-stat series to avoid interleaved values
    _plot_weightdist(data, out_dir, separate_subplots=weightdist_subplots, logy=logy_weightdist)

    # Exclude WeightDist tags from generic categorization; they're handled above
    non_weightdist_tags = [t for t in data.keys() if not t.startswith("WeightDist/")]
    categories = _categorize(non_weightdist_tags)
    excluded_sat_layers = _excluded_saturation_layers(data)

    # One PNG per metric, organized under category subfolders
    for category, tags in categories.items():
        if not tags:
            continue
        cat_dir = os.path.join(out_dir, category)
        for tag in sorted(tags):
            if category == "layer_saturation" and _extract_layer_num(tag) in excluded_sat_layers:
                continue
            series = data.get(tag)
            if series is None or len(series["steps"]) == 0:
                continue
            steps = np.array(series["steps"])  # epochs
            values = np.array(series["values"])  # scalars

            # Clean name for file
            safe = tag.replace("/", "_")
            out_path = os.path.join(cat_dir, f"{safe}.png")
            ylim = saturation_ylim if category == "layer_saturation" else None
            _save_line_plot(out_path, f"{tag}", steps, values, ylabel=tag, ylim=ylim)

    # Convenience: also write a compact summary grid of key metrics if present
    summary_keys = [
        "Accuracy/train", "Accuracy/test",
        "Error/train", "Error/test",
        "Cost/train", "Cost/test",
        "Energy/train", "Energy/test",
    ]
    present = [k for k in summary_keys if k in data and len(data[k]["steps"]) > 0]
    if present:
        n = len(present)
        cols = 2
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
        axes = np.atleast_1d(axes).ravel()
        for ax, key in zip(axes, present):
            s = data[key]
            ax.plot(s["steps"], s["values"], "b-", linewidth=1.5)
            ax.set_title(key)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(key)
            ax.grid(True, alpha=0.3)
        for ax in axes[len(present):]:
            ax.axis("off")
        fig.suptitle("Summary metrics", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "summary_metrics.png"), dpi=300, bbox_inches="tight")
        plt.close()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Plot all metrics from a single run/events file.")
    p.add_argument("path", nargs="?", help="Run directory or direct events.out.tfevents* file path")
    p.add_argument("--out", dest="out_dir", default=None, help="Output directory for plots (default: <path>/plots)")
    p.add_argument("--max-steps", type=int, default=None, help="Optional upper bound on steps to plot")
    p.add_argument("--compare-gradients", nargs=2, metavar=("RUN_A", "RUN_B"), help="Overlay Gradient/train tags between two runs or events files.")
    p.add_argument("--compare-normalize", choices=["globalmax", "max", "first", "l2", "none"], default="globalmax", help="Normalization to apply per-run when comparing gradients.")
    p.add_argument("--label-a", default=None, help="Optional label for RUN_A in gradient comparison.")
    p.add_argument("--label-b", default=None, help="Optional label for RUN_B in gradient comparison.")
    p.add_argument("--compare-gradients-dir", dest="compare_gradients_dir", default=None, help="Recursively find all events files under this directory and overlay Gradient/train tags across them.")
    p.add_argument("--compare-logy", action="store_true", help="Use symmetric log scale for gradient comparison plots.")
    p.add_argument("--compare-kind", choices=["conv", "all", "dense"], default="all", help="Restrict gradient comparison (default: all gradients).")
    p.add_argument("--weightdist-subplots", action="store_true", help="Plot WeightDist stats in separate subplots instead of overlaid on one axis.")
    p.add_argument("--weightdist-logy", action="store_true", help="Use log y-scale for WeightDist plots.")
    p.add_argument("--saturation-only", action="store_true", help="Only generate combined layer saturation plots.")
    p.add_argument("--saturation-ylim", nargs=2, type=float, metavar=("YMIN", "YMAX"), default=None, help="Set y-axis limits for saturation plots.")
    return p.parse_args(argv)

def find_all_events_files(root_dir : str):
    pattern = os.path.join(root_dir, "**", "events.out.tfevents.*")
    return glob.glob(pattern, recursive=True)

def main(argv=None):
    args = parse_args(argv)

    if args.compare_gradients_dir:
        compare_train_gradients_multi(
            args.compare_gradients_dir,
            out_dir=args.out_dir,
            normalize=args.compare_normalize,
            max_steps=args.max_steps,
            logy=args.compare_logy,
            kind=args.compare_kind,
        )
        return

    if args.compare_gradients:
        compare_train_gradients(
            args.compare_gradients[0],
            args.compare_gradients[1],
            out_dir=args.out_dir,
            normalize=args.compare_normalize,
            max_steps=args.max_steps,
            label_a=args.label_a,
            label_b=args.label_b,
            logy=args.compare_logy,
            kind=args.compare_kind,
        )
        return

    plot_all_events_file = False
    if plot_all_events_file:
        tfevents_list = find_all_events_files(args.path)
        
        for events_file in tfevents_list:
            default_out = os.path.join(os.path.dirname(events_file), "plots")
            out_dir = default_out
            plot_single_run(
                events_file,
                out_dir,
                max_steps=args.max_steps,
                weightdist_subplots=args.weightdist_subplots,
                saturation_only=args.saturation_only,
                saturation_ylim=tuple(args.saturation_ylim) if args.saturation_ylim else None,
            )
    else:
        debug = True
        debug_path = "/home/filip/server_code/simulation_results/runs/single_runs/nom-cool-2_100-epochs-perfect-diode/events.out.tfevents.1764177391.nom-cool-2.2967796.0"

        path = args.path or (debug_path if debug else None)
        if path is None:
            raise SystemExit("Please provide a path or enable debug_path.")

        default_out = os.path.join(path if os.path.isdir(path) else os.path.dirname(path), "plots")
        out_dir = args.out_dir or default_out
        plot_single_run(
            path,
            out_dir,
            max_steps=args.max_steps,
            weightdist_subplots=args.weightdist_subplots,
            logy_weightdist=args.weightdist_logy,
            saturation_only=args.saturation_only,
            saturation_ylim=tuple(args.saturation_ylim) if args.saturation_ylim else None,
        )


if __name__ == "__main__":
    main()
