#!/usr/bin/env python3
import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path
import math

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402
except Exception:  # pragma: no cover - matplotlib optional
    plt = None


LAYER_RE = re.compile(r"Layer_(\d+)_vs_Layer_(\d+)_error_summary\.json$")


def _iter_from_path(path: Path) -> int | None:
    match = re.search(r"iter(\d+)", str(path))
    if not match:
        return None
    return int(match.group(1))


def _layer_from_name(name: str) -> int | None:
    match = LAYER_RE.match(name)
    if not match:
        return None
    return int(match.group(1))


def extract_layer_series(root: Path, hidden: str | None, metrics: list[str]) -> tuple[dict, list]:
    patterns: list[str] = []
    if hidden:
        patterns.append(f"iter*/{hidden}/*/Layer_*_vs_Layer_*_error_summary.json")
    patterns.append("iter*/Layer_*_vs_Layer_*_error_summary.json")
    raw = {metric: {} for metric in metrics}
    duplicates: list[tuple[str, int, int, int]] = []

    seen: set[Path] = set()
    for pattern in patterns:
        for path in root.glob(pattern):
            if path in seen:
                continue
            seen.add(path)
            iter_val = _iter_from_path(path)
            if iter_val is None:
                continue
            layer = _layer_from_name(path.name)
            if layer is None:
                continue
            data = json.loads(path.read_text())
            for metric in metrics:
                if metric not in data:
                    continue
                raw.setdefault(metric, {}).setdefault(layer, {}).setdefault(iter_val, []).append(float(data[metric]))

    aggregated: dict[str, dict[int, list[tuple[int, float]]]] = {}
    for metric, layers in raw.items():
        aggregated[metric] = {}
        for layer, iter_map in layers.items():
            points: list[tuple[int, float]] = []
            for iter_val, values in iter_map.items():
                if len(values) > 1:
                    duplicates.append((metric, layer, iter_val, len(values)))
                points.append((iter_val, sum(values) / len(values)))
            aggregated[metric][layer] = sorted(points)
    return aggregated, duplicates


def write_csv(series: dict[int, list[tuple[int, float]]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w") as handle:
        handle.write("layer,iters,value\n")
        for layer in sorted(series):
            for iters, value in series[layer]:
                handle.write(f"{layer},{iters},{value}\n")


def _write_svg(
    series: dict[int, list[tuple[int, float]]],
    out_svg: Path,
    title: str,
    y_label: str,
    x_label: str,
    log_x: bool,
    log_y: bool,
) -> None:
    width, height = 800, 500
    margin = 60
    plot_w, plot_h = width - 2 * margin, height - 2 * margin
    all_iters = sorted({i for points in series.values() for i, _ in points})
    all_vals = [v for points in series.values() for _, v in points]
    if not all_iters:
        raise SystemExit("No data found to plot.")
    xmin, xmax = min(all_iters), max(all_iters)
    if log_x:
        lxmin, lxmax = math.log2(xmin), math.log2(xmax)
    ymin, ymax = min(all_vals), max(all_vals)
    if log_y:
        min_pos = min((v for v in all_vals if v > 0), default=None)
        if min_pos is None:
            raise SystemExit("Log-scale requested but no positive values found.")
        min_y, max_y = min_pos / 10.0, ymax
    else:
        pad = 0.05 * (ymax - ymin) if ymax > ymin else 0.1
        min_y, max_y = ymin - pad, ymax + pad

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    def x_to_px(x):
        if log_x:
            lx = math.log2(x)
            return margin + (lx - lxmin) / (lxmax - lxmin) * plot_w
        return margin + (x - xmin) / (xmax - xmin) * plot_w

    def y_to_px(y):
        if log_y:
            y = max(y, min_y)
            ly = math.log10(y)
            lmin = math.log10(min_y)
            lmax = math.log10(max_y)
            return margin + (1 - (ly - lmin) / (lmax - lmin)) * plot_h
        return margin + (1 - (y - min_y) / (max_y - min_y)) * plot_h

    lines = []
    lines.append(f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>")
    lines.append("<rect width='100%' height='100%' fill='white' />")
    lines.append(f"<line x1='{margin}' y1='{margin}' x2='{margin}' y2='{height-margin}' stroke='#333'/>")
    lines.append(f"<line x1='{margin}' y1='{height-margin}' x2='{width-margin}' y2='{height-margin}' stroke='#333'/>")

    for it in all_iters:
        x = x_to_px(it)
        lines.append(f"<line x1='{x}' y1='{height-margin}' x2='{x}' y2='{height-margin+5}' stroke='#333' />")
        lines.append(f"<text x='{x}' y='{height-margin+20}' font-size='10' text-anchor='middle' fill='#333'>{it}</text>")

    for idx in range(6):
        if log_y:
            lmin = math.log10(min_y)
            lmax = math.log10(max_y)
            yv = 10 ** (lmin + idx * (lmax - lmin) / 5)
        else:
            yv = min_y + idx * (max_y - min_y) / 5
        y = y_to_px(yv)
        lines.append(f"<line x1='{margin-5}' y1='{y}' x2='{margin}' y2='{y}' stroke='#333' />")
        lines.append(f"<text x='{margin-10}' y='{y+4}' font-size='10' text-anchor='end' fill='#333'>{yv:.3g}</text>")

    for it in all_iters:
        x = x_to_px(it)
        lines.append(f"<line x1='{x}' y1='{margin}' x2='{x}' y2='{height-margin}' stroke='#eee' />")
    for idx in range(6):
        if log_y:
            lmin = math.log10(min_y)
            lmax = math.log10(max_y)
            yv = 10 ** (lmin + idx * (lmax - lmin) / 5)
        else:
            yv = min_y + idx * (max_y - min_y) / 5
        y = y_to_px(yv)
        lines.append(f"<line x1='{margin}' y1='{y}' x2='{width-margin}' y2='{y}' stroke='#eee' />")

    lines.append(f"<text x='{width/2}' y='24' font-size='14' text-anchor='middle' fill='#111'>{title}</text>")
    lines.append(
        f"<text x='{width/2}' y='{height-10}' font-size='12' text-anchor='middle' fill='#111'>{x_label}</text>"
    )
    lines.append(
        f"<text x='18' y='{height/2}' font-size='12' text-anchor='middle' fill='#111' transform='rotate(-90 18 {height/2})'>{y_label}</text>"
    )

    legend_x, legend_y = width - margin + 10, margin
    for idx, layer in enumerate(sorted(series)):
        color = colors[idx % len(colors)]
        points = series[layer]
        path_d = " ".join(
            [("M" if i == 0 else "L") + f"{x_to_px(x):.2f},{y_to_px(y):.2f}" for i, (x, y) in enumerate(points)]
        )
        lines.append(f"<path d='{path_d}' fill='none' stroke='{color}' stroke-width='2'/>")
        for x, y in points:
            lines.append(f"<circle cx='{x_to_px(x):.2f}' cy='{y_to_px(y):.2f}' r='3' fill='{color}' />")
        lines.append(f"<rect x='{legend_x}' y='{legend_y + idx*18 - 8}' width='10' height='10' fill='{color}' />")
        lines.append(f"<text x='{legend_x + 16}' y='{legend_y + idx*18}' font-size='10' fill='#111'>Layer {layer}</text>")

    lines.append("</svg>")
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    out_svg.write_text("\n".join(lines))


def _find_binary(candidates: list[str]) -> str | None:
    for candidate in candidates:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return None


def _maybe_svg_to_pdf(svg_path: Path, pdf_path: Path) -> bool:
    rsvg = _find_binary(["rsvg-convert"])
    if not rsvg:
        return False
    subprocess.run([rsvg, "-f", "pdf", "-o", str(pdf_path), str(svg_path)], check=True)
    return pdf_path.exists()


def _plot_series_matplotlib(
    series: dict[int, list[tuple[int, float]]],
    out_path: Path,
    title: str,
    y_label: str,
    x_label: str,
    log_x: bool,
    log_y: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    for layer in sorted(series):
        points = series[layer]
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        if log_y:
            ys = [max(v, 1e-16) for v in ys]
        ax.plot(xs, ys, marker="o", linewidth=1.8, label=f"Layer {layer}")

    if log_x:
        ax.set_xscale("log", base=2)
    if log_y:
        ax.set_yscale("log")

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plot per-layer error summaries vs iterations.",
    )
    parser.add_argument(
        "--root",
        required=True,
        help="Root folder containing iter*/hidden_*/<run>/Layer_*_error_summary.json files.",
    )
    parser.add_argument(
        "--hidden",
        default="hidden_3",
        help="Hidden folder to scan (default: hidden_3).",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["mean_rel_l1", "median_rel_l1"],
        help="Metric keys to extract (default: mean_rel_l1 median_rel_l1).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write plots/CSVs (default: <root>/layer_error_vs_iters).",
    )
    parser.add_argument(
        "--log-x",
        action="store_true",
        help="Use log2 scale for the x-axis.",
    )
    parser.add_argument(
        "--log-y",
        action="store_true",
        help="Use log scale for the y-axis.",
    )
    args = parser.parse_args(argv)

    root = Path(args.root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else root / "layer_error_vs_iters"
    output_dir.mkdir(parents=True, exist_ok=True)

    series_by_metric, duplicates = extract_layer_series(root, args.hidden, args.metrics)
    if duplicates:
        for metric, layer, iter_val, count in duplicates:
            print(
                f"Warning: {count} entries for {metric} layer {layer} iter {iter_val}; using mean of values."
            )

    summary = {
        "root": str(root),
        "hidden": args.hidden,
        "metrics": {},
    }
    wrote_any = False
    y_label = "Relative Mean Absolute Error"
    x_label = "Number of iterations"

    for metric in args.metrics:
        series = series_by_metric.get(metric, {})
        if not series:
            print(f"No data found for metric '{metric}'.")
            continue
        if metric.startswith("mean"):
            title = "Mean Rel. MAE vs number of iterations"
        elif metric.startswith("median"):
            title = "Median Rel. MAE vs number of iterations"
        else:
            title = "Rel. MAE vs number of iterations"
        summary["metrics"][metric] = {
            str(layer): [{"iter": iters, "value": value} for iters, value in points]
            for layer, points in series.items()
        }
        out_png = output_dir / f"{metric}_vs_iters.png"
        out_svg = output_dir / f"{metric}_vs_iters.svg"
        out_pdf = output_dir / f"{metric}_vs_iters.pdf"
        out_csv = output_dir / f"{metric}_vs_iters.csv"
        if plt is not None:
            _plot_series_matplotlib(
                series,
                out_png,
                title,
                y_label,
                x_label,
                log_x=args.log_x,
                log_y=args.log_y,
            )
            print(f"Saved {out_png}")
            try:
                _plot_series_matplotlib(
                    series,
                    out_pdf,
                    title,
                    y_label,
                    x_label,
                    log_x=args.log_x,
                    log_y=args.log_y,
                )
                print(f"Saved {out_pdf}")
            except Exception as exc:  # pragma: no cover - optional output
                print(f"Warning: failed to save PDF via matplotlib ({exc}).")
        else:
            _write_svg(
                series,
                out_svg,
                title,
                y_label,
                x_label,
                log_x=args.log_x,
                log_y=args.log_y,
            )
            print(f"Saved {out_svg}")
            if _maybe_svg_to_pdf(out_svg, out_pdf):
                print(f"Saved {out_pdf}")
            else:
                print("Warning: PDF conversion skipped (rsvg-convert not available).")
        write_csv(series, out_csv)
        print(f"Saved {out_csv}")
        wrote_any = True

    if not wrote_any:
        raise SystemExit(f"No matching layer error summaries found under {root}")
    summary_path = output_dir / "layer_error_vs_iters.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
