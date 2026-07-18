#!/usr/bin/env python3
import argparse
import json
import math
import os
import re
import shutil
import subprocess
from pathlib import Path


def extract_percentile_series(root: Path, percentile: int, metric: str = "rel_l1"):
    pattern = f"iter*/hidden_3/*/p{percentile}/error_summary.json"
    series = {}
    for path in root.glob(pattern):
        match = re.search(r"iter(\d+)", str(path))
        if not match:
            continue
        iters = int(match.group(1))
        data = json.loads(path.read_text())
        layers = data.get("layers", {})
        for name, values in layers.items():
            layer_num = int(name.replace("Layer_", ""))
            value = values.get(metric)
            if value is None:
                continue
            series.setdefault(layer_num, []).append((iters, float(value)))
    for layer in series:
        series[layer] = sorted(series[layer])
    return series


def write_svg(series, out_svg: Path, title: str, y_label: str, log_y: bool):
    width, height = 800, 500
    margin = 60
    plot_w, plot_h = width - 2 * margin, height - 2 * margin
    all_iters = sorted({i for points in series.values() for i, _ in points})
    all_vals = [v for points in series.values() for _, v in points]
    if not all_iters:
        raise SystemExit("No data found to plot.")
    xmin, xmax = min(all_iters), max(all_iters)
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

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    def x_to_px(x):
        lx = math.log2(x)
        return margin + (lx - lxmin) / (lxmax - lxmin) * plot_w

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
        f"<text x='{width/2}' y='{height-10}' font-size='12' text-anchor='middle' fill='#111'>Iterations (log2 scale)</text>"
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
    out_svg.write_text("\n".join(lines))


def write_csv(series, out_csv: Path):
    with out_csv.open("w") as handle:
        handle.write("layer,iters,value\n")
        for layer in sorted(series):
            for iters, value in series[layer]:
                handle.write(f"{layer},{iters},{value}\n")


def _maybe_write_png(svg_path: Path, png_path: Path) -> bool:
    rsvg = _find_binary(["rsvg-convert"])
    if not rsvg:
        return False
    subprocess.run([rsvg, "-o", str(png_path), str(svg_path)], check=True)
    return png_path.is_file()


def _find_binary(candidates):
    for candidate in candidates:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return None


def _plot_matplotlib(series, out_svg: Path, out_png: Path | None, title: str, y_label: str, log_y: bool):
    os.environ.setdefault("KMP_DISABLE_SHM", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("MPLBACKEND", "Agg")

    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 4.5))
    for layer in sorted(series):
        points = series[layer]
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        plt.plot(xs, ys, marker="o", label=f"Layer {layer}")

    plt.xscale("log", base=2)
    if log_y:
        min_pos = min((v for pts in series.values() for _, v in pts if v > 0), default=None)
        if min_pos is None:
            raise SystemExit("Log-scale requested but no positive values found.")
        max_val = max(v for pts in series.values() for _, v in pts)
        plt.yscale("log")
        plt.ylim(bottom=min_pos / 10.0, top=max_val)
    plt.xlabel("Iterations")
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, which="both", linestyle=":", alpha=0.5)
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_svg, dpi=200)
    if out_png:
        plt.savefig(out_png, dpi=200)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot percentile error summaries vs iterations without matplotlib.",
    )
    parser.add_argument("root", help="Root folder containing iter*/hidden_3/*/pXX/error_summary.json")
    parser.add_argument(
        "--percentiles",
        nargs="+",
        type=int,
        default=[90, 99],
        help="Percentiles to plot (default: 90 99).",
    )
    parser.add_argument(
        "--metric",
        default="rel_l1",
        help="Metric key inside each layer block (default: rel_l1).",
    )
    parser.add_argument(
        "--png",
        action="store_true",
        help="Also write PNGs (requires rsvg-convert).",
    )
    parser.add_argument(
        "--log-y",
        action="store_true",
        help="Use log scale for Y axis.",
    )
    parser.add_argument(
        "--matplotlib",
        action="store_true",
        help="Use matplotlib for plotting (falls back to SVG if unavailable).",
    )
    args = parser.parse_args()

    root = Path(args.root)
    wrote_any = False
    for percentile in args.percentiles:
        series = extract_percentile_series(root, percentile, metric=args.metric)
        if not series:
            print(f"No data for p{percentile} under {root}")
            continue
        out_svg = root / f"p{percentile}_{args.metric}_vs_iters.svg"
        out_csv = root / f"p{percentile}_{args.metric}_vs_iters.csv"
        out_png = root / f"p{percentile}_{args.metric}_vs_iters.png" if args.png else None

        if args.matplotlib:
            try:
                _plot_matplotlib(
                    series,
                    out_svg,
                    out_png,
                    f"p{percentile} {args.metric} vs iterations",
                    args.metric,
                    args.log_y,
                )
            except Exception as exc:
                print(f"Matplotlib plot failed ({exc}); falling back to SVG.")
                write_svg(
                    series,
                    out_svg,
                    f"p{percentile} {args.metric} vs iterations",
                    args.metric,
                    args.log_y,
                )
        else:
            write_svg(
                series,
                out_svg,
                f"p{percentile} {args.metric} vs iterations",
                args.metric,
                args.log_y,
            )
        write_csv(series, out_csv)
        wrote_any = True
        print(f"Saved {out_svg}")
        print(f"Saved {out_csv}")
        if args.png:
            if out_png and out_png.exists():
                print(f"Saved {out_png}")
            elif out_png:
                if _maybe_write_png(out_svg, out_png):
                    print(f"Saved {out_png}")
                else:
                    raise RuntimeError(
                        "Expected --png to produce a PNG via matplotlib or "
                        f"rsvg-convert. Provided output path: {out_png}."
                    )
    if not wrote_any:
        raise RuntimeError(
            "Expected at least one percentile error summary to plot. "
            f"Provided root: {root}."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
