#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import plot_conditioning_vs_runtime as pcr


DEPTH_STYLE = {
    1: ("o", "H1"),
    2: ("s", "H2"),
    3: ("^", "H3"),
}

FAMILY_COLOR = {
    "single_diode_exponential": ("Single diode", "#1f77b4"),
    "double_diode_exponential": ("Double diode", "#d62728"),
    "experimental": ("Experimental I-V", "#2ca02c"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot frozen-Hessian rho against average outer sweeps across hidden-1/2/3 for single- and double-diode timing runs."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path('/home/filip/server_code/labs/cases/rho_vs_runtime_all_hiddens'),
        help="Directory where the combined CSV, plot, and note will be written.",
    )
    return parser.parse_args()


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    return pcr.pearson_corr(x.astype(np.float64), y.astype(np.float64))


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    return pcr.spearman_corr(x.astype(np.float64), y.astype(np.float64))


def stopping_time_proxies(rho: float) -> tuple[float, float]:
    rho = float(rho)
    if not (0.0 < rho < 1.0):
        return math.nan, math.nan
    return 1.0 / (-math.log(rho)), 1.0 / (1.0 - rho)


def collect_rows() -> list[dict]:
    rows = []
    for hidden_layers in [1, 2, 3]:
        for nonlinearity, width_map in pcr.CASE_RUNS_BY_DEPTH[hidden_layers].items():
            for width, meta_path in sorted(width_map.items()):
                row = pcr.analyze_case(nonlinearity, width, meta_path)
                row['hidden_layers'] = hidden_layers
                inv_neg_log_rho, inv_one_minus_rho = stopping_time_proxies(row['rho_full_mean_diag'])
                row['inv_neg_log_rho'] = inv_neg_log_rho
                row['inv_one_minus_rho'] = inv_one_minus_rho
                rows.append(row)
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    fieldnames = [
        'nonlinearity', 'hidden_layers', 'width', 'avg_outer_iterations', 'rho_full_mean_diag',
        'inv_neg_log_rho', 'inv_one_minus_rho',
        'one_minus_sigma_over_L_block_min', 'sigma', 'kappa_full', 'kappa_block_min', 'kappa_block_max',
        'config_path', 'weights_path', 'states_path'
    ]
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, '') for k in fieldnames})


def summarize(rows: list[dict]) -> dict:
    summary = {'all_runs': {}, 'by_family': {}, 'by_depth': {}}
    metrics = [
        'rho_full_mean_diag',
        'inv_neg_log_rho',
        'inv_one_minus_rho',
        'one_minus_sigma_over_L_block_min',
        'kappa_block_min',
    ]

    def stats(group_rows: list[dict]) -> dict:
        out = {}
        y = np.asarray([r['avg_outer_iterations'] for r in group_rows], float)
        for metric in metrics:
            x = np.asarray([r[metric] for r in group_rows], float)
            out[metric] = {
                'n': int(len(group_rows)),
                'pearson': pearson_corr(x, y),
                'spearman': spearman_corr(x, y),
            }
        return out

    summary['all_runs'] = stats(rows)
    for nonlinearity in FAMILY_COLOR:
        summary['by_family'][nonlinearity] = stats([r for r in rows if r['nonlinearity'] == nonlinearity])
    for hidden_layers in [1, 2, 3]:
        summary['by_depth'][str(hidden_layers)] = stats([r for r in rows if r['hidden_layers'] == hidden_layers])
    return summary


def make_plot(rows: list[dict], summary: dict, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.4), constrained_layout=True)
    metric_specs = [
        ('rho_full_mean_diag', r"$\rho(T)$"),
        ('inv_neg_log_rho', r"$1/(-\log \rho(T))$"),
        ('inv_one_minus_rho', r"$1/(1-\rho(T))$"),
    ]
    for ax, (metric, xlabel) in zip(axes, metric_specs):
        for nonlinearity in FAMILY_COLOR:
            family_rows = [r for r in rows if r['nonlinearity'] == nonlinearity]
            family_label, color = FAMILY_COLOR[nonlinearity]
            for hidden_layers in [1, 2, 3]:
                marker, depth_label = DEPTH_STYLE[hidden_layers]
                depth_rows = [r for r in family_rows if r['hidden_layers'] == hidden_layers]
                x = np.asarray([r[metric] for r in depth_rows], float)
                y = np.asarray([r['avg_outer_iterations'] for r in depth_rows], float)
                label = f'{family_label}, {depth_label}' if metric == 'rho_full_mean_diag' else None
                ax.scatter(x, y, s=64, marker=marker, color=color, alpha=0.88, label=label)
                for r in depth_rows:
                    ax.annotate(
                        f"{depth_label}-{r['width']}",
                        (r[metric], r['avg_outer_iterations']),
                        textcoords='offset points',
                        xytext=(4, 4),
                        fontsize=7,
                    )
        corr = summary['all_runs'][metric]
        p_txt = 'nan' if math.isnan(corr['pearson']) else f"{corr['pearson']:.3f}"
        s_txt = 'nan' if math.isnan(corr['spearman']) else f"{corr['spearman']:.3f}"
        ax.set_title(f"All runs\nPearson={p_txt}, Spearman={s_txt}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Average outer sweeps')
        ax.grid(True, alpha=0.3)
    axes[0].legend(frameon=False, fontsize=8, ncols=3)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def write_readme(summary: dict, path: Path) -> None:
    all_rho = summary['all_runs']['rho_full_mean_diag']
    all_inv_log = summary['all_runs']['inv_neg_log_rho']
    all_inv_gap = summary['all_runs']['inv_one_minus_rho']
    all_thm = summary['all_runs']['one_minus_sigma_over_L_block_min']
    lines = [
        '# Rho vs Runtime Across Hidden Depths',
        '',
        'Artifacts:',
        '- `rho_vs_runtime_all_hiddens.csv`: combined frozen-Hessian summary for hidden-1/2/3, single diode, double diode, and the experimental I-V diode.',
        '- `rho_vs_runtime_all_hiddens.png`: scatter of average outer sweeps versus `rho(T)` and the stopping-time transforms `1/(-log rho(T))` and `1/(1-rho(T))`.',
        '- `rho_vs_runtime_all_hiddens_summary.json`: correlation summary for `rho(T)` and a few simpler proxies.',
        '',
        'Main observation:',
        f"- Across all runs, raw `rho(T)` has Pearson `{all_rho['pearson']:.3f}` and Spearman `{all_rho['spearman']:.3f}` against average outer sweeps.",
        f"- The stopping-time proxy `1/(-log rho(T))` improves the linear fit slightly (Pearson `{all_inv_log['pearson']:.3f}`, Spearman `{all_inv_log['spearman']:.3f}`).",
        f"- The proxy `1/(1-rho(T))` gives Pearson `{all_inv_gap['pearson']:.3f}` and Spearman `{all_inv_gap['spearman']:.3f}`.",
        f"- The theorem-style proxy `1 - sigma / min(Lodd, Leven)` remains weaker overall (Pearson `{all_thm['pearson']:.3f}`, Spearman `{all_thm['spearman']:.3f}`).",
        '- Agreement is still qualitative rather than exact: the stopping-time transforms reduce some curvature in the scatter, but there are clear family- and depth-specific deviations.',
        '',
        'Family-level reading:',
    ]
    for nonlinearity, (label, _color) in FAMILY_COLOR.items():
        rho = summary['by_family'][nonlinearity]['rho_full_mean_diag']
        lines.append(f"- `{label}`: Pearson `{rho['pearson']:.3f}`, Spearman `{rho['spearman']:.3f}`.")
    lines += [
        '',
        'Interpretation:',
        '- `rho(T)` is the best structural proxy among the frozen-Hessian quantities checked here.',
        '- The remaining mismatch is expected because runtime is a stopping time of the nonlinear solver, while `rho(T)` is a local contraction factor of a frozen averaged linearization.',
    ]
    path.write_text('\n'.join(lines) + '\n')


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = collect_rows()
    csv_path = args.output_dir / 'rho_vs_runtime_all_hiddens.csv'
    png_path = args.output_dir / 'rho_vs_runtime_all_hiddens.png'
    json_path = args.output_dir / 'rho_vs_runtime_all_hiddens_summary.json'
    md_path = args.output_dir / 'README.md'
    write_csv(rows, csv_path)
    summary = summarize(rows)
    with json_path.open('w') as f:
        json.dump(summary, f, indent=2)
        f.write('\n')
    make_plot(rows, summary, png_path)
    write_readme(summary, md_path)
    print(f'Wrote {csv_path}')
    print(f'Wrote {png_path}')
    print(f'Wrote {json_path}')
    print(f'Wrote {md_path}')


if __name__ == '__main__':
    main()
