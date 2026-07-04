#!/usr/bin/env python3
import argparse
import json
import shlex
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path("/home/filip/server_code")
LABS_DIR = PROJECT_ROOT / "labs"
for path in (PROJECT_ROOT, LABS_DIR):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from paper_post_processing import (  # noqa: E402
    _layer_sort_key,
    compute_relative_L1_error,
    load_npz_params,
    plot_npz_error_comparison,
    write_npz_error_summary,
    write_percentile_error_summaries,
)

DEFAULT_REL_PERCENTILES = (50, 60, 70, 80, 90, 95, 99)
DEFAULT_REL_EPS = 1e-12


def _flatten_samples(values):
    values = np.asarray(values)
    if values.ndim == 0:
        raise ValueError("Expected array-like values with at least 1 dimension.")
    if values.ndim == 1:
        return values[:, None]
    return values.reshape(values.shape[0], -1)


def _percentile_summary(values, percentiles):
    return {f"p{int(p)}": float(np.percentile(values, p)) for p in percentiles}


def _write_cross_layer_percentile_average(
    cd_npz,
    spice_npz,
    layers,
    output_dir,
    percentiles=DEFAULT_REL_PERCENTILES,
):
    cd_data = load_npz_params(cd_npz)
    spice_data = load_npz_params(spice_npz)
    pcts = [int(p) for p in percentiles]
    per_layer = {}
    aggregate = {f"p{p}": [] for p in pcts}

    for layer in layers:
        rel_err, _, _ = compute_relative_L1_error(
            _flatten_samples(cd_data[layer]),
            _flatten_samples(spice_data[layer]),
            axis=1,
        )
        layer_stats = {f"p{p}": float(np.percentile(rel_err, p)) for p in pcts}
        per_layer[layer] = layer_stats
        for key, value in layer_stats.items():
            aggregate[key].append(value)

    averaged = {key: float(np.mean(values)) for key, values in aggregate.items()}
    payload = {
        "cd_npz": str(Path(cd_npz).expanduser().resolve()),
        "spice_npz": str(Path(spice_npz).expanduser().resolve()),
        "num_layers": len(layers),
        "layers": list(layers),
        "percentiles": pcts,
        "average_rel_l1_percentiles_across_layers": averaged,
        "per_layer_rel_l1_percentiles": per_layer,
    }
    out_path = Path(output_dir) / "cross_layer_rel_l1_percentiles.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Saved cross-layer percentile summary to {out_path}")
    return out_path


def _write_node_weighted_percentile_average(
    cd_npz,
    spice_npz,
    layers,
    output_dir,
    percentiles=DEFAULT_REL_PERCENTILES,
    eps=DEFAULT_REL_EPS,
):
    cd_data = load_npz_params(cd_npz)
    spice_data = load_npz_params(spice_npz)
    pcts = [int(p) for p in percentiles]
    node_counts = {}
    total_nodes = 0
    total_mae = None
    total_ref = None
    node_rel_errors = []
    node_abs_errors = []

    for layer in layers:
        cd_vals = _flatten_samples(cd_data[layer])
        spice_vals = _flatten_samples(spice_data[layer])
        if cd_vals.shape != spice_vals.shape:
            raise ValueError(
                f"Shape mismatch for {layer}: cd {cd_vals.shape} vs spice {spice_vals.shape}"
            )
        node_count = int(cd_vals.shape[1])
        node_counts[layer] = node_count
        total_nodes += node_count
        diff = np.abs(cd_vals - spice_vals)
        node_abs_errors.append(diff)
        node_rel_errors.append(diff / (np.abs(spice_vals) + eps))
        mae = np.mean(diff, axis=1)
        ref_mean_abs = np.mean(np.abs(spice_vals), axis=1)
        if total_mae is None:
            total_mae = mae * node_count
            total_ref = ref_mean_abs * node_count
        else:
            total_mae += mae * node_count
            total_ref += ref_mean_abs * node_count

    if total_nodes == 0:
        raise ValueError("No nodes found when aggregating across layers.")

    rel_err = total_mae / (total_ref + eps)
    node_rel_p90_per_sample = np.percentile(np.concatenate(node_rel_errors, axis=1), 90, axis=1)
    node_abs_p90_per_sample = np.percentile(np.concatenate(node_abs_errors, axis=1), 90, axis=1)
    payload = {
        "cd_npz": str(Path(cd_npz).expanduser().resolve()),
        "spice_npz": str(Path(spice_npz).expanduser().resolve()),
        "num_layers": len(layers),
        "layers": list(layers),
        "percentiles": pcts,
        "total_nodes": total_nodes,
        "node_counts": node_counts,
        "node_weighted_rel_l1_percentiles": _percentile_summary(rel_err, pcts),
        "node_rel_error_p90_over_nodes_percentiles_over_samples": _percentile_summary(
            node_rel_p90_per_sample,
            pcts,
        ),
        "node_abs_error_p90_over_nodes_percentiles_over_samples": _percentile_summary(
            node_abs_p90_per_sample,
            pcts,
        ),
    }
    out_path = Path(output_dir) / "cross_layer_rel_l1_percentiles_node_weighted.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Saved node-weighted percentile summary to {out_path}")
    return out_path


def _write_metadata(cd_npz, spice_npz, *, output_dir, keys=None, log_scale=False, cli=None):
    meta = {
        "cd_npz": str(Path(cd_npz).expanduser().resolve()),
        "spice_npz": str(Path(spice_npz).expanduser().resolve()),
        "keys": keys,
        "log_scale": log_scale,
        "cli": cli,
    }
    output_dir = Path(output_dir) if output_dir else Path(cd_npz).expanduser().resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = output_dir / "error_compare_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta_path


def compare_npz_errors(cd_npz, spice_npz, *, keys=None, output_dir=None, log_scale=False, cli=None):
    """Compute and plot relative + absolute errors between two npz files."""
    summary_path = plot_npz_error_comparison(
        [cd_npz, spice_npz],
        keys=keys,
        output_dir=output_dir,
        log_scale=log_scale,
    )
    _write_metadata(cd_npz, spice_npz, output_dir=output_dir, keys=keys, log_scale=log_scale, cli=cli)
    if cli:
        try:
            summary = json.loads(Path(summary_path).read_text())
        except (OSError, json.JSONDecodeError):
            summary = {}
        summary["cli"] = cli
        Path(summary_path).write_text(json.dumps(summary, indent=2))
    return summary_path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="error_compare_npz.py",
        description="Compare two npz files and plot relative/absolute errors.",
    )
    p.add_argument("cd_npz", help="Path to CD (or reference) .npz file.")
    p.add_argument("spice_npz", help="Path to SPICE (or comparison) .npz file.")
    p.add_argument(
        "--keys",
        nargs="+",
        help="Optional key (or two keys) to compare; defaults to common last layer.",
    )
    p.add_argument(
        "--all-layers",
        action="store_true",
        help="Compare all common layers between the two files.",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Where to write plots/summary (default: <spice_npz_dir>/error_npz).",
    )
    p.add_argument(
        "--log-scale",
        action="store_true",
        help="Plot log10 of error values.",
    )
    p.add_argument(
        "--plot",
        action="store_true",
        help="Generate plots (default: summaries only).",
    )
    p.add_argument(
        "--percentiles",
        nargs="+",
        type=float,
        default=None,
        help="Write percentile summaries (e.g., 50 90 99) into separate folders.",
    )
    args = p.parse_args(argv)
    cli = " ".join([shlex.quote(sys.executable), shlex.quote(str(Path(__file__).resolve()))] + [shlex.quote(a) for a in sys.argv[1:]])

    if not args.all_layers and not args.keys:
        args.all_layers = True

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = str(Path(args.spice_npz).expanduser().resolve().parent / "error_npz")

    if args.all_layers:
        if args.keys:
            raise SystemExit("Use --all-layers without --keys.")
        cd_data = load_npz_params(args.cd_npz)
        spice_data = load_npz_params(args.spice_npz)
        common = sorted(set(cd_data) & set(spice_data), key=_layer_sort_key)
        if not common:
            raise SystemExit("No common layers found between the two files.")
        output_root = None
        for key in common:
            if args.plot:
                summary_path = compare_npz_errors(
                    args.cd_npz,
                    args.spice_npz,
                    keys=key,
                    output_dir=output_dir,
                    log_scale=args.log_scale,
                    cli=cli,
                )
            else:
                _write_metadata(args.cd_npz, args.spice_npz, output_dir=output_dir, keys=key, log_scale=args.log_scale, cli=cli)
                summary_path = write_npz_error_summary(
                    [args.cd_npz, args.spice_npz],
                    keys=key,
                    output_dir=output_dir,
                    log_scale=args.log_scale,
                )
            if output_root is None:
                output_root = Path(summary_path).parent
        if output_root is not None:
            _write_cross_layer_percentile_average(
                args.cd_npz,
                args.spice_npz,
                common,
                output_dir=output_root,
            )
            _write_node_weighted_percentile_average(
                args.cd_npz,
                args.spice_npz,
                common,
                output_dir=output_root,
            )
    else:
        if args.plot:
            compare_npz_errors(
                args.cd_npz,
                args.spice_npz,
                keys=args.keys,
                output_dir=output_dir,
                log_scale=args.log_scale,
                cli=cli,
            )
        else:
            _write_metadata(args.cd_npz, args.spice_npz, output_dir=output_dir, keys=args.keys, log_scale=args.log_scale, cli=cli)
            write_npz_error_summary(
                [args.cd_npz, args.spice_npz],
                keys=args.keys,
                output_dir=output_dir,
                log_scale=args.log_scale,
            )
    if args.percentiles:
        write_percentile_error_summaries(
            args.cd_npz,
            args.spice_npz,
            output_dir=output_dir,
            percentiles=args.percentiles,
            keys=args.keys,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
