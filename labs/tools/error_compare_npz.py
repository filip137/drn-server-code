#!/usr/bin/env python3
import argparse
import json
import shlex
import sys
from pathlib import Path

PROJECT_ROOT = Path("/home/filip/server_code")
LABS_DIR = PROJECT_ROOT / "labs"
for path in (PROJECT_ROOT, LABS_DIR):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from paper_post_processing import (  # noqa: E402
    _layer_sort_key,
    load_npz_params,
    plot_npz_error_comparison,
    write_npz_error_summary,
    write_percentile_error_summaries,
)


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
        help="Where to write plots/summary (default: common parent of inputs).",
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

    if args.all_layers:
        if args.keys:
            raise SystemExit("Use --all-layers without --keys.")
        cd_data = load_npz_params(args.cd_npz)
        spice_data = load_npz_params(args.spice_npz)
        common = sorted(set(cd_data) & set(spice_data), key=_layer_sort_key)
        if not common:
            raise SystemExit("No common layers found between the two files.")
        for key in common:
            if args.plot:
                compare_npz_errors(
                    args.cd_npz,
                    args.spice_npz,
                    keys=key,
                    output_dir=args.output_dir,
                    log_scale=args.log_scale,
                    cli=cli,
                )
            else:
                _write_metadata(args.cd_npz, args.spice_npz, output_dir=args.output_dir, keys=key, log_scale=args.log_scale, cli=cli)
                write_npz_error_summary(
                    [args.cd_npz, args.spice_npz],
                    keys=key,
                    output_dir=args.output_dir,
                    log_scale=args.log_scale,
                )
    else:
        if args.plot:
            compare_npz_errors(
                args.cd_npz,
                args.spice_npz,
                keys=args.keys,
                output_dir=args.output_dir,
                log_scale=args.log_scale,
                cli=cli,
            )
        else:
            _write_metadata(args.cd_npz, args.spice_npz, output_dir=args.output_dir, keys=args.keys, log_scale=args.log_scale, cli=cli)
            write_npz_error_summary(
                [args.cd_npz, args.spice_npz],
                keys=args.keys,
                output_dir=args.output_dir,
                log_scale=args.log_scale,
            )
    if args.percentiles:
        write_percentile_error_summaries(
            args.cd_npz,
            args.spice_npz,
            output_dir=args.output_dir,
            percentiles=args.percentiles,
            keys=args.keys,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
