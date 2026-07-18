#!/usr/bin/env python3
"""Retirement shim for the archived ``small_network`` command."""

from __future__ import annotations

import argparse
import sys


ARCHIVE_TAG = "archive/small-network-v1"
REPLACEMENT = "python -m experiments.mnist_conv run --config RUN.json --results-root results"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--mode")
    parser.add_argument("--config")
    return parser


def main(argv: list[str] | None = None) -> int:
    args, unknown = _parser().parse_known_args(argv)
    print(
        "Expected format: "
        "python -m experiments.mnist_conv run --config RUN.json "
        "--results-root RESULTS_ROOT",
        file=sys.stderr,
    )
    print(
        "labs/small_network.py is retired; its final implementation is archived "
        f"at Git tag {ARCHIVE_TAG}.",
        file=sys.stderr,
    )
    print(f"Replacement: {REPLACEMENT}", file=sys.stderr)
    print(
        f"Provided legacy mode={args.mode!r}, config={args.config!r}, "
        f"extra_args={unknown!r}",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
