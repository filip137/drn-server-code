#!/usr/bin/env python3
"""Deprecated delegate to the canonical MNIST Conv command."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_COMMANDS = {"run", "sweep", "collect"}


def main(argv: list[str] | None = None) -> int:
    provided = list(sys.argv[1:] if argv is None else argv)
    print(
        "Expected format: python -m experiments.mnist_conv "
        "{run|sweep|collect} --config CONFIG.json --results-root RESULTS",
        file=sys.stderr,
    )
    print(
        "This historical runner is deprecated and delegates only the canonical "
        "MNIST Conv grammar.",
        file=sys.stderr,
    )
    if not provided or provided[0] not in CANONICAL_COMMANDS:
        print(f"Provided legacy arguments: {provided!r}", file=sys.stderr)
        return 2

    completed = subprocess.run(
        [sys.executable, "-m", "experiments.mnist_conv", *provided],
        cwd=REPO_ROOT,
        check=False,
    )
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
