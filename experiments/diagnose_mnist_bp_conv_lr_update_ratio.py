#!/usr/bin/env python3
"""Read checkpoint-update diagnostics from canonical MNIST Conv bundles."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.mnist_conv.update_ratio_diagnostic import main as _canonical_main  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Analyze validated bundles without reconstructing or launching a run."""

    return _canonical_main(
        argv,
        description=(
            "Diagnose best-to-final parameter update ratios from completed "
            "canonical MNIST Conv run bundles."
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
