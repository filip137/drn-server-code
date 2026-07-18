#!/usr/bin/env python3
"""Create a canonical MNIST Conv SweepSpec from frozen inputs."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.build_mnist_bp_conv_amp_calibrated_training_manifest import (  # noqa: E402
    main as _canonical_sweep_main,
)


def main() -> int:
    """Delegate to the canonical calibration-to-SweepSpec converter."""

    return _canonical_sweep_main()


if __name__ == "__main__":
    raise SystemExit(main())
