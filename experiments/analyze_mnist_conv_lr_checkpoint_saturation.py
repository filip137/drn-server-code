#!/usr/bin/env python3
"""Replay trained Conv LR-study checkpoints and measure hidden saturation.

Only the deterministic validation split carved from MNIST training data is
available through this command.  The official MNIST test split is prohibited.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.mnist_conv.saturation_replay import replay_saturation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    parser.add_argument("--entry-dir", action="append", required=True)
    parser.add_argument("--initialization-dir", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-batches", type=int, default=8)
    parser.add_argument(
        "--reference-inference-iterations",
        type=int,
        default=None,
        help=(
            "Optional diagnostic settling depth. Omit to use every row's frozen "
            "operational inference iterations."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = replay_saturation(
        study_path=args.study,
        entry_dirs=args.entry_dir,
        initialization_dir=args.initialization_dir,
        data_root=args.data_root,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        reference_inference_iterations=args.reference_inference_iterations,
    )
    output = Path(args.output_dir).expanduser().resolve()
    print(
        f"[done] checkpoints={len(result['checkpoints'])} "
        f"layers={len(result['layers'])} samples={result['cohort']['sample_count']} "
        f"output={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
