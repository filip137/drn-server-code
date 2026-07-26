#!/usr/bin/env python3
"""Replay trained Conv LR-study checkpoints and analyze their BP gradients."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.mnist_conv.gradient_replay import (
    DEFAULT_PROBE_BATCH_POSITIONS,
    replay_gradients,
)


def _positions(raw: str) -> tuple[int, ...]:
    return tuple(int(value.strip()) for value in raw.split(",") if value.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    parser.add_argument("--entry-dir", action="append", required=True)
    parser.add_argument("--initialization-dir", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--probe-batch-positions",
        default=",".join(str(value) for value in DEFAULT_PROBE_BATCH_POSITIONS),
        help="Comma-separated positions from the original 32-batch LR probe.",
    )
    parser.add_argument("--reference-training-iterations", type=int, default=64)
    parser.add_argument(
        "--reference-inference-iterations",
        type=int,
        default=None,
        help="Optional common T for the reference; omit to retain each row's T.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = replay_gradients(
        study_path=args.study,
        entry_dirs=args.entry_dir,
        initialization_dir=args.initialization_dir,
        data_root=args.data_root,
        output_dir=args.output_dir,
        device=args.device,
        probe_batch_positions=_positions(args.probe_batch_positions),
        reference_training_iterations=args.reference_training_iterations,
        reference_inference_iterations=args.reference_inference_iterations,
    )
    output = Path(args.output_dir).expanduser().resolve()
    print(
        f"[done] checkpoints={len(result['checkpoints'])} "
        f"parameters={len(result['parameters'])} "
        f"samples={result['cohort']['sample_count']} output={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
