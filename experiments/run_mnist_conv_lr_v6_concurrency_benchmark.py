"""Run the v6 Conv2 production-launcher concurrency benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from experiments.mnist_conv.lr_v6_benchmark import run_concurrency_benchmark


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    parser.add_argument("--candidate-manifest", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    parser.add_argument("--scratch-root", required=True)
    parser.add_argument("--poll-seconds", type=float, default=0.10)
    parser.add_argument("--ready-timeout-seconds", type=float, default=900.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = run_concurrency_benchmark(
            study_dir=args.study,
            candidate_manifest_path=args.candidate_manifest,
            data_root=args.data_root,
            output_path=args.output,
            scratch_root=args.scratch_root,
            device=args.device,
            poll_seconds=args.poll_seconds,
            ready_timeout_seconds=args.ready_timeout_seconds,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(str(exc), flush=True)
        return 2
    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(Path(args.output).expanduser()),
                "selected_concurrency": result["selection"]["selected_concurrency"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
