"""Measure immutable v5 candidate GPU peaks without reading MNIST test data."""

from __future__ import annotations

import argparse
import json

from experiments.mnist_conv.lr_v5_memory_preflight import run_v5_memory_preflight


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    parser.add_argument("--candidate-manifest", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps-per-runtime", type=int, default=3)
    args = parser.parse_args()
    result = run_v5_memory_preflight(
        study_dir=args.study,
        candidate_manifest_path=args.candidate_manifest,
        data_root=args.data_root,
        output_path=args.output,
        device=args.device,
        steps_per_runtime=args.steps_per_runtime,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "study_id": result["study_id"],
                "runtime_count": len(result["representative_measurements"]),
                "candidate_entry_count": len(
                    result["peak_gpu_memory_mib_by_entry_index"]
                ),
                "output": args.output,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
