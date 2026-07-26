"""Create an immutable v6 one-GPU sequential-wave manifest."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from experiments.mnist_conv.lr_v6_packing import create_v6_pack_manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-manifest", required=True)
    parser.add_argument("--benchmark")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        manifest = create_v6_pack_manifest(
            args.stage_manifest,
            args.output,
            benchmark_path=args.benchmark,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(str(exc), flush=True)
        return 2
    print(
        json.dumps(
            {
                "status": "complete",
                "study_id": manifest["study_id"],
                "stage": manifest["stage_name"],
                "wave_count": len(manifest["waves"]),
                "selected_concurrency": manifest["execution_contract"][
                    "selected_concurrency"
                ],
                "output": args.output,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
