"""Create an immutable Jean Zay pack manifest from measured GPU preflight data."""

from __future__ import annotations

import argparse
import json

from experiments.mnist_conv.lr_v5_packing import create_v5_pack_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", required=True)
    parser.add_argument("--memory-preflight", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = create_v5_pack_manifest(
        args.candidate_manifest,
        args.memory_preflight,
        args.output,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "study_id": manifest["study_id"],
                "pack_count": len(manifest["packs"]),
                "output": args.output,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
