"""Execute an immutable v6 sequential-wave pack inside one GPU allocation."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from experiments.mnist_conv.lr_v6_wave_executor import execute_packed_waves


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--pack-manifest", required=True)
    parser.add_argument("--benchmark")
    parser.add_argument("--stage", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--python", required=True)
    parser.add_argument("--log-dir", required=True)
    args = parser.parse_args(argv)
    try:
        result = execute_packed_waves(
            study_dir=args.study,
            manifest_path=args.manifest,
            pack_manifest_path=args.pack_manifest,
            benchmark_path=args.benchmark,
            stage=args.stage,
            data_root=args.data_root,
            device=args.device,
            python=args.python,
            log_dir=args.log_dir,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(str(exc), flush=True)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
