#!/usr/bin/env python3
"""CLI for the bounded perfect-diode Conv one-batch functional smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from experiments.mnist_conv.io import read_json
from experiments.mnist_conv.perfectdiode_functional_smoke import (
    run_perfectdiode_one_batch_smoke,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run exactly one real perfect-diode Conv training batch and write "
            "an immutable functional-preflight receipt."
        )
    )
    parser.add_argument("--study", required=True)
    parser.add_argument("--bundle", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--entry-id")
    group.add_argument("--entry-index", type=int)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--download", action="store_true")
    return parser


def _entry(manifest: Any, *, entry_id: str | None, entry_index: int | None) -> dict:
    if not isinstance(manifest, dict) or not isinstance(
        manifest.get("entries"), list
    ):
        raise ValueError(
            "Expected bundle manifest.entries to be a list. "
            f"Provided value: {manifest!r}."
        )
    if entry_id is not None:
        matches = [
            item
            for item in manifest["entries"]
            if isinstance(item, dict) and item.get("entry_id") == entry_id
        ]
        expected = f"exactly one manifest entry with entry_id={entry_id!r}"
    else:
        matches = [
            item
            for item in manifest["entries"]
            if isinstance(item, dict)
            and item.get("entry_index") == entry_index
        ]
        expected = (
            "exactly one manifest entry with "
            f"entry_index={entry_index!r}"
        )
    if len(matches) != 1:
        raise ValueError(
            f"Expected {expected}. Provided value: {len(matches)} matches."
        )
    return dict(matches[0])


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    bundle = Path(args.bundle).expanduser().resolve()
    study = read_json(args.study)
    manifest = read_json(bundle / "manifest.json")
    payload = _entry(
        manifest,
        entry_id=args.entry_id,
        entry_index=args.entry_index,
    )
    result = run_perfectdiode_one_batch_smoke(
        study,
        bundle,
        payload,
        attempt_id=args.attempt_id,
        target=args.target,
        data_root=args.data_root,
        device=args.device,
        output_dir=args.output_dir,
        download=args.download,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
