#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.function.interaction import (
    FUNCTION_CHECKPOINT_FORMAT,
    FUNCTION_CHECKPOINT_VERSION,
    load_function_checkpoint_artifact,
)


def zero_last_param(input_path: Path, output_path: Path) -> Path:
    params, schema, source_format = load_function_checkpoint_artifact(
        input_path,
        map_location="cpu",
    )
    if not params:
        raise ValueError(
            f"Expected checkpoint {input_path} to contain at least one parameter state. "
            "Provided value: 0 parameter states."
        )
    params[-1] = torch.zeros_like(params[-1])

    if source_format == "versioned":
        output_payload = {
            "format": FUNCTION_CHECKPOINT_FORMAT,
            "version": FUNCTION_CHECKPOINT_VERSION,
            "schema": schema,
            "states": params,
        }
    else:
        output_payload = params

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_payload, output_path)
    return output_path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="zero_last_param.py",
        description="Zero the last tensor in a versioned or historical parameter checkpoint.",
    )
    p.add_argument("input_pt", help="Path to an input model checkpoint.")
    p.add_argument(
        "--output-pt",
        default=None,
        help="Output path (default: model_zero_bias.pt next to input).",
    )
    args = p.parse_args(argv)

    input_path = Path(args.input_pt).expanduser().resolve()
    if args.output_pt:
        output_path = Path(args.output_pt).expanduser().resolve()
    else:
        output_path = input_path.with_name("model_zero_bias.pt")

    result = zero_last_param(input_path, output_path)
    print(f"Wrote: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
