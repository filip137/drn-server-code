#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch


def zero_last_param(input_path: Path, output_path: Path) -> Path:
    params = torch.load(input_path, map_location="cpu")
    if not isinstance(params, (list, tuple)) or not params:
        raise ValueError(f"Expected non-empty list/tuple in {input_path}, got {type(params)}")
    if not torch.is_tensor(params[-1]):
        raise ValueError(f"Last parameter is not a tensor in {input_path}")

    if isinstance(params, tuple):
        params = list(params)
        params[-1] = torch.zeros_like(params[-1])
        params = tuple(params)
    else:
        params[-1] = torch.zeros_like(params[-1])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(params, output_path)
    return output_path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="zero_last_param.py",
        description="Zero the last tensor in a .pt parameter list and save a new file.",
    )
    p.add_argument("input_pt", help="Path to input model .pt (list/tuple of tensors).")
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
