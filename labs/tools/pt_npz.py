#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path("/home/filip/server_code")
LABS_DIR = PROJECT_ROOT / "labs"
for path in (PROJECT_ROOT, LABS_DIR):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from common import export_pt_to_npz  # noqa: E402


def _flatten_weights_npz(weights_npz_path: Path, output_dir: Path):
    shapes = {"weights": {}}
    with np.load(weights_npz_path) as weights_data:
        flat_weights = {}
        for name, array in weights_data.items():
            if array.ndim > 2:
                flat_array = array.reshape(-1, array.shape[-1])
            else:
                flat_array = array
            flat_weights[name] = flat_array
            shapes["weights"][name] = {
                "original": list(array.shape),
                "flat": list(flat_array.shape),
            }
    weights_out = output_dir / f"{weights_npz_path.stem}_flat.npz"
    np.savez(weights_out, **flat_weights)
    shapes_path = output_dir / "flattened_shapes.json"
    shapes_path.write_text(json.dumps(shapes, indent=2))
    return weights_out, shapes_path

    
def _flatten_npz(inputs_npz_path: Path, output_dir: Path, input_layer: str | None = None):
    shapes = {"inputs": {}}
    with np.load(inputs_npz_path) as inputs_data:
        flat_inputs = {}
        layer_name = input_layer
        if layer_name is None:
            layer_name = inputs_data.files[0] if inputs_data.files else None
        for name, array in inputs_data.items():
            if name == layer_name and array.ndim >= 2:
                flat_array = array.reshape(array.shape[0], -1)
            else:
                flat_array = array
            flat_inputs[name] = flat_array
            shapes["inputs"][name] = {
                "original": list(array.shape),
                "flat": list(flat_array.shape),
            }
    inputs_out = output_dir / f"{inputs_npz_path.stem}_flat.npz"
    np.savez(inputs_out, **flat_inputs)
    return inputs_out, shapes

def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="pt_to_npz.py",
        description="Convert a model.pt (list/tuple of tensors) into a .npz file.",
    )
    p.add_argument("input_pt", help="Path to input model .pt.")
    p.add_argument(
        "--output-npz",
        default=None,
        help="Output path (default: same name with .npz extension).",
    )
    p.add_argument(
        "--flatten",
        action="store_true",
        help="Also write a flattened weights .npz and shapes JSON.",
    )
    args = p.parse_args(argv)

    input_path = Path(args.input_pt).expanduser().resolve()
    output_path = Path(args.output_npz).expanduser().resolve() if args.output_npz else None
    result = export_pt_to_npz(input_path, output_path)
    print(f"Wrote: {result}")
    if args.flatten:
        out_dir = result.parent
        flat_path, shapes_path = _flatten_weights_npz(result, out_dir)
        print(f"Wrote: {flat_path}")
        print(f"Wrote: {shapes_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
