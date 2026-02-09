import argparse
import json
import sys
from pathlib import Path

import numpy as np

LABS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = LABS_DIR.parent
for path in (LABS_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from common import export_pt_to_npz, flatten_weights_and_inputs


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
    return weights_out, shapes


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="flatten.py", description="Flatten weights and/or inputs saved in .npz files.")
    p.add_argument("--weights-npz", default=None, help="Path to weights .npz.")
    p.add_argument("--weights-pt", default=None, help="Path to weights .pt (will be exported to .npz first).")
    p.add_argument("--inputs-npz", default=None, help="Path to inputs/state .npz.")
    p.add_argument("--output-dir", required=True, help="Directory to write flattened outputs.")
    p.add_argument("--input-layer", default=None, help="Layer name to flatten for inputs (default: first key).")
    args = p.parse_args(argv)

    if not args.weights_npz and not args.weights_pt and not args.inputs_npz:
        raise SystemExit("Provide at least one of --weights-npz, --weights-pt, or --inputs-npz.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shapes = {}

    weights_npz = None
    if args.weights_pt:
        weights_pt = Path(args.weights_pt)
        weights_npz = output_dir / f"{weights_pt.stem}.npz"
        export_pt_to_npz(weights_pt, weights_npz)
    elif args.weights_npz:
        weights_npz = Path(args.weights_npz)

    if weights_npz and args.inputs_npz:
        weights_out, inputs_out = flatten_weights_and_inputs(
            weights_npz, Path(args.inputs_npz), output_dir, input_layer=args.input_layer
        )
        shapes_path = output_dir / "flattened_shapes.json"
        print(f"weights_flat: {weights_out}")
        print(f"inputs_flat: {inputs_out}")
        print(f"shapes: {shapes_path}")
        return 0

    if weights_npz:
        weights_out, weights_shapes = _flatten_weights_npz(weights_npz, output_dir)
        shapes.update(weights_shapes)
        print(f"weights_flat: {weights_out}")

    if args.inputs_npz:
        inputs_out, inputs_shapes = _flatten_npz(
            Path(args.inputs_npz), output_dir, input_layer=args.input_layer
        )
        shapes.update(inputs_shapes)
        print(f"inputs_flat: {inputs_out}")

    shapes_path = output_dir / "flattened_shapes.json"
    shapes_path.write_text(json.dumps(shapes, indent=2))
    print(f"shapes: {shapes_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
