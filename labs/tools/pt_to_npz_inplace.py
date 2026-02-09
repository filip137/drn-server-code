#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path("/home/filip/server_code")
LABS_DIR = PROJECT_ROOT / "labs"
for path in (PROJECT_ROOT, LABS_DIR):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from labs.common import export_pt_to_npz  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert a model.pt into a .npz saved next to the .pt file.",
    )
    parser.add_argument("model_pt", help="Path to model.pt")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output .npz path (defaults to same folder as model.pt).",
    )
    args = parser.parse_args(argv)

    pt_path = Path(args.model_pt).expanduser()
    if args.output:
        out_path = Path(args.output).expanduser()
    else:
        out_path = pt_path.with_suffix(".npz")

    result = export_pt_to_npz(pt_path, out_path)
    print(f"Wrote {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
