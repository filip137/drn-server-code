#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
  cat <<'USAGE'
Usage: compare_npz.sh <run_npz> <spice_npz> <output_dir> [extra args...]

Example:
  compare_npz.sh run.npz spice.npz out_dir --plot --log-scale
USAGE
  exit 2
fi

run_npz="$1"
spice_npz="$2"
out_dir="$3"
shift 3

python3 /home/filip/server_code/labs/tools/error_compare_npz.py \
  "$run_npz" \
  "$spice_npz" \
  --output-dir "$out_dir" \
  --all-layers \
  "$@"
