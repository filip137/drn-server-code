#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 3 ]]; then
  echo "Usage: $0 <config.json> <weights.pt> <base_dir>" >&2
  echo "Example: $0 /path/to/config.json /path/to/model.pt /path/to/iterations" >&2
  exit 2
fi

CONFIG="$1"
WEIGHTS="$2"
BASE="$3"

for it in 4 8 16 32 64 128 256 512; do
  OUT="$BASE/iter${it}"
  /home/filip/miniconda3/envs/py312/bin/python /home/filip/server_code/labs/small_network.py \
    --mode digits_validate \
    --dataset digits \
    --num-iterations "$it" \
    --config "$CONFIG" \
    --weights "$WEIGHTS" \
    --output-dir "$OUT"
done
