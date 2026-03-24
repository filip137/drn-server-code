#!/usr/bin/env bash
set -euo pipefail

CONFIG=/home/filip/server_code/simulation_results/digits_medium_network/hidden_1/single_diode_exponential/hidden_128/small_network_single_diode_exponential1h_hidden128.json
WEIGHTS=/home/filip/server_code/simulation_results/digits_medium_network/hidden_1/single_diode_exponential/hidden_128/model.pt
BASE=/home/filip/server_code/simulation_results/digits_medium_network/hidden_1/single_diode_exponential/hidden_128/iterations

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
