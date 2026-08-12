#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
study_config=${1:-"$repo_root/configs/conv/perfectdiode_conv3_baseline_ours_float64_beta_observables_20260811_v1.json"}
python_bin=${PYTHON_BIN:-/home/filip/miniconda3/envs/py312/bin/python}

mapfile -t resolved < <(
  "$python_bin" - "$study_config" <<'PY'
import json
from pathlib import Path
import sys

config_path = Path(sys.argv[1]).resolve()
config = json.loads(config_path.read_text(encoding="utf-8"))
root = config_path.parents[2]
print(root / config["source"]["config"])
print(root / config["source"]["run"])
print(root / config["output_root"])
print(config["replay"]["T"])
print(config["replay"]["K"])
print(config["replay"]["batch_indices"][0])
for beta in config["beta_contract"]["common_effective_betas"]:
    print(format(float(beta), ".17g"))
PY
)

source_config=${resolved[0]}
source_run=${resolved[1]}
output_root=${resolved[2]}
replay_t=${resolved[3]}
replay_k=${resolved[4]}
batch_index=${resolved[5]}
betas=("${resolved[@]:6}")

for index in "${!betas[@]}"; do
  injected_beta=${betas[$index]}
  ours_beta=$("$python_bin" -c 'import sys; print(format(float(sys.argv[1]) / 64.0, ".17g"))' "$injected_beta")
  run_id=$(printf 'beta-%02d' "$index")
  run_dir="$output_root/$run_id"
  if [[ -e "$run_dir" ]]; then
    echo "Refusing to overwrite existing run directory: $run_dir" >&2
    exit 2
  fi

  "$python_bin" "$repo_root/experiments/audit_eqprop_float64_shadow.py" \
    --config "$source_config" \
    --source-run "$source_run" \
    --case "conv3:baseline:best_validation:$injected_beta" \
    --case "conv3:ours:best_validation:$ours_beta" \
    --explicit-beta-label wide_equal_injected_observables \
    --allow-out-of-grid-beta \
    --tk-override "$replay_t" "$replay_k" \
    --batch-index "$batch_index" \
    --output-root "$output_root" \
    --run-id "$run_id" \
    --device cuda \
    --target local:RTX3090
done
