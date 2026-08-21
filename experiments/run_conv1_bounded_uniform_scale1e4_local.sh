#!/usr/bin/env bash
set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
study_id="perfectdiode-conv1-bounded-uniform-scale1e4-adam-3ep-seed0-20260813-v1"
config_root="${repo_root}/configs/conv/${study_id}"
result_root="${repo_root}/results/${study_id}"
launcher_root="${result_root}/launcher"
python_bin="/home/filip/miniconda3/envs/py312/bin/python"

mkdir -p "${launcher_root}" /tmp/matplotlib-conv1-bounded-scale1e4
date --iso-8601=seconds > "${launcher_root}/started_at"

export KMP_DISABLE_SHM=1
export KMP_SHM_DISABLE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/matplotlib-conv1-bounded-scale1e4
export PYTHONUNBUFFERED=1

set +e
"${python_bin}" -m experiments.exact_run \
  "${config_root}/conv1_baseline_adam.json" \
  "${config_root}/conv1_ours_adam.json" \
  "${config_root}/conv1_legacy_adam.json" \
  --output-root "${result_root}/runs" \
  --device cuda \
  --checkpoint-every-epoch \
  --study-id "${study_id}" \
  --target main \
  --summary-json "${result_root}/production_summary.json" \
  > "${launcher_root}/main.log" 2>&1
return_code=$?
set -e

date --iso-8601=seconds > "${launcher_root}/finished_at"
printf '%s\n' "${return_code}" > "${launcher_root}/exit_code"
exit "${return_code}"
