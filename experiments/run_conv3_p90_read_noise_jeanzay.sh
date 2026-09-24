#!/usr/bin/env bash
# Two-case H100 pack. SOURCE CONFIGS OUTPUT DATA [smoke].
set -euo pipefail
(( $# == 4 || $# == 5 )) || exit 2
mode=${5:-production}
[[ $mode == production || $mode == smoke ]] || exit 2
source_dir=$1
config_dir=$2
output_root=$3
data_dir=$4
task=${SLURM_ARRAY_TASK_ID:?Expected a Slurm array task index}
[[ $task =~ ^[0-7]$ ]] || exit 2
mapfile -t configs < "$config_dir/config-names.txt"
(( ${#configs[@]} == 15 )) || exit 2
case_dir="$output_root/task_$task"
mkdir -p "$case_dir"
mkdir "$case_dir/worker_started"
printf '%s\n' "$$" > "$case_dir/worker_started/pid"
date --iso-8601=seconds > "$case_dir/worker_started/started_at"
finish() {
  code=$?
  date --iso-8601=seconds > "$case_dir/worker_started/finished_at"
  printf '%s\n' "$code" > "$case_dir/worker_started/exit_code"
}
trap finish EXIT
export MODULESHOME=/lustre/fshomisc/sup/spack_soft/environment-modules/4.3.1/gcc-11.3.1-wf7m7j6whgecysm2fm5n73sm4jg7txup
export MODULEPATH=/lustre/fshomisc/sup/hpe/pub/module-rh/modulefiles:/lustre/fshomisc/sup/hpe/pub/modules-idris-env4/modulefiles/linux-rhel9-skylake_avx512
module_command="$MODULESHOME/bin/modulecmd"
eval "$("$module_command" bash purge)"
eval "$("$module_command" bash load arch/h100)"
eval "$("$module_command" bash load pytorch-gpu/py3/2.5.0)"
export KMP_DISABLE_SHM=1 KMP_SHM_DISABLE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2
export MPLCONFIGDIR="/tmp/mpl-refined-beta-${SLURM_JOB_ID}"
export GIT_CEILING_DIRECTORIES
GIT_CEILING_DIRECTORIES=$(dirname "$source_dir")
cd "$source_dir"
export EXPERIMENT_SOURCE_COMMIT EXPERIMENT_SOURCE_ARCHIVE_SHA256
EXPERIMENT_SOURCE_COMMIT=$(cat SOURCE_COMMIT)
EXPERIMENT_SOURCE_ARCHIVE_SHA256=$(cat SOURCE_ARCHIVE_SHA256)
sha256sum --check --quiet LAYERWISE_SHA256SUMS
extra=()
[[ $mode != smoke ]] || extra+=(--smoke)
for index in "$task" "$((task+8))"; do
  (( index < ${#configs[@]} )) || continue
  config="$config_dir/${configs[$index]}"
  run_dir="$case_dir/case_$index"
  mkdir "$run_dir"
  if timeout --signal=TERM --kill-after=20s 3540s \
    python -m experiments.exact_run "$config" --index 0 "${extra[@]}" \
    --output-root "$run_dir/runs" --device cuda --dataset-root "$data_dir" \
    --target jean-zay --gradient-trace-samples-per-epoch 8 \
    --summary-json "$run_dir/summary.json" > "$run_dir/train.log" 2>&1; then
    code=0
  else
    code=$?
  fi
  printf '%s\n' "$code" > "$run_dir/exit_code"
  if (( code != 0 )); then
    # Keep numerical failures as evidence, but stop on operational defects.
    python - "$run_dir" <<'PYCHECK'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); statuses=list((p/'runs').rglob('status.json'))
assert len(statuses)==1 and json.loads(statuses[0].read_text())['state']=='failed'
assert 'NonFiniteTrainingError' in (p/'train.log').read_text()
PYCHECK
  fi
done
