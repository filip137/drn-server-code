#!/usr/bin/env bash
# One unchanged exact config per Slurm array task. SOURCE CONFIGS OUTPUT DATA [smoke].
set -euo pipefail
(( $# == 4 || $# == 5 )) || exit 2
mode=${5:-production}
[[ $mode == production || $mode == smoke ]] || exit 2
source_dir=$1
config_dir=$2
output_root=$3
data_dir=$4
task=${SLURM_ARRAY_TASK_ID:?Expected a Slurm array task index}
[[ $task =~ ^[0-5]$ ]] || exit 2
configs=(
  conv3_baseline_cos95_seed0.json
  conv3_baseline_cos90_seed0.json
  conv3_legacy_cos95_seed0.json
  conv3_legacy_cos90_seed0.json
  conv3_ours_cos95_seed0.json
  conv3_ours_cos90_seed0.json
)
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
config="$config_dir/${configs[$task]}"
extra=()
[[ $mode != smoke ]] || extra+=(--smoke)
timeout --signal=TERM --kill-after=20s 7140s \
  python -m experiments.exact_run "$config" --index 0 "${extra[@]}" \
  --output-root "$case_dir/runs" --device cuda --dataset-root "$data_dir" \
  --target jean-zay --gradient-trace-samples-per-epoch 8 \
  --summary-json "$case_dir/summary.json"
