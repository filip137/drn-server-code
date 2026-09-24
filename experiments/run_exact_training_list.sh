#!/usr/bin/env bash
# Direct, serial exact-run transport. Account for its full per-case wall limits
# in the local GPU-hour ledger before calling this script. One array at a time.
set -euo pipefail
if (( $# != 6 )); then
  echo 'Usage: run_exact_training_list.sh SOURCE LIST OUTPUT DATA TARGET PYTHON' >&2
  exit 2
fi
source_dir=$1
config_list=$2
output_dir=$3
dataset_dir=$4
target=$5
python_bin=$6
cd "$source_dir"
export KMP_DISABLE_SHM=1 KMP_SHM_DISABLE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp/mpl-paper-training TF_CPP_MIN_LOG_LEVEL=2
export GIT_CEILING_DIRECTORIES
GIT_CEILING_DIRECTORIES=$(dirname "$source_dir")
export EXPERIMENT_SOURCE_COMMIT
export EXPERIMENT_SOURCE_ARCHIVE_SHA256
EXPERIMENT_SOURCE_COMMIT=$(cat SOURCE_COMMIT)
EXPERIMENT_SOURCE_ARCHIVE_SHA256=$(cat SOURCE_ARCHIVE_SHA256)
sha256sum --check --quiet INPUT_SHA256SUMS
mapfile -t configs < "$config_list"
if [[ -n ${SLURM_ARRAY_TASK_ID:-} ]]; then
  index=$SLURM_ARRAY_TASK_ID
  [[ $index =~ ^[0-9]+$ ]] && (( index < ${#configs[@]} )) || exit 2
  configs=("${configs[$index]}")
  output_dir="$output_dir/task_$index"
fi
mkdir -p "$output_dir"
# Atomic duplicate guard, retained even after termination. Retry in a new root.
mkdir "$output_dir/worker_started"
date --iso-8601=seconds > "$output_dir/worker_started/started_at"
printf '%s\n' "$$" > "$output_dir/worker_started/pid"
finish() {
  local code=$?
  date --iso-8601=seconds > "$output_dir/worker_started/ended_at"
  printf '%s\n' "$code" > "$output_dir/worker_started/exit_code"
}
trap finish EXIT
for config in "${configs[@]}"; do
  [[ -n $config ]] || continue
  # Recheck between serial cases: a different user may have occupied the lab
  # GPU since this worker started. Slurm owns device isolation for its tasks.
  if [[ -z ${SLURM_JOB_ID:-} ]]; then
    gpu_processes=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)
    if [[ -n ${gpu_processes//[[:space:]]/} ]]; then
      printf 'GPU_OCCUPIED: remaining configs held; compute PIDs: %s\n' "$gpu_processes" >&2
      exit 75
    fi
  fi
  wall_seconds=$("$python_bin" - "$config" <<'PY'
import hashlib,json,sys
from pathlib import Path
config=json.loads(Path(sys.argv[1]).read_text())
plan=config['completion_plan']
if plan['qualification']=='bounded_beta_pending':
    raise SystemExit('Bounded EP is not numerically qualified')
if plan['replication_gate']=='all_27_bounded_pilots_and_budget_review':
    raise SystemExit('Old bounded replication hold; prepare an explicitly released config')
if plan['replication_gate']=='qualified_group_seed0_pilots':
    from experiments.check_training_pilots import validate_bounded_replication
    gate=json.loads(Path('guards/bounded_pilots.json').read_text())
    validate_bounded_replication(config, gate)
if plan['replication_gate']=='all_9_wide_pilots':
    gate=json.loads(Path('guards/wide_pilots.json').read_text())
    assert gate['stable'] and not gate['bounded'] and len(gate['pilots'])==9
assert config['evaluation']['official_test']['policy']=='disabled'
asset=Path(config['init_checkpoint_path'])
assert hashlib.sha256(asset.read_bytes()).hexdigest()==config['initialization']['checkpoint_sha256']
depth=len(config['model_overrides'][config['lab']['model_key']]['conv_pipeline'])
print(7200 if depth==1 else 28800)
PY
  )
  # TERM then KILL both fit within the reserved 2/8 hours, including cleanup.
  timeout --signal=TERM --kill-after=20s "$((wall_seconds-30))s" \
    "$python_bin" -m experiments.exact_run "$config" --index 0 \
    --output-root "$output_dir" --device cuda --dataset-root "$dataset_dir" \
    --study-id paper-training-completion-20260911-v1 --target "$target" \
    --summary-json "$output_dir/$(basename "$config" .json).summary.json"
done
