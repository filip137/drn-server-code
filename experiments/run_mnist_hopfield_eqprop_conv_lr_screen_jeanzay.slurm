#!/bin/bash -l
#SBATCH --job-name=mnist_hopf_lr
#SBATCH --output=slurm/mnist_hopf_lr/%x-%A_%a.out
#SBATCH --error=slurm/mnist_hopf_lr/%x-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=06:00:00
#SBATCH --account=fmu@v100
#SBATCH --qos=qos_gpu-t3
#SBATCH --partition=gpu_p13
#SBATCH --constraint=v100-16g
#SBATCH --array=0-5%6

set -euo pipefail

PROJECT="${IDRPROJ_OVERRIDE:-fmu}"
if [ -x /gpfslocalsup/bin/idrenv ]; then
  eval "$(/gpfslocalsup/bin/idrenv -d "${PROJECT}")"
elif command -v idrenv >/dev/null 2>&1; then
  eval "$(idrenv -d "${PROJECT}")"
fi

module purge
module load "${PYTORCH_MODULE:-pytorch-gpu/py3/2.5.0}"

UMG_WORK="${WORK_umg:-/lustre/fswork/projects/rech/umg/${USER}}"
UMG_SCRATCH="${SCRATCH_umg:-/lustre/fsn1/projects/rech/umg/${USER}}"
FMU_SCRATCH="${SCRATCH_fmu:-/lustre/fsn1/projects/rech/fmu/${USER}}"
SOURCE_ROOT="${SOURCE_ROOT:-${UMG_WORK}/server_code}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${FMU_SCRATCH}/server_code/results/mnist_hopfield_eqprop_conv_saturation_20260628}"
DATASET_ROOT="${DATASET_ROOT:-${UMG_SCRATCH}/datasets/mnist}"
PYTHON="${PYTHON:-python}"
PACK_SIZE="${PACK_SIZE:-2}"
SUMMARY_ONLY="${SUMMARY_ONLY:-0}"

cd "${SOURCE_ROOT}"
mkdir -p "${OUTPUT_ROOT}" slurm/mnist_hopf_lr

export PYTHONUNBUFFERED=1
export PYTHONPATH="${SOURCE_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

if [ "${SUMMARY_ONLY}" = "1" ]; then
  "${PYTHON}" "${SOURCE_ROOT}/experiments/collect_mnist_hopfield_eqprop_lr_screen.py" \
    --input-root "${OUTPUT_ROOT}/lr_screen" \
    --summary-csv "${OUTPUT_ROOT}/lr_screen_summary.csv" \
    --selected-csv "${OUTPUT_ROOT}/selected_lr_by_depth.csv"
  exit 0
fi

DEPTHS=(1 1 1 1 2 2 2 2 3 3 3 3)
LRS=(0.25 0.5 1.0 2.0 0.25 0.5 1.0 2.0 0.25 0.5 1.0 2.0)

run_one() {
  local conv_depth="$1"
  local lr_multiplier="$2"
  local extra_args=()
  if [ -n "${MAX_BATCHES:-}" ]; then
    extra_args+=(--max-batches "${MAX_BATCHES}")
  fi
  if [ -n "${MAX_TEST_BATCHES:-}" ]; then
    extra_args+=(--max-test-batches "${MAX_TEST_BATCHES}")
  fi
  echo "[hopf-lr] job=${SLURM_JOB_ID:-manual} task=${SLURM_ARRAY_TASK_ID:-0} conv=${conv_depth} lr_mult=${lr_multiplier}"
  "${PYTHON}" "${SOURCE_ROOT}/experiments/train_mnist_hopfield_eqprop_conv_sweep.py" \
    --output-root "${OUTPUT_ROOT}" \
    --dataset-root "${DATASET_ROOT}" \
    --device cuda \
    --run-group lr_screen \
    --conv-depth "${conv_depth}" \
    --seed 0 \
    --epochs "${EPOCHS:-5}" \
    --batch-size "${BATCH_SIZE:-64}" \
    --test-batch-size 200 \
    --t1 "${T1:-200}" \
    --t2 "${T2:-10}" \
    --beta 0.4 \
    --lr-multiplier "${lr_multiplier}" \
    --log-interval "${LOG_INTERVAL:-100}" \
    --no-download \
    "${extra_args[@]}"
}

wait_batch() {
  local status=0
  local pid
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done
  pids=()
  active=0
  return "${status}"
}

base_task_id=$((SLURM_ARRAY_TASK_ID * PACK_SIZE))
pids=()
active=0
status=0
for ((offset = 0; offset < PACK_SIZE; offset++)); do
  task_id=$((base_task_id + offset))
  if [ "${task_id}" -ge "${#DEPTHS[@]}" ]; then
    continue
  fi
  run_one "${DEPTHS[$task_id]}" "${LRS[$task_id]}" &
  pids+=("$!")
  active=$((active + 1))
done
if [ "${active}" -gt 0 ]; then
  wait_batch || status=1
fi
exit "${status}"
