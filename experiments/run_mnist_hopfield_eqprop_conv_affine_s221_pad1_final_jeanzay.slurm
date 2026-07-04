#!/bin/bash -l
#SBATCH --job-name=mnist_hopf_aff_s221p1_final
#SBATCH --output=slurm/mnist_hopf_aff_s221p1_final/%x-%A_%a.out
#SBATCH --error=slurm/mnist_hopf_aff_s221p1_final/%x-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --account=fmu@v100
#SBATCH --qos=qos_gpu-t3
#SBATCH --partition=gpu_p13
#SBATCH --constraint=v100-16g
#SBATCH --array=0-2%3

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
OUTPUT_ROOT="${OUTPUT_ROOT:-${FMU_SCRATCH}/server_code/results/mnist_hopfield_eqprop_conv_affine_s221_pad1_20260704}"
DATASET_ROOT="${DATASET_ROOT:-${UMG_SCRATCH}/datasets/mnist}"
PYTHON="${PYTHON:-python}"
PACK_SIZE="${PACK_SIZE:-2}"
AFFINE_PRESET="${AFFINE_PRESET:-medium}"
RUN_GROUP="${RUN_GROUP:-final_affine_mnist_s221_pad1}"
TOTAL_RUNS=6

cd "${SOURCE_ROOT}"
mkdir -p "${OUTPUT_ROOT}" slurm/mnist_hopf_aff_s221p1_final
PACK_LOG_DIR="${OUTPUT_ROOT}/packed_logs/${SLURM_JOB_ID:-manual}_final"
mkdir -p "${PACK_LOG_DIR}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${SOURCE_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

run_one() {
  local task_id="$1"
  local config_id seed conv_depth
  config_id=$((task_id / 3))
  seed=$((task_id % 3))
  if [ "${config_id}" -eq 0 ]; then
    conv_depth=2
  else
    conv_depth=3
  fi

  echo "[mnist-hopf-aff-s221p1-final] job=${SLURM_JOB_ID:-manual} task=${SLURM_ARRAY_TASK_ID:-0} packed=${task_id} conv=${conv_depth} seed=${seed} affine=${AFFINE_PRESET}"
  "${PYTHON}" "${SOURCE_ROOT}/experiments/train_mnist_hopfield_eqprop_conv_sweep.py" \
    --output-root "${OUTPUT_ROOT}" \
    --dataset-root "${DATASET_ROOT}" \
    --dataset mnist \
    --device cuda \
    --run-group "${RUN_GROUP}" \
    --conv-depth "${conv_depth}" \
    --strides 2 2 1 \
    --padding 1 \
    --affine-preset "${AFFINE_PRESET}" \
    --seed "${seed}" \
    --epochs "${EPOCHS:-50}" \
    --batch-size "${BATCH_SIZE:-64}" \
    --test-batch-size "${TEST_BATCH_SIZE:-200}" \
    --t1 "${T1:-200}" \
    --t2 "${T2:-10}" \
    --beta 0.4 \
    --lr-multiplier 2.0 \
    --log-interval "${LOG_INTERVAL:-500}" \
    --no-download
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
  if [ "${task_id}" -ge "${TOTAL_RUNS}" ]; then
    continue
  fi
  run_one "${task_id}" >"${PACK_LOG_DIR}/task_${task_id}.log" 2>&1 &
  pids+=("$!")
  active=$((active + 1))
done
if [ "${active}" -gt 0 ]; then
  wait_batch || status=1
fi
exit "${status}"
