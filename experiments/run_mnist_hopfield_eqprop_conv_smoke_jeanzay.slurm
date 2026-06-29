#!/bin/bash -l
#SBATCH --job-name=mnist_hopf_smoke
#SBATCH --output=slurm/mnist_hopf_smoke/%x-%A_%a.out
#SBATCH --error=slurm/mnist_hopf_smoke/%x-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=01:30:00
#SBATCH --account=fmu@v100
#SBATCH --qos=qos_gpu-t3
#SBATCH --partition=gpu_p13
#SBATCH --constraint=v100-16g
#SBATCH --array=0-0

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

cd "${SOURCE_ROOT}"
mkdir -p "${OUTPUT_ROOT}" slurm/mnist_hopf_smoke

export PYTHONUNBUFFERED=1
export PYTHONPATH="${SOURCE_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

DEPTHS=(3 1)
LRS=(1.0 1.0)

run_one() {
  local conv_depth="$1"
  local lr_multiplier="$2"
  echo "[hopf-smoke] job=${SLURM_JOB_ID:-manual} conv=${conv_depth} lr_mult=${lr_multiplier} output=${OUTPUT_ROOT}"
  "${PYTHON}" "${SOURCE_ROOT}/experiments/train_mnist_hopfield_eqprop_conv_sweep.py" \
    --output-root "${OUTPUT_ROOT}" \
    --dataset-root "${DATASET_ROOT}" \
    --device cuda \
    --run-group smoke \
    --conv-depth "${conv_depth}" \
    --seed 0 \
    --epochs 1 \
    --batch-size "${BATCH_SIZE:-64}" \
    --test-batch-size 200 \
    --t1 "${T1:-200}" \
    --t2 "${T2:-10}" \
    --beta 0.4 \
    --lr-multiplier "${lr_multiplier}" \
    --max-batches "${MAX_BATCHES:-2}" \
    --max-test-batches "${MAX_TEST_BATCHES:-1}" \
    --log-interval 1 \
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

pids=()
active=0
status=0
for index in "${!DEPTHS[@]}"; do
  run_one "${DEPTHS[$index]}" "${LRS[$index]}" &
  pids+=("$!")
  active=$((active + 1))
  if [ "${active}" -ge "${PACK_SIZE}" ]; then
    wait_batch || status=1
  fi
done
if [ "${active}" -gt 0 ]; then
  wait_batch || status=1
fi

exit "${status}"
