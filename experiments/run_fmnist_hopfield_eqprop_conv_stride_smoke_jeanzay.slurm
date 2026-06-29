#!/bin/bash -l
#SBATCH --job-name=fmnist_hopf_stride_smoke
#SBATCH --output=slurm/fmnist_hopf_stride_smoke/%x-%A_%a.out
#SBATCH --error=slurm/fmnist_hopf_stride_smoke/%x-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=01:00:00
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
OUTPUT_ROOT="${OUTPUT_ROOT:-${FMU_SCRATCH}/server_code/results/fmnist_hopfield_eqprop_conv_stride_20260629}"
DATASET_ROOT="${DATASET_ROOT:-${UMG_SCRATCH}/datasets/fashion_mnist}"
PYTHON="${PYTHON:-python}"

cd "${SOURCE_ROOT}"
mkdir -p "${OUTPUT_ROOT}" slurm/fmnist_hopf_stride_smoke

export PYTHONUNBUFFERED=1
export PYTHONPATH="${SOURCE_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

DEPTHS=(1 1 2 2 3 3)
STRIDES=(1 2 1 2 1 2)
PADDINGS=(0 0 0 0 1 1)
BATCHES=(16 64 16 64 8 64)
TEST_BATCHES=(64 200 64 200 32 200)

task_id="${SLURM_ARRAY_TASK_ID:-0}"
conv_depth="${DEPTHS[$task_id]}"
stride="${STRIDES[$task_id]}"
padding="${PADDINGS[$task_id]}"
batch_size="${BATCHES[$task_id]}"
test_batch_size="${TEST_BATCHES[$task_id]}"

echo "[fmnist-hopf-smoke] job=${SLURM_JOB_ID:-manual} task=${task_id} conv=${conv_depth} stride=${stride}"
"${PYTHON}" "${SOURCE_ROOT}/experiments/train_mnist_hopfield_eqprop_conv_sweep.py" \
  --output-root "${OUTPUT_ROOT}" \
  --dataset-root "${DATASET_ROOT}" \
  --dataset fashion_mnist \
  --device cuda \
  --run-group "smoke_fashion_mnist_stride${stride}" \
  --conv-depth "${conv_depth}" \
  --stride "${stride}" \
  --padding "${padding}" \
  --seed 0 \
  --epochs 1 \
  --batch-size "${batch_size}" \
  --test-batch-size "${test_batch_size}" \
  --t1 "${T1:-200}" \
  --t2 "${T2:-10}" \
  --beta 0.4 \
  --lr-multiplier 2.0 \
  --max-batches "${MAX_BATCHES:-10}" \
  --max-test-batches "${MAX_TEST_BATCHES:-3}" \
  --log-interval "${LOG_INTERVAL:-2}" \
  --no-download
