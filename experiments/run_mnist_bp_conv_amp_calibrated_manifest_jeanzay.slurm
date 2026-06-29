#!/bin/bash -l
#SBATCH --job-name=mnist_conv_amp_cal
#SBATCH --output=slurm/mnist_conv_amp_cal/%x-%A_%a.out
#SBATCH --error=slurm/mnist_conv_amp_cal/%x-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=02:45:00
#SBATCH --account=fmu@v100
#SBATCH --qos=qos_gpu-t3
#SBATCH --partition=gpu_p13
#SBATCH --constraint=v100-16g
#SBATCH --array=0-41%42

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
PYTHON_BIN="${PYTHON_BIN:-python}"
MANIFEST_CSV="${MANIFEST_CSV:?Set MANIFEST_CSV to the calibrated training manifest.}"
DATASET_ROOT="${DATASET_ROOT:-${UMG_SCRATCH}/datasets/mnist}"

cd "${SOURCE_ROOT}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${SOURCE_ROOT}:${SOURCE_ROOT}/labs:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

mkdir -p slurm/mnist_conv_amp_cal

echo "[jean-zay-train] job=${SLURM_JOB_ID:-manual} array_task=${SLURM_ARRAY_TASK_ID:-0}"
echo "[jean-zay-train] source_root=${SOURCE_ROOT}"
echo "[jean-zay-train] manifest=${MANIFEST_CSV}"
echo "[jean-zay-train] dataset_root=${DATASET_ROOT}"
echo "[jean-zay-train] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"

MANIFEST_CSV="${MANIFEST_CSV}" \
REPO_ROOT="${SOURCE_ROOT}" \
PYTHON_BIN="${PYTHON_BIN}" \
DATASET_ROOT="${DATASET_ROOT}" \
DEVICE=cuda \
EPOCHS="${EPOCHS:-10}" \
BATCH_SIZE="${BATCH_SIZE:-4}" \
NUM_SHARDS="${NUM_SHARDS:-42}" \
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}" \
GPU_UTIL_THRESHOLD=100 \
GPU_MEM_THRESHOLD_MB=12000 \
srun bash "${SOURCE_ROOT}/experiments/run_mnist_bp_conv_amp_calibrated_manifest_lane.sh"
