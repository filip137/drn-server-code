#!/bin/bash -l
#SBATCH --job-name=mnist_conv_fix_post
#SBATCH --output=slurm/mnist_conv_fix_post/%x-%A_%a.out
#SBATCH --error=slurm/mnist_conv_fix_post/%x-%A_%a.err
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
#SBATCH --array=0-1%2

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
RESULT_ROOT="${RESULT_ROOT:-${FMU_SCRATCH}/server_code/results/mnist_bp_conv_fixed_step_redo_r3_20260703}"
INPUT_ROOT="${INPUT_ROOT:-${RESULT_ROOT}/final_seeds012_50epoch}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${RESULT_ROOT}/post_diagnostics}"
DATASET_ROOT="${DATASET_ROOT:-${UMG_SCRATCH}/datasets/mnist}"
PYTHON="${PYTHON:-python}"

cd "${SOURCE_ROOT}"
mkdir -p "${OUTPUT_ROOT}" slurm/mnist_conv_fix_post

export PYTHONUNBUFFERED=1
export PYTHONPATH="${SOURCE_ROOT}:${SOURCE_ROOT}/labs:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

KINDS=(best final)
checkpoint_kind="${KINDS[${SLURM_ARRAY_TASK_ID:-0}]}"
job_root="${OUTPUT_ROOT}/${checkpoint_kind}"

echo "[post-diagnostics] checkpoint=${checkpoint_kind} input=${INPUT_ROOT} output=${job_root}"

srun "${PYTHON}" "${SOURCE_ROOT}/experiments/diagnose_mnist_bp_conv_saturation_gradients.py" \
  --input-root "${INPUT_ROOT}" \
  --output-root "${job_root}" \
  --dataset-root "${DATASET_ROOT}" \
  --device cuda \
  --checkpoint-kind "${checkpoint_kind}" \
  --split train \
  --batch-size "${BATCH_SIZE:-16}" \
  --max-batches "${MAX_BATCHES:-8}" \
  --no-download
