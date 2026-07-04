#!/bin/bash -l
#SBATCH --job-name=mnist_hopf_tanh_collect
#SBATCH --output=slurm/mnist_hopf_tanh_collect/%x-%j.out
#SBATCH --error=slurm/mnist_hopf_tanh_collect/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=00:30:00
#SBATCH --account=fmu@v100
#SBATCH --qos=qos_gpu-t3
#SBATCH --partition=gpu_p13
#SBATCH --constraint=v100-16g

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
FMU_SCRATCH="${SCRATCH_fmu:-/lustre/fsn1/projects/rech/fmu/${USER}}"
SOURCE_ROOT="${SOURCE_ROOT:-${UMG_WORK}/server_code}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${FMU_SCRATCH}/server_code/results/mnist_hopfield_eqprop_conv_tanh_identity_20260630}"
PYTHON="${PYTHON:-python}"

cd "${SOURCE_ROOT}"
mkdir -p "${OUTPUT_ROOT}" slurm/mnist_hopf_tanh_collect

export PYTHONUNBUFFERED=1
export PYTHONPATH="${SOURCE_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

"${PYTHON}" "${SOURCE_ROOT}/experiments/collect_mnist_hopfield_eqprop_lr_screen.py" \
  --input-root "${OUTPUT_ROOT}/lr_screen_tanh_identity" \
  --summary-csv "${OUTPUT_ROOT}/lr_screen_tanh_identity_summary.csv" \
  --selected-csv "${OUTPUT_ROOT}/selected_lr_tanh_identity_by_depth.csv"
