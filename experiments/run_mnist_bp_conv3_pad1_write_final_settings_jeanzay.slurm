#!/bin/bash -l
#SBATCH --job-name=mnist_conv3_pad1_settings
#SBATCH --output=slurm/mnist_conv3_pad1_settings/%x-%j.out
#SBATCH --error=slurm/mnist_conv3_pad1_settings/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --hint=nomultithread
#SBATCH --time=00:20:00
#SBATCH --account=umg@v100
#SBATCH --qos=qos_gpu-t3
#SBATCH --partition=gpu_p13
#SBATCH --constraint=v100-16g

set -euo pipefail

if command -v idrenv >/dev/null 2>&1; then
  eval "$(idrenv)"
fi
if [ -z "${WORK:-}" ]; then
  WORK="/lustre/fswork/projects/rech/${IDRPROJ:-umg}/${USER}"
fi
if [ -z "${SCRATCH:-}" ]; then
  SCRATCH="/lustre/fsn1/projects/rech/${IDRPROJ:-umg}/${USER}"
fi

module purge
module load "${PYTORCH_MODULE:-pytorch-gpu/py3/2.5.0}"

SOURCE_ROOT="${SOURCE_ROOT:-${WORK}/server_code}"
cd "${SOURCE_ROOT}"

PYTHON="${PYTHON:-python}"
SCRIPT="${SOURCE_ROOT}/experiments/write_mnist_bp_conv_final_settings_from_lr_selection.py"
SELECTED_LR_CSV="${SELECTED_LR_CSV:-${SCRATCH}/server_code/results/mnist_bp_conv3_pad1_long_lr_check_seed0_50epoch/selected_lr_by_nonlinearity_amp.csv}"
OUTPUT="${OUTPUT:-${SCRATCH}/server_code/results/mnist_bp_conv3_pad1_final_settings.csv}"

"${PYTHON}" "${SCRIPT}" \
  --selected-lr-csv "${SELECTED_LR_CSV}" \
  --output "${OUTPUT}"
