#!/bin/bash -l
#SBATCH --job-name=mnist_conv_fix_collr
#SBATCH --output=slurm/mnist_conv_fix_collr/%x-%j.out
#SBATCH --error=slurm/mnist_conv_fix_collr/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=00:30:00
#SBATCH --account=fmu@v100
#SBATCH --qos=qos_gpu-dev
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
RESULT_ROOT="${RESULT_ROOT:-${FMU_SCRATCH}/server_code/results/mnist_bp_conv_fixed_step_redo_r3_20260703}"
LR_ROOT="${LR_ROOT:-${RESULT_ROOT}/lr_screen_seed0_10epoch}"
SELECTED_SETTINGS_CSV="${SELECTED_SETTINGS_CSV:-${LR_ROOT}/selected_final_settings.csv}"
PYTHON="${PYTHON:-python}"

cd "${SOURCE_ROOT}"
mkdir -p "${LR_ROOT}" slurm/mnist_conv_fix_collr

export PYTHONUNBUFFERED=1
export PYTHONPATH="${SOURCE_ROOT}:${SOURCE_ROOT}/labs:${PYTHONPATH:-}"

srun "${PYTHON}" "${SOURCE_ROOT}/experiments/collect_mnist_bp_conv_lr_screen.py" \
  --root "${LR_ROOT}"

srun "${PYTHON}" "${SOURCE_ROOT}/experiments/write_mnist_bp_conv_final_settings_from_lr_selection.py" \
  --selected-lr-csv "${LR_ROOT}/selected_lr_by_nonlinearity_amp.csv" \
  --output "${SELECTED_SETTINGS_CSV}"
