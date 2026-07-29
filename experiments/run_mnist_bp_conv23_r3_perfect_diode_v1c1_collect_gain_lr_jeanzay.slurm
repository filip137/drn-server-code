#!/bin/bash -l
#SBATCH --job-name=mnist_c23_pd_collect
#SBATCH --output=slurm/mnist_c23_pd_collect/%x-%A.out
#SBATCH --error=slurm/mnist_c23_pd_collect/%x-%A.err
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
GRID_ROOT="${GRID_ROOT:-${FMU_SCRATCH}/server_code/results/mnist_bp_conv23_s221_pad1_perfect_diode_v1c1_gain_lr_downbias_k4k8_seed0_10epoch}"
PYTHON="${PYTHON:-python}"

cd "${SOURCE_ROOT}"
mkdir -p slurm/mnist_c23_pd_collect

echo "[conv23-pd-collect] job=${SLURM_JOB_ID:-manual} grid_root=${GRID_ROOT}"

"${PYTHON}" "${SOURCE_ROOT}/experiments/collect_mnist_bp_conv23_v1c1_perfect_diode_gain_lr_grid.py" \
  --root "${GRID_ROOT}"
