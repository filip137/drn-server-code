#!/bin/bash -l
#SBATCH --job-name=conv2_hs_ig60_cont
#SBATCH --output=slurm/conv2_hs_ig60_cont/%x-%A_%a.out
#SBATCH --error=slurm/conv2_hs_ig60_cont/%x-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=12:00:00
#SBATCH --account=umg@v100
#SBATCH --qos=qos_gpu-t3
#SBATCH --partition=gpu_p13
#SBATCH --constraint=v100-16g
#SBATCH --array=0-0

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

export SOURCE_ROOT
export PYTHON="${PYTHON:-python}"
export DATASET_ROOT="${DATASET_ROOT:-${SCRATCH}/datasets/mnist}"
export INIT_ROOT="${INIT_ROOT:-${SCRATCH}/server_code/results/mnist_bp_conv2_hardsigmoid_inputgain60_lr_sweep_seed0_50epoch}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRATCH}/server_code/results/mnist_bp_conv2_hardsigmoid_inputgain60_lr_sweep_seed0_continue${EPOCHS:-20}_from_jz_best}"
export LANE_ID="${LANE_ID:-2}"
export NUM_LANES="${NUM_LANES:-3}"
export PACK_SIZE="${PACK_SIZE:-4}"
export EPOCHS="${EPOCHS:-20}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p slurm/conv2_hs_ig60_cont "${OUTPUT_ROOT}"

echo "[jz-cont] source_root=${SOURCE_ROOT}"
echo "[jz-cont] init_root=${INIT_ROOT}"
echo "[jz-cont] output_root=${OUTPUT_ROOT}"
echo "[jz-cont] lane=${LANE_ID}/${NUM_LANES} pack=${PACK_SIZE} epochs=${EPOCHS}"

bash "${SOURCE_ROOT}/experiments/run_mnist_bp_conv2_hardsigmoid_ig60_continue_from_best_lane.sh"
