#!/bin/bash -l
#SBATCH --mnist-conv-legacy-launcher-is-deprecated
# MNIST_CONV_LEGACY_LAUNCHER_DEPRECATED
printf "%s\n" "Expected format: python -m experiments.mnist_conv {run|sweep|collect} ..." >&2
printf "%s\n" "Provided deprecated MNIST Conv launcher invocation: $0 $*" >&2
printf "%s\n" "Deprecated MNIST Conv launcher; no setup, allocation, or work was performed." >&2
exit 2
#SBATCH --job-name=mnist_conv_sat35_45
#SBATCH --output=slurm/mnist_conv_sat35_45/%x-%A_%a.out
#SBATCH --error=slurm/mnist_conv_sat35_45/%x-%A_%a.err
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
#SBATCH --array=0-3%4

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
if [ ! -d "${SOURCE_ROOT}" ]; then
  echo "Expected SOURCE_ROOT to exist on Jean Zay: ${SOURCE_ROOT}" >&2
  exit 2
fi
cd "${SOURCE_ROOT}"

QUEUE="${SOURCE_ROOT}/experiments/run_mnist_bp_conv_hardsigmoid_noamp_saturation_targets_10epoch_lr2_queue.sh"
SEARCH="${SOURCE_ROOT}/experiments/search_mnist_bp_conv_hardsigmoid_saturation_targets.py"
TRAIN="${SOURCE_ROOT}/experiments/train_mnist_bp_conv_amplification_sweep.py"
for path in "${QUEUE}" "${SEARCH}" "${TRAIN}"; do
  if [ ! -f "${path}" ]; then
    echo "Expected required experiment file at ${path}" >&2
    exit 2
  fi
done

export PYTHONUNBUFFERED=1
export PYTHONPATH="${SOURCE_ROOT}:${SOURCE_ROOT}/labs:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

task_id="${SLURM_ARRAY_TASK_ID:-0}"
case "${task_id}" in
  0)
    conv_depth=2
    padding=0
    num_iterations=6
    v_off=3.5
    ;;
  1)
    conv_depth=2
    padding=0
    num_iterations=6
    v_off=4.5
    ;;
  2)
    conv_depth=3
    padding=1
    num_iterations=16
    v_off=3.5
    ;;
  3)
    conv_depth=3
    padding=1
    num_iterations=16
    v_off=4.5
    ;;
  *)
    echo "Invalid SLURM_ARRAY_TASK_ID=${task_id}; expected 0..3." >&2
    exit 2
    ;;
esac

DATASET_ROOT="${DATASET_ROOT:-${SCRATCH}/datasets/mnist}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRATCH}/server_code/results/mnist_bp_conv${conv_depth}_hardsigmoid_noamp_saturation_targets_seed0_10epoch_lr2_voff35_45_firsthidden_jeanzay}"

mkdir -p "${OUTPUT_ROOT}" slurm/mnist_conv_sat35_45

echo "[jean-zay] job=${SLURM_JOB_ID:-manual} array_task=${task_id}"
echo "[jean-zay] conv_depth=${conv_depth} padding=${padding} num_iterations=${num_iterations} v_off=${v_off}"
echo "[jean-zay] source_root=${SOURCE_ROOT}"
echo "[jean-zay] dataset_root=${DATASET_ROOT}"
echo "[jean-zay] output_root=${OUTPUT_ROOT}"
echo "[jean-zay] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"

export REPO_ROOT="${SOURCE_ROOT}"
export PYTHON_BIN="${PYTHON:-python}"
export CONV_DEPTH="${conv_depth}"
export PADDING="${padding}"
export NUM_ITERATIONS="${num_iterations}"
export OUTPUT_ROOT
export DATASET_ROOT
export DEVICE=cuda
export V_OFFS="${v_off}"
export LR_NUMERATOR="${LR_NUMERATOR:-1.44}"
export SEARCH_STEPS="${SEARCH_STEPS:-24}"
export SEARCH_SAMPLES="${SEARCH_SAMPLES:-256}"
export SATURATION_SCOPE=first_hidden
export GPU_MEM_THRESHOLD_MB="${GPU_MEM_THRESHOLD_MB:-12000}"
export GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-30}"

srun bash "${QUEUE}"
