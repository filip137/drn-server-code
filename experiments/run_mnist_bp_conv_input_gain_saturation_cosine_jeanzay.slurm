#!/bin/bash -l
#SBATCH --job-name=mnist_conv_igcos
#SBATCH --output=slurm/mnist_conv_igcos/%x-%A_%a.out
#SBATCH --error=slurm/mnist_conv_igcos/%x-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --hint=nomultithread
#SBATCH --time=08:00:00
#SBATCH --account=umg@v100
#SBATCH --qos=qos_gpu-t3
#SBATCH --partition=gpu_p13
#SBATCH --constraint=v100-16g
#SBATCH --array=0-9%10

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
SCRIPT="${SOURCE_ROOT}/experiments/evaluate_mnist_bp_conv_input_gain_saturation_cosine.py"
INPUT_ROOT_CONV1="${INPUT_ROOT_CONV1:-${SCRATCH}/server_code/results/mnist_bp_conv1_amplification_sweep_64ch_s2_valid}"
INPUT_ROOT_CONV2="${INPUT_ROOT_CONV2:-${SCRATCH}/server_code/results/mnist_bp_conv2_best_lr_50epoch_64_128ch_s2_valid_iter6_seed0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRATCH}/server_code/results/mnist_bp_conv_input_gain_saturation_cosine_hardsigmoid}"
NUM_SHARDS="${NUM_SHARDS:-10}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
SUMMARY_ONLY="${SUMMARY_ONLY:-0}"

if [ ! -d "${SOURCE_ROOT}" ]; then
  echo "Expected SOURCE_ROOT to exist: ${SOURCE_ROOT}" >&2
  exit 2
fi
if [ ! -f "${SCRIPT}" ]; then
  echo "Expected evaluator to exist: ${SCRIPT}" >&2
  exit 2
fi
if [ ! -d "${INPUT_ROOT_CONV1}" ]; then
  echo "Expected INPUT_ROOT_CONV1 to exist: ${INPUT_ROOT_CONV1}" >&2
  exit 2
fi
if [ ! -d "${INPUT_ROOT_CONV2}" ]; then
  echo "Expected INPUT_ROOT_CONV2 to exist: ${INPUT_ROOT_CONV2}" >&2
  exit 2
fi

cd "${SOURCE_ROOT}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${SOURCE_ROOT}:${SOURCE_ROOT}/labs:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/mplconfig}"

mkdir -p "${OUTPUT_ROOT}" slurm/mnist_conv_igcos

common_args=(
  --input-root-conv1 "${INPUT_ROOT_CONV1}"
  --input-root-conv2 "${INPUT_ROOT_CONV2}"
  --output-root "${OUTPUT_ROOT}"
  --model-label "${MODEL_LABEL:-conv_hardsigmoid_input_gain}"
  --split "${SPLIT:-train}"
  --checkpoint-kind "${CHECKPOINT_KIND:-best}"
  --device cuda
  --conv-depth 1 2
  --non-linearity hard_sigmoid
  --training-seeds 0
  --input-gains ${INPUT_GAINS:-25 50 75 100}
  --iteration-counts-conv1 ${ITERATION_COUNTS_CONV1:-4 6 8 12 16}
  --iteration-counts-conv2 ${ITERATION_COUNTS_CONV2:-4 6 8 12 16 24 32}
  --betas ${BETAS:-0.1 0.25 0.5 1.0}
  --primary-beta "${PRIMARY_BETA:-0.25}"
  --batch-size "${BATCH_SIZE:-4}"
  --max-batches "${MAX_BATCHES:-32}"
  --selection-accuracy-tolerance "${SELECTION_ACCURACY_TOLERANCE:-0.02}"
  --selection-max-layer0-outside "${SELECTION_MAX_LAYER0_OUTSIDE:-0.60}"
  --no-download
)

if [ "${SUMMARY_ONLY}" = "1" ]; then
  python "${SCRIPT}" "${common_args[@]}" --summary-only
  exit 0
fi

echo "[jean-zay] job=${SLURM_JOB_ID:-manual} shard=${SHARD_INDEX}/${NUM_SHARDS}"
echo "[jean-zay] source_root=${SOURCE_ROOT}"
echo "[jean-zay] conv1=${INPUT_ROOT_CONV1}"
echo "[jean-zay] conv2=${INPUT_ROOT_CONV2}"
echo "[jean-zay] output_root=${OUTPUT_ROOT}"
echo "[jean-zay] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"

python "${SCRIPT}" \
  "${common_args[@]}" \
  --num-shards "${NUM_SHARDS}" \
  --shard-index "${SHARD_INDEX}"
