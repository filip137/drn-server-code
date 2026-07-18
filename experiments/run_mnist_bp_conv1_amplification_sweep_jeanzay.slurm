#!/bin/bash -l
#SBATCH --mnist-conv-legacy-launcher-is-deprecated
# MNIST_CONV_LEGACY_LAUNCHER_DEPRECATED
printf "%s\n" "Expected format: python -m experiments.mnist_conv {run|sweep|collect} ..." >&2
printf "%s\n" "Provided deprecated MNIST Conv launcher invocation: $0 $*" >&2
printf "%s\n" "Deprecated MNIST Conv launcher; no setup, allocation, or work was performed." >&2
exit 2
#SBATCH --job-name=mnist_conv1_amp
#SBATCH --output=slurm/mnist_conv1_amp/%x-%A_%a.out
#SBATCH --error=slurm/mnist_conv1_amp/%x-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=08:00:00
#SBATCH --qos=qos_gpu-t3
#SBATCH --partition=gpu_p13
#SBATCH --constraint=v100-16g
#SBATCH --array=0-29%30

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

PYTHON="${PYTHON:-python}"
TRAIN="${SOURCE_ROOT}/experiments/train_mnist_bp_conv_amplification_sweep.py"

if [ ! -f "${TRAIN}" ]; then
  echo "Expected conv sweep runner at ${TRAIN}" >&2
  exit 2
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${SOURCE_ROOT}:${SOURCE_ROOT}/labs:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

DATASET_ROOT="${DATASET_ROOT:-${SCRATCH}/datasets/mnist}"
BASE_OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRATCH}/server_code/results/mnist_bp_conv1_amplification_sweep_64ch_s2_valid}"
SMOKE="${SMOKE:-0}"
SUMMARY_ONLY="${SUMMARY_ONLY:-0}"
read -r -a SEED_ARGS <<< "${SEEDS:-0 1 2}"
NUM_SHARDS_VALUE="${NUM_SHARDS:-30}"

if [ "${SMOKE}" = "1" ]; then
  EPOCHS="${EPOCHS:-1}"
  MAX_BATCHES="${MAX_BATCHES:-2}"
  MAX_TEST_BATCHES="${MAX_TEST_BATCHES:-2}"
  OUTPUT_ROOT="${SMOKE_OUTPUT_ROOT:-${BASE_OUTPUT_ROOT}_smoke}"
else
  EPOCHS="${EPOCHS:-10}"
  MAX_BATCHES="${MAX_BATCHES:-}"
  MAX_TEST_BATCHES="${MAX_TEST_BATCHES:-}"
  OUTPUT_ROOT="${BASE_OUTPUT_ROOT}"
fi

mkdir -p "${OUTPUT_ROOT}" slurm/mnist_conv1_amp

common_args=(
  --output-root "${OUTPUT_ROOT}"
  --dataset-root "${DATASET_ROOT}"
  --device cuda
  --model-key mnist_bp_conv_amp
  --seeds "${SEED_ARGS[@]}"
  --non-linearity hard_sigmoid perfect_diode
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE:-4}"
  --num-iterations "${NUM_ITERATIONS:-4}"
  --learning-rate "${LR:-0.006}"
  --beta "${BETA:-1.0}"
  --lr-decay "${LR_DECAY:-0.99}"
  --conv-depth 1
  --conv-channels 64 128 256
  --kernel-size 3
  --stride 2
  --padding 0
  --output-dim 20
  --weight-gains 1.0
  --weight-min 0.0
  --weight-max 100.0
  --weight-init-mode kaiming_uniform
  --input-gain "${INPUT_GAIN:-100.0}"
  --hard-sigmoid-g-on "${HARD_SIGMOID_G_ON:-100.0}"
  --hard-sigmoid-g-off "${HARD_SIGMOID_G_OFF:-0.0}"
  --hard-sigmoid-v-min "${HARD_SIGMOID_V_MIN:--1.5}"
  --hard-sigmoid-v-max "${HARD_SIGMOID_V_MAX:-1.5}"
  --normalize-mean 0.1307
  --normalize-std 0.3081
  --normalize-scale 0.3
  --no-download
)

if [ -n "${INIT_CHECKPOINT_ROOT:-}" ]; then
  common_args+=(--init-checkpoint-root "${INIT_CHECKPOINT_ROOT}")
fi

if [ -n "${INIT_CHECKPOINT_NAME:-}" ]; then
  common_args+=(--init-checkpoint-name "${INIT_CHECKPOINT_NAME}")
fi

if [ "${SUMMARY_ONLY}" = "1" ]; then
  "${PYTHON}" "${TRAIN}" "${common_args[@]}" --summary-only
  exit 0
fi

if [ -n "${MAX_BATCHES}" ]; then
  common_args+=(--max-batches "${MAX_BATCHES}")
fi

if [ -n "${MAX_TEST_BATCHES}" ]; then
  common_args+=(--max-test-batches "${MAX_TEST_BATCHES}")
fi

echo "[jean-zay] job=${SLURM_JOB_ID:-manual} array_task=${SLURM_ARRAY_TASK_ID:-0}"
echo "[jean-zay] output_root=${OUTPUT_ROOT}"
echo "[jean-zay] dataset_root=${DATASET_ROOT}"
echo "[jean-zay] source_root=${SOURCE_ROOT}"

srun "${PYTHON}" "${TRAIN}" \
  "${common_args[@]}" \
  --num-shards "${NUM_SHARDS_VALUE}" \
  --shard-index "${SLURM_ARRAY_TASK_ID:-0}"
