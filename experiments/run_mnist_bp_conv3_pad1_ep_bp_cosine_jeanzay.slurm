#!/bin/bash -l
#SBATCH --job-name=mnist_conv3_pad1_epbpK
#SBATCH --output=slurm/mnist_conv3_pad1_epbpK/%x-%A_%a.out
#SBATCH --error=slurm/mnist_conv3_pad1_epbpK/%x-%A_%a.err
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
cd "${SOURCE_ROOT}"

PYTHON="${PYTHON:-python}"
SCRIPT="${SOURCE_ROOT}/experiments/evaluate_mnist_bp_conv_ep_bp_cosine_vs_iterations.py"
INPUT_ROOT="${INPUT_ROOT:-${SCRATCH}/server_code/results/mnist_bp_conv3_pad1_bootstrap_k16_inputgain100_seed0_10epoch}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRATCH}/server_code/results/mnist_bp_conv3_pad1_k_preflight_seed0/epbp}"
NUM_SHARDS="${NUM_SHARDS:-10}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
SUMMARY_ONLY="${SUMMARY_ONLY:-0}"

if [ ! -f "${SCRIPT}" ]; then
  echo "Expected EP/BP evaluator at ${SCRIPT}" >&2
  exit 2
fi
if [ ! -d "${INPUT_ROOT}" ]; then
  echo "Expected INPUT_ROOT to exist: ${INPUT_ROOT}" >&2
  exit 2
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${SOURCE_ROOT}:${SOURCE_ROOT}/labs:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/mplconfig}"

mkdir -p "${OUTPUT_ROOT}" slurm/mnist_conv3_pad1_epbpK

common_args=(
  --input-root "${INPUT_ROOT}"
  --output-root "${OUTPUT_ROOT}"
  --model-label "${MODEL_LABEL:-conv3_pad1_bootstrap_seed0}"
  --split "${SPLIT:-train}"
  --checkpoint-kind "${CHECKPOINT_KIND:-best}"
  --device cuda
  --non-linearity hard_sigmoid perfect_diode
  --training-seeds 0
  --iteration-counts ${ITERATION_COUNTS:-4 6 8 12 16 24 32}
  --betas ${BETAS:-0.1 0.25 0.5 1.0}
  --primary-beta "${PRIMARY_BETA:-0.25}"
  --plateau-cosine-tolerance "${PLATEAU_COSINE_TOLERANCE:-0.01}"
  --batch-size "${BATCH_SIZE:-4}"
  --max-batches "${MAX_BATCHES:-32}"
  --no-download
)

if [ "${SUMMARY_ONLY}" = "1" ]; then
  "${PYTHON}" "${SCRIPT}" "${common_args[@]}" --summary-only
  exit 0
fi

"${PYTHON}" "${SCRIPT}" \
  "${common_args[@]}" \
  --num-shards "${NUM_SHARDS}" \
  --shard-index "${SHARD_INDEX}" \
  --raw-results-name "raw_cosine_shard_${SHARD_INDEX}.csv"
