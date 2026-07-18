#!/bin/bash -l
#SBATCH --mnist-conv-legacy-launcher-is-deprecated
# MNIST_CONV_LEGACY_LAUNCHER_DEPRECATED
printf "%s\n" "Expected format: python -m experiments.mnist_conv {run|sweep|collect} ..." >&2
printf "%s\n" "Provided deprecated MNIST Conv launcher invocation: $0 $*" >&2
printf "%s\n" "Deprecated MNIST Conv launcher; no setup, allocation, or work was performed." >&2
exit 2
#SBATCH --job-name=mnist_conv3_pad1_residK
#SBATCH --output=slurm/mnist_conv3_pad1_residK/%x-%A_%a.out
#SBATCH --error=slurm/mnist_conv3_pad1_residK/%x-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --hint=nomultithread
#SBATCH --time=04:00:00
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
SCRIPT="${SOURCE_ROOT}/experiments/evaluate_mnist_bp_conv_residual_vs_iterations.py"
INPUT_ROOT="${INPUT_ROOT:-${SCRATCH}/server_code/results/mnist_bp_conv3_pad1_bootstrap_k16_inputgain100_seed0_10epoch}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRATCH}/server_code/results/mnist_bp_conv3_pad1_k_preflight_seed0/residual}"
NUM_SHARDS="${NUM_SHARDS:-10}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
SUMMARY_ONLY="${SUMMARY_ONLY:-0}"

if [ ! -f "${SCRIPT}" ]; then
  echo "Expected residual evaluator at ${SCRIPT}" >&2
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

mkdir -p "${OUTPUT_ROOT}" slurm/mnist_conv3_pad1_residK

common_args=(
  --input-root "${INPUT_ROOT}"
  --output-root "${OUTPUT_ROOT}"
  --model-label "${MODEL_LABEL:-conv3_pad1_bootstrap_seed0}"
  --checkpoint-kind "${CHECKPOINT_KIND:-best}"
  --device cuda
  --non-linearity hard_sigmoid perfect_diode
  --training-seeds 0
  --iteration-counts ${ITERATION_COUNTS:-4 6 8 12 16 24 32}
  --eval-batch-size "${EVAL_BATCH_SIZE:-128}"
  --max-test-samples "${MAX_TEST_SAMPLES:-1024}"
  --residual-mode "${RESIDUAL_MODE:-auto}"
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
  --raw-results-name "raw_residual_shard_${SHARD_INDEX}.csv"
