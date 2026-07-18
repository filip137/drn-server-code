#!/bin/bash -l
#SBATCH --mnist-conv-legacy-launcher-is-deprecated
# MNIST_CONV_LEGACY_LAUNCHER_DEPRECATED
printf "%s\n" "Expected format: python -m experiments.mnist_conv {run|sweep|collect} ..." >&2
printf "%s\n" "Provided deprecated MNIST Conv launcher invocation: $0 $*" >&2
printf "%s\n" "Deprecated MNIST Conv launcher; no setup, allocation, or work was performed." >&2
exit 2
#SBATCH --job-name=mnist_conv2_residK
#SBATCH --output=slurm/mnist_conv2_residK/%x-%A_%a.out
#SBATCH --error=slurm/mnist_conv2_residK/%x-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --hint=nomultithread
#SBATCH --time=03:00:00
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
if [ ! -d "${SOURCE_ROOT}" ]; then
  echo "Expected SOURCE_ROOT to exist on Jean Zay: ${SOURCE_ROOT}" >&2
  exit 2
fi
cd "${SOURCE_ROOT}"

PYTHON="${PYTHON:-python}"
SCRIPT="${SOURCE_ROOT}/experiments/evaluate_mnist_bp_conv_residual_vs_iterations.py"
INPUT_ROOT="${INPUT_ROOT:-${SCRATCH}/server_code/results/mnist_bp_conv2_best_lr_50epoch_64_128ch_s2_valid_iter6_seed0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRATCH}/server_code/results/mnist_bp_conv2_residual_vs_iterations_seed0_corrected_kkt}"
NUM_SHARDS="${NUM_SHARDS:-10}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
SUMMARY_ONLY="${SUMMARY_ONLY:-0}"

if [ ! -f "${SCRIPT}" ]; then
  echo "Expected residual evaluator at ${SCRIPT}" >&2
  exit 2
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${SOURCE_ROOT}:${SOURCE_ROOT}/labs:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/mplconfig}"

mkdir -p "${OUTPUT_ROOT}" slurm/mnist_conv2_residK

read -r -a NONLINEARITY_ARGS <<< "${NON_LINEARITIES:-hard_sigmoid perfect_diode}"

common_args=(
  --input-root "${INPUT_ROOT}"
  --output-root "${OUTPUT_ROOT}"
  --model-label "${MODEL_LABEL:-conv2_bestlr50_seed0}"
  --checkpoint-kind "${CHECKPOINT_KIND:-best}"
  --device cuda
  --non-linearity "${NONLINEARITY_ARGS[@]}"
  --training-seeds 0
  --iteration-counts ${ITERATION_COUNTS:-1 2 3 4 5 6 8 10 12 16 24 32}
  --eval-batch-size "${EVAL_BATCH_SIZE:-128}"
  --max-test-samples "${MAX_TEST_SAMPLES:-1024}"
  --residual-mode "${RESIDUAL_MODE:-auto}"
  --no-download
)

if [ "${SUMMARY_ONLY}" = "1" ]; then
  "${PYTHON}" "${SCRIPT}" "${common_args[@]}" --summary-only
  exit 0
fi

echo "[jean-zay] job=${SLURM_JOB_ID:-manual} shard=${SHARD_INDEX}/${NUM_SHARDS}"
echo "[jean-zay] source_root=${SOURCE_ROOT}"
echo "[jean-zay] input_root=${INPUT_ROOT}"
echo "[jean-zay] output_root=${OUTPUT_ROOT}"
echo "[jean-zay] nonlinearities=${NONLINEARITY_ARGS[*]}"
echo "[jean-zay] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"

"${PYTHON}" "${SCRIPT}" \
  "${common_args[@]}" \
  --num-shards "${NUM_SHARDS}" \
  --shard-index "${SHARD_INDEX}" \
  --raw-results-name "raw_residual_shard_${SHARD_INDEX}.csv"
