#!/bin/bash -l
#SBATCH --mnist-conv-legacy-launcher-is-deprecated
# MNIST_CONV_LEGACY_LAUNCHER_DEPRECATED
printf "%s\n" "Expected format: python -m experiments.mnist_conv {run|sweep|collect} ..." >&2
printf "%s\n" "Provided deprecated MNIST Conv launcher invocation: $0 $*" >&2
printf "%s\n" "Deprecated MNIST Conv launcher; no setup, allocation, or work was performed." >&2
exit 2
#SBATCH --job-name=mnist_conv_cost_hess
#SBATCH --output=slurm/mnist_conv_cost_hess/%x-%A_%a.out
#SBATCH --error=slurm/mnist_conv_cost_hess/%x-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --hint=nomultithread
#SBATCH --time=06:00:00
#SBATCH --account=umg@v100
#SBATCH --qos=qos_gpu-t3
#SBATCH --partition=gpu_p13
#SBATCH --constraint=v100-16g
#SBATCH --array=0-0%1

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
SCRIPT="${SOURCE_ROOT}/experiments/evaluate_mnist_bp_cost_hessian_noise_curvature.py"
INPUT_ROOT="${INPUT_ROOT:?Set INPUT_ROOT to a completed conv amplification sweep root.}"
OUTPUT_ROOT="${OUTPUT_ROOT:?Set OUTPUT_ROOT for cost-Hessian outputs.}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
SUMMARY_ONLY="${SUMMARY_ONLY:-0}"

if [ ! -f "${SCRIPT}" ]; then
  echo "Expected cost Hessian evaluator at ${SCRIPT}" >&2
  exit 2
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${SOURCE_ROOT}:${SOURCE_ROOT}/labs:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/mplconfig}"

mkdir -p "${OUTPUT_ROOT}" slurm/mnist_conv_cost_hess

common_args=(
  --input-root "${INPUT_ROOT}"
  --output-root "${OUTPUT_ROOT}"
  --noise-root ""
  --checkpoint-kind "${CHECKPOINT_KIND:-best}"
  --device cuda
  --eval-batch-size "${EVAL_BATCH_SIZE:-512}"
  --max-eval-batches "${MAX_EVAL_BATCHES:-20}"
  --num-directions "${NUM_DIRECTIONS:-8}"
  --epsilons ${EPSILONS:-0.02}
  --direction-seed-offset "${DIRECTION_SEED_OFFSET:-24680}"
  --clamp-perturbed
  --layerwise
  --no-download
)

if [ -n "${INFERENCE_ITERATIONS:-}" ]; then
  common_args+=(--inference-iterations "${INFERENCE_ITERATIONS}")
fi

if [ "${INCLUDE_GRADIENT_DIRECTION:-0}" = "1" ]; then
  common_args+=(--include-gradient-direction)
  if [ -n "${GRADIENT_EPSILONS:-}" ]; then
    common_args+=(--gradient-epsilons ${GRADIENT_EPSILONS})
  fi
fi

if [ "${SUMMARY_ONLY}" = "1" ]; then
  "${PYTHON}" "${SCRIPT}" "${common_args[@]}" --summary-only
  exit 0
fi

echo "[jean-zay] job=${SLURM_JOB_ID:-manual} shard=${SHARD_INDEX}/${NUM_SHARDS}"
echo "[jean-zay] source_root=${SOURCE_ROOT}"
echo "[jean-zay] input_root=${INPUT_ROOT}"
echo "[jean-zay] output_root=${OUTPUT_ROOT}"
echo "[jean-zay] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"

"${PYTHON}" "${SCRIPT}" \
  "${common_args[@]}" \
  --num-shards "${NUM_SHARDS}" \
  --shard-index "${SHARD_INDEX}" \
  --raw-results-name "raw_curvature_shard_${SHARD_INDEX}.csv" \
  --gradient-raw-results-name "raw_gradient_curvature_shard_${SHARD_INDEX}.csv"
