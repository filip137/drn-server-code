#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-/home/filip/miniconda3/envs/py312/bin/python}"
EVAL="/home/filip/server_code/experiments/evaluate_mnist_bp_cost_hessian_noise_curvature.py"

INPUT_ROOT="${INPUT_ROOT:-/home/filip/server_code/results/mnist_bp_amplification_sweep_hardsigmoid_voff15_legacy}"
NOISE_ROOT="${NOISE_ROOT:-/home/filip/server_code/results/mnist_bp_write_noise_sweep_hardsigmoid_voff15_sigma_response_iter16_30x}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/filip/server_code/results/mnist_bp_cost_hessian_noise_curvature_hardsigmoid_voff15_iter16}"
LOG_DIR="${OUTPUT_ROOT}/logs"

EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1024}"
NUM_DIRECTIONS="${NUM_DIRECTIONS:-16}"
EPSILON="${EPSILON:-0.02}"
NUM_SHARDS="${NUM_SHARDS:-15}"

mkdir -p "${LOG_DIR}"

export KMP_DISABLE_SHM=1
export KMP_SHM_DISABLE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-mnist-cost-hessian-noise-curvature}"

echo "$(date) launching MNIST BP cost-Hessian/noise-curvature sweep" | tee -a "${LOG_DIR}/queue.log"
echo "$(date) batch=${EVAL_BATCH_SIZE} directions=${NUM_DIRECTIONS} epsilon=${EPSILON}" | tee -a "${LOG_DIR}/queue.log"

common_args=(
  "${EVAL}"
  --input-root "${INPUT_ROOT}"
  --output-root "${OUTPUT_ROOT}"
  --noise-root "${NOISE_ROOT}"
  --device cuda
  --checkpoint-kind best
  --eval-batch-size "${EVAL_BATCH_SIZE}"
  --inference-iterations 16
  --num-directions "${NUM_DIRECTIONS}"
  --epsilons "${EPSILON}"
  --layerwise
  --no-download
  --num-shards "${NUM_SHARDS}"
)

pids=()
status=0
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  log_path="${LOG_DIR}/cost_hessian_shard_${shard}.log"
  echo "$(date) shard ${shard}/${NUM_SHARDS} -> ${log_path}" | tee -a "${LOG_DIR}/queue.log"
  "${PYTHON}" "${common_args[@]}" \
    --shard-index "${shard}" \
    --raw-results-name "raw_curvature_shard_${shard}.csv" \
    >"${log_path}" 2>&1 &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

"${PYTHON}" "${EVAL}" \
  --output-root "${OUTPUT_ROOT}" \
  --noise-root "${NOISE_ROOT}" \
  --summary-only >"${LOG_DIR}/summary.log" 2>&1 || status=1

if [ "${status}" -eq 0 ]; then
  echo "$(date) MNIST BP cost-Hessian/noise-curvature sweep finished successfully" | tee -a "${LOG_DIR}/queue.log"
else
  echo "$(date) MNIST BP cost-Hessian/noise-curvature sweep finished with errors" | tee -a "${LOG_DIR}/queue.log"
fi

exit "${status}"
