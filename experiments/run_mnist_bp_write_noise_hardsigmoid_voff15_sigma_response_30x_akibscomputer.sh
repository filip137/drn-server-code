#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-/home/filip/miniconda3/envs/py312/bin/python}"
EVAL="/home/filip/server_code/experiments/evaluate_mnist_bp_write_noise_sweep.py"
PLOT="/home/filip/server_code/experiments/plot_mnist_bp_write_noise_sigma_response.py"

INPUT_ROOT="${INPUT_ROOT:-/home/filip/server_code/results/mnist_bp_amplification_sweep_hardsigmoid_voff15_legacy}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/filip/server_code/results/mnist_bp_write_noise_sweep_hardsigmoid_voff15_sigma_response_30x}"
LOG_DIR="${OUTPUT_ROOT}/logs"

mkdir -p "${LOG_DIR}"

export KMP_DISABLE_SHM=1
export KMP_SHM_DISABLE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="/tmp/matplotlib-mnist-write-noise-hardsigmoid-voff15-sigma-response"

echo "$(date) launching hard-sigmoid MNIST BP write-noise sigma-response sweep" | tee -a "${LOG_DIR}/queue.log"

common_args=(
  "${EVAL}"
  --input-root "${INPUT_ROOT}"
  --output-root "${OUTPUT_ROOT}"
  --device cuda
  --checkpoint-kind best
  --sigmas 0 0.025 0.05 0.075 0.1 0.125 0.15 0.175 0.2 0.25 0.3 0.4 0.5 0.75 1.0
  --noise-seeds
)

for seed in $(seq 0 29); do
  common_args+=("${seed}")
done

common_args+=(
  --eval-batch-size 256
  --no-download
  --num-shards 15
)

pids=()
status=0
for shard in $(seq 0 14); do
  log_path="${LOG_DIR}/write_noise_sigma_response_shard_${shard}.log"
  echo "$(date) shard ${shard}/15 -> ${log_path}" | tee -a "${LOG_DIR}/queue.log"
  "${PYTHON}" "${common_args[@]}" \
    --shard-index "${shard}" \
    --raw-results-name "raw_results_shard_${shard}.csv" \
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
  --summary-only >"${LOG_DIR}/summary.log" 2>&1 || status=1

"${PYTHON}" "${PLOT}" \
  --input-root "${OUTPUT_ROOT}" >"${LOG_DIR}/plot_sigma_response.log" 2>&1 || status=1

if [ "${status}" -eq 0 ]; then
  echo "$(date) hard-sigmoid write-noise sigma-response sweep finished successfully" | tee -a "${LOG_DIR}/queue.log"
else
  echo "$(date) hard-sigmoid write-noise sigma-response sweep finished with errors" | tee -a "${LOG_DIR}/queue.log"
fi

exit "${status}"
