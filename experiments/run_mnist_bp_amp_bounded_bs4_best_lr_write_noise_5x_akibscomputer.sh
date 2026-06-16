#!/usr/bin/env bash
set -euo pipefail

PYTHON="/home/filip/miniconda3/envs/py312/bin/python"
EVAL="/home/filip/server_code/experiments/evaluate_mnist_bp_write_noise_sweep.py"

INPUT_ROOT="/home/filip/server_code/results/mnist_bp_amp_bounded_bs4_best_lr"
OUTPUT_ROOT="/home/filip/server_code/results/mnist_bp_amp_bounded_bs4_best_lr_write_noise_5x"
LOG_DIR="${OUTPUT_ROOT}/logs"

mkdir -p "${LOG_DIR}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="/tmp/matplotlib-mnist-amp-bounded-bs4-best-lr-write-noise-5x"

echo "$(date) launching MNIST BP amplification write-noise sweep on bounded bs4 best-LR checkpoints" | tee -a "${LOG_DIR}/queue.log"

common_args=(
  "${EVAL}"
  --input-root "${INPUT_ROOT}"
  --output-root "${OUTPUT_ROOT}"
  --device cuda
  --checkpoint-kind best
  --sigmas 0 0.005 0.01 0.02 0.05 0.1 0.2 0.5 1.0 1.5 2.0
  --noise-seeds 0 1 2 3 4
  --eval-batch-size 256
  --no-download
  --num-shards 15
)

pids=()
status=0
for shard in $(seq 0 14); do
  log_path="${LOG_DIR}/write_noise_shard_${shard}.log"
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

if [ "${status}" -eq 0 ]; then
  echo "$(date) MNIST BP amplification write-noise sweep finished successfully" | tee -a "${LOG_DIR}/queue.log"
else
  echo "$(date) MNIST BP amplification write-noise sweep finished with errors" | tee -a "${LOG_DIR}/queue.log"
fi

exit "${status}"
