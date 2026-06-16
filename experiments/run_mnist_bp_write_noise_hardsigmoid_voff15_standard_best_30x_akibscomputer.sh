#!/usr/bin/env bash
set -euo pipefail

PYTHON="/home/filip/miniconda3/envs/py312/bin/python"
EVAL="/home/filip/server_code/experiments/evaluate_mnist_bp_write_noise_sweep.py"

INPUT_ROOT="/home/filip/server_code/results/mnist_bp_amplification_sweep_hardsigmoid_voff15_legacy"
OUTPUT_ROOT="/home/filip/server_code/results/mnist_bp_write_noise_sweep_hardsigmoid_voff15_standard_best_30x"
LOG_DIR="${OUTPUT_ROOT}/logs"

mkdir -p "${LOG_DIR}"

export KMP_DISABLE_SHM=1
export KMP_SHM_DISABLE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="/tmp/matplotlib-mnist-write-noise-hardsigmoid-voff15"

echo "$(date) launching hard-sigmoid MNIST BP write-noise sweep, best checkpoints, standard 30x grid" | tee -a "${LOG_DIR}/queue.log"

common_args=(
  "${EVAL}"
  --input-root "${INPUT_ROOT}"
  --output-root "${OUTPUT_ROOT}"
  --device cuda
  --checkpoint-kind best
  --sigmas 0 0.005 0.01 0.02 0.05 0.1 0.15 0.2
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
  echo "$(date) hard-sigmoid write-noise sweep finished successfully" | tee -a "${LOG_DIR}/queue.log"
else
  echo "$(date) hard-sigmoid write-noise sweep finished with errors" | tee -a "${LOG_DIR}/queue.log"
fi

exit "${status}"
