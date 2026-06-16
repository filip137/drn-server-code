#!/usr/bin/env bash
set -euo pipefail

PYTHON="/home/filip/miniconda3/envs/py312/bin/python"
EVAL="/home/filip/server_code/experiments/evaluate_mnist_bp_input_quantization_sweep.py"

INPUT_ROOT="/home/filip/server_code/results/mnist_bp_amplification_sweep_drn_xs_10epoch_legacy_preproc"
OUTPUT_ROOT="/home/filip/server_code/results/mnist_bp_input_quantization_sweep_standard_best_bits4_8"
LOG_DIR="${OUTPUT_ROOT}/logs"

mkdir -p "${LOG_DIR}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="/tmp/matplotlib-mnist-input-quantization"

echo "$(date) launching MNIST BP input-quantization sweep" | tee -a "${LOG_DIR}/queue.log"

common_args=(
  "${EVAL}"
  --input-root "${INPUT_ROOT}"
  --output-root "${OUTPUT_ROOT}"
  --device cuda
  --checkpoint-kind best
  --run-name mnist_bp_amp_v1_c1
  --run-name mnist_bp_amp_v2_c1
  --run-name mnist_bp_amp_v4_c1
  --run-name mnist_bp_amp_v1_c2
  --run-name mnist_bp_amp_v1_c4
  --bits clean 8 4
  --eval-batch-size 256
  --no-download
  --num-shards 15
)

pids=()
status=0
for shard in $(seq 0 14); do
  log_path="${LOG_DIR}/input_quant_shard_${shard}.log"
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
  echo "$(date) input-quantization sweep finished successfully" | tee -a "${LOG_DIR}/queue.log"
else
  echo "$(date) input-quantization sweep finished with errors" | tee -a "${LOG_DIR}/queue.log"
fi

exit "${status}"
