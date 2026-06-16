#!/usr/bin/env bash
set -euo pipefail

PYTHON="/home/filip/miniconda3/envs/py312/bin/python"
EVAL="/home/filip/server_code/experiments/evaluate_mnist_bp_directional_hessian_response.py"

INPUT_ROOT="/home/filip/server_code/results/mnist_bp_hessian_lpw_amplification_sweep_legacy_drnxs"
OUTPUT_ROOT="/home/filip/server_code/results/mnist_bp_hessian_directional_response_legacy_drnxs"
LOG_DIR="${OUTPUT_ROOT}/logs"

mkdir -p "${LOG_DIR}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="/tmp/matplotlib-mnist-bp-directional-hessian-legacy"

echo "$(date) launching MNIST BP directional Hessian response analysis" | tee -a "${LOG_DIR}/queue.log"

run_names=(
  mnist_bp_amp_v1_c1
  mnist_bp_amp_v2_c1
  mnist_bp_amp_v4_c1
  mnist_bp_amp_v1_c2
  mnist_bp_amp_v1_c4
)
seeds=(0 1 2)

common_args=(
  --input-root "${INPUT_ROOT}"
  --output-root "${OUTPUT_ROOT}"
  --device cuda
  --batch-size 256
  --no-download
)

pids=()
status=0

for run_name in "${run_names[@]}"; do
  for seed in "${seeds[@]}"; do
    log_path="${LOG_DIR}/${run_name}_seed_${seed}.log"
    echo "$(date) run=${run_name} seed=${seed} -> ${log_path}" | tee -a "${LOG_DIR}/queue.log"
    "${PYTHON}" "${EVAL}" \
      "${common_args[@]}" \
      --run-name "${run_name}" \
      --seeds "${seed}" \
      --no-summary-after-run >"${log_path}" 2>&1 &
    pids+=("$!")
  done
done

for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

"${PYTHON}" "${EVAL}" \
  --input-root "${INPUT_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --seeds "${seeds[@]}" \
  --summary-only >"${LOG_DIR}/summary.log" 2>&1 || status=1

if [ "${status}" -eq 0 ]; then
  echo "$(date) MNIST BP directional Hessian response analysis finished successfully" | tee -a "${LOG_DIR}/queue.log"
else
  echo "$(date) MNIST BP directional Hessian response analysis finished with errors" | tee -a "${LOG_DIR}/queue.log"
fi

exit "${status}"
