#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-/home/filip/miniconda3/envs/py312/bin/python}"
EVAL="/home/filip/server_code/experiments/evaluate_mnist_bp_physical_cost_sharpness.py"

INPUT_ROOT="${INPUT_ROOT:-/home/filip/server_code/results/mnist_bp_amplification_sweep_hardsigmoid_voff15_legacy}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/filip/server_code/results/mnist_bp_physical_cost_sharpness_hardsigmoid_voff15_iter16}"
LOG_DIR="${OUTPUT_ROOT}/logs"
BATCH_SIZE="${BATCH_SIZE:-512}"
INFERENCE_ITERATIONS="${INFERENCE_ITERATIONS:-16}"
NUM_CLASSES="${NUM_CLASSES:-10}"

run_names=(
  mnist_bp_amp_v1_c1
  mnist_bp_amp_v2_c1
  mnist_bp_amp_v4_c1
  mnist_bp_amp_v1_c2
  mnist_bp_amp_v1_c4
)
seeds=(0 1 2)

mkdir -p "${LOG_DIR}"

export KMP_DISABLE_SHM=1
export KMP_SHM_DISABLE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-mnist-physical-cost-sharpness}"

echo "$(date) launching physical-cost sharpness jobs, batch=${BATCH_SIZE}, inference_iterations=${INFERENCE_ITERATIONS}" | tee -a "${LOG_DIR}/queue.log"

pids=()
status=0
for run_name in "${run_names[@]}"; do
  for seed in "${seeds[@]}"; do
    log_path="${LOG_DIR}/${run_name}_seed_${seed}.log"
    echo "$(date) ${run_name} seed=${seed} -> ${log_path}" | tee -a "${LOG_DIR}/queue.log"
    "${PYTHON}" "${EVAL}" \
      --input-root "${INPUT_ROOT}" \
      --output-root "${OUTPUT_ROOT}" \
      --checkpoint best \
      --device cuda \
      --batch-size "${BATCH_SIZE}" \
      --inference-iterations "${INFERENCE_ITERATIONS}" \
      --num-classes "${NUM_CLASSES}" \
      --run-name "${run_name}" \
      --seeds "${seed}" \
      --no-download \
      --no-summary-after-run \
      >"${log_path}" 2>&1 &
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
  --checkpoint best \
  --inference-iterations "${INFERENCE_ITERATIONS}" \
  --summary-only >"${LOG_DIR}/summary.log" 2>&1 || status=1

if [ "${status}" -eq 0 ]; then
  echo "$(date) physical-cost sharpness jobs finished successfully" | tee -a "${LOG_DIR}/queue.log"
else
  echo "$(date) physical-cost sharpness jobs finished with errors" | tee -a "${LOG_DIR}/queue.log"
fi

exit "${status}"
