#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-/home/filip/miniconda3/envs/py312/bin/python}"
EVAL="/home/filip/server_code/experiments/evaluate_mnist_bp_stern_incidence_validation.py"

INPUT_ROOT="/home/filip/server_code/results/mnist_bp_amplification_sweep_hardsigmoid_voff15_legacy"
OUTPUT_ROOT="/home/filip/server_code/results/drn_stern_incidence_validation_hardsigmoid_voff15_iter16"
LOG_DIR="${OUTPUT_ROOT}/logs"

BATCH_SIZE="${BATCH_SIZE:-256}"
MAX_TEST_SAMPLES="${MAX_TEST_SAMPLES:-128}"
APHI_SAMPLES="${APHI_SAMPLES:-128}"
SCALING_SAMPLES="${SCALING_SAMPLES:-64}"
SWRITE_SAMPLES="${SWRITE_SAMPLES:-128}"
SAME_COORDINATE_SAMPLES="${SAME_COORDINATE_SAMPLES:-128}"
NUM_DIRECTIONS="${NUM_DIRECTIONS:-8}"

mkdir -p "${LOG_DIR}"

export KMP_DISABLE_SHM=1
export KMP_SHM_DISABLE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-mnist-stern-incidence-validation}"

echo "$(date) launching Stern incidence validation" | tee -a "${LOG_DIR}/queue.log"
echo "$(date) samples=${MAX_TEST_SAMPLES} directions=${NUM_DIRECTIONS}" | tee -a "${LOG_DIR}/queue.log"

pids=()
status=0
runs=(mnist_bp_amp_v1_c1 mnist_bp_amp_v2_c1 mnist_bp_amp_v4_c1 mnist_bp_amp_v1_c2 mnist_bp_amp_v1_c4)

for run_name in "${runs[@]}"; do
  for seed in 0 1 2; do
    log_path="${LOG_DIR}/${run_name}_seed_${seed}.log"
    echo "$(date) ${run_name} seed=${seed} -> ${log_path}" | tee -a "${LOG_DIR}/queue.log"
    "${PYTHON}" "${EVAL}" \
      --input-root "${INPUT_ROOT}" \
      --output-root "${OUTPUT_ROOT}" \
      --device cuda \
      --checkpoint best \
      --run-name "${run_name}" \
      --seeds "${seed}" \
      --batch-size "${BATCH_SIZE}" \
      --max-test-samples "${MAX_TEST_SAMPLES}" \
      --aphi-samples "${APHI_SAMPLES}" \
      --scaling-samples "${SCALING_SAMPLES}" \
      --swrite-samples "${SWRITE_SAMPLES}" \
      --same-coordinate-samples "${SAME_COORDINATE_SAMPLES}" \
      --num-directions "${NUM_DIRECTIONS}" \
      --inference-iterations 16 \
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
  --summary-only >"${LOG_DIR}/summary.log" 2>&1 || status=1

if [ "${status}" -eq 0 ]; then
  echo "$(date) Stern incidence validation finished successfully" | tee -a "${LOG_DIR}/queue.log"
else
  echo "$(date) Stern incidence validation finished with errors" | tee -a "${LOG_DIR}/queue.log"
fi

exit "${status}"
