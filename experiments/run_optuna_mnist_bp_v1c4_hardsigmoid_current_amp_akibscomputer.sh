#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-/home/filip/miniconda3/envs/py312/bin/python}"
SCRIPT="/home/filip/server_code/experiments/optuna_mnist_bp_v1c4_hardsigmoid_current_amp_search.py"

OUTPUT_ROOT="${OUTPUT_ROOT:-/home/filip/server_code/results/mnist_bp_v1c4_optuna_hardsigmoid_current_amp_dense_amp_fix}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-/home/filip/server_code/results/mnist_bp_amplification_sweep_hardsigmoid_voff15_current_amp_seed0_lr0p004_iter16_continue_dense_amp_fix/iter_16/lr_0p0040/mnist_bp_amp_v1_c4/seed_0/best_model.pt}"
DATASET_ROOT="${DATASET_ROOT:-/home/filip/server_code/model/resistive/data}"
STUDY_NAME="${STUDY_NAME:-mnist_bp_v1c4_hardsigmoid_current_amp_dense_fix}"
N_WORKERS="${N_WORKERS:-6}"
TRIALS_PER_WORKER="${TRIALS_PER_WORKER:-4}"
EPOCHS="${EPOCHS:-10}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda}"
LOG_DIR="${OUTPUT_ROOT}/logs"

mkdir -p "${LOG_DIR}"

export KMP_DISABLE_SHM=1
export KMP_SHM_DISABLE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-mnist-v1c4-optuna}"

if [ ! -f "${INIT_CHECKPOINT}" ]; then
  echo "Expected INIT_CHECKPOINT to exist: ${INIT_CHECKPOINT}" >&2
  exit 1
fi

echo "$(date) launching v1/c4 Optuna search" | tee -a "${LOG_DIR}/queue.log"
echo "$(date) output=${OUTPUT_ROOT} init=${INIT_CHECKPOINT} workers=${N_WORKERS} trials_per_worker=${TRIALS_PER_WORKER} epochs=${EPOCHS}" | tee -a "${LOG_DIR}/queue.log"

pids=()
status=0
for worker_idx in $(seq 0 $((N_WORKERS - 1))); do
  log_path="${LOG_DIR}/worker_${worker_idx}.log"
  echo "$(date) worker=${worker_idx} -> ${log_path}" | tee -a "${LOG_DIR}/queue.log"
  "${PYTHON}" "${SCRIPT}" \
    --output-root "${OUTPUT_ROOT}" \
    --study-name "${STUDY_NAME}" \
    --storage auto \
    --load-if-exists \
    --trials "${TRIALS_PER_WORKER}" \
    --epochs "${EPOCHS}" \
    --seed "${SEED}" \
    --sampler-seed "$((1729 + worker_idx))" \
    --worker-id "worker_${worker_idx}" \
    --device "${DEVICE}" \
    --dataset-root "${DATASET_ROOT}" \
    --init-checkpoint "${INIT_CHECKPOINT}" \
    --batch-size 4 \
    --num-iterations 16 \
    --no-download \
    --input-gain-log \
    --tune-v-off \
    >"${log_path}" 2>&1 &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

"${PYTHON}" "${SCRIPT}" \
  --output-root "${OUTPUT_ROOT}" \
  --study-name "${STUDY_NAME}" \
  --storage auto \
  --load-if-exists \
  --trials 0 \
  --epochs "${EPOCHS}" \
  --seed "${SEED}" \
  --worker-id collector \
  --device "${DEVICE}" \
  --dataset-root "${DATASET_ROOT}" \
  --init-checkpoint "${INIT_CHECKPOINT}" \
  --batch-size 4 \
  --num-iterations 16 \
  --no-download \
  --input-gain-log \
  --tune-v-off \
  >"${LOG_DIR}/collector.log" 2>&1 || status=1

if [ "${status}" -eq 0 ]; then
  echo "$(date) v1/c4 Optuna search finished successfully" | tee -a "${LOG_DIR}/queue.log"
else
  echo "$(date) v1/c4 Optuna search finished with errors" | tee -a "${LOG_DIR}/queue.log"
fi

exit "${status}"
