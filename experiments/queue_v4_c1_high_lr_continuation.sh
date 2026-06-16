#!/usr/bin/env bash
set -euo pipefail

BASE_ROOT="/home/filip/server_code/results/mnist_bp_amplification_sweep_drn_xs_10epoch_legacy_preproc"
OUT_ROOT="/home/filip/server_code/results/mnist_bp_amp_v4_c1_continue_lr0p012_10epoch_from_legacy_preproc"
LOG_DIR="${OUT_ROOT}/logs"
PYTHON="/home/filip/miniconda3/envs/py312/bin/python"
RUNNER="/home/filip/server_code/experiments/train_mnist_bp_amplification_sweep.py"

mkdir -p "${LOG_DIR}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "$(date) waiting for current sweep to finish" | tee -a "${LOG_DIR}/queue.log"
while true; do
  metrics_count=$(find "${BASE_ROOT}" -maxdepth 3 -name metrics.json | wc -l)
  echo "$(date) base metrics=${metrics_count}/15" | tee -a "${LOG_DIR}/queue.log"
  if [ "${metrics_count}" -ge 15 ]; then
    break
  fi
  sleep 60
done

echo "$(date) launching v4/c1 high-LR continuation" | tee -a "${LOG_DIR}/queue.log"

common_args=(
  "${RUNNER}"
  --device cuda
  --output-root "${OUT_ROOT}"
  --run-name mnist_bp_amp_v4_c1
  --epochs 10
  --batch-size 4
  --num-iterations 4
  --input-gain 100
  --learning-rate 0.012 0.012 0.012
  --lr-decay 0.99
  --beta 1.0
  --weight-gains 1 1
  --weight-min 0.0
  --weight-max 100.0
  --weight-init-mode kaiming_uniform
  --normalize-mean 0.1307
  --normalize-std 0.3081
  --normalize-scale 0.3
  --init-checkpoint-template "${BASE_ROOT}/{run_name}/seed_{seed}/final_model.pt"
  --no-download
)

pids=()
for seed in 0 1 2; do
  log_path="${LOG_DIR}/mnist_bp_amp_v4_c1_seed_${seed}.log"
  echo "$(date) seed ${seed} -> ${log_path}" | tee -a "${LOG_DIR}/queue.log"
  "${PYTHON}" "${common_args[@]}" --seeds "${seed}" >"${log_path}" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

"${PYTHON}" "${RUNNER}" \
  --output-root "${OUT_ROOT}" \
  --run-name mnist_bp_amp_v4_c1 \
  --seeds 0 1 2 \
  --summary-only >"${LOG_DIR}/summary.log" 2>&1 || status=1

if [ "${status}" -eq 0 ]; then
  echo "$(date) v4/c1 high-LR continuation finished successfully" | tee -a "${LOG_DIR}/queue.log"
else
  echo "$(date) v4/c1 high-LR continuation finished with errors" | tee -a "${LOG_DIR}/queue.log"
fi

exit "${status}"
