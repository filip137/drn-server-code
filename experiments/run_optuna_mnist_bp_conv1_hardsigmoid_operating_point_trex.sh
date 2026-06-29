#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/filip/server_code}"
PYTHON_BIN="${PYTHON_BIN:-/home/filip/miniconda3/envs/py312/bin/python}"
SCRIPT="${SCRIPT:-${REPO_ROOT}/experiments/optuna_mnist_bp_conv1_hardsigmoid_operating_point.py}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/results/mnist_bp_conv1_hardsigmoid_optuna_operating_point_seed0_10epoch_trex}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/model/resistive/data}"
TRIALS_PER_WORKER="${TRIALS_PER_WORKER:-8}"
WORKERS_PER_RUN="${WORKERS_PER_RUN:-2}"
EPOCHS="${EPOCHS:-10}"
DEVICE="${DEVICE:-cuda}"
MAX_BATCHES="${MAX_BATCHES:-}"
MAX_TEST_BATCHES="${MAX_TEST_BATCHES:-}"
INPUT_GAIN_MIN="${INPUT_GAIN_MIN:-10.0}"
INPUT_GAIN_MAX="${INPUT_GAIN_MAX:-220.0}"
V_OFF_MIN="${V_OFF_MIN:-1.0}"
V_OFF_MAX="${V_OFF_MAX:-5.0}"

mkdir -p "${OUTPUT_ROOT}/logs"

launch_worker() {
  local run_name="$1"
  local worker_index="$2"
  local session="conv1_hs_optuna_${run_name#mnist_bp_amp_}_w${worker_index}"
  local worker_id="${run_name}_worker${worker_index}"
  local log_path="${OUTPUT_ROOT}/logs/${worker_id}.log"

  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "[skip] tmux session already exists: ${session}"
    return
  fi

  local cmd=(
    "${PYTHON_BIN}" "${SCRIPT}"
    --output-root "${OUTPUT_ROOT}"
    --run-name "${run_name}"
    --worker-id "${worker_id}"
    --sampler-seed "${worker_index}"
    --trials "${TRIALS_PER_WORKER}"
    --epochs "${EPOCHS}"
    --seed 0
    --device "${DEVICE}"
    --dataset-root "${DATASET_ROOT}"
    --batch-size 4
    --num-iterations 4
    --lr-decay 0.99
    --lr-min 0.003
    --lr-max 0.08
    --input-gain-min "${INPUT_GAIN_MIN}"
    --input-gain-max "${INPUT_GAIN_MAX}"
    --v-off-min "${V_OFF_MIN}"
    --v-off-max "${V_OFF_MAX}"
    --log-interval 1000
    --no-download
  )

  if [[ -n "${MAX_BATCHES}" ]]; then
    cmd+=(--max-batches "${MAX_BATCHES}")
  fi
  if [[ -n "${MAX_TEST_BATCHES}" ]]; then
    cmd+=(--max-test-batches "${MAX_TEST_BATCHES}")
  fi

  printf '[launch] %s %s\n' "${session}" "${cmd[*]}" | tee -a "${OUTPUT_ROOT}/logs/launch.log"
  tmux new-session -d -s "${session}" "cd '${REPO_ROOT}' && ${cmd[*]} > '${log_path}' 2>&1"
}

for run_name in mnist_bp_amp_v1_c1 mnist_bp_amp_v1_c2; do
  for worker_index in $(seq 0 $((WORKERS_PER_RUN - 1))); do
    launch_worker "${run_name}" "${worker_index}"
  done
done

echo "[done] launched Conv1 hard-sigmoid operating-point Optuna workers"
tmux list-sessions | grep '^conv1_hs_optuna_' || true
