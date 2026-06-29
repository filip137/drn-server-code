#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/filip/server_code}"
PYTHON_BIN="${PYTHON_BIN:-/home/filip/miniconda3/envs/py312/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/results/mnist_bp_conv2_hardsigmoid_inputgain60_k6_lr_protocol_seed0_10epoch}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/model/resistive/data}"
NUM_LANES="${NUM_LANES:-4}"
WAIT_FOR_CONV1="${WAIT_FOR_CONV1:-1}"
POLL_SECONDS="${POLL_SECONDS:-60}"

RUNNER="${REPO_ROOT}/experiments/run_mnist_bp_conv2_hardsigmoid_inputgain60_k6_lr_protocol_10epoch_lane.sh"
COLLECTOR="${REPO_ROOT}/experiments/collect_mnist_bp_conv2_hardsigmoid_inputgain60_k6_lr_protocol_10epoch.py"

if [[ ! -f "${RUNNER}" ]]; then
  echo "Expected runner at ${RUNNER}" >&2
  exit 2
fi
if [[ ! -f "${COLLECTOR}" ]]; then
  echo "Expected collector at ${COLLECTOR}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}/lane_logs"

conv1_active() {
  local lane
  for lane in 0 1 2 3; do
    if tmux has-session -t "conv1_hs_ig40_highlr50_l${lane}" 2>/dev/null; then
      return 0
    fi
  done
  return 1
}

if [[ "${WAIT_FOR_CONV1}" == "1" ]]; then
  while conv1_active; do
    echo "$(date) waiting for Conv1 high-LR Trex sessions to finish before launching Conv2 K=6 protocol"
    sleep "${POLL_SECONDS}"
  done
fi

for lane in $(seq 0 $((NUM_LANES - 1))); do
  session="conv2_hs_ig60_k6_lrproto_l${lane}"
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "$(date) session already exists: ${session}"
    continue
  fi
  tmux new-session -d -s "${session}" \
    "cd '${REPO_ROOT}' && REPO_ROOT='${REPO_ROOT}' PYTHON_BIN='${PYTHON_BIN}' OUTPUT_ROOT='${OUTPUT_ROOT}' DATASET_ROOT='${DATASET_ROOT}' bash '${RUNNER}' '${lane}' '${NUM_LANES}' > '${OUTPUT_ROOT}/lane_logs/lane_${lane}.out' 2>&1"
  echo "$(date) launched ${session}"
done

while true; do
  active=0
  for lane in $(seq 0 $((NUM_LANES - 1))); do
    if tmux has-session -t "conv2_hs_ig60_k6_lrproto_l${lane}" 2>/dev/null; then
      active=1
    fi
  done
  if [[ "${active}" == "0" ]]; then
    break
  fi
  echo "$(date) waiting for Conv2 K=6 protocol lanes to finish"
  sleep "${POLL_SECONDS}"
done

"${PYTHON_BIN}" "${COLLECTOR}" --root "${OUTPUT_ROOT}" > "${OUTPUT_ROOT}/collector.out" 2>&1
echo "$(date) collected Conv2 K=6 protocol results at ${OUTPUT_ROOT}"
