#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/filip/server_code}"
PYTHON_BIN="${PYTHON_BIN:-/home/filip/miniconda3/envs/py312/bin/python}"
ROOT="${ROOT:-${REPO_ROOT}/results/mnist_bp_conv1_inputgain40_selected_lr_hardsigmoid_voff15_seed0_50epoch}"
REMOTE="${REMOTE:-filip@trex}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/filip/server_code/results/mnist_bp_conv1_inputgain40_selected_lr_hardsigmoid_voff15_seed0_50epoch}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-1800}"

REMOTE_LRS=(0p012 0p036 0p048)

count_local() {
  find "${ROOT}" -path '*/hard_sigmoid/*/seed_0/metrics.json' -type f 2>/dev/null | wc -l
}

count_remote() {
  ssh "${REMOTE}" "find '${REMOTE_ROOT}' -path '*/hard_sigmoid/*/seed_0/metrics.json' -type f 2>/dev/null | wc -l" || printf '0\n'
}

remote_sessions_done() {
  for lane in 1 3 4; do
    if ssh "${REMOTE}" "tmux has-session -t conv1_ig40_50ep_lane${lane} 2>/dev/null"; then
      return 1
    fi
  done
  return 0
}

local_sessions_done() {
  for lane in 0 2; do
    if tmux has-session -t "conv1_ig40_50ep_lane${lane}" 2>/dev/null; then
      return 1
    fi
  done
  return 0
}

copy_remote_results() {
  mkdir -p "${ROOT}/lane_logs"
  for lr_label in "${REMOTE_LRS[@]}"; do
    scp -r "${REMOTE}:${REMOTE_ROOT}/lr_${lr_label}" "${ROOT}/"
  done
  scp "${REMOTE}:${REMOTE_ROOT}/lane_logs/trex_lane"*.log "${ROOT}/lane_logs/" || true
}

while true; do
  timestamp="$(date --iso-8601=seconds)"
  local_count="$(count_local | tr -d ' ')"
  remote_count="$(count_remote | tr -d ' ')"
  echo "[${timestamp}] local_metrics=${local_count}/5 remote_metrics=${remote_count}/3"

  if local_sessions_done && remote_sessions_done; then
    echo "[${timestamp}] lanes finished; copying trex results"
    copy_remote_results
    "${PYTHON_BIN}" "${REPO_ROOT}/experiments/collect_mnist_bp_conv1_inputgain40_selected_lr_50epoch.py" --root "${ROOT}"
    echo "[${timestamp}] monitor complete"
    exit 0
  fi

  sleep "${INTERVAL_SECONDS}"
done
