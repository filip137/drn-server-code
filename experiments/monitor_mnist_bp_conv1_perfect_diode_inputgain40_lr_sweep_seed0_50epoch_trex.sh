#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/filip/server_code}"
PYTHON_BIN="${PYTHON_BIN:-/home/filip/miniconda3/envs/py312/bin/python}"
ROOT="${ROOT:-${REPO_ROOT}/results/mnist_bp_conv1_perfect_diode_inputgain40_lr_sweep_seed0_50epoch}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-1800}"
NUM_LANES="${NUM_LANES:-5}"

count_metrics() {
  find "${ROOT}" -path '*/perfect_diode/*/seed_0/metrics.json' -type f 2>/dev/null | wc -l
}

lanes_done() {
  for lane in $(seq 0 $((NUM_LANES - 1))); do
    if tmux has-session -t "conv1_pd_ig40_50ep_lane${lane}" 2>/dev/null; then
      return 1
    fi
  done
  return 0
}

while true; do
  timestamp="$(date --iso-8601=seconds)"
  count="$(count_metrics | tr -d ' ')"
  echo "[${timestamp}] metrics=${count}/10"

  if lanes_done; then
    "${PYTHON_BIN}" "${REPO_ROOT}/experiments/collect_mnist_bp_conv1_perfect_diode_inputgain40_lr_sweep_seed0_50epoch.py" --root "${ROOT}"
    echo "[${timestamp}] monitor complete"
    exit 0
  fi

  sleep "${INTERVAL_SECONDS}"
done
