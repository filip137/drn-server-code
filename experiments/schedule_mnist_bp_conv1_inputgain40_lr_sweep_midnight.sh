#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/filip/server_code}"
PYTHON_BIN="${PYTHON_BIN:-/home/filip/miniconda3/envs/py312/bin/python}"
REMOTE_HOST="${REMOTE_HOST:-filip@trex}"
REMOTE_REPO_ROOT="${REMOTE_REPO_ROOT:-/home/filip/server_code}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/results/mnist_bp_conv1_inputgain40_lr_sweep_hardsigmoid_voff15_seed0_10epoch}"
REMOTE_OUTPUT_ROOT="${REMOTE_OUTPUT_ROOT:-${OUTPUT_ROOT}}"
LOCAL_DATASET_ROOT="${LOCAL_DATASET_ROOT:-/home/filip/datasets/mnist}"
REMOTE_DATASET_ROOT="${REMOTE_DATASET_ROOT:-/home/filip/server_code/model/resistive/data}"
DEVICE="${DEVICE:-cuda}"
TARGET_TIME="${TARGET_TIME:-2026-06-18 00:00:00 CEST}"
EXPECTED_JOBS="${EXPECTED_JOBS:-29}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-1800}"
NUM_LANES="${NUM_LANES:-6}"
LOCAL_LANES="${LOCAL_LANES:-0 2 4}"
REMOTE_LANES="${REMOTE_LANES:-1 3 5}"

LOG_DIR="${OUTPUT_ROOT}/scheduler_logs"
LANE_LOG_DIR="${OUTPUT_ROOT}/lane_logs"
LOG_PATH="${LOG_DIR}/midnight_scheduler.log"
mkdir -p "${LOG_DIR}" "${LANE_LOG_DIR}"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$1" | tee -a "${LOG_PATH}"
}

target_epoch() {
  date -d "${TARGET_TIME}" +%s
}

sleep_until_target() {
  local target
  local now
  local delay
  target="$(target_epoch)"
  now="$(date +%s)"
  delay=$((target - now))
  if (( delay > 0 )); then
    log "sleeping ${delay}s until ${TARGET_TIME}"
    sleep "${delay}"
  else
    log "target time ${TARGET_TIME} is in the past; launching immediately"
  fi
}

count_local() {
  find "${OUTPUT_ROOT}" -path '*seed_0/metrics.json' 2>/dev/null | wc -l
}

count_remote() {
  ssh "${REMOTE_HOST}" "find '${REMOTE_OUTPUT_ROOT}' -path '*seed_0/metrics.json' 2>/dev/null | wc -l" 2>/dev/null || printf '0\n'
}

copy_remote_outputs() {
  mkdir -p "${OUTPUT_ROOT}"
  for lr_dir in 0p003 0p006 0p009 0p012 0p018 0p024 0p036 0p048; do
    if ssh "${REMOTE_HOST}" "test -d '${REMOTE_OUTPUT_ROOT}/lr_${lr_dir}'" 2>/dev/null; then
      mkdir -p "${OUTPUT_ROOT}/lr_${lr_dir}"
      scp -r "${REMOTE_HOST}:${REMOTE_OUTPUT_ROOT}/lr_${lr_dir}/"* "${OUTPUT_ROOT}/lr_${lr_dir}/" >/dev/null 2>&1 || true
    fi
  done
  if ssh "${REMOTE_HOST}" "test -d '${REMOTE_OUTPUT_ROOT}/lane_logs'" 2>/dev/null; then
    mkdir -p "${OUTPUT_ROOT}/lane_logs"
    scp -r "${REMOTE_HOST}:${REMOTE_OUTPUT_ROOT}/lane_logs/"* "${OUTPUT_ROOT}/lane_logs/" >/dev/null 2>&1 || true
  fi
}

stage_remote_scripts() {
  log "staging focused launcher to ${REMOTE_HOST}"
  scp "${REPO_ROOT}/experiments/run_mnist_bp_conv1_inputgain40_lr_sweep_hardsigmoid_voff15_seed0_10epoch_local_lane.sh" \
    "${REMOTE_HOST}:${REMOTE_REPO_ROOT}/experiments/" >/dev/null
}

launch_local_lane() {
  local lane="$1"
  local session="conv1_ig40_lr_lane${lane}"
  local lane_log="${LANE_LOG_DIR}/main_lane${lane}.log"
  tmux kill-session -t "${session}" 2>/dev/null || true
  tmux new-session -d -s "${session}" \
    "cd '${REPO_ROOT}' && OUTPUT_ROOT='${OUTPUT_ROOT}' DATASET_ROOT='${LOCAL_DATASET_ROOT}' DEVICE='${DEVICE}' '${REPO_ROOT}/experiments/run_mnist_bp_conv1_inputgain40_lr_sweep_hardsigmoid_voff15_seed0_10epoch_local_lane.sh' '${lane}' '${NUM_LANES}' > '${lane_log}' 2>&1"
  log "launched local lane ${lane} in tmux ${session}"
}

launch_remote_lane() {
  local lane="$1"
  local session="conv1_ig40_lr_lane${lane}"
  local lane_log="${REMOTE_OUTPUT_ROOT}/lane_logs/trex_lane${lane}.log"
  ssh "${REMOTE_HOST}" "mkdir -p '${REMOTE_OUTPUT_ROOT}/lane_logs'"
  ssh "${REMOTE_HOST}" "tmux kill-session -t '${session}' 2>/dev/null || true; tmux new-session -d -s '${session}' \"cd '${REMOTE_REPO_ROOT}' && OUTPUT_ROOT='${REMOTE_OUTPUT_ROOT}' DATASET_ROOT='${REMOTE_DATASET_ROOT}' DEVICE='${DEVICE}' '${REMOTE_REPO_ROOT}/experiments/run_mnist_bp_conv1_inputgain40_lr_sweep_hardsigmoid_voff15_seed0_10epoch_local_lane.sh' '${lane}' '${NUM_LANES}' > '${lane_log}' 2>&1\""
  log "launched remote lane ${lane} in tmux ${session}"
}

launch_lanes() {
  stage_remote_scripts
  for lane in ${LOCAL_LANES}; do
    launch_local_lane "${lane}"
  done
  for lane in ${REMOTE_LANES}; do
    launch_remote_lane "${lane}"
  done
}

collect_when_complete() {
  while true; do
    local local_count
    local remote_count
    local total_count
    local_count="$(count_local)"
    remote_count="$(count_remote)"
    total_count=$((local_count + remote_count))
    log "focused LR sweep progress: local=${local_count} remote=${remote_count} total=${total_count}/${EXPECTED_JOBS}"
    if (( total_count >= EXPECTED_JOBS )); then
      log "copying remote outputs"
      copy_remote_outputs
      log "collecting focused LR sweep"
      "${PYTHON_BIN}" "${REPO_ROOT}/experiments/collect_mnist_bp_conv1_inputgain40_lr_sweep.py" \
        --root "${OUTPUT_ROOT}" | tee -a "${LOG_PATH}"
      log "focused LR sweep collection complete"
      return
    fi
    log "sleeping ${CHECK_INTERVAL_SECONDS}s before next focused LR check"
    sleep "${CHECK_INTERVAL_SECONDS}"
  done
}

log "midnight focused Conv1 gain40 LR scheduler starting"
sleep_until_target
launch_lanes
collect_when_complete
