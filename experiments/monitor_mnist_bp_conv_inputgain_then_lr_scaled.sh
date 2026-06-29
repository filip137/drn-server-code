#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/filip/server_code}"
PYTHON_BIN="${PYTHON_BIN:-/home/filip/miniconda3/envs/py312/bin/python}"
REMOTE_HOST="${REMOTE_HOST:-filip@trex}"
REMOTE_REPO_ROOT="${REMOTE_REPO_ROOT:-/home/filip/server_code}"

CURRENT_ROOT="${CURRENT_ROOT:-${REPO_ROOT}/results/mnist_bp_conv_inputgain_train_sweep_hardsigmoid_voff15_seed0_5epoch}"
REMOTE_CURRENT_ROOT="${REMOTE_CURRENT_ROOT:-${CURRENT_ROOT}}"
SCALED_ROOT="${SCALED_ROOT:-${REPO_ROOT}/results/mnist_bp_conv_inputgain_train_sweep_hardsigmoid_voff15_seed0_5epoch_lr_scaled}"
REMOTE_SCALED_ROOT="${REMOTE_SCALED_ROOT:-${SCALED_ROOT}}"

LOCAL_DATASET_ROOT="${LOCAL_DATASET_ROOT:-/home/filip/datasets/mnist}"
REMOTE_DATASET_ROOT="${REMOTE_DATASET_ROOT:-/home/filip/server_code/model/resistive/data}"
DEVICE="${DEVICE:-cuda}"
EXPECTED_JOBS="${EXPECTED_JOBS:-45}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-3600}"
INITIAL_SLEEP_SECONDS="${INITIAL_SLEEP_SECONDS:-0}"
MAX_CHECKS="${MAX_CHECKS:-0}"
NUM_LANES="${NUM_LANES:-6}"
LOCAL_LANES="${LOCAL_LANES:-0 2 4}"
REMOTE_LANES="${REMOTE_LANES:-1 3 5}"
BASE_GAIN="${BASE_GAIN:-10}"
BASE_LR="${BASE_LR:-0.006}"

STATE_DIR="${CURRENT_ROOT}/monitor_state"
LOG_DIR="${CURRENT_ROOT}/monitor_logs"
CURRENT_COLLECTED_MARKER="${STATE_DIR}/current_collected.ok"
SCALED_LAUNCHED_MARKER="${STATE_DIR}/scaled_launched.ok"
SCALED_COLLECTED_MARKER="${STATE_DIR}/scaled_collected.ok"
MONITOR_LOG="${LOG_DIR}/hourly_checker.log"

mkdir -p "${STATE_DIR}" "${LOG_DIR}" "${SCALED_ROOT}/lane_logs"

log() {
  local message="$1"
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "${message}" | tee -a "${MONITOR_LOG}"
}

count_local() {
  local root="$1"
  find "${root}" -path '*seed_0/metrics.json' 2>/dev/null | wc -l
}

count_remote() {
  local root="$1"
  ssh "${REMOTE_HOST}" "find '${root}' -path '*seed_0/metrics.json' 2>/dev/null | wc -l" 2>/dev/null || printf '0\n'
}

copy_remote_tree() {
  local remote_root="$1"
  local local_root="$2"
  mkdir -p "${local_root}/conv1" "${local_root}/conv2" "${local_root}/lane_logs"
  if ssh "${REMOTE_HOST}" "test -d '${remote_root}/conv1'" 2>/dev/null; then
    scp -r "${REMOTE_HOST}:${remote_root}/conv1/"* "${local_root}/conv1/" >/dev/null 2>&1 || true
  fi
  if ssh "${REMOTE_HOST}" "test -d '${remote_root}/conv2'" 2>/dev/null; then
    scp -r "${REMOTE_HOST}:${remote_root}/conv2/"* "${local_root}/conv2/" >/dev/null 2>&1 || true
  fi
  if ssh "${REMOTE_HOST}" "test -d '${remote_root}/lane_logs'" 2>/dev/null; then
    scp -r "${REMOTE_HOST}:${remote_root}/lane_logs/"* "${local_root}/lane_logs/" >/dev/null 2>&1 || true
  fi
}

collect_current() {
  log "copying current sweep outputs from ${REMOTE_HOST}"
  copy_remote_tree "${REMOTE_CURRENT_ROOT}" "${CURRENT_ROOT}"
  log "collecting current fixed-LR sweep"
  "${PYTHON_BIN}" "${REPO_ROOT}/experiments/collect_mnist_bp_conv_inputgain_train_sweep.py" \
    --root "${CURRENT_ROOT}" | tee -a "${MONITOR_LOG}"
  touch "${CURRENT_COLLECTED_MARKER}"
}

collect_scaled() {
  log "copying scaled-LR sweep outputs from ${REMOTE_HOST}"
  copy_remote_tree "${REMOTE_SCALED_ROOT}" "${SCALED_ROOT}"
  log "collecting scaled-LR sweep"
  "${PYTHON_BIN}" "${REPO_ROOT}/experiments/collect_mnist_bp_conv_inputgain_train_sweep.py" \
    --root "${SCALED_ROOT}" \
    --lr-mode gain_scaled \
    --base-lr "${BASE_LR}" \
    --base-gain "${BASE_GAIN}" \
    --path-includes-lr | tee -a "${MONITOR_LOG}"
  touch "${SCALED_COLLECTED_MARKER}"
}

stage_remote_scripts() {
  log "staging scaled-LR launcher to ${REMOTE_HOST}"
  scp "${REPO_ROOT}/experiments/run_mnist_bp_conv_inputgain_train_sweep_hardsigmoid_voff15_seed0_5epoch_lr_scaled_local_lane.sh" \
    "${REMOTE_HOST}:${REMOTE_REPO_ROOT}/experiments/" >/dev/null
}

launch_local_lane() {
  local lane="$1"
  local session="conv_ig_lrs_lane${lane}"
  local log_path="${SCALED_ROOT}/lane_logs/main_lane${lane}.log"
  if tmux has-session -t "${session}" 2>/dev/null; then
    log "local tmux session ${session} already exists; not relaunching"
    return
  fi
  tmux new-session -d -s "${session}" \
    "cd '${REPO_ROOT}' && OUTPUT_ROOT='${SCALED_ROOT}' DATASET_ROOT='${LOCAL_DATASET_ROOT}' DEVICE='${DEVICE}' BASE_GAIN='${BASE_GAIN}' BASE_LR='${BASE_LR}' '${REPO_ROOT}/experiments/run_mnist_bp_conv_inputgain_train_sweep_hardsigmoid_voff15_seed0_5epoch_lr_scaled_local_lane.sh' '${lane}' '${NUM_LANES}' > '${log_path}' 2>&1"
  log "launched local scaled-LR lane ${lane} in tmux ${session}"
}

launch_remote_lane() {
  local lane="$1"
  local session="conv_ig_lrs_lane${lane}"
  local log_path="${REMOTE_SCALED_ROOT}/lane_logs/trex_lane${lane}.log"
  ssh "${REMOTE_HOST}" "mkdir -p '${REMOTE_SCALED_ROOT}/lane_logs'"
  ssh "${REMOTE_HOST}" "tmux has-session -t '${session}' 2>/dev/null || tmux new-session -d -s '${session}' \"cd '${REMOTE_REPO_ROOT}' && OUTPUT_ROOT='${REMOTE_SCALED_ROOT}' DATASET_ROOT='${REMOTE_DATASET_ROOT}' DEVICE='${DEVICE}' BASE_GAIN='${BASE_GAIN}' BASE_LR='${BASE_LR}' '${REMOTE_REPO_ROOT}/experiments/run_mnist_bp_conv_inputgain_train_sweep_hardsigmoid_voff15_seed0_5epoch_lr_scaled_local_lane.sh' '${lane}' '${NUM_LANES}' > '${log_path}' 2>&1\""
  log "ensured remote scaled-LR lane ${lane} in tmux ${session}"
}

launch_scaled_sweep() {
  if [[ -f "${SCALED_LAUNCHED_MARKER}" ]]; then
    log "scaled-LR sweep already launched; not relaunching"
    return
  fi
  mkdir -p "${SCALED_ROOT}/lane_logs"
  stage_remote_scripts
  for lane in ${LOCAL_LANES}; do
    launch_local_lane "${lane}"
  done
  for lane in ${REMOTE_LANES}; do
    launch_remote_lane "${lane}"
  done
  touch "${SCALED_LAUNCHED_MARKER}"
  log "scaled-LR sweep launched with base_lr=${BASE_LR} at base_gain=${BASE_GAIN}"
}

check_sweep() {
  local name="$1"
  local local_root="$2"
  local remote_root="$3"
  local local_count
  local remote_count
  local total_count
  local_count="$(count_local "${local_root}")"
  remote_count="$(count_remote "${remote_root}")"
  total_count=$((local_count + remote_count))
  log "${name}: local=${local_count} remote=${remote_count} total=${total_count}/${EXPECTED_JOBS}"
  [[ "${total_count}" -ge "${EXPECTED_JOBS}" ]]
}

if [[ "${INITIAL_SLEEP_SECONDS}" != "0" ]]; then
  log "initial sleep ${INITIAL_SLEEP_SECONDS}s"
  sleep "${INITIAL_SLEEP_SECONDS}"
fi

check_index=0
while true; do
  check_index=$((check_index + 1))
  if [[ "${MAX_CHECKS}" != "0" && "${check_index}" -gt "${MAX_CHECKS}" ]]; then
    log "max checks reached (${MAX_CHECKS}); exiting"
    exit 0
  fi

  if [[ ! -f "${CURRENT_COLLECTED_MARKER}" ]]; then
    if check_sweep "fixed-LR input-gain sweep" "${CURRENT_ROOT}" "${REMOTE_CURRENT_ROOT}"; then
      collect_current
      launch_scaled_sweep
    fi
  else
    launch_scaled_sweep
  fi

  if [[ -f "${SCALED_LAUNCHED_MARKER}" && ! -f "${SCALED_COLLECTED_MARKER}" ]]; then
    if check_sweep "scaled-LR input-gain sweep" "${SCALED_ROOT}" "${REMOTE_SCALED_ROOT}"; then
      collect_scaled
      log "scaled-LR sweep complete; monitor exiting"
      exit 0
    fi
  elif [[ -f "${SCALED_COLLECTED_MARKER}" ]]; then
    log "scaled-LR sweep already collected; monitor exiting"
    exit 0
  fi

  log "sleeping ${CHECK_INTERVAL_SECONDS}s before next check"
  sleep "${CHECK_INTERVAL_SECONDS}"
done
