#!/usr/bin/env bash
# MNIST_CONV_LEGACY_LAUNCHER_DEPRECATED
printf "%s\n" "Expected format: python -m experiments.mnist_conv {run|sweep|collect} ..." >&2
printf "%s\n" "Provided deprecated MNIST Conv launcher invocation: $0 $*" >&2
printf "%s\n" "Deprecated MNIST Conv launcher; no setup, allocation, or work was performed." >&2
exit 2
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/filip/server_code}"
PYTHON_BIN="${PYTHON_BIN:-/home/filip/miniconda3/envs/py312/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/data}"
DEVICE="${DEVICE:-cuda}"
NUM_LANES="${NUM_LANES:-4}"
WAIT_FOR_PID="${WAIT_FOR_PID:-}"
WAIT_POLL_SECONDS="${WAIT_POLL_SECONDS:-60}"
RUN_NAMES="${RUN_NAMES:-mnist_bp_amp_v2_c1 mnist_bp_amp_v4_c1 mnist_bp_amp_v1_c2 mnist_bp_amp_v1_c4}"

SCRIPT="${SCRIPT:-${REPO_ROOT}/experiments/run_mnist_bp_conv1_hardsigmoid_amp_saturation_targets_csv_10epoch_local.sh}"
TARGET_CSV="${TARGET_CSV:-${REPO_ROOT}/results/mnist_bp_conv1_hardsigmoid_noamp_saturation_targets_voff4_seed0_10epoch_lr0p5_akib/target_input_gains.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/results/mnist_bp_conv1_hardsigmoid_amp_saturation_targets_voff4_seed0_10epoch_lr0p5_akib}"
ORCH_ROOT="${ORCH_ROOT:-${REPO_ROOT}/results/mnist_bp_conv1_hardsigmoid_amp_saturation_targets_voff4_seed0_10epoch_lr0p5_main}"

mkdir -p "${OUTPUT_ROOT}/logs" "${ORCH_ROOT}/logs"

echo "$(date) starting v_off=4.0 amp queue on main NUM_LANES=${NUM_LANES} RUN_NAMES=${RUN_NAMES}" | tee -a "${ORCH_ROOT}/logs/orchestrator.log"

pids=()
for lane in $(seq 0 $((NUM_LANES - 1))); do
  lane_log="${OUTPUT_ROOT}/logs/main_lane_${lane}.nohup.out"
  (
    export REPO_ROOT PYTHON_BIN DATASET_ROOT DEVICE RUN_NAMES
    export TARGET_CSV OUTPUT_ROOT NUM_LANES
    export LANE_ID="${lane}"
    export WAIT_FOR_PID WAIT_POLL_SECONDS
    bash "${SCRIPT}"
  ) >"${lane_log}" 2>&1 &
  pid="$!"
  pids+=("${pid}")
  echo "$(date) launched v_off=4.0 main lane=${lane} pid=${pid} log=${lane_log}" | tee -a "${ORCH_ROOT}/logs/orchestrator.log"
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    echo "$(date) lane pid=${pid} failed" | tee -a "${ORCH_ROOT}/logs/orchestrator.log" >&2
    status=1
  fi
done
if [[ "${status}" -ne 0 ]]; then
  exit "${status}"
fi
echo "$(date) v_off=4.0 main orchestrator complete" | tee -a "${ORCH_ROOT}/logs/orchestrator.log"
