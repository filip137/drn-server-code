#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/filip/server_code}"
PYTHON_BIN="${PYTHON_BIN:-/home/filiposana/miniconda3/envs/py312/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/data}"
DEVICE="${DEVICE:-cuda}"
NUM_LANES="${NUM_LANES:-4}"
WAIT_FOR_PID="${WAIT_FOR_PID:-}"
WAIT_POLL_SECONDS="${WAIT_POLL_SECONDS:-60}"
RUN_NAMES="${RUN_NAMES:-mnist_bp_amp_v2_c1 mnist_bp_amp_v4_c1 mnist_bp_amp_v1_c2 mnist_bp_amp_v1_c4}"

SCRIPT="${SCRIPT:-${REPO_ROOT}/experiments/run_mnist_bp_conv1_hardsigmoid_amp_saturation_targets_csv_10epoch_local.sh}"
CSV_35="${CSV_35:-${REPO_ROOT}/results/mnist_bp_conv1_hardsigmoid_noamp_saturation_targets_voff3p5_seed0_10epoch_lr0p5_akib/target_input_gains.csv}"
CSV_40="${CSV_40:-${REPO_ROOT}/results/mnist_bp_conv1_hardsigmoid_noamp_saturation_targets_voff4_seed0_10epoch_lr0p5_akib/target_input_gains.csv}"
OUT_35="${OUT_35:-${REPO_ROOT}/results/mnist_bp_conv1_hardsigmoid_amp_saturation_targets_voff3p5_seed0_10epoch_lr0p5_akib}"
OUT_40="${OUT_40:-${REPO_ROOT}/results/mnist_bp_conv1_hardsigmoid_amp_saturation_targets_voff4_seed0_10epoch_lr0p5_akib}"
ORCH_ROOT="${ORCH_ROOT:-${REPO_ROOT}/results/mnist_bp_conv1_hardsigmoid_amp_saturation_targets_voff35_40_seed0_10epoch_lr0p5_akib}"

mkdir -p "${OUT_35}/logs" "${OUT_40}/logs" "${ORCH_ROOT}/logs"

echo "$(date) starting v_off=3.5/4.0 amp queue NUM_LANES=${NUM_LANES} RUN_NAMES=${RUN_NAMES}" | tee -a "${ORCH_ROOT}/logs/orchestrator.log"

pids=()
for lane in $(seq 0 $((NUM_LANES - 1))); do
  lane_log="${OUT_35}/logs/lane_${lane}.nohup.out"
  (
    export REPO_ROOT PYTHON_BIN DATASET_ROOT DEVICE RUN_NAMES
    export TARGET_CSV="${CSV_35}"
    export OUTPUT_ROOT="${OUT_35}"
    export NUM_LANES LANE_ID="${lane}"
    export WAIT_FOR_PID WAIT_POLL_SECONDS
    bash "${SCRIPT}"
  ) >"${lane_log}" 2>&1 &
  pid="$!"
  pids+=("${pid}")
  echo "$(date) launched v_off=3.5 lane=${lane} pid=${pid} log=${lane_log}" | tee -a "${ORCH_ROOT}/logs/orchestrator.log"
done

for lane in $(seq 0 $((NUM_LANES - 1))); do
  lane_log="${OUT_40}/logs/lane_${lane}.nohup.out"
  (
    export REPO_ROOT PYTHON_BIN DATASET_ROOT DEVICE RUN_NAMES
    export TARGET_CSV="${CSV_40}"
    export OUTPUT_ROOT="${OUT_40}"
    export NUM_LANES LANE_ID="${lane}"
    export WAIT_FOR_PID="${pids[$lane]}"
    export WAIT_POLL_SECONDS
    bash "${SCRIPT}"
  ) >"${lane_log}" 2>&1 &
  pid="$!"
  echo "$(date) launched v_off=4.0 lane=${lane} pid=${pid} waits_for=${pids[$lane]} log=${lane_log}" | tee -a "${ORCH_ROOT}/logs/orchestrator.log"
done

wait
echo "$(date) orchestrator complete" | tee -a "${ORCH_ROOT}/logs/orchestrator.log"
