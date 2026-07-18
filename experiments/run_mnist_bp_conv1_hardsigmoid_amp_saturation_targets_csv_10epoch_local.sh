#!/usr/bin/env bash
# MNIST_CONV_LEGACY_LAUNCHER_DEPRECATED
printf "%s\n" "Expected format: python -m experiments.mnist_conv {run|sweep|collect} ..." >&2
printf "%s\n" "Provided deprecated MNIST Conv launcher invocation: $0 $*" >&2
printf "%s\n" "Deprecated MNIST Conv launcher; no setup, allocation, or work was performed." >&2
exit 2
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/filip/server_code}"
PYTHON_BIN="${PYTHON_BIN:-/home/filip/miniconda3/envs/py312/bin/python}"
TARGET_CSV="${TARGET_CSV:?TARGET_CSV must point to a target_input_gains.csv file}"
OUTPUT_ROOT="${OUTPUT_ROOT:?OUTPUT_ROOT must be set}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/data}"
DEVICE="${DEVICE:-cuda}"
GPU_UTIL_THRESHOLD="${GPU_UTIL_THRESHOLD:-98}"
GPU_MEM_THRESHOLD_MB="${GPU_MEM_THRESHOLD_MB:-9300}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-120}"
WAIT_FOR_PID="${WAIT_FOR_PID:-}"
WAIT_POLL_SECONDS="${WAIT_POLL_SECONDS:-60}"
NUM_LANES="${NUM_LANES:-1}"
LANE_ID="${LANE_ID:-0}"
RUN_NAMES="${RUN_NAMES:-mnist_bp_amp_v2_c1 mnist_bp_amp_v4_c1 mnist_bp_amp_v1_c2 mnist_bp_amp_v1_c4}"

mkdir -p "${OUTPUT_ROOT}/logs"

if [[ "${LANE_ID}" -lt 0 || "${LANE_ID}" -ge "${NUM_LANES}" ]]; then
  echo "expected 0 <= LANE_ID < NUM_LANES, got LANE_ID=${LANE_ID} NUM_LANES=${NUM_LANES}" >&2
  exit 2
fi

if [[ -n "${WAIT_FOR_PID}" ]]; then
  while ps -p "${WAIT_FOR_PID}" >/dev/null 2>&1; do
    {
      date
      echo "lane=${LANE_ID}/${NUM_LANES} waiting for pid ${WAIT_FOR_PID}"
    } | tee -a "${OUTPUT_ROOT}/logs/lane_${LANE_ID}.queue.log"
    sleep "${WAIT_POLL_SECONDS}"
  done
fi

wait_for_gpu() {
  if [[ "${DEVICE}" != cuda* ]]; then
    return
  fi

  while true; do
    local util mem
    util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d ' ')"
    mem="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')"
    {
      date
      echo "lane=${LANE_ID}/${NUM_LANES} gpu util=${util} mem=${mem}MiB threshold_util=${GPU_UTIL_THRESHOLD} threshold_mem=${GPU_MEM_THRESHOLD_MB}MiB"
    } | tee -a "${OUTPUT_ROOT}/logs/lane_${LANE_ID}.queue.log"

    if [[ "${util}" -lt "${GPU_UTIL_THRESHOLD}" && "${mem}" -lt "${GPU_MEM_THRESHOLD_MB}" ]]; then
      break
    fi
    sleep "${GPU_POLL_SECONDS}"
  done
}

label_float() {
  local value="$1"
  value="${value//$'\r'/}"
  value="${value//./p}"
  value="${value//-/m}"
  printf '%s' "${value}"
}

target_label() {
  local target="$1"
  awk -v t="${target}" 'BEGIN { printf "sat%.0f", t * 100 }'
}

run_job() {
  local target="$1"
  local input_gain="$2"
  local learning_rate="$3"
  local v_off="$4"
  local run_name="$5"
  local target_label_value gain_label lr_label job_root v_min run_log

  target="${target//$'\r'/}"
  input_gain="${input_gain//$'\r'/}"
  learning_rate="${learning_rate//$'\r'/}"
  v_off="${v_off//$'\r'/}"
  run_name="${run_name//$'\r'/}"
  target_label_value="$(target_label "${target}")"
  gain_label="$(label_float "${input_gain}")"
  lr_label="$(label_float "${learning_rate}")"
  job_root="${OUTPUT_ROOT}/target_${target_label_value}/input_gain_${gain_label}/lr_${lr_label}"
  run_log="${OUTPUT_ROOT}/logs/target_${target_label_value}_${run_name}.log"
  v_min="-${v_off}"

  {
    date
    echo "START lane=${LANE_ID}/${NUM_LANES} target=${target} input_gain=${input_gain} lr=${learning_rate} v_off=${v_off} run_name=${run_name} job_root=${job_root}"
  } | tee -a "${OUTPUT_ROOT}/logs/lane_${LANE_ID}.queue.log"
  wait_for_gpu

  "${PYTHON_BIN}" "${REPO_ROOT}/experiments/train_mnist_bp_conv_amplification_sweep.py" \
    --output-root "${job_root}" \
    --dataset-root "${DATASET_ROOT}" \
    --device "${DEVICE}" \
    --model-key mnist_bp_conv_amp \
    --non-linearity hard_sigmoid \
    --seeds 0 \
    --epochs 10 \
    --batch-size 4 \
    --beta 1.0 \
    --lr-decay 0.99 \
    --conv-depth 1 \
    --conv-channels 64 \
    --kernel-size 3 \
    --stride 2 \
    --padding 0 \
    --output-dim 20 \
    --weight-gains 1.0 \
    --weight-min 0.0 \
    --weight-max 100.0 \
    --weight-init-mode kaiming_uniform \
    --input-gain "${input_gain}" \
    --num-iterations 4 \
    --learning-rate "${learning_rate}" \
    --hard-sigmoid-g-on 100.0 \
    --hard-sigmoid-g-off 0.0 \
    --hard-sigmoid-v-min "${v_min}" \
    --hard-sigmoid-v-max "${v_off}" \
    --normalize-mean 0.1307 \
    --normalize-std 0.3081 \
    --normalize-scale 0.3 \
    --log-interval 500 \
    --no-download \
    --run-name "${run_name}" \
    2>&1 | tee "${run_log}"

  {
    date
    echo "DONE lane=${LANE_ID}/${NUM_LANES} target=${target} input_gain=${input_gain} lr=${learning_rate} v_off=${v_off} run_name=${run_name}"
  } | tee -a "${OUTPUT_ROOT}/logs/lane_${LANE_ID}.queue.log"
}

job_index=0
while IFS=, read -r target input_gain _measured learning_rate v_off _lr_num _seed _num_samples _num_iterations; do
  [[ -n "${target}" ]] || continue
  for run_name in ${RUN_NAMES}; do
    if (( job_index % NUM_LANES == LANE_ID )); then
      run_job "${target}" "${input_gain}" "${learning_rate}" "${v_off}" "${run_name}"
    fi
    job_index=$((job_index + 1))
  done
done < <(tail -n +2 "${TARGET_CSV}")
