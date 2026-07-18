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
GPU_UTIL_THRESHOLD="${GPU_UTIL_THRESHOLD:-35}"
GPU_MEM_THRESHOLD_MB="${GPU_MEM_THRESHOLD_MB:-6000}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-300}"
WAIT_FOR_PID="${WAIT_FOR_PID:-}"
WAIT_POLL_SECONDS="${WAIT_POLL_SECONDS:-60}"

mkdir -p "${OUTPUT_ROOT}/logs"

if [[ -n "${WAIT_FOR_PID}" ]]; then
  while ps -p "${WAIT_FOR_PID}" >/dev/null 2>&1; do
    {
      date
      echo "waiting for pid ${WAIT_FOR_PID} before starting CSV saturation queue"
    } | tee -a "${OUTPUT_ROOT}/logs/queue.log"
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
      echo "gpu util=${util} mem=${mem}MiB threshold_util=${GPU_UTIL_THRESHOLD} threshold_mem=${GPU_MEM_THRESHOLD_MB}MiB"
    } | tee -a "${OUTPUT_ROOT}/logs/queue.log"

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
  local target_label_value gain_label lr_label job_root v_min

  target="${target//$'\r'/}"
  input_gain="${input_gain//$'\r'/}"
  learning_rate="${learning_rate//$'\r'/}"
  v_off="${v_off//$'\r'/}"
  target_label_value="$(target_label "${target}")"
  gain_label="$(label_float "${input_gain}")"
  lr_label="$(label_float "${learning_rate}")"
  job_root="${OUTPUT_ROOT}/target_${target_label_value}/input_gain_${gain_label}/lr_${lr_label}"
  v_min="-${v_off}"

  echo "START target=${target} input_gain=${input_gain} lr=${learning_rate} v_off=${v_off} job_root=${job_root}" | tee -a "${OUTPUT_ROOT}/logs/queue.log"
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
    --run-name mnist_bp_amp_v1_c1 \
    2>&1 | tee "${OUTPUT_ROOT}/logs/target_${target_label_value}.log"

  echo "DONE target=${target} input_gain=${input_gain} lr=${learning_rate} v_off=${v_off}" | tee -a "${OUTPUT_ROOT}/logs/queue.log"
}

tail -n +2 "${TARGET_CSV}" | while IFS=, read -r target input_gain _measured learning_rate v_off _lr_num _seed _num_samples _num_iterations; do
  [[ -n "${target}" ]] || continue
  run_job "${target}" "${input_gain}" "${learning_rate}" "${v_off}"
done
