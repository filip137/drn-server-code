#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/filip/server_code}"
PYTHON_BIN="${PYTHON_BIN:-/home/filip/miniconda3/envs/py312/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/results/mnist_bp_conv1_hardsigmoid_noamp_saturation_targets_voff2p5_seed0_10epoch_lr2}"
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
      echo "waiting for pid ${WAIT_FOR_PID} before starting v_off=2.5 queue"
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
  value="${value//./p}"
  value="${value//-/m}"
  printf '%s' "${value}"
}

run_job() {
  local target="$1"
  local input_gain="$2"
  local learning_rate="$3"
  local target_label="$4"
  local gain_label lr_label job_root

  gain_label="$(label_float "${input_gain}")"
  lr_label="$(label_float "${learning_rate}")"
  job_root="${OUTPUT_ROOT}/target_${target_label}/input_gain_${gain_label}/lr_${lr_label}"

  echo "START target=${target} input_gain=${input_gain} lr=${learning_rate} v_off=2.5 job_root=${job_root}" | tee -a "${OUTPUT_ROOT}/logs/queue.log"
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
    --hard-sigmoid-v-min -2.5 \
    --hard-sigmoid-v-max 2.5 \
    --normalize-mean 0.1307 \
    --normalize-std 0.3081 \
    --normalize-scale 0.3 \
    --log-interval 500 \
    --no-download \
    --run-name mnist_bp_amp_v1_c1 \
    2>&1 | tee "${OUTPUT_ROOT}/logs/target_${target_label}.log"

  echo "DONE target=${target} input_gain=${input_gain} lr=${learning_rate} v_off=2.5" | tee -a "${OUTPUT_ROOT}/logs/queue.log"
}

# Same initialized no-amplification Conv1 saturation-target protocol as the
# LR x2 run, with hard-sigmoid v_off=2.5. Targets were measured as |h1| > 2.5
# on 256 MNIST train samples. Learning rates use lr = 1.44 / input_gain.
run_job 0.10 16.524959564208984 0.08714090914442306 sat10
run_job 0.30 41.03604507446289 0.0350911009427691 sat30
run_job 0.50 75.98670196533203 0.01895068430074754 sat50
run_job 0.70 111.79586029052734 0.012880620053889541 sat70
run_job 0.90 292.7659606933594 0.004918604596619222 sat90
