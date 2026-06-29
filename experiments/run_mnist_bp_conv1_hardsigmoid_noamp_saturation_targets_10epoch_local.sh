#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/filip/server_code}"
PYTHON_BIN="${PYTHON_BIN:-/home/filip/miniconda3/envs/py312/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/results/mnist_bp_conv1_hardsigmoid_noamp_saturation_targets_seed0_10epoch}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/data}"
DEVICE="${DEVICE:-cuda}"
GPU_UTIL_THRESHOLD="${GPU_UTIL_THRESHOLD:-35}"
GPU_MEM_THRESHOLD_MB="${GPU_MEM_THRESHOLD_MB:-6000}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-300}"

mkdir -p "${OUTPUT_ROOT}/logs"

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

  echo "START target=${target} input_gain=${input_gain} lr=${learning_rate} job_root=${job_root}" | tee -a "${OUTPUT_ROOT}/logs/queue.log"
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
    --hard-sigmoid-v-min -1.5 \
    --hard-sigmoid-v-max 1.5 \
    --normalize-mean 0.1307 \
    --normalize-std 0.3081 \
    --normalize-scale 0.3 \
    --log-interval 500 \
    --no-download \
    --run-name mnist_bp_amp_v1_c1 \
    2>&1 | tee "${OUTPUT_ROOT}/logs/target_${target_label}.log"

  echo "DONE target=${target} input_gain=${input_gain} lr=${learning_rate}" | tee -a "${OUTPUT_ROOT}/logs/queue.log"
}

# Target input gains were selected by binary search on initialized no-amplification
# Conv1 h1 saturation, measured as fraction of |h1| > 1.5 on 256 MNIST train samples.
# LR is inverse-scaled from the 10-epoch no-amp Conv1 reference lr(IG=40)=0.018.
run_job 0.10 10.2362 0.070338561 sat10
run_job 0.30 24.845 0.028979708 sat30
run_job 0.50 45.8423 0.015706023 sat50
run_job 0.70 67.3022 0.010698008 sat70
run_job 0.90 176.104 0.0040885044 sat90
