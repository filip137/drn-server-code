#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/filiposana/server_code}"
PYTHON_BIN="${PYTHON_BIN:-/home/filiposana/miniconda3/envs/py312/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/data}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/results/mnist_bp_conv1_hardsigmoid_target30_v2c2_legacy_lowlr_voff4_seed0_10epoch_akib}"
GPU_UTIL_THRESHOLD="${GPU_UTIL_THRESHOLD:-90}"
GPU_MEM_THRESHOLD_MB="${GPU_MEM_THRESHOLD_MB:-9300}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-120}"

mkdir -p "${OUTPUT_ROOT}/logs"

label_float() {
  local value="$1"
  value="${value//$'\r'/}"
  value="${value//./p}"
  value="${value//-/m}"
  printf '%s' "${value}"
}

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

run_job() {
  local run_name="$1"
  local input_gain="$2"
  local base_lr="$3"
  local lr_mult="$4"
  local note="$5"
  local lr gain_label base_lr_label mult_label lr_label job_root run_log

  lr="$(awk -v lr="${base_lr}" -v mult="${lr_mult}" 'BEGIN { printf "%.17g", lr * mult }')"
  gain_label="$(label_float "${input_gain}")"
  base_lr_label="$(label_float "${base_lr}")"
  mult_label="$(label_float "${lr_mult}")"
  lr_label="$(label_float "${lr}")"
  job_root="${OUTPUT_ROOT}/target_sat30/input_gain_${gain_label}/base_lr_${base_lr_label}/lr_mult_${mult_label}/lr_${lr_label}"
  run_log="${OUTPUT_ROOT}/logs/${note}_${run_name}_lr_mult_${mult_label}.log"

  {
    date
    echo "START note=${note} run_name=${run_name} input_gain=${input_gain} base_lr=${base_lr} lr_mult=${lr_mult} lr=${lr} job_root=${job_root}"
  } | tee -a "${OUTPUT_ROOT}/logs/queue.log"

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
    --learning-rate "${lr}" \
    --hard-sigmoid-g-on 100.0 \
    --hard-sigmoid-g-off 0.0 \
    --hard-sigmoid-v-min -4.0 \
    --hard-sigmoid-v-max 4.0 \
    --normalize-mean 0.1307 \
    --normalize-std 0.3081 \
    --normalize-scale 0.3 \
    --log-interval 500 \
    --no-download \
    --run-name "${run_name}" \
    2>&1 | tee "${run_log}"

  {
    date
    echo "DONE note=${note} run_name=${run_name} input_gain=${input_gain} base_lr=${base_lr} lr_mult=${lr_mult} lr=${lr}"
  } | tee -a "${OUTPUT_ROOT}/logs/queue.log"
}

# Calibrated from:
# results/mnist_bp_conv1_hardsigmoid_amp_calibrated_saturation_targets_voff4_sat10_30_50_70_sat50_lr012_seed0_10epoch_akib/targets/conv1_amp_calibrated_targets.csv
run_job "mnist_bp_amp_v2_c2" "65.65995025634766" "0.0054827942847123475" "1" "v2c2"

for mult in 0.0625 0.125 0.25 0.5 1; do
  run_job "mnist_bp_amp_v4_c0p25" "7.207874298095703" "0.04994537711279327" "${mult}" "legacy"
done
