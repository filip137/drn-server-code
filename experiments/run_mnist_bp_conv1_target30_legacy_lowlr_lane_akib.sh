#!/usr/bin/env bash
# MNIST_CONV_LEGACY_LAUNCHER_DEPRECATED
printf "%s\n" "Expected format: python -m experiments.mnist_conv {run|sweep|collect} ..." >&2
printf "%s\n" "Provided deprecated MNIST Conv launcher invocation: $0 $*" >&2
printf "%s\n" "Deprecated MNIST Conv launcher; no setup, allocation, or work was performed." >&2
exit 2
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/filiposana/server_code}"
PYTHON_BIN="${PYTHON_BIN:-/home/filiposana/miniconda3/envs/py312/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/data}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/results/mnist_bp_conv1_hardsigmoid_target30_v2c2_legacy_lowlr_voff4_seed0_10epoch_akib}"
GPU_UTIL_THRESHOLD="${GPU_UTIL_THRESHOLD:-90}"
GPU_MEM_THRESHOLD_MB="${GPU_MEM_THRESHOLD_MB:-9300}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-90}"
NUM_LANES="${NUM_LANES:-1}"
LANE_ID="${LANE_ID:-0}"

mkdir -p "${OUTPUT_ROOT}/logs"

if [[ "${LANE_ID}" -lt 0 || "${LANE_ID}" -ge "${NUM_LANES}" ]]; then
  echo "expected 0 <= LANE_ID < NUM_LANES, got LANE_ID=${LANE_ID} NUM_LANES=${NUM_LANES}" >&2
  exit 2
fi

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
      echo "lane=${LANE_ID}/${NUM_LANES} gpu util=${util} mem=${mem}MiB threshold_util=${GPU_UTIL_THRESHOLD} threshold_mem=${GPU_MEM_THRESHOLD_MB}MiB"
    } | tee -a "${OUTPUT_ROOT}/logs/legacy_lane_${LANE_ID}.queue.log"
    if [[ "${util}" -lt "${GPU_UTIL_THRESHOLD}" && "${mem}" -lt "${GPU_MEM_THRESHOLD_MB}" ]]; then
      break
    fi
    sleep "${GPU_POLL_SECONDS}"
  done
}

run_job() {
  local lr_mult="$1"
  local run_name="mnist_bp_amp_v4_c0p25"
  local input_gain="7.207874298095703"
  local base_lr="0.04994537711279327"
  local lr gain_label base_lr_label mult_label lr_label job_root run_log metrics_path

  lr="$(awk -v lr="${base_lr}" -v mult="${lr_mult}" 'BEGIN { printf "%.17g", lr * mult }')"
  gain_label="$(label_float "${input_gain}")"
  base_lr_label="$(label_float "${base_lr}")"
  mult_label="$(label_float "${lr_mult}")"
  lr_label="$(label_float "${lr}")"
  job_root="${OUTPUT_ROOT}/target_sat30/input_gain_${gain_label}/base_lr_${base_lr_label}/lr_mult_${mult_label}/lr_${lr_label}"
  run_log="${OUTPUT_ROOT}/logs/legacy_lane_${LANE_ID}_lr_mult_${mult_label}.log"
  metrics_path="${job_root}/hard_sigmoid/${run_name}/seed_0/metrics.json"

  if [[ -f "${metrics_path}" ]]; then
    echo "SKIP lane=${LANE_ID}/${NUM_LANES} lr_mult=${lr_mult} metrics=${metrics_path}" | tee -a "${OUTPUT_ROOT}/logs/legacy_lane_${LANE_ID}.queue.log"
    return
  fi

  {
    date
    echo "START lane=${LANE_ID}/${NUM_LANES} run_name=${run_name} input_gain=${input_gain} base_lr=${base_lr} lr_mult=${lr_mult} lr=${lr} job_root=${job_root}"
  } | tee -a "${OUTPUT_ROOT}/logs/legacy_lane_${LANE_ID}.queue.log"

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
    echo "DONE lane=${LANE_ID}/${NUM_LANES} run_name=${run_name} input_gain=${input_gain} base_lr=${base_lr} lr_mult=${lr_mult} lr=${lr}"
  } | tee -a "${OUTPUT_ROOT}/logs/legacy_lane_${LANE_ID}.queue.log"
}

job_index=0
for lr_mult in 0.0625 0.125 0.25 0.5 1; do
  if (( job_index % NUM_LANES == LANE_ID )); then
    run_job "${lr_mult}"
  fi
  job_index=$((job_index + 1))
done
