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
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/results/mnist_bp_conv2_hardsigmoid_stride1_recalibrated_sat30_voff4_k6_seed0_30epoch}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/logs}"
DEVICE="${DEVICE:-cuda}"

mkdir -p "${LOG_DIR}"
cd "${REPO_ROOT}"

run_job() {
  local run_name="$1"
  local gain="$2"
  local gain_label="$3"
  local lr_mult="$4"
  local lr_mult_label="$5"
  local lr="$6"
  local lr_label="$7"

  local job_root="${OUTPUT_ROOT}/target_sat30/input_gain_${gain_label}/lr_mult_${lr_mult_label}/lr_${lr_label}"
  local log_path="${LOG_DIR}/${run_name}_stride1_recalibrated_sat30_30epoch.log"

  echo "[conv2-stride1-recal] $(date -Is) start run=${run_name} gain=${gain} lr_mult=${lr_mult} lr=${lr} root=${job_root}" | tee "${log_path}"

  "${PYTHON_BIN}" "${REPO_ROOT}/experiments/train_mnist_bp_conv_amplification_sweep.py" \
    --output-root "${job_root}" \
    --dataset-root "${DATASET_ROOT}" \
    --device "${DEVICE}" \
    --model-key mnist_bp_conv_amp \
    --non-linearity hard_sigmoid \
    --seeds 0 \
    --epochs "${EPOCHS:-30}" \
    --batch-size "${BATCH_SIZE:-4}" \
    --beta 1.0 \
    --lr-decay 0.99 \
    --conv-depth 2 \
    --conv-channels 64 128 256 \
    --kernel-size 3 \
    --stride 1 \
    --padding 0 \
    --output-dim 20 \
    --weight-gains 1.0 \
    --weight-min 0.0 \
    --weight-max 100.0 \
    --weight-init-mode kaiming_uniform \
    --input-gain "${gain}" \
    --num-iterations 6 \
    --learning-rate "${lr}" \
    --hard-sigmoid-g-on 100.0 \
    --hard-sigmoid-g-off 0.0 \
    --hard-sigmoid-v-min -4.0 \
    --hard-sigmoid-v-max 4.0 \
    --normalize-mean 0.1307 \
    --normalize-std 0.3081 \
    --normalize-scale 0.3 \
    --log-interval "${LOG_INTERVAL:-500}" \
    --no-download \
    --run-name "${run_name}" \
    2>&1 | tee -a "${log_path}"
}

# Stride-1 sat30 gains recalibrated per amplification with v_off=4, K=6.
# LR multipliers reuse the best fixed-amplification Conv2 sat30 choices:
# A=1,B=1 uses x4; A=4,B=1 uses x0.5.
run_job mnist_bp_amp_v1_c1 621.01953125 621p01953125 4 4 0.00927507060592 0p00927507060592 &
pid_v1_c1=$!

run_job mnist_bp_amp_v4_c1 2233.578125 2233p578125 0.5 0p5 0.000322352727196 0p000322352727196 &
pid_v4_c1=$!

status=0
wait "${pid_v1_c1}" || status=$?
wait "${pid_v4_c1}" || status=$?

echo "[conv2-stride1-recal] $(date -Is) done output=${OUTPUT_ROOT}"
exit "${status}"
