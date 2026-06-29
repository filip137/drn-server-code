#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/filip/server_code}"
PYTHON_BIN="${PYTHON_BIN:-/home/filip/miniconda3/envs/py312/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/results/mnist_bp_conv2_hardsigmoid_stride1_best_sat30_voff4_k6_seed0_30epoch}"
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
  local log_path="${LOG_DIR}/${run_name}_stride1_30epoch.log"

  echo "[conv2-stride1] $(date -Is) start run=${run_name} gain=${gain} lr_mult=${lr_mult} lr=${lr} root=${job_root}" | tee "${log_path}"

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

run_job mnist_bp_amp_v1_c1 221.06837463378906 221p068374634 4 4 0.0260552872365 0p0260552872365 &
pid_v1_c1=$!

run_job mnist_bp_amp_v4_c1 158.0352325439453 158p035232544 0.5 0p5 0.00455594609132 0p00455594609132 &
pid_v4_c1=$!

status=0
wait "${pid_v1_c1}" || status=$?
wait "${pid_v4_c1}" || status=$?

echo "[conv2-stride1] $(date -Is) done output=${OUTPUT_ROOT}"
exit "${status}"
