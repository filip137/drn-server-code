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
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/results/mnist_bp_conv_fixed_step_v4c0p25_wide_lr_trex_20260704}"
LANE="${LANE:?Expected LANE=hs or LANE=pd.}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MINIMIZER_CONFIG="${MINIMIZER_CONFIG:-${REPO_ROOT}/labs/configs/mnist_minimizer_fixed_iterations.json}"

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_ROOT}/lane_logs"

label_float() {
  local value="$1"
  value="${value//./p}"
  value="${value//-/m}"
  printf '%s' "${value}"
}

run_train() {
  local nonlinearity="$1"
  local factor="$2"
  local input_gain="$3"
  local center_lr="$4"
  local T="$5"
  local K="$6"
  local learning_rate
  learning_rate="$("${PYTHON_BIN}" - "${center_lr}" "${factor}" <<'PY'
import sys
print(f"{float(sys.argv[1]) * float(sys.argv[2]):.12g}")
PY
)"

  local factor_label lr_label gain_label job_root
  factor_label="$(label_float "${factor}")"
  lr_label="$(label_float "${learning_rate}")"
  gain_label="$(label_float "${input_gain}")"
  job_root="${OUTPUT_ROOT}/conv2/${nonlinearity}/mnist_bp_amp_v4_c0p25/input_gain_${gain_label}/T_${T}_K_${K}/lr_center_$(label_float "${center_lr}")/lr_factor_${factor_label}/lr_${lr_label}"

  echo "[trex-wide-lr] $(date -Is) lane=${LANE} nonlinearity=${nonlinearity} factor=${factor} lr=${learning_rate} root=${job_root}"
  "${PYTHON_BIN}" "${REPO_ROOT}/experiments/train_mnist_bp_conv_amplification_sweep.py" \
    --output-root "${job_root}" \
    --dataset-root "${DATASET_ROOT}" \
    --device cuda \
    --model-key mnist_bp_conv_amp \
    --non-linearity "${nonlinearity}" \
    --seeds 0 \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --beta 1.0 \
    --lr-decay 0.99 \
    --conv-depth 2 \
    --conv-channels 64 128 \
    --kernel-size 3 \
    --strides 2 2 \
    --paddings 1 1 \
    --output-dim 20 \
    --weight-gains 1.0 \
    --weight-min 0.0 \
    --weight-max 100.0 \
    --weight-init-mode kaiming_uniform \
    --input-gain "${input_gain}" \
    --num-iterations-inference "${T}" \
    --num-iterations-training "${K}" \
    --learning-rate "${learning_rate}" \
    --minimizer-config "${MINIMIZER_CONFIG}" \
    --hard-sigmoid-g-on 100.0 \
    --hard-sigmoid-g-off 0.0 \
    --hard-sigmoid-v-min -4.0 \
    --hard-sigmoid-v-max 4.0 \
    --normalize-mean 0.1307 \
    --normalize-std 0.3081 \
    --normalize-scale 0.3 \
    --log-interval "${LOG_INTERVAL:-500}" \
    --no-download \
    --run-name mnist_bp_amp_v4_c0p25
}

case "${LANE}" in
  hs)
    # R3 tested factors 0.5, 1, 2 around this center. This lane expands below,
    # between, and above that neighborhood for the sat30-calibrated gain.
    input_gain="${HS_INPUT_GAIN:-643.8777465820312}"
    center_lr="${HS_CENTER_LR:-0.000559112349992}"
    T="${HS_T:-6}"
    K="${HS_K:-16}"
    read -r -a factors <<< "${HS_FACTORS:-0.125 0.25 0.75 1.5 3 4 8}"
    for factor in "${factors[@]}"; do
      run_train hard_sigmoid "${factor}" "${input_gain}" "${center_lr}" "${T}" "${K}"
    done
    ;;
  pd)
    # R3 tested factors 0.5, 1, 2 and the row degraded upward, so this lane
    # focuses below the old center while keeping one intermediate upper probe.
    input_gain="${PD_INPUT_GAIN:-100}"
    center_lr="${PD_CENTER_LR:-0.012}"
    T="${PD_T:-6}"
    K="${PD_K:-4}"
    read -r -a factors <<< "${PD_FACTORS:-0.03125 0.0625 0.125 0.25 0.375 0.75}"
    for factor in "${factors[@]}"; do
      run_train perfect_diode "${factor}" "${input_gain}" "${center_lr}" "${T}" "${K}"
    done
    ;;
  *)
    echo "Expected LANE=hs or LANE=pd, got ${LANE}." >&2
    exit 2
    ;;
esac
