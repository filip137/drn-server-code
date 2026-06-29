#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/filip/server_code}"
PYTHON_BIN="${PYTHON_BIN:-/home/filip/miniconda3/envs/py312/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/data}"
DEVICE="${DEVICE:-cuda}"

BASELINE_ROOT="${BASELINE_ROOT:-${REPO_ROOT}/results/mnist_bp_conv2_hardsigmoid_amp_saturation_targets_voff4_sat10_30_50_70_seed0_10epoch}"
LR_SCREEN_ROOT="${LR_SCREEN_ROOT:-${REPO_ROOT}/results/mnist_bp_conv2_hardsigmoid_voff4_existing_targets_lr_screen_seed0_10epoch}"
LONGER_ROOT="${LONGER_ROOT:-${REPO_ROOT}/results/mnist_bp_conv2_hardsigmoid_voff4_existing_targets_longer_x1_seed0_30epoch}"
LR_SCREEN_MANIFEST="${LR_SCREEN_MANIFEST:-${LR_SCREEN_ROOT}/manifests/lr_screen_manifest.csv}"
LONGER_MANIFEST="${LONGER_MANIFEST:-${LONGER_ROOT}/manifests/longer_x1_manifest.csv}"
GRADIENT_CSV="${GRADIENT_CSV:-${REPO_ROOT}/results/good_conv_saturation_analysis/conv2_voff4_lr_diagnostic_final_gradients.csv}"
SUMMARY_CSV="${SUMMARY_CSV:-${REPO_ROOT}/results/good_conv_saturation_analysis/conv2_voff4_lr_diagnostic_summary.csv}"
SUMMARY_MD="${SUMMARY_MD:-${REPO_ROOT}/results/good_conv_saturation_analysis/conv2_voff4_lr_diagnostic_summary.md}"

RUN_TRAINING="${RUN_TRAINING:-1}"
RUN_ANALYSIS="${RUN_ANALYSIS:-1}"
DRY_RUN="${DRY_RUN:-0}"
BUILD_MANIFESTS="${BUILD_MANIFESTS:-1}"

cd "${REPO_ROOT}"

if [[ "${BUILD_MANIFESTS}" == "1" ]]; then
  "${PYTHON_BIN}" "${REPO_ROOT}/experiments/build_mnist_bp_conv2_voff4_lr_diagnostic_manifest.py" \
    --baseline-root "${BASELINE_ROOT}" \
    --lr-screen-root "${LR_SCREEN_ROOT}" \
    --longer-root "${LONGER_ROOT}" \
    --lr-screen-manifest "${LR_SCREEN_MANIFEST}" \
    --longer-manifest "${LONGER_MANIFEST}"
else
  test -s "${LR_SCREEN_MANIFEST}"
  test -s "${LONGER_MANIFEST}"
fi

echo "[diagnostic] LR screen jobs: $(( $(wc -l < "${LR_SCREEN_MANIFEST}") - 1 ))"
echo "[diagnostic] longer-control jobs: $(( $(wc -l < "${LONGER_MANIFEST}") - 1 ))"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[diagnostic] DRY_RUN=1; manifests built but no training launched."
  exit 0
fi

if [[ "${RUN_TRAINING}" == "1" ]]; then
  MANIFEST_CSV="${LR_SCREEN_MANIFEST}" \
  REPO_ROOT="${REPO_ROOT}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  DATASET_ROOT="${DATASET_ROOT}" \
  DEVICE="${DEVICE}" \
  EPOCHS=10 \
  BATCH_SIZE="${BATCH_SIZE:-4}" \
  LOG_INTERVAL="${LOG_INTERVAL:-500}" \
  GPU_UTIL_THRESHOLD="${GPU_UTIL_THRESHOLD:-90}" \
  GPU_MEM_THRESHOLD_MB="${GPU_MEM_THRESHOLD_MB:-9300}" \
  "${REPO_ROOT}/experiments/run_mnist_bp_conv_amp_calibrated_manifest_lane.sh"

  MANIFEST_CSV="${LONGER_MANIFEST}" \
  REPO_ROOT="${REPO_ROOT}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  DATASET_ROOT="${DATASET_ROOT}" \
  DEVICE="${DEVICE}" \
  EPOCHS=30 \
  BATCH_SIZE="${BATCH_SIZE:-4}" \
  LOG_INTERVAL="${LOG_INTERVAL:-500}" \
  GPU_UTIL_THRESHOLD="${GPU_UTIL_THRESHOLD:-90}" \
  GPU_MEM_THRESHOLD_MB="${GPU_MEM_THRESHOLD_MB:-9300}" \
  "${REPO_ROOT}/experiments/run_mnist_bp_conv_amp_calibrated_manifest_lane.sh"
fi

if [[ "${RUN_ANALYSIS}" == "1" ]]; then
  "${PYTHON_BIN}" "${REPO_ROOT}/experiments/analyze_mnist_bp_conv_good_run_gradient_saturation.py" \
    --input-root "${BASELINE_ROOT}" \
    --input-root "${LR_SCREEN_ROOT}" \
    --input-root "${LONGER_ROOT}" \
    --output-csv "${GRADIENT_CSV}" \
    --phase final \
    --split train \
    --batch-size 16 \
    --max-samples 128 \
    --dataset-root "${DATASET_ROOT}" \
    --device "${DEVICE}" \
    --no-download

  "${PYTHON_BIN}" "${REPO_ROOT}/experiments/summarize_mnist_bp_conv2_voff4_lr_diagnostic.py" \
    --baseline-root "${BASELINE_ROOT}" \
    --lr-screen-root "${LR_SCREEN_ROOT}" \
    --longer-root "${LONGER_ROOT}" \
    --gradient-csv "${GRADIENT_CSV}" \
    --output-csv "${SUMMARY_CSV}" \
    --output-md "${SUMMARY_MD}"
fi

echo "[diagnostic] done"
