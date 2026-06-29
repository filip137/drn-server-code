#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/filip/server_code}"
PYTHON_BIN="${PYTHON_BIN:-/home/filip/miniconda3/envs/py312/bin/python}"
CONV_DEPTH="${CONV_DEPTH:?Expected CONV_DEPTH, e.g. 1 or 2.}"
PADDING="${PADDING:?Expected PADDING, e.g. 0.}"
NUM_ITERATIONS="${NUM_ITERATIONS:?Expected NUM_ITERATIONS, e.g. 4 or 6.}"
LR_NUMERATOR="${LR_NUMERATOR:?Expected LR_NUMERATOR, e.g. 0.36 or 1.44.}"

DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/data}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/results/mnist_bp_conv${CONV_DEPTH}_hardsigmoid_amp_calibrated_saturation_targets_voff4_sat10_30_50_70_sat50_lr012_seed0_10epoch}"
WORK_ROOT="${WORK_ROOT:-${OUTPUT_ROOT}/targets}"
CALIBRATION_CSV="${CALIBRATION_CSV:-${WORK_ROOT}/conv${CONV_DEPTH}_amp_calibrated_targets.csv}"
MANIFEST_CSV="${MANIFEST_CSV:-${WORK_ROOT}/conv${CONV_DEPTH}_amp_calibrated_manifest.csv}"
TARGETS="${TARGETS:-0.1 0.3 0.5 0.7}"
V_OFF="${V_OFF:-4.0}"
SEARCH_STEPS="${SEARCH_STEPS:-24}"
SEARCH_SAMPLES="${SEARCH_SAMPLES:-256}"
SEARCH_BATCH_SIZE="${SEARCH_BATCH_SIZE:-64}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"
GPU_UTIL_THRESHOLD="${GPU_UTIL_THRESHOLD:-98}"
GPU_MEM_THRESHOLD_MB="${GPU_MEM_THRESHOLD_MB:-9300}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-120}"

wait_for_gpu() {
  if [[ "${DEVICE}" != cuda* ]]; then
    return
  fi
  while true; do
    local util mem
    util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d ' ')"
    mem="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')"
    echo "[$(date --iso-8601=seconds)] local-pipeline gpu util=${util} mem=${mem}MiB thresholds util<${GPU_UTIL_THRESHOLD} mem<${GPU_MEM_THRESHOLD_MB}"
    if [[ "${util}" -lt "${GPU_UTIL_THRESHOLD}" && "${mem}" -lt "${GPU_MEM_THRESHOLD_MB}" ]]; then
      break
    fi
    sleep "${GPU_POLL_SECONDS}"
  done
}

mkdir -p "${WORK_ROOT}"

if [[ ! -s "${CALIBRATION_CSV}" ]]; then
  wait_for_gpu
  "${PYTHON_BIN}" "${REPO_ROOT}/experiments/search_mnist_bp_conv_hardsigmoid_saturation_targets.py" \
    --dataset-root "${DATASET_ROOT}" \
    --output-csv "${CALIBRATION_CSV}" \
    --device "${DEVICE}" \
    --seed 0 \
    --num-samples "${SEARCH_SAMPLES}" \
    --batch-size "${SEARCH_BATCH_SIZE}" \
    --num-iterations "${NUM_ITERATIONS}" \
    --conv-depth "${CONV_DEPTH}" \
    --conv-channels 64 128 256 \
    --kernel-size 3 \
    --stride 2 \
    --padding "${PADDING}" \
    --output-dim 20 \
    --v-off "${V_OFF}" \
    --saturation-scope first_hidden \
    --targets ${TARGETS} \
    --search-steps "${SEARCH_STEPS}" \
    --lr-numerator "${LR_NUMERATOR}" \
    --all-amplifications
fi

"${PYTHON_BIN}" "${REPO_ROOT}/experiments/build_mnist_bp_conv_amp_calibrated_training_manifest.py" \
  --calibration-csv "${CALIBRATION_CSV}" \
  --output-csv "${MANIFEST_CSV}" \
  --output-root "${OUTPUT_ROOT}" \
  --targets ${TARGETS} \
  --sat50-lr-multipliers 0.5 1.0 2.0

MANIFEST_CSV="${MANIFEST_CSV}" \
REPO_ROOT="${REPO_ROOT}" \
PYTHON_BIN="${PYTHON_BIN}" \
DATASET_ROOT="${DATASET_ROOT}" \
DEVICE="${DEVICE}" \
EPOCHS="${EPOCHS}" \
BATCH_SIZE="${BATCH_SIZE}" \
NUM_SHARDS="${NUM_SHARDS}" \
SHARD_INDEX="${SHARD_INDEX}" \
GPU_UTIL_THRESHOLD="${GPU_UTIL_THRESHOLD}" \
GPU_MEM_THRESHOLD_MB="${GPU_MEM_THRESHOLD_MB}" \
GPU_POLL_SECONDS="${GPU_POLL_SECONDS}" \
bash "${REPO_ROOT}/experiments/run_mnist_bp_conv_amp_calibrated_manifest_lane.sh"
