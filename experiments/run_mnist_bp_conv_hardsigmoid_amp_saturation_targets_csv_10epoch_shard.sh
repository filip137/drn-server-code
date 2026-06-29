#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/filip/server_code}"
PYTHON_BIN="${PYTHON_BIN:-/home/filip/miniconda3/envs/py312/bin/python}"
TARGET_CSV="${TARGET_CSV:?TARGET_CSV must point to a saturation target CSV.}"
OUTPUT_ROOT="${OUTPUT_ROOT:?OUTPUT_ROOT must be set.}"
CONV_DEPTH="${CONV_DEPTH:?Expected CONV_DEPTH, e.g. 2 or 3.}"
PADDING="${PADDING:?Expected PADDING, e.g. 0 or 1.}"
NUM_ITERATIONS="${NUM_ITERATIONS:?Expected NUM_ITERATIONS, e.g. 6 or 16.}"

DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/data}"
DEVICE="${DEVICE:-cuda}"
GPU_UTIL_THRESHOLD="${GPU_UTIL_THRESHOLD:-98}"
GPU_MEM_THRESHOLD_MB="${GPU_MEM_THRESHOLD_MB:-9300}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-120}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"
TARGETS="${TARGETS:-0.1 0.3 0.5 0.7}"
RUN_NAMES="${RUN_NAMES:-mnist_bp_amp_v1_c1 mnist_bp_amp_v2_c1 mnist_bp_amp_v4_c1 mnist_bp_amp_v1_c2 mnist_bp_amp_v1_c4 mnist_bp_amp_v4_c0p25 mnist_bp_amp_v2_c2}"
CONV_CHANNELS="${CONV_CHANNELS:-64 128 256}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LOG_INTERVAL="${LOG_INTERVAL:-500}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Expected executable PYTHON_BIN, got ${PYTHON_BIN}" >&2
  exit 2
fi
if [[ ! -s "${TARGET_CSV}" ]]; then
  echo "Expected non-empty TARGET_CSV, got ${TARGET_CSV}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}/logs"

if [[ "${SHARD_INDEX}" -lt 0 || "${SHARD_INDEX}" -ge "${NUM_SHARDS}" ]]; then
  echo "expected 0 <= SHARD_INDEX < NUM_SHARDS, got SHARD_INDEX=${SHARD_INDEX} NUM_SHARDS=${NUM_SHARDS}" >&2
  exit 2
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
      echo "shard=${SHARD_INDEX}/${NUM_SHARDS} gpu util=${util} mem=${mem}MiB threshold_util=${GPU_UTIL_THRESHOLD} threshold_mem=${GPU_MEM_THRESHOLD_MB}MiB"
    } | tee -a "${OUTPUT_ROOT}/logs/shard_${SHARD_INDEX}.queue.log"

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

emit_rows() {
  TARGETS="${TARGETS}" "${PYTHON_BIN}" - "${TARGET_CSV}" <<'PY'
import csv
import os
import sys
from pathlib import Path

targets = {round(float(value), 8) for value in os.environ["TARGETS"].split()}
with Path(sys.argv[1]).open(newline="") as handle:
    for row in csv.DictReader(handle):
        target = round(float(row["target_saturation"]), 8)
        if target not in targets:
            continue
        print(
            "\t".join(
                [
                    row["target_saturation"],
                    row["input_gain"],
                    row["learning_rate"],
                    row["v_off"],
                ]
            )
        )
PY
}

run_job() {
  local target="$1"
  local input_gain="$2"
  local learning_rate="$3"
  local v_off="$4"
  local run_name="$5"
  local target_label_value gain_label lr_label job_root v_min run_log
  local conv_channels_array

  target="${target//$'\r'/}"
  input_gain="${input_gain//$'\r'/}"
  learning_rate="${learning_rate//$'\r'/}"
  v_off="${v_off//$'\r'/}"
  run_name="${run_name//$'\r'/}"
  target_label_value="$(target_label "${target}")"
  gain_label="$(label_float "${input_gain}")"
  lr_label="$(label_float "${learning_rate}")"
  job_root="${OUTPUT_ROOT}/target_${target_label_value}/input_gain_${gain_label}/lr_${lr_label}"
  run_log="${OUTPUT_ROOT}/logs/conv${CONV_DEPTH}_target_${target_label_value}_${run_name}.log"
  v_min="-${v_off}"

  {
    date
    echo "START conv_depth=${CONV_DEPTH} shard=${SHARD_INDEX}/${NUM_SHARDS} target=${target} input_gain=${input_gain} lr=${learning_rate} v_off=${v_off} run_name=${run_name} job_root=${job_root}"
  } | tee -a "${OUTPUT_ROOT}/logs/shard_${SHARD_INDEX}.queue.log"
  wait_for_gpu

  read -r -a conv_channels_array <<< "${CONV_CHANNELS}"
  "${PYTHON_BIN}" "${REPO_ROOT}/experiments/train_mnist_bp_conv_amplification_sweep.py" \
    --output-root "${job_root}" \
    --dataset-root "${DATASET_ROOT}" \
    --device "${DEVICE}" \
    --model-key mnist_bp_conv_amp \
    --non-linearity hard_sigmoid \
    --seeds 0 \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --beta 1.0 \
    --lr-decay 0.99 \
    --conv-depth "${CONV_DEPTH}" \
    --conv-channels "${conv_channels_array[@]}" \
    --kernel-size 3 \
    --stride 2 \
    --padding "${PADDING}" \
    --output-dim 20 \
    --weight-gains 1.0 \
    --weight-min 0.0 \
    --weight-max 100.0 \
    --weight-init-mode kaiming_uniform \
    --input-gain "${input_gain}" \
    --num-iterations "${NUM_ITERATIONS}" \
    --learning-rate "${learning_rate}" \
    --hard-sigmoid-g-on 100.0 \
    --hard-sigmoid-g-off 0.0 \
    --hard-sigmoid-v-min "${v_min}" \
    --hard-sigmoid-v-max "${v_off}" \
    --normalize-mean 0.1307 \
    --normalize-std 0.3081 \
    --normalize-scale 0.3 \
    --log-interval "${LOG_INTERVAL}" \
    --no-download \
    --run-name "${run_name}" \
    2>&1 | tee "${run_log}"

  {
    date
    echo "DONE conv_depth=${CONV_DEPTH} shard=${SHARD_INDEX}/${NUM_SHARDS} target=${target} input_gain=${input_gain} lr=${learning_rate} v_off=${v_off} run_name=${run_name}"
  } | tee -a "${OUTPUT_ROOT}/logs/shard_${SHARD_INDEX}.queue.log"
}

job_index=0
while IFS=$'\t' read -r target input_gain learning_rate v_off; do
  [[ -n "${target}" ]] || continue
  for run_name in ${RUN_NAMES}; do
    if (( job_index % NUM_SHARDS == SHARD_INDEX )); then
      run_job "${target}" "${input_gain}" "${learning_rate}" "${v_off}" "${run_name}"
    fi
    job_index=$((job_index + 1))
  done
done < <(emit_rows)

echo "QUEUE_DONE conv_depth=${CONV_DEPTH} shard=${SHARD_INDEX}/${NUM_SHARDS} output_root=${OUTPUT_ROOT}" | tee -a "${OUTPUT_ROOT}/logs/shard_${SHARD_INDEX}.queue.log"
