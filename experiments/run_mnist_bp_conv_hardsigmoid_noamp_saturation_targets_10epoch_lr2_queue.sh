#!/usr/bin/env bash
# MNIST_CONV_LEGACY_LAUNCHER_DEPRECATED
printf "%s\n" "Expected format: python -m experiments.mnist_conv {run|sweep|collect} ..." >&2
printf "%s\n" "Provided deprecated MNIST Conv launcher invocation: $0 $*" >&2
printf "%s\n" "Deprecated MNIST Conv launcher; no setup, allocation, or work was performed." >&2
exit 2
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/filip/server_code}"
PYTHON_BIN="${PYTHON_BIN:-/home/filip/miniconda3/envs/py312/bin/python}"
CONV_DEPTH="${CONV_DEPTH:?Expected CONV_DEPTH, e.g. 2 or 3.}"
PADDING="${PADDING:?Expected PADDING, e.g. 0 or 1.}"
NUM_ITERATIONS="${NUM_ITERATIONS:?Expected NUM_ITERATIONS, e.g. 6 or 16.}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/results/mnist_bp_conv${CONV_DEPTH}_hardsigmoid_noamp_saturation_targets_seed0_10epoch_lr2_voff15_25}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/data}"
DEVICE="${DEVICE:-cuda}"
GPU_UTIL_THRESHOLD="${GPU_UTIL_THRESHOLD:-35}"
GPU_MEM_THRESHOLD_MB="${GPU_MEM_THRESHOLD_MB:-12000}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-300}"
V_OFFS="${V_OFFS:-1.5 2.5}"
LR_NUMERATOR="${LR_NUMERATOR:-1.44}"
SEARCH_STEPS="${SEARCH_STEPS:-24}"
SEARCH_SAMPLES="${SEARCH_SAMPLES:-256}"
SATURATION_SCOPE="${SATURATION_SCOPE:-first_hidden}"

mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/targets"

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

target_label() {
  "${PYTHON_BIN}" - "$1" <<'PY'
import sys
value = float(sys.argv[1])
print(f"sat{int(round(value * 100)):02d}")
PY
}

emit_rows() {
  "${PYTHON_BIN}" - "$1" <<'PY'
import csv
import sys
from pathlib import Path

for row in csv.DictReader(Path(sys.argv[1]).open()):
    print("\t".join([row["target_saturation"], row["input_gain"], row["learning_rate"]]))
PY
}

run_search_if_needed() {
  local v_off="$1"
  local v_label="$2"
  local csv_path="${OUTPUT_ROOT}/targets/conv${CONV_DEPTH}_voff_${v_label}_targets.csv"
  if [[ -s "${csv_path}" ]]; then
    echo "SEARCH_SKIP conv_depth=${CONV_DEPTH} v_off=${v_off} csv=${csv_path}" | tee -a "${OUTPUT_ROOT}/logs/queue.log"
    return
  fi

  echo "SEARCH_START conv_depth=${CONV_DEPTH} v_off=${v_off} csv=${csv_path}" | tee -a "${OUTPUT_ROOT}/logs/queue.log"
  wait_for_gpu
  "${PYTHON_BIN}" "${REPO_ROOT}/experiments/search_mnist_bp_conv_hardsigmoid_saturation_targets.py" \
    --dataset-root "${DATASET_ROOT}" \
    --output-csv "${csv_path}" \
    --device "${DEVICE}" \
    --seed 0 \
    --num-samples "${SEARCH_SAMPLES}" \
    --batch-size 64 \
    --num-iterations "${NUM_ITERATIONS}" \
    --conv-depth "${CONV_DEPTH}" \
    --conv-channels 64 128 256 \
    --kernel-size 3 \
    --stride 2 \
    --padding "${PADDING}" \
    --output-dim 20 \
    --v-off "${v_off}" \
    --saturation-scope "${SATURATION_SCOPE}" \
    --g-on 100.0 \
    --g-off 0.0 \
    --normalize-mean 0.1307 \
    --normalize-std 0.3081 \
    --normalize-scale 0.3 \
    --targets 0.10 0.30 0.50 0.70 0.90 \
    --initial-hi 64.0 \
    --max-gain 8192.0 \
    --search-steps "${SEARCH_STEPS}" \
    --lr-numerator "${LR_NUMERATOR}" \
    2>&1 | tee "${OUTPUT_ROOT}/logs/search_voff_${v_label}.log"
  echo "SEARCH_DONE conv_depth=${CONV_DEPTH} v_off=${v_off} csv=${csv_path}" | tee -a "${OUTPUT_ROOT}/logs/queue.log"
}

run_job() {
  local v_off="$1"
  local target="$2"
  local input_gain="$3"
  local learning_rate="$4"
  local v_label t_label gain_label lr_label job_root

  v_label="$(label_float "${v_off}")"
  t_label="$(target_label "${target}")"
  gain_label="$(label_float "${input_gain}")"
  lr_label="$(label_float "${learning_rate}")"
  job_root="${OUTPUT_ROOT}/v_off_${v_label}/target_${t_label}/input_gain_${gain_label}/lr_${lr_label}"

  echo "START conv_depth=${CONV_DEPTH} v_off=${v_off} target=${target} input_gain=${input_gain} lr=${learning_rate} job_root=${job_root}" | tee -a "${OUTPUT_ROOT}/logs/queue.log"
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
    --conv-depth "${CONV_DEPTH}" \
    --conv-channels 64 128 256 \
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
    --hard-sigmoid-v-min "-${v_off}" \
    --hard-sigmoid-v-max "${v_off}" \
    --normalize-mean 0.1307 \
    --normalize-std 0.3081 \
    --normalize-scale 0.3 \
    --log-interval 500 \
    --no-download \
    --run-name mnist_bp_amp_v1_c1 \
    2>&1 | tee "${OUTPUT_ROOT}/logs/conv${CONV_DEPTH}_voff_${v_label}_target_${t_label}.log"

  echo "DONE conv_depth=${CONV_DEPTH} v_off=${v_off} target=${target} input_gain=${input_gain} lr=${learning_rate}" | tee -a "${OUTPUT_ROOT}/logs/queue.log"
}

for v_off in ${V_OFFS}; do
  v_label="$(label_float "${v_off}")"
  csv_path="${OUTPUT_ROOT}/targets/conv${CONV_DEPTH}_voff_${v_label}_targets.csv"
  run_search_if_needed "${v_off}" "${v_label}"
  while IFS=$'\t' read -r target input_gain learning_rate; do
    run_job "${v_off}" "${target}" "${input_gain}" "${learning_rate}"
  done < <(emit_rows "${csv_path}")
done

echo "QUEUE_DONE conv_depth=${CONV_DEPTH} output_root=${OUTPUT_ROOT}" | tee -a "${OUTPUT_ROOT}/logs/queue.log"
