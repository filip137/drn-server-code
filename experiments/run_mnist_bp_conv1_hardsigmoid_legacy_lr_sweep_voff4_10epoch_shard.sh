#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/filip/server_code}"
PYTHON_BIN="${PYTHON_BIN:-/home/filip/miniconda3/envs/py312/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/data}"
DEVICE="${DEVICE:-cuda}"
BASE_TARGET_CSV="${BASE_TARGET_CSV:-${REPO_ROOT}/results/mnist_bp_conv1_hardsigmoid_noamp_saturation_targets_voff4_seed0_10epoch_lr0p5_akib/target_input_gains.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/results/mnist_bp_conv1_hardsigmoid_legacy_lr_sweep_voff4_sat50_70_90_seed0_10epoch}"
TARGETS="${TARGETS:-0.5 0.7 0.9}"
LR_MULTIPLIERS="${LR_MULTIPLIERS:-0.5 1 2 4 8}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"
RUN_NAME="${RUN_NAME:-mnist_bp_amp_v4_c0p25}"
GPU_UTIL_THRESHOLD="${GPU_UTIL_THRESHOLD:-98}"
GPU_MEM_THRESHOLD_MB="${GPU_MEM_THRESHOLD_MB:-9300}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-120}"
WAIT_FOR_PID="${WAIT_FOR_PID:-}"
WAIT_POLL_SECONDS="${WAIT_POLL_SECONDS:-60}"

mkdir -p "${OUTPUT_ROOT}/logs"

if [[ "${SHARD_INDEX}" -lt 0 || "${SHARD_INDEX}" -ge "${NUM_SHARDS}" ]]; then
  echo "expected 0 <= SHARD_INDEX < NUM_SHARDS, got SHARD_INDEX=${SHARD_INDEX} NUM_SHARDS=${NUM_SHARDS}" >&2
  exit 2
fi

if [[ -n "${WAIT_FOR_PID}" ]]; then
  while ps -p "${WAIT_FOR_PID}" >/dev/null 2>&1; do
    {
      date
      echo "shard=${SHARD_INDEX}/${NUM_SHARDS} waiting for pid ${WAIT_FOR_PID}"
    } | tee -a "${OUTPUT_ROOT}/logs/shard_${SHARD_INDEX}.queue.log"
    sleep "${WAIT_POLL_SECONDS}"
  done
fi

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

target_allowed() {
  local target="$1"
  local allowed
  for allowed in ${TARGETS}; do
    if [[ "${target}" == "${allowed}" ]]; then
      return 0
    fi
  done
  return 1
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
      echo "shard=${SHARD_INDEX}/${NUM_SHARDS} gpu util=${util} mem=${mem}MiB threshold_util=${GPU_UTIL_THRESHOLD} threshold_mem=${GPU_MEM_THRESHOLD_MB}MiB"
    } | tee -a "${OUTPUT_ROOT}/logs/shard_${SHARD_INDEX}.queue.log"

    if [[ "${util}" -lt "${GPU_UTIL_THRESHOLD}" && "${mem}" -lt "${GPU_MEM_THRESHOLD_MB}" ]]; then
      break
    fi
    sleep "${GPU_POLL_SECONDS}"
  done
}

run_job() {
  local target="$1"
  local input_gain="$2"
  local base_learning_rate="$3"
  local v_off="$4"
  local lr_multiplier="$5"
  local learning_rate target_label_value gain_label base_lr_label mult_label lr_label job_root run_log v_min

  target="${target//$'\r'/}"
  input_gain="${input_gain//$'\r'/}"
  base_learning_rate="${base_learning_rate//$'\r'/}"
  v_off="${v_off//$'\r'/}"
  lr_multiplier="${lr_multiplier//$'\r'/}"
  learning_rate="$(awk -v lr="${base_learning_rate}" -v mult="${lr_multiplier}" 'BEGIN { printf "%.17g", lr * mult }')"
  target_label_value="$(target_label "${target}")"
  gain_label="$(label_float "${input_gain}")"
  base_lr_label="$(label_float "${base_learning_rate}")"
  mult_label="$(label_float "${lr_multiplier}")"
  lr_label="$(label_float "${learning_rate}")"
  job_root="${OUTPUT_ROOT}/target_${target_label_value}/input_gain_${gain_label}/base_lr_${base_lr_label}/lr_mult_${mult_label}/lr_${lr_label}"
  run_log="${OUTPUT_ROOT}/logs/target_${target_label_value}_lr_mult_${mult_label}_${RUN_NAME}.log"
  v_min="-${v_off}"

  {
    date
    echo "START shard=${SHARD_INDEX}/${NUM_SHARDS} target=${target} input_gain=${input_gain} base_lr=${base_learning_rate} lr_mult=${lr_multiplier} lr=${learning_rate} v_off=${v_off} run_name=${RUN_NAME} job_root=${job_root}"
  } | tee -a "${OUTPUT_ROOT}/logs/shard_${SHARD_INDEX}.queue.log"
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
    --hard-sigmoid-v-min "${v_min}" \
    --hard-sigmoid-v-max "${v_off}" \
    --normalize-mean 0.1307 \
    --normalize-std 0.3081 \
    --normalize-scale 0.3 \
    --log-interval 500 \
    --no-download \
    --run-name "${RUN_NAME}" \
    2>&1 | tee "${run_log}"

  {
    date
    echo "DONE shard=${SHARD_INDEX}/${NUM_SHARDS} target=${target} input_gain=${input_gain} base_lr=${base_learning_rate} lr_mult=${lr_multiplier} lr=${learning_rate} v_off=${v_off} run_name=${RUN_NAME}"
  } | tee -a "${OUTPUT_ROOT}/logs/shard_${SHARD_INDEX}.queue.log"
}

job_index=0
while IFS=, read -r target input_gain _measured base_learning_rate v_off _lr_num _seed _num_samples _num_iterations; do
  [[ -n "${target}" ]] || continue
  if ! target_allowed "${target}"; then
    continue
  fi
  for lr_multiplier in ${LR_MULTIPLIERS}; do
    if (( job_index % NUM_SHARDS == SHARD_INDEX )); then
      run_job "${target}" "${input_gain}" "${base_learning_rate}" "${v_off}" "${lr_multiplier}"
    fi
    job_index=$((job_index + 1))
  done
done < <(tail -n +2 "${BASE_TARGET_CSV}")
