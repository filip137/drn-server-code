#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/filip/server_code}"
PYTHON_BIN="${PYTHON_BIN:-/home/filip/miniconda3/envs/py312/bin/python}"
MANIFEST_CSV="${MANIFEST_CSV:?MANIFEST_CSV must point to a training manifest CSV.}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/data}"
DEVICE="${DEVICE:-cuda}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LOG_INTERVAL="${LOG_INTERVAL:-500}"
CONV_CHANNELS="${CONV_CHANNELS:-64 128 256}"
GPU_UTIL_THRESHOLD="${GPU_UTIL_THRESHOLD:-98}"
GPU_MEM_THRESHOLD_MB="${GPU_MEM_THRESHOLD_MB:-9300}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-120}"
TRAINABLE_AMPLIFICATION="${TRAINABLE_AMPLIFICATION:-0}"
AMPLIFICATION_MIN="${AMPLIFICATION_MIN:-1e-6}"
AMPLIFICATION_MAX="${AMPLIFICATION_MAX:-}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1 && [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Expected executable PYTHON_BIN or command on PATH, got ${PYTHON_BIN}" >&2
  exit 2
fi
if [[ ! -s "${MANIFEST_CSV}" ]]; then
  echo "Expected non-empty MANIFEST_CSV, got ${MANIFEST_CSV}" >&2
  exit 2
fi
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
    echo "[$(date --iso-8601=seconds)] gpu util=${util} mem=${mem}MiB thresholds util<${GPU_UTIL_THRESHOLD} mem<${GPU_MEM_THRESHOLD_MB}" >&2
    if [[ "${util}" -lt "${GPU_UTIL_THRESHOLD}" && "${mem}" -lt "${GPU_MEM_THRESHOLD_MB}" ]]; then
      break
    fi
    sleep "${GPU_POLL_SECONDS}"
  done
}

emit_rows() {
  "${PYTHON_BIN}" - "${MANIFEST_CSV}" "${NUM_SHARDS}" "${SHARD_INDEX}" <<'PY'
import csv
import sys

manifest, num_shards, shard_index = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
with open(manifest, newline="") as handle:
    for row in csv.DictReader(handle):
        job_index = int(row["job_index"])
        if job_index % num_shards != shard_index:
            continue
        fields = [
            "job_index",
            "job_root",
            "conv_depth",
            "padding",
            "num_iterations",
            "input_gain",
            "learning_rate",
            "v_off",
            "run_name",
            "target_label",
            "lr_multiplier",
        ]
        print("\t".join(row[field] for field in fields))
PY
}

run_row() {
  local job_index="$1"
  local job_root="$2"
  local conv_depth="$3"
  local padding="$4"
  local num_iterations="$5"
  local input_gain="$6"
  local learning_rate="$7"
  local v_off="$8"
  local run_name="$9"
  local target_label="${10}"
  local lr_multiplier="${11}"
  local v_min
  local log_dir
  local run_log
  local conv_channels_array

  v_min="-${v_off}"
  log_dir="${job_root}/logs"
  mkdir -p "${log_dir}"
  run_log="${log_dir}/manifest_job_${job_index}_${run_name}.log"

  echo "[$(date --iso-8601=seconds)] START job_index=${job_index} conv=${conv_depth} ${target_label} lr_mult=${lr_multiplier} run=${run_name} root=${job_root}" | tee -a "${run_log}"
  wait_for_gpu
  read -r -a conv_channels_array <<< "${CONV_CHANNELS}"
  extra_train_args=()
  if [[ "${TRAINABLE_AMPLIFICATION}" == "1" || "${TRAINABLE_AMPLIFICATION}" == "true" ]]; then
    extra_train_args+=(--trainable-amplification --amplification-min "${AMPLIFICATION_MIN}")
    if [[ -n "${AMPLIFICATION_MAX}" ]]; then
      extra_train_args+=(--amplification-max "${AMPLIFICATION_MAX}")
    fi
  fi

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
    --conv-depth "${conv_depth}" \
    --conv-channels "${conv_channels_array[@]}" \
    --kernel-size 3 \
    --stride 2 \
    --padding "${padding}" \
    --output-dim 20 \
    --weight-gains 1.0 \
    --weight-min 0.0 \
    --weight-max 100.0 \
    --weight-init-mode kaiming_uniform \
    --input-gain "${input_gain}" \
    --num-iterations "${num_iterations}" \
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
    "${extra_train_args[@]}" \
    2>&1 | tee -a "${run_log}"

  echo "[$(date --iso-8601=seconds)] DONE job_index=${job_index} conv=${conv_depth} ${target_label} lr_mult=${lr_multiplier} run=${run_name}" | tee -a "${run_log}"
}

while IFS=$'\t' read -r job_index job_root conv_depth padding num_iterations input_gain learning_rate v_off run_name target_label lr_multiplier; do
  [[ -n "${job_index}" ]] || continue
  run_row "${job_index}" "${job_root}" "${conv_depth}" "${padding}" "${num_iterations}" "${input_gain}" "${learning_rate}" "${v_off}" "${run_name}" "${target_label}" "${lr_multiplier}"
done < <(emit_rows)

echo "[$(date --iso-8601=seconds)] LANE_DONE manifest=${MANIFEST_CSV} shard=${SHARD_INDEX}/${NUM_SHARDS}"
