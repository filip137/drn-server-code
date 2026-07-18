#!/usr/bin/env bash
# MNIST_CONV_LEGACY_LAUNCHER_DEPRECATED
printf "%s\n" "Expected format: python -m experiments.mnist_conv {run|sweep|collect} ..." >&2
printf "%s\n" "Provided deprecated MNIST Conv launcher invocation: $0 $*" >&2
printf "%s\n" "Deprecated MNIST Conv launcher; no setup, allocation, or work was performed." >&2
exit 2
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <lane_index> <num_lanes>" >&2
  exit 2
fi

LANE_INDEX="$1"
NUM_LANES="$2"

REPO_ROOT="${REPO_ROOT:-/home/filip/server_code}"
PYTHON_BIN="${PYTHON_BIN:-/home/filip/miniconda3/envs/py312/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/results/mnist_bp_conv_inputgain_train_sweep_hardsigmoid_voff15_seed0_5epoch}"
DATASET_ROOT="${DATASET_ROOT:-/home/filip/datasets/mnist}"
DEVICE="${DEVICE:-cuda}"

COMMON_ARGS=(
  --dataset-root "${DATASET_ROOT}"
  --device "${DEVICE}"
  --non-linearity hard_sigmoid
  --seeds 0
  --epochs 5
  --batch-size 4
  --learning-rate 0.006
  --beta 1.0
  --lr-decay 0.99
  --kernel-size 3
  --stride 2
  --padding 0
  --output-dim 20
  --weight-gains 1.0
  --weight-min 0.0
  --weight-max 100.0
  --weight-init-mode kaiming_uniform
  --hard-sigmoid-g-on 100.0
  --hard-sigmoid-g-off 0.0
  --hard-sigmoid-v-min -1.5
  --hard-sigmoid-v-max 1.5
  --normalize-mean 0.1307
  --normalize-std 0.3081
  --normalize-scale 0.3
  --log-interval 500
  --no-download
)

label_float() {
  local value="$1"
  value="${value//./p}"
  value="${value//-/m}"
  printf '%s' "${value}"
}

run_job() {
  local index="$1"
  local conv_depth="$2"
  local input_gain="$3"
  local iterations="$4"
  local run_name="$5"

  if (( index % NUM_LANES != LANE_INDEX )); then
    return
  fi

  local gain_label
  local k_label
  gain_label="$(label_float "${input_gain}")"
  k_label="$(label_float "${iterations}")"
  local job_root="${OUTPUT_ROOT}/conv${conv_depth}/input_gain_${gain_label}/k_${k_label}"

  echo "[lane ${LANE_INDEX}/${NUM_LANES}] job=${index} conv${conv_depth} ${run_name} input_gain=${input_gain} K=${iterations} output=${job_root}"
  "${PYTHON_BIN}" "${REPO_ROOT}/experiments/train_mnist_bp_conv_amplification_sweep.py" \
    --output-root "${job_root}" \
    "${COMMON_ARGS[@]}" \
    --conv-depth "${conv_depth}" \
    --input-gain "${input_gain}" \
    --num-iterations "${iterations}" \
    --run-name "${run_name}"
}

mkdir -p "${OUTPUT_ROOT}/lane_logs"

job_index=0
for input_gain in 10 20 30 40 50; do
  for run_name in mnist_bp_amp_v1_c1 mnist_bp_amp_v2_c1 mnist_bp_amp_v4_c1 mnist_bp_amp_v1_c2 mnist_bp_amp_v1_c4; do
    run_job "${job_index}" 1 "${input_gain}" 4 "${run_name}"
    job_index=$((job_index + 1))
  done
done

for input_gain in 25 50 75 100; do
  run_job "${job_index}" 2 "${input_gain}" 12 mnist_bp_amp_v1_c1
  job_index=$((job_index + 1))
  run_job "${job_index}" 2 "${input_gain}" 8 mnist_bp_amp_v2_c1
  job_index=$((job_index + 1))
  run_job "${job_index}" 2 "${input_gain}" 12 mnist_bp_amp_v4_c1
  job_index=$((job_index + 1))
  run_job "${job_index}" 2 "${input_gain}" 16 mnist_bp_amp_v1_c2
  job_index=$((job_index + 1))
  run_job "${job_index}" 2 "${input_gain}" 16 mnist_bp_amp_v1_c4
  job_index=$((job_index + 1))
done

echo "[lane ${LANE_INDEX}/${NUM_LANES}] complete"
