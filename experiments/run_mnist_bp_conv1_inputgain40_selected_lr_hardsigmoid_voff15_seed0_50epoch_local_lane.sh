#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <lane_index> <num_lanes>" >&2
  exit 2
fi

LANE_INDEX="$1"
NUM_LANES="$2"

REPO_ROOT="${REPO_ROOT:-/home/filip/server_code}"
PYTHON_BIN="${PYTHON_BIN:-/home/filip/miniconda3/envs/py312/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/results/mnist_bp_conv1_inputgain40_selected_lr_hardsigmoid_voff15_seed0_50epoch}"
DATASET_ROOT="${DATASET_ROOT:-/home/filip/datasets/mnist}"
DEVICE="${DEVICE:-cuda}"
MAX_BATCHES="${MAX_BATCHES:-}"
MAX_TEST_BATCHES="${MAX_TEST_BATCHES:-}"

COMMON_ARGS=(
  --dataset-root "${DATASET_ROOT}"
  --device "${DEVICE}"
  --non-linearity hard_sigmoid
  --seeds 0
  --epochs 50
  --batch-size 4
  --beta 1.0
  --lr-decay 0.99
  --conv-depth 1
  --conv-channels 64
  --kernel-size 3
  --stride 2
  --padding 0
  --output-dim 20
  --weight-gains 1.0
  --weight-min 0.0
  --weight-max 100.0
  --weight-init-mode kaiming_uniform
  --input-gain 40.0
  --num-iterations 4
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

if [[ -n "${MAX_BATCHES}" ]]; then
  COMMON_ARGS+=(--max-batches "${MAX_BATCHES}")
fi
if [[ -n "${MAX_TEST_BATCHES}" ]]; then
  COMMON_ARGS+=(--max-test-batches "${MAX_TEST_BATCHES}")
fi

label_float() {
  local value="$1"
  value="${value//./p}"
  value="${value//-/m}"
  printf '%s' "${value}"
}

run_job() {
  local index="$1"
  local run_name="$2"
  local lr="$3"

  if (( index % NUM_LANES != LANE_INDEX )); then
    return
  fi

  local lr_label
  lr_label="$(label_float "${lr}")"
  local job_root="${OUTPUT_ROOT}/lr_${lr_label}"

  echo "[lane ${LANE_INDEX}/${NUM_LANES}] job=${index} ${run_name} input_gain=40 K=4 epochs=50 lr=${lr} output=${job_root}"
  "${PYTHON_BIN}" "${REPO_ROOT}/experiments/train_mnist_bp_conv_amplification_sweep.py" \
    --output-root "${job_root}" \
    "${COMMON_ARGS[@]}" \
    --learning-rate "${lr}" \
    --run-name "${run_name}"
}

mkdir -p "${OUTPUT_ROOT}/lane_logs"

job_index=0
run_job "${job_index}" mnist_bp_amp_v1_c1 0.018
job_index=$((job_index + 1))
run_job "${job_index}" mnist_bp_amp_v2_c1 0.012
job_index=$((job_index + 1))
run_job "${job_index}" mnist_bp_amp_v4_c1 0.009
job_index=$((job_index + 1))
run_job "${job_index}" mnist_bp_amp_v1_c2 0.036
job_index=$((job_index + 1))
run_job "${job_index}" mnist_bp_amp_v1_c4 0.048

echo "[lane ${LANE_INDEX}/${NUM_LANES}] complete"
