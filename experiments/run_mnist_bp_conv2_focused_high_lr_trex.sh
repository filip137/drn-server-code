#!/bin/bash
set -euo pipefail

SOURCE_ROOT="${SOURCE_ROOT:-/home/filip/server_code}"
PYTHON="${PYTHON:-/home/filip/miniconda3/envs/py312/bin/python}"
TRAIN="${SOURCE_ROOT}/experiments/train_mnist_bp_conv_amplification_sweep.py"
COLLECT="${SOURCE_ROOT}/experiments/collect_mnist_bp_conv_lr_smoke.py"
NUM_ITERATIONS_VALUE="${NUM_ITERATIONS:-6}"
BASE_OUTPUT_ROOT="${OUTPUT_ROOT:-${SOURCE_ROOT}/results/mnist_bp_conv2_focused_high_lr_seed0_iter${NUM_ITERATIONS_VALUE}}"
DATASET_ROOT="${DATASET_ROOT:-${SOURCE_ROOT}/data}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
TASK_IDS="${TASK_IDS:-0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25}"

if [ ! -f "${TRAIN}" ]; then
  echo "Expected conv sweep runner at ${TRAIN}" >&2
  exit 2
fi
if [ ! -f "${COLLECT}" ]; then
  echo "Expected conv collector at ${COLLECT}" >&2
  exit 2
fi
if [ "${MAX_PARALLEL}" -lt 1 ]; then
  echo "Expected MAX_PARALLEL >= 1, got ${MAX_PARALLEL}." >&2
  exit 2
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${SOURCE_ROOT}:${SOURCE_ROOT}/labs:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

mkdir -p "${BASE_OUTPUT_ROOT}" "${SOURCE_ROOT}/logs/mnist_conv2_highlr"
cd "${SOURCE_ROOT}"

common_args=(
  --dataset-root "${DATASET_ROOT}"
  --device cuda
  --model-key mnist_bp_conv_amp
  --seeds 0
  --epochs "${EPOCHS:-50}"
  --batch-size "${BATCH_SIZE:-4}"
  --num-iterations "${NUM_ITERATIONS_VALUE}"
  --beta "${BETA:-1.0}"
  --lr-decay "${LR_DECAY:-0.99}"
  --conv-depth 2
  --conv-channels 64 128 256
  --kernel-size 3
  --stride 2
  --padding 0
  --output-dim 20
  --weight-gains 1.0
  --weight-min 0.0
  --weight-max 100.0
  --weight-init-mode kaiming_uniform
  --input-gain "${INPUT_GAIN:-100.0}"
  --hard-sigmoid-g-on "${HARD_SIGMOID_G_ON:-100.0}"
  --hard-sigmoid-g-off "${HARD_SIGMOID_G_OFF:-0.0}"
  --hard-sigmoid-v-min "${HARD_SIGMOID_V_MIN:--1.5}"
  --hard-sigmoid-v-max "${HARD_SIGMOID_V_MAX:-1.5}"
  --normalize-mean 0.1307
  --normalize-std 0.3081
  --normalize-scale 0.3
  --log-interval "${LOG_INTERVAL:-500}"
  --no-download
)

LR_VALUES=(
  0.018 0.024 0.036 0.048
  0.018 0.024 0.036 0.048
  0.018 0.024 0.036 0.048
  0.018 0.024 0.036 0.048
  0.018 0.024 0.036 0.048
  0.036 0.048
  0.036 0.048
  0.036 0.048
)
LR_LABELS=(
  0p018 0p024 0p036 0p048
  0p018 0p024 0p036 0p048
  0p018 0p024 0p036 0p048
  0p018 0p024 0p036 0p048
  0p018 0p024 0p036 0p048
  0p036 0p048
  0p036 0p048
  0p036 0p048
)
NONLINEARITIES=(
  hard_sigmoid hard_sigmoid hard_sigmoid hard_sigmoid
  hard_sigmoid hard_sigmoid hard_sigmoid hard_sigmoid
  hard_sigmoid hard_sigmoid hard_sigmoid hard_sigmoid
  hard_sigmoid hard_sigmoid hard_sigmoid hard_sigmoid
  hard_sigmoid hard_sigmoid hard_sigmoid hard_sigmoid
  perfect_diode perfect_diode
  perfect_diode perfect_diode
  perfect_diode perfect_diode
)
RUN_NAMES=(
  mnist_bp_amp_v1_c1 mnist_bp_amp_v1_c1 mnist_bp_amp_v1_c1 mnist_bp_amp_v1_c1
  mnist_bp_amp_v2_c1 mnist_bp_amp_v2_c1 mnist_bp_amp_v2_c1 mnist_bp_amp_v2_c1
  mnist_bp_amp_v4_c1 mnist_bp_amp_v4_c1 mnist_bp_amp_v4_c1 mnist_bp_amp_v4_c1
  mnist_bp_amp_v1_c2 mnist_bp_amp_v1_c2 mnist_bp_amp_v1_c2 mnist_bp_amp_v1_c2
  mnist_bp_amp_v1_c4 mnist_bp_amp_v1_c4 mnist_bp_amp_v1_c4 mnist_bp_amp_v1_c4
  mnist_bp_amp_v1_c1 mnist_bp_amp_v1_c1
  mnist_bp_amp_v2_c1 mnist_bp_amp_v2_c1
  mnist_bp_amp_v4_c1 mnist_bp_amp_v4_c1
)

run_task() {
  local logical_id="$1"
  if [ "${logical_id}" -lt 0 ] || [ "${logical_id}" -ge "${#LR_VALUES[@]}" ]; then
    echo "Invalid logical task id ${logical_id}; expected 0..$((${#LR_VALUES[@]} - 1))." >&2
    return 2
  fi
  local lr="${LR_VALUES[$logical_id]}"
  local lr_label="${LR_LABELS[$logical_id]}"
  local nonlinearity="${NONLINEARITIES[$logical_id]}"
  local run_name="${RUN_NAMES[$logical_id]}"
  local output_root="${BASE_OUTPUT_ROOT}/lr_${lr_label}"
  local log="${SOURCE_ROOT}/logs/mnist_conv2_highlr/task_${logical_id}.out"
  local err="${SOURCE_ROOT}/logs/mnist_conv2_highlr/task_${logical_id}.err"

  echo "[trex] launching logical_id=${logical_id} nonlinearity=${nonlinearity} run_name=${run_name} seed=0 lr=${lr}"
  "${PYTHON}" "${TRAIN}" \
    "${common_args[@]}" \
    --output-root "${output_root}" \
    --learning-rate "${lr}" \
    --non-linearity "${nonlinearity}" \
    --run-name "${run_name}" \
    >"${log}" 2>"${err}"
  echo "[trex] completed logical_id=${logical_id}"
}

echo "[trex] source_root=${SOURCE_ROOT}"
echo "[trex] dataset_root=${DATASET_ROOT}"
echo "[trex] base_output_root=${BASE_OUTPUT_ROOT}"
echo "[trex] epochs=${EPOCHS:-50} num_iterations=${NUM_ITERATIONS_VALUE} max_parallel=${MAX_PARALLEL}"
echo "[trex] task_ids=${TASK_IDS}"

status=0
active=0
for logical_id in ${TASK_IDS}; do
  run_task "${logical_id}" &
  active=$((active + 1))
  if [ "${active}" -ge "${MAX_PARALLEL}" ]; then
    if ! wait -n; then
      status=1
    fi
    active=$((active - 1))
  fi
done

while [ "${active}" -gt 0 ]; do
  if ! wait -n; then
    status=1
  fi
  active=$((active - 1))
done

"${PYTHON}" "${COLLECT}" --root "${BASE_OUTPUT_ROOT}" || status=1
exit "${status}"
