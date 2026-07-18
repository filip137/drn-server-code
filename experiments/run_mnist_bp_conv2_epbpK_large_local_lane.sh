#!/usr/bin/env bash
# MNIST_CONV_LEGACY_LAUNCHER_DEPRECATED
printf "%s\n" "Expected format: python -m experiments.mnist_conv {run|sweep|collect} ..." >&2
printf "%s\n" "Provided deprecated MNIST Conv launcher invocation: $0 $*" >&2
printf "%s\n" "Deprecated MNIST Conv launcher; no setup, allocation, or work was performed." >&2
exit 2
# Run a shard/lane of the Conv2 EP/BP-selected-K seed-0 jobs on a local GPU.

set -euo pipefail

LANE_ID="${LANE_ID:?Set LANE_ID to a zero-based lane index.}"
NUM_LANES="${NUM_LANES:?Set NUM_LANES to the total number of lanes.}"

SOURCE_ROOT="${SOURCE_ROOT:-/home/filip/server_code}"
PYTHON="${PYTHON:-/home/filip/miniconda3/envs/py312/bin/python}"
if [ ! -x "${PYTHON}" ]; then
  PYTHON="python"
fi

TRAIN="${SOURCE_ROOT}/experiments/train_mnist_bp_conv_amplification_sweep.py"
EPOCHS_VALUE="${EPOCHS:-50}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SOURCE_ROOT}/results/mnist_bp_conv2_epbpK_large_${EPOCHS_VALUE}epoch_64_128ch_s2_valid_seed0_local}"
DATASET_ROOT="${DATASET_ROOT:-${SOURCE_ROOT}/model/resistive/data}"

if [ ! -d "${SOURCE_ROOT}" ]; then
  echo "Expected SOURCE_ROOT to exist, got ${SOURCE_ROOT}." >&2
  exit 2
fi
if [ ! -f "${TRAIN}" ]; then
  echo "Expected training runner to exist, got ${TRAIN}." >&2
  exit 2
fi
if [ ! -d "${DATASET_ROOT}" ]; then
  echo "Expected DATASET_ROOT to exist, got ${DATASET_ROOT}." >&2
  exit 2
fi

cd "${SOURCE_ROOT}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${SOURCE_ROOT}:${SOURCE_ROOT}/labs:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "${OUTPUT_ROOT}/lane_logs"

common_args=(
  --output-root "${OUTPUT_ROOT}"
  --dataset-root "${DATASET_ROOT}"
  --device cuda
  --model-key mnist_bp_conv_amp
  --seeds 0
  --epochs "${EPOCHS_VALUE}"
  --batch-size "${BATCH_SIZE:-4}"
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

NONLINEARITIES=(
  hard_sigmoid
  hard_sigmoid
  hard_sigmoid
  hard_sigmoid
  hard_sigmoid
  perfect_diode
  perfect_diode
  perfect_diode
  perfect_diode
  perfect_diode
)
RUN_NAMES=(
  mnist_bp_amp_v1_c1
  mnist_bp_amp_v2_c1
  mnist_bp_amp_v4_c1
  mnist_bp_amp_v1_c2
  mnist_bp_amp_v1_c4
  mnist_bp_amp_v1_c1
  mnist_bp_amp_v2_c1
  mnist_bp_amp_v4_c1
  mnist_bp_amp_v1_c2
  mnist_bp_amp_v1_c4
)
LEARNING_RATES=(
  0.012
  0.012
  0.012
  0.024
  0.024
  0.024
  0.012
  0.012
  0.006
  0.0015
)
ITERATION_COUNTS=(
  12
  8
  12
  4
  4
  4
  4
  4
  4
  4
)

if [ "${SUMMARY_ONLY:-0}" = "1" ]; then
  "${PYTHON}" "${TRAIN}" \
    "${common_args[@]}" \
    --num-iterations 4 \
    --learning-rate 0.006 \
    --non-linearity hard_sigmoid perfect_diode \
    --summary-only
  exit 0
fi

echo "[lane] host=$(hostname) lane=${LANE_ID}/${NUM_LANES} output=${OUTPUT_ROOT}"
echo "[lane] cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"

for task_id in "${!RUN_NAMES[@]}"; do
  if [ $((task_id % NUM_LANES)) -ne "${LANE_ID}" ]; then
    continue
  fi

  nonlinearity="${NONLINEARITIES[$task_id]}"
  run_name="${RUN_NAMES[$task_id]}"
  learning_rate="${LEARNING_RATES[$task_id]}"
  num_iterations="${ITERATION_COUNTS[$task_id]}"
  log_path="${OUTPUT_ROOT}/lane_logs/task_${task_id}_${nonlinearity}_${run_name}.log"

  echo "[lane] start task=${task_id} nonlinearity=${nonlinearity} run=${run_name} lr=${learning_rate} K=${num_iterations}"
  "${PYTHON}" "${TRAIN}" \
    "${common_args[@]}" \
    --num-iterations "${num_iterations}" \
    --learning-rate "${learning_rate}" \
    --non-linearity "${nonlinearity}" \
    --run-name "${run_name}" \
    2>&1 | tee "${log_path}"
  echo "[lane] done task=${task_id} log=${log_path}"
done

echo "[lane] complete lane=${LANE_ID}/${NUM_LANES}"
