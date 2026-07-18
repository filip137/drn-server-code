#!/usr/bin/env bash
# MNIST_CONV_LEGACY_LAUNCHER_DEPRECATED
printf "%s\n" "Expected format: python -m experiments.mnist_conv {run|sweep|collect} ..." >&2
printf "%s\n" "Provided deprecated MNIST Conv launcher invocation: $0 $*" >&2
printf "%s\n" "Deprecated MNIST Conv launcher; no setup, allocation, or work was performed." >&2
exit 2
# Continue the interrupted Conv2 hard-sigmoid input-gain-60 LR sweep from best_model.pt.

set -euo pipefail

LANE_ID="${LANE_ID:?Set LANE_ID to a zero-based lane index.}"
NUM_LANES="${NUM_LANES:?Set NUM_LANES to the total number of lanes.}"
PACK_SIZE="${PACK_SIZE:-1}"

SOURCE_ROOT="${SOURCE_ROOT:-/home/filip/server_code}"
PYTHON="${PYTHON:-/home/filip/miniconda3/envs/py312/bin/python}"
if [ ! -x "${PYTHON}" ] && [ -x "/home/filiposana/miniconda3/envs/py312/bin/python" ]; then
  PYTHON="/home/filiposana/miniconda3/envs/py312/bin/python"
fi
if [ ! -x "${PYTHON}" ]; then
  PYTHON="python"
fi

TRAIN="${SOURCE_ROOT}/experiments/train_mnist_bp_conv_amplification_sweep.py"
COLLECT="${SOURCE_ROOT}/experiments/collect_mnist_bp_conv_lr_smoke.py"
EPOCHS_VALUE="${EPOCHS:-20}"
INIT_ROOT="${INIT_ROOT:-${SOURCE_ROOT}/results/mnist_bp_conv2_hardsigmoid_inputgain60_lr_sweep_seed0_50epoch}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SOURCE_ROOT}/results/mnist_bp_conv2_hardsigmoid_inputgain60_lr_sweep_seed0_continue${EPOCHS_VALUE}_from_jz_best}"
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
if [ ! -d "${INIT_ROOT}" ]; then
  echo "Expected INIT_ROOT to exist, got ${INIT_ROOT}." >&2
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

RUN_NAMES=(
  mnist_bp_amp_v1_c1
  mnist_bp_amp_v1_c1
  mnist_bp_amp_v1_c1
  mnist_bp_amp_v1_c1
  mnist_bp_amp_v2_c1
  mnist_bp_amp_v2_c1
  mnist_bp_amp_v2_c1
  mnist_bp_amp_v2_c1
  mnist_bp_amp_v4_c1
  mnist_bp_amp_v4_c1
  mnist_bp_amp_v4_c1
  mnist_bp_amp_v4_c1
  mnist_bp_amp_v1_c2
  mnist_bp_amp_v1_c2
  mnist_bp_amp_v1_c2
  mnist_bp_amp_v1_c2
  mnist_bp_amp_v1_c2
  mnist_bp_amp_v1_c4
  mnist_bp_amp_v1_c4
  mnist_bp_amp_v1_c4
  mnist_bp_amp_v1_c4
  mnist_bp_amp_v1_c4
)
LEARNING_RATES=(
  0.006
  0.0075
  0.009
  0.012
  0.006
  0.0075
  0.009
  0.012
  0.006
  0.0075
  0.009
  0.012
  0.009
  0.012
  0.015
  0.018
  0.024
  0.009
  0.012
  0.015
  0.018
  0.024
)
LR_LABELS=(
  0p006
  0p0075
  0p009
  0p012
  0p006
  0p0075
  0p009
  0p012
  0p006
  0p0075
  0p009
  0p012
  0p009
  0p012
  0p015
  0p018
  0p024
  0p009
  0p012
  0p015
  0p018
  0p024
)
ITERATION_COUNTS=(
  12
  12
  12
  12
  8
  8
  8
  8
  12
  12
  12
  12
  16
  16
  16
  16
  16
  16
  16
  16
  16
  16
)

common_args=(
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
  --input-gain 60.0
  --hard-sigmoid-g-on 100.0
  --hard-sigmoid-g-off 0.0
  --hard-sigmoid-v-min -1.5
  --hard-sigmoid-v-max 1.5
  --normalize-mean 0.1307
  --normalize-std 0.3081
  --normalize-scale 0.3
  --log-interval "${LOG_INTERVAL:-500}"
  --no-download
)

run_task() {
  local task_id="$1"
  local run_name="${RUN_NAMES[$task_id]}"
  local learning_rate="${LEARNING_RATES[$task_id]}"
  local lr_label="${LR_LABELS[$task_id]}"
  local num_iterations="${ITERATION_COUNTS[$task_id]}"
  local init_lr_root="${INIT_ROOT}/lr_${lr_label}"
  local checkpoint="${init_lr_root}/hard_sigmoid/${run_name}/seed_0/best_model.pt"
  local output_lr_root="${OUTPUT_ROOT}/lr_${lr_label}"
  local log_path="${OUTPUT_ROOT}/lane_logs/task_${task_id}_${run_name}_lr_${lr_label}.log"

  if [ ! -f "${checkpoint}" ]; then
    echo "Expected checkpoint to exist, got ${checkpoint}." >&2
    return 2
  fi

  echo "[lane] start task=${task_id} host=$(hostname) lane=${LANE_ID}/${NUM_LANES} run=${run_name} lr=${learning_rate} K=${num_iterations}"
  echo "[lane] init_checkpoint=${checkpoint}"
  "${PYTHON}" "${TRAIN}" \
    "${common_args[@]}" \
    --output-root "${output_lr_root}" \
    --num-iterations "${num_iterations}" \
    --learning-rate "${learning_rate}" \
    --non-linearity hard_sigmoid \
    --run-name "${run_name}" \
    --init-checkpoint-root "${init_lr_root}" \
    --init-checkpoint-name best_model.pt \
    2>&1 | tee "${log_path}"
  echo "[lane] done task=${task_id} log=${log_path}"
}

if [ "${SUMMARY_ONLY:-0}" = "1" ]; then
  "${PYTHON}" "${COLLECT}" --root "${OUTPUT_ROOT}"
  exit 0
fi

echo "[lane] host=$(hostname) lane=${LANE_ID}/${NUM_LANES} pack=${PACK_SIZE} epochs=${EPOCHS_VALUE}"
echo "[lane] init_root=${INIT_ROOT}"
echo "[lane] output_root=${OUTPUT_ROOT}"
echo "[lane] cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"

task_ids=()
for task_id in "${!RUN_NAMES[@]}"; do
  if [ $((task_id % NUM_LANES)) -eq "${LANE_ID}" ]; then
    task_ids+=("${task_id}")
  fi
done

for ((start = 0; start < ${#task_ids[@]}; start += PACK_SIZE)); do
  pids=()
  for ((offset = 0; offset < PACK_SIZE; offset++)); do
    index=$((start + offset))
    if [ "${index}" -ge "${#task_ids[@]}" ]; then
      break
    fi
    run_task "${task_ids[$index]}" &
    pids+=("$!")
  done

  status=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done
  if [ "${status}" -ne 0 ]; then
    exit "${status}"
  fi
done

echo "[lane] complete lane=${LANE_ID}/${NUM_LANES}"
