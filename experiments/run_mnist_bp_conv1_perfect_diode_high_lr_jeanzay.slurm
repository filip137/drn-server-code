#!/bin/bash -l
#SBATCH --job-name=conv1_pd_hilr
#SBATCH --output=slurm/conv1_pd_hilr/%x-%A_%a.out
#SBATCH --error=slurm/conv1_pd_hilr/%x-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=12:00:00
#SBATCH --account=umg@v100
#SBATCH --qos=qos_gpu-t3
#SBATCH --partition=gpu_p13
#SBATCH --constraint=v100-16g
#SBATCH --array=0-10%11

set -euo pipefail

if command -v idrenv >/dev/null 2>&1; then
  eval "$(idrenv)"
fi
if [ -z "${WORK:-}" ]; then
  WORK="/lustre/fswork/projects/rech/${IDRPROJ:-umg}/${USER}"
fi
if [ -z "${SCRATCH:-}" ]; then
  SCRATCH="/lustre/fsn1/projects/rech/${IDRPROJ:-umg}/${USER}"
fi

module purge
module load "${PYTORCH_MODULE:-pytorch-gpu/py3/2.5.0}"

SOURCE_ROOT="${SOURCE_ROOT:-${WORK}/server_code}"
cd "${SOURCE_ROOT}"

PYTHON="${PYTHON:-python}"
TRAIN="${SOURCE_ROOT}/experiments/train_mnist_bp_conv_amplification_sweep.py"
COLLECT="${SOURCE_ROOT}/experiments/collect_mnist_bp_conv1_perfect_diode_high_lr.py"
DATASET_ROOT="${DATASET_ROOT:-${SCRATCH}/datasets/mnist}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRATCH}/server_code/results/mnist_bp_conv1_perfect_diode_inputgain40_high_lr_seed0_50epoch}"
SUMMARY_ONLY="${SUMMARY_ONLY:-0}"
PACK_SIZE="${PACK_SIZE:-1}"

RUN_NAMES=(
  mnist_bp_amp_v1_c1
  mnist_bp_amp_v1_c1
  mnist_bp_amp_v1_c1
  mnist_bp_amp_v2_c1
  mnist_bp_amp_v2_c1
  mnist_bp_amp_v2_c1
  mnist_bp_amp_v1_c4
  mnist_bp_amp_v1_c4
  mnist_bp_amp_v1_c4
  mnist_bp_amp_v1_c4
  mnist_bp_amp_v1_c4
)
LRS=(0.012 0.018 0.024 0.012 0.018 0.024 0.012 0.018 0.024 0.036 0.048)
LR_LABELS=(0p012 0p018 0p024 0p012 0p018 0p024 0p012 0p018 0p024 0p036 0p048)

export PYTHONUNBUFFERED=1
export PYTHONPATH="${SOURCE_ROOT}:${SOURCE_ROOT}/labs:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

mkdir -p "${OUTPUT_ROOT}" slurm/conv1_pd_hilr

if [ "${SUMMARY_ONLY}" = "1" ]; then
  "${PYTHON}" "${COLLECT}" --root "${OUTPUT_ROOT}"
  exit 0
fi

run_task() {
  local task_id="$1"
  if [ "${task_id}" -ge "${#RUN_NAMES[@]}" ]; then
    return 0
  fi

  local run_name="${RUN_NAMES[$task_id]}"
  local learning_rate="${LRS[$task_id]}"
  local lr_label="${LR_LABELS[$task_id]}"
  local job_root="${OUTPUT_ROOT}/lr_${lr_label}"

  echo "[conv1-perfect-diode-high-lr] task=${task_id} run=${run_name}"
  echo "[conv1-perfect-diode-high-lr] input_gain=${INPUT_GAIN:-40.0} lr=${learning_rate} K=${NUM_ITERATIONS:-4} output=${job_root}"

  "${PYTHON}" "${TRAIN}" \
    --output-root "${job_root}" \
    --dataset-root "${DATASET_ROOT}" \
    --device cuda \
    --model-key mnist_bp_conv_amp \
    --seeds 0 \
    --epochs "${EPOCHS:-50}" \
    --batch-size "${BATCH_SIZE:-4}" \
    --beta "${BETA:-1.0}" \
    --lr-decay "${LR_DECAY:-0.99}" \
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
    --input-gain "${INPUT_GAIN:-40.0}" \
    --num-iterations "${NUM_ITERATIONS:-4}" \
    --learning-rate "${learning_rate}" \
    --hard-sigmoid-g-on 100.0 \
    --hard-sigmoid-g-off 0.0 \
    --hard-sigmoid-v-min -1.5 \
    --hard-sigmoid-v-max 1.5 \
    --normalize-mean 0.1307 \
    --normalize-std 0.3081 \
    --normalize-scale 0.3 \
    --log-interval "${LOG_INTERVAL:-500}" \
    --no-download \
    --non-linearity perfect_diode \
    --run-name "${run_name}"
}

array_task_id="${SLURM_ARRAY_TASK_ID:-0}"
base_task_id=$((array_task_id * PACK_SIZE))
echo "[conv1-perfect-diode-high-lr] array_task=${array_task_id} pack_size=${PACK_SIZE} base_task=${base_task_id}"

pids=()
for ((offset = 0; offset < PACK_SIZE; offset++)); do
  task_id=$((base_task_id + offset))
  run_task "${task_id}" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
exit "${status}"
