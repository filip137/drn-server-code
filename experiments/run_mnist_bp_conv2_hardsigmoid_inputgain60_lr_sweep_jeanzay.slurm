#!/bin/bash -l
#SBATCH --mnist-conv-legacy-launcher-is-deprecated
# MNIST_CONV_LEGACY_LAUNCHER_DEPRECATED
printf "%s\n" "Expected format: python -m experiments.mnist_conv {run|sweep|collect} ..." >&2
printf "%s\n" "Provided deprecated MNIST Conv launcher invocation: $0 $*" >&2
printf "%s\n" "Deprecated MNIST Conv launcher; no setup, allocation, or work was performed." >&2
exit 2
#SBATCH --job-name=mnist_conv2_hs_ig60_lr
#SBATCH --output=slurm/mnist_conv2_hs_ig60_lr/%x-%A_%a.out
#SBATCH --error=slurm/mnist_conv2_hs_ig60_lr/%x-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --account=umg@v100
#SBATCH --qos=qos_gpu-t3
#SBATCH --partition=gpu_p13
#SBATCH --constraint=v100-16g
#SBATCH --array=0-21%22

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
if [ ! -d "${SOURCE_ROOT}" ]; then
  echo "Expected SOURCE_ROOT to exist on Jean Zay: ${SOURCE_ROOT}" >&2
  exit 2
fi
cd "${SOURCE_ROOT}"

PYTHON="${PYTHON:-python}"
TRAIN="${SOURCE_ROOT}/experiments/train_mnist_bp_conv_amplification_sweep.py"
COLLECT="${SOURCE_ROOT}/experiments/collect_mnist_bp_conv_lr_smoke.py"
EPOCHS_VALUE="${EPOCHS:-50}"
BASE_OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRATCH}/server_code/results/mnist_bp_conv2_hardsigmoid_inputgain60_lr_sweep_seed0_${EPOCHS_VALUE}epoch}"
DATASET_ROOT="${DATASET_ROOT:-${SCRATCH}/datasets/mnist}"
SUMMARY_ONLY="${SUMMARY_ONLY:-0}"

if [ ! -f "${TRAIN}" ]; then
  echo "Expected conv sweep runner at ${TRAIN}" >&2
  exit 2
fi
if [ ! -f "${COLLECT}" ]; then
  echo "Expected LR collector at ${COLLECT}" >&2
  exit 2
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${SOURCE_ROOT}:${SOURCE_ROOT}/labs:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

mkdir -p "${BASE_OUTPUT_ROOT}" slurm/mnist_conv2_hs_ig60_lr

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

if [ "${SUMMARY_ONLY}" = "1" ]; then
  "${PYTHON}" "${COLLECT}" --root "${BASE_OUTPUT_ROOT}"
  exit 0
fi

task_id="${SLURM_ARRAY_TASK_ID:-0}"
if [ "${task_id}" -ge "${#RUN_NAMES[@]}" ]; then
  echo "task_id=${task_id} is outside total_tasks=${#RUN_NAMES[@]}; exiting."
  exit 0
fi

run_name="${RUN_NAMES[$task_id]}"
learning_rate="${LEARNING_RATES[$task_id]}"
lr_label="${LR_LABELS[$task_id]}"
num_iterations="${ITERATION_COUNTS[$task_id]}"
output_root="${BASE_OUTPUT_ROOT}/lr_${lr_label}"

echo "[jean-zay] job=${SLURM_JOB_ID:-manual} array_task=${task_id}/${#RUN_NAMES[@]}"
echo "[jean-zay] source_root=${SOURCE_ROOT}"
echo "[jean-zay] dataset_root=${DATASET_ROOT}"
echo "[jean-zay] base_output_root=${BASE_OUTPUT_ROOT}"
echo "[jean-zay] run_name=${run_name} nonlinearity=hard_sigmoid seed=0"
echo "[jean-zay] input_gain=60 lr=${learning_rate} K=${num_iterations} epochs=${EPOCHS_VALUE}"
echo "[jean-zay] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"

srun "${PYTHON}" "${TRAIN}" \
  "${common_args[@]}" \
  --output-root "${output_root}" \
  --num-iterations "${num_iterations}" \
  --learning-rate "${learning_rate}" \
  --non-linearity hard_sigmoid \
  --run-name "${run_name}"
