#!/bin/bash -l
#SBATCH --mnist-conv-legacy-launcher-is-deprecated
# MNIST_CONV_LEGACY_LAUNCHER_DEPRECATED
printf "%s\n" "Expected format: python -m experiments.mnist_conv {run|sweep|collect} ..." >&2
printf "%s\n" "Provided deprecated MNIST Conv launcher invocation: $0 $*" >&2
printf "%s\n" "Deprecated MNIST Conv launcher; no setup, allocation, or work was performed." >&2
exit 2
#SBATCH --job-name=mnist_conv1_hs_probe
#SBATCH --output=slurm/mnist_conv1_hs_probe/%x-%A_%a.out
#SBATCH --error=slurm/mnist_conv1_hs_probe/%x-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=04:00:00
#SBATCH --qos=qos_gpu-t3
#SBATCH --partition=gpu_p13
#SBATCH --constraint=v100-16g
#SBATCH --array=0-17%18

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

export PYTHONUNBUFFERED=1
export PYTHONPATH="${SOURCE_ROOT}:${SOURCE_ROOT}/labs:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

PYTHON="${PYTHON:-python}"
TRAIN="${SOURCE_ROOT}/experiments/train_mnist_bp_conv_amplification_sweep.py"
COLLECT="${SOURCE_ROOT}/experiments/collect_mnist_bp_conv1_hardsigmoid_lr_inputgain_probe.py"
DATASET_ROOT="${DATASET_ROOT:-${SCRATCH}/datasets/mnist}"
BASE_OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRATCH}/server_code/results/mnist_bp_conv1_hardsigmoid_current_lr_inputgain_probe_seed0_20epoch}"

mkdir -p "${BASE_OUTPUT_ROOT}" slurm/mnist_conv1_hs_probe

if [ "${SUMMARY_ONLY:-0}" = "1" ]; then
  "${PYTHON}" "${COLLECT}" --root "${BASE_OUTPUT_ROOT}" --seed 0
  exit 0
fi

RUN_NAMES=(mnist_bp_amp_v1_c2 mnist_bp_amp_v1_c4)
LRS=(0.006 0.009 0.012)
LR_LABELS=(0p006 0p009 0p012)
INPUT_GAINS=(50.0 100.0 200.0)
INPUT_GAIN_LABELS=(50 100 200)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
NUM_RUNS="${#RUN_NAMES[@]}"
NUM_LRS="${#LRS[@]}"
NUM_INPUT_GAINS="${#INPUT_GAINS[@]}"
TOTAL_TASKS=$((NUM_RUNS * NUM_LRS * NUM_INPUT_GAINS))
if [ "${TASK_ID}" -lt 0 ] || [ "${TASK_ID}" -ge "${TOTAL_TASKS}" ]; then
  echo "Task ${TASK_ID} is outside 0..$((TOTAL_TASKS - 1))" >&2
  exit 2
fi

RUN_INDEX=$((TASK_ID % NUM_RUNS))
LR_INDEX=$(((TASK_ID / NUM_RUNS) % NUM_LRS))
INPUT_GAIN_INDEX=$((TASK_ID / (NUM_RUNS * NUM_LRS)))

RUN_NAME="${RUN_NAMES[$RUN_INDEX]}"
LR="${LRS[$LR_INDEX]}"
LR_LABEL="${LR_LABELS[$LR_INDEX]}"
INPUT_GAIN="${INPUT_GAINS[$INPUT_GAIN_INDEX]}"
INPUT_GAIN_LABEL="${INPUT_GAIN_LABELS[$INPUT_GAIN_INDEX]}"
OUTPUT_ROOT="${BASE_OUTPUT_ROOT}/input_gain_${INPUT_GAIN_LABEL}/lr_${LR_LABEL}"

echo "[probe] job=${SLURM_JOB_ID:-manual} task=${TASK_ID}/${TOTAL_TASKS}"
echo "[probe] run_name=${RUN_NAME} lr=${LR} input_gain=${INPUT_GAIN}"
echo "[probe] output_root=${OUTPUT_ROOT}"

srun "${PYTHON}" "${TRAIN}" \
  --output-root "${OUTPUT_ROOT}" \
  --dataset-root "${DATASET_ROOT}" \
  --device cuda \
  --model-key mnist_bp_conv_amp \
  --seeds 0 \
  --non-linearity hard_sigmoid \
  --run-name "${RUN_NAME}" \
  --epochs "${EPOCHS:-20}" \
  --batch-size "${BATCH_SIZE:-4}" \
  --num-iterations "${NUM_ITERATIONS:-4}" \
  --learning-rate "${LR}" \
  --beta "${BETA:-1.0}" \
  --lr-decay "${LR_DECAY:-0.99}" \
  --conv-depth 1 \
  --conv-channels 64 128 256 \
  --kernel-size 3 \
  --stride 2 \
  --padding 0 \
  --output-dim 20 \
  --weight-gains 1.0 \
  --weight-min 0.0 \
  --weight-max 100.0 \
  --weight-init-mode kaiming_uniform \
  --input-gain "${INPUT_GAIN}" \
  --hard-sigmoid-g-on "${HARD_SIGMOID_G_ON:-100.0}" \
  --hard-sigmoid-g-off "${HARD_SIGMOID_G_OFF:-0.0}" \
  --hard-sigmoid-v-min "${HARD_SIGMOID_V_MIN:--1.5}" \
  --hard-sigmoid-v-max "${HARD_SIGMOID_V_MAX:-1.5}" \
  --normalize-mean 0.1307 \
  --normalize-std 0.3081 \
  --normalize-scale 0.3 \
  --no-download
