#!/bin/bash -l
#SBATCH --job-name=mnist_conv2_epbpK
#SBATCH --output=slurm/mnist_conv2_epbpK/%x-%A_%a.out
#SBATCH --error=slurm/mnist_conv2_epbpK/%x-%A_%a.err
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
#SBATCH --array=0-9%10

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
EPOCHS_VALUE="${EPOCHS:-50}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRATCH}/server_code/results/mnist_bp_conv2_epbpK_large_${EPOCHS_VALUE}epoch_64_128ch_s2_valid_seed0}"
DATASET_ROOT="${DATASET_ROOT:-${SCRATCH}/datasets/mnist}"
SUMMARY_ONLY="${SUMMARY_ONLY:-0}"

if [ ! -f "${TRAIN}" ]; then
  echo "Expected conv sweep runner at ${TRAIN}" >&2
  exit 2
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${SOURCE_ROOT}:${SOURCE_ROOT}/labs:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

mkdir -p "${OUTPUT_ROOT}" slurm/mnist_conv2_epbpK

read -r -a SEEDS <<< "${SEEDS:-0}"

common_args=(
  --output-root "${OUTPUT_ROOT}"
  --dataset-root "${DATASET_ROOT}"
  --device cuda
  --model-key mnist_bp_conv_amp
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

if [ "${SUMMARY_ONLY}" = "1" ]; then
  "${PYTHON}" "${TRAIN}" \
    "${common_args[@]}" \
    --seeds "${SEEDS[@]}" \
    --num-iterations "${SUMMARY_ITERATIONS_PLACEHOLDER:-4}" \
    --learning-rate "${SUMMARY_LR_PLACEHOLDER:-0.006}" \
    --non-linearity hard_sigmoid perfect_diode \
    --summary-only
  exit 0
fi

task_id="${SLURM_ARRAY_TASK_ID:-0}"
num_configs="${#RUN_NAMES[@]}"
num_seeds="${#SEEDS[@]}"
total_tasks=$((num_configs * num_seeds))
if [ "${task_id}" -ge "${total_tasks}" ]; then
  echo "task_id=${task_id} is outside total_tasks=${total_tasks}; exiting."
  exit 0
fi

config_id=$((task_id / num_seeds))
seed_id=$((task_id % num_seeds))
seed="${SEEDS[$seed_id]}"
nonlinearity="${NONLINEARITIES[$config_id]}"
run_name="${RUN_NAMES[$config_id]}"
learning_rate="${LEARNING_RATES[$config_id]}"
num_iterations="${ITERATION_COUNTS[$config_id]}"

echo "[jean-zay] job=${SLURM_JOB_ID:-manual} array_task=${task_id}/${total_tasks}"
echo "[jean-zay] source_root=${SOURCE_ROOT}"
echo "[jean-zay] dataset_root=${DATASET_ROOT}"
echo "[jean-zay] output_root=${OUTPUT_ROOT}"
echo "[jean-zay] seed=${seed} nonlinearity=${nonlinearity} run_name=${run_name}"
echo "[jean-zay] lr=${learning_rate} num_iterations=${num_iterations} epochs=${EPOCHS_VALUE}"
echo "[jean-zay] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"

"${PYTHON}" "${TRAIN}" \
  "${common_args[@]}" \
  --seeds "${seed}" \
  --num-iterations "${num_iterations}" \
  --learning-rate "${learning_rate}" \
  --non-linearity "${nonlinearity}" \
  --run-name "${run_name}"
