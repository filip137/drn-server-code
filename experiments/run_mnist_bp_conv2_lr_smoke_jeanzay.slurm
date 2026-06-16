#!/bin/bash -l
#SBATCH --job-name=mnist_conv2_lr_smoke
#SBATCH --output=slurm/mnist_conv2_lr_smoke/%x-%A_%a.out
#SBATCH --error=slurm/mnist_conv2_lr_smoke/%x-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=04:00:00
#SBATCH --qos=qos_gpu-t3
#SBATCH --partition=gpu_p13
#SBATCH --constraint=v100-16g
#SBATCH --array=0-39%40

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

DATASET_ROOT="${DATASET_ROOT:-${SCRATCH}/datasets/mnist}"
BASE_OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRATCH}/server_code/results/mnist_bp_conv2_lr_smoke_64_128ch_s2_valid_iter6_seed0}"
SUMMARY_ONLY="${SUMMARY_ONLY:-0}"

LR_VALUES=(0.003 0.006 0.009 0.012)
LR_LABELS=(0p003 0p006 0p009 0p012)
NONLINEARITIES=(hard_sigmoid perfect_diode)
RUN_NAMES=(
  mnist_bp_amp_v1_c1
  mnist_bp_amp_v2_c1
  mnist_bp_amp_v4_c1
  mnist_bp_amp_v1_c2
  mnist_bp_amp_v1_c4
)

mkdir -p "${BASE_OUTPUT_ROOT}" slurm/mnist_conv2_lr_smoke

common_args=(
  --dataset-root "${DATASET_ROOT}"
  --device cuda
  --model-key mnist_bp_conv_amp
  --seeds 0
  --epochs "${EPOCHS:-5}"
  --batch-size "${BATCH_SIZE:-4}"
  --num-iterations "${NUM_ITERATIONS:-6}"
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
  --max-batches "${MAX_BATCHES:-1500}"
  --max-test-batches "${MAX_TEST_BATCHES:-1000}"
  --log-interval "${LOG_INTERVAL:-250}"
  --no-download
)

if [ "${SUMMARY_ONLY}" = "1" ]; then
  for idx in "${!LR_VALUES[@]}"; do
    "${PYTHON}" "${TRAIN}" \
      "${common_args[@]}" \
      --output-root "${BASE_OUTPUT_ROOT}/lr_${LR_LABELS[$idx]}" \
      --learning-rate "${LR_VALUES[$idx]}" \
      --non-linearity "${NONLINEARITIES[@]}" \
      --summary-only
  done
  "${PYTHON}" "${COLLECT}" --root "${BASE_OUTPUT_ROOT}"
  exit 0
fi

task_id="${SLURM_ARRAY_TASK_ID:-0}"
num_lrs="${#LR_VALUES[@]}"
num_runs="${#RUN_NAMES[@]}"
lr_index=$((task_id % num_lrs))
run_index=$(((task_id / num_lrs) % num_runs))
nonlinearity_index=$((task_id / (num_lrs * num_runs)))

if [ "${nonlinearity_index}" -ge "${#NONLINEARITIES[@]}" ]; then
  echo "Invalid task_id=${task_id}; expected 0-39." >&2
  exit 2
fi

lr="${LR_VALUES[$lr_index]}"
lr_label="${LR_LABELS[$lr_index]}"
run_name="${RUN_NAMES[$run_index]}"
nonlinearity="${NONLINEARITIES[$nonlinearity_index]}"
output_root="${BASE_OUTPUT_ROOT}/lr_${lr_label}"

echo "[jean-zay] job=${SLURM_JOB_ID:-manual} array_task=${task_id}"
echo "[jean-zay] source_root=${SOURCE_ROOT}"
echo "[jean-zay] dataset_root=${DATASET_ROOT}"
echo "[jean-zay] output_root=${output_root}"
echo "[jean-zay] nonlinearity=${nonlinearity} run_name=${run_name} seed=0 lr=${lr}"

srun "${PYTHON}" "${TRAIN}" \
  "${common_args[@]}" \
  --output-root "${output_root}" \
  --learning-rate "${lr}" \
  --non-linearity "${nonlinearity}" \
  --run-name "${run_name}"
