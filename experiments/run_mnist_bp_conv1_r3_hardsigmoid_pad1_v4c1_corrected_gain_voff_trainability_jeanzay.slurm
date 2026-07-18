#!/bin/bash -l
#SBATCH --mnist-conv-legacy-launcher-is-deprecated
# MNIST_CONV_LEGACY_LAUNCHER_DEPRECATED
printf "%s\n" "Expected format: python -m experiments.mnist_conv {run|sweep|collect} ..." >&2
printf "%s\n" "Provided deprecated MNIST Conv launcher invocation: $0 $*" >&2
printf "%s\n" "Deprecated MNIST Conv launcher; no setup, allocation, or work was performed." >&2
exit 2
#SBATCH --job-name=mnist_c1_v4_voff_fix
#SBATCH --output=slurm/mnist_c1_v4_voff_fix/%x-%A_%a.out
#SBATCH --error=slurm/mnist_c1_v4_voff_fix/%x-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=06:00:00
#SBATCH --account=fmu@v100
#SBATCH --qos=qos_gpu-t3
#SBATCH --partition=gpu_p13
#SBATCH --constraint=v100-16g
#SBATCH --array=0-1%2

set -euo pipefail

PROJECT="${IDRPROJ_OVERRIDE:-fmu}"
if [ -x /gpfslocalsup/bin/idrenv ]; then
  eval "$(/gpfslocalsup/bin/idrenv -d "${PROJECT}")"
elif command -v idrenv >/dev/null 2>&1; then
  eval "$(idrenv -d "${PROJECT}")"
fi

module purge
module load "${PYTORCH_MODULE:-pytorch-gpu/py3/2.5.0}"

UMG_WORK="${WORK_umg:-/lustre/fswork/projects/rech/umg/${USER}}"
UMG_SCRATCH="${SCRATCH_umg:-/lustre/fsn1/projects/rech/umg/${USER}}"
FMU_SCRATCH="${SCRATCH_fmu:-/lustre/fsn1/projects/rech/fmu/${USER}}"
SOURCE_ROOT="${SOURCE_ROOT:-${UMG_WORK}/server_code}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${FMU_SCRATCH}/server_code/results/mnist_bp_conv1_s2_pad1_hardsigmoid_v4c1_shared_sat30_gain65p657_voff_trainability_seed0_10epoch}"
DATASET_ROOT="${DATASET_ROOT:-${UMG_SCRATCH}/datasets/mnist}"
PYTHON="${PYTHON:-python}"
BATCH_SIZE="${BATCH_SIZE:-4}"

cd "${SOURCE_ROOT}"
mkdir -p "${OUTPUT_ROOT}" slurm/mnist_c1_v4_voff_fix

export PYTHONUNBUFFERED=1
export PYTHONPATH="${SOURCE_ROOT}:${SOURCE_ROOT}/labs:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

VOFF_MODES=(fixed trainable)

task_id="${SLURM_ARRAY_TASK_ID:-0}"
if [ "${task_id}" -ge "${#VOFF_MODES[@]}" ]; then
  echo "task_id=${task_id} outside run matrix"
  exit 0
fi

run_name="mnist_bp_amp_v4_c1"
input_gain="65.65764617919922"
learning_rate="0.00548298668852"
voff_mode="${VOFF_MODES[$task_id]}"

label_float() {
  local value="$1"
  value="${value//./p}"
  value="${value//-/m}"
  printf '%s' "${value}"
}

gain_label="$(label_float "${input_gain}")"
lr_label="$(label_float "${learning_rate}")"
job_root="${OUTPUT_ROOT}/target_sat30_shared_gain/input_gain_${gain_label}/voff_${voff_mode}/lr_${lr_label}"

hard_sigmoid_args=(--hard-sigmoid-v-min -4.0 --hard-sigmoid-v-max 4.0)
if [ "${voff_mode}" = "trainable" ]; then
  hard_sigmoid_args=(
    --hard-sigmoid-v-off 4.0
    --trainable-hard-sigmoid-v-off
    --hard-sigmoid-v-off-min 0.0
  )
fi

echo "[conv1-v4-corrected-voff] job=${SLURM_JOB_ID:-manual} task=${task_id} project=${IDRPROJ:-unset}"
echo "[conv1-v4-corrected-voff] run=${run_name} mode=${voff_mode} input_gain=${input_gain} lr=${learning_rate}"
echo "[conv1-v4-corrected-voff] source=${SOURCE_ROOT} dataset=${DATASET_ROOT} output=${job_root}"
echo "[conv1-v4-corrected-voff] architecture=conv1 channels=64 stride=2 padding=1 K=4 epochs=10"

srun "${PYTHON}" "${SOURCE_ROOT}/experiments/train_mnist_bp_conv_amplification_sweep.py" \
  --output-root "${job_root}" \
  --dataset-root "${DATASET_ROOT}" \
  --device cuda \
  --model-key mnist_bp_conv_amp \
  --non-linearity hard_sigmoid \
  --seeds 0 \
  --epochs 10 \
  --batch-size "${BATCH_SIZE}" \
  --beta 1.0 \
  --lr-decay 0.99 \
  --conv-depth 1 \
  --conv-channels 64 128 256 \
  --kernel-size 3 \
  --strides 2 \
  --paddings 1 \
  --output-dim 20 \
  --weight-gains 1.0 \
  --weight-min 0.0 \
  --weight-max 100.0 \
  --weight-init-mode kaiming_uniform \
  --input-gain "${input_gain}" \
  --num-iterations 4 \
  --learning-rate "${learning_rate}" \
  --hard-sigmoid-g-on 100.0 \
  --hard-sigmoid-g-off 0.0 \
  "${hard_sigmoid_args[@]}" \
  --normalize-mean 0.1307 \
  --normalize-std 0.3081 \
  --normalize-scale 0.3 \
  --log-interval "${LOG_INTERVAL:-1000}" \
  --no-download \
  --run-name "${run_name}"

echo "[conv1-v4-corrected-voff] done task=${task_id} run=${run_name} mode=${voff_mode}"
