#!/bin/bash -l
#SBATCH --mnist-conv-legacy-launcher-is-deprecated
# MNIST_CONV_LEGACY_LAUNCHER_DEPRECATED
printf "%s\n" "Expected format: python -m experiments.mnist_conv {run|sweep|collect} ..." >&2
printf "%s\n" "Provided deprecated MNIST Conv launcher invocation: $0 $*" >&2
printf "%s\n" "Deprecated MNIST Conv launcher; no setup, allocation, or work was performed." >&2
exit 2
#SBATCH --job-name=mnist_c12_p1_voff
#SBATCH --output=slurm/mnist_c12_p1_voff/%x-%A_%a.out
#SBATCH --error=slurm/mnist_c12_p1_voff/%x-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --account=fmu@v100
#SBATCH --qos=qos_gpu-t3
#SBATCH --partition=gpu_p13
#SBATCH --constraint=v100-16g
#SBATCH --array=0-7%8

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
OUTPUT_ROOT="${OUTPUT_ROOT:-${FMU_SCRATCH}/server_code/results/mnist_bp_conv12_s2s2_pad1_hardsigmoid_sat30_v1c1_v4c1_voff_trainability_k4k6_seed0_10_30epoch}"
DATASET_ROOT="${DATASET_ROOT:-${UMG_SCRATCH}/datasets/mnist}"
PYTHON="${PYTHON:-python}"
BATCH_SIZE="${BATCH_SIZE:-4}"

cd "${SOURCE_ROOT}"
mkdir -p "${OUTPUT_ROOT}" slurm/mnist_c12_p1_voff

export PYTHONUNBUFFERED=1
export PYTHONPATH="${SOURCE_ROOT}:${SOURCE_ROOT}/labs:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

CONV_DEPTHS=(1 1 1 1 2 2 2 2)
RUN_NAMES=(
  mnist_bp_amp_v1_c1
  mnist_bp_amp_v1_c1
  mnist_bp_amp_v4_c1
  mnist_bp_amp_v4_c1
  mnist_bp_amp_v1_c1
  mnist_bp_amp_v1_c1
  mnist_bp_amp_v4_c1
  mnist_bp_amp_v4_c1
)
INPUT_GAINS=(
  65.65764617919922
  65.65764617919922
  18.72927474975586
  18.72927474975586
  221.06837463378906
  221.06837463378906
  158.0352325439453
  158.0352325439453
)
LEARNING_RATES=(
  0.00548298668852
  0.00548298668852
  0.0192212461406
  0.0192212461406
  0.0260552872365
  0.0260552872365
  0.00455594609132
  0.00455594609132
)
VOFF_MODES=(fixed trainable fixed trainable fixed trainable fixed trainable)
EPOCHS_BY_TASK=(10 10 10 10 30 30 30 30)
K_BY_TASK=(4 4 4 4 6 6 6 6)
STRIDES_BY_TASK=("2" "2" "2" "2" "2 2" "2 2" "2 2" "2 2")
PADDINGS_BY_TASK=("1" "1" "1" "1" "1 1" "1 1" "1 1" "1 1")

task_id="${SLURM_ARRAY_TASK_ID:-0}"
if [ "${task_id}" -ge "${#RUN_NAMES[@]}" ]; then
  echo "task_id=${task_id} outside run matrix"
  exit 0
fi

conv_depth="${CONV_DEPTHS[$task_id]}"
run_name="${RUN_NAMES[$task_id]}"
input_gain="${INPUT_GAINS[$task_id]}"
learning_rate="${LEARNING_RATES[$task_id]}"
voff_mode="${VOFF_MODES[$task_id]}"
epochs="${EPOCHS_BY_TASK[$task_id]}"
num_iterations="${K_BY_TASK[$task_id]}"
read -r -a strides <<< "${STRIDES_BY_TASK[$task_id]}"
read -r -a paddings <<< "${PADDINGS_BY_TASK[$task_id]}"

label_float() {
  local value="$1"
  value="${value//./p}"
  value="${value//-/m}"
  printf '%s' "${value}"
}

gain_label="$(label_float "${input_gain}")"
lr_label="$(label_float "${learning_rate}")"
job_root="${OUTPUT_ROOT}/conv${conv_depth}/target_sat30/input_gain_${gain_label}/voff_${voff_mode}/lr_${lr_label}"

hard_sigmoid_args=(--hard-sigmoid-v-min -4.0 --hard-sigmoid-v-max 4.0)
if [ "${voff_mode}" = "trainable" ]; then
  hard_sigmoid_args=(
    --hard-sigmoid-v-off 4.0
    --trainable-hard-sigmoid-v-off
    --hard-sigmoid-v-off-min 0.0
  )
fi

echo "[conv12-pad1-voff] job=${SLURM_JOB_ID:-manual} task=${task_id} project=${IDRPROJ:-unset}"
echo "[conv12-pad1-voff] conv=${conv_depth} run=${run_name} mode=${voff_mode} input_gain=${input_gain} lr=${learning_rate}"
echo "[conv12-pad1-voff] source=${SOURCE_ROOT} dataset=${DATASET_ROOT} output=${job_root}"
echo "[conv12-pad1-voff] strides=${strides[*]} paddings=${paddings[*]} K=${num_iterations} epochs=${epochs} target=sat30"

srun "${PYTHON}" "${SOURCE_ROOT}/experiments/train_mnist_bp_conv_amplification_sweep.py" \
  --output-root "${job_root}" \
  --dataset-root "${DATASET_ROOT}" \
  --device cuda \
  --model-key mnist_bp_conv_amp \
  --non-linearity hard_sigmoid \
  --seeds 0 \
  --epochs "${epochs}" \
  --batch-size "${BATCH_SIZE}" \
  --beta 1.0 \
  --lr-decay 0.99 \
  --conv-depth "${conv_depth}" \
  --conv-channels 64 128 256 \
  --kernel-size 3 \
  --strides "${strides[@]}" \
  --paddings "${paddings[@]}" \
  --output-dim 20 \
  --weight-gains 1.0 \
  --weight-min 0.0 \
  --weight-max 100.0 \
  --weight-init-mode kaiming_uniform \
  --input-gain "${input_gain}" \
  --num-iterations "${num_iterations}" \
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

echo "[conv12-pad1-voff] done task=${task_id} conv=${conv_depth} run=${run_name} mode=${voff_mode}"
