#!/bin/bash -l
#SBATCH --mnist-conv-legacy-launcher-is-deprecated
# MNIST_CONV_LEGACY_LAUNCHER_DEPRECATED
printf "%s\n" "Expected format: python -m experiments.mnist_conv {run|sweep|collect} ..." >&2
printf "%s\n" "Provided deprecated MNIST Conv launcher invocation: $0 $*" >&2
printf "%s\n" "Deprecated MNIST Conv launcher; no setup, allocation, or work was performed." >&2
exit 2
#SBATCH --job-name=mnist_c2_ab_train
#SBATCH --output=slurm/mnist_c2_ab_train/%x-%A_%a.out
#SBATCH --error=slurm/mnist_c2_ab_train/%x-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=12:00:00
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
PYTHON="${PYTHON:-python}"
DATASET_ROOT="${DATASET_ROOT:-${UMG_SCRATCH}/datasets/mnist}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${FMU_SCRATCH}/server_code/results/mnist_bp_conv2_hardsigmoid_trainable_amp_sat30_voff4_seed0_30epoch}"

cd "${SOURCE_ROOT}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${SOURCE_ROOT}/labs:${SOURCE_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

RUN_NAMES=(mnist_bp_amp_v1_c1 mnist_bp_amp_v4_c1)
INPUT_GAINS=(221.06837463378906 158.0352325439453)
LEARNING_RATES=(0.00651382180914 0.00911189218265)
GAIN_LABELS=(221p068374634 158p035232544)
LR_LABELS=(0p00651382180914 0p00911189218265)

task_id="${SLURM_ARRAY_TASK_ID:-0}"
if [ "${task_id}" -ge "${#RUN_NAMES[@]}" ]; then
  echo "task_id=${task_id} outside trainable A/B Conv2 matrix"
  exit 0
fi

run_name="${RUN_NAMES[$task_id]}"
input_gain="${INPUT_GAINS[$task_id]}"
learning_rate="${LEARNING_RATES[$task_id]}"
gain_label="${GAIN_LABELS[$task_id]}"
lr_label="${LR_LABELS[$task_id]}"
job_root="${OUTPUT_ROOT}/target_sat30/input_gain_${gain_label}/lr_${lr_label}"

mkdir -p "${job_root}" slurm/mnist_c2_ab_train

extra_train_args=()
if [ -n "${MAX_BATCHES:-}" ]; then
  extra_train_args+=(--max-batches "${MAX_BATCHES}")
fi
if [ -n "${MAX_TEST_BATCHES:-}" ]; then
  extra_train_args+=(--max-test-batches "${MAX_TEST_BATCHES}")
fi

echo "[conv2-trainable-ab] job=${SLURM_JOB_ID:-manual} task=${task_id}"
echo "[conv2-trainable-ab] source=${SOURCE_ROOT} dataset=${DATASET_ROOT}"
echo "[conv2-trainable-ab] output=${job_root}"
echo "[conv2-trainable-ab] run=${run_name} target=sat30 input_gain=${input_gain} lr=${learning_rate} v_off=4 K=6 epochs=${EPOCHS:-30}"
echo "[conv2-trainable-ab] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"

srun "${PYTHON}" "${SOURCE_ROOT}/experiments/train_mnist_bp_conv_amplification_sweep.py" \
  --output-root "${job_root}" \
  --dataset-root "${DATASET_ROOT}" \
  --device cuda \
  --model-key mnist_bp_conv_amp \
  --non-linearity hard_sigmoid \
  --seeds 0 \
  --epochs "${EPOCHS:-30}" \
  --batch-size "${BATCH_SIZE:-4}" \
  --beta 1.0 \
  --lr-decay 0.99 \
  --conv-depth 2 \
  --conv-channels 64 128 256 \
  --kernel-size 3 \
  --stride 2 \
  --padding 0 \
  --output-dim 20 \
  --weight-gains 1.0 \
  --weight-min 0.0 \
  --weight-max 100.0 \
  --weight-init-mode kaiming_uniform \
  --input-gain "${input_gain}" \
  --num-iterations 6 \
  --learning-rate "${learning_rate}" \
  --hard-sigmoid-g-on 100.0 \
  --hard-sigmoid-g-off 0.0 \
  --hard-sigmoid-v-min -4.0 \
  --hard-sigmoid-v-max 4.0 \
  --normalize-mean 0.1307 \
  --normalize-std 0.3081 \
  --normalize-scale 0.3 \
  --log-interval "${LOG_INTERVAL:-500}" \
  --no-download \
  --run-name "${run_name}" \
  --trainable-amplification \
  --amplification-min "${AMPLIFICATION_MIN:-1e-6}" \
  "${extra_train_args[@]}"
