#!/bin/bash -l
#SBATCH --job-name=mnist_c3_r3_legacy_lowg
#SBATCH --output=slurm/mnist_c3_r3_legacy_lowg/%x-%A_%a.out
#SBATCH --error=slurm/mnist_c3_r3_legacy_lowg/%x-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=03:00:00
#SBATCH --account=fmu@v100
#SBATCH --qos=qos_gpu-t3
#SBATCH --partition=gpu_p13
#SBATCH --constraint=v100-16g
#SBATCH --array=0-4%5

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
OUTPUT_ROOT="${OUTPUT_ROOT:-${FMU_SCRATCH}/server_code/results/mnist_bp_conv3_pad1_hardsigmoid_legacy_lowgain_voff4_k8_seed0_10epoch}"
DATASET_ROOT="${DATASET_ROOT:-${UMG_SCRATCH}/datasets/mnist}"
PYTHON="${PYTHON:-python}"
EPOCHS="${EPOCHS:-10}"
LR_NUMERATOR="${LR_NUMERATOR:-1.44}"
LR_MULT="${LR_MULT:-0.5}"
V_OFF="${V_OFF:-4.0}"

cd "${SOURCE_ROOT}"
mkdir -p "${OUTPUT_ROOT}" slurm/mnist_c3_r3_legacy_lowg

export PYTHONUNBUFFERED=1
export PYTHONPATH="${SOURCE_ROOT}:${SOURCE_ROOT}/labs:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

INPUT_GAINS=(50 100 150 200 250)
RUN_NAME="mnist_bp_amp_v4_c0p25"

task_id="${SLURM_ARRAY_TASK_ID:-0}"
if [ "${task_id}" -ge "${#INPUT_GAINS[@]}" ]; then
  echo "task_id=${task_id} outside legacy low-gain matrix"
  exit 0
fi

input_gain="${INPUT_GAINS[$task_id]}"
learning_rate="$(awk -v num="${LR_NUMERATOR}" -v gain="${input_gain}" -v mult="${LR_MULT}" 'BEGIN { printf "%.12g", (num / gain) * mult }')"
label_float() {
  local value="$1"
  value="${value//./p}"
  value="${value//-/m}"
  printf '%s' "${value}"
}
gain_label="$(label_float "${input_gain}")"
mult_label="$(label_float "${LR_MULT}")"
lr_label="$(label_float "${learning_rate}")"
job_root="${OUTPUT_ROOT}/input_gain_${gain_label}/lr_mult_${mult_label}/lr_${lr_label}"
v_min="-${V_OFF}"

echo "[conv3-r3-legacy-lowgain] job=${SLURM_JOB_ID:-manual} task=${task_id} project=${IDRPROJ:-unset}"
echo "[conv3-r3-legacy-lowgain] run=${RUN_NAME} gain=${input_gain} lr_mult=${LR_MULT} lr=${learning_rate}"
echo "[conv3-r3-legacy-lowgain] source=${SOURCE_ROOT} dataset=${DATASET_ROOT} output=${job_root}"

EXTRA_TRAIN_ARGS=()
if [ -n "${MAX_BATCHES:-}" ]; then
  EXTRA_TRAIN_ARGS+=(--max-batches "${MAX_BATCHES}")
fi
if [ -n "${MAX_TEST_BATCHES:-}" ]; then
  EXTRA_TRAIN_ARGS+=(--max-test-batches "${MAX_TEST_BATCHES}")
fi

srun "${PYTHON}" "${SOURCE_ROOT}/experiments/train_mnist_bp_conv_amplification_sweep.py" \
  --output-root "${job_root}" \
  --dataset-root "${DATASET_ROOT}" \
  --device cuda \
  --model-key mnist_bp_conv_amp \
  --non-linearity hard_sigmoid \
  --seeds 0 \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE:-4}" \
  --beta 1.0 \
  --lr-decay 0.99 \
  --conv-depth 3 \
  --conv-channels 64 128 256 \
  --kernel-size 3 \
  --stride 2 \
  --padding 1 \
  --output-dim 20 \
  --weight-gains 1.0 \
  --weight-min 0.0 \
  --weight-max 100.0 \
  --weight-init-mode kaiming_uniform \
  --input-gain "${input_gain}" \
  --num-iterations 8 \
  --learning-rate "${learning_rate}" \
  --hard-sigmoid-g-on 100.0 \
  --hard-sigmoid-g-off 0.0 \
  --hard-sigmoid-v-min "${v_min}" \
  --hard-sigmoid-v-max "${V_OFF}" \
  --normalize-mean 0.1307 \
  --normalize-std 0.3081 \
  --normalize-scale 0.3 \
  --log-interval "${LOG_INTERVAL:-500}" \
  "${EXTRA_TRAIN_ARGS[@]}" \
  --no-download \
  --run-name "${RUN_NAME}"

echo "[conv3-r3-legacy-lowgain] done task=${task_id} gain=${input_gain}"
