#!/bin/bash -l
#SBATCH --job-name=mnist_conv_pd_ig
#SBATCH --output=slurm/mnist_conv_pd_ig/%x-%A_%a.out
#SBATCH --error=slurm/mnist_conv_pd_ig/%x-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=08:00:00
#SBATCH --account=umg@v100
#SBATCH --qos=qos_gpu-t3
#SBATCH --partition=gpu_p13
#SBATCH --constraint=v100-16g
#SBATCH --array=0-24%25

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
COLLECT="${SOURCE_ROOT}/experiments/collect_mnist_bp_conv_inputgain_screen.py"
DATASET_ROOT="${DATASET_ROOT:-${SCRATCH}/datasets/mnist}"

CONV_DEPTH="${CONV_DEPTH:-1}"
NUM_ITERATIONS="${NUM_ITERATIONS:-4}"
PADDING="${PADDING:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRATCH}/server_code/results/mnist_bp_conv${CONV_DEPTH}_perfect_diode_inputgain_screen_seed0_10epoch}"
SUMMARY_ONLY="${SUMMARY_ONLY:-0}"
PACK_SIZE="${PACK_SIZE:-1}"

read -r -a GAINS <<< "${GAINS:-60 80 100 120 140}"
read -r -a GAIN_LABELS <<< "${GAIN_LABELS:-60 80 100 120 140}"
read -r -a LRS <<< "${LRS:-0.0036 0.0048 0.006 0.0072 0.0084}"
read -r -a LR_LABELS <<< "${LR_LABELS:-0p0036 0p0048 0p006 0p0072 0p0084}"
RUN_NAMES=(
  mnist_bp_amp_v1_c1
  mnist_bp_amp_v2_c1
  mnist_bp_amp_v4_c1
  mnist_bp_amp_v1_c2
  mnist_bp_amp_v1_c4
)

if [ "${#GAINS[@]}" -ne "${#LRS[@]}" ] || [ "${#GAINS[@]}" -ne "${#GAIN_LABELS[@]}" ] || [ "${#GAINS[@]}" -ne "${#LR_LABELS[@]}" ]; then
  echo "Expected GAINS, GAIN_LABELS, LRS, and LR_LABELS to have the same length." >&2
  echo "Provided GAINS='${GAINS[*]}' LRS='${LRS[*]}'" >&2
  exit 2
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${SOURCE_ROOT}:${SOURCE_ROOT}/labs:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

mkdir -p "${OUTPUT_ROOT}" slurm/mnist_conv_pd_ig

if [ "${SUMMARY_ONLY}" = "1" ]; then
  "${PYTHON}" "${COLLECT}" \
    --root "${OUTPUT_ROOT}" \
    --input-gains "${GAINS[@]}" \
    --non-linearities perfect_diode \
    --seed 0
  exit 0
fi

run_task() {
  local task_id="$1"
  local num_gains="${#GAINS[@]}"
  local total_jobs=$((num_gains * 5))
  if [ "${task_id}" -ge "${total_jobs}" ]; then
    return 0
  fi

  local gain_index=$((task_id / 5))
  local run_index=$((task_id % 5))
  local input_gain="${GAINS[$gain_index]}"
  local gain_label="${GAIN_LABELS[$gain_index]}"
  local learning_rate="${LRS[$gain_index]}"
  local lr_label="${LR_LABELS[$gain_index]}"
  local run_name="${RUN_NAMES[$run_index]}"
  local job_root="${OUTPUT_ROOT}/input_gain_${gain_label}/lr_${lr_label}/k_${NUM_ITERATIONS}"

  echo "[conv-perfect-diode-input-gain] task=${task_id} depth=${CONV_DEPTH} run=${run_name}"
  echo "[conv-perfect-diode-input-gain] gain=${input_gain} lr=${learning_rate} K=${NUM_ITERATIONS} output=${job_root}"

  "${PYTHON}" "${TRAIN}" \
    --output-root "${job_root}" \
    --dataset-root "${DATASET_ROOT}" \
    --device cuda \
    --model-key mnist_bp_conv_amp \
    --seeds 0 \
    --epochs "${EPOCHS:-10}" \
    --batch-size "${BATCH_SIZE:-4}" \
    --beta "${BETA:-1.0}" \
    --lr-decay "${LR_DECAY:-0.99}" \
    --conv-depth "${CONV_DEPTH}" \
    --conv-channels 64 128 256 \
    --kernel-size 3 \
    --stride 2 \
    --padding "${PADDING}" \
    --output-dim 20 \
    --weight-gains 1.0 \
    --weight-min 0.0 \
    --weight-max 100.0 \
    --weight-init-mode kaiming_uniform \
    --input-gain "${input_gain}" \
    --num-iterations "${NUM_ITERATIONS}" \
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
echo "[conv-perfect-diode-input-gain] array_task=${array_task_id} pack_size=${PACK_SIZE} base_task=${base_task_id}"

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
