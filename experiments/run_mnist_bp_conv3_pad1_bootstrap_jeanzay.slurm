#!/bin/bash -l
#SBATCH --job-name=mnist_conv3_pad1_boot
#SBATCH --output=slurm/mnist_conv3_pad1_boot/%x-%A_%a.out
#SBATCH --error=slurm/mnist_conv3_pad1_boot/%x-%A_%a.err
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
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRATCH}/server_code/results/mnist_bp_conv3_pad1_bootstrap_k16_inputgain100_seed0_10epoch}"
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

mkdir -p "${OUTPUT_ROOT}" slurm/mnist_conv3_pad1_boot

common_args=(
  --output-root "${OUTPUT_ROOT}"
  --dataset-root "${DATASET_ROOT}"
  --device cuda
  --model-key mnist_bp_conv_amp
  --seeds 0
  --epochs "${EPOCHS:-10}"
  --batch-size "${BATCH_SIZE:-4}"
  --beta "${BETA:-1.0}"
  --lr-decay "${LR_DECAY:-0.99}"
  --conv-depth 3
  --conv-channels 64 128 256
  --kernel-size 3
  --stride 2
  --padding 1
  --output-dim 20
  --weight-gains 1.0
  --weight-min 0.0
  --weight-max 100.0
  --weight-init-mode kaiming_uniform
  --input-gain "${INPUT_GAIN:-100.0}"
  --num-iterations "${NUM_ITERATIONS:-16}"
  --learning-rate "${LEARNING_RATE:-0.006}"
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

if [ -n "${MAX_BATCHES:-}" ]; then
  common_args+=(--max-batches "${MAX_BATCHES}")
fi
if [ -n "${MAX_TEST_BATCHES:-}" ]; then
  common_args+=(--max-test-batches "${MAX_TEST_BATCHES}")
fi

if [ "${SUMMARY_ONLY}" = "1" ]; then
  "${PYTHON}" "${TRAIN}" "${common_args[@]}" --non-linearity hard_sigmoid perfect_diode --summary-only
  exit 0
fi

NONLINEARITIES=(hard_sigmoid perfect_diode)
RUN_NAMES=(
  mnist_bp_amp_v1_c1
  mnist_bp_amp_v2_c1
  mnist_bp_amp_v4_c1
  mnist_bp_amp_v1_c2
  mnist_bp_amp_v1_c4
)

task_id="${SLURM_ARRAY_TASK_ID:-0}"
nonlinearity_index=$((task_id / 5))
run_index=$((task_id % 5))
if [ "${nonlinearity_index}" -ge "${#NONLINEARITIES[@]}" ]; then
  echo "task_id=${task_id} outside bootstrap matrix; exiting."
  exit 0
fi
nonlinearity="${NONLINEARITIES[$nonlinearity_index]}"
run_name="${RUN_NAMES[$run_index]}"

echo "[conv3-bootstrap] task=${task_id} nonlinearity=${nonlinearity} run=${run_name}"
echo "[conv3-bootstrap] output_root=${OUTPUT_ROOT}"
echo "[conv3-bootstrap] input_gain=${INPUT_GAIN:-100.0} K=${NUM_ITERATIONS:-16} lr=${LEARNING_RATE:-0.006}"

srun "${PYTHON}" "${TRAIN}" \
  "${common_args[@]}" \
  --non-linearity "${nonlinearity}" \
  --run-name "${run_name}"
