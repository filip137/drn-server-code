#!/bin/bash -l
#SBATCH --mnist-conv-legacy-launcher-is-deprecated
# MNIST_CONV_LEGACY_LAUNCHER_DEPRECATED
printf "%s\n" "Expected format: python -m experiments.mnist_conv {run|sweep|collect} ..." >&2
printf "%s\n" "Provided deprecated MNIST Conv launcher invocation: $0 $*" >&2
printf "%s\n" "Deprecated MNIST Conv launcher; no setup, allocation, or work was performed." >&2
exit 2
#SBATCH --job-name=mnist_conv3_pad1_ig
#SBATCH --output=slurm/mnist_conv3_pad1_ig/%x-%A_%a.out
#SBATCH --error=slurm/mnist_conv3_pad1_ig/%x-%A_%a.err
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
#SBATCH --array=0-49%30

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
K_CSV="${K_CSV:-${SCRATCH}/server_code/results/mnist_bp_conv3_pad1_k_preflight_seed0/selected_k_by_nonlinearity.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRATCH}/server_code/results/mnist_bp_conv3_pad1_inputgain_screen_seed0_10epoch}"
DATASET_ROOT="${DATASET_ROOT:-${SCRATCH}/datasets/mnist}"
SUMMARY_ONLY="${SUMMARY_ONLY:-0}"

if [ ! -f "${K_CSV}" ]; then
  echo "Expected selected K CSV at ${K_CSV}" >&2
  exit 2
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${SOURCE_ROOT}:${SOURCE_ROOT}/labs:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

mkdir -p "${OUTPUT_ROOT}" slurm/mnist_conv3_pad1_ig

if [ "${SUMMARY_ONLY}" = "1" ]; then
  "${PYTHON}" "${COLLECT}" \
    --root "${OUTPUT_ROOT}" \
    --input-gains 50 75 100 125 150 \
    --non-linearities hard_sigmoid perfect_diode \
    --seed 0
  exit 0
fi

NONLINEARITIES=(hard_sigmoid perfect_diode)
GAINS=(50 75 100 125 150)
GAIN_LABELS=(50 75 100 125 150)
LRS=(0.003 0.0045 0.006 0.0075 0.009)
LR_LABELS=(0p003 0p0045 0p006 0p0075 0p009)
RUN_NAMES=(
  mnist_bp_amp_v1_c1
  mnist_bp_amp_v2_c1
  mnist_bp_amp_v4_c1
  mnist_bp_amp_v1_c2
  mnist_bp_amp_v1_c4
)

task_id="${SLURM_ARRAY_TASK_ID:-0}"
nonlinearity_index=$((task_id / 25))
gain_index=$(((task_id / 5) % 5))
run_index=$((task_id % 5))
if [ "${nonlinearity_index}" -ge "${#NONLINEARITIES[@]}" ]; then
  echo "task_id=${task_id} outside input-gain matrix; exiting."
  exit 0
fi

nonlinearity="${NONLINEARITIES[$nonlinearity_index]}"
input_gain="${GAINS[$gain_index]}"
gain_label="${GAIN_LABELS[$gain_index]}"
learning_rate="${LRS[$gain_index]}"
lr_label="${LR_LABELS[$gain_index]}"
run_name="${RUN_NAMES[$run_index]}"
selected_k="$("${PYTHON}" - "${K_CSV}" "${nonlinearity}" <<'PY'
import csv, sys
path, nonlinearity = sys.argv[1], sys.argv[2]
with open(path, newline="") as handle:
    for row in csv.DictReader(handle):
        if row["non_linearity"] == nonlinearity:
            print(int(float(row["selected_iteration_count"])))
            raise SystemExit(0)
raise SystemExit(f"missing selected K for {nonlinearity}")
PY
)"

job_root="${OUTPUT_ROOT}/input_gain_${gain_label}/lr_${lr_label}/k_${selected_k}"

echo "[conv3-input-gain] task=${task_id} nonlinearity=${nonlinearity} run=${run_name}"
echo "[conv3-input-gain] gain=${input_gain} lr=${learning_rate} K=${selected_k} output=${job_root}"

srun "${PYTHON}" "${TRAIN}" \
  --output-root "${job_root}" \
  --dataset-root "${DATASET_ROOT}" \
  --device cuda \
  --model-key mnist_bp_conv_amp \
  --seeds 0 \
  --epochs "${EPOCHS:-10}" \
  --batch-size "${BATCH_SIZE:-4}" \
  --beta "${BETA:-1.0}" \
  --lr-decay "${LR_DECAY:-0.99}" \
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
  --num-iterations "${selected_k}" \
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
  --non-linearity "${nonlinearity}" \
  --run-name "${run_name}"
