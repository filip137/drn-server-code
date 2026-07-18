#!/bin/bash -l
#SBATCH --mnist-conv-legacy-launcher-is-deprecated
# MNIST_CONV_LEGACY_LAUNCHER_DEPRECATED
printf "%s\n" "Expected format: python -m experiments.mnist_conv {run|sweep|collect} ..." >&2
printf "%s\n" "Provided deprecated MNIST Conv launcher invocation: $0 $*" >&2
printf "%s\n" "Deprecated MNIST Conv launcher; no setup, allocation, or work was performed." >&2
exit 2
#SBATCH --job-name=mnist_conv3_pad1_final
#SBATCH --output=slurm/mnist_conv3_pad1_final/%x-%A_%a.out
#SBATCH --error=slurm/mnist_conv3_pad1_final/%x-%A_%a.err
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
#SBATCH --array=0-29%30

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
SETTINGS_CSV="${SETTINGS_CSV:-${SCRATCH}/server_code/results/mnist_bp_conv3_pad1_final_settings.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRATCH}/server_code/results/mnist_bp_conv3_pad1_final_paper_seed012_50epoch}"
DATASET_ROOT="${DATASET_ROOT:-${SCRATCH}/datasets/mnist}"
SUMMARY_ONLY="${SUMMARY_ONLY:-0}"

if [ ! -f "${SETTINGS_CSV}" ]; then
  echo "Expected frozen settings CSV at ${SETTINGS_CSV}" >&2
  echo "Required columns: non_linearity,run_name,input_gain,iteration_count,lr" >&2
  exit 2
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${SOURCE_ROOT}:${SOURCE_ROOT}/labs:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

mkdir -p "${OUTPUT_ROOT}" slurm/mnist_conv3_pad1_final

if [ "${SUMMARY_ONLY}" = "1" ]; then
  "${PYTHON}" "${TRAIN}" \
    --output-root "${OUTPUT_ROOT}" \
    --dataset-root "${DATASET_ROOT}" \
    --device cuda \
    --model-key mnist_bp_conv_amp \
    --seeds 0 1 2 \
    --epochs "${EPOCHS:-50}" \
    --batch-size "${BATCH_SIZE:-4}" \
    --conv-depth 3 \
    --conv-channels 64 128 256 \
    --kernel-size 3 \
    --stride 2 \
    --padding 1 \
    --output-dim 20 \
    --non-linearity hard_sigmoid perfect_diode \
    --summary-only
  exit 0
fi

task_id="${SLURM_ARRAY_TASK_ID:-0}"
setting_index=$((task_id / 3))
seed_index=$((task_id % 3))
SEEDS=(0 1 2)
seed="${SEEDS[$seed_index]}"

setting_line="$("${PYTHON}" - "${SETTINGS_CSV}" "${setting_index}" <<'PY'
import csv, sys
path, index = sys.argv[1], int(sys.argv[2])
rows = list(csv.DictReader(open(path, newline="")))
if index >= len(rows):
    raise SystemExit(0)
row = rows[index]
print(row["non_linearity"], row["run_name"], row["input_gain"], row["iteration_count"], row["lr"])
PY
)"
if [ -z "${setting_line}" ]; then
  echo "task_id=${task_id} outside frozen settings; exiting."
  exit 0
fi
read -r nonlinearity run_name input_gain selected_k learning_rate <<< "${setting_line}"

echo "[conv3-final] task=${task_id} seed=${seed} nonlinearity=${nonlinearity} run=${run_name}"
echo "[conv3-final] gain=${input_gain} lr=${learning_rate} K=${selected_k} output=${OUTPUT_ROOT}"

srun "${PYTHON}" "${TRAIN}" \
  --output-root "${OUTPUT_ROOT}" \
  --dataset-root "${DATASET_ROOT}" \
  --device cuda \
  --model-key mnist_bp_conv_amp \
  --seeds "${seed}" \
  --epochs "${EPOCHS:-50}" \
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
