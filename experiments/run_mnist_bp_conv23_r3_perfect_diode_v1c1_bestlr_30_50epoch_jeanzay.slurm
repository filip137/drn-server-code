#!/bin/bash -l
#SBATCH --mnist-conv-legacy-launcher-is-deprecated
# MNIST_CONV_LEGACY_LAUNCHER_DEPRECATED
printf "%s\n" "Expected format: python -m experiments.mnist_conv {run|sweep|collect} ..." >&2
printf "%s\n" "Provided deprecated MNIST Conv launcher invocation: $0 $*" >&2
printf "%s\n" "Deprecated MNIST Conv launcher; no setup, allocation, or work was performed." >&2
exit 2
#SBATCH --job-name=mnist_c23_pd_bestlr
#SBATCH --output=slurm/mnist_c23_pd_bestlr/%x-%A_%a.out
#SBATCH --error=slurm/mnist_c23_pd_bestlr/%x-%A_%a.err
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
#SBATCH --array=0-3%4

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
GRID_ROOT="${GRID_ROOT:-${FMU_SCRATCH}/server_code/results/mnist_bp_conv23_s221_pad1_perfect_diode_v1c1_gain_lr_downbias_k4k8_seed0_10epoch}"
SELECTED_CSV="${SELECTED_CSV:-${GRID_ROOT}/selected_gain_lr_by_conv_depth.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${FMU_SCRATCH}/server_code/results/mnist_bp_conv23_s221_pad1_perfect_diode_v1c1_gain_lr_downbias_bestlr_k4k8_seed0_30_50epoch}"
DATASET_ROOT="${DATASET_ROOT:-${UMG_SCRATCH}/datasets/mnist}"
PYTHON="${PYTHON:-python}"
RUN_NAME="mnist_bp_amp_v1_c1"
EPOCH_OPTIONS=(30 50)

cd "${SOURCE_ROOT}"
mkdir -p "${OUTPUT_ROOT}" slurm/mnist_c23_pd_bestlr

export PYTHONUNBUFFERED=1
export PYTHONPATH="${SOURCE_ROOT}:${SOURCE_ROOT}/labs:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

if [ ! -s "${SELECTED_CSV}" ]; then
  echo "Expected selected gain/LR CSV at ${SELECTED_CSV}" >&2
  exit 2
fi

label_float() {
  local value="$1"
  value="${value//./p}"
  value="${value//-/m}"
  printf '%s' "${value}"
}

task_id="${SLURM_ARRAY_TASK_ID:-0}"
conv_index=$((task_id / ${#EPOCH_OPTIONS[@]}))
epoch_index=$((task_id % ${#EPOCH_OPTIONS[@]}))
conv_depth=$((conv_index + 2))
epochs="${EPOCH_OPTIONS[$epoch_index]}"

if [ "${conv_depth}" -gt 3 ]; then
  echo "task_id=${task_id} outside final-run matrix"
  exit 0
fi

IFS='|' read -r input_gain learning_rate selected_k strides_text paddings_text < <("${PYTHON}" - "${SELECTED_CSV}" "${conv_depth}" <<'PY'
import csv
import sys

path = sys.argv[1]
conv_depth = int(sys.argv[2])
with open(path, newline="") as handle:
    for row in csv.DictReader(handle):
        if int(float(row["conv_depth"])) == conv_depth:
            print("|".join([
                f"{float(row['selected_input_gain']):g}",
                f"{float(row['selected_lr']):g}",
                str(int(float(row["selected_iteration_count"]))),
                row["strides"],
                row["paddings"],
            ]))
            break
    else:
        raise SystemExit(f"missing selected row for conv_depth={conv_depth}")
PY
)
read -r -a strides <<< "${strides_text}"
read -r -a paddings <<< "${paddings_text}"

gain_label="$(label_float "${input_gain}")"
lr_label="$(label_float "${learning_rate}")"
job_root="${OUTPUT_ROOT}/conv${conv_depth}/epochs_${epochs}/input_gain_${gain_label}/lr_${lr_label}/k_${selected_k}"

if [ "${task_id}" = "0" ]; then
  cp "${SELECTED_CSV}" "${OUTPUT_ROOT}/selected_gain_lr_by_conv_depth.csv"
fi

echo "[conv23-pd-bestlr] job=${SLURM_JOB_ID:-manual} task=${task_id} project=${IDRPROJ:-unset}"
echo "[conv23-pd-bestlr] conv_depth=${conv_depth} run=${RUN_NAME} input_gain=${input_gain} lr=${learning_rate}"
echo "[conv23-pd-bestlr] epochs=${epochs} K=${selected_k} strides=${strides[*]} paddings=${paddings[*]}"
echo "[conv23-pd-bestlr] source=${SOURCE_ROOT} dataset=${DATASET_ROOT} output=${job_root}"

srun "${PYTHON}" "${SOURCE_ROOT}/experiments/train_mnist_bp_conv_amplification_sweep.py" \
  --output-root "${job_root}" \
  --dataset-root "${DATASET_ROOT}" \
  --device cuda \
  --model-key mnist_bp_conv_amp \
  --non-linearity perfect_diode \
  --seeds 0 \
  --epochs "${epochs}" \
  --batch-size "${BATCH_SIZE:-4}" \
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
  --num-iterations "${selected_k}" \
  --learning-rate "${learning_rate}" \
  --hard-sigmoid-g-on 100.0 \
  --hard-sigmoid-g-off 0.0 \
  --hard-sigmoid-v-min -1.5 \
  --hard-sigmoid-v-max 1.5 \
  --normalize-mean 0.1307 \
  --normalize-std 0.3081 \
  --normalize-scale 0.3 \
  --log-interval "${LOG_INTERVAL:-1000}" \
  --no-download \
  --run-name "${RUN_NAME}"

echo "[conv23-pd-bestlr] done task=${task_id} conv_depth=${conv_depth} epochs=${epochs}"
