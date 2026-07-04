#!/bin/bash -l
#SBATCH --job-name=mnist_conv_fix_train
#SBATCH --output=slurm/mnist_conv_fix_train/%x-%A_%a.out
#SBATCH --error=slurm/mnist_conv_fix_train/%x-%A_%a.err
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
#SBATCH --array=0-35%12

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
RESULT_ROOT="${RESULT_ROOT:-${FMU_SCRATCH}/server_code/results/mnist_bp_conv_fixed_step_redo_r3_20260703}"
PHASE="${PHASE:-long}"
if [ "${PHASE}" = "long" ]; then
  SETTINGS_CSV="${SETTINGS_CSV:-${RESULT_ROOT}/lr_screen_seed0_10epoch/selected_final_settings.csv}"
  OUTPUT_ROOT="${OUTPUT_ROOT:-${RESULT_ROOT}/long_check_seed0_50epoch}"
  EPOCHS="${EPOCHS:-50}"
  SEEDS=(${SEEDS:-0})
elif [ "${PHASE}" = "final" ]; then
  SETTINGS_CSV="${SETTINGS_CSV:-${RESULT_ROOT}/long_check_seed0_50epoch/selected_final_settings.csv}"
  OUTPUT_ROOT="${OUTPUT_ROOT:-${RESULT_ROOT}/final_seeds012_50epoch}"
  EPOCHS="${EPOCHS:-50}"
  SEEDS=(${SEEDS:-0 1 2})
else
  echo "Expected PHASE to be long or final, got ${PHASE}." >&2
  exit 2
fi
DATASET_ROOT="${DATASET_ROOT:-${UMG_SCRATCH}/datasets/mnist}"
MINIMIZER_CONFIG="${MINIMIZER_CONFIG:-${SOURCE_ROOT}/labs/configs/mnist_minimizer_fixed_iterations.json}"
PYTHON="${PYTHON:-python}"

cd "${SOURCE_ROOT}"
mkdir -p "${OUTPUT_ROOT}" slurm/mnist_conv_fix_train

export PYTHONUNBUFFERED=1
export PYTHONPATH="${SOURCE_ROOT}:${SOURCE_ROOT}/labs:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

if [ ! -s "${SETTINGS_CSV}" ]; then
  echo "Expected selected settings CSV at ${SETTINGS_CSV}" >&2
  exit 2
fi

num_seeds="${#SEEDS[@]}"
num_rows="$("${PYTHON}" - "${SETTINGS_CSV}" <<'PY'
import csv
import sys
with open(sys.argv[1], newline="") as handle:
    print(sum(1 for _ in csv.DictReader(handle)))
PY
)"
num_tasks=$((num_rows * num_seeds))

run_one_task() {
task_id="$1"
if [ "${task_id}" -ge "${num_tasks}" ]; then
  echo "task_id=${task_id} outside matrix size=${num_tasks}"
  return 0
fi

row_index=$((task_id / num_seeds))
seed="${SEEDS[$((task_id % num_seeds))]}"

IFS=$'\t' read -r conv_depth strides_csv paddings_csv nonlinearity run_name input_gain T K learning_rate < <("${PYTHON}" - "${SETTINGS_CSV}" "${row_index}" <<'PY'
import csv
import sys
path, index = sys.argv[1], int(sys.argv[2])
with open(path, newline="") as handle:
    rows = list(csv.DictReader(handle))
row = rows[index]
print(
    row["conv_depth"],
    row["strides"],
    row["paddings"],
    row["non_linearity"],
    row["run_name"],
    f"{float(row['input_gain']):.12g}",
    int(float(row.get("num_iterations_inference") or row["iteration_count"])),
    int(float(row.get("num_iterations_training") or row["iteration_count"])),
    f"{float(row['lr']):.12g}",
    sep="\t",
)
PY
)

IFS=',' read -r -a strides <<< "${strides_csv}"
IFS=',' read -r -a paddings <<< "${paddings_csv}"
if [ "${conv_depth}" = "2" ]; then
  arch=conv2
  channels=(64 128)
elif [ "${conv_depth}" = "3" ]; then
  arch=conv3
  channels=(64 128 256)
else
  echo "Expected conv_depth 2 or 3, got ${conv_depth}." >&2
  exit 2
fi

label_float() {
  local value="$1"
  value="${value//./p}"
  value="${value//-/m}"
  printf '%s' "${value}"
}
gain_label="$(label_float "${input_gain}")"
lr_label="$(label_float "${learning_rate}")"
job_root="${OUTPUT_ROOT}/${arch}/input_gain_${gain_label}/T_${T}_K_${K}/lr_${lr_label}"

echo "[train-selected] phase=${PHASE} arch=${arch} nonlinearity=${nonlinearity} run=${run_name} seed=${seed}"
echo "[train-selected] gain=${input_gain} T=${T} K=${K} lr=${learning_rate} output=${job_root}"

srun "${PYTHON}" "${SOURCE_ROOT}/experiments/train_mnist_bp_conv_amplification_sweep.py" \
  --output-root "${job_root}" \
  --dataset-root "${DATASET_ROOT}" \
  --device cuda \
  --model-key mnist_bp_conv_amp \
  --non-linearity "${nonlinearity}" \
  --seeds "${seed}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE:-4}" \
  --beta 1.0 \
  --lr-decay 0.99 \
  --conv-depth "${conv_depth}" \
  --conv-channels "${channels[@]}" \
  --kernel-size 3 \
  --strides "${strides[@]}" \
  --paddings "${paddings[@]}" \
  --output-dim 20 \
  --weight-gains 1.0 \
  --weight-min 0.0 \
  --weight-max 100.0 \
  --weight-init-mode kaiming_uniform \
  --input-gain "${input_gain}" \
  --num-iterations-inference "${T}" \
  --num-iterations-training "${K}" \
  --learning-rate "${learning_rate}" \
  --minimizer-config "${MINIMIZER_CONFIG}" \
  --hard-sigmoid-g-on 100.0 \
  --hard-sigmoid-g-off 0.0 \
  --hard-sigmoid-v-min -4.0 \
  --hard-sigmoid-v-max 4.0 \
  --normalize-mean 0.1307 \
  --normalize-std 0.3081 \
  --normalize-scale 0.3 \
  --log-interval "${LOG_INTERVAL:-500}" \
  --no-download \
  --run-name "${run_name}"
}

if [ "${TASK_PACK_STRIDE:-0}" -gt 0 ]; then
  task_id="${SLURM_ARRAY_TASK_ID:-0}"
  while [ "${task_id}" -lt "${TASK_PACK_TOTAL:-${num_tasks}}" ]; do
    run_one_task "${task_id}"
    task_id=$((task_id + TASK_PACK_STRIDE))
  done
else
  run_one_task "${SLURM_ARRAY_TASK_ID:-0}"
fi
