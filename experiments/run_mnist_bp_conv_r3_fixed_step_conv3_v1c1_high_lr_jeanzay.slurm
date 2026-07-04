#!/bin/bash -l
#SBATCH --job-name=mnist_conv_c3v1_hi
#SBATCH --output=slurm/mnist_conv_c3v1_hi/%x-%A_%a.out
#SBATCH --error=slurm/mnist_conv_c3v1_hi/%x-%A_%a.err
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
RESULT_ROOT="${RESULT_ROOT:-${FMU_SCRATCH}/server_code/results/mnist_bp_conv_fixed_step_redo_r3_settled_tk_20260703}"
SELECTED_TK_CSV="${SELECTED_TK_CSV:-${RESULT_ROOT}/tk_diagnostics/selected_tk.csv}"
CAL_ROOT="${CAL_ROOT:-${RESULT_ROOT}/hs_sat30_calibration}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${RESULT_ROOT}/lr_screen_seed0_10epoch_conv3_v1c1_highlr_20260704}"
DATASET_ROOT="${DATASET_ROOT:-${UMG_SCRATCH}/datasets/mnist}"
MINIMIZER_CONFIG="${MINIMIZER_CONFIG:-${SOURCE_ROOT}/labs/configs/mnist_minimizer_fixed_iterations.json}"
PYTHON="${PYTHON:-python}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-4}"

cd "${SOURCE_ROOT}"
mkdir -p "${OUTPUT_ROOT}" slurm/mnist_conv_c3v1_hi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${SOURCE_ROOT}:${SOURCE_ROOT}/labs:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

RUN_NAME="mnist_bp_amp_v1_c1"
DEPTH=3
CHANNELS=(64 128 256)
STRIDES=(2 2 1)
PADDINGS=(1 1 1)
NONLINEARITY="hard_sigmoid"
HS_LR_MULT="${HS_LR_MULT:-2}"

read -r -a LR_FACTORS <<< "${LR_FACTORS:-3.0 4.0 6.0 8.0 12.0}"

task_id="${SLURM_ARRAY_TASK_ID:-0}"
factor_count="${#LR_FACTORS[@]}"
if [ "${task_id}" -lt 0 ] || [ "${task_id}" -ge "${factor_count}" ]; then
  echo "Expected SLURM_ARRAY_TASK_ID in [0,$((factor_count - 1))], got ${task_id}." >&2
  exit 2
fi
lr_factor="${LR_FACTORS[${task_id}]}"

if [ ! -s "${SELECTED_TK_CSV}" ]; then
  echo "Expected selected T/K CSV at ${SELECTED_TK_CSV}" >&2
  exit 2
fi

read -r T K < <("${PYTHON}" - "${SELECTED_TK_CSV}" "${DEPTH}" "${NONLINEARITY}" <<'PY'
import csv
import sys

path, depth, nonlinearity = sys.argv[1], int(sys.argv[2]), sys.argv[3]
with open(path, newline="") as handle:
    for row in csv.DictReader(handle):
        if int(float(row["conv_depth"])) == depth and row["non_linearity"] == nonlinearity:
            print(int(float(row["selected_T"])), int(float(row["selected_K"])))
            raise SystemExit(0)
raise SystemExit(f"missing selected T/K for depth={depth} nonlinearity={nonlinearity}")
PY
)

cal_csv="${CAL_ROOT}/targets/conv3_${RUN_NAME}_sat30.csv"
if [ ! -s "${cal_csv}" ]; then
  echo "Expected hard-sigmoid calibration CSV at ${cal_csv}" >&2
  exit 2
fi

read -r input_gain base_lr < <("${PYTHON}" - "${cal_csv}" <<'PY'
import csv
import sys

with open(sys.argv[1], newline="") as handle:
    for row in csv.DictReader(handle):
        if abs(float(row["target_saturation"]) - 0.30) < 1e-8:
            print(f"{float(row['input_gain']):.12g}", f"{float(row['learning_rate']):.12g}")
            raise SystemExit(0)
raise SystemExit("missing target_saturation=0.30 row")
PY
)

lr_center="$("${PYTHON}" - "${base_lr}" "${HS_LR_MULT}" <<'PY'
import sys
print(f"{float(sys.argv[1]) * float(sys.argv[2]):.12g}")
PY
)"
learning_rate="$("${PYTHON}" - "${lr_center}" "${lr_factor}" <<'PY'
import sys
print(f"{float(sys.argv[1]) * float(sys.argv[2]):.12g}")
PY
)"

label_float() {
  local value="$1"
  value="${value//./p}"
  value="${value//-/m}"
  printf '%s' "${value}"
}

gain_label="$(label_float "${input_gain}")"
center_label="$(label_float "${lr_center}")"
factor_label="$(label_float "${lr_factor}")"
lr_label="$(label_float "${learning_rate}")"
job_root="${OUTPUT_ROOT}/conv3/${NONLINEARITY}/${RUN_NAME}/input_gain_${gain_label}/T_${T}_K_${K}/lr_center_${center_label}/lr_factor_${factor_label}/lr_${lr_label}"

echo "[conv3-v1c1-high-lr] task=${task_id}/${factor_count} factor=${lr_factor}"
echo "[conv3-v1c1-high-lr] T=${T} K=${K} gain=${input_gain} center=${lr_center} lr=${learning_rate}"
echo "[conv3-v1c1-high-lr] output=${job_root}"

srun "${PYTHON}" "${SOURCE_ROOT}/experiments/train_mnist_bp_conv_amplification_sweep.py" \
  --output-root "${job_root}" \
  --dataset-root "${DATASET_ROOT}" \
  --device cuda \
  --model-key mnist_bp_conv_amp \
  --non-linearity "${NONLINEARITY}" \
  --seeds 0 \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --beta 1.0 \
  --lr-decay 0.99 \
  --conv-depth "${DEPTH}" \
  --conv-channels "${CHANNELS[@]}" \
  --kernel-size 3 \
  --strides "${STRIDES[@]}" \
  --paddings "${PADDINGS[@]}" \
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
  --run-name "${RUN_NAME}"
