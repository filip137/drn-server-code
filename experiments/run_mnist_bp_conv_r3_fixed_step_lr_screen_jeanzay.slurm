#!/bin/bash -l
#SBATCH --job-name=mnist_conv_fix_lr
#SBATCH --output=slurm/mnist_conv_fix_lr/%x-%A_%a.out
#SBATCH --error=slurm/mnist_conv_fix_lr/%x-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=08:00:00
#SBATCH --account=fmu@v100
#SBATCH --qos=qos_gpu-t3
#SBATCH --partition=gpu_p13
#SBATCH --constraint=v100-16g
#SBATCH --array=0-35%18

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
SELECTED_TK_CSV="${SELECTED_TK_CSV:-${RESULT_ROOT}/tk_diagnostics/selected_tk.csv}"
CAL_ROOT="${CAL_ROOT:-${RESULT_ROOT}/hs_sat30_calibration}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${RESULT_ROOT}/lr_screen_seed0_10epoch}"
DATASET_ROOT="${DATASET_ROOT:-${UMG_SCRATCH}/datasets/mnist}"
MINIMIZER_CONFIG="${MINIMIZER_CONFIG:-${SOURCE_ROOT}/labs/configs/mnist_minimizer_fixed_iterations.json}"
PYTHON="${PYTHON:-python}"
EPOCHS="${EPOCHS:-10}"

cd "${SOURCE_ROOT}"
mkdir -p "${OUTPUT_ROOT}" slurm/mnist_conv_fix_lr

export PYTHONUNBUFFERED=1
export PYTHONPATH="${SOURCE_ROOT}:${SOURCE_ROOT}/labs:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

RUN_NAMES=(mnist_bp_amp_v1_c1 mnist_bp_amp_v4_c1 mnist_bp_amp_v4_c0p25)
LR_FACTORS=(0.5 1.0 2.0)

run_one_task() {
task_id="$1"
factor_index=$((task_id % 3))
run_index=$(((task_id / 3) % 3))
nonlin_index=$(((task_id / 9) % 2))
arch_index=$((task_id / 18))
arch="$([ "${arch_index}" -eq 0 ] && printf conv2 || printf conv3)"
nonlinearity="$([ "${nonlin_index}" -eq 0 ] && printf hard_sigmoid || printf perfect_diode)"
run_name="${RUN_NAMES[$run_index]}"
lr_factor="${LR_FACTORS[$factor_index]}"

case "${arch}" in
  conv2)
    depth=2
    channels=(64 128)
    strides=(2 2)
    paddings=(1 1)
    pd_gain=100
    pd_center_lr=0.012
    case "${run_name}" in
      mnist_bp_amp_v1_c1) hs_lr_mult=4 ;;
      mnist_bp_amp_v4_c1) hs_lr_mult=0.5 ;;
      mnist_bp_amp_v4_c0p25) hs_lr_mult=0.25 ;;
    esac
    ;;
  conv3)
    depth=3
    channels=(64 128 256)
    strides=(2 2 1)
    paddings=(1 1 1)
    pd_gain=360
    pd_center_lr=0.0216
    case "${run_name}" in
      mnist_bp_amp_v1_c1) hs_lr_mult=2 ;;
      mnist_bp_amp_v4_c1) hs_lr_mult=2 ;;
      mnist_bp_amp_v4_c0p25) hs_lr_mult=0.5 ;;
    esac
    ;;
  *)
    echo "Unknown arch ${arch}" >&2
    exit 2
    ;;
esac

if [ ! -s "${SELECTED_TK_CSV}" ]; then
  echo "Expected selected T/K CSV at ${SELECTED_TK_CSV}" >&2
  exit 2
fi

read -r T K < <("${PYTHON}" - "${SELECTED_TK_CSV}" "${depth}" "${nonlinearity}" <<'PY'
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

if [ "${nonlinearity}" = "hard_sigmoid" ]; then
  cal_csv="${CAL_ROOT}/targets/${arch}_${run_name}_sat30.csv"
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
  lr_center="$(awk -v lr="${base_lr}" -v mult="${hs_lr_mult}" 'BEGIN { printf "%.12g", lr * mult }')"
else
  input_gain="${pd_gain}"
  lr_center="${pd_center_lr}"
fi

learning_rate="$(awk -v lr="${lr_center}" -v factor="${lr_factor}" 'BEGIN { printf "%.12g", lr * factor }')"
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
job_root="${OUTPUT_ROOT}/${arch}/${nonlinearity}/${run_name}/input_gain_${gain_label}/T_${T}_K_${K}/lr_center_${center_label}/lr_factor_${factor_label}/lr_${lr_label}"

echo "[lr-screen] arch=${arch} nonlinearity=${nonlinearity} run=${run_name} T=${T} K=${K}"
echo "[lr-screen] gain=${input_gain} center=${lr_center} factor=${lr_factor} lr=${learning_rate} output=${job_root}"

srun "${PYTHON}" "${SOURCE_ROOT}/experiments/train_mnist_bp_conv_amplification_sweep.py" \
  --output-root "${job_root}" \
  --dataset-root "${DATASET_ROOT}" \
  --device cuda \
  --model-key mnist_bp_conv_amp \
  --non-linearity "${nonlinearity}" \
  --seeds 0 \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE:-4}" \
  --beta 1.0 \
  --lr-decay 0.99 \
  --conv-depth "${depth}" \
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
  while [ "${task_id}" -lt "${TASK_PACK_TOTAL:-36}" ]; do
    run_one_task "${task_id}"
    task_id=$((task_id + TASK_PACK_STRIDE))
  done
else
  run_one_task "${SLURM_ARRAY_TASK_ID:-0}"
fi
