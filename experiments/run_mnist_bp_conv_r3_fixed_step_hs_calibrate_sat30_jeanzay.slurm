#!/bin/bash -l
#SBATCH --mnist-conv-legacy-launcher-is-deprecated
# MNIST_CONV_LEGACY_LAUNCHER_DEPRECATED
printf "%s\n" "Expected format: python -m experiments.mnist_conv {run|sweep|collect} ..." >&2
printf "%s\n" "Provided deprecated MNIST Conv launcher invocation: $0 $*" >&2
printf "%s\n" "Deprecated MNIST Conv launcher; no setup, allocation, or work was performed." >&2
exit 2
#SBATCH --job-name=mnist_conv_fix_cal
#SBATCH --output=slurm/mnist_conv_fix_cal/%x-%A_%a.out
#SBATCH --error=slurm/mnist_conv_fix_cal/%x-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=00:30:00
#SBATCH --account=fmu@v100
#SBATCH --qos=qos_gpu-dev
#SBATCH --partition=gpu_p13
#SBATCH --constraint=v100-16g
#SBATCH --array=0-5%6

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
OUTPUT_ROOT="${OUTPUT_ROOT:-${RESULT_ROOT}/hs_sat30_calibration}"
DATASET_ROOT="${DATASET_ROOT:-${UMG_SCRATCH}/datasets/mnist}"
MINIMIZER_CONFIG="${MINIMIZER_CONFIG:-${SOURCE_ROOT}/labs/configs/mnist_minimizer_fixed_iterations.json}"
PYTHON="${PYTHON:-python}"

cd "${SOURCE_ROOT}"
mkdir -p "${OUTPUT_ROOT}/targets" slurm/mnist_conv_fix_cal

export PYTHONUNBUFFERED=1
export PYTHONPATH="${SOURCE_ROOT}:${SOURCE_ROOT}/labs:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

RUN_NAMES=(mnist_bp_amp_v1_c1 mnist_bp_amp_v4_c1 mnist_bp_amp_v4_c0p25)
task_id="${SLURM_ARRAY_TASK_ID:-0}"
arch="$([ "$((task_id / 3))" -eq 0 ] && printf conv2 || printf conv3)"
run_name="${RUN_NAMES[$((task_id % 3))]}"

case "${arch}" in
  conv2)
    depth=2
    channels=(64 128)
    strides=(2 2)
    paddings=(1 1)
    ;;
  conv3)
    depth=3
    channels=(64 128 256)
    strides=(2 2 1)
    paddings=(1 1 1)
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

T="$("${PYTHON}" - "${SELECTED_TK_CSV}" "${depth}" "hard_sigmoid" <<'PY'
import csv
import sys

path, depth, nonlinearity = sys.argv[1], int(sys.argv[2]), sys.argv[3]
with open(path, newline="") as handle:
    for row in csv.DictReader(handle):
        if int(float(row["conv_depth"])) == depth and row["non_linearity"] == nonlinearity:
            print(int(float(row["selected_T"])))
            raise SystemExit(0)
raise SystemExit(f"missing selected T/K for depth={depth} nonlinearity={nonlinearity}")
PY
)"

output_csv="${OUTPUT_ROOT}/targets/${arch}_${run_name}_sat30.csv"
echo "[hs-calibrate] arch=${arch} run=${run_name} T=${T} output=${output_csv}"

srun "${PYTHON}" "${SOURCE_ROOT}/experiments/search_mnist_bp_conv_hardsigmoid_saturation_targets.py" \
  --dataset-root "${DATASET_ROOT}" \
  --output-csv "${output_csv}" \
  --device cuda \
  --seed 0 \
  --num-samples 256 \
  --batch-size 64 \
  --num-iterations-inference "${T}" \
  --minimizer-config "${MINIMIZER_CONFIG}" \
  --run-name "${run_name}" \
  --conv-depth "${depth}" \
  --conv-channels "${channels[@]}" \
  --kernel-size 3 \
  --strides "${strides[@]}" \
  --paddings "${paddings[@]}" \
  --output-dim 20 \
  --v-off 4.0 \
  --saturation-scope first_hidden \
  --targets 0.30 \
  --lr-numerator 1.44
