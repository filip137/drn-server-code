#!/bin/bash -l
#SBATCH --job-name=mnist_conv_fix_pre
#SBATCH --output=slurm/mnist_conv_fix_pre/%x-%A_%a.out
#SBATCH --error=slurm/mnist_conv_fix_pre/%x-%A_%a.err
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
#SBATCH --array=0-11%12

set -euo pipefail

PROJECT="${IDRPROJ_OVERRIDE:-fmu}"
if [ -x /gpfslocalsup/bin/idrenv ]; then
  eval "$(/gpfslocalsup/bin/idrenv -d "${PROJECT}")"
elif command -v idrenv >/dev/null 2>&1; then
  eval "$(idrenv -d "${PROJECT}")"
fi
if command -v idrproj >/dev/null 2>&1 && ! idrproj | grep -q "^${PROJECT}\\b"; then
  echo "Expected idrproj to expose project ${PROJECT}." >&2
  idrproj >&2 || true
  exit 2
fi

module purge
module load "${PYTORCH_MODULE:-pytorch-gpu/py3/2.5.0}"

UMG_WORK="${WORK_umg:-/lustre/fswork/projects/rech/umg/${USER}}"
UMG_SCRATCH="${SCRATCH_umg:-/lustre/fsn1/projects/rech/umg/${USER}}"
FMU_SCRATCH="${SCRATCH_fmu:-/lustre/fsn1/projects/rech/fmu/${USER}}"
SOURCE_ROOT="${SOURCE_ROOT:-${UMG_WORK}/server_code}"
RESULT_ROOT="${RESULT_ROOT:-${FMU_SCRATCH}/server_code/results/mnist_bp_conv_fixed_step_redo_r3_20260703}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${RESULT_ROOT}/preflight}"
DATASET_ROOT="${DATASET_ROOT:-${UMG_SCRATCH}/datasets/mnist}"
MINIMIZER_CONFIG="${MINIMIZER_CONFIG:-${SOURCE_ROOT}/labs/configs/mnist_minimizer_fixed_iterations.json}"
PYTHON="${PYTHON:-python}"

cd "${SOURCE_ROOT}"
mkdir -p "${OUTPUT_ROOT}" slurm/mnist_conv_fix_pre

export PYTHONUNBUFFERED=1
export PYTHONPATH="${SOURCE_ROOT}:${SOURCE_ROOT}/labs:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

task_id="${SLURM_ARRAY_TASK_ID:-0}"
RUN_NAMES=(mnist_bp_amp_v1_c1 mnist_bp_amp_v4_c1 mnist_bp_amp_v4_c0p25)
arch_index=$((task_id / 6))
nonlin_index=$(((task_id / 3) % 2))
run_index=$((task_id % 3))
arch="$([ "${arch_index}" -eq 0 ] && printf conv2 || printf conv3)"
nonlinearity="$([ "${nonlin_index}" -eq 0 ] && printf hard_sigmoid || printf perfect_diode)"
run_name="${RUN_NAMES[$run_index]}"

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

job_root="${OUTPUT_ROOT}/${arch}/${nonlinearity}/${run_name}"
echo "[preflight] arch=${arch} nonlinearity=${nonlinearity} run=${run_name} output=${job_root}"

srun "${PYTHON}" "${SOURCE_ROOT}/experiments/train_mnist_bp_conv_amplification_sweep.py" \
  --output-root "${job_root}" \
  --dataset-root "${DATASET_ROOT}" \
  --device cuda \
  --model-key mnist_bp_conv_amp \
  --non-linearity "${nonlinearity}" \
  --seeds 0 \
  --epochs 1 \
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
  --input-gain 100 \
  --num-iterations-inference 4 \
  --num-iterations-training 4 \
  --learning-rate 0.006 \
  --minimizer-config "${MINIMIZER_CONFIG}" \
  --hard-sigmoid-g-on 100.0 \
  --hard-sigmoid-g-off 0.0 \
  --hard-sigmoid-v-min -4.0 \
  --hard-sigmoid-v-max 4.0 \
  --normalize-mean 0.1307 \
  --normalize-std 0.3081 \
  --normalize-scale 0.3 \
  --log-interval 500 \
  --no-download \
  --run-name "${run_name}" \
  --dry-run

"${PYTHON}" - "${job_root}" "${depth}" "${strides[*]}" "${paddings[*]}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected_depth = int(sys.argv[2])
expected_strides = [int(x) for x in sys.argv[3].split()]
expected_paddings = [int(x) for x in sys.argv[4].split()]
configs = list(root.glob("**/source_config.json"))
if len(configs) != 1:
    raise SystemExit(f"Expected exactly one source_config.json under {root}, got {len(configs)}.")
cfg = json.loads(configs[0].read_text())
model = cfg["model_base"]
pipeline = cfg["model_overrides"][cfg["lab"]["model_key"]]["conv_pipeline"]
if len(pipeline) != expected_depth:
    raise SystemExit(f"Expected conv depth {expected_depth}, got {len(pipeline)}.")
if [int(item["stride"]) for item in pipeline] != expected_strides:
    raise SystemExit(f"Unexpected strides: {pipeline}")
if [int(item["padding"]) for item in pipeline] != expected_paddings:
    raise SystemExit(f"Unexpected paddings: {pipeline}")
if model["num_iterations_inference"] != 4 or model["num_iterations_training"] != 4:
    raise SystemExit(f"Unexpected T/K: {model['num_iterations_inference']}/{model['num_iterations_training']}")
if model["minimizer"]["adaptive_equilibrium"] is not False:
    raise SystemExit("Expected adaptive_equilibrium=false.")
print(f"[preflight-ok] {configs[0]}")
PY
