#!/bin/bash -l
#SBATCH --mnist-conv-legacy-launcher-is-deprecated
# MNIST_CONV_LEGACY_LAUNCHER_DEPRECATED
printf "%s\n" "Expected format: python -m experiments.mnist_conv {run|sweep|collect} ..." >&2
printf "%s\n" "Provided deprecated MNIST Conv launcher invocation: $0 $*" >&2
printf "%s\n" "Deprecated MNIST Conv launcher; no setup, allocation, or work was performed." >&2
exit 2
#SBATCH --job-name=mnist_c3_r3_cal
#SBATCH --output=slurm/mnist_c3_r3_cal/%x-%A_%a.out
#SBATCH --error=slurm/mnist_c3_r3_cal/%x-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=01:30:00
#SBATCH --account=fmu@v100
#SBATCH --qos=qos_gpu-t3
#SBATCH --partition=gpu_p13
#SBATCH --constraint=v100-16g
#SBATCH --array=0-6%7

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
OUTPUT_ROOT="${OUTPUT_ROOT:-${FMU_SCRATCH}/server_code/results/mnist_bp_conv3_pad1_hardsigmoid_amp_targets_voff4_sat10_20_30_50_k8_seed0}"
DATASET_ROOT="${DATASET_ROOT:-${UMG_SCRATCH}/datasets/mnist}"
PYTHON="${PYTHON:-python}"

cd "${SOURCE_ROOT}"
mkdir -p "${OUTPUT_ROOT}/targets" slurm/mnist_c3_r3_cal

export PYTHONUNBUFFERED=1
export PYTHONPATH="${SOURCE_ROOT}:${SOURCE_ROOT}/labs:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

RUN_NAMES=(
  mnist_bp_amp_v1_c1
  mnist_bp_amp_v2_c1
  mnist_bp_amp_v4_c1
  mnist_bp_amp_v1_c2
  mnist_bp_amp_v1_c4
  mnist_bp_amp_v4_c0p25
  mnist_bp_amp_v2_c2
)

task_id="${SLURM_ARRAY_TASK_ID:-0}"
if [ "${task_id}" -ge "${#RUN_NAMES[@]}" ]; then
  echo "task_id=${task_id} outside calibration matrix"
  exit 0
fi
run_name="${RUN_NAMES[$task_id]}"
output_csv="${OUTPUT_ROOT}/targets/conv3_voff4_k8_${run_name}_targets.csv"

echo "[conv3-r3-cal] job=${SLURM_JOB_ID:-manual} task=${task_id} project=${IDRPROJ:-unset}"
echo "[conv3-r3-cal] run_name=${run_name} output=${output_csv}"
echo "[conv3-r3-cal] source=${SOURCE_ROOT} dataset=${DATASET_ROOT}"

srun "${PYTHON}" "${SOURCE_ROOT}/experiments/search_mnist_bp_conv_hardsigmoid_saturation_targets.py" \
  --dataset-root "${DATASET_ROOT}" \
  --output-csv "${output_csv}" \
  --device cuda \
  --seed 0 \
  --num-samples 256 \
  --batch-size 64 \
  --num-iterations 8 \
  --run-name "${run_name}" \
  --conv-depth 3 \
  --conv-channels 64 128 256 \
  --kernel-size 3 \
  --stride 2 \
  --padding 1 \
  --output-dim 20 \
  --v-off 4.0 \
  --saturation-scope first_hidden \
  --targets 0.1 0.2 0.3 0.5 \
  --lr-numerator 1.44

echo "[conv3-r3-cal] done ${output_csv}"
