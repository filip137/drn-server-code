#!/bin/bash -l
#SBATCH --job-name=mnist_conv2_amp_calib
#SBATCH --output=slurm/mnist_conv2_amp_calib/%x-%j.out
#SBATCH --error=slurm/mnist_conv2_amp_calib/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=04:00:00
#SBATCH --account=fmu@v100
#SBATCH --qos=qos_gpu-t3
#SBATCH --partition=gpu_p13
#SBATCH --constraint=v100-16g

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
PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_ROOT="${DATASET_ROOT:-${UMG_SCRATCH}/datasets/mnist}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${FMU_SCRATCH}/server_code/results/mnist_bp_conv2_hardsigmoid_amp_calibrated_saturation_targets_voff4_sat10_30_50_70_sat50_lr012_seed0_10epoch}"
TARGET_ROOT="${TARGET_ROOT:-${OUTPUT_ROOT}/targets}"
CALIBRATION_CSV="${CALIBRATION_CSV:-${TARGET_ROOT}/conv2_amp_calibrated_targets.csv}"
MANIFEST_CSV="${MANIFEST_CSV:-${TARGET_ROOT}/conv2_amp_calibrated_manifest.csv}"
TRAIN_SLURM="${TRAIN_SLURM:-${SOURCE_ROOT}/experiments/run_mnist_bp_conv_amp_calibrated_manifest_jeanzay.slurm}"
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-42}"

cd "${SOURCE_ROOT}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${SOURCE_ROOT}:${SOURCE_ROOT}/labs:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

mkdir -p "${TARGET_ROOT}" slurm/mnist_conv2_amp_calib

echo "[conv2-calib] source_root=${SOURCE_ROOT}"
echo "[conv2-calib] output_root=${OUTPUT_ROOT}"
echo "[conv2-calib] calibration_csv=${CALIBRATION_CSV}"
echo "[conv2-calib] manifest_csv=${MANIFEST_CSV}"
echo "[conv2-calib] dataset_root=${DATASET_ROOT}"

srun "${PYTHON_BIN}" "${SOURCE_ROOT}/experiments/search_mnist_bp_conv_hardsigmoid_saturation_targets.py" \
  --dataset-root "${DATASET_ROOT}" \
  --output-csv "${CALIBRATION_CSV}" \
  --device cuda \
  --seed 0 \
  --num-samples "${SEARCH_SAMPLES:-256}" \
  --batch-size "${SEARCH_BATCH_SIZE:-64}" \
  --num-iterations 6 \
  --conv-depth 2 \
  --conv-channels 64 128 256 \
  --kernel-size 3 \
  --stride 2 \
  --padding 0 \
  --output-dim 20 \
  --v-off 4.0 \
  --saturation-scope first_hidden \
  --targets 0.1 0.3 0.5 0.7 \
  --search-steps "${SEARCH_STEPS:-24}" \
  --lr-numerator 1.44 \
  --all-amplifications

"${PYTHON_BIN}" "${SOURCE_ROOT}/experiments/build_mnist_bp_conv_amp_calibrated_training_manifest.py" \
  --calibration-csv "${CALIBRATION_CSV}" \
  --output-csv "${MANIFEST_CSV}" \
  --output-root "${OUTPUT_ROOT}" \
  --targets 0.1 0.3 0.5 0.7 \
  --sat50-lr-multipliers 0.5 1.0 2.0

row_count="$("${PYTHON_BIN}" - "${MANIFEST_CSV}" <<'PY'
import csv
import sys
with open(sys.argv[1], newline="") as handle:
    print(sum(1 for _ in csv.DictReader(handle)))
PY
)"
if [ "${row_count}" -ne 42 ]; then
  echo "Expected 42 manifest rows, got ${row_count}" >&2
  exit 2
fi

train_job_id="$(sbatch --parsable \
  --array="0-41%${ARRAY_CONCURRENCY}" \
  --export=ALL,SOURCE_ROOT="${SOURCE_ROOT}",MANIFEST_CSV="${MANIFEST_CSV}",DATASET_ROOT="${DATASET_ROOT}",NUM_SHARDS=42,IDRPROJ_OVERRIDE="${PROJECT}" \
  "${TRAIN_SLURM}")"
echo "${train_job_id}" > "${TARGET_ROOT}/conv2_training_array_job_id.txt"
echo "[conv2-calib] submitted training_array_job_id=${train_job_id}"
