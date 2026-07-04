#!/bin/bash -l
#SBATCH --job-name=pd_grad_survey
#SBATCH --output=slurm/pd_grad_survey/%x-%j.out
#SBATCH --error=slurm/pd_grad_survey/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=01:00:00
#SBATCH --account=fmu@v100
#SBATCH --qos=qos_gpu-dev
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
DATASET_ROOT="${DATASET_ROOT:-${UMG_SCRATCH}/datasets/mnist}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${FMU_SCRATCH}/server_code/results/perfect_diode_gradient_survey_20260701}"
PYTHON="${PYTHON:-python}"

cd "${SOURCE_ROOT}"
mkdir -p "${OUTPUT_ROOT}" slurm/pd_grad_survey

export PYTHONUNBUFFERED=1
export PYTHONPATH="${SOURCE_ROOT}:${SOURCE_ROOT}/labs:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

REPEAT_ROOT="${FMU_SCRATCH}/server_code/results/mnist_bp_conv123_s221_pad1_perfect_diode_v1c1_v4c1_bestparams_k4k8_seed0_10_30_50epoch"
RESCUE_ROOT="${FMU_SCRATCH}/server_code/results/mnist_bp_conv23_s221_pad1_perfect_diode_v1c1_gain_lr_downbias_bestlr_k4k8_seed0_30_50epoch"
OUT_CSV="${OUTPUT_ROOT}/perfect_diode_conv123_gradient_survey_final_train256_gpu.csv"

echo "[pd-grad-survey] job=${SLURM_JOB_ID:-manual} project=${IDRPROJ:-unset}"
echo "[pd-grad-survey] source=${SOURCE_ROOT}"
echo "[pd-grad-survey] dataset=${DATASET_ROOT}"
echo "[pd-grad-survey] output=${OUT_CSV}"

srun "${PYTHON}" "${SOURCE_ROOT}/experiments/analyze_mnist_bp_conv_good_run_gradient_saturation.py" \
  --run-dir "${REPEAT_ROOT}/conv1/epochs_10/input_gain_40/lr_0p006/k_4/perfect_diode/mnist_bp_amp_v1_c1/seed_0" \
  --run-dir "${REPEAT_ROOT}/conv1/epochs_10/input_gain_40/lr_0p009/k_4/perfect_diode/mnist_bp_amp_v4_c1/seed_0" \
  --run-dir "${REPEAT_ROOT}/conv2/epochs_30/input_gain_100/lr_0p024/k_4/perfect_diode/mnist_bp_amp_v1_c1/seed_0" \
  --run-dir "${REPEAT_ROOT}/conv2/epochs_30/input_gain_100/lr_0p012/k_4/perfect_diode/mnist_bp_amp_v4_c1/seed_0" \
  --run-dir "${REPEAT_ROOT}/conv3/epochs_50/input_gain_360/lr_0p0432/k_8/perfect_diode/mnist_bp_amp_v1_c1/seed_0" \
  --run-dir "${REPEAT_ROOT}/conv3/epochs_50/input_gain_360/lr_0p0864/k_8/perfect_diode/mnist_bp_amp_v4_c1/seed_0" \
  --run-dir "${RESCUE_ROOT}/conv2/epochs_50/input_gain_400/lr_0p012/k_4/perfect_diode/mnist_bp_amp_v1_c1/seed_0" \
  --run-dir "${RESCUE_ROOT}/conv3/epochs_50/input_gain_480/lr_0p0864/k_8/perfect_diode/mnist_bp_amp_v1_c1/seed_0" \
  --output-csv "${OUT_CSV}" \
  --dataset-root "${DATASET_ROOT}" \
  --device cuda \
  --phase final \
  --split train \
  --batch-size "${BATCH_SIZE:-32}" \
  --max-samples "${MAX_SAMPLES:-256}" \
  --no-download

echo "[pd-grad-survey] done"
