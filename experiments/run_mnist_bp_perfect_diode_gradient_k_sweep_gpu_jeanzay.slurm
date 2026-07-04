#!/bin/bash -l
#SBATCH --job-name=pd_grad_ksweep
#SBATCH --output=slurm/pd_grad_ksweep/%x-%j.out
#SBATCH --error=slurm/pd_grad_ksweep/%x-%j.err
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
OUTPUT_ROOT="${OUTPUT_ROOT:-${FMU_SCRATCH}/server_code/results/perfect_diode_gradient_k_sweep_20260701}"
PYTHON="${PYTHON:-python}"

cd "${SOURCE_ROOT}"
mkdir -p "${OUTPUT_ROOT}" slurm/pd_grad_ksweep

export PYTHONUNBUFFERED=1
export PYTHONPATH="${SOURCE_ROOT}:${SOURCE_ROOT}/labs:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

REPEAT_ROOT="${FMU_SCRATCH}/server_code/results/mnist_bp_conv123_s221_pad1_perfect_diode_v1c1_v4c1_bestparams_k4k8_seed0_10_30_50epoch"
RESCUE_ROOT="${FMU_SCRATCH}/server_code/results/mnist_bp_conv23_s221_pad1_perfect_diode_v1c1_gain_lr_downbias_bestlr_k4k8_seed0_30_50epoch"

RUN_ARGS=(
  --run-dir "${REPEAT_ROOT}/conv2/epochs_30/input_gain_100/lr_0p024/k_4/perfect_diode/mnist_bp_amp_v1_c1/seed_0"
  --run-dir "${RESCUE_ROOT}/conv2/epochs_50/input_gain_400/lr_0p012/k_4/perfect_diode/mnist_bp_amp_v1_c1/seed_0"
  --run-dir "${REPEAT_ROOT}/conv2/epochs_30/input_gain_100/lr_0p012/k_4/perfect_diode/mnist_bp_amp_v4_c1/seed_0"
  --run-dir "${RESCUE_ROOT}/conv3/epochs_50/input_gain_480/lr_0p0864/k_8/perfect_diode/mnist_bp_amp_v1_c1/seed_0"
)

echo "[pd-grad-ksweep] job=${SLURM_JOB_ID:-manual} project=${IDRPROJ:-unset}"
echo "[pd-grad-ksweep] source=${SOURCE_ROOT}"
echo "[pd-grad-ksweep] dataset=${DATASET_ROOT}"
echo "[pd-grad-ksweep] output_root=${OUTPUT_ROOT}"

for k in ${K_VALUES:-1 2 4 8 16}; do
  out_csv="${OUTPUT_ROOT}/perfect_diode_gradient_k${k}_init_final_train256_gpu.csv"
  echo "[pd-grad-ksweep] k=${k} out=${out_csv}"
  srun "${PYTHON}" "${SOURCE_ROOT}/experiments/analyze_mnist_bp_conv_good_run_gradient_saturation.py" \
    "${RUN_ARGS[@]}" \
    --output-csv "${out_csv}" \
    --dataset-root "${DATASET_ROOT}" \
    --device cuda \
    --phase both \
    --split train \
    --batch-size "${BATCH_SIZE:-32}" \
    --max-samples "${MAX_SAMPLES:-256}" \
    --inference-iterations "${k}" \
    --training-iterations "${k}" \
    --no-download
done

echo "[pd-grad-ksweep] done"
