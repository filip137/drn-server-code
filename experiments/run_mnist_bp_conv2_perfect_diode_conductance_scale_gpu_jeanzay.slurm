#!/bin/bash -l
#SBATCH --job-name=conv2_pd_cond_gpu
#SBATCH --output=slurm/conv2_pd_cond_gpu/%x-%j.out
#SBATCH --error=slurm/conv2_pd_cond_gpu/%x-%j.err
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
FMU_SCRATCH="${SCRATCH_fmu:-/lustre/fsn1/projects/rech/fmu/${USER}}"
SOURCE_ROOT="${SOURCE_ROOT:-${UMG_WORK}/server_code}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${FMU_SCRATCH}/server_code/results/conv2_perfect_diode_v4_vs_v1_saturation_gpu_20260701}"
PYTHON="${PYTHON:-python}"

cd "${SOURCE_ROOT}"
mkdir -p "${OUTPUT_ROOT}" slurm/conv2_pd_cond_gpu

export PYTHONUNBUFFERED=1
export PYTHONPATH="${SOURCE_ROOT}:${SOURCE_ROOT}/labs:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

V4_RUN="${FMU_SCRATCH}/server_code/results/mnist_bp_conv123_s221_pad1_perfect_diode_v1c1_v4c1_bestparams_k4k8_seed0_10_30_50epoch/conv2/epochs_30/input_gain_100/lr_0p012/k_4/perfect_diode/mnist_bp_amp_v4_c1/seed_0"
V1_HIGH_RUN="${FMU_SCRATCH}/server_code/results/mnist_bp_conv23_s221_pad1_perfect_diode_v1c1_gain_lr_downbias_bestlr_k4k8_seed0_30_50epoch/conv2/epochs_50/input_gain_400/lr_0p012/k_4/perfect_diode/mnist_bp_amp_v1_c1/seed_0"
V1_OLD_RUN="${FMU_SCRATCH}/server_code/results/mnist_bp_conv123_s221_pad1_perfect_diode_v1c1_v4c1_bestparams_k4k8_seed0_10_30_50epoch/conv2/epochs_30/input_gain_100/lr_0p024/k_4/perfect_diode/mnist_bp_amp_v1_c1/seed_0"
OUT_CSV="${OUTPUT_ROOT}/conv2_pd_v4_vs_v1_conductance_scale.csv"

echo "[conv2-pd-cond-gpu] job=${SLURM_JOB_ID:-manual} project=${IDRPROJ:-unset}"
echo "[conv2-pd-cond-gpu] source=${SOURCE_ROOT}"
echo "[conv2-pd-cond-gpu] output=${OUT_CSV}"
echo "[conv2-pd-cond-gpu] v4_run=${V4_RUN}"
echo "[conv2-pd-cond-gpu] v1_high_run=${V1_HIGH_RUN}"
echo "[conv2-pd-cond-gpu] v1_old_run=${V1_OLD_RUN}"

srun "${PYTHON}" "${SOURCE_ROOT}/experiments/analyze_mnist_bp_conv_conductance_scale.py" \
  --run "v4c1_gain100_lr012_30ep=${V4_RUN}" \
  --run "v1c1_gain400_lr012_50ep=${V1_HIGH_RUN}" \
  --run "v1c1_gain100_lr024_30ep=${V1_OLD_RUN}" \
  --output-csv "${OUT_CSV}"

echo "[conv2-pd-cond-gpu] done"
