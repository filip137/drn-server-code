#!/bin/bash -l
#SBATCH --mnist-conv-legacy-launcher-is-deprecated
# MNIST_CONV_LEGACY_LAUNCHER_DEPRECATED
printf "%s\n" "Expected format: python -m experiments.mnist_conv {run|sweep|collect} ..." >&2
printf "%s\n" "Provided deprecated MNIST Conv launcher invocation: $0 $*" >&2
printf "%s\n" "Deprecated MNIST Conv launcher; no setup, allocation, or work was performed." >&2
exit 2
#SBATCH --job-name=conv2_pd_grad_gpu
#SBATCH --output=slurm/conv2_pd_grad_gpu/%x-%j.out
#SBATCH --error=slurm/conv2_pd_grad_gpu/%x-%j.err
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
OUTPUT_ROOT="${OUTPUT_ROOT:-${FMU_SCRATCH}/server_code/results/conv2_perfect_diode_v4_vs_v1_saturation_gpu_20260701}"
PYTHON="${PYTHON:-python}"

cd "${SOURCE_ROOT}"
mkdir -p "${OUTPUT_ROOT}" slurm/conv2_pd_grad_gpu

export PYTHONUNBUFFERED=1
export PYTHONPATH="${SOURCE_ROOT}:${SOURCE_ROOT}/labs:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

V4_RUN="${FMU_SCRATCH}/server_code/results/mnist_bp_conv123_s221_pad1_perfect_diode_v1c1_v4c1_bestparams_k4k8_seed0_10_30_50epoch/conv2/epochs_30/input_gain_100/lr_0p012/k_4/perfect_diode/mnist_bp_amp_v4_c1/seed_0"
V1_RUN="${FMU_SCRATCH}/server_code/results/mnist_bp_conv23_s221_pad1_perfect_diode_v1c1_gain_lr_downbias_bestlr_k4k8_seed0_30_50epoch/conv2/epochs_50/input_gain_400/lr_0p012/k_4/perfect_diode/mnist_bp_amp_v1_c1/seed_0"

echo "[conv2-pd-grad-gpu] job=${SLURM_JOB_ID:-manual} project=${IDRPROJ:-unset}"
echo "[conv2-pd-grad-gpu] source=${SOURCE_ROOT}"
echo "[conv2-pd-grad-gpu] dataset=${DATASET_ROOT}"
echo "[conv2-pd-grad-gpu] output_root=${OUTPUT_ROOT}"
echo "[conv2-pd-grad-gpu] v4_run=${V4_RUN}"
echo "[conv2-pd-grad-gpu] v1_run=${V1_RUN}"

for split in train test; do
  out_csv="${OUTPUT_ROOT}/conv2_pd_v4_vs_v1_${split}_gradient_init_final_256_gpu.csv"
  echo "[conv2-pd-grad-gpu] split=${split} out=${out_csv}"
  srun "${PYTHON}" "${SOURCE_ROOT}/experiments/analyze_mnist_bp_conv_good_run_gradient_saturation.py" \
    --run-dir "${V4_RUN}" \
    --run-dir "${V1_RUN}" \
    --output-csv "${out_csv}" \
    --dataset-root "${DATASET_ROOT}" \
    --device cuda \
    --phase both \
    --split "${split}" \
    --batch-size "${BATCH_SIZE:-32}" \
    --max-samples "${MAX_SAMPLES:-256}" \
    --no-download
done

echo "[conv2-pd-grad-gpu] done"
