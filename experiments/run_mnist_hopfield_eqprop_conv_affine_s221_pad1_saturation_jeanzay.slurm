#!/bin/bash -l
#SBATCH --job-name=mnist_hopf_aff_s221p1_sat
#SBATCH --output=slurm/mnist_hopf_aff_s221p1_sat/%x-%j.out
#SBATCH --error=slurm/mnist_hopf_aff_s221p1_sat/%x-%j.err
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
OUTPUT_ROOT="${OUTPUT_ROOT:-${FMU_SCRATCH}/server_code/results/mnist_hopfield_eqprop_conv_affine_s221_pad1_20260704}"
DATASET_ROOT="${DATASET_ROOT:-${UMG_SCRATCH}/datasets/mnist}"
PYTHON="${PYTHON:-python}"
RUN_GROUP="${RUN_GROUP:-final_affine_mnist_s221_pad1}"
INPUT_ROOT="${INPUT_ROOT:-${OUTPUT_ROOT}/${RUN_GROUP}}"
OUTPUT_CSV="${OUTPUT_CSV:-${OUTPUT_ROOT}/${RUN_GROUP}_saturation_train_test_2048.csv}"

cd "${SOURCE_ROOT}"
mkdir -p "${OUTPUT_ROOT}" slurm/mnist_hopf_aff_s221p1_sat

export PYTHONUNBUFFERED=1
export PYTHONPATH="${SOURCE_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

"${PYTHON}" "${SOURCE_ROOT}/experiments/measure_mnist_hopfield_eqprop_saturation.py" \
  --input-root "${INPUT_ROOT}" \
  --output-csv "${OUTPUT_CSV}" \
  --dataset-root "${DATASET_ROOT}" \
  --dataset mnist \
  --device cuda \
  --split train test \
  --checkpoint-kind init best final \
  --max-samples "${MAX_SAMPLES:-2048}" \
  --batch-size "${BATCH_SIZE:-64}" \
  --no-download
