#!/bin/bash -l
#SBATCH --job-name=mnist_c3_satcheck
#SBATCH --output=slurm/mnist_c3_satcheck/%x-%j.out
#SBATCH --error=slurm/mnist_c3_satcheck/%x-%j.err
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
RESULT_ROOT="${RESULT_ROOT:-${FMU_SCRATCH}/server_code/results/mnist_bp_conv3_pad1_hardsigmoid_amp_saturation_targets_voff4_sat10_20_30_50_k8_seed0_30epoch}"
DATASET_ROOT="${DATASET_ROOT:-${UMG_SCRATCH}/datasets/mnist}"
OUTPUT_CSV="${OUTPUT_CSV:-${RESULT_ROOT}/saturation_init_final_train256.csv}"
PYTHON="${PYTHON:-python}"

cd "${SOURCE_ROOT}"
mkdir -p slurm/mnist_c3_satcheck "$(dirname "${OUTPUT_CSV}")"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${SOURCE_ROOT}:${SOURCE_ROOT}/labs:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

echo "[satcheck] result_root=${RESULT_ROOT}"
echo "[satcheck] output_csv=${OUTPUT_CSV}"
echo "[satcheck] dataset_root=${DATASET_ROOT}"

srun "${PYTHON}" "${SOURCE_ROOT}/experiments/analyze_mnist_bp_conv_good_run_saturation.py" \
  --input-root "${RESULT_ROOT}" \
  --output-csv "${OUTPUT_CSV}" \
  --dataset-root "${DATASET_ROOT}" \
  --device cuda \
  --split "${SPLIT:-train}" \
  --max-samples "${MAX_SAMPLES:-256}" \
  --batch-size "${BATCH_SIZE:-16}" \
  --no-download

echo "[satcheck] done"
