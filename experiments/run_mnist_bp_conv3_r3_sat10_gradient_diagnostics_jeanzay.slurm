#!/bin/bash -l
#SBATCH --mnist-conv-legacy-launcher-is-deprecated
# MNIST_CONV_LEGACY_LAUNCHER_DEPRECATED
printf "%s\n" "Expected format: python -m experiments.mnist_conv {run|sweep|collect} ..." >&2
printf "%s\n" "Provided deprecated MNIST Conv launcher invocation: $0 $*" >&2
printf "%s\n" "Deprecated MNIST Conv launcher; no setup, allocation, or work was performed." >&2
exit 2
#SBATCH --job-name=mnist_c3_grad10
#SBATCH --output=slurm/mnist_c3_grad10/%x-%j.out
#SBATCH --error=slurm/mnist_c3_grad10/%x-%j.err
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
INPUT_ROOT="${INPUT_ROOT:-${RESULT_ROOT}/target_sat10}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${RESULT_ROOT}/gradient_diagnostics_sat10_train64}"
DATASET_ROOT="${DATASET_ROOT:-${UMG_SCRATCH}/datasets/mnist}"
PYTHON="${PYTHON:-python}"

cd "${SOURCE_ROOT}"
mkdir -p slurm/mnist_c3_grad10 "${OUTPUT_ROOT}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${SOURCE_ROOT}:${SOURCE_ROOT}/labs:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

echo "[grad10] input_root=${INPUT_ROOT}"
echo "[grad10] output_root=${OUTPUT_ROOT}"
echo "[grad10] dataset_root=${DATASET_ROOT}"

for checkpoint_kind in init final; do
  kind_output="${OUTPUT_ROOT}/${checkpoint_kind}"
  echo "[grad10] checkpoint_kind=${checkpoint_kind} output=${kind_output}"
  srun "${PYTHON}" "${SOURCE_ROOT}/experiments/diagnose_mnist_bp_conv_saturation_gradients.py" \
    --input-root "${INPUT_ROOT}" \
    --output-root "${kind_output}" \
    --dataset-root "${DATASET_ROOT}" \
    --device cuda \
    --checkpoint-kind "${checkpoint_kind}" \
    --split "${SPLIT:-train}" \
    --batch-size "${BATCH_SIZE:-8}" \
    --max-batches "${MAX_BATCHES:-8}" \
    --no-download
done

echo "[grad10] done"
