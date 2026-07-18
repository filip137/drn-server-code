#!/bin/bash -l
#SBATCH --mnist-conv-legacy-launcher-is-deprecated
# MNIST_CONV_LEGACY_LAUNCHER_DEPRECATED
printf "%s\n" "Expected format: python -m experiments.mnist_conv {run|sweep|collect} ..." >&2
printf "%s\n" "Provided deprecated MNIST Conv launcher invocation: $0 $*" >&2
printf "%s\n" "Deprecated MNIST Conv launcher; no setup, allocation, or work was performed." >&2
exit 2
#SBATCH --job-name=mnist_conv3_pad1_selectK
#SBATCH --output=slurm/mnist_conv3_pad1_selectK/%x-%j.out
#SBATCH --error=slurm/mnist_conv3_pad1_selectK/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=00:30:00
#SBATCH --account=umg@v100
#SBATCH --qos=qos_gpu-t3
#SBATCH --partition=gpu_p13
#SBATCH --constraint=v100-16g

set -euo pipefail

if command -v idrenv >/dev/null 2>&1; then
  eval "$(idrenv)"
fi
if [ -z "${WORK:-}" ]; then
  WORK="/lustre/fswork/projects/rech/${IDRPROJ:-umg}/${USER}"
fi
if [ -z "${SCRATCH:-}" ]; then
  SCRATCH="/lustre/fsn1/projects/rech/${IDRPROJ:-umg}/${USER}"
fi

module purge
module load "${PYTORCH_MODULE:-pytorch-gpu/py3/2.5.0}"

SOURCE_ROOT="${SOURCE_ROOT:-${WORK}/server_code}"
cd "${SOURCE_ROOT}"

PYTHON="${PYTHON:-python}"
K_ROOT="${K_ROOT:-${SCRATCH}/server_code/results/mnist_bp_conv3_pad1_k_preflight_seed0}"
RESIDUAL_SCRIPT="${SOURCE_ROOT}/experiments/evaluate_mnist_bp_conv_residual_vs_iterations.py"
COSINE_SCRIPT="${SOURCE_ROOT}/experiments/evaluate_mnist_bp_conv_ep_bp_cosine_vs_iterations.py"
SELECT_SCRIPT="${SOURCE_ROOT}/experiments/select_mnist_bp_conv_k_by_nonlinearity.py"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${SOURCE_ROOT}:${SOURCE_ROOT}/labs:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/mplconfig}"

"${PYTHON}" "${RESIDUAL_SCRIPT}" \
  --input-root "${INPUT_ROOT:-${SCRATCH}/server_code/results/mnist_bp_conv3_pad1_bootstrap_k16_inputgain100_seed0_10epoch}" \
  --output-root "${K_ROOT}/residual" \
  --non-linearity hard_sigmoid perfect_diode \
  --summary-only

"${PYTHON}" "${COSINE_SCRIPT}" \
  --input-root "${INPUT_ROOT:-${SCRATCH}/server_code/results/mnist_bp_conv3_pad1_bootstrap_k16_inputgain100_seed0_10epoch}" \
  --output-root "${K_ROOT}/epbp" \
  --non-linearity hard_sigmoid perfect_diode \
  --summary-only

"${PYTHON}" "${SELECT_SCRIPT}" \
  --cosine-selected "${K_ROOT}/epbp/selected_k_by_amp_beta0p25.csv" \
  --output "${K_ROOT}/selected_k_by_nonlinearity.csv" \
  --cap "${K_CAP:-32}"

echo "[select-k] wrote ${K_ROOT}/selected_k_by_nonlinearity.csv"
