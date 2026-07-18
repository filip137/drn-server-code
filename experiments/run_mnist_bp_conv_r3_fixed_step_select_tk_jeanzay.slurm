#!/bin/bash -l
#SBATCH --mnist-conv-legacy-launcher-is-deprecated
# MNIST_CONV_LEGACY_LAUNCHER_DEPRECATED
printf "%s\n" "Expected format: python -m experiments.mnist_conv {run|sweep|collect} ..." >&2
printf "%s\n" "Provided deprecated MNIST Conv launcher invocation: $0 $*" >&2
printf "%s\n" "Deprecated MNIST Conv launcher; no setup, allocation, or work was performed." >&2
exit 2
#SBATCH --job-name=mnist_conv_fix_seltk
#SBATCH --output=slurm/mnist_conv_fix_seltk/%x-%j.out
#SBATCH --error=slurm/mnist_conv_fix_seltk/%x-%j.err
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
RESULT_ROOT="${RESULT_ROOT:-${FMU_SCRATCH}/server_code/results/mnist_bp_conv_fixed_step_redo_r3_20260703}"
DIAGNOSTICS_ROOT="${DIAGNOSTICS_ROOT:-${RESULT_ROOT}/tk_diagnostics}"
SELECTED_TK_CSV="${SELECTED_TK_CSV:-${DIAGNOSTICS_ROOT}/selected_tk.csv}"
PYTHON="${PYTHON:-python}"

cd "${SOURCE_ROOT}"
mkdir -p "${DIAGNOSTICS_ROOT}" slurm/mnist_conv_fix_seltk

export PYTHONUNBUFFERED=1
export PYTHONPATH="${SOURCE_ROOT}:${SOURCE_ROOT}/labs:${PYTHONPATH:-}"

read -r -a MIN_T_BY_CONV_DEPTH_ARGS <<< "${MIN_T_BY_CONV_DEPTH:-2:6 3:10}"

srun "${PYTHON}" "${SOURCE_ROOT}/experiments/select_mnist_bp_conv_fixed_step_tk.py" \
  --diagnostics-root "${DIAGNOSTICS_ROOT}" \
  --output "${SELECTED_TK_CSV}" \
  --T-values 4 6 8 10 16 32 \
  --K-values 4 6 8 16 \
  --selection-mode settled \
  --settle-relative-tolerance "${SETTLE_RELATIVE_TOLERANCE:-0.10}" \
  --settle-zero-fraction-tolerance "${SETTLE_ZERO_FRACTION_TOLERANCE:-0.02}" \
  --min-T-by-conv-depth "${MIN_T_BY_CONV_DEPTH_ARGS[@]}" \
  --expected-run-names mnist_bp_amp_v1_c1 mnist_bp_amp_v4_c1 mnist_bp_amp_v4_c0p25
