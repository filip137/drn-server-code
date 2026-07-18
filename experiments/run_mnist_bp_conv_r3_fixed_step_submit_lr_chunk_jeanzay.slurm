#!/bin/bash -l
#SBATCH --mnist-conv-legacy-launcher-is-deprecated
# MNIST_CONV_LEGACY_LAUNCHER_DEPRECATED
printf "%s\n" "Expected format: python -m experiments.mnist_conv {run|sweep|collect} ..." >&2
printf "%s\n" "Provided deprecated MNIST Conv launcher invocation: $0 $*" >&2
printf "%s\n" "Deprecated MNIST Conv launcher; no setup, allocation, or work was performed." >&2
exit 2
#SBATCH --job-name=mnist_conv_fix_lrnxt
#SBATCH --output=slurm/mnist_conv_fix_lrnxt/%x-%j.out
#SBATCH --error=slurm/mnist_conv_fix_lrnxt/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --time=00:10:00
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

UMG_WORK="${WORK_umg:-/lustre/fswork/projects/rech/umg/${USER}}"
FMU_SCRATCH="${SCRATCH_fmu:-/lustre/fsn1/projects/rech/fmu/${USER}}"
SOURCE_ROOT="${SOURCE_ROOT:-${UMG_WORK}/server_code}"
RESULT_ROOT="${RESULT_ROOT:-${FMU_SCRATCH}/server_code/results/mnist_bp_conv_fixed_step_redo_r3_20260703}"
CHUNK_START="${CHUNK_START:?Expected CHUNK_START.}"
CHUNK_SIZE="${CHUNK_SIZE:-6}"
TOTAL_TASKS="${TOTAL_TASKS:-36}"

cd "${SOURCE_ROOT}"
mkdir -p slurm/mnist_conv_fix_lrnxt

chunk_end=$((CHUNK_START + CHUNK_SIZE - 1))
last_task=$((TOTAL_TASKS - 1))
if [ "${chunk_end}" -gt "${last_task}" ]; then
  chunk_end="${last_task}"
fi

echo "[lr-next] submit chunk ${CHUNK_START}-${chunk_end} total=${TOTAL_TASKS}"
lr_job="$(
  sbatch --parsable \
    --array="${CHUNK_START}-${chunk_end}%${CHUNK_SIZE}" \
    --export="ALL,RESULT_ROOT=${RESULT_ROOT}" \
    experiments/run_mnist_bp_conv_r3_fixed_step_lr_screen_jeanzay.slurm
)"
echo "[lr-next] lr_job=${lr_job}"

next_start=$((chunk_end + 1))
if [ "${next_start}" -lt "${TOTAL_TASKS}" ]; then
  next_submit="$(
    sbatch --parsable \
      --dependency="afterok:${lr_job}" \
      --export="ALL,RESULT_ROOT=${RESULT_ROOT},CHUNK_START=${next_start},CHUNK_SIZE=${CHUNK_SIZE},TOTAL_TASKS=${TOTAL_TASKS}" \
      experiments/run_mnist_bp_conv_r3_fixed_step_submit_lr_chunk_jeanzay.slurm
  )"
  echo "[lr-next] next_submit=${next_submit} next_start=${next_start}"
else
  collect_job="$(
    sbatch --parsable \
      --dependency="afterok:${lr_job}" \
      --export="ALL,RESULT_ROOT=${RESULT_ROOT}" \
      experiments/run_mnist_bp_conv_r3_fixed_step_collect_lr_jeanzay.slurm
  )"
  echo "[lr-next] collect_lr=${collect_job}"
  long_submit="$(
    sbatch --parsable \
      --dependency="afterok:${collect_job}" \
      --export="ALL,RESULT_ROOT=${RESULT_ROOT},PHASE=long,CHUNK_START=0,CHUNK_SIZE=${CHUNK_SIZE},TOTAL_TASKS=12" \
      experiments/run_mnist_bp_conv_r3_fixed_step_submit_train_chunk_jeanzay.slurm
  )"
  echo "[lr-next] long_submit=${long_submit}"
fi
