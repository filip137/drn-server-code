#!/bin/bash -l
#SBATCH --job-name=mnist_conv_fix_trnxt
#SBATCH --output=slurm/mnist_conv_fix_trnxt/%x-%j.out
#SBATCH --error=slurm/mnist_conv_fix_trnxt/%x-%j.err
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
PHASE="${PHASE:?Expected PHASE=long or PHASE=final.}"
CHUNK_START="${CHUNK_START:?Expected CHUNK_START.}"
CHUNK_SIZE="${CHUNK_SIZE:-6}"
TOTAL_TASKS="${TOTAL_TASKS:?Expected TOTAL_TASKS.}"

cd "${SOURCE_ROOT}"
mkdir -p slurm/mnist_conv_fix_trnxt

chunk_end=$((CHUNK_START + CHUNK_SIZE - 1))
last_task=$((TOTAL_TASKS - 1))
if [ "${chunk_end}" -gt "${last_task}" ]; then
  chunk_end="${last_task}"
fi

echo "[train-next] phase=${PHASE} submit chunk ${CHUNK_START}-${chunk_end} total=${TOTAL_TASKS}"
train_job="$(
  sbatch --parsable \
    --array="${CHUNK_START}-${chunk_end}%${CHUNK_SIZE}" \
    --export="ALL,RESULT_ROOT=${RESULT_ROOT},PHASE=${PHASE}" \
    experiments/run_mnist_bp_conv_r3_fixed_step_train_selected_jeanzay.slurm
)"
echo "[train-next] train_job=${train_job}"

next_start=$((chunk_end + 1))
if [ "${next_start}" -lt "${TOTAL_TASKS}" ]; then
  next_submit="$(
    sbatch --parsable \
      --dependency="afterok:${train_job}" \
      --export="ALL,RESULT_ROOT=${RESULT_ROOT},PHASE=${PHASE},CHUNK_START=${next_start},CHUNK_SIZE=${CHUNK_SIZE},TOTAL_TASKS=${TOTAL_TASKS}" \
      experiments/run_mnist_bp_conv_r3_fixed_step_submit_train_chunk_jeanzay.slurm
  )"
  echo "[train-next] next_submit=${next_submit} next_start=${next_start}"
elif [ "${PHASE}" = "long" ]; then
  collect_long="$(
    sbatch --parsable \
      --dependency="afterok:${train_job}" \
      --export="ALL,RESULT_ROOT=${RESULT_ROOT},LR_ROOT=${RESULT_ROOT}/long_check_seed0_50epoch,SELECTED_SETTINGS_CSV=${RESULT_ROOT}/long_check_seed0_50epoch/selected_final_settings.csv" \
      experiments/run_mnist_bp_conv_r3_fixed_step_collect_lr_jeanzay.slurm
  )"
  echo "[train-next] collect_long=${collect_long}"
  gate_long="$(
    sbatch --parsable \
      --dependency="afterok:${collect_long}" \
      --export="ALL,RESULT_ROOT=${RESULT_ROOT}" \
      experiments/run_mnist_bp_conv_r3_fixed_step_gate_long_check_jeanzay.slurm
  )"
  echo "[train-next] gate_long=${gate_long}"
  final_submit="$(
    sbatch --parsable \
      --dependency="afterok:${gate_long}" \
      --export="ALL,RESULT_ROOT=${RESULT_ROOT},PHASE=final,CHUNK_START=0,CHUNK_SIZE=${CHUNK_SIZE},TOTAL_TASKS=36" \
      experiments/run_mnist_bp_conv_r3_fixed_step_submit_train_chunk_jeanzay.slurm
  )"
  echo "[train-next] final_submit=${final_submit}"
elif [ "${PHASE}" = "final" ]; then
  post_diag="$(
    sbatch --parsable \
      --dependency="afterok:${train_job}" \
      --array="0-1%2" \
      --export="ALL,RESULT_ROOT=${RESULT_ROOT}" \
      experiments/run_mnist_bp_conv_r3_fixed_step_post_diagnostics_jeanzay.slurm
  )"
  echo "[train-next] post_diag=${post_diag}"
else
  echo "Expected PHASE=long or final, got ${PHASE}." >&2
  exit 2
fi
