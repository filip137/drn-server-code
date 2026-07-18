#!/bin/bash -l
#SBATCH --mnist-conv-legacy-launcher-is-deprecated
# MNIST_CONV_LEGACY_LAUNCHER_DEPRECATED
printf "%s\n" "Expected format: python -m experiments.mnist_conv {run|sweep|collect} ..." >&2
printf "%s\n" "Provided deprecated MNIST Conv launcher invocation: $0 $*" >&2
printf "%s\n" "Deprecated MNIST Conv launcher; no setup, allocation, or work was performed." >&2
exit 2
#SBATCH --job-name=mnist_c3_pd_30ep
#SBATCH --output=slurm/mnist_c3_pd_30ep/%x-%A_%a.out
#SBATCH --error=slurm/mnist_c3_pd_30ep/%x-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --account=fmu@v100
#SBATCH --qos=qos_gpu-t3
#SBATCH --partition=gpu_p13
#SBATCH --constraint=v100-16g
#SBATCH --array=0-0%1

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
LR_ROOT="${LR_ROOT:-${FMU_SCRATCH}/server_code/results/mnist_bp_conv3_pad1_perfect_diode_nonlegacy_lr_screen_k8_seed0_10epoch}"
SELECTED_LR_CSV="${SELECTED_LR_CSV:-${LR_ROOT}/selected_lr_by_nonlinearity_amp.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${FMU_SCRATCH}/server_code/results/mnist_bp_conv3_pad1_perfect_diode_nonlegacy_bestlr_k8_seed0_30epoch}"
DATASET_ROOT="${DATASET_ROOT:-${UMG_SCRATCH}/datasets/mnist}"
PYTHON="${PYTHON:-python}"
EPOCHS="${EPOCHS:-30}"
PACK_SIZE="${PACK_SIZE:-3}"
SUMMARY_ONLY="${SUMMARY_ONLY:-0}"

cd "${SOURCE_ROOT}"
mkdir -p "${OUTPUT_ROOT}" slurm/mnist_c3_pd_30ep

export PYTHONUNBUFFERED=1
export PYTHONPATH="${SOURCE_ROOT}:${SOURCE_ROOT}/labs:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

RUN_NAMES=(
  mnist_bp_amp_v1_c1
  mnist_bp_amp_v2_c1
  mnist_bp_amp_v4_c1
)

if [ "${PACK_SIZE}" -lt 1 ]; then
  echo "Expected PACK_SIZE >= 1, got ${PACK_SIZE}." >&2
  exit 2
fi

if [ "${SUMMARY_ONLY}" = "1" ]; then
  "${PYTHON}" "${SOURCE_ROOT}/experiments/train_mnist_bp_conv_amplification_sweep.py" \
    --output-root "${OUTPUT_ROOT}" \
    --dataset-root "${DATASET_ROOT}" \
    --device cuda \
    --model-key mnist_bp_conv_amp \
    --non-linearity perfect_diode \
    --seeds 0 \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE:-4}" \
    --conv-depth 3 \
    --conv-channels 64 128 256 \
    --kernel-size 3 \
    --stride 2 \
    --padding 1 \
    --output-dim 20 \
    --summary-only
  exit 0
fi

if [ ! -s "${SELECTED_LR_CSV}" ]; then
  echo "Expected selected LR CSV at ${SELECTED_LR_CSV}" >&2
  exit 2
fi

mapfile -t TASKS < <("${PYTHON}" - "${SELECTED_LR_CSV}" "${RUN_NAMES[@]}" <<'PY'
import csv
import sys

path = sys.argv[1]
run_order = sys.argv[2:]
rows = {}
with open(path, newline="") as handle:
    for row in csv.DictReader(handle):
        if row.get("non_linearity") == "perfect_diode":
            rows[row["run_name"]] = row

missing = [run_name for run_name in run_order if run_name not in rows]
if missing:
    raise SystemExit(f"missing selected LR rows for: {' '.join(missing)}")

for run_name in run_order:
    row = rows[run_name]
    print(
        run_name,
        f"{float(row['input_gain']):g}",
        int(float(row["iteration_count"])),
        f"{float(row['selected_lr']):g}",
    )
PY
)

run_one() {
  local run_name="$1"
  local input_gain="$2"
  local selected_k="$3"
  local learning_rate="$4"

  echo "[conv3-pd-bestlr30] job=${SLURM_JOB_ID:-manual} task=${SLURM_ARRAY_TASK_ID:-0} run=${run_name}"
  echo "[conv3-pd-bestlr30] gain=${input_gain} lr=${learning_rate} K=${selected_k} output=${OUTPUT_ROOT}"

  "${PYTHON}" "${SOURCE_ROOT}/experiments/train_mnist_bp_conv_amplification_sweep.py" \
    --output-root "${OUTPUT_ROOT}" \
    --dataset-root "${DATASET_ROOT}" \
    --device cuda \
    --model-key mnist_bp_conv_amp \
    --non-linearity perfect_diode \
    --seeds 0 \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE:-4}" \
    --beta 1.0 \
    --lr-decay 0.99 \
    --conv-depth 3 \
    --conv-channels 64 128 256 \
    --kernel-size 3 \
    --stride 2 \
    --padding 1 \
    --output-dim 20 \
    --weight-gains 1.0 \
    --weight-min 0.0 \
    --weight-max 100.0 \
    --weight-init-mode kaiming_uniform \
    --input-gain "${input_gain}" \
    --num-iterations "${selected_k}" \
    --learning-rate "${learning_rate}" \
    --hard-sigmoid-g-on 100.0 \
    --hard-sigmoid-g-off 0.0 \
    --hard-sigmoid-v-min -1.5 \
    --hard-sigmoid-v-max 1.5 \
    --normalize-mean 0.1307 \
    --normalize-std 0.3081 \
    --normalize-scale 0.3 \
    --log-interval "${LOG_INTERVAL:-1000}" \
    --no-download \
    --run-name "${run_name}"
}

wait_batch() {
  local status=0
  local pid
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done
  pids=()
  active=0
  return "${status}"
}

task_id="${SLURM_ARRAY_TASK_ID:-0}"
base_task_id=$((task_id * PACK_SIZE))
if [ "${base_task_id}" -ge "${#TASKS[@]}" ]; then
  echo "task_id=${task_id} outside generated tasks=${#TASKS[@]}"
  exit 0
fi

echo "[conv3-pd-bestlr30] array_task=${task_id} pack_size=${PACK_SIZE} base_task=${base_task_id}"

pids=()
active=0
status=0
for ((offset = 0; offset < PACK_SIZE; offset++)); do
  index=$((base_task_id + offset))
  if [ "${index}" -ge "${#TASKS[@]}" ]; then
    break
  fi
  read -r run_name input_gain selected_k learning_rate <<< "${TASKS[$index]}"
  run_one "${run_name}" "${input_gain}" "${selected_k}" "${learning_rate}" &
  pids+=("$!")
  active=$((active + 1))
done
if [ "${active}" -gt 0 ]; then
  wait_batch || status=1
fi

if [ "${status}" -eq 0 ]; then
  "${PYTHON}" "${SOURCE_ROOT}/experiments/train_mnist_bp_conv_amplification_sweep.py" \
    --output-root "${OUTPUT_ROOT}" \
    --dataset-root "${DATASET_ROOT}" \
    --device cuda \
    --model-key mnist_bp_conv_amp \
    --non-linearity perfect_diode \
    --seeds 0 \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE:-4}" \
    --conv-depth 3 \
    --conv-channels 64 128 256 \
    --kernel-size 3 \
    --stride 2 \
    --padding 1 \
    --output-dim 20 \
    --summary-only
fi

exit "${status}"
