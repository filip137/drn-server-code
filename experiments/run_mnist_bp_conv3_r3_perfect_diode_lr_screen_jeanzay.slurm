#!/bin/bash -l
#SBATCH --job-name=mnist_c3_pd_lr
#SBATCH --output=slurm/mnist_c3_pd_lr/%x-%A_%a.out
#SBATCH --error=slurm/mnist_c3_pd_lr/%x-%A_%a.err
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
#SBATCH --array=0-4%5

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
INPUT_GAIN_ROOT="${INPUT_GAIN_ROOT:-${FMU_SCRATCH}/server_code/results/mnist_bp_conv3_pad1_perfect_diode_inputgain_screen_k8_seed0_10epoch}"
SELECTED_GAIN_CSV="${SELECTED_GAIN_CSV:-${INPUT_GAIN_ROOT}/selected_input_gain_by_nonlinearity.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${FMU_SCRATCH}/server_code/results/mnist_bp_conv3_pad1_perfect_diode_nonlegacy_lr_screen_k8_seed0_10epoch}"
DATASET_ROOT="${DATASET_ROOT:-${UMG_SCRATCH}/datasets/mnist}"
PYTHON="${PYTHON:-python}"
EPOCHS="${EPOCHS:-10}"
PACK_SIZE="${PACK_SIZE:-3}"
SUMMARY_ONLY="${SUMMARY_ONLY:-0}"

cd "${SOURCE_ROOT}"
mkdir -p "${OUTPUT_ROOT}" slurm/mnist_c3_pd_lr

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
LR_MULTS=(0.25 0.5 1 2 4)

if [ "${PACK_SIZE}" -lt 1 ]; then
  echo "Expected PACK_SIZE >= 1, got ${PACK_SIZE}." >&2
  exit 2
fi

label_float() {
  local value="$1"
  value="${value//./p}"
  value="${value//-/m}"
  printf '%s' "${value}"
}

if [ "${SUMMARY_ONLY}" = "1" ]; then
  "${PYTHON}" "${SOURCE_ROOT}/experiments/collect_mnist_bp_conv_lr_screen.py" \
    --root "${OUTPUT_ROOT}"
  exit 0
fi

if [ ! -s "${SELECTED_GAIN_CSV}" ]; then
  echo "Expected selected input-gain CSV at ${SELECTED_GAIN_CSV}" >&2
  exit 2
fi

read -r input_gain base_lr selected_k < <("${PYTHON}" - "${SELECTED_GAIN_CSV}" <<'PY'
import csv
import sys

path = sys.argv[1]
with open(path, newline="") as handle:
    for row in csv.DictReader(handle):
        if row.get("non_linearity") == "perfect_diode":
            print(
                f"{float(row['selected_input_gain']):g}",
                f"{float(row['selected_lr']):g}",
                int(float(row["selected_iteration_count"])),
            )
            raise SystemExit(0)
raise SystemExit(f"missing perfect_diode row in {path}")
PY
)

run_one() {
  local run_name="$1"
  local learning_rate="$2"
  local gain_label="$3"
  local mult_label="$4"
  local lr_label="$5"
  local job_root="${OUTPUT_ROOT}/input_gain_${gain_label}/lr_mult_${mult_label}/lr_${lr_label}"

  echo "[conv3-pd-lr] job=${SLURM_JOB_ID:-manual} task=${SLURM_ARRAY_TASK_ID:-0} run=${run_name}"
  echo "[conv3-pd-lr] gain=${input_gain} lr=${learning_rate} K=${selected_k} output=${job_root}"

  "${PYTHON}" "${SOURCE_ROOT}/experiments/train_mnist_bp_conv_amplification_sweep.py" \
    --output-root "${job_root}" \
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
    --log-interval "${LOG_INTERVAL:-500}" \
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
if [ "${task_id}" -ge "${#LR_MULTS[@]}" ]; then
  echo "task_id=${task_id} outside LR matrix"
  exit 0
fi

lr_mult="${LR_MULTS[$task_id]}"
learning_rate="$(awk -v lr="${base_lr}" -v mult="${lr_mult}" 'BEGIN { printf "%.12g", lr * mult }')"
gain_label="$(label_float "${input_gain}")"
mult_label="$(label_float "${lr_mult}")"
lr_label="$(label_float "${learning_rate}")"

echo "[conv3-pd-lr] array_task=${task_id} pack_size=${PACK_SIZE} gain=${input_gain} base_lr=${base_lr} lr_mult=${lr_mult}"

pids=()
active=0
status=0
for run_name in "${RUN_NAMES[@]}"; do
  run_one "${run_name}" "${learning_rate}" "${gain_label}" "${mult_label}" "${lr_label}" &
  pids+=("$!")
  active=$((active + 1))
  if [ "${active}" -ge "${PACK_SIZE}" ]; then
    wait_batch || status=1
  fi
done
if [ "${active}" -gt 0 ]; then
  wait_batch || status=1
fi

exit "${status}"
