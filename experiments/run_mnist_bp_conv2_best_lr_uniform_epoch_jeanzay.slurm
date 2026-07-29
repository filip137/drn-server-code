#!/bin/bash -l
#SBATCH --job-name=mnist_conv2_bestlr
#SBATCH --output=slurm/mnist_conv2_bestlr/%x-%A_%a.out
#SBATCH --error=slurm/mnist_conv2_bestlr/%x-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --account=umg@v100
#SBATCH --qos=qos_gpu-t3
#SBATCH --partition=gpu_p13
#SBATCH --constraint=v100-16g
#SBATCH --array=0-4%5

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
if [ ! -d "${SOURCE_ROOT}" ]; then
  echo "Expected SOURCE_ROOT to exist on Jean Zay: ${SOURCE_ROOT}" >&2
  exit 2
fi
cd "${SOURCE_ROOT}"

PYTHON="${PYTHON:-python}"
TRAIN="${SOURCE_ROOT}/experiments/train_mnist_bp_conv_amplification_sweep.py"
EPOCHS_VALUE="${EPOCHS:-50}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRATCH}/server_code/results/mnist_bp_conv2_best_lr_${EPOCHS_VALUE}epoch_64_128ch_s2_valid_iter6_seed0}"
DATASET_ROOT="${DATASET_ROOT:-${SCRATCH}/datasets/mnist}"
RESCUE_SELECTED="${RESCUE_SELECTED:-${SCRATCH}/server_code/results/mnist_bp_conv2_focused_lr_rescue_64_128ch_s2_valid_iter6_seed0/selected_best_by_run.csv}"
SMOKE_SUMMARY="${SMOKE_SUMMARY:-${SCRATCH}/server_code/results/mnist_bp_conv2_lr_smoke_64_128ch_s2_valid_iter6_seed0/summary_by_lr_nonlinearity_amp.csv}"
SUMMARY_ONLY="${SUMMARY_ONLY:-0}"

if [ ! -f "${TRAIN}" ]; then
  echo "Expected conv sweep runner at ${TRAIN}" >&2
  exit 2
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${SOURCE_ROOT}:${SOURCE_ROOT}/labs:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

mkdir -p "${OUTPUT_ROOT}" slurm/mnist_conv2_bestlr/subtasks

common_args=(
  --output-root "${OUTPUT_ROOT}"
  --dataset-root "${DATASET_ROOT}"
  --device cuda
  --model-key mnist_bp_conv_amp
  --seeds 0
  --epochs "${EPOCHS_VALUE}"
  --batch-size "${BATCH_SIZE:-4}"
  --num-iterations "${NUM_ITERATIONS:-6}"
  --beta "${BETA:-1.0}"
  --lr-decay "${LR_DECAY:-0.99}"
  --conv-depth 2
  --conv-channels 64 128 256
  --kernel-size 3
  --stride 2
  --padding 0
  --output-dim 20
  --weight-gains 1.0
  --weight-min 0.0
  --weight-max 100.0
  --weight-init-mode kaiming_uniform
  --input-gain "${INPUT_GAIN:-100.0}"
  --hard-sigmoid-g-on "${HARD_SIGMOID_G_ON:-100.0}"
  --hard-sigmoid-g-off "${HARD_SIGMOID_G_OFF:-0.0}"
  --hard-sigmoid-v-min "${HARD_SIGMOID_V_MIN:--1.5}"
  --hard-sigmoid-v-max "${HARD_SIGMOID_V_MAX:-1.5}"
  --normalize-mean 0.1307
  --normalize-std 0.3081
  --normalize-scale 0.3
  --log-interval "${LOG_INTERVAL:-500}"
  --no-download
)

NONLINEARITIES=(
  hard_sigmoid
  hard_sigmoid
  hard_sigmoid
  hard_sigmoid
  hard_sigmoid
  perfect_diode
  perfect_diode
  perfect_diode
  perfect_diode
  perfect_diode
)
RUN_NAMES=(
  mnist_bp_amp_v1_c1
  mnist_bp_amp_v2_c1
  mnist_bp_amp_v4_c1
  mnist_bp_amp_v1_c2
  mnist_bp_amp_v1_c4
  mnist_bp_amp_v1_c1
  mnist_bp_amp_v2_c1
  mnist_bp_amp_v4_c1
  mnist_bp_amp_v1_c2
  mnist_bp_amp_v1_c4
)

if [ "${SUMMARY_ONLY}" = "1" ]; then
  "${PYTHON}" "${TRAIN}" \
    "${common_args[@]}" \
    --learning-rate "${SUMMARY_LR_PLACEHOLDER:-0.006}" \
    --non-linearity hard_sigmoid perfect_diode \
    --summary-only
  exit 0
fi

selected_lr_for() {
  local nonlinearity="$1"
  local run_name="$2"
  "${PYTHON}" - "${RESCUE_SELECTED}" "${SMOKE_SUMMARY}" "${nonlinearity}" "${run_name}" <<'PY'
import csv
import sys

rescue_selected, smoke_summary, nonlinearity, run_name = sys.argv[1:5]

try:
    with open(rescue_selected, newline="") as handle:
        for row in csv.DictReader(handle):
            if row["non_linearity"] == nonlinearity and row["run_name"] == run_name:
                print(row["selected_lr"])
                raise SystemExit(0)
except FileNotFoundError:
    pass

rows = []
with open(smoke_summary, newline="") as handle:
    for row in csv.DictReader(handle):
        if row["non_linearity"] == nonlinearity and row["run_name"] == run_name:
            rows.append(row)

if not rows:
    raise SystemExit(
        f"No LR candidates found for non_linearity={nonlinearity} run_name={run_name}."
    )

def score(row):
    return (
        float(row["mean_best_test_accuracy"]),
        -float(row["mean_final_test_loss"]),
        -float(row["lr"]),
    )

print(max(rows, key=score)["lr"])
PY
}

task_id="${SLURM_ARRAY_TASK_ID:-0}"
tasks_per_gpu="${TASKS_PER_GPU:-2}"
if [ "${tasks_per_gpu}" -lt 1 ]; then
  echo "Expected TASKS_PER_GPU >= 1, got ${tasks_per_gpu}." >&2
  exit 2
fi
start_index=$((task_id * tasks_per_gpu))
if [ "${start_index}" -ge "${#RUN_NAMES[@]}" ]; then
  echo "Invalid task_id=${task_id}; no logical paper task starts at ${start_index}." >&2
  exit 2
fi

echo "[jean-zay] job=${SLURM_JOB_ID:-manual} array_task=${task_id}"
echo "[jean-zay] source_root=${SOURCE_ROOT}"
echo "[jean-zay] dataset_root=${DATASET_ROOT}"
echo "[jean-zay] output_root=${OUTPUT_ROOT}"
echo "[jean-zay] rescue_selected=${RESCUE_SELECTED}"
echo "[jean-zay] smoke_summary=${SMOKE_SUMMARY}"
echo "[jean-zay] epochs=${EPOCHS_VALUE} tasks_per_gpu=${tasks_per_gpu}"
echo "[jean-zay] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"

pids=()
logical_ids=()

for local_idx in $(seq 0 $((tasks_per_gpu - 1))); do
  logical_id=$((start_index + local_idx))
  if [ "${logical_id}" -ge "${#RUN_NAMES[@]}" ]; then
    continue
  fi

  nonlinearity="${NONLINEARITIES[$logical_id]}"
  run_name="${RUN_NAMES[$logical_id]}"
  selected_lr="$(selected_lr_for "${nonlinearity}" "${run_name}")"
  child_log="slurm/mnist_conv2_bestlr/subtasks/${SLURM_ARRAY_JOB_ID:-manual}_${logical_id}.out"
  child_err="slurm/mnist_conv2_bestlr/subtasks/${SLURM_ARRAY_JOB_ID:-manual}_${logical_id}.err"

  echo "[jean-zay] launching logical_id=${logical_id} nonlinearity=${nonlinearity} run_name=${run_name} seed=0 lr=${selected_lr}"
  (
    "${PYTHON}" "${TRAIN}" \
      "${common_args[@]}" \
      --learning-rate "${selected_lr}" \
      --non-linearity "${nonlinearity}" \
      --run-name "${run_name}"
  ) >"${child_log}" 2>"${child_err}" &
  pids+=("$!")
  logical_ids+=("${logical_id}")
done

status=0
for idx in "${!pids[@]}"; do
  pid="${pids[$idx]}"
  logical_id="${logical_ids[$idx]}"
  if wait "${pid}"; then
    echo "[jean-zay] logical_id=${logical_id} completed"
  else
    echo "[jean-zay] logical_id=${logical_id} failed" >&2
    status=1
  fi
done
exit "${status}"
