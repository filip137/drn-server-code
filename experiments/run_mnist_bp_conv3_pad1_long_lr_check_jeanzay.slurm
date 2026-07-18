#!/bin/bash -l
#SBATCH --mnist-conv-legacy-launcher-is-deprecated
# MNIST_CONV_LEGACY_LAUNCHER_DEPRECATED
printf "%s\n" "Expected format: python -m experiments.mnist_conv {run|sweep|collect} ..." >&2
printf "%s\n" "Provided deprecated MNIST Conv launcher invocation: $0 $*" >&2
printf "%s\n" "Deprecated MNIST Conv launcher; no setup, allocation, or work was performed." >&2
exit 2
#SBATCH --job-name=mnist_conv3_pad1_longlr
#SBATCH --output=slurm/mnist_conv3_pad1_longlr/%x-%A_%a.out
#SBATCH --error=slurm/mnist_conv3_pad1_longlr/%x-%A_%a.err
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
#SBATCH --array=0-19%20

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
TRAIN="${SOURCE_ROOT}/experiments/train_mnist_bp_conv_amplification_sweep.py"
COLLECT="${SOURCE_ROOT}/experiments/collect_mnist_bp_conv_lr_screen.py"
SELECTED_LR_CSV="${SELECTED_LR_CSV:-${SCRATCH}/server_code/results/mnist_bp_conv3_pad1_lr_screen_seed0_10epoch/selected_lr_by_nonlinearity_amp.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRATCH}/server_code/results/mnist_bp_conv3_pad1_long_lr_check_seed0_50epoch}"
DATASET_ROOT="${DATASET_ROOT:-${SCRATCH}/datasets/mnist}"
SUMMARY_ONLY="${SUMMARY_ONLY:-0}"

if [ ! -f "${SELECTED_LR_CSV}" ]; then
  echo "Expected selected LR CSV at ${SELECTED_LR_CSV}" >&2
  exit 2
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${SOURCE_ROOT}:${SOURCE_ROOT}/labs:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

mkdir -p "${OUTPUT_ROOT}" slurm/mnist_conv3_pad1_longlr

if [ "${SUMMARY_ONLY}" = "1" ]; then
  "${PYTHON}" "${COLLECT}" --root "${OUTPUT_ROOT}"
  exit 0
fi

mapfile -t TASKS < <("${PYTHON}" - "${SELECTED_LR_CSV}" <<'PY'
import csv, sys

selected_csv = sys.argv[1]
grids = {
    ("hard_sigmoid", "mnist_bp_amp_v1_c1"): [0.003, 0.006, 0.009, 0.012, 0.018],
    ("hard_sigmoid", "mnist_bp_amp_v2_c1"): [0.003, 0.006, 0.009, 0.012, 0.018],
    ("hard_sigmoid", "mnist_bp_amp_v4_c1"): [0.003, 0.006, 0.009, 0.012, 0.018],
    ("hard_sigmoid", "mnist_bp_amp_v1_c2"): [0.006, 0.009, 0.012, 0.018, 0.024, 0.036],
    ("hard_sigmoid", "mnist_bp_amp_v1_c4"): [0.006, 0.009, 0.012, 0.018, 0.024, 0.036],
    ("perfect_diode", "mnist_bp_amp_v1_c1"): [0.003, 0.006, 0.009, 0.012, 0.018],
    ("perfect_diode", "mnist_bp_amp_v2_c1"): [0.003, 0.006, 0.009, 0.012, 0.018],
    ("perfect_diode", "mnist_bp_amp_v4_c1"): [0.003, 0.006, 0.009, 0.012, 0.018],
    ("perfect_diode", "mnist_bp_amp_v1_c2"): [0.0015, 0.003, 0.006, 0.009, 0.012],
    ("perfect_diode", "mnist_bp_amp_v1_c4"): [0.0015, 0.003, 0.006, 0.009, 0.012],
}

def label(value):
    return f"{float(value):g}".replace("-", "m").replace(".", "p")

def b(value):
    return str(value).strip().lower() in {"1", "true", "yes"}

rows = list(csv.DictReader(open(selected_csv, newline="")))
tasks = []
for row in rows:
    nonlinearity = row["non_linearity"]
    run_name = row["run_name"]
    lr = float(row["selected_lr"])
    tasks.append((nonlinearity, run_name, lr, label(lr), "selected", row["input_gain"], row["iteration_count"]))
    grid = grids.get((nonlinearity, run_name), [])
    if lr in grid:
        idx = grid.index(lr)
        adjacent = None
        if b(row.get("unstable", "")) and idx > 0:
            adjacent = grid[idx - 1]
        elif b(row.get("epoch_limited", "")) and idx < len(grid) - 1:
            adjacent = grid[idx + 1]
        if adjacent is not None:
            tasks.append((nonlinearity, run_name, adjacent, label(adjacent), "adjacent", row["input_gain"], row["iteration_count"]))
for task in tasks:
    print(" ".join(str(part) for part in task))
PY
)

task_id="${SLURM_ARRAY_TASK_ID:-0}"
if [ "${task_id}" -ge "${#TASKS[@]}" ]; then
  echo "task_id=${task_id} outside generated long-LR tasks=${#TASKS[@]}; exiting."
  exit 0
fi

read -r nonlinearity run_name learning_rate lr_label lr_tag input_gain selected_k <<< "${TASKS[$task_id]}"
job_root="${OUTPUT_ROOT}/lr_${lr_label}"

echo "[conv3-long-lr] task=${task_id}/${#TASKS[@]} nonlinearity=${nonlinearity} run=${run_name}"
echo "[conv3-long-lr] tag=${lr_tag} gain=${input_gain} lr=${learning_rate} K=${selected_k} output=${job_root}"

srun "${PYTHON}" "${TRAIN}" \
  --output-root "${job_root}" \
  --dataset-root "${DATASET_ROOT}" \
  --device cuda \
  --model-key mnist_bp_conv_amp \
  --seeds 0 \
  --epochs "${EPOCHS:-50}" \
  --batch-size "${BATCH_SIZE:-4}" \
  --beta "${BETA:-1.0}" \
  --lr-decay "${LR_DECAY:-0.99}" \
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
  --non-linearity "${nonlinearity}" \
  --run-name "${run_name}"
