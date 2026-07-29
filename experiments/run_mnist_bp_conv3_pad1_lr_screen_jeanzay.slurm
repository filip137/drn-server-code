#!/bin/bash -l
#SBATCH --job-name=mnist_conv3_pad1_lr
#SBATCH --output=slurm/mnist_conv3_pad1_lr/%x-%A_%a.out
#SBATCH --error=slurm/mnist_conv3_pad1_lr/%x-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=12:00:00
#SBATCH --account=umg@v100
#SBATCH --qos=qos_gpu-t3
#SBATCH --partition=gpu_p13
#SBATCH --constraint=v100-16g
#SBATCH --array=0-51%40

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
K_CSV="${K_CSV:-${SCRATCH}/server_code/results/mnist_bp_conv3_pad1_k_preflight_seed0/selected_k_by_nonlinearity.csv}"
GAIN_CSV="${GAIN_CSV:-${SCRATCH}/server_code/results/mnist_bp_conv3_pad1_inputgain_screen_seed0_10epoch/selected_input_gain_by_nonlinearity.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRATCH}/server_code/results/mnist_bp_conv3_pad1_lr_screen_seed0_10epoch}"
DATASET_ROOT="${DATASET_ROOT:-${SCRATCH}/datasets/mnist}"
SUMMARY_ONLY="${SUMMARY_ONLY:-0}"

if [ ! -f "${K_CSV}" ]; then
  echo "Expected selected K CSV at ${K_CSV}" >&2
  exit 2
fi
if [ ! -f "${GAIN_CSV}" ]; then
  echo "Expected selected input-gain CSV at ${GAIN_CSV}" >&2
  exit 2
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${SOURCE_ROOT}:${SOURCE_ROOT}/labs:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

mkdir -p "${OUTPUT_ROOT}" slurm/mnist_conv3_pad1_lr

if [ "${SUMMARY_ONLY}" = "1" ]; then
  "${PYTHON}" "${COLLECT}" --root "${OUTPUT_ROOT}"
  exit 0
fi

NONLINEARITIES=(
  hard_sigmoid hard_sigmoid hard_sigmoid hard_sigmoid hard_sigmoid
  hard_sigmoid hard_sigmoid hard_sigmoid hard_sigmoid hard_sigmoid
  hard_sigmoid hard_sigmoid hard_sigmoid hard_sigmoid hard_sigmoid
  hard_sigmoid hard_sigmoid hard_sigmoid hard_sigmoid hard_sigmoid hard_sigmoid
  hard_sigmoid hard_sigmoid hard_sigmoid hard_sigmoid hard_sigmoid hard_sigmoid
  perfect_diode perfect_diode perfect_diode perfect_diode perfect_diode
  perfect_diode perfect_diode perfect_diode perfect_diode perfect_diode
  perfect_diode perfect_diode perfect_diode perfect_diode perfect_diode
  perfect_diode perfect_diode perfect_diode perfect_diode perfect_diode
  perfect_diode perfect_diode perfect_diode perfect_diode perfect_diode
)
RUN_NAMES=(
  mnist_bp_amp_v1_c1 mnist_bp_amp_v1_c1 mnist_bp_amp_v1_c1 mnist_bp_amp_v1_c1 mnist_bp_amp_v1_c1
  mnist_bp_amp_v2_c1 mnist_bp_amp_v2_c1 mnist_bp_amp_v2_c1 mnist_bp_amp_v2_c1 mnist_bp_amp_v2_c1
  mnist_bp_amp_v4_c1 mnist_bp_amp_v4_c1 mnist_bp_amp_v4_c1 mnist_bp_amp_v4_c1 mnist_bp_amp_v4_c1
  mnist_bp_amp_v1_c2 mnist_bp_amp_v1_c2 mnist_bp_amp_v1_c2 mnist_bp_amp_v1_c2 mnist_bp_amp_v1_c2 mnist_bp_amp_v1_c2
  mnist_bp_amp_v1_c4 mnist_bp_amp_v1_c4 mnist_bp_amp_v1_c4 mnist_bp_amp_v1_c4 mnist_bp_amp_v1_c4 mnist_bp_amp_v1_c4
  mnist_bp_amp_v1_c1 mnist_bp_amp_v1_c1 mnist_bp_amp_v1_c1 mnist_bp_amp_v1_c1 mnist_bp_amp_v1_c1
  mnist_bp_amp_v2_c1 mnist_bp_amp_v2_c1 mnist_bp_amp_v2_c1 mnist_bp_amp_v2_c1 mnist_bp_amp_v2_c1
  mnist_bp_amp_v4_c1 mnist_bp_amp_v4_c1 mnist_bp_amp_v4_c1 mnist_bp_amp_v4_c1 mnist_bp_amp_v4_c1
  mnist_bp_amp_v1_c2 mnist_bp_amp_v1_c2 mnist_bp_amp_v1_c2 mnist_bp_amp_v1_c2 mnist_bp_amp_v1_c2
  mnist_bp_amp_v1_c4 mnist_bp_amp_v1_c4 mnist_bp_amp_v1_c4 mnist_bp_amp_v1_c4 mnist_bp_amp_v1_c4
)
LRS=(
  0.003 0.006 0.009 0.012 0.018
  0.003 0.006 0.009 0.012 0.018
  0.003 0.006 0.009 0.012 0.018
  0.006 0.009 0.012 0.018 0.024 0.036
  0.006 0.009 0.012 0.018 0.024 0.036
  0.003 0.006 0.009 0.012 0.018
  0.003 0.006 0.009 0.012 0.018
  0.003 0.006 0.009 0.012 0.018
  0.0015 0.003 0.006 0.009 0.012
  0.0015 0.003 0.006 0.009 0.012
)
LR_LABELS=(
  0p003 0p006 0p009 0p012 0p018
  0p003 0p006 0p009 0p012 0p018
  0p003 0p006 0p009 0p012 0p018
  0p006 0p009 0p012 0p018 0p024 0p036
  0p006 0p009 0p012 0p018 0p024 0p036
  0p003 0p006 0p009 0p012 0p018
  0p003 0p006 0p009 0p012 0p018
  0p003 0p006 0p009 0p012 0p018
  0p0015 0p003 0p006 0p009 0p012
  0p0015 0p003 0p006 0p009 0p012
)

task_id="${SLURM_ARRAY_TASK_ID:-0}"
if [ "${task_id}" -ge "${#RUN_NAMES[@]}" ]; then
  echo "task_id=${task_id} outside LR matrix; exiting."
  exit 0
fi
nonlinearity="${NONLINEARITIES[$task_id]}"
run_name="${RUN_NAMES[$task_id]}"
learning_rate="${LRS[$task_id]}"
lr_label="${LR_LABELS[$task_id]}"

read -r selected_k selected_gain < <("${PYTHON}" - "${K_CSV}" "${GAIN_CSV}" "${nonlinearity}" <<'PY'
import csv, sys
k_csv, gain_csv, nonlinearity = sys.argv[1], sys.argv[2], sys.argv[3]
selected_k = None
selected_gain = None
with open(k_csv, newline="") as handle:
    for row in csv.DictReader(handle):
        if row["non_linearity"] == nonlinearity:
            selected_k = int(float(row["selected_iteration_count"]))
with open(gain_csv, newline="") as handle:
    for row in csv.DictReader(handle):
        if row["non_linearity"] == nonlinearity:
            selected_gain = float(row["selected_input_gain"])
if selected_k is None or selected_gain is None:
    raise SystemExit(f"missing K/gain for {nonlinearity}")
print(selected_k, f"{selected_gain:g}")
PY
)

job_root="${OUTPUT_ROOT}/lr_${lr_label}"

echo "[conv3-lr] task=${task_id} nonlinearity=${nonlinearity} run=${run_name}"
echo "[conv3-lr] gain=${selected_gain} lr=${learning_rate} K=${selected_k} output=${job_root}"

srun "${PYTHON}" "${TRAIN}" \
  --output-root "${job_root}" \
  --dataset-root "${DATASET_ROOT}" \
  --device cuda \
  --model-key mnist_bp_conv_amp \
  --seeds 0 \
  --epochs "${EPOCHS:-10}" \
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
  --input-gain "${selected_gain}" \
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
