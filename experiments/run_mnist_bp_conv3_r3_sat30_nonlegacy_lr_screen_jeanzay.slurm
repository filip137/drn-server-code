#!/bin/bash -l
#SBATCH --mnist-conv-legacy-launcher-is-deprecated
# MNIST_CONV_LEGACY_LAUNCHER_DEPRECATED
printf "%s\n" "Expected format: python -m experiments.mnist_conv {run|sweep|collect} ..." >&2
printf "%s\n" "Provided deprecated MNIST Conv launcher invocation: $0 $*" >&2
printf "%s\n" "Deprecated MNIST Conv launcher; no setup, allocation, or work was performed." >&2
exit 2
#SBATCH --job-name=mnist_c3_r3_lr30_v
#SBATCH --output=slurm/mnist_c3_r3_lr30_v/%x-%A_%a.out
#SBATCH --error=slurm/mnist_c3_r3_lr30_v/%x-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=03:00:00
#SBATCH --account=fmu@v100
#SBATCH --qos=qos_gpu-t3
#SBATCH --partition=gpu_p13
#SBATCH --constraint=v100-16g
#SBATCH --array=0-14%15

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
TARGET_ROOT="${TARGET_ROOT:-${FMU_SCRATCH}/server_code/results/mnist_bp_conv3_pad1_hardsigmoid_amp_targets_voff4_sat10_20_30_50_k8_seed0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${FMU_SCRATCH}/server_code/results/mnist_bp_conv3_pad1_hardsigmoid_nonlegacy_sat30_lr_screen_voff4_k8_seed0_10epoch}"
DATASET_ROOT="${DATASET_ROOT:-${UMG_SCRATCH}/datasets/mnist}"
PYTHON="${PYTHON:-python}"
EPOCHS="${EPOCHS:-10}"
TARGET="${TARGET:-0.3}"
SAT_LABEL="${SAT_LABEL:-30}"

cd "${SOURCE_ROOT}"
mkdir -p "${OUTPUT_ROOT}" slurm/mnist_c3_r3_lr30_v

export PYTHONUNBUFFERED=1
export PYTHONPATH="${SOURCE_ROOT}:${SOURCE_ROOT}/labs:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

RUN_NAMES=(
  mnist_bp_amp_v1_c1 mnist_bp_amp_v1_c1 mnist_bp_amp_v1_c1 mnist_bp_amp_v1_c1 mnist_bp_amp_v1_c1
  mnist_bp_amp_v2_c1 mnist_bp_amp_v2_c1 mnist_bp_amp_v2_c1 mnist_bp_amp_v2_c1 mnist_bp_amp_v2_c1
  mnist_bp_amp_v4_c1 mnist_bp_amp_v4_c1 mnist_bp_amp_v4_c1 mnist_bp_amp_v4_c1 mnist_bp_amp_v4_c1
)
LR_MULTS=(
  0.25 0.5 1 2 4
  0.25 0.5 1 2 4
  0.25 0.5 1 2 4
)

task_id="${SLURM_ARRAY_TASK_ID:-0}"
if [ "${task_id}" -ge "${#RUN_NAMES[@]}" ]; then
  echo "task_id=${task_id} outside nonlegacy sat${SAT_LABEL} LR matrix"
  exit 0
fi

run_name="${RUN_NAMES[$task_id]}"
lr_mult="${LR_MULTS[$task_id]}"
target_csv="${TARGET_ROOT}/targets/conv3_voff4_k8_${run_name}_targets.csv"
if [ ! -s "${target_csv}" ]; then
  echo "Expected calibration CSV at ${target_csv}" >&2
  exit 2
fi

read -r input_gain base_lr v_off < <("${PYTHON}" -c 'import csv, sys
target = float(sys.argv[2])
with open(sys.argv[1], newline="") as handle:
    for row in csv.DictReader(handle):
        if abs(float(row["target_saturation"]) - target) < 1e-8:
            print(row["input_gain"], row["learning_rate"], row["v_off"])
            raise SystemExit(0)
raise SystemExit(f"missing target row {target}")' "${target_csv}" "${TARGET}")

learning_rate="$(awk -v lr="${base_lr}" -v mult="${lr_mult}" 'BEGIN { printf "%.12g", lr * mult }')"
label_float() {
  local value="$1"
  value="${value//./p}"
  value="${value//-/m}"
  printf '%s' "${value}"
}
gain_label="$(label_float "${input_gain}")"
mult_label="$(label_float "${lr_mult}")"
lr_label="$(label_float "${learning_rate}")"
job_root="${OUTPUT_ROOT}/target_sat${SAT_LABEL}/input_gain_${gain_label}/lr_mult_${mult_label}/lr_${lr_label}"
v_min="-${v_off}"

echo "[conv3-r3-lr30-nonlegacy] job=${SLURM_JOB_ID:-manual} task=${task_id} project=${IDRPROJ:-unset}"
echo "[conv3-r3-lr30-nonlegacy] run=${run_name} target=${TARGET} lr_mult=${lr_mult} lr=${learning_rate} gain=${input_gain}"
echo "[conv3-r3-lr30-nonlegacy] source=${SOURCE_ROOT} dataset=${DATASET_ROOT} output=${job_root}"

EXTRA_TRAIN_ARGS=()
if [ -n "${MAX_BATCHES:-}" ]; then
  EXTRA_TRAIN_ARGS+=(--max-batches "${MAX_BATCHES}")
fi
if [ -n "${MAX_TEST_BATCHES:-}" ]; then
  EXTRA_TRAIN_ARGS+=(--max-test-batches "${MAX_TEST_BATCHES}")
fi

srun "${PYTHON}" "${SOURCE_ROOT}/experiments/train_mnist_bp_conv_amplification_sweep.py" \
  --output-root "${job_root}" \
  --dataset-root "${DATASET_ROOT}" \
  --device cuda \
  --model-key mnist_bp_conv_amp \
  --non-linearity hard_sigmoid \
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
  --num-iterations 8 \
  --learning-rate "${learning_rate}" \
  --hard-sigmoid-g-on 100.0 \
  --hard-sigmoid-g-off 0.0 \
  --hard-sigmoid-v-min "${v_min}" \
  --hard-sigmoid-v-max "${v_off}" \
  --normalize-mean 0.1307 \
  --normalize-std 0.3081 \
  --normalize-scale 0.3 \
  --log-interval "${LOG_INTERVAL:-500}" \
  "${EXTRA_TRAIN_ARGS[@]}" \
  --no-download \
  --run-name "${run_name}"

echo "[conv3-r3-lr30-nonlegacy] done task=${task_id} run=${run_name} lr_mult=${lr_mult}"
