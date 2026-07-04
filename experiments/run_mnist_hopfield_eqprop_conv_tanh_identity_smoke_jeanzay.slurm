#!/bin/bash -l
#SBATCH --job-name=mnist_hopf_tanh_smoke
#SBATCH --output=slurm/mnist_hopf_tanh_smoke/%x-%A_%a.out
#SBATCH --error=slurm/mnist_hopf_tanh_smoke/%x-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --account=fmu@v100
#SBATCH --qos=qos_gpu-t3
#SBATCH --partition=gpu_p13
#SBATCH --constraint=v100-16g
#SBATCH --array=0-2%3

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
OUTPUT_ROOT="${OUTPUT_ROOT:-${FMU_SCRATCH}/server_code/results/mnist_hopfield_eqprop_conv_tanh_identity_20260630}"
DATASET_ROOT="${DATASET_ROOT:-${UMG_SCRATCH}/datasets/mnist}"
PYTHON="${PYTHON:-python}"
PACK_SIZE="${PACK_SIZE:-2}"

cd "${SOURCE_ROOT}"
mkdir -p "${OUTPUT_ROOT}" slurm/mnist_hopf_tanh_smoke

export PYTHONUNBUFFERED=1
export PYTHONPATH="${SOURCE_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

run_one() {
  local config_id="$1"
  local conv_depth batch run_group
  local stride_args=()
  local padding_args=()

  case "${config_id}" in
    0)
      conv_depth=1
      batch="${BATCH_SIZE:-64}"
      run_group="smoke_tanh_identity_conv1_s2"
      ;;
    1)
      conv_depth=2
      batch="${BATCH_SIZE:-64}"
      run_group="smoke_tanh_identity_conv2_s2"
      ;;
    2)
      conv_depth=3
      batch="${BATCH_SIZE:-64}"
      run_group="smoke_tanh_identity_conv3_s2"
      ;;
    3)
      conv_depth=3
      batch="${STRIDE1_BATCH_SIZE:-16}"
      run_group="smoke_tanh_identity_conv3_s1_pad1"
      stride_args=(--strides 1 1 1)
      padding_args=(--padding 1)
      ;;
    4)
      conv_depth=3
      batch="${BATCH_SIZE:-64}"
      run_group="smoke_tanh_identity_conv3_s221_pad1"
      stride_args=(--strides 2 2 1)
      padding_args=(--padding 1)
      ;;
    *)
      return 0
      ;;
  esac

  echo "[hopf-tanh-smoke] job=${SLURM_JOB_ID:-manual} task=${SLURM_ARRAY_TASK_ID:-0} config=${config_id} conv=${conv_depth} group=${run_group}"
  "${PYTHON}" "${SOURCE_ROOT}/experiments/train_mnist_hopfield_eqprop_conv_sweep.py" \
    --output-root "${OUTPUT_ROOT}" \
    --dataset-root "${DATASET_ROOT}" \
    --dataset mnist \
    --device cuda \
    --run-group "${run_group}" \
    --conv-depth "${conv_depth}" \
    "${stride_args[@]}" \
    "${padding_args[@]}" \
    --activation tanh \
    --input-preprocessing identity \
    --seed 0 \
    --epochs 1 \
    --batch-size "${batch}" \
    --test-batch-size "${TEST_BATCH_SIZE:-200}" \
    --t1 "${T1:-200}" \
    --t2 "${T2:-10}" \
    --beta 0.4 \
    --lr-multiplier 1.0 \
    --max-batches "${MAX_BATCHES:-3}" \
    --max-test-batches "${MAX_TEST_BATCHES:-1}" \
    --log-interval 1 \
    --no-download
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

base_task_id=$((SLURM_ARRAY_TASK_ID * PACK_SIZE))
pids=()
active=0
status=0
for ((offset = 0; offset < PACK_SIZE; offset++)); do
  config_id=$((base_task_id + offset))
  if [ "${config_id}" -ge 5 ]; then
    continue
  fi
  run_one "${config_id}" &
  pids+=("$!")
  active=$((active + 1))
done
if [ "${active}" -gt 0 ]; then
  wait_batch || status=1
fi
exit "${status}"
