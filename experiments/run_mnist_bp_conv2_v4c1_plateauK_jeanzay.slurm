#!/bin/bash -l
#SBATCH --mnist-conv-legacy-launcher-is-deprecated
# MNIST_CONV_LEGACY_LAUNCHER_DEPRECATED
printf "%s\n" "Expected format: python -m experiments.mnist_conv {run|sweep|collect} ..." >&2
printf "%s\n" "Provided deprecated MNIST Conv launcher invocation: $0 $*" >&2
printf "%s\n" "Deprecated MNIST Conv launcher; no setup, allocation, or work was performed." >&2
exit 2
#SBATCH --job-name=mnist_c2_v4_plateauK
#SBATCH --output=slurm/mnist_c2_v4_plateauK/%x-%A_%a.out
#SBATCH --error=slurm/mnist_c2_v4_plateauK/%x-%A_%a.err
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
#SBATCH --array=0-1%2

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
OUTPUT_ROOT="${OUTPUT_ROOT:-${FMU_SCRATCH}/server_code/results/mnist_bp_conv2_v4c1_plateauK_seed0_50epoch}"
DATASET_ROOT="${DATASET_ROOT:-${UMG_SCRATCH}/datasets/mnist}"
PYTHON="${PYTHON:-python}"

cd "${SOURCE_ROOT}"
mkdir -p "${OUTPUT_ROOT}" slurm/mnist_c2_v4_plateauK

export PYTHONUNBUFFERED=1
export PYTHONPATH="${SOURCE_ROOT}:${SOURCE_ROOT}/labs:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

NONLINEARITIES=(perfect_diode hard_sigmoid)
K_VALUES=(16 32)

task_id="${SLURM_ARRAY_TASK_ID:-0}"
if [ "${task_id}" -ge "${#NONLINEARITIES[@]}" ]; then
  echo "task_id=${task_id} outside run matrix"
  exit 0
fi

nonlinearity="${NONLINEARITIES[$task_id]}"
selected_k="${K_VALUES[$task_id]}"
job_root="${OUTPUT_ROOT}/k_${selected_k}"

if [ "${task_id}" = "0" ]; then
  cat > "${OUTPUT_ROOT}/manifest.csv" <<'CSV'
task_id,non_linearity,run_name,input_gain,learning_rate,epochs,num_iterations,strides,paddings,selection_reason
0,perfect_diode,mnist_bp_amp_v4_c1,100,0.012,50,16,2 2,0 0,first high-K point where early-conv gradients are stable/dead in diagnostics
1,hard_sigmoid,mnist_bp_amp_v4_c1,100,0.012,50,32,2 2,0 0,first post-fix point where residual/voltage updates have effectively plateaued
CSV
fi

echo "[conv2-v4-plateauK] job=${SLURM_JOB_ID:-manual} task=${task_id} project=${IDRPROJ:-unset}"
echo "[conv2-v4-plateauK] nonlinearity=${nonlinearity} run=mnist_bp_amp_v4_c1 input_gain=100 lr=0.012"
echo "[conv2-v4-plateauK] epochs=50 K=${selected_k} strides=2 2 paddings=0 0"
echo "[conv2-v4-plateauK] source=${SOURCE_ROOT} dataset=${DATASET_ROOT} output=${job_root}"

srun "${PYTHON}" "${SOURCE_ROOT}/experiments/train_mnist_bp_conv_amplification_sweep.py" \
  --output-root "${job_root}" \
  --dataset-root "${DATASET_ROOT}" \
  --device cuda \
  --model-key mnist_bp_conv_amp \
  --non-linearity "${nonlinearity}" \
  --seeds 0 \
  --epochs 50 \
  --batch-size "${BATCH_SIZE:-4}" \
  --beta 1.0 \
  --lr-decay 0.99 \
  --conv-depth 2 \
  --conv-channels 64 128 256 \
  --kernel-size 3 \
  --strides 2 2 \
  --paddings 0 0 \
  --output-dim 20 \
  --weight-gains 1.0 \
  --weight-min 0.0 \
  --weight-max 100.0 \
  --weight-init-mode kaiming_uniform \
  --input-gain 100.0 \
  --num-iterations "${selected_k}" \
  --learning-rate 0.012 \
  --hard-sigmoid-g-on 100.0 \
  --hard-sigmoid-g-off 0.0 \
  --hard-sigmoid-v-min -1.5 \
  --hard-sigmoid-v-max 1.5 \
  --normalize-mean 0.1307 \
  --normalize-std 0.3081 \
  --normalize-scale 0.3 \
  --log-interval "${LOG_INTERVAL:-1000}" \
  --no-download \
  --run-name mnist_bp_amp_v4_c1

echo "[conv2-v4-plateauK] done task=${task_id} nonlinearity=${nonlinearity} K=${selected_k}"
