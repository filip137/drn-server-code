#!/bin/bash -l
#SBATCH --job-name=mnist_c2_p1_v4_k16
#SBATCH --output=slurm/mnist_c2_p1_v4_k16/%x-%A_%a.out
#SBATCH --error=slurm/mnist_c2_p1_v4_k16/%x-%A_%a.err
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
OUTPUT_ROOT="${OUTPUT_ROOT:-${FMU_SCRATCH}/server_code/results/mnist_bp_conv2_pad1_v4c1_k16_seed0_30epoch}"
DATASET_ROOT="${DATASET_ROOT:-${UMG_SCRATCH}/datasets/mnist}"
PYTHON="${PYTHON:-python}"

cd "${SOURCE_ROOT}"
mkdir -p "${OUTPUT_ROOT}" slurm/mnist_c2_p1_v4_k16

export PYTHONUNBUFFERED=1
export PYTHONPATH="${SOURCE_ROOT}:${SOURCE_ROOT}/labs:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

NONLINEARITIES=(perfect_diode hard_sigmoid)
INPUT_GAINS=(100 158.0352325439453)
LEARNING_RATES=(0.012 0.00455594609132)
HARD_SIGMOID_MINS=(-1.5 -4.0)
HARD_SIGMOID_MAXS=(1.5 4.0)

task_id="${SLURM_ARRAY_TASK_ID:-0}"
if [ "${task_id}" -ge "${#NONLINEARITIES[@]}" ]; then
  echo "task_id=${task_id} outside run matrix"
  exit 0
fi

nonlinearity="${NONLINEARITIES[$task_id]}"
input_gain="${INPUT_GAINS[$task_id]}"
learning_rate="${LEARNING_RATES[$task_id]}"
hard_sigmoid_min="${HARD_SIGMOID_MINS[$task_id]}"
hard_sigmoid_max="${HARD_SIGMOID_MAXS[$task_id]}"
job_root="${OUTPUT_ROOT}/k_16"

if [ "${task_id}" = "0" ]; then
  cat > "${OUTPUT_ROOT}/manifest.csv" <<'CSV'
task_id,non_linearity,run_name,input_gain,learning_rate,epochs,num_iterations,strides,paddings,hard_sigmoid_bounds,selection_reason
0,perfect_diode,mnist_bp_amp_v4_c1,100,0.012,30,16,2 2,1 1,-1.5..1.5,Conv2 pad1 perfect-diode v4/c1 row rerun at plateau K
1,hard_sigmoid,mnist_bp_amp_v4_c1,158.0352325439453,0.00455594609132,30,16,2 2,1 1,-4.0..4.0,Conv2 pad1 hard-sigmoid fixed-v_off v4/c1 row rerun at K16
CSV
fi

echo "[conv2-pad1-v4-k16] job=${SLURM_JOB_ID:-manual} task=${task_id} project=${IDRPROJ:-unset}"
echo "[conv2-pad1-v4-k16] nonlinearity=${nonlinearity} run=mnist_bp_amp_v4_c1 input_gain=${input_gain} lr=${learning_rate}"
echo "[conv2-pad1-v4-k16] epochs=30 K=16 strides=2 2 paddings=1 1 hard_sigmoid_bounds=${hard_sigmoid_min}..${hard_sigmoid_max}"
echo "[conv2-pad1-v4-k16] source=${SOURCE_ROOT} dataset=${DATASET_ROOT} output=${job_root}"

srun "${PYTHON}" "${SOURCE_ROOT}/experiments/train_mnist_bp_conv_amplification_sweep.py" \
  --output-root "${job_root}" \
  --dataset-root "${DATASET_ROOT}" \
  --device cuda \
  --model-key mnist_bp_conv_amp \
  --non-linearity "${nonlinearity}" \
  --seeds 0 \
  --epochs 30 \
  --batch-size "${BATCH_SIZE:-4}" \
  --beta 1.0 \
  --lr-decay 0.99 \
  --conv-depth 2 \
  --conv-channels 64 128 256 \
  --kernel-size 3 \
  --strides 2 2 \
  --paddings 1 1 \
  --output-dim 20 \
  --weight-gains 1.0 \
  --weight-min 0.0 \
  --weight-max 100.0 \
  --weight-init-mode kaiming_uniform \
  --input-gain "${input_gain}" \
  --num-iterations 16 \
  --learning-rate "${learning_rate}" \
  --hard-sigmoid-g-on 100.0 \
  --hard-sigmoid-g-off 0.0 \
  --hard-sigmoid-v-min "${hard_sigmoid_min}" \
  --hard-sigmoid-v-max "${hard_sigmoid_max}" \
  --normalize-mean 0.1307 \
  --normalize-std 0.3081 \
  --normalize-scale 0.3 \
  --log-interval "${LOG_INTERVAL:-1000}" \
  --no-download \
  --run-name mnist_bp_amp_v4_c1

echo "[conv2-pad1-v4-k16] done task=${task_id} nonlinearity=${nonlinearity}"
