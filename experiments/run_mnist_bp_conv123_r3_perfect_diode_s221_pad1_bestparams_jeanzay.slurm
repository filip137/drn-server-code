#!/bin/bash -l
#SBATCH --mnist-conv-legacy-launcher-is-deprecated
# MNIST_CONV_LEGACY_LAUNCHER_DEPRECATED
printf "%s\n" "Expected format: python -m experiments.mnist_conv {run|sweep|collect} ..." >&2
printf "%s\n" "Provided deprecated MNIST Conv launcher invocation: $0 $*" >&2
printf "%s\n" "Deprecated MNIST Conv launcher; no setup, allocation, or work was performed." >&2
exit 2
#SBATCH --job-name=mnist_c123_pd_s221
#SBATCH --output=slurm/mnist_c123_pd_s221/%x-%A_%a.out
#SBATCH --error=slurm/mnist_c123_pd_s221/%x-%A_%a.err
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
#SBATCH --array=0-5%6

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
OUTPUT_ROOT="${OUTPUT_ROOT:-${FMU_SCRATCH}/server_code/results/mnist_bp_conv123_s221_pad1_perfect_diode_v1c1_v4c1_bestparams_k4k8_seed0_10_30_50epoch}"
DATASET_ROOT="${DATASET_ROOT:-${UMG_SCRATCH}/datasets/mnist}"
PYTHON="${PYTHON:-python}"

cd "${SOURCE_ROOT}"
mkdir -p "${OUTPUT_ROOT}" slurm/mnist_c123_pd_s221

export PYTHONUNBUFFERED=1
export PYTHONPATH="${SOURCE_ROOT}:${SOURCE_ROOT}/labs:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

CONV_DEPTHS=(1 1 2 2 3 3)
RUN_NAMES=(
  mnist_bp_amp_v1_c1
  mnist_bp_amp_v4_c1
  mnist_bp_amp_v1_c1
  mnist_bp_amp_v4_c1
  mnist_bp_amp_v1_c1
  mnist_bp_amp_v4_c1
)
INPUT_GAINS=(40 40 100 100 360 360)
LEARNING_RATES=(0.006 0.009 0.024 0.012 0.0432 0.0864)
EPOCHS_BY_TASK=(10 10 30 30 50 50)
NUM_ITERATIONS=(4 4 4 4 8 8)
STRIDE_ARGS=("2" "2" "2 2" "2 2" "2 2 1" "2 2 1")
PADDING_ARGS=("1" "1" "1 1" "1 1" "1 1 1" "1 1 1")

task_id="${SLURM_ARRAY_TASK_ID:-0}"
if [ "${task_id}" -ge "${#RUN_NAMES[@]}" ]; then
  echo "task_id=${task_id} outside run matrix"
  exit 0
fi

conv_depth="${CONV_DEPTHS[$task_id]}"
run_name="${RUN_NAMES[$task_id]}"
input_gain="${INPUT_GAINS[$task_id]}"
learning_rate="${LEARNING_RATES[$task_id]}"
epochs="${EPOCHS_BY_TASK[$task_id]}"
selected_k="${NUM_ITERATIONS[$task_id]}"
read -r -a strides <<< "${STRIDE_ARGS[$task_id]}"
read -r -a paddings <<< "${PADDING_ARGS[$task_id]}"

label_float() {
  local value="$1"
  value="${value//./p}"
  value="${value//-/m}"
  printf '%s' "${value}"
}

gain_label="$(label_float "${input_gain}")"
lr_label="$(label_float "${learning_rate}")"
job_root="${OUTPUT_ROOT}/conv${conv_depth}/epochs_${epochs}/input_gain_${gain_label}/lr_${lr_label}/k_${selected_k}"

if [ "${task_id}" = "0" ]; then
  cat > "${OUTPUT_ROOT}/manifest.csv" <<'CSV'
task_id,conv_depth,run_name,input_gain,learning_rate,epochs,num_iterations,strides,paddings,source
0,1,mnist_bp_amp_v1_c1,40,0.006,10,4,2,1,conv1 perfect_diode best-per-amp summary
1,1,mnist_bp_amp_v4_c1,40,0.009,10,4,2,1,conv1 perfect_diode best-per-amp summary
2,2,mnist_bp_amp_v1_c1,100,0.024,30,4,2 2,1 1,conv2 perfect_diode best-per-amp summary
3,2,mnist_bp_amp_v4_c1,100,0.012,30,4,2 2,1 1,conv2 perfect_diode best-per-amp summary
4,3,mnist_bp_amp_v1_c1,360,0.0432,50,8,2 2 1,1 1 1,conv3 perfect_diode selected LR screen
5,3,mnist_bp_amp_v4_c1,360,0.0864,50,8,2 2 1,1 1 1,conv3 perfect_diode selected LR screen
CSV
fi

echo "[conv123-pd-s221] job=${SLURM_JOB_ID:-manual} task=${task_id} project=${IDRPROJ:-unset}"
echo "[conv123-pd-s221] conv_depth=${conv_depth} run=${run_name} input_gain=${input_gain} lr=${learning_rate}"
echo "[conv123-pd-s221] epochs=${epochs} K=${selected_k} strides=${strides[*]} paddings=${paddings[*]}"
echo "[conv123-pd-s221] source=${SOURCE_ROOT} dataset=${DATASET_ROOT} output=${job_root}"

srun "${PYTHON}" "${SOURCE_ROOT}/experiments/train_mnist_bp_conv_amplification_sweep.py" \
  --output-root "${job_root}" \
  --dataset-root "${DATASET_ROOT}" \
  --device cuda \
  --model-key mnist_bp_conv_amp \
  --non-linearity perfect_diode \
  --seeds 0 \
  --epochs "${epochs}" \
  --batch-size "${BATCH_SIZE:-4}" \
  --beta 1.0 \
  --lr-decay 0.99 \
  --conv-depth "${conv_depth}" \
  --conv-channels 64 128 256 \
  --kernel-size 3 \
  --strides "${strides[@]}" \
  --paddings "${paddings[@]}" \
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

echo "[conv123-pd-s221] done task=${task_id} conv_depth=${conv_depth} run=${run_name}"
