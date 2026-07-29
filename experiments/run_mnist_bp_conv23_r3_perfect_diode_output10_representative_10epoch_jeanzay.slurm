#!/bin/bash -l
#SBATCH --job-name=mnist_c23_pd_o10e10
#SBATCH --output=slurm/mnist_c23_pd_o10e10/%x-%A_%a.out
#SBATCH --error=slurm/mnist_c23_pd_o10e10/%x-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=06:00:00
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
OUTPUT_ROOT="${OUTPUT_ROOT:-${FMU_SCRATCH}/server_code/results/mnist_bp_conv23_s221_pad1_perfect_diode_output10_representative_k4k8_seed0_10epoch}"
DATASET_ROOT="${DATASET_ROOT:-${UMG_SCRATCH}/datasets/mnist}"
PYTHON="${PYTHON:-python}"

cd "${SOURCE_ROOT}"
mkdir -p "${OUTPUT_ROOT}" slurm/mnist_c23_pd_o10e10

export PYTHONUNBUFFERED=1
export PYTHONPATH="${SOURCE_ROOT}:${SOURCE_ROOT}/labs:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

CONV_DEPTHS=(2 2 2 3 3 3)
RUN_NAMES=(
  mnist_bp_amp_v1_c1
  mnist_bp_amp_v1_c1
  mnist_bp_amp_v4_c1
  mnist_bp_amp_v1_c1
  mnist_bp_amp_v1_c1
  mnist_bp_amp_v4_c1
)
INPUT_GAINS=(100 400 100 360 480 360)
LEARNING_RATES=(0.024 0.012 0.012 0.0432 0.0864 0.0864)
NUM_ITERATIONS=(4 4 4 8 8 8)
STRIDE_ARGS=("2 2" "2 2" "2 2" "2 2 1" "2 2 1" "2 2 1")
PADDING_ARGS=("1 1" "1 1" "1 1" "1 1 1" "1 1 1" "1 1 1")
LABELS=(
  conv2_old_v1c1
  conv2_tuned_v1c1
  conv2_v4c1
  conv3_repeat_v1c1
  conv3_tuned_v1c1
  conv3_v4c1
)
epochs=10

task_id="${SLURM_ARRAY_TASK_ID:-0}"
if [ "${task_id}" -ge "${#RUN_NAMES[@]}" ]; then
  echo "task_id=${task_id} outside run matrix"
  exit 0
fi

conv_depth="${CONV_DEPTHS[$task_id]}"
run_name="${RUN_NAMES[$task_id]}"
input_gain="${INPUT_GAINS[$task_id]}"
learning_rate="${LEARNING_RATES[$task_id]}"
selected_k="${NUM_ITERATIONS[$task_id]}"
row_label="${LABELS[$task_id]}"
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
job_root="${OUTPUT_ROOT}/${row_label}/conv${conv_depth}/epochs_${epochs}/input_gain_${gain_label}/lr_${lr_label}/k_${selected_k}/output_dim_10"

if [ "${task_id}" = "0" ]; then
  cat > "${OUTPUT_ROOT}/manifest.csv" <<'CSV'
task_id,label,conv_depth,run_name,input_gain,learning_rate,epochs,num_iterations,strides,paddings,output_dim,source
0,conv2_old_v1c1,2,mnist_bp_amp_v1_c1,100,0.024,10,4,2 2,1 1,10,Conv2 old perfect-diode v1/c1 output-10 smoke
1,conv2_tuned_v1c1,2,mnist_bp_amp_v1_c1,400,0.012,10,4,2 2,1 1,10,Conv2 tuned perfect-diode v1/c1 output-10 smoke
2,conv2_v4c1,2,mnist_bp_amp_v4_c1,100,0.012,10,4,2 2,1 1,10,Conv2 perfect-diode v4/c1 output-10 smoke
3,conv3_repeat_v1c1,3,mnist_bp_amp_v1_c1,360,0.0432,10,8,2 2 1,1 1 1,10,Conv3 repeat perfect-diode v1/c1 output-10 smoke
4,conv3_tuned_v1c1,3,mnist_bp_amp_v1_c1,480,0.0864,10,8,2 2 1,1 1 1,10,Conv3 tuned perfect-diode v1/c1 output-10 smoke
5,conv3_v4c1,3,mnist_bp_amp_v4_c1,360,0.0864,10,8,2 2 1,1 1 1,10,Conv3 perfect-diode v4/c1 output-10 smoke
CSV
fi

echo "[pd-o10-10ep] job=${SLURM_JOB_ID:-manual} task=${task_id} project=${IDRPROJ:-unset}"
echo "[pd-o10-10ep] label=${row_label} conv_depth=${conv_depth} run=${run_name} input_gain=${input_gain} lr=${learning_rate}"
echo "[pd-o10-10ep] epochs=${epochs} K=${selected_k} strides=${strides[*]} paddings=${paddings[*]} output_dim=10"
echo "[pd-o10-10ep] source=${SOURCE_ROOT} dataset=${DATASET_ROOT} output=${job_root}"

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
  --output-dim 10 \
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

echo "[pd-o10-10ep] done task=${task_id} label=${row_label}"
