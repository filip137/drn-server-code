#!/bin/bash -l
#SBATCH --job-name=mnist_c23_pd_mgain
#SBATCH --output=slurm/mnist_c23_pd_mgain/%x-%A_%a.out
#SBATCH --error=slurm/mnist_c23_pd_mgain/%x-%A_%a.err
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
#SBATCH --array=0-39%12

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
OUTPUT_ROOT="${OUTPUT_ROOT:-${FMU_SCRATCH}/server_code/results/mnist_bp_conv23_s221_pad1_perfect_diode_v1c1_massive_gain_lr30_k4k8_seed0_30epoch}"
DATASET_ROOT="${DATASET_ROOT:-${UMG_SCRATCH}/datasets/mnist}"
PYTHON="${PYTHON:-python}"

cd "${SOURCE_ROOT}"
mkdir -p "${OUTPUT_ROOT}" slurm/mnist_c23_pd_mgain

export PYTHONUNBUFFERED=1
export PYTHONPATH="${SOURCE_ROOT}:${SOURCE_ROOT}/labs:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

RUN_NAME="mnist_bp_amp_v1_c1"
EPOCHS=30
CONV2_GAINS=(800 1200 1600 2400 3200)
CONV3_GAINS=(960 1440 1920 2880 3840)
LR_MULTS=(0.25 0.5 1 2)

# Center the LR grid at ref_gain * ref_lr / new_gain.
CONV2_REF_GAIN=400
CONV2_REF_LR=0.012
CONV3_REF_GAIN=480
CONV3_REF_LR=0.0864

label_float() {
  local value="$1"
  value="${value//./p}"
  value="${value//-/m}"
  printf '%s' "${value}"
}

scaled_lr() {
  "${PYTHON}" - "$1" "$2" "$3" "$4" <<'PY'
import sys
gain = float(sys.argv[1])
ref_gain = float(sys.argv[2])
ref_lr = float(sys.argv[3])
mult = float(sys.argv[4])
print(f"{(ref_gain * ref_lr / gain) * mult:.12g}")
PY
}

task_id="${SLURM_ARRAY_TASK_ID:-0}"
num_lrs="${#LR_MULTS[@]}"
num_gains="${#CONV2_GAINS[@]}"
per_depth_tasks=$((num_lrs * num_gains))
total_tasks=$((per_depth_tasks * 2))

if [ "${task_id}" -ge "${total_tasks}" ]; then
  echo "task_id=${task_id} outside total_tasks=${total_tasks}"
  exit 0
fi

if [ "${task_id}" = "0" ]; then
  cat > "${OUTPUT_ROOT}/manifest.csv" <<'CSV'
conv_depth,run_name,input_gains,lr_multiplier_grid,epochs,num_iterations,strides,paddings,lr_scaling_rule,reference_gain,reference_lr,selection_note
2,mnist_bp_amp_v1_c1,"800 1200 1600 2400 3200","0.25 0.5 1 2",30,4,"2 2","1 1","lr = reference_gain * reference_lr / input_gain * lr_multiplier",400,0.012,"massive-gain grid; no pruning; optimize directly on 30 epochs"
3,mnist_bp_amp_v1_c1,"960 1440 1920 2880 3840","0.25 0.5 1 2",30,8,"2 2 1","1 1 1","lr = reference_gain * reference_lr / input_gain * lr_multiplier",480,0.0864,"massive-gain grid; no pruning; optimize directly on 30 epochs"
CSV
fi

depth_block=$((task_id / per_depth_tasks))
within_depth=$((task_id % per_depth_tasks))
gain_index=$((within_depth / num_lrs))
lr_index=$((within_depth % num_lrs))
lr_mult="${LR_MULTS[$lr_index]}"

if [ "${depth_block}" -eq 0 ]; then
  conv_depth=2
  input_gain="${CONV2_GAINS[$gain_index]}"
  learning_rate="$(scaled_lr "${input_gain}" "${CONV2_REF_GAIN}" "${CONV2_REF_LR}" "${lr_mult}")"
  selected_k=4
  strides=(2 2)
  paddings=(1 1)
else
  conv_depth=3
  input_gain="${CONV3_GAINS[$gain_index]}"
  learning_rate="$(scaled_lr "${input_gain}" "${CONV3_REF_GAIN}" "${CONV3_REF_LR}" "${lr_mult}")"
  selected_k=8
  strides=(2 2 1)
  paddings=(1 1 1)
fi

gain_label="$(label_float "${input_gain}")"
lr_label="$(label_float "${learning_rate}")"
job_root="${OUTPUT_ROOT}/conv${conv_depth}/epochs_${EPOCHS}/input_gain_${gain_label}/lr_${lr_label}/k_${selected_k}"

echo "[conv23-pd-massive-gain] job=${SLURM_JOB_ID:-manual} task=${task_id} project=${IDRPROJ:-unset}"
echo "[conv23-pd-massive-gain] conv_depth=${conv_depth} run=${RUN_NAME} input_gain=${input_gain} lr=${learning_rate} lr_mult=${lr_mult}"
echo "[conv23-pd-massive-gain] epochs=${EPOCHS} K=${selected_k} strides=${strides[*]} paddings=${paddings[*]}"
echo "[conv23-pd-massive-gain] source=${SOURCE_ROOT} dataset=${DATASET_ROOT} output=${job_root}"

srun "${PYTHON}" "${SOURCE_ROOT}/experiments/train_mnist_bp_conv_amplification_sweep.py" \
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
  --run-name "${RUN_NAME}"

echo "[conv23-pd-massive-gain] done task=${task_id} conv_depth=${conv_depth} run=${RUN_NAME}"
