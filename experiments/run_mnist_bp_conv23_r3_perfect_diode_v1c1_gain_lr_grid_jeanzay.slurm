#!/bin/bash -l
#SBATCH --mnist-conv-legacy-launcher-is-deprecated
# MNIST_CONV_LEGACY_LAUNCHER_DEPRECATED
printf "%s\n" "Expected format: python -m experiments.mnist_conv {run|sweep|collect} ..." >&2
printf "%s\n" "Provided deprecated MNIST Conv launcher invocation: $0 $*" >&2
printf "%s\n" "Deprecated MNIST Conv launcher; no setup, allocation, or work was performed." >&2
exit 2
#SBATCH --job-name=mnist_c23_pd_gainlr
#SBATCH --output=slurm/mnist_c23_pd_gainlr/%x-%A_%a.out
#SBATCH --error=slurm/mnist_c23_pd_gainlr/%x-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=08:00:00
#SBATCH --account=fmu@v100
#SBATCH --qos=qos_gpu-t3
#SBATCH --partition=gpu_p13
#SBATCH --constraint=v100-16g
#SBATCH --array=0-59%20

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
OUTPUT_ROOT="${OUTPUT_ROOT:-${FMU_SCRATCH}/server_code/results/mnist_bp_conv23_s221_pad1_perfect_diode_v1c1_gain_lr_downbias_k4k8_seed0_10epoch}"
DATASET_ROOT="${DATASET_ROOT:-${UMG_SCRATCH}/datasets/mnist}"
PYTHON="${PYTHON:-python}"

cd "${SOURCE_ROOT}"
mkdir -p "${OUTPUT_ROOT}" slurm/mnist_c23_pd_gainlr

export PYTHONUNBUFFERED=1
export PYTHONPATH="${SOURCE_ROOT}:${SOURCE_ROOT}/labs:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

RUN_NAME="mnist_bp_amp_v1_c1"
EPOCHS=10
CONV2_GAINS=(100 140 200 280 400)
CONV3_GAINS=(360 480 640 900 1200)
CONV2_LRS=(0.0015 0.003 0.006 0.012 0.024 0.048)
CONV3_LRS=(0.0027 0.0054 0.0108 0.0216 0.0432 0.0864)
CONV2_PRUNE_MIN_BEST_ACC="${CONV2_PRUNE_MIN_BEST_ACC:-0.9537}"
CONV3_PRUNE_MIN_BEST_ACC="${CONV3_PRUNE_MIN_BEST_ACC:-0.8921}"
PRUNE_AFTER_EPOCHS="${PRUNE_AFTER_EPOCHS:-5}"

label_float() {
  local value="$1"
  value="${value//./p}"
  value="${value//-/m}"
  printf '%s' "${value}"
}

task_id="${SLURM_ARRAY_TASK_ID:-0}"
num_lrs="${#CONV2_LRS[@]}"
num_gains="${#CONV2_GAINS[@]}"
per_depth_tasks=$((num_lrs * num_gains))
total_tasks=$((per_depth_tasks * 2))

if [ "${task_id}" -ge "${total_tasks}" ]; then
  echo "task_id=${task_id} outside total_tasks=${total_tasks}"
  exit 0
fi

if [ "${task_id}" = "0" ]; then
  cat > "${OUTPUT_ROOT}/manifest.csv" <<'CSV'
conv_depth,run_name,input_gains,learning_rates,epochs,num_iterations,strides,paddings,prune_after_epochs,prune_min_best_test_accuracy,selection_note
2,mnist_bp_amp_v1_c1,"100 140 200 280 400","0.0015 0.003 0.006 0.012 0.024 0.048",10,4,"2 2","1 1",5,0.9537,"downward-biased LR grid; current LR 0.024 retained"
3,mnist_bp_amp_v1_c1,"360 480 640 900 1200","0.0027 0.0054 0.0108 0.0216 0.0432 0.0864",10,8,"2 2 1","1 1 1",5,0.8921,"downward-biased LR grid; current LR 0.0432 retained"
CSV
fi

depth_block=$((task_id / per_depth_tasks))
within_depth=$((task_id % per_depth_tasks))
gain_index=$((within_depth / num_lrs))
lr_index=$((within_depth % num_lrs))

if [ "${depth_block}" -eq 0 ]; then
  conv_depth=2
  input_gain="${CONV2_GAINS[$gain_index]}"
  learning_rate="${CONV2_LRS[$lr_index]}"
  selected_k=4
  prune_min_best_acc="${CONV2_PRUNE_MIN_BEST_ACC}"
  strides=(2 2)
  paddings=(1 1)
else
  conv_depth=3
  input_gain="${CONV3_GAINS[$gain_index]}"
  learning_rate="${CONV3_LRS[$lr_index]}"
  selected_k=8
  prune_min_best_acc="${CONV3_PRUNE_MIN_BEST_ACC}"
  strides=(2 2 1)
  paddings=(1 1 1)
fi

gain_label="$(label_float "${input_gain}")"
lr_label="$(label_float "${learning_rate}")"
job_root="${OUTPUT_ROOT}/conv${conv_depth}/input_gain_${gain_label}/lr_${lr_label}/k_${selected_k}"

echo "[conv23-pd-gainlr] job=${SLURM_JOB_ID:-manual} task=${task_id} project=${IDRPROJ:-unset}"
echo "[conv23-pd-gainlr] conv_depth=${conv_depth} run=${RUN_NAME} input_gain=${input_gain} lr=${learning_rate}"
echo "[conv23-pd-gainlr] epochs=${EPOCHS} K=${selected_k} strides=${strides[*]} paddings=${paddings[*]}"
echo "[conv23-pd-gainlr] prune_after=${PRUNE_AFTER_EPOCHS} prune_min_best_acc=${prune_min_best_acc}"
echo "[conv23-pd-gainlr] source=${SOURCE_ROOT} dataset=${DATASET_ROOT} output=${job_root}"

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
  --prune-after-epochs "${PRUNE_AFTER_EPOCHS}" \
  --prune-min-best-test-accuracy "${prune_min_best_acc}" \
  --no-download \
  --run-name "${RUN_NAME}"

echo "[conv23-pd-gainlr] done task=${task_id} conv_depth=${conv_depth} run=${RUN_NAME}"
