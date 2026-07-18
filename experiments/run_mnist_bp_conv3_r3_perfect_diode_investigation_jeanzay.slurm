#!/bin/bash -l
#SBATCH --mnist-conv-legacy-launcher-is-deprecated
# MNIST_CONV_LEGACY_LAUNCHER_DEPRECATED
printf "%s\n" "Expected format: python -m experiments.mnist_conv {run|sweep|collect} ..." >&2
printf "%s\n" "Provided deprecated MNIST Conv launcher invocation: $0 $*" >&2
printf "%s\n" "Deprecated MNIST Conv launcher; no setup, allocation, or work was performed." >&2
exit 2
#SBATCH --job-name=mnist_c3_pd_inv
#SBATCH --output=slurm/mnist_c3_pd_inv/%x-%A_%a.out
#SBATCH --error=slurm/mnist_c3_pd_inv/%x-%A_%a.err
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
#SBATCH --array=0-6%7

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
OUTPUT_ROOT="${OUTPUT_ROOT:-${FMU_SCRATCH}/server_code/results/mnist_bp_conv3_s221_pad1_perfect_diode_investigate_v1v4_k8_seed0_1_2_50epoch}"
DATASET_ROOT="${DATASET_ROOT:-${UMG_SCRATCH}/datasets/mnist}"
PYTHON="${PYTHON:-python}"

cd "${SOURCE_ROOT}"
mkdir -p "${OUTPUT_ROOT}" slurm/mnist_c3_pd_inv

export PYTHONUNBUFFERED=1
export PYTHONPATH="${SOURCE_ROOT}:${SOURCE_ROOT}/labs:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

EPOCHS=50
CONV_DEPTH=3
K=8
STRIDES=(2 2 1)
PADDINGS=(1 1 1)

# tag|run_name|input_gain|learning_rate|seed|purpose
ROWS=(
  "v1_selected_seed1|mnist_bp_amp_v1_c1|480|0.0864|1|seed check for selected v1/c1 row"
  "v1_selected_seed2|mnist_bp_amp_v1_c1|480|0.0864|2|seed check for selected v1/c1 row"
  "v4_repeat_seed1|mnist_bp_amp_v4_c1|360|0.0864|1|seed check for v4/c1 comparator"
  "v4_repeat_seed2|mnist_bp_amp_v4_c1|360|0.0864|2|seed check for v4/c1 comparator"
  "v1_gain480_lr0432_seed0|mnist_bp_amp_v1_c1|480|0.0432|0|no-prune lower-LR follow-up pruned at epoch 5"
  "v1_gain360_lr0864_seed0|mnist_bp_amp_v1_c1|360|0.0864|0|no-prune old-gain faster-LR follow-up"
  "v1_gain640_lr0864_seed0|mnist_bp_amp_v1_c1|640|0.0864|0|no-prune high-gain surviving-row follow-up"
)

label_float() {
  local value="$1"
  value="${value//./p}"
  value="${value//-/m}"
  printf '%s' "${value}"
}

task_id="${SLURM_ARRAY_TASK_ID:-0}"
if [ "${task_id}" -ge "${#ROWS[@]}" ]; then
  echo "task_id=${task_id} outside investigation matrix"
  exit 0
fi

IFS='|' read -r tag run_name input_gain learning_rate seed purpose <<< "${ROWS[$task_id]}"
gain_label="$(label_float "${input_gain}")"
lr_label="$(label_float "${learning_rate}")"
job_root="${OUTPUT_ROOT}/conv3/epochs_${EPOCHS}/input_gain_${gain_label}/lr_${lr_label}/k_${K}"

if [ "${task_id}" = "0" ]; then
  cat > "${OUTPUT_ROOT}/manifest.csv" <<'CSV'
task_id,tag,run_name,input_gain,learning_rate,seed,epochs,num_iterations,strides,paddings,purpose
0,v1_selected_seed1,mnist_bp_amp_v1_c1,480,0.0864,1,50,8,"2 2 1","1 1 1","seed check for selected v1/c1 row"
1,v1_selected_seed2,mnist_bp_amp_v1_c1,480,0.0864,2,50,8,"2 2 1","1 1 1","seed check for selected v1/c1 row"
2,v4_repeat_seed1,mnist_bp_amp_v4_c1,360,0.0864,1,50,8,"2 2 1","1 1 1","seed check for v4/c1 comparator"
3,v4_repeat_seed2,mnist_bp_amp_v4_c1,360,0.0864,2,50,8,"2 2 1","1 1 1","seed check for v4/c1 comparator"
4,v1_gain480_lr0432_seed0,mnist_bp_amp_v1_c1,480,0.0432,0,50,8,"2 2 1","1 1 1","no-prune lower-LR follow-up pruned at epoch 5"
5,v1_gain360_lr0864_seed0,mnist_bp_amp_v1_c1,360,0.0864,0,50,8,"2 2 1","1 1 1","no-prune old-gain faster-LR follow-up"
6,v1_gain640_lr0864_seed0,mnist_bp_amp_v1_c1,640,0.0864,0,50,8,"2 2 1","1 1 1","no-prune high-gain surviving-row follow-up"
CSV
fi

echo "[conv3-pd-investigate] job=${SLURM_JOB_ID:-manual} task=${task_id} project=${IDRPROJ:-unset}"
echo "[conv3-pd-investigate] tag=${tag} purpose=${purpose}"
echo "[conv3-pd-investigate] run=${run_name} input_gain=${input_gain} lr=${learning_rate} seed=${seed}"
echo "[conv3-pd-investigate] epochs=${EPOCHS} K=${K} strides=${STRIDES[*]} paddings=${PADDINGS[*]}"
echo "[conv3-pd-investigate] source=${SOURCE_ROOT} dataset=${DATASET_ROOT} output=${job_root}"

srun "${PYTHON}" "${SOURCE_ROOT}/experiments/train_mnist_bp_conv_amplification_sweep.py" \
  --output-root "${job_root}" \
  --dataset-root "${DATASET_ROOT}" \
  --device cuda \
  --model-key mnist_bp_conv_amp \
  --non-linearity perfect_diode \
  --seeds "${seed}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE:-4}" \
  --beta 1.0 \
  --lr-decay 0.99 \
  --conv-depth "${CONV_DEPTH}" \
  --conv-channels 64 128 256 \
  --kernel-size 3 \
  --strides "${STRIDES[@]}" \
  --paddings "${PADDINGS[@]}" \
  --output-dim 20 \
  --weight-gains 1.0 \
  --weight-min 0.0 \
  --weight-max 100.0 \
  --weight-init-mode kaiming_uniform \
  --input-gain "${input_gain}" \
  --num-iterations "${K}" \
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

echo "[conv3-pd-investigate] done task=${task_id} tag=${tag}"
