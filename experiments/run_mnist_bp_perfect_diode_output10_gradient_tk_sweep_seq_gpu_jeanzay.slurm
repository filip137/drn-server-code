#!/bin/bash -l
#SBATCH --job-name=pd_o10_grad_tk
#SBATCH --output=slurm/pd_o10_grad_tk/%x-%j.out
#SBATCH --error=slurm/pd_o10_grad_tk/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=01:00:00
#SBATCH --account=fmu@v100
#SBATCH --qos=qos_gpu-dev
#SBATCH --partition=gpu_p13
#SBATCH --constraint=v100-16g

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
TRAIN_ROOT="${TRAIN_ROOT:-${FMU_SCRATCH}/server_code/results/mnist_bp_conv23_s221_pad1_perfect_diode_output10_representative_k4k8_seed0_30_50epoch}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${FMU_SCRATCH}/server_code/results/perfect_diode_output10_gradient_tk_sweep_20260701}"
DATASET_ROOT="${DATASET_ROOT:-${UMG_SCRATCH}/datasets/mnist}"
PYTHON="${PYTHON:-python}"

read -r -a T_VALUES <<< "${T_VALUES:-4 8 16}"
read -r -a K_VALUES <<< "${K_VALUES:-1 2 4 6 8}"

cd "${SOURCE_ROOT}"
mkdir -p "${OUTPUT_ROOT}" slurm/pd_o10_grad_tk

export PYTHONUNBUFFERED=1
export PYTHONPATH="${SOURCE_ROOT}:${SOURCE_ROOT}/labs:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

RUN_ARGS=(
  --run-dir "${TRAIN_ROOT}/conv2_old_v1c1/conv2/epochs_30/input_gain_100/lr_0p024/k_4/output_dim_10/perfect_diode/mnist_bp_amp_v1_c1/seed_0"
  --run-dir "${TRAIN_ROOT}/conv2_tuned_v1c1/conv2/epochs_50/input_gain_400/lr_0p012/k_4/output_dim_10/perfect_diode/mnist_bp_amp_v1_c1/seed_0"
  --run-dir "${TRAIN_ROOT}/conv2_v4c1/conv2/epochs_30/input_gain_100/lr_0p012/k_4/output_dim_10/perfect_diode/mnist_bp_amp_v4_c1/seed_0"
  --run-dir "${TRAIN_ROOT}/conv3_repeat_v1c1/conv3/epochs_50/input_gain_360/lr_0p0432/k_8/output_dim_10/perfect_diode/mnist_bp_amp_v1_c1/seed_0"
  --run-dir "${TRAIN_ROOT}/conv3_tuned_v1c1/conv3/epochs_50/input_gain_480/lr_0p0864/k_8/output_dim_10/perfect_diode/mnist_bp_amp_v1_c1/seed_0"
  --run-dir "${TRAIN_ROOT}/conv3_v4c1/conv3/epochs_50/input_gain_360/lr_0p0864/k_8/output_dim_10/perfect_diode/mnist_bp_amp_v4_c1/seed_0"
)

echo "[pd-o10-grad-tk] job=${SLURM_JOB_ID:-manual} project=${IDRPROJ:-unset}"
echo "[pd-o10-grad-tk] source=${SOURCE_ROOT}"
echo "[pd-o10-grad-tk] train_root=${TRAIN_ROOT}"
echo "[pd-o10-grad-tk] dataset=${DATASET_ROOT}"
echo "[pd-o10-grad-tk] output_root=${OUTPUT_ROOT}"
echo "[pd-o10-grad-tk] T_VALUES=${T_VALUES[*]}"
echo "[pd-o10-grad-tk] K_VALUES=${K_VALUES[*]}"

for t_inference in "${T_VALUES[@]}"; do
  for k_training in "${K_VALUES[@]}"; do
    out_csv="${OUTPUT_ROOT}/perfect_diode_output10_gradient_T${t_inference}_K${k_training}_init_final_train128_gpu.csv"
    if [ -s "${out_csv}" ] && [ "${FORCE:-0}" != "1" ]; then
      echo "[pd-o10-grad-tk] skip existing T=${t_inference} K=${k_training}: ${out_csv}"
      continue
    fi
    echo "[pd-o10-grad-tk] run T=${t_inference} K=${k_training} out=${out_csv}"
    srun "${PYTHON}" "${SOURCE_ROOT}/experiments/analyze_mnist_bp_conv_good_run_gradient_saturation.py" \
      "${RUN_ARGS[@]}" \
      --output-csv "${out_csv}" \
      --dataset-root "${DATASET_ROOT}" \
      --device cuda \
      --phase both \
      --split train \
      --batch-size "${BATCH_SIZE:-32}" \
      --max-samples "${MAX_SAMPLES:-128}" \
      --inference-iterations "${t_inference}" \
      --training-iterations "${k_training}" \
      --no-download
  done
done

summary_csv="${OUTPUT_ROOT}/perfect_diode_output10_gradient_tk_sweep_final_summary.csv"
summary_md="${OUTPUT_ROOT}/perfect_diode_output10_gradient_tk_sweep_final_summary.md"
srun "${PYTHON}" "${SOURCE_ROOT}/experiments/summarize_mnist_bp_perfect_diode_gradient_tk_sweep.py" \
  --input-root "${OUTPUT_ROOT}" \
  --output-csv "${summary_csv}" \
  --markdown-output "${summary_md}" \
  --phase final

echo "[pd-o10-grad-tk] summary_csv=${summary_csv}"
echo "[pd-o10-grad-tk] summary_md=${summary_md}"
echo "[pd-o10-grad-tk] done"
