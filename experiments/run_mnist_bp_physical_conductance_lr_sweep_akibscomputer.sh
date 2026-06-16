#!/usr/bin/env bash
set -euo pipefail

PYTHON="/home/filip/miniconda3/envs/py312/bin/python"
TRAIN="/home/filip/server_code/experiments/train_mnist_bp_amplification_sweep.py"
SUMMARIZE="/home/filip/server_code/experiments/summarize_mnist_bp_physical_conductance_lr_sweep.py"

OUTPUT_ROOT="/home/filip/server_code/results/mnist_bp_amp_physical_conductance_lr_sweep"
LOG_DIR="${OUTPUT_ROOT}/logs"

mkdir -p "${LOG_DIR}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="/tmp/matplotlib-mnist-physical-conductance-lr"

echo "$(date) launching MNIST BP physical-conductance amplification LR sweep" | tee -a "${LOG_DIR}/queue.log"

run_names=(
  mnist_bp_amp_v1_c1
  mnist_bp_amp_v2_c1
  mnist_bp_amp_v4_c1
  mnist_bp_amp_v1_c2
  mnist_bp_amp_v1_c4
)

seeds=(0 1 2)

lr_specs=(
  "1p25e-6:0.00000125"
  "1p75e-6:0.00000175"
  "2p5e-6:0.0000025"
  "3p5e-6:0.0000035"
)

pids=()
status=0

for lr_spec in "${lr_specs[@]}"; do
  lr_label="${lr_spec%%:*}"
  lr="${lr_spec##*:}"
  lr_root="${OUTPUT_ROOT}/lr_${lr_label}"
  for run_name in "${run_names[@]}"; do
    for seed in "${seeds[@]}"; do
      log_path="${LOG_DIR}/lr_${lr_label}_${run_name}_seed_${seed}.log"
      echo "$(date) lr=${lr} run=${run_name} seed=${seed} -> ${log_path}" | tee -a "${LOG_DIR}/queue.log"
      "${PYTHON}" "${TRAIN}" \
        --output-root "${lr_root}" \
        --dataset-root /home/filip/datasets/mnist \
        --device cuda \
        --seeds "${seed}" \
        --epochs 10 \
        --batch-size 4 \
        --num-iterations 4 \
        --learning-rate "${lr}" "${lr}" "${lr}" \
        --beta 1.0 \
        --lr-decay 1.0 \
        --weight-gains 0.007127636354360399 0.0018000000000000002 \
        --weight-min 0.0 \
        --weight-max 0.00018 \
        --weight-init-mode kaiming_uniform \
        --input-gain 50.0 \
        --normalize-mean 0.13066 \
        --normalize-std 0.308108 \
        --normalize-scale 1.0 \
        --run-name "${run_name}" \
        --no-download \
        >"${log_path}" 2>&1 &
      pids+=("$!")
    done
  done
done

for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

for lr_spec in "${lr_specs[@]}"; do
  lr_label="${lr_spec%%:*}"
  lr_root="${OUTPUT_ROOT}/lr_${lr_label}"
  "${PYTHON}" "${TRAIN}" \
    --output-root "${lr_root}" \
    --summary-only >"${LOG_DIR}/summary_lr_${lr_label}.log" 2>&1 || status=1
done

"${PYTHON}" "${SUMMARIZE}" \
  --output-root "${OUTPUT_ROOT}" >"${LOG_DIR}/summary_all.log" 2>&1 || status=1

if [ "${status}" -eq 0 ]; then
  echo "$(date) MNIST BP physical-conductance amplification LR sweep finished successfully" | tee -a "${LOG_DIR}/queue.log"
else
  echo "$(date) MNIST BP physical-conductance amplification LR sweep finished with errors" | tee -a "${LOG_DIR}/queue.log"
fi

exit "${status}"
