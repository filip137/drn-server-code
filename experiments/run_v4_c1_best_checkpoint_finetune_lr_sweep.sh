#!/usr/bin/env bash
set -euo pipefail

BASE_ROOT="/home/filip/server_code/results/mnist_bp_amp_v4_c1_continue_lr0p012_10epoch_from_legacy_preproc"
PYTHON="/home/filip/miniconda3/envs/py312/bin/python"
RUNNER="/home/filip/server_code/experiments/train_mnist_bp_amplification_sweep.py"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

launch_group() {
  local lr="$1"
  local tag="$2"
  local out_root="/home/filip/server_code/results/mnist_bp_amp_v4_c1_finetune_${tag}_20epoch_from_lr0p012_best"
  local log_dir="${out_root}/logs"
  mkdir -p "${log_dir}"

  echo "$(date) launching v4/c1 fine-tune ${tag}" | tee -a "${log_dir}/queue.log"

  local common_args=(
    "${RUNNER}"
    --device cuda
    --output-root "${out_root}"
    --run-name mnist_bp_amp_v4_c1
    --epochs 20
    --batch-size 4
    --num-iterations 4
    --input-gain 100
    --learning-rate "${lr}" "${lr}" "${lr}"
    --lr-decay 0.99
    --beta 1.0
    --weight-gains 1 1
    --weight-min 0.0
    --weight-max 100.0
    --weight-init-mode kaiming_uniform
    --normalize-mean 0.1307
    --normalize-std 0.3081
    --normalize-scale 0.3
    --init-checkpoint-template "${BASE_ROOT}/{run_name}/seed_{seed}/best_model.pt"
    --no-download
  )

  local pids=()
  for seed in 0 1 2; do
    local log_path="${log_dir}/mnist_bp_amp_v4_c1_seed_${seed}.log"
    echo "$(date) ${tag} seed ${seed} -> ${log_path}" | tee -a "${log_dir}/queue.log"
    "${PYTHON}" "${common_args[@]}" --seeds "${seed}" >"${log_path}" 2>&1 &
    pids+=("$!")
  done

  local status=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done

  "${PYTHON}" "${RUNNER}" \
    --output-root "${out_root}" \
    --run-name mnist_bp_amp_v4_c1 \
    --seeds 0 1 2 \
    --summary-only >"${log_dir}/summary.log" 2>&1 || status=1

  if [ "${status}" -eq 0 ]; then
    echo "$(date) v4/c1 fine-tune ${tag} finished successfully" | tee -a "${log_dir}/queue.log"
  else
    echo "$(date) v4/c1 fine-tune ${tag} finished with errors" | tee -a "${log_dir}/queue.log"
  fi

  return "${status}"
}

status=0
launch_group 0.006 lr0p006 &
pid_lr006="$!"
launch_group 0.003 lr0p003 &
pid_lr003="$!"

wait "${pid_lr006}" || status=1
wait "${pid_lr003}" || status=1

exit "${status}"
