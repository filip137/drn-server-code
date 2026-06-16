#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-/home/filip/miniconda3/envs/py312/bin/python}"
EVAL="/home/filip/server_code/experiments/evaluate_mnist_bp_write_noise_sweep.py"
PLOT="/home/filip/server_code/experiments/plot_mnist_bp_write_noise_sigma_response.py"

INPUT_ROOT="/home/filip/server_code/results/mnist_bp_amplification_sweep_hardsigmoid_voff15_legacy"
RESULTS_BASE="/home/filip/server_code/results"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-512}"
NUM_SHARDS="${NUM_SHARDS:-15}"
ITERATIONS=(${ITERATIONS:-8 16})

export KMP_DISABLE_SHM=1
export KMP_SHM_DISABLE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-mnist-write-noise-hardsigmoid-voff15-iters8-16}"

sigmas=(0 0.025 0.05 0.075 0.1 0.125 0.15 0.175 0.2 0.25 0.3 0.4 0.5 0.75 1.0)
noise_seeds=()
for seed in $(seq 0 29); do
  noise_seeds+=("${seed}")
done

pids=()
status=0
output_roots=()

for iter_count in "${ITERATIONS[@]}"; do
  output_root="${RESULTS_BASE}/mnist_bp_write_noise_sweep_hardsigmoid_voff15_sigma_response_iter${iter_count}_30x"
  log_dir="${output_root}/logs"
  mkdir -p "${log_dir}"
  output_roots+=("${output_root}")

  echo "$(date) launching hard-sigmoid sigma-response sweep with inference iterations=${iter_count}, batch=${EVAL_BATCH_SIZE}" | tee -a "${log_dir}/queue.log"

  common_args=(
    "${EVAL}"
    --input-root "${INPUT_ROOT}"
    --output-root "${output_root}"
    --device cuda
    --checkpoint-kind best
    --sigmas "${sigmas[@]}"
    --noise-seeds "${noise_seeds[@]}"
    --eval-batch-size "${EVAL_BATCH_SIZE}"
    --inference-iterations "${iter_count}"
    --no-download
    --num-shards "${NUM_SHARDS}"
  )

  for shard in $(seq 0 $((NUM_SHARDS - 1))); do
    log_path="${log_dir}/write_noise_sigma_response_iter${iter_count}_shard_${shard}.log"
    echo "$(date) iter=${iter_count} shard ${shard}/${NUM_SHARDS} -> ${log_path}" | tee -a "${log_dir}/queue.log"
    "${PYTHON}" "${common_args[@]}" \
      --shard-index "${shard}" \
      --raw-results-name "raw_results_shard_${shard}.csv" \
      >"${log_path}" 2>&1 &
    pids+=("$!")
  done
done

for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

for output_root in "${output_roots[@]}"; do
  log_dir="${output_root}/logs"
  "${PYTHON}" "${EVAL}" \
    --output-root "${output_root}" \
    --summary-only >"${log_dir}/summary.log" 2>&1 || status=1

  "${PYTHON}" "${PLOT}" \
    --input-root "${output_root}" >"${log_dir}/plot_sigma_response.log" 2>&1 || status=1
done

if [ "${status}" -eq 0 ]; then
  echo "$(date) hard-sigmoid iter8/iter16 write-noise sweeps finished successfully"
else
  echo "$(date) hard-sigmoid iter8/iter16 write-noise sweeps finished with errors"
fi

exit "${status}"
