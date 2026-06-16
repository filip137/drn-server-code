#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-/home/filip/miniconda3/envs/py312/bin/python}"
EVAL="/home/filip/server_code/experiments/evaluate_mnist_bp_output_decomposition_physical_logg_noise.py"

INPUT_ROOT="/home/filip/server_code/results/mnist_bp_amplification_sweep_hardsigmoid_voff15_legacy"
SWRITE_ROOT="/home/filip/server_code/results/mnist_bp_physical_cost_sharpness_hardsigmoid_voff15_iter16"
OUTPUT_ROOT="/home/filip/server_code/results/mnist_bp_output_decomp_physical_logg_noise_hardsigmoid_iter16"
LOG_DIR="${OUTPUT_ROOT}/logs"

EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1024}"
NUM_DIRECTIONS="${NUM_DIRECTIONS:-16}"
EPSILON="${EPSILON:-0.02}"
NUM_SHARDS="${NUM_SHARDS:-15}"
MAX_EVAL_BATCHES="${MAX_EVAL_BATCHES:-}"
NOISE_SEED_COUNT="${NOISE_SEED_COUNT:-30}"

mkdir -p "${LOG_DIR}"

export KMP_DISABLE_SHM=1
export KMP_SHM_DISABLE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-mnist-output-decomp-physical-logg}"

sigmas=(0 0.025 0.05 0.075 0.1 0.125 0.15 0.175 0.2 0.25 0.3 0.4 0.5 0.75 1.0)
noise_seeds=()
for seed in $(seq 0 $((NOISE_SEED_COUNT - 1))); do
  noise_seeds+=("${seed}")
done

echo "$(date) launching output-decomposition + physical-logG noise diagnostic" | tee -a "${LOG_DIR}/queue.log"
echo "$(date) batch=${EVAL_BATCH_SIZE} directions=${NUM_DIRECTIONS} epsilon=${EPSILON} noise_seeds=${NOISE_SEED_COUNT}" | tee -a "${LOG_DIR}/queue.log"

common_args=(
  "${EVAL}"
  --input-root "${INPUT_ROOT}"
  --output-root "${OUTPUT_ROOT}"
  --swrite-root "${SWRITE_ROOT}"
  --device cuda
  --checkpoint-kind best
  --eval-batch-size "${EVAL_BATCH_SIZE}"
  --inference-iterations 16
  --num-directions "${NUM_DIRECTIONS}"
  --epsilons "${EPSILON}"
  --sigmas "${sigmas[@]}"
  --noise-seeds "${noise_seeds[@]}"
  --no-download
  --num-shards "${NUM_SHARDS}"
)

if [ -n "${MAX_EVAL_BATCHES}" ]; then
  common_args+=(--max-eval-batches "${MAX_EVAL_BATCHES}")
fi

pids=()
status=0
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  log_path="${LOG_DIR}/output_decomp_physical_logg_shard_${shard}.log"
  echo "$(date) shard ${shard}/${NUM_SHARDS} -> ${log_path}" | tee -a "${LOG_DIR}/queue.log"
  "${PYTHON}" "${common_args[@]}" \
    --shard-index "${shard}" \
    --raw-decomposition-name "raw_decomposition_shard_${shard}.csv" \
    --raw-noise-name "raw_physical_logg_noise_shard_${shard}.csv" \
    >"${log_path}" 2>&1 &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

"${PYTHON}" "${EVAL}" \
  --output-root "${OUTPUT_ROOT}" \
  --swrite-root "${SWRITE_ROOT}" \
  --summary-only >"${LOG_DIR}/summary.log" 2>&1 || status=1

if [ "${status}" -eq 0 ]; then
  echo "$(date) output-decomposition + physical-logG noise diagnostic finished successfully" | tee -a "${LOG_DIR}/queue.log"
else
  echo "$(date) output-decomposition + physical-logG noise diagnostic finished with errors" | tee -a "${LOG_DIR}/queue.log"
fi

exit "${status}"
