#!/usr/bin/env bash
set -euo pipefail

CURRENT_ROOT="/home/filip/server_code/results/mnist_bp_amplification_sweep"
NEXT_ROOT="/home/filip/server_code/results/mnist_bp_amplification_sweep_bias_lower_lr"
EXPECTED_RUNS=15
POLL_SECONDS=300
PYTHON="/home/filip/miniconda3/envs/py312/bin/python"
RUNNER="/home/filip/server_code/experiments/train_mnist_bp_amplification_sweep.py"
DATASET_ROOT="/home/filip/server_code/model/resistive/data"
LR_ARGS=(0.03 0.015 0.003)
MODE="${1:---wait}"

case "${MODE}" in
  --wait)
    echo "[queue] waiting for ${EXPECTED_RUNS} completed metrics under ${CURRENT_ROOT}"
    while true; do
      completed=$(find "${CURRENT_ROOT}" -name metrics.json -print 2>/dev/null | wc -l)
      echo "[queue] $(date '+%Y-%m-%d %H:%M:%S %Z') completed=${completed}/${EXPECTED_RUNS}"
      if [ "${completed}" -ge "${EXPECTED_RUNS}" ]; then
        break
      fi
      sleep "${POLL_SECONDS}"
    done
    ;;
  --now)
    echo "[queue] launching immediately without waiting for ${CURRENT_ROOT}"
    ;;
  -h|--help)
    echo "Usage: $0 [--wait|--now]"
    exit 0
    ;;
  *)
    echo "Usage: $0 [--wait|--now]" >&2
    exit 2
    ;;
esac

echo "[queue] launching bias+lower-lr sweep under ${NEXT_ROOT}"
for shard_index in $(seq 0 14); do
  if [ "${shard_index}" -lt 8 ]; then
    session="main"
  else
    session="akibscomputer"
  fi
  window=$(printf 'mnist_bias_%02d' "${shard_index}")
  if tmux list-windows -t "${session}" | grep -q "${window}"; then
    echo "[queue] ${session}:${window} already exists; skipping launch"
    continue
  fi
  tmux new-window -t "${session}" -n "${window}" -c /home/filip/server_code
  tmux send-keys -t "${session}:${window}" \
    "KMP_DISABLE_SHM=1 KMP_SHM_DISABLE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mpl-mnist-bp-bias-${shard_index} TF_CPP_MIN_LOG_LEVEL=2 ${PYTHON} ${RUNNER} --output-root ${NEXT_ROOT} --dataset-root ${DATASET_ROOT} --no-download --device cuda --num-shards 15 --shard-index ${shard_index} --learning-rate ${LR_ARGS[*]}" \
    C-m
  echo "[queue] launched ${session}:${window}"
done

echo "[queue] done"
