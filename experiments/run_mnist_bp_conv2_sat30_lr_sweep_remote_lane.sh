#!/usr/bin/env bash
set -euo pipefail

MACHINE_TAG="${MACHINE_TAG:?Expected MACHINE_TAG=akib or trex.}"
NUM_SHARDS="${NUM_SHARDS:?Expected NUM_SHARDS.}"
SHARD_INDEX="${SHARD_INDEX:?Expected SHARD_INDEX.}"

case "${MACHINE_TAG}" in
  akib)
    REPO_ROOT="${REPO_ROOT:-/home/filiposana/server_code}"
    PYTHON_BIN="${PYTHON_BIN:-/home/filiposana/miniconda3/envs/py312/bin/python}"
    ;;
  trex)
    REPO_ROOT="${REPO_ROOT:-/home/filip/server_code}"
    PYTHON_BIN="${PYTHON_BIN:-/home/filip/miniconda3/envs/py312/bin/python}"
    ;;
  *)
    echo "Expected MACHINE_TAG to be akib or trex, got ${MACHINE_TAG}" >&2
    exit 2
    ;;
esac

ROOT_NAME="mnist_bp_conv2_hardsigmoid_amp_calibrated_sat30_lr_sweep_voff4_seed0_10epoch_${MACHINE_TAG}"
MANIFEST_CSV="${MANIFEST_CSV:-${REPO_ROOT}/results/${ROOT_NAME}/targets/conv2_sat30_lr_sweep_manifest_${MACHINE_TAG}.csv}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/data}"

cd "${REPO_ROOT}"

echo "[sat30-lr-sweep] machine=${MACHINE_TAG} shard=${SHARD_INDEX}/${NUM_SHARDS}"
echo "[sat30-lr-sweep] manifest=${MANIFEST_CSV}"

MANIFEST_CSV="${MANIFEST_CSV}" \
REPO_ROOT="${REPO_ROOT}" \
PYTHON_BIN="${PYTHON_BIN}" \
DATASET_ROOT="${DATASET_ROOT}" \
DEVICE="${DEVICE:-cuda}" \
EPOCHS="${EPOCHS:-10}" \
BATCH_SIZE="${BATCH_SIZE:-4}" \
NUM_SHARDS="${NUM_SHARDS}" \
SHARD_INDEX="${SHARD_INDEX}" \
GPU_UTIL_THRESHOLD="${GPU_UTIL_THRESHOLD:-90}" \
GPU_MEM_THRESHOLD_MB="${GPU_MEM_THRESHOLD_MB:-9300}" \
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-60}" \
bash "${REPO_ROOT}/experiments/run_mnist_bp_conv_amp_calibrated_manifest_lane.sh"
