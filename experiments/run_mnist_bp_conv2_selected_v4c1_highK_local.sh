#!/usr/bin/env bash
# MNIST_CONV_LEGACY_LAUNCHER_DEPRECATED
printf "%s\n" "Expected format: python -m experiments.mnist_conv {run|sweep|collect} ..." >&2
printf "%s\n" "Provided deprecated MNIST Conv launcher invocation: $0 $*" >&2
printf "%s\n" "Deprecated MNIST Conv launcher; no setup, allocation, or work was performed." >&2
exit 2
set -euo pipefail

if [[ $# -gt 1 ]]; then
  echo "usage: $0 [perfect_diode|hard_sigmoid|both]" >&2
  exit 2
fi

MODE="${1:-both}"
case "${MODE}" in
  perfect_diode|hard_sigmoid|both) ;;
  *)
    echo "Expected mode to be perfect_diode, hard_sigmoid, or both; got ${MODE}." >&2
    exit 2
    ;;
esac

REPO_ROOT="${REPO_ROOT:-/home/filip/server_code}"
PYTHON_BIN="${PYTHON_BIN:-/home/filip/miniconda3/envs/py312/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/model/resistive/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/results/mnist_bp_conv2_selected_v4c1_highK64_seed0_50epoch}"
DEVICE="${DEVICE:-cuda}"
K_VALUE="${K_VALUE:-64}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python"
fi
if [[ ! -d "${REPO_ROOT}" ]]; then
  echo "Expected REPO_ROOT to exist, got ${REPO_ROOT}." >&2
  exit 2
fi
if [[ ! -d "${DATASET_ROOT}" ]]; then
  echo "Expected DATASET_ROOT to exist, got ${DATASET_ROOT}." >&2
  exit 2
fi

cd "${REPO_ROOT}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/labs:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "${OUTPUT_ROOT}/logs"

common_args=(
  --output-root "${OUTPUT_ROOT}"
  --dataset-root "${DATASET_ROOT}"
  --device "${DEVICE}"
  --model-key mnist_bp_conv_amp
  --seeds 0
  --epochs 50
  --batch-size 4
  --num-iterations "${K_VALUE}"
  --beta 1.0
  --lr-decay 0.99
  --conv-depth 2
  --conv-channels 64 128 256
  --kernel-size 3
  --stride 2
  --padding 0
  --output-dim 20
  --weight-gains 1.0
  --weight-min 0.0
  --weight-max 100.0
  --weight-init-mode kaiming_uniform
  --input-gain 100.0
  --learning-rate 0.012
  --run-name mnist_bp_amp_v4_c1
  --hard-sigmoid-g-on 100.0
  --hard-sigmoid-g-off 0.0
  --hard-sigmoid-v-min -1.5
  --hard-sigmoid-v-max 1.5
  --normalize-mean 0.1307
  --normalize-std 0.3081
  --normalize-scale 0.3
  --log-interval 500
  --no-download
)

run_one() {
  local nonlinearity="$1"
  local log_path="${OUTPUT_ROOT}/logs/${nonlinearity}_v4c1_K${K_VALUE}.log"
  echo "[run] nonlinearity=${nonlinearity} run=mnist_bp_amp_v4_c1 K=${K_VALUE} output=${OUTPUT_ROOT}"
  "${PYTHON_BIN}" "${REPO_ROOT}/experiments/train_mnist_bp_conv_amplification_sweep.py" \
    "${common_args[@]}" \
    --non-linearity "${nonlinearity}" \
    2>&1 | tee "${log_path}"
  echo "[done] nonlinearity=${nonlinearity} log=${log_path}"
}

if [[ "${MODE}" == "perfect_diode" || "${MODE}" == "both" ]]; then
  run_one perfect_diode
fi
if [[ "${MODE}" == "hard_sigmoid" || "${MODE}" == "both" ]]; then
  run_one hard_sigmoid
fi
