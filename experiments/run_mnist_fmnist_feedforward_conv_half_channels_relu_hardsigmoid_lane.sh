#!/usr/bin/env bash
set -euo pipefail

lane_index="${1:-0}"
num_lanes="${2:-1}"

if [ "${num_lanes}" -lt 1 ]; then
  echo "Expected num_lanes >= 1, got ${num_lanes}." >&2
  exit 2
fi
if [ "${lane_index}" -lt 0 ] || [ "${lane_index}" -ge "${num_lanes}" ]; then
  echo "Expected lane_index in [0, num_lanes), got ${lane_index} for num_lanes=${num_lanes}." >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${PYTHON:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/results/mnist_fmnist_feedforward_conv_half_channels_relu_hardsigmoid_20260701}"
MNIST_DATASET_ROOT="${MNIST_DATASET_ROOT:-${REPO_ROOT}/data}"
FMNIST_DATASET_ROOT="${FMNIST_DATASET_ROOT:-${REPO_ROOT}/data}"
CIFAR_DATASET_ROOT="${CIFAR_DATASET_ROOT:-${REPO_ROOT}/data}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-128}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-512}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
LOSS="${LOSS:-cross_entropy}"
LOG_INTERVAL="${LOG_INTERVAL:-100}"
KERNEL_SIZE="${KERNEL_SIZE:-3}"
PADDINGS="${PADDINGS:-1 1 1}"
STRIDES="${STRIDES:-2 2 1}"
CONV_CHANNELS="${CONV_CHANNELS:-32 64 128}"
CONV_DEPTHS="${CONV_DEPTHS:-1 2 3}"
ROTATION_DEGREES="${ROTATION_DEGREES:-45.0}"
ROTATION_SEED="${ROTATION_SEED:-1729}"
TRANSLATION_FRACTION="${TRANSLATION_FRACTION:-0.0}"
SCALE_MIN="${SCALE_MIN:-1.0}"
SCALE_MAX="${SCALE_MAX:-1.0}"
SHEAR_DEGREES="${SHEAR_DEGREES:-0.0}"
SEEDS="${SEEDS:-0 1 2}"
ACTIVATIONS="${ACTIVATIONS:-relu hard-sigmoid}"
DATASETS="${DATASETS:-mnist fashion_mnist}"
INPUT_PREPROCESSING="${INPUT_PREPROCESSING:-identity}"
NO_DOWNLOAD="${NO_DOWNLOAD:-1}"
RUN_GROUP="${RUN_GROUP:-final_halfch_s221_pad1}"

mkdir -p "${OUTPUT_ROOT}" "${REPO_ROOT}/logs"

dataset_root_for() {
  case "$1" in
    mnist|mnist_rotated|mnist_affine) printf '%s\n' "${MNIST_DATASET_ROOT}" ;;
    fashion_mnist) printf '%s\n' "${FMNIST_DATASET_ROOT}" ;;
    cifar10_gray|cifar10_rgb) printf '%s\n' "${CIFAR_DATASET_ROOT}" ;;
    *) echo "Expected dataset 'mnist', 'mnist_rotated', 'mnist_affine', 'fashion_mnist', 'cifar10_gray', or 'cifar10_rgb', got $1." >&2; return 2 ;;
  esac
}

run_id=0
for dataset in ${DATASETS}; do
  for activation in ${ACTIVATIONS}; do
    for conv_depth in ${CONV_DEPTHS}; do
      for seed in ${SEEDS}; do
        if [ $((run_id % num_lanes)) -ne "${lane_index}" ]; then
          run_id=$((run_id + 1))
          continue
        fi

        dataset_root="$(dataset_root_for "${dataset}")"
        echo "[run] lane=${lane_index}/${num_lanes} dataset=${dataset} activation=${activation} conv${conv_depth} seed=${seed}"
        cmd=(
          "${PYTHON}" "${REPO_ROOT}/experiments/train_mnist_feedforward_conv.py"
          --output-root "${OUTPUT_ROOT}"
          --dataset-root "${dataset_root}"
          --dataset "${dataset}"
          --device "${DEVICE}"
          --run-group "${RUN_GROUP}"
          --conv-depth "${conv_depth}"
          --conv-channels ${CONV_CHANNELS}
          --activation "${activation}"
          --input-preprocessing "${INPUT_PREPROCESSING}"
          --kernel-size "${KERNEL_SIZE}"
          --strides ${STRIDES}
          --paddings ${PADDINGS}
          --seed "${seed}"
          --epochs "${EPOCHS}"
          --batch-size "${BATCH_SIZE}"
          --test-batch-size "${TEST_BATCH_SIZE}"
          --learning-rate "${LEARNING_RATE}"
          --loss "${LOSS}"
          --log-interval "${LOG_INTERVAL}"
          --rotation-degrees "${ROTATION_DEGREES}"
          --rotation-seed "${ROTATION_SEED}"
          --translation-fraction "${TRANSLATION_FRACTION}"
          --scale-min "${SCALE_MIN}"
          --scale-max "${SCALE_MAX}"
          --shear-degrees "${SHEAR_DEGREES}"
          --skip-complete
        )
        if [ "${NO_DOWNLOAD}" = "1" ]; then
          cmd+=(--no-download)
        fi
        "${cmd[@]}"
        run_id=$((run_id + 1))
      done
    done
  done
done

echo "[done] lane=${lane_index}/${num_lanes} total_specs=${run_id} output_root=${OUTPUT_ROOT}"
