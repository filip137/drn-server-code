#!/usr/bin/env bash
set -euo pipefail

cd "${MNIST_CONV_REPO_ROOT:?}"
if [[ -n "${MNIST_CONV_MODULE:-}" ]]; then
  module purge
  module load "${MNIST_CONV_MODULE}"
fi
if [[ -n "${MNIST_CONV_IDRENV_PROJECT:-}" ]] && command -v idrenv >/dev/null 2>&1; then
  eval "$(idrenv -d "${MNIST_CONV_IDRENV_PROJECT}")"
fi

exec "${MNIST_CONV_PYTHON:?}" -m experiments.mnist_conv lr-study \
  --stage "${MNIST_CONV_LR_STAGE:?}" \
  --study "${MNIST_CONV_LR_STUDY:?}" \
  --manifest "${MNIST_CONV_LR_MANIFEST:?}" \
  --finalize-stage \
  --data-root "${MNIST_CONV_DATASET_ROOT:?}" \
  --device "${MNIST_CONV_DEVICE:?}"
