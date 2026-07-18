#!/usr/bin/env bash
set -euo pipefail

require_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "Expected environment variable ${name} to be set." >&2
    echo "Provided value: ${!name-<unset>}" >&2
    exit 2
  fi
}

for name in \
  MNIST_CONV_REPO_ROOT \
  MNIST_CONV_PYTHON \
  MNIST_CONV_MANIFEST \
  MNIST_CONV_RESULTS_ROOT \
  MNIST_CONV_DATASET_ROOT \
  MNIST_CONV_DEVICE \
  SLURM_ARRAY_TASK_ID
do
  require_env "${name}"
done

if [[ -n "${MNIST_CONV_IDRENV_PROJECT:-}" ]]; then
  if [[ -x /gpfslocalsup/bin/idrenv ]]; then
    eval "$(/gpfslocalsup/bin/idrenv -d "${MNIST_CONV_IDRENV_PROJECT}")"
  elif command -v idrenv >/dev/null 2>&1; then
    eval "$(idrenv -d "${MNIST_CONV_IDRENV_PROJECT}")"
  fi
fi

if [[ -n "${MNIST_CONV_MODULE:-}" ]]; then
  module purge
  module load "${MNIST_CONV_MODULE}"
fi

cd "${MNIST_CONV_REPO_ROOT}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${MNIST_CONV_REPO_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

args=(
  "${MNIST_CONV_PYTHON}" -m experiments.mnist_conv sweep
  --manifest "${MNIST_CONV_MANIFEST}"
  --job-index "${SLURM_ARRAY_TASK_ID}"
  --results-root "${MNIST_CONV_RESULTS_ROOT}"
  --data-root "${MNIST_CONV_DATASET_ROOT}"
  --device "${MNIST_CONV_DEVICE}"
)
if [[ "${MNIST_CONV_RETRY_FAILED:-0}" == "1" ]]; then
  args+=(--retry-failed)
fi
if [[ "${MNIST_CONV_RECOVER_STALE:-0}" == "1" ]]; then
  args+=(--recover-stale)
fi
exec srun "${args[@]}"
