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
  MNIST_CONV_MANIFEST
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

# Do not allow partial collection here: this afterany job must report any failed,
# pruned, missing, or corrupt worker as a non-zero Slurm job.
exec srun "${MNIST_CONV_PYTHON}" -m experiments.mnist_conv collect \
  --sweep "$(dirname "${MNIST_CONV_MANIFEST}")"
