#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 0 ]]; then
  echo "Expected canonical sweep infrastructure arguments." >&2
  echo "Provided no arguments." >&2
  echo "Usage: $0 --config SWEEP.json --results-root ROOT --data-root MNIST_ROOT --workers N [other infrastructure options]" >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-python}"

exec "${PYTHON_BIN}" -m experiments.mnist_conv sweep --executor local "$@"
