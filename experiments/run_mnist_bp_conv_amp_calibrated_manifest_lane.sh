#!/usr/bin/env bash
# MNIST_CONV_LEGACY_LAUNCHER_DEPRECATED
printf "%s\n" "Expected format: python -m experiments.mnist_conv {run|sweep|collect} ..." >&2
printf "%s\n" "Provided deprecated MNIST Conv launcher invocation: $0 $*" >&2
printf "%s\n" "Deprecated MNIST Conv launcher; no setup, allocation, or work was performed." >&2
exit 2
set -euo pipefail

echo "Expected the canonical local wrapper: experiments/run_mnist_conv_local.sh" >&2
echo "Provided deprecated CSV-manifest lane: $0 $*" >&2
echo "Build a canonical sweep JSON and plan it with: python -m experiments.mnist_conv sweep --config SWEEP.json --results-root RESULTS --plan-only" >&2
exit 2
