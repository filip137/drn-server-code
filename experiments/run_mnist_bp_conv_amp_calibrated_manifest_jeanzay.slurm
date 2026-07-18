#!/usr/bin/env bash
#SBATCH --mnist-conv-legacy-launcher-is-deprecated
# MNIST_CONV_LEGACY_LAUNCHER_DEPRECATED
printf "%s\n" "Expected format: python -m experiments.mnist_conv {run|sweep|collect} ..." >&2
printf "%s\n" "Provided deprecated MNIST Conv launcher invocation: $0 $*" >&2
printf "%s\n" "Deprecated MNIST Conv launcher; no setup, allocation, or work was performed." >&2
exit 2
set -euo pipefail

echo "Expected the canonical command: python -m experiments.mnist_conv sweep --config SWEEP.json --results-root RESULTS --executor slurm --profile PROFILE.json" >&2
echo "Provided deprecated fixed-size CSV-manifest Slurm launcher: $0 $*" >&2
echo "Publish a canonical manifest with plan-only before submitting it." >&2
exit 2
