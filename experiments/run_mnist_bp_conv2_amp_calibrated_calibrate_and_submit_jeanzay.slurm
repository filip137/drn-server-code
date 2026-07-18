#!/usr/bin/env bash
#SBATCH --mnist-conv-legacy-launcher-is-deprecated
# MNIST_CONV_LEGACY_LAUNCHER_DEPRECATED
printf "%s\n" "Expected format: python -m experiments.mnist_conv {run|sweep|collect} ..." >&2
printf "%s\n" "Provided deprecated MNIST Conv launcher invocation: $0 $*" >&2
printf "%s\n" "Deprecated MNIST Conv launcher; no setup, allocation, or work was performed." >&2
exit 2
set -euo pipefail

echo "Expected separate canonical calibration records, sweep planning, and generic Slurm submission." >&2
echo "Provided deprecated Conv2 calibration-and-training launcher: $0 $*" >&2
echo "This launcher is disabled because operational T/K and training settings are not frozen." >&2
echo "See docs/mnist_conv_workflow.md." >&2
exit 2
