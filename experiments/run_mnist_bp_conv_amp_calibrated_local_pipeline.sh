#!/usr/bin/env bash
# MNIST_CONV_LEGACY_LAUNCHER_DEPRECATED
printf "%s\n" "Expected format: python -m experiments.mnist_conv {run|sweep|collect} ..." >&2
printf "%s\n" "Provided deprecated MNIST Conv launcher invocation: $0 $*" >&2
printf "%s\n" "Deprecated MNIST Conv launcher; no setup, allocation, or work was performed." >&2
exit 2
set -euo pipefail

echo "Expected complete run/calibration JSON inputs and the canonical MNIST Conv workflow." >&2
echo "Provided deprecated calibration-and-training pipeline: $0 $*" >&2
echo "This launcher is disabled because it selected targets, T/K, LR, and training defaults in shell." >&2
echo "See docs/mnist_conv_workflow.md." >&2
exit 2
