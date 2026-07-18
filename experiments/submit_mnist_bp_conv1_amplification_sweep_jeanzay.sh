#!/usr/bin/env bash
# MNIST_CONV_LEGACY_LAUNCHER_DEPRECATED
printf "%s\n" "Expected format: python -m experiments.mnist_conv {run|sweep|collect} ..." >&2
printf "%s\n" "Provided deprecated MNIST Conv launcher invocation: $0 $*" >&2
printf "%s\n" "Deprecated MNIST Conv launcher; no setup, allocation, or work was performed." >&2
exit 2
set -euo pipefail

if command -v idrenv >/dev/null 2>&1; then
  eval "$(idrenv)"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPT="${SCRIPT_DIR}/run_mnist_bp_conv1_amplification_sweep_jeanzay.slurm"
MODE="${1:-smoke}"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "Expected to run this submit helper on a Jean Zay login node with sbatch available." >&2
  exit 2
fi

resolve_account() {
  if [ -n "${JZ_ACCOUNT:-}" ]; then
    printf '%s\n' "${JZ_ACCOUNT}"
    return 0
  fi
  if ! command -v sacctmgr >/dev/null 2>&1; then
    echo "Set JZ_ACCOUNT; sacctmgr is unavailable for account auto-detection." >&2
    return 2
  fi
  mapfile -t accounts < <(
    sacctmgr -nP show assoc where user="${USER}" format=Account,Partition |
      awk -F'|' '$1 != "" && $2 ~ /^gpu/ {print $1}' |
      sort -u
  )
  if [ "${#accounts[@]}" -eq 1 ]; then
    printf '%s\n' "${accounts[0]}"
    return 0
  fi
  if [ "${#accounts[@]}" -eq 0 ]; then
    echo "Set JZ_ACCOUNT; no GPU account was auto-detected for user ${USER}." >&2
  else
    echo "Set JZ_ACCOUNT; multiple GPU accounts were found:" >&2
    printf '  %s\n' "${accounts[@]}" >&2
  fi
  return 2
}

ACCOUNT="$(resolve_account)"
mkdir -p slurm/mnist_conv1_amp

case "${MODE}" in
  smoke)
    echo "[submit] smoke array tasks 0 and 15 with account ${ACCOUNT}"
    sbatch \
      --account="${ACCOUNT}" \
      --array=0,15%2 \
      --export=ALL,SMOKE=1 \
      "${SLURM_SCRIPT}"
    ;;
  full)
    echo "[submit] full 30-job array with account ${ACCOUNT}"
    sbatch \
      --account="${ACCOUNT}" \
      --array=0-29%30 \
      --export=ALL,SMOKE=0 \
      "${SLURM_SCRIPT}"
    ;;
  summary)
    echo "[submit] summary collector with account ${ACCOUNT}"
    sbatch \
      --account="${ACCOUNT}" \
      --array=0 \
      --export=ALL,SUMMARY_ONLY=1 \
      "${SLURM_SCRIPT}"
    ;;
  *)
    echo "Usage: $0 [smoke|full|summary]" >&2
    exit 2
    ;;
esac
