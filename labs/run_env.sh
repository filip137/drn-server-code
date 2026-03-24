#!/usr/bin/env bash
# Environment defaults for reproducible runs.
# Source this file before launching a run:
#   source /home/filip/server_code/labs/run_env.sh

# ---- Numerics-affecting ----
# Clamp for b-term in diode updates. Required for CustomExponentialSingleDiodeUpdater.
# Example: export DRN_B_CLAMP=50
: "${DRN_B_CLAMP:=1e6}"

# Path for experimental IV curve data (used when non_linearity=experimental).
# Example: export LABS_IV_CURVE_PATH=/home/filip/server_code/labs/iv_data/iv_curve.json
: "${LABS_IV_CURVE_PATH:=/home/filip/server_code/labs/i_v_npz/experimental_curve_voff_0.8_200_points.npz}"

# ---- Run-time behavior / outputs ----
# Store layer states during runs (1 to enable).
: "${DRN_STORE_STATES:=}"

# ---- Debug / safety ----
# Enable diode debug prints (set to 1). Warning: triggers pdb.set_trace() in custom_minimizer init.
: "${DRN_DEBUG_DIODE:=}"

# Break on non-finite values when DRN_DEBUG_DIODE=1 or DRN_DEBUG_BREAK_ON_ERROR=1.
: "${DRN_DEBUG_BREAK_ON_ERROR:=}"

# Threshold for state magnitude checks (default 1e10 if unset).
: "${DRN_DEBUG_STATE_MAX_ABS:=}"

# ---- Timing / verbosity ----
# Print timing for single-diode updater (1 to enable).
: "${LABS_PRINT_SINGLE_DIODE_TIMING:=}"
# Print timing for double-diode updater (1 to enable).
: "${LABS_PRINT_DOUBLE_DIODE_TIMING:=}"
: "${LABS_PRINT_DOUBLE_DIODE_TIMING_EVERY:=}"

# Newton timing/iters for experimental IV curve updater (1 to enable).
: "${LABS_PRINT_NEWTON_ITERS:=}"
: "${LABS_PRINT_NEWTON_ITERS_EVERY:=}"
: "${LABS_PRINT_NEWTON_TIMING:=}"
: "${LABS_PRINT_NEWTON_TIMING_EVERY:=}"

# Export only if set to a non-empty value.
_export_if_set() {
  local name="$1"
  local value="${!name}"
  if [[ -n "$value" ]]; then
    export "$name=$value"
  fi
}

for var in \
  DRN_B_CLAMP \
  LABS_IV_CURVE_PATH \
  DRN_STORE_STATES \
  DRN_DEBUG_DIODE \
  DRN_DEBUG_BREAK_ON_ERROR \
  DRN_DEBUG_STATE_MAX_ABS \
  LABS_PRINT_SINGLE_DIODE_TIMING \
  LABS_PRINT_DOUBLE_DIODE_TIMING \
  LABS_PRINT_DOUBLE_DIODE_TIMING_EVERY \
  LABS_PRINT_NEWTON_ITERS \
  LABS_PRINT_NEWTON_ITERS_EVERY \
  LABS_PRINT_NEWTON_TIMING \
  LABS_PRINT_NEWTON_TIMING_EVERY
  do
  _export_if_set "$var"
done

unset -f _export_if_set
