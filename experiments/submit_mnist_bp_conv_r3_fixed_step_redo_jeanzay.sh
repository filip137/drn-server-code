#!/bin/bash
# MNIST_CONV_LEGACY_LAUNCHER_DEPRECATED
printf "%s\n" "Expected format: python -m experiments.mnist_conv {run|sweep|collect} ..." >&2
printf "%s\n" "Provided deprecated MNIST Conv launcher invocation: $0 $*" >&2
printf "%s\n" "Deprecated MNIST Conv launcher; no setup, allocation, or work was performed." >&2
exit 2
# Submit the Jean Zay R3 fixed-step Conv2/Conv3 redo dependency chain.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_RESULT_ROOT="${SCRATCH_fmu:-/lustre/fsn1/projects/rech/fmu/${USER}}/server_code/results/mnist_bp_conv_fixed_step_redo_r3_20260703"
RESULT_ROOT="${RESULT_ROOT:-${DEFAULT_RESULT_ROOT}}"
BASE_EXPORT="ALL,RESULT_ROOT=${RESULT_ROOT}"
STOP_AFTER_STAGE="${STOP_AFTER_STAGE:-}"

submit() {
  local dep="$1"
  local script="$2"
  local extra_export="${3:-}"
  shift 2
  local args=(--export="${BASE_EXPORT}${extra_export:+,${extra_export}}")
  if [ -n "${dep}" ]; then
    args+=(--dependency="afterok:${dep}")
  fi
  sbatch --parsable "${args[@]}" "${SCRIPT_DIR}/${script}"
}

maybe_stop() {
  local stage="$1"
  if [ "${STOP_AFTER_STAGE}" = "${stage}" ]; then
    echo "[submitted] stopped_after=${stage}"
    echo "[submitted] result_root=${RESULT_ROOT}"
    exit 0
  fi
}

preflight="$(submit "" run_mnist_bp_conv_r3_fixed_step_preflight_jeanzay.slurm)"
maybe_stop preflight
tk_diag="$(submit "${preflight}" run_mnist_bp_conv_r3_fixed_step_tk_diagnostics_jeanzay.slurm)"
maybe_stop tk_diag
select_tk="$(submit "${tk_diag}" run_mnist_bp_conv_r3_fixed_step_select_tk_jeanzay.slurm)"
maybe_stop select_tk
calibrate="$(submit "${select_tk}" run_mnist_bp_conv_r3_fixed_step_hs_calibrate_sat30_jeanzay.slurm)"
maybe_stop calibrate
lr_screen="$(submit "${calibrate}" run_mnist_bp_conv_r3_fixed_step_lr_screen_jeanzay.slurm)"
maybe_stop lr_screen
collect_lr="$(submit "${lr_screen}" run_mnist_bp_conv_r3_fixed_step_collect_lr_jeanzay.slurm)"
maybe_stop collect_lr
long_check="$(submit "${collect_lr}" run_mnist_bp_conv_r3_fixed_step_train_selected_jeanzay.slurm PHASE=long)"
maybe_stop long_check
collect_long="$(submit "${long_check}" run_mnist_bp_conv_r3_fixed_step_collect_lr_jeanzay.slurm "LR_ROOT=${RESULT_ROOT}/long_check_seed0_50epoch,SELECTED_SETTINGS_CSV=${RESULT_ROOT}/long_check_seed0_50epoch/selected_final_settings.csv")"
maybe_stop collect_long
gate_long="$(submit "${collect_long}" run_mnist_bp_conv_r3_fixed_step_gate_long_check_jeanzay.slurm)"
maybe_stop gate_long
final="$(submit "${gate_long}" run_mnist_bp_conv_r3_fixed_step_train_selected_jeanzay.slurm PHASE=final)"
maybe_stop final
post_diag="$(submit "${final}" run_mnist_bp_conv_r3_fixed_step_post_diagnostics_jeanzay.slurm)"
maybe_stop post_diag

cat <<EOF
[submitted] preflight=${preflight}
[submitted] tk_diag=${tk_diag}
[submitted] select_tk=${select_tk}
[submitted] calibrate=${calibrate}
[submitted] lr_screen=${lr_screen}
[submitted] collect_lr=${collect_lr}
[submitted] long_check=${long_check}
[submitted] collect_long=${collect_long}
[submitted] gate_long=${gate_long}
[submitted] final=${final}
[submitted] post_diag=${post_diag}
EOF
