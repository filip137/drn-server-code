#!/usr/bin/env bash

# Shared helpers for NPZ analysis wrappers. Experiment orchestration belongs
# to the versioned `python -m ebl` CLI and `campaigns/`.

_compare_npz_usage() {
  cat <<'USAGE'
Usage: compare_npz.sh <run_npz> <spice_npz> [output_dir] [extra args...]

Example:
  compare_npz.sh run.npz spice.npz out_dir --plot --log-scale
  compare_npz.sh run.npz spice.npz --plot --log-scale
USAGE
}

_labs_python_bin() {
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    printf '%s\n' "$PYTHON_BIN"
    return 0
  fi
  printf '%s\n' python3
}

compare_npz() {
  if [[ "$#" -lt 2 ]]; then
    _compare_npz_usage
    return 2
  fi

  local run_npz="$1"
  local spice_npz="$2"
  shift 2

  local out_dir
  if [[ "$#" -gt 0 && "${1:0:1}" != "-" ]]; then
    out_dir="$1"
    shift
  fi

  local python_bin
  python_bin="$(_labs_python_bin)"
  local tools_dir
  tools_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

  local cmd=(
    "$python_bin"
    "$tools_dir/error_compare_npz.py"
    "$run_npz"
    "$spice_npz"
  )
  if [[ -n "${out_dir:-}" ]]; then
    cmd+=(--output-dir "$out_dir")
  fi
  cmd+=("$@")

  "${cmd[@]}"
}
