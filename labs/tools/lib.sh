#!/usr/bin/env bash

# Shell helpers retained for NPZ comparison utilities.

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
    echo "$PYTHON_BIN"
    return 0
  fi
  if [[ -x /home/filip/miniconda3/envs/py312/bin/python ]]; then
    echo /home/filip/miniconda3/envs/py312/bin/python
    return 0
  fi
  echo python3
}

_compare_npz_python_bin() {
  _labs_python_bin
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
  python_bin="$(_compare_npz_python_bin)"

  local cmd=(
    "$python_bin"
    /home/filip/server_code/labs/tools/error_compare_npz.py
    "$run_npz"
    "$spice_npz"
  )
  if [[ -n "${out_dir:-}" ]]; then
    cmd+=(--output-dir "$out_dir")
  fi
  cmd+=("$@")

  "${cmd[@]}"
}
