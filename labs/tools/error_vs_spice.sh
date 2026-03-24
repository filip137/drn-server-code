#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: error_vs_spice.sh <run_npz> [spice_npz] [output_dir] [extra args...]

Examples:
  error_vs_spice.sh /path/to/validation_states.npz
  error_vs_spice.sh run.npz spice.npz
  error_vs_spice.sh run.npz spice.npz out_dir --percentiles 50 90 99

Notes:
  - If spice_npz is omitted, the script will search upward from the run_npz
    directory for validation_states_spice_layers.npz.
  - If output_dir is omitted, it defaults to <run_npz_dir>/error_vs_spice_layers.
USAGE
}

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

strip_bracketed_paste() {
  local val="$1"
  val="${val#\[200~}"
  val="${val%\[201~}"
  printf '%s' "$val"
}

run_npz="$(strip_bracketed_paste "$1")"
shift

if [[ ! -f "$run_npz" ]]; then
  echo "Run npz not found: $run_npz" >&2
  exit 2
fi

spice_npz=""
out_dir=""

if [[ $# -gt 0 && "${1##*.}" == "npz" ]]; then
  spice_npz="$(strip_bracketed_paste "$1")"
  shift
fi

if [[ $# -gt 0 && "${1:0:1}" != "-" ]]; then
  out_dir="$(strip_bracketed_paste "$1")"
  shift
fi

if [[ -z "$spice_npz" && -n "${SPICE_NPZ:-}" ]]; then
  spice_npz="$SPICE_NPZ"
fi

if [[ -z "$spice_npz" ]]; then
  search_dir="$(cd "$(dirname "$run_npz")" && pwd)"
  for _ in {1..6}; do
    candidate="$search_dir/validation_states_spice_layers.npz"
    if [[ -f "$candidate" ]]; then
      spice_npz="$candidate"
      break
    fi
    search_dir="$(dirname "$search_dir")"
  done
fi

if [[ -z "$spice_npz" ]]; then
  echo "Could not find validation_states_spice_layers.npz. Provide spice_npz explicitly." >&2
  exit 2
fi

if [[ ! -f "$spice_npz" ]]; then
  echo "Spice npz not found: $spice_npz" >&2
  exit 2
fi

if [[ -z "$out_dir" ]]; then
  out_dir="$(dirname "$run_npz")/error_vs_spice_layers"
fi

/home/filip/server_code/labs/tools/compare_npz.sh "$run_npz" "$spice_npz" "$out_dir" "$@"
