#!/usr/bin/env bash

# Shell helpers for labs/tools wrappers.

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

resolve_moons_run_dir() {
  if [[ "$#" -ne 1 ]]; then
    echo "Usage: resolve_moons_run_dir <run_dir>" >&2
    return 2
  fi
  local run_dir="$1"
  if [[ ! -d "$run_dir" ]]; then
    echo "Run directory not found: $run_dir" >&2
    return 2
  fi

  local model_matches
  mapfile -t model_matches < <(find "$run_dir" -maxdepth 1 -type f -name "model.pt")
  if [[ "${#model_matches[@]}" -ne 1 ]]; then
    if [[ "${#model_matches[@]}" -eq 0 ]]; then
      echo "No model.pt found in $run_dir" >&2
    else
      echo "Multiple model.pt files found in $run_dir" >&2
    fi
    return 2
  fi

  local json_matches
  mapfile -t json_matches < <(find "$run_dir" -maxdepth 1 -type f -name "*.json")
  if [[ "${#json_matches[@]}" -ne 1 ]]; then
    if [[ "${#json_matches[@]}" -eq 0 ]]; then
      echo "No .json config found in $run_dir" >&2
    else
      echo "Multiple .json configs found in $run_dir" >&2
    fi
    return 2
  fi

  printf '%s|%s\n' "${model_matches[0]}" "${json_matches[0]}"
}

run_moons_linspace_dir() {
  if [[ "$#" -lt 1 ]]; then
    echo "Usage: run_moons_linspace_dir <run_dir> [extra small_network.py args...]" >&2
    return 2
  fi
  local run_dir="$1"
  shift

  local resolved
  resolved="$(resolve_moons_run_dir "$run_dir")" || return 2

  local model_path config_path
  model_path="${resolved%%|*}"
  config_path="${resolved##*|}"

  local python_bin
  python_bin="$(_labs_python_bin)"

  local add_output_dir="true"
  for arg in "$@"; do
    if [[ "$arg" == "--output-dir" || "$arg" == --output-dir=* ]]; then
      add_output_dir="false"
      break
    fi
  done

  local extra_args=("$@")
  if [[ "$add_output_dir" == "true" ]]; then
    extra_args+=(--output-dir "$run_dir")
  fi

  "$python_bin" /home/filip/server_code/labs/small_network.py \
    --mode linspace \
    --config "$config_path" \
    --weights "$model_path" \
    "${extra_args[@]}"
}

run_moons_linspace_dirs() {
  if [[ "$#" -lt 1 ]]; then
    echo "Usage: run_moons_linspace_dirs <dir1> <dir2> ... -- [extra args...]" >&2
    return 2
  fi

  local dirs=()
  local extra_args=()
  local seen_sep="false"

  for arg in "$@"; do
    if [[ "$arg" == "--" ]]; then
      seen_sep="true"
      continue
    fi
    if [[ "$seen_sep" == "true" ]]; then
      extra_args+=("$arg")
    else
      dirs+=("$arg")
    fi
  done

  if [[ "${#dirs[@]}" -eq 0 ]]; then
    echo "No run directories provided." >&2
    return 2
  fi

  for dir in "${dirs[@]}"; do
    run_moons_linspace_dir "$dir" "${extra_args[@]}"
  done
}

_digits_train_validate_usage() {
  cat <<'USAGE'
Usage: run_digits_train_validate.sh <config.json> [output_dir] [-- extra args...]

Trains with the given config, writes outputs under output_dir (default: config directory),
then runs digits_validate using the latest model.pt for that hidden depth.

Examples:
  run_digits_train_validate.sh /path/to/config.json
  run_digits_train_validate.sh /path/to/config.json /path/to/output -- --num-epochs 50
USAGE
}

_digits_train_usage() {
  cat <<'USAGE'
Usage: run_digits_train.sh <config.json> [output_dir] [-- extra args...]

Trains with the given config, writing outputs under output_dir (default: config directory).

Examples:
  run_digits_train.sh /path/to/config.json
  run_digits_train.sh /path/to/config.json /path/to/output -- --num-epochs 50
USAGE
}

_digits_validate_usage() {
  cat <<'USAGE'
Usage: run_digits_validate.sh <config.json> [weights.pt] [output_dir] [-- extra args...]

Runs digits_validate for the given config. weights.pt is required.

Examples:
  run_digits_validate.sh /path/to/config.json /path/to/model.pt
  run_digits_validate.sh /path/to/config.json /path/to/model.pt /path/to/output
USAGE
}

_digits_latest_model_for_config() {
  if [[ "$#" -ne 2 ]]; then
    echo "Usage: _digits_latest_model_for_config <config.json> <output_dir>" >&2
    return 2
  fi
  local config_path="$1"
  local output_dir="$2"

  local python_bin
  python_bin="$(_labs_python_bin)"

  "$python_bin" - "$config_path" "$output_dir" <<'PY'
import json
import sys
from pathlib import Path

config_path = Path(sys.argv[1]).expanduser()
output_dir = Path(sys.argv[2]).expanduser()

cfg = json.loads(config_path.read_text())
dims = cfg.get("dims")
if not isinstance(dims, list) or len(dims) < 2:
    raise SystemExit("Config missing valid 'dims' list.")

hidden_count = max(len(dims) - 2, 0)
root = output_dir / f"hidden_{hidden_count}"
candidates = sorted(root.glob("*/model.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
if not candidates:
    raise SystemExit(f"No model.pt found under {root}")
print(candidates[0])
PY
}

run_digits_train_validate() {
  if [[ "$#" -lt 1 ]]; then
    _digits_train_validate_usage
    return 2
  fi

  local config_path="$1"
  shift

  local output_dir
  if [[ "$#" -gt 0 && "${1:0:1}" != "-" ]]; then
    output_dir="$1"
    shift
  else
    output_dir="$(dirname "$config_path")"
  fi

  local extra_args=()
  if [[ "$#" -gt 0 && "$1" == "--" ]]; then
    shift
  fi
  extra_args=("$@")

  local python_bin
  python_bin="$(_labs_python_bin)"

  "$python_bin" /home/filip/server_code/labs/small_network.py \
    --mode train \
    --config "$config_path" \
    --dataset digits \
    --output-dir "$output_dir" \
    "${extra_args[@]}"

  local model_path
  model_path="$(_digits_latest_model_for_config "$config_path" "$output_dir")"

  "$python_bin" /home/filip/server_code/labs/small_network.py \
    --mode digits_validate \
    --config "$config_path" \
    --weights "$model_path" \
    --dataset digits \
    --output-dir "$output_dir" \
    "${extra_args[@]}"
}

run_digits_train() {
  if [[ "$#" -lt 1 ]]; then
    _digits_train_usage
    return 2
  fi

  local config_path="$1"
  shift

  local output_dir
  if [[ "$#" -gt 0 && "${1:0:1}" != "-" ]]; then
    output_dir="$1"
    shift
  else
    output_dir="$(dirname "$config_path")"
  fi

  local extra_args=()
  if [[ "$#" -gt 0 && "$1" == "--" ]]; then
    shift
  fi
  extra_args=("$@")

  local python_bin
  python_bin="$(_labs_python_bin)"

  "$python_bin" /home/filip/server_code/labs/small_network.py \
    --mode train \
    --config "$config_path" \
    --dataset digits \
    --output-dir "$output_dir" \
    "${extra_args[@]}"
}

run_digits_validate() {
  if [[ "$#" -lt 2 ]]; then
    _digits_validate_usage
    return 2
  fi

  local config_path="$1"
  shift

  local weights_path=""
  local output_dir=""

  weights_path="$1"
  shift

  if [[ -z "$output_dir" ]]; then
    if [[ "$#" -gt 0 && "${1:0:1}" != "-" ]]; then
      output_dir="$1"
      shift
    else
      output_dir="$(dirname "$config_path")"
    fi
  fi

  local extra_args=()
  if [[ "$#" -gt 0 && "$1" == "--" ]]; then
    shift
  fi
  extra_args=("$@")

  local python_bin
  python_bin="$(_labs_python_bin)"

  "$python_bin" /home/filip/server_code/labs/small_network.py \
    --mode digits_validate \
    --config "$config_path" \
    --weights "$weights_path" \
    --dataset digits \
    --output-dir "$output_dir" \
    "${extra_args[@]}"
}
