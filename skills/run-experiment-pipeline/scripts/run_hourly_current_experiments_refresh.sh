#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
prompt_path="${repo_root}/skills/run-experiment-pipeline/assets/hourly-current-experiments-refresh.md"
tracker_path="${repo_root}/docs/current_experiments.md"
validator_path="${repo_root}/skills/run-experiment-pipeline/scripts/validate_current_experiments.py"
python_bin="${CURRENT_EXPERIMENTS_REFRESH_PYTHON:-/home/filip/miniconda3/envs/py312/bin/python}"
codex_bin="${CURRENT_EXPERIMENTS_REFRESH_CODEX:-/home/filip/.local/bin/codex}"
interval_seconds="${CURRENT_EXPERIMENTS_REFRESH_INTERVAL_SECONDS:-3600}"
log_dir="${repo_root}/results/_tracker_refresh/hourly"
lock_path="${log_dir}/refresh.lock"

check_contract() {
  if [[ ! "${interval_seconds}" =~ ^[0-9]+$ ]] || (( interval_seconds < 300 )); then
    printf 'Expected CURRENT_EXPERIMENTS_REFRESH_INTERVAL_SECONDS >= 300; got %q\n' \
      "${interval_seconds}" >&2
    return 1
  fi
  if [[ ! -x "${python_bin}" ]]; then
    printf 'Expected an executable Python interpreter; got %q\n' \
      "${python_bin}" >&2
    return 1
  fi
  if [[ ! -x "${codex_bin}" ]]; then
    printf 'Expected an executable Codex CLI; got %q\n' "${codex_bin}" >&2
    return 1
  fi
  if [[ ! -r "${prompt_path}" ]]; then
    printf 'Expected a readable refresh prompt; got %q\n' "${prompt_path}" >&2
    return 1
  fi
  "${python_bin}" "${validator_path}" "${tracker_path}"
}

run_once() {
  mkdir -p "${log_dir}"
  local log_path="${log_dir}/refresh-$(date +%Y%m%d).log"
  (
    if ! flock -n 9; then
      printf '%s skipped: another refresh holds the lock\n' \
        "$(date --iso-8601=seconds)"
      return 0
    fi

    printf '%s refresh started\n' "$(date --iso-8601=seconds)"
    local exit_code=0
    "${codex_bin}" --ask-for-approval never exec \
      --config 'model_reasoning_effort="medium"' \
      --ephemeral \
      --color never \
      --sandbox workspace-write \
      --cd "${repo_root}" \
      - <"${prompt_path}" || exit_code=$?
    printf '%s refresh finished exit_code=%s\n' \
      "$(date --iso-8601=seconds)" "${exit_code}"
  ) 9>"${lock_path}" >>"${log_path}" 2>&1
}

case "${1:---loop}" in
  --check)
    check_contract
    ;;
  --once)
    check_contract
    run_once
    ;;
  --loop)
    check_contract
    while true; do
      run_once
      sleep "${interval_seconds}"
    done
    ;;
  *)
    printf 'Expected one of --check, --once, or --loop; got %q\n' "$1" >&2
    exit 2
    ;;
esac
