#!/bin/bash -l
# Canonical bounded source/bootstrap validation for the Jean Zay successor.

set -euo pipefail

# Enforce the advertised deadline around the whole login-shell/bootstrap
# sequence, including idrenv, module initialization, verifier staging, and
# the individually bounded validation commands below.
if [[ "${PD_SUCCESSOR_STAGED_VALIDATION_INNER:-0}" != "1" ]]; then
  outer_timeout_seconds=900
  outer_arguments=("$@")
  for (( argument_index = 0; argument_index < ${#outer_arguments[@]}; argument_index++ )); do
    if [[ "${outer_arguments[argument_index]}" == "--timeout-seconds" ]]; then
      next_index=$(( argument_index + 1 ))
      if (( next_index >= ${#outer_arguments[@]} )); then
        printf 'Expected --timeout-seconds to have a value.\n' >&2
        printf 'Provided value: <missing>\n' >&2
        exit 2
      fi
      outer_timeout_seconds="${outer_arguments[next_index]}"
    fi
  done
  if [[ ! "${outer_timeout_seconds}" =~ ^[1-9][0-9]*$ ]]; then
    printf 'Expected --timeout-seconds to be a positive integer.\n' >&2
    printf 'Provided value: %s\n' "${outer_timeout_seconds}" >&2
    exit 2
  fi
  if (( outer_timeout_seconds > 3600 )); then
    printf 'Expected --timeout-seconds not to exceed 3600.\n' >&2
    printf 'Provided value: %s\n' "${outer_timeout_seconds}" >&2
    exit 2
  fi
  if ! command -v timeout >/dev/null 2>&1; then
    printf 'Expected timeout to enforce the staged-validation deadline.\n' >&2
    printf 'Provided value: not found\n' >&2
    exit 2
  fi
  set +e
  PD_SUCCESSOR_STAGED_VALIDATION_INNER=1 \
    timeout --kill-after=30s "${outer_timeout_seconds}s" \
    bash "$0" "${outer_arguments[@]}"
  outer_status=$?
  set -e
  if (( outer_status == 124 || outer_status == 137 )); then
    printf 'Expected staged validation to finish within %s seconds.\n' \
      "${outer_timeout_seconds}" >&2
    printf 'Provided value: whole-script hard deadline reached\n' >&2
  fi
  exit "${outer_status}"
fi
unset PD_SUCCESSOR_STAGED_VALIDATION_INNER

experiment_id="perfectdiode-conv12-best-observed-confirmation-20260727-v1"
hard_timeout_seconds=900
repo_root=""
approved_plan=""
environment_contract=""
started_seconds="${SECONDS}"

fail_value() {
  local expected="$1"
  local provided="$2"
  printf 'Expected %s.\n' "${expected}" >&2
  printf 'Provided value: %s\n' "${provided}" >&2
  exit 2
}

while (( $# > 0 )); do
  case "$1" in
    --repo-root)
      (( $# >= 2 )) || fail_value "--repo-root to have a value" "<missing>"
      repo_root="$2"
      shift 2
      ;;
    --approved-plan)
      (( $# >= 2 )) || fail_value "--approved-plan to have a value" "<missing>"
      approved_plan="$2"
      shift 2
      ;;
    --environment-contract)
      (( $# >= 2 )) || fail_value \
        "--environment-contract to have a value" "<missing>"
      environment_contract="$2"
      shift 2
      ;;
    --timeout-seconds)
      (( $# >= 2 )) || fail_value "--timeout-seconds to have a value" "<missing>"
      hard_timeout_seconds="$2"
      shift 2
      ;;
    *)
      fail_value \
        "arguments to be --repo-root, --approved-plan, --environment-contract, or --timeout-seconds" \
        "$1"
      ;;
  esac
done

[[ -n "${repo_root}" ]] || fail_value "--repo-root to be supplied" "<unset>"
[[ -n "${approved_plan}" ]] || fail_value "--approved-plan to be supplied" "<unset>"
[[ -n "${environment_contract}" ]] || fail_value \
  "--environment-contract to be supplied" "<unset>"
[[ "${hard_timeout_seconds}" =~ ^[1-9][0-9]*$ ]] || \
  fail_value "--timeout-seconds to be a positive integer" "${hard_timeout_seconds}"
(( hard_timeout_seconds <= 3600 )) || \
  fail_value "--timeout-seconds not to exceed 3600" "${hard_timeout_seconds}"

provided_repo_root="${repo_root}"
provided_approved_plan="${approved_plan}"
provided_environment_contract="${environment_contract}"
repo_root="$(realpath -e -- "${provided_repo_root}")" || fail_value \
  "repo root to be an existing path" "${provided_repo_root}"
approved_plan="$(realpath -e -- "${provided_approved_plan}")" || fail_value \
  "approved plan to be an existing path" "${provided_approved_plan}"
environment_contract="$(
  realpath -e -- "${provided_environment_contract}"
)" || fail_value \
  "environment contract to be an existing path" \
  "${provided_environment_contract}"
[[ -d "${repo_root}" ]] || fail_value \
  "repo root to be a directory" "${provided_repo_root}"
[[ -f "${approved_plan}" ]] || fail_value \
  "approved plan to be an existing file" "${provided_approved_plan}"
[[ -f "${environment_contract}" ]] || fail_value \
  "environment contract to be an existing file" \
  "${provided_environment_contract}"

run_bounded() {
  local elapsed=$(( SECONDS - started_seconds ))
  local remaining=$(( hard_timeout_seconds - elapsed ))
  (( remaining > 0 )) || fail_value \
    "staged validation to finish within ${hard_timeout_seconds} seconds" \
    "deadline reached before command: $*"
  timeout --kill-after=30s "${remaining}s" "$@"
}

if [[ -x /gpfslocalsup/bin/idrenv ]]; then
  eval "$(/gpfslocalsup/bin/idrenv -d umg)"
elif command -v idrenv >/dev/null 2>&1; then
  eval "$(idrenv -d umg)"
else
  fail_value "idrenv to initialize project umg" "not found"
fi

module_name="pytorch-gpu/py3/2.5.0"
if command -v module >/dev/null 2>&1; then
  module purge
  module load "${module_name}"
else
  module_home="/lustre/fshomisc/sup/spack_soft/environment-modules/4.3.1/gcc-11.3.1-wf7m7j6whgecysm2fm5n73sm4jg7txup"
  module_command="${module_home}/bin/modulecmd"
  [[ -x "${module_command}" ]] || fail_value \
    "module or the reviewed Jean Zay modulecmd backend" "${module_command}"
  export MODULESHOME="${module_home}"
  export MODULEPATH="/lustre/fshomisc/sup/hpe/pub/module-rh/modulefiles:/lustre/fshomisc/sup/hpe/pub/modules-idris-env4/modulefiles/linux-rhel9-skylake_avx512"
  eval "$("${module_command}" bash purge)"
  eval "$("${module_command}" bash load "${module_name}")"
fi

python_bin="/lustre/fshomisc/sup/hpe/pub/miniforge/24.9.0/envs/pytorch-gpu-2.5.0+py3.12.7/bin/python"
observed_python="$(command -v python || true)"
[[ "${observed_python}" == "${python_bin}" ]] || fail_value \
  "the loaded module Python to be ${python_bin}" "${observed_python:-<unset>}"
[[ -x "${python_bin}" ]] || fail_value \
  "the reviewed module Python to be executable" "${python_bin}"

readarray -t verifier_contract < <(
  run_bounded "${python_bin}" -c '
import json
import os
import pathlib
import sys

value = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
verification = value["verification"]
print(
    verification["official_canary_verifier_path_template"].format(
        user=os.environ["USER"]
    )
)
print(verification["official_canary_verifier_sha256"])
' "${environment_contract}"
)
(( ${#verifier_contract[@]} == 2 )) || fail_value \
  "environment contract to emit verifier path and SHA-256" \
  "${verifier_contract[*]-<empty>}"
official_verifier="${verifier_contract[0]}"
official_verifier_sha256="${verifier_contract[1]}"
verifier_source="${repo_root}/experiments/verify_jeanzay_canary.py"
[[ -f "${verifier_source}" && ! -L "${verifier_source}" ]] || fail_value \
  "reviewed repo verifier source to be a regular non-symlink file" \
  "${verifier_source}"
source_verifier_sha256="$(sha256sum "${verifier_source}" | awk '{print $1}')"
[[ "${source_verifier_sha256}" == "${official_verifier_sha256}" ]] || fail_value \
  "repo verifier SHA-256 to match the environment contract" \
  "${source_verifier_sha256}"
if [[ -e "${official_verifier}" || -L "${official_verifier}" ]]; then
  [[ -f "${official_verifier}" && ! -L "${official_verifier}" ]] || fail_value \
    "staged verifier to be a regular non-symlink file" "${official_verifier}"
  observed_verifier_sha256="$(
    sha256sum "${official_verifier}" | awk '{print $1}'
  )"
  [[ "${observed_verifier_sha256}" == "${official_verifier_sha256}" ]] || \
    fail_value \
      "existing staged verifier SHA-256 to remain immutable" \
      "${observed_verifier_sha256}"
else
  install -d -m 700 "$(dirname "${official_verifier}")"
  verifier_temporary="$(
    dirname "${official_verifier}"
  )/.verify_canary.$$.tmp"
  trap 'rm -f "${verifier_temporary:-}"' EXIT
  install -m 700 "${verifier_source}" "${verifier_temporary}"
  sync "${verifier_temporary}"
  ln "${verifier_temporary}" "${official_verifier}"
  rm -f "${verifier_temporary}"
  trap - EXIT
fi

printf 'STAGED_VALIDATION stage=syntax status=running\n'
run_bounded bash -n \
  "${repo_root}/experiments/run_mnist_conv_perfectdiode_successor_confirmation_jeanzay.slurm"
run_bounded "${python_bin}" -m py_compile \
  "${repo_root}/experiments/run_mnist_conv_perfectdiode_successor_confirmation.py" \
  "${repo_root}/experiments/mnist_conv/perfectdiode_successor_confirmation.py" \
  "${repo_root}/experiments/submit_mnist_conv_perfectdiode_successor_confirmation_jeanzay.py" \
  "${repo_root}/experiments/supervise_mnist_conv_perfectdiode_successor_confirmation_jeanzay.py" \
  "${repo_root}/experiments/verify_mnist_conv_perfectdiode_successor_canary.py" \
  "${repo_root}/experiments/verify_jeanzay_canary.py" \
  "${official_verifier}"

printf 'STAGED_VALIDATION stage=plan status=running\n'
run_bounded "${python_bin}" \
  "${repo_root}/skills/run-experiment-pipeline/scripts/validate_experiment_plan.py" \
  "${approved_plan}" \
  --require-approved \
  --verify-files

printf 'STAGED_VALIDATION stage=tracker status=running\n'
run_bounded "${python_bin}" \
  "${repo_root}/skills/run-experiment-pipeline/scripts/validate_current_experiments.py" \
  "${repo_root}/docs/current_experiments.md" \
  --require-experiment-id "${experiment_id}"

printf 'STAGED_VALIDATION stage=focused-tests status=running\n'
run_bounded "${python_bin}" -m pytest -q \
  "${repo_root}/labs/tests/test_agent_fail_stop_policy.py" \
  "${repo_root}/labs/tests/test_mnist_conv_perfectdiode_successor_runtime.py" \
  "${repo_root}/labs/tests/test_mnist_conv_perfectdiode_successor_jeanzay.py"

printf 'STAGED_VALIDATION status=passed elapsed_seconds=%s\n' \
  "$(( SECONDS - started_seconds ))"
