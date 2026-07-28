#!/bin/bash -l
#
# Stable Jean Zay validation runner.
#
# Slurm resources and log paths are supplied by the launch CLI. Experiment
# variation lives only in the immutable package's launch-spec.json.

set -euo pipefail

fail_value() {
  local expected="$1"
  local provided="$2"
  printf 'Expected %s.\n' "${expected}" >&2
  printf 'Provided value: %s\n' "${provided}" >&2
  exit 2
}

preflight=false
package_root=""
project=""
module_name=""
python_executable=""

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --preflight)
      preflight=true
      shift
      ;;
    --package-root)
      [[ "$#" -ge 2 ]] || fail_value "a value after --package-root" "<missing>"
      package_root="$2"
      shift 2
      ;;
    --project)
      [[ "$#" -ge 2 ]] || fail_value "a value after --project" "<missing>"
      project="$2"
      shift 2
      ;;
    --module)
      [[ "$#" -ge 2 ]] || fail_value "a value after --module" "<missing>"
      module_name="$2"
      shift 2
      ;;
    --python)
      [[ "$#" -ge 2 ]] || fail_value "a value after --python" "<missing>"
      python_executable="$2"
      shift 2
      ;;
    *)
      fail_value "only documented runner arguments" "$1"
      ;;
  esac
done

[[ "${package_root}" == /* ]] || \
  fail_value "an absolute --package-root" "${package_root:-<unset>}"
[[ -n "${project}" ]] || fail_value "a non-empty --project" "<unset>"
[[ -n "${module_name}" ]] || fail_value "a non-empty --module" "<unset>"
[[ "${python_executable}" == /* ]] || \
  fail_value "an absolute --python" "${python_executable:-<unset>}"

readonly source_root="${package_root}/source"
readonly launch_spec="${package_root}/launch-spec.json"

if [[ -x /gpfslocalsup/bin/idrenv ]]; then
  eval "$(/gpfslocalsup/bin/idrenv -d "${project}")"
else
  fail_value "the canonical Jean Zay idrenv helper" "/gpfslocalsup/bin/idrenv"
fi

if command -v module >/dev/null 2>&1; then
  module purge
  module load "${module_name}"
else
  readonly module_home="/lustre/fshomisc/sup/spack_soft/environment-modules/4.3.1/gcc-11.3.1-wf7m7j6whgecysm2fm5n73sm4jg7txup"
  readonly module_command="${module_home}/bin/modulecmd"
  [[ -x "${module_command}" ]] || \
    fail_value "module or the reviewed modulecmd backend" "${module_command}"
  export MODULESHOME="${module_home}"
  export MODULEPATH="/lustre/fshomisc/sup/hpe/pub/module-rh/modulefiles:/lustre/fshomisc/sup/hpe/pub/modules-idris-env4/modulefiles/linux-rhel9-skylake_avx512"
  eval "$("${module_command}" bash purge)"
  eval "$("${module_command}" bash load "${module_name}")"
fi

observed_python="$(command -v python || true)"
[[ "${observed_python}" == "${python_executable}" ]] || \
  fail_value "the target-profile Python ${python_executable}" "${observed_python:-<unset>}"

export PYTHONPATH="${source_root}:${source_root}/labs"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

if [[ "${preflight}" == true ]]; then
  "${python_executable}" -c \
    'import json, sys, torch; import custom_minimizer; from labs.custom_classes import FlexibleDeepResistiveEnergy; import experiments.run_mnist_conv_perfectdiode_job; import experiments.jeanzay_validation_worker; print(json.dumps({"schema_version": "jean-zay-import-preflight/v1", "status": "passed", "python": sys.executable, "torch": torch.__version__, "cuda_build": torch.version.cuda, "imports": ["custom_minimizer", "labs.custom_classes.FlexibleDeepResistiveEnergy", "experiments.run_mnist_conv_perfectdiode_job", "experiments.jeanzay_validation_worker"], "side_effects_performed": False}, sort_keys=True))'
  exit 0
fi

[[ -f "${launch_spec}" ]] || \
  fail_value "the immutable package launch specification" "${launch_spec}"
[[ "${SLURM_ARRAY_TASK_ID:-}" == "0" ]] || \
  fail_value "the singleton Slurm array task id 0" "${SLURM_ARRAY_TASK_ID:-<unset>}"
[[ "${SLURM_ARRAY_TASK_COUNT:-}" == "1" ]] || \
  fail_value "one Slurm array task" "${SLURM_ARRAY_TASK_COUNT:-<unset>}"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || \
  fail_value "one visible Slurm GPU" "${CUDA_VISIBLE_DEVICES:-<unset>}"

exec "${python_executable}" -m experiments.jeanzay_validation_worker \
  --launch-spec "${launch_spec}"
