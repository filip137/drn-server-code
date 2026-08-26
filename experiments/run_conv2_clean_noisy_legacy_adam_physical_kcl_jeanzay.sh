#!/bin/bash -l
# Jean Zay wrapper for corrected legacy-Adam Conv2 clean/noisy EqProp repeats.

set -euo pipefail

: "${PHYSKCL_C2_CASE:?Set the Conv2 EqProp case identifier.}"
: "${PHYSKCL_C2_SOURCE_ROOT:?Set PHYSKCL_C2_SOURCE_ROOT to the frozen staged source.}"
: "${PHYSKCL_C2_SOURCE_ARCHIVE:?Set PHYSKCL_C2_SOURCE_ARCHIVE to its retained archive.}"
: "${PHYSKCL_C2_RESULT_ROOT:?Set PHYSKCL_C2_RESULT_ROOT to the shared result root.}"
: "${PHYSKCL_C2_DATASET_ROOT:?Set PHYSKCL_C2_DATASET_ROOT to the Jean Zay MNIST root.}"
: "${EXPERIMENT_SOURCE_COMMIT:?Set EXPERIMENT_SOURCE_COMMIT.}"
: "${EXPERIMENT_SOURCE_ARCHIVE_SHA256:?Set EXPERIMENT_SOURCE_ARCHIVE_SHA256.}"
: "${SLURM_JOB_ID:?Submit this wrapper through Slurm.}"

STUDY_ID="perfectdiode-conv2-legacy-adam-clean-noisy-eqprop-physical-kcl-reruns-seed0-20260826-v1"
MODULE_ID="pytorch-gpu/py3/2.5.0"

case "${PHYSKCL_C2_CASE}" in
  clean_eqprop)
    CONFIG_RELATIVE="configs/conv/perfectdiode_conv123_zero_bias_adam_eqprop_one_decade_ordinary_mnist_10_30_30ep_seed0_20260816_v1.json"
    CONFIG_SHA256="adeec73eca2a2182912668a583855b24f4900c0c76a78e53cef6e33de91ee990"
    RESOLVED_CONFIG_SHA256="83609bea33d5ad85f4def238ccb8d98a7a7bbcf1f54c58cfca910cf3aa404c2a"
    CASE_STUDY_ID="perfectdiode-conv123-zero-bias-adam-centered-float64-eqprop-one-decade-ordinary-mnist-10-30-30ep-seed0-20260816-v1"
    EVIDENCE_CLASS="ordinary_mnist_eqprop_zero_bias_training_qualification"
    RUNNER_MODULE="experiments.run_conv123_zero_bias_adam_eqprop_one_decade"
    EXPECTED_CONFIG_COUNT=9
    EXACT_INDEX=8
    EXPECTED_NOISE_STD="0.0"
    ;;
  noisy_eqprop_5em4)
    CONFIG_RELATIVE="configs/conv/perfectdiode_conv2_zero_bias_adam_eqprop_one_decade_sigma_5em4_ordinary_mnist_30ep_seed0_20260817_v1.json"
    CONFIG_SHA256="ca4519e6b2839c5c8c1a5ad51850916054ea8f52cf2cf7454628e855599193c7"
    RESOLVED_CONFIG_SHA256="863f0ee3aab5b103253492db68fbd85e10c17f11580f3bc3b45725843ccfe1ea"
    CASE_STUDY_ID="perfectdiode-conv2-zero-bias-adam-centered-float64-eqprop-one-decade-sigma-5em4-ordinary-mnist-30ep-seed0-20260817-v1"
    EVIDENCE_CLASS="ordinary_mnist_eqprop_zero_bias_read_noise_training_diagnostic"
    RUNNER_MODULE="experiments.run_conv2_zero_bias_adam_eqprop_read_noise_5em4"
    EXPECTED_CONFIG_COUNT=3
    EXACT_INDEX=2
    EXPECTED_NOISE_STD="0.0005"
    ;;
  *)
    echo "Unknown PHYSKCL_C2_CASE: ${PHYSKCL_C2_CASE}." >&2
    exit 3
    ;;
esac

if [[ ! "${EXPERIMENT_SOURCE_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "EXPERIMENT_SOURCE_COMMIT must be a lowercase 40-character SHA." >&2
  exit 3
fi
if [[ ! "${EXPERIMENT_SOURCE_ARCHIVE_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "EXPERIMENT_SOURCE_ARCHIVE_SHA256 must be a lowercase SHA-256." >&2
  exit 3
fi
if [[ ! -d "${PHYSKCL_C2_SOURCE_ROOT}" || ! -s "${PHYSKCL_C2_SOURCE_ARCHIVE}" ]]; then
  echo "Frozen source directory or retained archive is missing." >&2
  exit 3
fi
if [[ ! -d "${PHYSKCL_C2_DATASET_ROOT}" ]]; then
  echo "MNIST dataset root is missing: ${PHYSKCL_C2_DATASET_ROOT}." >&2
  exit 3
fi

CONFIG_PATH="${PHYSKCL_C2_SOURCE_ROOT}/${CONFIG_RELATIVE}"
if [[ ! -s "${CONFIG_PATH}" ]]; then
  echo "Scientific authority is missing: ${CONFIG_PATH}." >&2
  exit 3
fi

archive_sha="$(sha256sum -- "${PHYSKCL_C2_SOURCE_ARCHIVE}")"
archive_sha="${archive_sha%% *}"
config_sha="$(sha256sum -- "${CONFIG_PATH}")"
config_sha="${config_sha%% *}"
if [[ "${archive_sha}" != "${EXPERIMENT_SOURCE_ARCHIVE_SHA256}" ]]; then
  echo "Frozen source archive SHA mismatch." >&2
  exit 3
fi
if [[ "${config_sha}" != "${CONFIG_SHA256}" ]]; then
  echo "Scientific-authority SHA mismatch for ${PHYSKCL_C2_CASE}." >&2
  exit 3
fi

source_root_canonical="$(realpath -e -- "${PHYSKCL_C2_SOURCE_ROOT}")"
result_root_canonical="$(realpath -m -- "${PHYSKCL_C2_RESULT_ROOT}")"
if [[ "${source_root_canonical}" == "${result_root_canonical}" \
   || "${source_root_canonical}" == "${result_root_canonical}/"* \
   || "${result_root_canonical}" == "${source_root_canonical}/"* ]]; then
  echo "Frozen source and result roots must be disjoint." >&2
  exit 3
fi

export MODULESHOME="/lustre/fshomisc/sup/spack_soft/environment-modules/4.3.1/gcc-11.3.1-wf7m7j6whgecysm2fm5n73sm4jg7txup"
export MODULEPATH="/lustre/fshomisc/sup/hpe/pub/module-rh/modulefiles:/lustre/fshomisc/sup/hpe/pub/modules-idris-env4/modulefiles/linux-rhel9-skylake_avx512"
MODULECMD="${MODULESHOME}/bin/modulecmd"
eval "$("${MODULECMD}" bash purge)"
eval "$("${MODULECMD}" bash load "${MODULE_ID}")"
PYTHON_BIN="$(command -v python)"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "${MODULE_ID} did not provide Python." >&2
  exit 3
fi

export KMP_DISABLE_SHM=1
export KMP_SHM_DISABLE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PHYSKCL_C2_SOURCE_ROOT}:${PYTHONPATH:-}"

CASE_ROOT="${PHYSKCL_C2_RESULT_ROOT}/production/${PHYSKCL_C2_CASE}"
SUMMARY_ROOT="${PHYSKCL_C2_RESULT_ROOT}/task_summaries"
VALIDATION_ROOT="${PHYSKCL_C2_RESULT_ROOT}/validation"
LOG_ROOT="${PHYSKCL_C2_RESULT_ROOT}/logs"
LOCK_ROOT="${PHYSKCL_C2_RESULT_ROOT}/locks"
mkdir -p "${SUMMARY_ROOT}" "${VALIDATION_ROOT}" "${LOG_ROOT}" "${LOCK_ROOT}"
if [[ -d "${CASE_ROOT}" ]] && find "${CASE_ROOT}" -type f -name result.json -print -quit | grep -q .; then
  echo "Refusing duplicate completed case ${PHYSKCL_C2_CASE}." >&2
  exit 17
fi
LOCK_PATH="${LOCK_ROOT}/${PHYSKCL_C2_CASE}.lock"
if ! mkdir "${LOCK_PATH}"; then
  echo "Refusing concurrent duplicate case ${PHYSKCL_C2_CASE}." >&2
  exit 17
fi
cleanup_lock() {
  rmdir "${LOCK_PATH}" 2>/dev/null || true
}
trap cleanup_lock EXIT

cd "${PHYSKCL_C2_SOURCE_ROOT}"
"${PYTHON_BIN}" - \
  "${RUNNER_MODULE}" "${CONFIG_PATH}" "${PHYSKCL_C2_DATASET_ROOT}" \
  "${EXPECTED_NOISE_STD}" <<'PY'
import importlib
import json
import math
import sys
from pathlib import Path

import torch

runner_module, authority_arg, dataset_arg, noise_arg = sys.argv[1:]
runner = importlib.import_module(runner_module)
study = runner.load_and_validate_study(Path(authority_arg))
spec = next(
    row for row in runner.CASE_SPECS
    if row.architecture == "conv2" and row.scheme == "legacy"
)
config = runner._resolved_case_config(study, spec)
model = config["model_base"]
parameter_order = config["parameter_order"]
named_rates = config["learning_rates_by_parameter"]
checks = {
    "conv2_legacy": config["arm_id"].startswith("conv2_legacy_"),
    "seed": config["seed"] == 0,
    "optimizer": config["optimizer"]["name"] == "Adam",
    "algorithm": config["training_algorithm"] == "EP",
    "epochs": config["lab"]["epochs"] == 30,
    "T": model["num_iterations_inference"] == 6,
    "K": model["num_iterations_training"] == 6,
    "voltage_amp": model["voltage_amp"] == 4.0,
    "current_amp": model["current_amp"] == 0.25,
    "input_gain": model["input_gain"] == 100.0,
    "perfect_diode": model["non_linearity"] == "perfect_diode",
    "explicit_diode_params": isinstance(model.get("exponential_diode_param"), dict)
        and isinstance(model.get("quadratic_diode_param"), dict),
    "weight_min": model["weight_min"] == 0.0,
    "weight_max": model["weight_max"] == 100.0,
    "bias_rates": all(
        float(named_rates[name]) == 0.0
        for name in parameter_order
        if name.startswith("Bias_")
    ),
    "runtime_float64": config["runtime_dtype"] == "float64",
    "base_beta": math.isclose(
        float(config["beta"]), 0.0001171875, rel_tol=0.0, abs_tol=0.0
    ),
    "injected_beta": math.isclose(
        float(config["eqprop"]["injected_beta_B"]),
        0.03,
        rel_tol=0.0,
        abs_tol=0.0,
    ),
    "endpoint_noise": math.isclose(
        float(config["eqprop"]["endpoint_read_noise_std"]),
        float(noise_arg),
        rel_tol=0.0,
        abs_tol=0.0,
    ),
    "official_test_disabled": config["evaluation"]["official_test"]["policy"]
        == "disabled",
}
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise SystemExit(f"Scientific preflight failed: {failed}")

dataset_root = Path(dataset_arg)
for relative in (
    "MNIST/raw/train-images-idx3-ubyte",
    "MNIST/raw/train-labels-idx1-ubyte",
):
    path = dataset_root / relative
    if not path.is_file() or path.stat().st_size <= 0:
        raise SystemExit(f"Required MNIST source is missing or empty: {path}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in the allocated task.")
device_name = torch.cuda.get_device_name(0)
device_bytes = int(torch.cuda.get_device_properties(0).total_memory)
if "V100" not in device_name.upper() or device_bytes < 31 * 1024**3:
    raise SystemExit(
        f"Expected V100 with at least 31 GiB, got {device_name!r}/{device_bytes}."
    )
print(
    json.dumps(
        {
            "case": config["arm_id"],
            "checks": checks,
            "device": device_name,
            "device_bytes": device_bytes,
            "pytorch": torch.__version__,
        },
        sort_keys=True,
    ),
    flush=True,
)
PY

echo "PHYSKCL_C2_SOURCE_PASS case=${PHYSKCL_C2_CASE} archive_sha256=${archive_sha} config_sha256=${config_sha}"
RESOLVED_CONFIG_ROOT="${PHYSKCL_C2_RESULT_ROOT}/resolved_configs/${PHYSKCL_C2_CASE}"
"${PYTHON_BIN}" -m "${RUNNER_MODULE}" \
  --study "${CONFIG_PATH}" materialize \
  --directory "${RESOLVED_CONFIG_ROOT}" \
  >"${LOG_ROOT}/${PHYSKCL_C2_CASE}.materialize.log"
mapfile -t EXACT_CONFIGS < <(find "${RESOLVED_CONFIG_ROOT}" -maxdepth 1 -type f -name '*.json' | sort)
if [[ "${#EXACT_CONFIGS[@]}" -ne "${EXPECTED_CONFIG_COUNT}" ]]; then
  echo "Expected ${EXPECTED_CONFIG_COUNT} exact configs; found ${#EXACT_CONFIGS[@]}." >&2
  exit 3
fi
resolved_config_sha="$(sha256sum -- "${EXACT_CONFIGS[EXACT_INDEX]}")"
resolved_config_sha="${resolved_config_sha%% *}"
if [[ "${resolved_config_sha}" != "${RESOLVED_CONFIG_SHA256}" ]]; then
  echo "Materialized exact-config SHA mismatch for ${PHYSKCL_C2_CASE}." >&2
  exit 3
fi

"${PYTHON_BIN}" -m experiments.exact_run "${EXACT_CONFIGS[@]}" \
  --index "${EXACT_INDEX}" \
  --output-root "${CASE_ROOT}" \
  --device cuda \
  --dataset-root "${PHYSKCL_C2_DATASET_ROOT}" \
  --skip-terminal-official-test \
  --study-id "${CASE_STUDY_ID}" \
  --evidence-class "${EVIDENCE_CLASS}" \
  --target jean-zay \
  --summary-json "${SUMMARY_ROOT}/${PHYSKCL_C2_CASE}.json" \
  >"${LOG_ROOT}/${PHYSKCL_C2_CASE}.log" 2>&1

mapfile -t result_paths < <(find "${CASE_ROOT}" -type f -name result.json | sort)
if [[ "${#result_paths[@]}" -ne 1 ]]; then
  echo "Expected one successful canonical run; found ${#result_paths[@]}." >&2
  exit 1
fi
RUN_DIR="${result_paths[0]%/result.json}"
for artifact in manifest.json status.json metrics.jsonl result.json; do
  if [[ ! -s "${RUN_DIR}/${artifact}" ]]; then
    echo "Canonical artifact is missing: ${RUN_DIR}/${artifact}." >&2
    exit 1
  fi
done
"${PYTHON_BIN}" -m experiments.reporting validate-run "${RUN_DIR}" \
  >"${VALIDATION_ROOT}/${PHYSKCL_C2_CASE}.txt"
echo "PHYSKCL_C2_SEMANTIC_PASS case=${PHYSKCL_C2_CASE} run_dir=${RUN_DIR} validation=${VALIDATION_ROOT}/${PHYSKCL_C2_CASE}.txt"
