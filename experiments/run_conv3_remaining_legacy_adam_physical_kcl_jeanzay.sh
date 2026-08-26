#!/bin/bash -l
# Jean Zay wrapper for one remaining corrected legacy-Adam Conv3 run.

set -euo pipefail

: "${PHYSKCL_CASE:?Set the remaining Conv3 case identifier.}"
: "${PHYSKCL_SOURCE_ROOT:?Set PHYSKCL_SOURCE_ROOT to the frozen staged source.}"
: "${PHYSKCL_SOURCE_ARCHIVE:?Set PHYSKCL_SOURCE_ARCHIVE to its retained archive.}"
: "${PHYSKCL_RESULT_ROOT:?Set PHYSKCL_RESULT_ROOT to the shared result root.}"
: "${PHYSKCL_DATASET_ROOT:?Set PHYSKCL_DATASET_ROOT to the Jean Zay MNIST root.}"
: "${EXPERIMENT_SOURCE_COMMIT:?Set EXPERIMENT_SOURCE_COMMIT.}"
: "${EXPERIMENT_SOURCE_ARCHIVE_SHA256:?Set EXPERIMENT_SOURCE_ARCHIVE_SHA256.}"
: "${SLURM_JOB_ID:?Submit this wrapper through Slurm.}"

STUDY_ID="perfectdiode-conv3-remaining-legacy-adam-physical-kcl-reruns-seed0-20260826-v1"
MODULE_ID="pytorch-gpu/py3/2.5.0"
INITIALIZER_RELATIVE="results/perfectdiode-conv123-fixed-uniform-init-wmax-adam-10-30-30ep-seed0-20260814-v1/assets/bounded_uniform/conv3/final_model.pt"
INITIALIZER_SHA256="8ebe9dd916e1e9299c31053e2bb37b4f5121f8322f26af7746a92875e555438f"

case "${PHYSKCL_CASE}" in
  clean_eqprop)
    RUN_KIND="clean_eqprop"
    CONFIG_RELATIVE="configs/conv/perfectdiode_conv123_zero_bias_adam_eqprop_one_decade_ordinary_mnist_10_30_30ep_seed0_20260816_v1.json"
    CONFIG_SHA256="adeec73eca2a2182912668a583855b24f4900c0c76a78e53cef6e33de91ee990"
    RESOLVED_CONFIG_SHA256="28ca8e35e790788501dd430d7fcd13e86383cf78fef3559e56168b815d2932ed"
    CASE_STUDY_ID="perfectdiode-conv123-zero-bias-adam-centered-float64-eqprop-one-decade-ordinary-mnist-10-30-30ep-seed0-20260816-v1"
    EVIDENCE_CLASS="ordinary_mnist_eqprop_zero_bias_training_qualification"
    EXPECTED_NOISE_STD="0.0"
    EXPECTED_WEIGHT_MAX="100.0"
    ;;
  noisy_eqprop_5em4)
    RUN_KIND="noisy_eqprop"
    CONFIG_RELATIVE="configs/conv/perfectdiode_conv13_zero_bias_adam_eqprop_one_decade_sigma_5em4_ordinary_mnist_10_30ep_seed0_20260817_v1.json"
    CONFIG_SHA256="fdc0620998eca33874a46c054b8191fed14071c57773c9acdb5b9f11efd50605"
    RESOLVED_CONFIG_SHA256="72b4da95519eaccfc743dc995684850edd9a928afc3866f49f8c265b21b782b1"
    CASE_STUDY_ID="perfectdiode-conv13-zero-bias-adam-centered-float64-eqprop-one-decade-sigma-5em4-ordinary-mnist-10-30ep-seed0-20260817-v1"
    EVIDENCE_CLASS="ordinary_mnist_eqprop_zero_bias_read_noise_training_diagnostic"
    EXPECTED_NOISE_STD="0.0005"
    EXPECTED_WEIGHT_MAX="100.0"
    ;;
  bounded_wmax_1em4)
    RUN_KIND="bounded_bptt"
    CONFIG_RELATIVE="configs/conv/perfectdiode-conv123-fixed-uniform-init-wmax-adam-10-30-30ep-seed0-20260814-v1/conv3/02_wmax_1em4_legacy_adam.json"
    CONFIG_SHA256="c72c3716b5ad77928bcf012d14795cbb7101455a6e80c6f99e4c870c848a7128"
    CASE_STUDY_ID="perfectdiode-conv123-fixed-uniform-init-wmax-adam-10-30-30ep-seed0-20260814-v1"
    EVIDENCE_CLASS="ordinary_mnist_weight_ceiling_sensitivity_exploratory"
    EXPECTED_NOISE_STD="0.0"
    EXPECTED_WEIGHT_MAX="0.0001"
    ;;
  bounded_wmax_5em4)
    RUN_KIND="bounded_bptt"
    CONFIG_RELATIVE="configs/conv/perfectdiode-conv123-fixed-uniform-init-wmax-adam-10-30-30ep-seed0-20260814-v1/conv3/05_wmax_5em4_legacy_adam.json"
    CONFIG_SHA256="75f83208bf24ba2c13e5cec6ed011ff4fad2f346c653f8985ea61c17d1f0397d"
    CASE_STUDY_ID="perfectdiode-conv123-fixed-uniform-init-wmax-adam-10-30-30ep-seed0-20260814-v1"
    EVIDENCE_CLASS="ordinary_mnist_weight_ceiling_sensitivity_exploratory"
    EXPECTED_NOISE_STD="0.0"
    EXPECTED_WEIGHT_MAX="0.0005"
    ;;
  bounded_wmax_1em3)
    RUN_KIND="bounded_bptt"
    CONFIG_RELATIVE="configs/conv/perfectdiode-conv123-fixed-uniform-init-wmax-adam-10-30-30ep-seed0-20260814-v1/conv3/08_wmax_1em3_legacy_adam.json"
    CONFIG_SHA256="078b7ea378047f771a15e3901af3e3d1a05a34671b6942971d0394ceee9283fb"
    CASE_STUDY_ID="perfectdiode-conv123-fixed-uniform-init-wmax-adam-10-30-30ep-seed0-20260814-v1"
    EVIDENCE_CLASS="ordinary_mnist_weight_ceiling_sensitivity_exploratory"
    EXPECTED_NOISE_STD="0.0"
    EXPECTED_WEIGHT_MAX="0.001"
    ;;
  *)
    echo "Unknown PHYSKCL_CASE: ${PHYSKCL_CASE}." >&2
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
if [[ ! -d "${PHYSKCL_SOURCE_ROOT}" || ! -s "${PHYSKCL_SOURCE_ARCHIVE}" ]]; then
  echo "Frozen source directory or retained archive is missing." >&2
  exit 3
fi
if [[ ! -d "${PHYSKCL_DATASET_ROOT}" ]]; then
  echo "MNIST dataset root is missing: ${PHYSKCL_DATASET_ROOT}." >&2
  exit 3
fi

CONFIG_PATH="${PHYSKCL_SOURCE_ROOT}/${CONFIG_RELATIVE}"
INITIALIZER_PATH="${PHYSKCL_SOURCE_ROOT}/${INITIALIZER_RELATIVE}"
if [[ ! -s "${CONFIG_PATH}" ]]; then
  echo "Scientific authority is missing: ${CONFIG_PATH}." >&2
  exit 3
fi
if [[ "${RUN_KIND}" == "bounded_bptt" && ! -s "${INITIALIZER_PATH}" ]]; then
  echo "Fixed Conv3 initializer is missing: ${INITIALIZER_PATH}." >&2
  exit 3
fi

archive_sha="$(sha256sum -- "${PHYSKCL_SOURCE_ARCHIVE}")"
archive_sha="${archive_sha%% *}"
config_sha="$(sha256sum -- "${CONFIG_PATH}")"
config_sha="${config_sha%% *}"
if [[ "${archive_sha}" != "${EXPERIMENT_SOURCE_ARCHIVE_SHA256}" ]]; then
  echo "Frozen source archive SHA mismatch." >&2
  exit 3
fi
if [[ "${config_sha}" != "${CONFIG_SHA256}" ]]; then
  echo "Scientific-authority SHA mismatch for ${PHYSKCL_CASE}." >&2
  exit 3
fi
if [[ "${RUN_KIND}" == "bounded_bptt" ]]; then
  initializer_sha="$(sha256sum -- "${INITIALIZER_PATH}")"
  initializer_sha="${initializer_sha%% *}"
  if [[ "${initializer_sha}" != "${INITIALIZER_SHA256}" ]]; then
    echo "Fixed Conv3 initializer SHA mismatch." >&2
    exit 3
  fi
fi

source_root_canonical="$(realpath -e -- "${PHYSKCL_SOURCE_ROOT}")"
result_root_canonical="$(realpath -m -- "${PHYSKCL_RESULT_ROOT}")"
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
export PYTHONPATH="${PHYSKCL_SOURCE_ROOT}:${PYTHONPATH:-}"

CASE_ROOT="${PHYSKCL_RESULT_ROOT}/production/${PHYSKCL_CASE}"
SUMMARY_ROOT="${PHYSKCL_RESULT_ROOT}/task_summaries"
VALIDATION_ROOT="${PHYSKCL_RESULT_ROOT}/validation"
LOG_ROOT="${PHYSKCL_RESULT_ROOT}/logs"
LOCK_ROOT="${PHYSKCL_RESULT_ROOT}/locks"
mkdir -p "${SUMMARY_ROOT}" "${VALIDATION_ROOT}" "${LOG_ROOT}" "${LOCK_ROOT}"
if [[ -d "${CASE_ROOT}" ]] && find "${CASE_ROOT}" -type f -name result.json -print -quit | grep -q .; then
  echo "Refusing duplicate completed case ${PHYSKCL_CASE}." >&2
  exit 17
fi
LOCK_PATH="${LOCK_ROOT}/${PHYSKCL_CASE}.lock"
if ! mkdir "${LOCK_PATH}"; then
  echo "Refusing concurrent duplicate case ${PHYSKCL_CASE}." >&2
  exit 17
fi
cleanup_lock() {
  rmdir "${LOCK_PATH}" 2>/dev/null || true
}
trap cleanup_lock EXIT

cd "${PHYSKCL_SOURCE_ROOT}"
"${PYTHON_BIN}" - \
  "${RUN_KIND}" "${CONFIG_PATH}" "${PHYSKCL_DATASET_ROOT}" \
  "${EXPECTED_NOISE_STD}" "${EXPECTED_WEIGHT_MAX}" <<'PY'
import json
import math
import sys
from pathlib import Path

import torch

run_kind, authority_arg, dataset_arg, noise_arg, weight_max_arg = sys.argv[1:]
authority_path = Path(authority_arg)
if run_kind == "clean_eqprop":
    from experiments import run_conv123_zero_bias_adam_eqprop_one_decade as runner

    study = runner.load_and_validate_study(authority_path)
    spec = runner.CASE_SPECS[5]
    config = runner._resolved_case_config(study, spec)
elif run_kind == "noisy_eqprop":
    from experiments import run_conv13_zero_bias_adam_eqprop_read_noise_5em4 as runner

    study = runner.load_and_validate_study(authority_path)
    spec = runner.CASE_SPECS[5]
    config = runner._resolved_case_config(study, spec)
else:
    config = json.loads(authority_path.read_text(encoding="utf-8"))
    spec = None

model = config["model_base"]
parameter_order = config["parameter_order"]
named_rates = config["learning_rates_by_parameter"]
expected_algorithm = "EP" if run_kind.endswith("eqprop") else "BP"
checks = {
    "conv3_legacy": config["arm_id"].startswith("conv3_legacy_")
        or "_legacy_adam_" in config["arm_id"],
    "seed": config["seed"] == 0,
    "optimizer": config["optimizer"]["name"] == "Adam",
    "algorithm": config["training_algorithm"] == expected_algorithm,
    "epochs": config["lab"]["epochs"] == 30,
    "T": model["num_iterations_inference"] == 8,
    "K": model["num_iterations_training"] == 8,
    "voltage_amp": model["voltage_amp"] == 4.0,
    "current_amp": model["current_amp"] == 0.25,
    "input_gain": model["input_gain"] == 360.0,
    "perfect_diode": model["non_linearity"] == "perfect_diode",
    "explicit_diode_params": isinstance(model.get("exponential_diode_param"), dict)
        and isinstance(model.get("quadratic_diode_param"), dict),
    "weight_max": math.isclose(
        float(model["weight_max"]), float(weight_max_arg), rel_tol=0.0, abs_tol=0.0
    ),
    "bias_rates": all(
        float(named_rates[name]) == 0.0
        for name in parameter_order
        if name.startswith("Bias_")
    ),
    "official_test_disabled": config["evaluation"]["official_test"]["policy"]
        == "disabled",
}
if expected_algorithm == "EP":
    checks["runtime_float64"] = config["runtime_dtype"] == "float64"
    checks["endpoint_noise"] = math.isclose(
        float(config["eqprop"]["endpoint_read_noise_std"]),
        float(noise_arg),
        rel_tol=0.0,
        abs_tol=0.0,
    )
else:
    checks["fixed_initializer"] = config["init_checkpoint_path"].endswith(
        "/conv3/final_model.pt"
    )
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

echo "PHYSKCL_C3_SOURCE_PASS case=${PHYSKCL_CASE} archive_sha256=${archive_sha} config_sha256=${config_sha}"
EXACT_CONFIGS=("${CONFIG_PATH}")
EXACT_INDEX=0
case "${RUN_KIND}" in
  clean_eqprop)
    RESOLVED_CONFIG_ROOT="${PHYSKCL_RESULT_ROOT}/resolved_configs/${PHYSKCL_CASE}"
    "${PYTHON_BIN}" -m experiments.run_conv123_zero_bias_adam_eqprop_one_decade \
      --study "${CONFIG_PATH}" materialize \
      --directory "${RESOLVED_CONFIG_ROOT}" \
      >"${LOG_ROOT}/${PHYSKCL_CASE}.materialize.log"
    mapfile -t EXACT_CONFIGS < <(find "${RESOLVED_CONFIG_ROOT}" -maxdepth 1 -type f -name '*.json' | sort)
    if [[ "${#EXACT_CONFIGS[@]}" -ne 9 ]]; then
      echo "Expected nine clean EqProp exact configs; found ${#EXACT_CONFIGS[@]}." >&2
      exit 3
    fi
    EXACT_INDEX=5
    ;;
  noisy_eqprop)
    RESOLVED_CONFIG_ROOT="${PHYSKCL_RESULT_ROOT}/resolved_configs/${PHYSKCL_CASE}"
    "${PYTHON_BIN}" -m experiments.run_conv13_zero_bias_adam_eqprop_read_noise_5em4 \
      --study "${CONFIG_PATH}" materialize \
      --directory "${RESOLVED_CONFIG_ROOT}" \
      >"${LOG_ROOT}/${PHYSKCL_CASE}.materialize.log"
    mapfile -t EXACT_CONFIGS < <(find "${RESOLVED_CONFIG_ROOT}" -maxdepth 1 -type f -name '*.json' | sort)
    if [[ "${#EXACT_CONFIGS[@]}" -ne 6 ]]; then
      echo "Expected six noisy EqProp exact configs; found ${#EXACT_CONFIGS[@]}." >&2
      exit 3
    fi
    EXACT_INDEX=5
    ;;
esac

if [[ "${RUN_KIND}" != "bounded_bptt" ]]; then
  resolved_config_sha="$(sha256sum -- "${EXACT_CONFIGS[EXACT_INDEX]}")"
  resolved_config_sha="${resolved_config_sha%% *}"
  if [[ "${resolved_config_sha}" != "${RESOLVED_CONFIG_SHA256}" ]]; then
    echo "Materialized exact-config SHA mismatch for ${PHYSKCL_CASE}." >&2
    exit 3
  fi
fi
"${PYTHON_BIN}" -m experiments.exact_run "${EXACT_CONFIGS[@]}" \
  --index "${EXACT_INDEX}" \
  --output-root "${CASE_ROOT}" \
  --device cuda \
  --dataset-root "${PHYSKCL_DATASET_ROOT}" \
  --skip-terminal-official-test \
  --study-id "${CASE_STUDY_ID}" \
  --evidence-class "${EVIDENCE_CLASS}" \
  --target jean-zay \
  --summary-json "${SUMMARY_ROOT}/${PHYSKCL_CASE}.json" \
  >"${LOG_ROOT}/${PHYSKCL_CASE}.log" 2>&1

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
  >"${VALIDATION_ROOT}/${PHYSKCL_CASE}.txt"
echo "PHYSKCL_C3_SEMANTIC_PASS case=${PHYSKCL_CASE} run_dir=${RUN_DIR} validation=${VALIDATION_ROOT}/${PHYSKCL_CASE}.txt"
