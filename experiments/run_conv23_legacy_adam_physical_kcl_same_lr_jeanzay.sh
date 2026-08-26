#!/bin/bash -l
# Jean Zay wrapper for one corrected legacy-Adam Conv2 or Conv3 run.

set -euo pipefail

: "${PHYSKCL_ARCH:?Set PHYSKCL_ARCH to conv2 or conv3.}"
: "${PHYSKCL_SOURCE_ROOT:?Set PHYSKCL_SOURCE_ROOT to the frozen staged source.}"
: "${PHYSKCL_SOURCE_ARCHIVE:?Set PHYSKCL_SOURCE_ARCHIVE to its retained archive.}"
: "${PHYSKCL_RESULT_ROOT:?Set PHYSKCL_RESULT_ROOT to the shared result root.}"
: "${PHYSKCL_DATASET_ROOT:?Set PHYSKCL_DATASET_ROOT to the Jean Zay MNIST root.}"
: "${EXPERIMENT_SOURCE_COMMIT:?Set EXPERIMENT_SOURCE_COMMIT.}"
: "${EXPERIMENT_SOURCE_ARCHIVE_SHA256:?Set EXPERIMENT_SOURCE_ARCHIVE_SHA256.}"
: "${SLURM_JOB_ID:?Submit this wrapper through Slurm.}"

STUDY_ID="perfectdiode-conv23-legacy-adam-physical-kcl-same-lr-bptt-rerun-seed0-20260826-v1"
EVIDENCE_CLASS="ordinary_mnist_physical_kcl_same_lr_bptt_compatibility"
MODULE_ID="pytorch-gpu/py3/2.5.0"

case "${PHYSKCL_ARCH}" in
  conv2)
    CONFIG_RELATIVE="configs/conv/perfectdiode_conv123_zero_bias_ordinary_mnist_seed0_20260805_v1/conv2/05_legacy_adam_bias_zero_seed0.json"
    CONFIG_SHA256="6db2f139cfa023d9af6f83b4dea91dfa615b99ecf5a11bb959a063cef5180540"
    EXPECTED_ITERATIONS=6
    EXPECTED_INPUT_GAIN=100.0
    ;;
  conv3)
    CONFIG_RELATIVE="configs/conv/perfectdiode_conv123_zero_bias_ordinary_mnist_seed0_20260805_v1/conv3/05_legacy_adam_bias_zero_seed0.json"
    CONFIG_SHA256="06f658e108463da0612ca9322c7c6e6bd92c18640ff0f12c37259a96b79e9784"
    EXPECTED_ITERATIONS=8
    EXPECTED_INPUT_GAIN=360.0
    ;;
  *)
    echo "PHYSKCL_ARCH must be conv2 or conv3; got ${PHYSKCL_ARCH}." >&2
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
if [[ ! -s "${CONFIG_PATH}" ]]; then
  echo "Exact config is missing: ${CONFIG_PATH}." >&2
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
  echo "Exact-config SHA mismatch for ${PHYSKCL_ARCH}." >&2
  exit 3
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

mkdir -p "${PHYSKCL_RESULT_ROOT}/production" \
  "${PHYSKCL_RESULT_ROOT}/task_summaries" \
  "${PHYSKCL_RESULT_ROOT}/locks"
LOCK_PATH="${PHYSKCL_RESULT_ROOT}/locks/${PHYSKCL_ARCH}.lock"
if ! mkdir "${LOCK_PATH}"; then
  echo "Refusing concurrent or duplicate ${PHYSKCL_ARCH} execution." >&2
  exit 17
fi
cleanup_lock() {
  rmdir "${LOCK_PATH}" 2>/dev/null || true
}
trap cleanup_lock EXIT

cd "${PHYSKCL_SOURCE_ROOT}"
"${PYTHON_BIN}" - "${CONFIG_PATH}" "${PHYSKCL_ARCH}" \
  "${EXPECTED_ITERATIONS}" "${EXPECTED_INPUT_GAIN}" <<'PY'
import json
import sys
from pathlib import Path

import torch

config_path, architecture, expected_iterations, expected_input_gain = sys.argv[1:]
config = json.loads(Path(config_path).read_text(encoding="utf-8"))
expected_biases = 2 if architecture == "conv2" else 3
checks = {
    "optimizer": config["optimizer"]["name"] == "Adam",
    "algorithm": config["training_algorithm"] == "BP",
    "epochs": config["lab"]["epochs"] == 30,
    "seed": config["seed"] == 0,
    "voltage_amp": config["model_base"]["voltage_amp"] == 4.0,
    "current_amp": config["model_base"]["current_amp"] == 0.25,
    "perfect_diode": config["model_base"]["non_linearity"] == "perfect_diode",
    "inference_iterations": config["model_base"]["num_iterations_inference"]
        == int(expected_iterations),
    "training_iterations": config["model_base"]["num_iterations_training"]
        == int(expected_iterations),
    "input_gain": config["model_base"]["input_gain"] == float(expected_input_gain),
    "bias_rates": all(
        config["learning_rates_by_parameter"][f"Bias_{index}"] == 0.0
        for index in range(expected_biases)
    ),
    "official_test_disabled": config["evaluation"]["official_test"]["policy"]
        == "disabled",
}
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise SystemExit(f"Scientific preflight failed: {failed}")
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot see the allocated CUDA device.")
print(
    json.dumps(
        {
            "architecture": architecture,
            "checks": checks,
            "cuda_device": torch.cuda.get_device_name(0),
            "pytorch": torch.__version__,
        },
        sort_keys=True,
    )
)
PY

"${PYTHON_BIN}" -m experiments.exact_run "${CONFIG_PATH}" \
  --output-root "${PHYSKCL_RESULT_ROOT}/production" \
  --device cuda \
  --dataset-root "${PHYSKCL_DATASET_ROOT}" \
  --skip-terminal-official-test \
  --study-id "${STUDY_ID}" \
  --evidence-class "${EVIDENCE_CLASS}" \
  --target jean-zay \
  --summary-json "${PHYSKCL_RESULT_ROOT}/task_summaries/${PHYSKCL_ARCH}.json"
