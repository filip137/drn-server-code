#!/usr/bin/env bash

set -euo pipefail

: "${BWMAX_SOURCE_ROOT:?Set BWMAX_SOURCE_ROOT to the frozen staged source.}"
: "${BWMAX_SOURCE_ARCHIVE:?Set BWMAX_SOURCE_ARCHIVE to the staged source archive.}"
: "${BWMAX_RESULT_ROOT:?Set BWMAX_RESULT_ROOT to the study result directory.}"
: "${BWMAX_ARCHITECTURE:?Set BWMAX_ARCHITECTURE to conv1 or conv2.}"
: "${BWMAX_TARGET:?Set BWMAX_TARGET to akib or trex.}"
: "${BWMAX_DATASET_ROOT:?Set BWMAX_DATASET_ROOT to the target MNIST root.}"
: "${BWMAX_PYTHON:?Set BWMAX_PYTHON to the target Python executable.}"
: "${BWMAX_CONFIG_SET_SHA256:?Set BWMAX_CONFIG_SET_SHA256 to the architecture config-set digest.}"
: "${EXPERIMENT_SOURCE_COMMIT:?Set EXPERIMENT_SOURCE_COMMIT to the frozen parent source commit.}"
: "${EXPERIMENT_SOURCE_ARCHIVE_SHA256:?Set EXPERIMENT_SOURCE_ARCHIVE_SHA256 to the staged archive digest.}"

STUDY_ID="perfectdiode-bounded-wmax-sweep-signed-scaled-bias-conv123-seed0-20260809-v1"
EXPECTED_SOURCE_COMMIT="08dd52e44818adf43b011542748e140dc0e01aeb"
CONFIG_ROOT="${BWMAX_SOURCE_ROOT}/configs/conv/perfectdiode_bounded_wmax_sweep_signed_scaled_bias_conv123_seed0_20260809_v1"

if [[ "${EXPERIMENT_SOURCE_COMMIT}" != "${EXPECTED_SOURCE_COMMIT}" ]]; then
  echo "Unexpected source commit: ${EXPERIMENT_SOURCE_COMMIT}." >&2
  exit 3
fi
if [[ ! "${EXPERIMENT_SOURCE_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "EXPERIMENT_SOURCE_COMMIT must be a lowercase forty-character Git identity." >&2
  exit 3
fi
if [[ ! -d "${BWMAX_SOURCE_ROOT}" ]]; then
  echo "Frozen source directory is missing: ${BWMAX_SOURCE_ROOT}." >&2
  exit 3
fi
if [[ ! -s "${BWMAX_SOURCE_ARCHIVE}" ]]; then
  echo "Frozen source archive is missing or empty: ${BWMAX_SOURCE_ARCHIVE}." >&2
  exit 3
fi
if [[ ! -d "${BWMAX_DATASET_ROOT}" ]]; then
  echo "MNIST dataset root is missing: ${BWMAX_DATASET_ROOT}." >&2
  exit 3
fi
if [[ ! "${BWMAX_CONFIG_SET_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "BWMAX_CONFIG_SET_SHA256 must be a lowercase hexadecimal SHA-256." >&2
  exit 3
fi
if [[ ! "${EXPERIMENT_SOURCE_ARCHIVE_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "EXPERIMENT_SOURCE_ARCHIVE_SHA256 must be a lowercase hexadecimal SHA-256." >&2
  exit 3
fi
ACTUAL_SOURCE_ARCHIVE_SHA256="$(sha256sum -- "${BWMAX_SOURCE_ARCHIVE}")"
ACTUAL_SOURCE_ARCHIVE_SHA256="${ACTUAL_SOURCE_ARCHIVE_SHA256%% *}"
if [[ "${ACTUAL_SOURCE_ARCHIVE_SHA256}" != "${EXPERIMENT_SOURCE_ARCHIVE_SHA256}" ]]; then
  echo "Source archive SHA-256 mismatch: expected ${EXPERIMENT_SOURCE_ARCHIVE_SHA256}, got ${ACTUAL_SOURCE_ARCHIVE_SHA256}." >&2
  exit 3
fi
SOURCE_ROOT_CANONICAL="$(realpath -e -- "${BWMAX_SOURCE_ROOT}")"
RESULT_ROOT_CANONICAL="$(realpath -m -- "${BWMAX_RESULT_ROOT}")"
if [[ "${SOURCE_ROOT_CANONICAL}" == "${RESULT_ROOT_CANONICAL}" \
   || "${SOURCE_ROOT_CANONICAL}" == "${RESULT_ROOT_CANONICAL}/"* \
   || "${RESULT_ROOT_CANONICAL}" == "${SOURCE_ROOT_CANONICAL}/"* ]]; then
  echo "Frozen source and result roots must be disjoint." >&2
  exit 3
fi
case "${BWMAX_ARCHITECTURE}:${BWMAX_TARGET}" in
  conv1:akib | conv2:trex) ;;
  *)
    echo "Expected conv1:akib or conv2:trex; got ${BWMAX_ARCHITECTURE}:${BWMAX_TARGET}." >&2
    exit 3
    ;;
esac
if [[ ! -x "${BWMAX_PYTHON}" ]]; then
  echo "Python is missing or non-executable: ${BWMAX_PYTHON}." >&2
  exit 3
fi

export LC_ALL=C
shopt -s nullglob
CONFIGS=("${CONFIG_ROOT}/${BWMAX_ARCHITECTURE}/"*.json)
shopt -u nullglob
if [[ "${#CONFIGS[@]}" -ne 30 ]]; then
  echo "Expected exactly thirty ${BWMAX_ARCHITECTURE} configs; got ${#CONFIGS[@]}." >&2
  exit 3
fi
if [[ ! -s "${CONFIG_ROOT}/study_manifest.json" ]]; then
  echo "Missing or empty study manifest: ${CONFIG_ROOT}/study_manifest.json." >&2
  exit 3
fi

BWMAX_RUN_MODE="${BWMAX_RUN_MODE:-production}"
case "${BWMAX_RUN_MODE}" in
  production)
    START_INDEX="${BWMAX_START_INDEX:-0}"
    END_INDEX="${BWMAX_END_INDEX:-29}"
    MODE_ARGS=()
    VALIDATOR_MODE_ARGS=()
    RUN_ROOT="${BWMAX_RESULT_ROOT}/${BWMAX_ARCHITECTURE}/runs"
    SUMMARY_ROOT="${BWMAX_RESULT_ROOT}/${BWMAX_ARCHITECTURE}/task_summaries/production"
    VALIDATION_ROOT="${BWMAX_RESULT_ROOT}/${BWMAX_ARCHITECTURE}/validation/production"
    ;;
  smoke)
    START_INDEX="${BWMAX_INDEX:-0}"
    END_INDEX="${START_INDEX}"
    MODE_ARGS=(--smoke)
    VALIDATOR_MODE_ARGS=(--smoke)
    RUN_ROOT="${BWMAX_RESULT_ROOT}/target_smoke/${BWMAX_ARCHITECTURE}"
    SUMMARY_ROOT="${BWMAX_RESULT_ROOT}/target_smoke/task_summaries/${BWMAX_ARCHITECTURE}"
    VALIDATION_ROOT="${BWMAX_RESULT_ROOT}/target_smoke/validation/${BWMAX_ARCHITECTURE}"
    ;;
  *)
    echo "BWMAX_RUN_MODE must be production or smoke; got ${BWMAX_RUN_MODE}." >&2
    exit 3
    ;;
esac
if [[ ! "${START_INDEX}" =~ ^[0-9]+$ || ! "${END_INDEX}" =~ ^[0-9]+$ ]]; then
  echo "Requested indices must be non-negative integers." >&2
  exit 3
fi
START_INDEX=$((10#${START_INDEX}))
END_INDEX=$((10#${END_INDEX}))
if (( START_INDEX < 0 || END_INDEX >= 30 || START_INDEX > END_INDEX )); then
  echo "Invalid requested index interval [${START_INDEX},${END_INDEX}]." >&2
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

mkdir -p "${RUN_ROOT}" "${SUMMARY_ROOT}" "${VALIDATION_ROOT}"
cd "${BWMAX_SOURCE_ROOT}"

"${BWMAX_PYTHON}" -m experiments.validate_bounded_wmax_run \
  --config-root "${CONFIG_ROOT}" \
  --architecture "${BWMAX_ARCHITECTURE}" \
  --expected-config-set-sha256 "${BWMAX_CONFIG_SET_SHA256}" \
  --output "${VALIDATION_ROOT}/config_set.json"

for (( index=START_INDEX; index<=END_INDEX; index++ )); do
  SUMMARY_PATH="${SUMMARY_ROOT}/${index}.json"
  "${BWMAX_PYTHON}" -m experiments.exact_run \
    "${CONFIGS[@]}" \
    --output-root "${RUN_ROOT}" \
    --device cuda \
    --dataset-root "${BWMAX_DATASET_ROOT}" \
    --index "${index}" \
    --study-id "${STUDY_ID}" \
    --evidence-class ordinary_mnist_bounded_wmax_sensitivity \
    --target "${BWMAX_TARGET}" \
    --summary-json "${SUMMARY_PATH}" \
    "${MODE_ARGS[@]}"

  VALIDATION_PATH="${VALIDATION_ROOT}/${index}.json"
  "${BWMAX_PYTHON}" -m experiments.validate_bounded_wmax_run \
    --config-root "${CONFIG_ROOT}" \
    --architecture "${BWMAX_ARCHITECTURE}" \
    --expected-config-set-sha256 "${BWMAX_CONFIG_SET_SHA256}" \
    --summary-json "${SUMMARY_PATH}" \
    --index "${index}" \
    --source-archive-sha256 "${EXPERIMENT_SOURCE_ARCHIVE_SHA256}" \
    "${VALIDATOR_MODE_ARGS[@]}" \
    --output "${VALIDATION_PATH}"

  RUN_DIR="$("${BWMAX_PYTHON}" - "${VALIDATION_PATH}" <<'PY'
import json
import sys
from pathlib import Path

print(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["run_dir"])
PY
)"
  "${BWMAX_PYTHON}" -m experiments.reporting validate-run "${RUN_DIR}"
  echo "BWMAX_SEMANTIC_PASS architecture=${BWMAX_ARCHITECTURE} index=${index} mode=${BWMAX_RUN_MODE}"
done
