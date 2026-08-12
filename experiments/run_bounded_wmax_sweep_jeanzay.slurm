#!/bin/bash -l
#SBATCH --job-name=pd-bwmax-c3
#SBATCH --account=fmu@v100
#SBATCH --partition=gpu_p13
#SBATCH --qos=qos_gpu-t3
#SBATCH --constraint=v100-32g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=10:00:00
#SBATCH --array=0-29%10

set -euo pipefail

: "${BWMAX_SOURCE_ROOT:?Set BWMAX_SOURCE_ROOT to the frozen staged source.}"
: "${BWMAX_SOURCE_ARCHIVE:?Set BWMAX_SOURCE_ARCHIVE to the staged source archive.}"
: "${BWMAX_RESULT_ROOT:?Set BWMAX_RESULT_ROOT to the study result directory.}"
: "${BWMAX_CONFIG_SET_SHA256:?Set BWMAX_CONFIG_SET_SHA256 to the Conv3 config-set digest.}"
: "${EXPERIMENT_SOURCE_COMMIT:?Set EXPERIMENT_SOURCE_COMMIT to the frozen parent source commit.}"
: "${EXPERIMENT_SOURCE_ARCHIVE_SHA256:?Set EXPERIMENT_SOURCE_ARCHIVE_SHA256 to the staged archive digest.}"
: "${SLURM_ARRAY_TASK_ID:?Submit this wrapper as a Slurm array.}"

STUDY_ID="perfectdiode-bounded-wmax-sweep-signed-scaled-bias-conv123-seed0-20260809-v1"
EXPECTED_SOURCE_COMMIT="08dd52e44818adf43b011542748e140dc0e01aeb"
CONFIG_ROOT="${BWMAX_SOURCE_ROOT}/configs/conv/perfectdiode_bounded_wmax_sweep_signed_scaled_bias_conv123_seed0_20260809_v1"
CONFIGS=(
  "${CONFIG_ROOT}/conv3/00_wmax_1em4_baseline_sgd_seed0.json"
  "${CONFIG_ROOT}/conv3/01_wmax_1em4_baseline_adam_seed0.json"
  "${CONFIG_ROOT}/conv3/02_wmax_1em4_ours_sgd_seed0.json"
  "${CONFIG_ROOT}/conv3/03_wmax_1em4_ours_adam_seed0.json"
  "${CONFIG_ROOT}/conv3/04_wmax_1em4_legacy_sgd_seed0.json"
  "${CONFIG_ROOT}/conv3/05_wmax_1em4_legacy_adam_seed0.json"
  "${CONFIG_ROOT}/conv3/06_wmax_3em4_baseline_sgd_seed0.json"
  "${CONFIG_ROOT}/conv3/07_wmax_3em4_baseline_adam_seed0.json"
  "${CONFIG_ROOT}/conv3/08_wmax_3em4_ours_sgd_seed0.json"
  "${CONFIG_ROOT}/conv3/09_wmax_3em4_ours_adam_seed0.json"
  "${CONFIG_ROOT}/conv3/10_wmax_3em4_legacy_sgd_seed0.json"
  "${CONFIG_ROOT}/conv3/11_wmax_3em4_legacy_adam_seed0.json"
  "${CONFIG_ROOT}/conv3/12_wmax_1em3_baseline_sgd_seed0.json"
  "${CONFIG_ROOT}/conv3/13_wmax_1em3_baseline_adam_seed0.json"
  "${CONFIG_ROOT}/conv3/14_wmax_1em3_ours_sgd_seed0.json"
  "${CONFIG_ROOT}/conv3/15_wmax_1em3_ours_adam_seed0.json"
  "${CONFIG_ROOT}/conv3/16_wmax_1em3_legacy_sgd_seed0.json"
  "${CONFIG_ROOT}/conv3/17_wmax_1em3_legacy_adam_seed0.json"
  "${CONFIG_ROOT}/conv3/18_wmax_3em3_baseline_sgd_seed0.json"
  "${CONFIG_ROOT}/conv3/19_wmax_3em3_baseline_adam_seed0.json"
  "${CONFIG_ROOT}/conv3/20_wmax_3em3_ours_sgd_seed0.json"
  "${CONFIG_ROOT}/conv3/21_wmax_3em3_ours_adam_seed0.json"
  "${CONFIG_ROOT}/conv3/22_wmax_3em3_legacy_sgd_seed0.json"
  "${CONFIG_ROOT}/conv3/23_wmax_3em3_legacy_adam_seed0.json"
  "${CONFIG_ROOT}/conv3/24_wmax_1em2_baseline_sgd_seed0.json"
  "${CONFIG_ROOT}/conv3/25_wmax_1em2_baseline_adam_seed0.json"
  "${CONFIG_ROOT}/conv3/26_wmax_1em2_ours_sgd_seed0.json"
  "${CONFIG_ROOT}/conv3/27_wmax_1em2_ours_adam_seed0.json"
  "${CONFIG_ROOT}/conv3/28_wmax_1em2_legacy_sgd_seed0.json"
  "${CONFIG_ROOT}/conv3/29_wmax_1em2_legacy_adam_seed0.json"
)

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
if [[ ! -s "${CONFIG_ROOT}/study_manifest.json" ]]; then
  echo "Missing or empty study manifest: ${CONFIG_ROOT}/study_manifest.json." >&2
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
if [[ "${#CONFIGS[@]}" -ne 30 ]]; then
  echo "Expected exactly thirty frozen Conv3 configs." >&2
  exit 3
fi
if [[ ! "${SLURM_ARRAY_TASK_ID}" =~ ^[0-9]+$ ]]; then
  echo "SLURM_ARRAY_TASK_ID must be a non-negative integer." >&2
  exit 3
fi
SLURM_ARRAY_TASK_ID=$((10#${SLURM_ARRAY_TASK_ID}))
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ${#CONFIGS[@]} )); then
  echo "Array task ${SLURM_ARRAY_TASK_ID} is outside [0,29]." >&2
  exit 3
fi
for config in "${CONFIGS[@]}"; do
  if [[ ! -s "${config}" ]]; then
    echo "Missing or empty exact config: ${config}." >&2
    exit 3
  fi
done

BWMAX_RUN_MODE="${BWMAX_RUN_MODE:-production}"
case "${BWMAX_RUN_MODE}" in
  production)
    MODE_ARGS=()
    VALIDATOR_MODE_ARGS=()
    RUN_ROOT="${BWMAX_RESULT_ROOT}/conv3/runs"
    SUMMARY_ROOT="${BWMAX_RESULT_ROOT}/conv3/task_summaries/production"
    VALIDATION_ROOT="${BWMAX_RESULT_ROOT}/conv3/validation/production"
    ;;
  smoke)
    MODE_ARGS=(--smoke)
    VALIDATOR_MODE_ARGS=(--smoke)
    RUN_ROOT="${BWMAX_RESULT_ROOT}/operational_canary/conv3"
    SUMMARY_ROOT="${BWMAX_RESULT_ROOT}/operational_canary/task_summaries"
    VALIDATION_ROOT="${BWMAX_RESULT_ROOT}/operational_canary/validation"
    ;;
  *)
    echo "BWMAX_RUN_MODE must be production or smoke; got ${BWMAX_RUN_MODE}." >&2
    exit 3
    ;;
esac

export MODULESHOME="/lustre/fshomisc/sup/spack_soft/environment-modules/4.3.1/gcc-11.3.1-wf7m7j6whgecysm2fm5n73sm4jg7txup"
export MODULEPATH="/lustre/fshomisc/sup/hpe/pub/module-rh/modulefiles:/lustre/fshomisc/sup/hpe/pub/modules-idris-env4/modulefiles/linux-rhel9-skylake_avx512"
MODULECMD="${MODULESHOME}/bin/modulecmd"
eval "$("${MODULECMD}" bash purge)"
eval "$("${MODULECMD}" bash load pytorch-gpu/py3/2.5.0)"

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

python -m experiments.validate_bounded_wmax_run \
  --config-root "${CONFIG_ROOT}" \
  --architecture conv3 \
  --expected-config-set-sha256 "${BWMAX_CONFIG_SET_SHA256}" \
  --output "${VALIDATION_ROOT}/config_set_${SLURM_ARRAY_TASK_ID}.json"

SUMMARY_PATH="${SUMMARY_ROOT}/${SLURM_ARRAY_TASK_ID}.json"
python -m experiments.exact_run \
  "${CONFIGS[@]}" \
  --output-root "${RUN_ROOT}" \
  --device cuda \
  --study-id "${STUDY_ID}" \
  --evidence-class ordinary_mnist_bounded_wmax_sensitivity \
  --target jean-zay \
  --summary-json "${SUMMARY_PATH}" \
  "${MODE_ARGS[@]}"

VALIDATION_PATH="${VALIDATION_ROOT}/${SLURM_ARRAY_TASK_ID}.json"
python -m experiments.validate_bounded_wmax_run \
  --config-root "${CONFIG_ROOT}" \
  --architecture conv3 \
  --expected-config-set-sha256 "${BWMAX_CONFIG_SET_SHA256}" \
  --summary-json "${SUMMARY_PATH}" \
  --index "${SLURM_ARRAY_TASK_ID}" \
  --source-archive-sha256 "${EXPERIMENT_SOURCE_ARCHIVE_SHA256}" \
  "${VALIDATOR_MODE_ARGS[@]}" \
  --output "${VALIDATION_PATH}"

RUN_DIR="$(python - "${VALIDATION_PATH}" <<'PY'
import json
import sys
from pathlib import Path

print(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["run_dir"])
PY
)"
python -m experiments.reporting validate-run "${RUN_DIR}"
echo "BWMAX_SEMANTIC_PASS architecture=conv3 index=${SLURM_ARRAY_TASK_ID} mode=${BWMAX_RUN_MODE}"
