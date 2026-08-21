#!/bin/bash -l
#SBATCH --job-name=pd-c2-wmax-fixed
#SBATCH --account=umg@v100
#SBATCH --partition=gpu_p13
#SBATCH --qos=qos_gpu-t3
#SBATCH --constraint=v100-16g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=01:00:00
#SBATCH --array=0-23%24

set -euo pipefail

: "${WMAX_SWEEP_SOURCE_ROOT:?Set the frozen staged source directory.}"
: "${WMAX_SWEEP_RESULT_ROOT:?Set the remote study result directory.}"
: "${WMAX_SWEEP_DATASET_ROOT:?Set the remote ordinary-MNIST dataset root.}"
: "${WMAX_SWEEP_CONFIG_SET_SHA256:?Set the ordered config-set SHA-256.}"
: "${EXPERIMENT_SOURCE_COMMIT:?Set the source commit recorded for this snapshot.}"
: "${EXPERIMENT_SOURCE_ARCHIVE_SHA256:?Set the frozen source archive SHA-256.}"
: "${SLURM_ARRAY_TASK_ID:?Submit this wrapper as a twenty-four-task Slurm array.}"

STUDY_ID="perfectdiode-conv2-fixed-uniform-init-wmax-sweep-sgd-adam-10ep-seed0-20260813-v1"
EVIDENCE_CLASS="ordinary_mnist_weight_ceiling_sensitivity_exploratory"
WMAX_SWEEP_SOURCE_ARCHIVE="${WMAX_SWEEP_SOURCE_ROOT}.tar.gz"
WMAX_SWEEP_CANARY_SMOKE="${WMAX_SWEEP_CANARY_SMOKE:-0}"
CONFIG_ROOT="${WMAX_SWEEP_SOURCE_ROOT}/configs/conv/${STUDY_ID}"
MANIFEST="${CONFIG_ROOT}/study_manifest.json"
CONFIGS=(
  "${CONFIG_ROOT}/00_wmax_1em4_baseline_sgd.json"
  "${CONFIG_ROOT}/01_wmax_1em4_baseline_adam.json"
  "${CONFIG_ROOT}/02_wmax_1em4_ours_sgd.json"
  "${CONFIG_ROOT}/03_wmax_1em4_ours_adam.json"
  "${CONFIG_ROOT}/04_wmax_1em4_legacy_sgd.json"
  "${CONFIG_ROOT}/05_wmax_1em4_legacy_adam.json"
  "${CONFIG_ROOT}/06_wmax_2em4_baseline_sgd.json"
  "${CONFIG_ROOT}/07_wmax_2em4_baseline_adam.json"
  "${CONFIG_ROOT}/08_wmax_2em4_ours_sgd.json"
  "${CONFIG_ROOT}/09_wmax_2em4_ours_adam.json"
  "${CONFIG_ROOT}/10_wmax_2em4_legacy_sgd.json"
  "${CONFIG_ROOT}/11_wmax_2em4_legacy_adam.json"
  "${CONFIG_ROOT}/12_wmax_5em4_baseline_sgd.json"
  "${CONFIG_ROOT}/13_wmax_5em4_baseline_adam.json"
  "${CONFIG_ROOT}/14_wmax_5em4_ours_sgd.json"
  "${CONFIG_ROOT}/15_wmax_5em4_ours_adam.json"
  "${CONFIG_ROOT}/16_wmax_5em4_legacy_sgd.json"
  "${CONFIG_ROOT}/17_wmax_5em4_legacy_adam.json"
  "${CONFIG_ROOT}/18_wmax_1em3_baseline_sgd.json"
  "${CONFIG_ROOT}/19_wmax_1em3_baseline_adam.json"
  "${CONFIG_ROOT}/20_wmax_1em3_ours_sgd.json"
  "${CONFIG_ROOT}/21_wmax_1em3_ours_adam.json"
  "${CONFIG_ROOT}/22_wmax_1em3_legacy_sgd.json"
  "${CONFIG_ROOT}/23_wmax_1em3_legacy_adam.json"
)
INITIALIZER="${WMAX_SWEEP_SOURCE_ROOT}/results/${STUDY_ID}/assets/bounded_uniform/conv2/final_model.pt"
INITIALIZER_SHA256="6469bc8f465e9abd2b4720231ad03d0116fc176ef4e437dc005027204c4d88d2"

if [[ ! "${SLURM_ARRAY_TASK_ID}" =~ ^[0-9]+$ ]]; then
  echo "SLURM_ARRAY_TASK_ID must be a non-negative integer." >&2
  exit 3
fi
TASK_INDEX=$((10#${SLURM_ARRAY_TASK_ID}))
if (( TASK_INDEX < 0 || TASK_INDEX >= ${#CONFIGS[@]} )); then
  echo "Array task ${TASK_INDEX} is outside [0,23]." >&2
  exit 3
fi
if [[ "${#CONFIGS[@]}" -ne 24 ]]; then
  echo "Expected exactly twenty-four configs." >&2
  exit 3
fi
if [[ ! "${WMAX_SWEEP_CANARY_SMOKE}" =~ ^[01]$ ]]; then
  echo "WMAX_SWEEP_CANARY_SMOKE must be 0 or 1." >&2
  exit 3
fi
if [[ ! "${EXPERIMENT_SOURCE_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "EXPERIMENT_SOURCE_COMMIT must be a forty-character lowercase Git identity." >&2
  exit 3
fi
for digest in "${WMAX_SWEEP_CONFIG_SET_SHA256}" "${EXPERIMENT_SOURCE_ARCHIVE_SHA256}"; do
  if [[ ! "${digest}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "Expected lowercase hexadecimal SHA-256, got ${digest}." >&2
    exit 3
  fi
done
if [[ ! -d "${WMAX_SWEEP_SOURCE_ROOT}" || ! -s "${WMAX_SWEEP_SOURCE_ARCHIVE}" ]]; then
  echo "Frozen source directory or archive is missing." >&2
  exit 3
fi
if [[ ! -s "${MANIFEST}" || ! -s "${INITIALIZER}" ]]; then
  echo "Study manifest or shared initializer is missing." >&2
  exit 3
fi
for config in "${CONFIGS[@]}"; do
  if [[ ! -s "${config}" ]]; then
    echo "Missing exact config: ${config}." >&2
    exit 3
  fi
done

actual_archive_sha256="$(sha256sum "${WMAX_SWEEP_SOURCE_ARCHIVE}" | awk '{print $1}')"
if [[ "${actual_archive_sha256}" != "${EXPERIMENT_SOURCE_ARCHIVE_SHA256}" ]]; then
  echo "Source archive SHA-256 mismatch." >&2
  exit 3
fi
actual_config_set_sha256="$({
  for config in "${CONFIGS[@]}"; do
    sha256sum "${config}" | awk '{print $1}'
  done
} | sha256sum | awk '{print $1}')"
if [[ "${actual_config_set_sha256}" != "${WMAX_SWEEP_CONFIG_SET_SHA256}" ]]; then
  echo "Config-set SHA-256 mismatch." >&2
  exit 3
fi
if [[ "$(sha256sum "${INITIALIZER}" | awk '{print $1}')" != "${INITIALIZER_SHA256}" ]]; then
  echo "Shared initializer SHA-256 mismatch." >&2
  exit 3
fi

source_root_canonical="$(realpath -e "${WMAX_SWEEP_SOURCE_ROOT}")"
result_root_canonical="$(realpath -m "${WMAX_SWEEP_RESULT_ROOT}")"
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
eval "$("${MODULECMD}" bash load pytorch-gpu/py3/2.5.0)"

export KMP_DISABLE_SHM=1
export KMP_SHM_DISABLE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

RUN_ROOT="${WMAX_SWEEP_RESULT_ROOT}/runs"
SUMMARY_ROOT="${WMAX_SWEEP_RESULT_ROOT}/task_summaries/production"
mkdir -p "${RUN_ROOT}" "${SUMMARY_ROOT}"
cd "${WMAX_SWEEP_SOURCE_ROOT}"

SUMMARY_PATH="${SUMMARY_ROOT}/${TASK_INDEX}.json"
EXACT_RUN_MODE=()
if [[ "${WMAX_SWEEP_CANARY_SMOKE}" == "1" ]]; then
  EXACT_RUN_MODE+=(--smoke)
fi
python -m experiments.exact_run \
  "${CONFIGS[@]}" \
  --output-root "${RUN_ROOT}" \
  --device cuda \
  --dataset-root "${WMAX_SWEEP_DATASET_ROOT}" \
  --study-id "${STUDY_ID}" \
  --evidence-class "${EVIDENCE_CLASS}" \
  --target jean-zay \
  --summary-json "${SUMMARY_PATH}" \
  --checkpoint-every-epoch \
  --skip-terminal-official-test \
  "${EXACT_RUN_MODE[@]}"

RUN_DIR="$(python -c 'import json,sys; rows=json.load(open(sys.argv[1])); assert len(rows)==1 and rows[0]["status"]=="complete"; print(rows[0]["output_dir"])' "${SUMMARY_PATH}")"
python -m experiments.reporting validate-run "${RUN_DIR}" > "${SUMMARY_ROOT}/${TASK_INDEX}.validate.txt"
test -s "${RUN_DIR}/result.json"
echo "WMAX_FIXED_INIT_SEMANTIC_PASS index=${TASK_INDEX} smoke=${WMAX_SWEEP_CANARY_SMOKE} run_dir=${RUN_DIR}"
