#!/bin/bash -l
#SBATCH --job-name=pd-c12-kai-adam
#SBATCH --account=umg@v100
#SBATCH --partition=gpu_p13
#SBATCH --qos=qos_gpu-dev
#SBATCH --constraint=v100-16g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=01:00:00
#SBATCH --array=0-5%6

set -euo pipefail

: "${KAIMING_TRANSFER_SOURCE_ROOT:?Set the frozen staged source directory.}"
: "${KAIMING_TRANSFER_RESULT_ROOT:?Set the remote study result directory.}"
: "${KAIMING_TRANSFER_DATASET_ROOT:?Set the remote ordinary-MNIST dataset root.}"
: "${KAIMING_TRANSFER_CONFIG_SET_SHA256:?Set the frozen ordered config-set SHA-256.}"
: "${EXPERIMENT_SOURCE_COMMIT:?Set the source commit recorded for this snapshot.}"
: "${EXPERIMENT_SOURCE_ARCHIVE_SHA256:?Set the frozen source archive SHA-256.}"
: "${SLURM_ARRAY_TASK_ID:?Submit this wrapper as a six-task Slurm array.}"

STUDY_ID="perfectdiode-conv12-bounded-kaiming-adam-lr-transfer-3ep-seed0-20260813-v1"
EVIDENCE_CLASS="ordinary_mnist_initializer_transfer_exploratory"
CONFIG_ROOT="${KAIMING_TRANSFER_SOURCE_ROOT}/configs/conv/${STUDY_ID}"
MANIFEST="${CONFIG_ROOT}/study_manifest.json"
CONFIGS=(
  "${CONFIG_ROOT}/conv1_baseline_adam.json"
  "${CONFIG_ROOT}/conv1_ours_adam.json"
  "${CONFIG_ROOT}/conv1_legacy_adam.json"
  "${CONFIG_ROOT}/conv2_baseline_adam.json"
  "${CONFIG_ROOT}/conv2_ours_adam.json"
  "${CONFIG_ROOT}/conv2_legacy_adam.json"
)
CONV1_ASSET="${KAIMING_TRANSFER_SOURCE_ROOT}/results/${STUDY_ID}/assets/bounded_kaiming_uniform/conv1/final_model.pt"
CONV2_ASSET="${KAIMING_TRANSFER_SOURCE_ROOT}/results/${STUDY_ID}/assets/bounded_kaiming_uniform/conv2/final_model.pt"
CONV1_ASSET_SHA256="d603b26c86a622263572705dd693987e5af24da78e38e629256097d603c2a646"
CONV2_ASSET_SHA256="8ecf6240d572619308203301f7c4b462f7224fa7b9913778588fdadd2594d21b"

if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ${#CONFIGS[@]} )); then
  echo "Array task ${SLURM_ARRAY_TASK_ID} is outside [0,5]." >&2
  exit 3
fi
if [[ ! -s "${MANIFEST}" ]]; then
  echo "Missing study manifest: ${MANIFEST}" >&2
  exit 3
fi
for config in "${CONFIGS[@]}"; do
  if [[ ! -s "${config}" ]]; then
    echo "Missing exact config: ${config}" >&2
    exit 3
  fi
done

actual_config_set_sha256="$({
  for config in "${CONFIGS[@]}"; do
    sha256sum "${config}" | awk '{print $1}'
  done
} | sha256sum | awk '{print $1}')"
if [[ "${actual_config_set_sha256}" != "${KAIMING_TRANSFER_CONFIG_SET_SHA256}" ]]; then
  echo "Config-set digest mismatch: expected ${KAIMING_TRANSFER_CONFIG_SET_SHA256}, got ${actual_config_set_sha256}." >&2
  exit 3
fi
if [[ "$(sha256sum "${CONV1_ASSET}" | awk '{print $1}')" != "${CONV1_ASSET_SHA256}" ]]; then
  echo "Conv1 initializer checkpoint digest mismatch." >&2
  exit 3
fi
if [[ "$(sha256sum "${CONV2_ASSET}" | awk '{print $1}')" != "${CONV2_ASSET_SHA256}" ]]; then
  echo "Conv2 initializer checkpoint digest mismatch." >&2
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

RUN_ROOT="${KAIMING_TRANSFER_RESULT_ROOT}/runs"
SUMMARY_ROOT="${KAIMING_TRANSFER_RESULT_ROOT}/task_summaries/production"
mkdir -p "${RUN_ROOT}" "${SUMMARY_ROOT}"
cd "${KAIMING_TRANSFER_SOURCE_ROOT}"

SUMMARY_PATH="${SUMMARY_ROOT}/${SLURM_ARRAY_TASK_ID}.json"
python -m experiments.exact_run \
  "${CONFIGS[@]}" \
  --output-root "${RUN_ROOT}" \
  --device cuda \
  --dataset-root "${KAIMING_TRANSFER_DATASET_ROOT}" \
  --study-id "${STUDY_ID}" \
  --evidence-class "${EVIDENCE_CLASS}" \
  --target jean-zay \
  --summary-json "${SUMMARY_PATH}" \
  --checkpoint-every-epoch \
  --skip-terminal-official-test

RUN_DIR="$(python -c 'import json,sys; rows=json.load(open(sys.argv[1])); assert len(rows)==1 and rows[0]["status"]=="complete"; print(rows[0]["output_dir"])' "${SUMMARY_PATH}")"
python -m experiments.reporting validate-run "${RUN_DIR}" > "${SUMMARY_ROOT}/${SLURM_ARRAY_TASK_ID}.validate.txt"
test -s "${RUN_DIR}/result.json"
