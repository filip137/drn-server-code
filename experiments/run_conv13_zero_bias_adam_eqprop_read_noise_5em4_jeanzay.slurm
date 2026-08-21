#!/bin/bash -l
#SBATCH --job-name=pd-c13-rn5e4
#SBATCH --account=umg@v100
#SBATCH --partition=gpu_p13
#SBATCH --qos=qos_gpu-t3
#SBATCH --constraint=v100-32g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --time=08:00:00
#SBATCH --array=0-2%3
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err

set -euo pipefail

: "${C13RN_SOURCE_ROOT:?Set C13RN_SOURCE_ROOT to the frozen staged source.}"
: "${C13RN_SOURCE_ARCHIVE:?Set C13RN_SOURCE_ARCHIVE to the frozen source archive.}"
: "${C13RN_CONFIG_ROOT:?Set C13RN_CONFIG_ROOT to the six materialized exact configs.}"
: "${C13RN_RESULT_ROOT:?Set C13RN_RESULT_ROOT to the Jean Zay result root.}"
: "${C13RN_DATASET_ROOT:?Set C13RN_DATASET_ROOT to the Jean Zay MNIST root.}"
: "${C13RN_ENVIRONMENT_ID:?Set C13RN_ENVIRONMENT_ID.}"
: "${C13RN_STUDY_CONFIG_SHA256:?Set C13RN_STUDY_CONFIG_SHA256.}"
: "${C13RN_CONFIG_SET_SHA256:?Set C13RN_CONFIG_SET_SHA256.}"
: "${C13RN_RUN_MODE:?Set C13RN_RUN_MODE to canary or production.}"
: "${EXPERIMENT_SOURCE_COMMIT:?Set EXPERIMENT_SOURCE_COMMIT.}"
: "${EXPERIMENT_SOURCE_ARCHIVE_SHA256:?Set EXPERIMENT_SOURCE_ARCHIVE_SHA256.}"
: "${SLURM_JOB_ID:?Submit this wrapper through Slurm.}"
: "${SLURM_ARRAY_TASK_ID:?Submit this wrapper as an array.}"

STUDY_ID="perfectdiode-conv13-zero-bias-adam-centered-float64-eqprop-one-decade-sigma-5em4-ordinary-mnist-10-30ep-seed0-20260817-v1"
EVIDENCE_CLASS="ordinary_mnist_eqprop_zero_bias_read_noise_training_diagnostic"
STUDY_CONFIG_RELATIVE="configs/conv/perfectdiode_conv13_zero_bias_adam_eqprop_one_decade_sigma_5em4_ordinary_mnist_10_30ep_seed0_20260817_v1.json"
STUDY_CONFIG="${C13RN_SOURCE_ROOT}/${STUDY_CONFIG_RELATIVE}"
MODULE_ID="pytorch-gpu/py3/2.5.0"
TASK_ID=$((10#${SLURM_ARRAY_TASK_ID}))
PARENT_JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}"

case "${C13RN_RUN_MODE}" in
  canary)
    EXPECTED_TASK_COUNT=1
    MAX_TASK_ID=0
    RUN_ROOT="${C13RN_RESULT_ROOT}/canary/${PARENT_JOB_ID}/runs"
    RUN_SUMMARY_ROOT="${C13RN_RESULT_ROOT}/task_summaries/canary/${PARENT_JOB_ID}/runs"
    PACK_SUMMARY_ROOT="${C13RN_RESULT_ROOT}/task_summaries/canary/${PARENT_JOB_ID}/packs"
    RUN_RECEIPT_ROOT="${C13RN_RESULT_ROOT}/transport_receipts/canary/${PARENT_JOB_ID}/runs"
    PACK_RECEIPT_ROOT="${C13RN_RESULT_ROOT}/transport_receipts/canary/${PARENT_JOB_ID}/packs"
    LOG_ROOT="${C13RN_RESULT_ROOT}/packed_logs/canary/${PARENT_JOB_ID}"
    LOCK_ROOT="${C13RN_RESULT_ROOT}/task_locks/canary/${PARENT_JOB_ID}"
    EXACT_MODE_ARGS=(--smoke)
    VALIDATOR_RUN_MODE="smoke"
    ;;
  production)
    EXPECTED_TASK_COUNT=3
    MAX_TASK_ID=2
    RUN_ROOT="${C13RN_RESULT_ROOT}/production"
    RUN_SUMMARY_ROOT="${C13RN_RESULT_ROOT}/task_summaries/production/runs"
    PACK_SUMMARY_ROOT="${C13RN_RESULT_ROOT}/task_summaries/production/packs"
    RUN_RECEIPT_ROOT="${C13RN_RESULT_ROOT}/transport_receipts/production/runs"
    PACK_RECEIPT_ROOT="${C13RN_RESULT_ROOT}/transport_receipts/production/packs"
    LOG_ROOT="${C13RN_RESULT_ROOT}/packed_logs/production"
    LOCK_ROOT="${C13RN_RESULT_ROOT}/task_locks/production"
    EXACT_MODE_ARGS=()
    VALIDATOR_RUN_MODE="production"
    ;;
  *)
    echo "C13RN_RUN_MODE must be canary or production." >&2
    exit 3
    ;;
esac

if (( TASK_ID < 0 || TASK_ID > MAX_TASK_ID )); then
  echo "Array task ${TASK_ID} is outside [0,${MAX_TASK_ID}] for ${C13RN_RUN_MODE}." >&2
  exit 3
fi
if [[ -n "${SLURM_ARRAY_TASK_COUNT:-}" ]] \
   && (( 10#${SLURM_ARRAY_TASK_COUNT} != EXPECTED_TASK_COUNT )); then
  echo "Expected ${EXPECTED_TASK_COUNT} array tasks; got ${SLURM_ARRAY_TASK_COUNT}." >&2
  exit 3
fi
if [[ ! "${EXPERIMENT_SOURCE_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "EXPERIMENT_SOURCE_COMMIT must be a lowercase 40-character SHA." >&2
  exit 3
fi
for digest_name in \
  EXPERIMENT_SOURCE_ARCHIVE_SHA256 C13RN_STUDY_CONFIG_SHA256 C13RN_CONFIG_SET_SHA256
do
  digest="${!digest_name}"
  if [[ ! "${digest}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "${digest_name} must be a lowercase SHA-256." >&2
    exit 3
  fi
done
if [[ ! -d "${C13RN_SOURCE_ROOT}" || ! -s "${C13RN_SOURCE_ARCHIVE}" ]]; then
  echo "Frozen source directory or archive is missing." >&2
  exit 3
fi
if [[ ! -d "${C13RN_CONFIG_ROOT}" || ! -s "${STUDY_CONFIG}" ]]; then
  echo "Exact-config root or study config is missing." >&2
  exit 3
fi
if [[ ! -d "${C13RN_DATASET_ROOT}" ]]; then
  echo "MNIST dataset root is missing: ${C13RN_DATASET_ROOT}." >&2
  exit 3
fi

mapfile -t CONFIGS < <(find "${C13RN_CONFIG_ROOT}" -maxdepth 1 -type f -name '*.json' | sort)
if [[ "${#CONFIGS[@]}" -ne 6 ]]; then
  echo "Expected six materialized exact configs; found ${#CONFIGS[@]}." >&2
  exit 3
fi

archive_sha="$(sha256sum -- "${C13RN_SOURCE_ARCHIVE}")"
archive_sha="${archive_sha%% *}"
study_config_sha="$(sha256sum -- "${STUDY_CONFIG}")"
study_config_sha="${study_config_sha%% *}"
config_set_sha="$({
  for config in "${CONFIGS[@]}"; do
    sha256sum -- "${config}" | awk '{print $1}'
  done
} | sha256sum | awk '{print $1}')"
if [[ "${archive_sha}" != "${EXPERIMENT_SOURCE_ARCHIVE_SHA256}" ]]; then
  echo "Frozen source archive SHA mismatch." >&2
  exit 3
fi
if [[ "${study_config_sha}" != "${C13RN_STUDY_CONFIG_SHA256}" ]]; then
  echo "Study config SHA mismatch." >&2
  exit 3
fi
if [[ "${config_set_sha}" != "${C13RN_CONFIG_SET_SHA256}" ]]; then
  echo "Ordered exact-config set SHA mismatch." >&2
  exit 3
fi

source_root_canonical="$(realpath -e -- "${C13RN_SOURCE_ROOT}")"
result_root_canonical="$(realpath -m -- "${C13RN_RESULT_ROOT}")"
if [[ "${source_root_canonical}" == "${result_root_canonical}" \
   || "${source_root_canonical}" == "${result_root_canonical}/"* \
   || "${result_root_canonical}" == "${source_root_canonical}/"* ]]; then
  echo "Frozen source and result roots must be disjoint." >&2
  exit 3
fi

mkdir -p "${RUN_ROOT}" "${RUN_SUMMARY_ROOT}" "${PACK_SUMMARY_ROOT}" \
  "${RUN_RECEIPT_ROOT}" "${PACK_RECEIPT_ROOT}" "${LOG_ROOT}" "${LOCK_ROOT}"

PACK_SUMMARY_PATH="${PACK_SUMMARY_ROOT}/${TASK_ID}.json"
PACK_RECEIPT_PATH="${PACK_RECEIPT_ROOT}/${TASK_ID}.json"
LOCK_PATH="${LOCK_ROOT}/${TASK_ID}.lock"
if [[ -e "${PACK_SUMMARY_PATH}" || -e "${PACK_RECEIPT_PATH}" ]]; then
  echo "Refusing duplicate ${C13RN_RUN_MODE} task ${TASK_ID}: terminal path exists." >&2
  exit 17
fi
if ! mkdir "${LOCK_PATH}"; then
  echo "Refusing concurrent duplicate ${C13RN_RUN_MODE} task ${TASK_ID}." >&2
  exit 17
fi
cleanup_lock() {
  rmdir "${LOCK_PATH}" 2>/dev/null || true
}
trap cleanup_lock EXIT

PACK_MAP=("0 1" "2 3" "4 5")
read -r -a LOGICAL_INDICES <<< "${PACK_MAP[TASK_ID]}"
for logical_index in "${LOGICAL_INDICES[@]}"; do
  if [[ -e "${RUN_SUMMARY_ROOT}/${logical_index}.json" \
     || -e "${RUN_RECEIPT_ROOT}/${logical_index}.json" ]]; then
    echo "Refusing duplicate logical run ${logical_index}: terminal path exists." >&2
    exit 17
  fi
done

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
export PYTHONPATH="${C13RN_SOURCE_ROOT}:${PYTHONPATH:-}"

logical_csv="$(IFS=,; echo "${LOGICAL_INDICES[*]}")"
cd "${C13RN_SOURCE_ROOT}"
"${PYTHON_BIN}" - \
  "${STUDY_CONFIG}" "${C13RN_CONFIG_ROOT}" "${TASK_ID}" \
  "${logical_csv}" "${C13RN_DATASET_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

import torch

from experiments.run_conv13_zero_bias_adam_eqprop_read_noise_5em4 import (
    CASE_SPECS,
    NOISE_STD,
    PACKS,
    load_and_validate_study,
)

study_arg, config_root_arg, pack_arg, logical_csv_arg, dataset_arg = sys.argv[1:]
study = load_and_validate_study(Path(study_arg))
pack_index = int(pack_arg)
logical_indices = tuple(int(value) for value in logical_csv_arg.split(","))
if PACKS[pack_index] != logical_indices:
    raise SystemExit("Packed logical-index mapping changed.")
configs = sorted(Path(config_root_arg).glob("*.json"))
if len(configs) != len(CASE_SPECS):
    raise SystemExit("Expected exactly six materialized configs.")
contract = study["scientific_contract"]
for logical_index in logical_indices:
    config = json.loads(configs[logical_index].read_text(encoding="utf-8"))
    spec = CASE_SPECS[logical_index]
    if config.get("study_id") != study.get("study_id") or config.get("arm_id") != spec.arm_id:
        raise SystemExit(f"Config/logical-index mapping mismatch at {logical_index}.")
    if config.get("training_algorithm") != "EP" or config.get("runtime_dtype") != "float64":
        raise SystemExit("Training algorithm or runtime dtype changed.")
    if config.get("optimizer", {}).get("name") != "Adam":
        raise SystemExit("Optimizer/config mapping changed.")
    eqprop = config.get("eqprop", {})
    if float(eqprop.get("endpoint_read_noise_std", -1.0)) != NOISE_STD:
        raise SystemExit("Endpoint read-noise sigma changed.")
    if int(eqprop.get("endpoint_read_noise_seed", -1)) != int(contract["endpoint_read_noise_seed"]):
        raise SystemExit("Endpoint read-noise seed changed.")
    if eqprop.get("input_read_noise") is not False:
        raise SystemExit("Input read noise became enabled.")
    model = config.get("model_base", {})
    if int(model.get("num_iterations_inference", -1)) != int(contract["T"][spec.architecture]):
        raise SystemExit("T changed.")
    if int(model.get("num_iterations_training", -1)) != int(contract["K"][spec.architecture]):
        raise SystemExit("K changed.")
    if int(config.get("lab", {}).get("epochs", -1)) != int(contract["epochs"][spec.architecture]):
        raise SystemExit("Epoch budget changed.")
    if config.get("evaluation", {}).get("official_test", {}).get("policy") != "disabled":
        raise SystemExit("Official-test policy is not disabled.")
    for name in config.get("parameter_order", []):
        if name.startswith("Bias_") and float(config["learning_rates_by_parameter"][name]) != 0.0:
            raise SystemExit(f"Bias learning rate changed: {name}.")
dataset_root = Path(dataset_arg)
for path in (
    dataset_root / "MNIST/raw/train-images-idx3-ubyte",
    dataset_root / "MNIST/raw/train-labels-idx1-ubyte",
):
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
    f"C13RN_ENVIRONMENT_PASS pack={pack_index} logical={logical_indices} "
    f"python={sys.executable} torch={torch.__version__} "
    f"device={device_name!r} bytes={device_bytes}",
    flush=True,
)
PY

echo "C13RN_SOURCE_PASS mode=${C13RN_RUN_MODE} task=${TASK_ID} logical=${logical_csv} archive_sha256=${archive_sha} study_config_sha256=${study_config_sha} config_set_sha256=${config_set_sha}"

pids=()
for logical_index in "${LOGICAL_INDICES[@]}"; do
  summary_path="${RUN_SUMMARY_ROOT}/${logical_index}.json"
  log_path="${LOG_ROOT}/${logical_index}.log"
  "${PYTHON_BIN}" -m experiments.exact_run \
    "${CONFIGS[@]}" \
    --index "${logical_index}" \
    --output-root "${RUN_ROOT}" \
    --device cuda \
    --dataset-root "${C13RN_DATASET_ROOT}" \
    --study-id "${STUDY_ID}" \
    --evidence-class "${EVIDENCE_CLASS}" \
    --target jean-zay \
    --summary-json "${summary_path}" \
    --skip-terminal-official-test \
    "${EXACT_MODE_ARGS[@]}" \
    >"${log_path}" 2>&1 &
  child_pid="$!"
  pids+=("${child_pid}")
  echo "C13RN_LAUNCHED mode=${C13RN_RUN_MODE} logical=${logical_index} pid=${child_pid} log=${log_path}"
done

child_statuses=()
set +e
for child_pid in "${pids[@]}"; do
  wait "${child_pid}"
  child_statuses+=("$?")
done
set -e

"${PYTHON_BIN}" -m experiments.validate_conv13_zero_bias_adam_eqprop_read_noise_5em4_pack \
  --study "${STUDY_CONFIG}" \
  --config-root "${C13RN_CONFIG_ROOT}" \
  --run-root "${RUN_ROOT}" \
  --summary-root "${RUN_SUMMARY_ROOT}" \
  --log-root "${LOG_ROOT}" \
  --receipt-root "${RUN_RECEIPT_ROOT}" \
  --pack-receipt "${PACK_RECEIPT_PATH}" \
  --semantic-summary "${PACK_SUMMARY_PATH}" \
  --pack-index "${TASK_ID}" \
  --logical-indices "${LOGICAL_INDICES[@]}" \
  --run-mode "${VALIDATOR_RUN_MODE}" \
  --execution-mode concurrent \
  --job-id "${SLURM_JOB_ID}" \
  --parent-job-id "${PARENT_JOB_ID}" \
  --environment-id "${C13RN_ENVIRONMENT_ID}" \
  --source-commit "${EXPERIMENT_SOURCE_COMMIT}" \
  --source-archive-sha256 "${EXPERIMENT_SOURCE_ARCHIVE_SHA256}" \
  --config-set-sha256 "${C13RN_CONFIG_SET_SHA256}"

echo "C13RN_SEMANTIC_PASS mode=${C13RN_RUN_MODE} task=${TASK_ID} logical=${logical_csv} child_statuses=${child_statuses[*]} summary=${PACK_SUMMARY_PATH}"
