#!/bin/bash -l
#SBATCH --job-name=pd-c2-rn-sgd
#SBATCH --account=fmu@v100
#SBATCH --partition=gpu_p13
#SBATCH --qos=qos_gpu-t3
#SBATCH --constraint=v100-16g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --hint=nomultithread
#SBATCH --time=03:00:00
#SBATCH --array=0-2%3
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err

set -euo pipefail

: "${C2RN_SOURCE_ROOT:?Set C2RN_SOURCE_ROOT to the frozen staged source.}"
: "${C2RN_SOURCE_ARCHIVE:?Set C2RN_SOURCE_ARCHIVE to the frozen source archive.}"
: "${C2RN_CONFIG_ROOT:?Set C2RN_CONFIG_ROOT to the materialized exact configs.}"
: "${C2RN_RESULT_ROOT:?Set C2RN_RESULT_ROOT to the Jean Zay result root.}"
: "${C2RN_DATASET_ROOT:?Set C2RN_DATASET_ROOT to the Jean Zay MNIST root.}"
: "${C2RN_ENVIRONMENT_ID:?Set C2RN_ENVIRONMENT_ID.}"
: "${C2RN_RUN_MODE:?Set C2RN_RUN_MODE to canary or production.}"
: "${C2RN_STUDY_CONFIG_SHA256:?Set C2RN_STUDY_CONFIG_SHA256.}"
: "${C2RN_CONFIG_SET_SHA256:?Set C2RN_CONFIG_SET_SHA256.}"
: "${EXPERIMENT_SOURCE_COMMIT:?Set EXPERIMENT_SOURCE_COMMIT.}"
: "${EXPERIMENT_SOURCE_ARCHIVE_SHA256:?Set EXPERIMENT_SOURCE_ARCHIVE_SHA256.}"
: "${SLURM_JOB_ID:?Submit this wrapper through Slurm.}"
: "${SLURM_ARRAY_TASK_ID:?Submit this wrapper as an array.}"

STUDY_ID="${C2RN_STUDY_ID:-perfectdiode-conv2-centered-float64-eqprop-read-noise-one-decade-sigma-5em4-sgd-10ep-seed0-20260816-v1}"
EVIDENCE_CLASS="${C2RN_EVIDENCE_CLASS:-ordinary_mnist_eqprop_read_noise_training_diagnostic}"
STUDY_CONFIG_RELATIVE="${C2RN_STUDY_CONFIG_RELATIVE:-configs/conv/perfectdiode_conv2_centered_float64_eqprop_read_noise_one_decade_sigma_5em4_sgd_10ep_seed0_20260816_v1.json}"
EXPECTED_CONFIGS="${C2RN_EXPECTED_CONFIGS:-3}"
STUDY_CONFIG="${C2RN_SOURCE_ROOT}/${STUDY_CONFIG_RELATIVE}"
MODULE_ID="pytorch-gpu/py3/2.5.0"
TASK_ID=$((10#${SLURM_ARRAY_TASK_ID}))
PARENT_JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}"

case "${C2RN_RUN_MODE}" in
  canary)
    EXPECTED_ARRAY_TASKS=1
    EXPECTED_EPOCHS=1
    EXPECTED_NOISE_DRAWS=6
    RUN_ROOT="${C2RN_RESULT_ROOT}/remote_canary"
    SUMMARY_ROOT="${C2RN_RESULT_ROOT}/task_summaries/remote_canary"
    RECEIPT_ROOT="${C2RN_RESULT_ROOT}/transport_receipts/remote_canary"
    MODE_ARGS=(--smoke)
    ;;
  production)
    EXPECTED_ARRAY_TASKS="${EXPECTED_CONFIGS}"
    EXPECTED_EPOCHS=10
    EXPECTED_NOISE_DRAWS=206280
    RUN_ROOT="${C2RN_RESULT_ROOT}/production"
    SUMMARY_ROOT="${C2RN_RESULT_ROOT}/task_summaries/production"
    RECEIPT_ROOT="${C2RN_RESULT_ROOT}/transport_receipts/production"
    MODE_ARGS=()
    ;;
  *)
    echo "C2RN_RUN_MODE must be canary or production; got ${C2RN_RUN_MODE}." >&2
    exit 3
    ;;
esac

if [[ ! "${EXPECTED_CONFIGS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "C2RN_EXPECTED_CONFIGS must be a positive integer; got ${EXPECTED_CONFIGS}." >&2
  exit 3
fi
MAX_TASK_ID=$((EXPECTED_CONFIGS - 1))
if (( TASK_ID < 0 || TASK_ID > MAX_TASK_ID )); then
  echo "Array task ${TASK_ID} is outside [0,${MAX_TASK_ID}]." >&2
  exit 3
fi
if [[ -n "${SLURM_ARRAY_TASK_COUNT:-}" ]] \
   && (( 10#${SLURM_ARRAY_TASK_COUNT} != EXPECTED_ARRAY_TASKS )); then
  echo "Expected ${EXPECTED_ARRAY_TASKS} array task(s); got ${SLURM_ARRAY_TASK_COUNT}." >&2
  exit 3
fi
if [[ ! "${EXPERIMENT_SOURCE_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "EXPERIMENT_SOURCE_COMMIT must be a lowercase 40-character SHA." >&2
  exit 3
fi
for digest_name in \
  EXPERIMENT_SOURCE_ARCHIVE_SHA256 C2RN_STUDY_CONFIG_SHA256 C2RN_CONFIG_SET_SHA256
do
  digest="${!digest_name}"
  if [[ ! "${digest}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "${digest_name} must be a lowercase SHA-256." >&2
    exit 3
  fi
done
if [[ ! -d "${C2RN_SOURCE_ROOT}" || ! -s "${C2RN_SOURCE_ARCHIVE}" ]]; then
  echo "Frozen source directory or archive is missing." >&2
  exit 3
fi
if [[ ! -d "${C2RN_CONFIG_ROOT}" || ! -s "${STUDY_CONFIG}" ]]; then
  echo "Exact-config root or study config is missing." >&2
  exit 3
fi
if [[ ! -d "${C2RN_DATASET_ROOT}" ]]; then
  echo "MNIST dataset root is missing: ${C2RN_DATASET_ROOT}." >&2
  exit 3
fi

mapfile -t CONFIGS < <(find "${C2RN_CONFIG_ROOT}" -maxdepth 1 -type f -name '*.json' | sort)
if [[ "${#CONFIGS[@]}" -ne "${EXPECTED_CONFIGS}" ]]; then
  echo "Expected ${EXPECTED_CONFIGS} materialized exact configs; found ${#CONFIGS[@]}." >&2
  exit 3
fi

archive_sha="$(sha256sum -- "${C2RN_SOURCE_ARCHIVE}")"
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
if [[ "${study_config_sha}" != "${C2RN_STUDY_CONFIG_SHA256}" ]]; then
  echo "Study config SHA mismatch." >&2
  exit 3
fi
if [[ "${config_set_sha}" != "${C2RN_CONFIG_SET_SHA256}" ]]; then
  echo "Ordered exact-config set SHA mismatch." >&2
  exit 3
fi

source_root_canonical="$(realpath -e -- "${C2RN_SOURCE_ROOT}")"
result_root_canonical="$(realpath -m -- "${C2RN_RESULT_ROOT}")"
if [[ "${source_root_canonical}" == "${result_root_canonical}" \
   || "${source_root_canonical}" == "${result_root_canonical}/"* \
   || "${result_root_canonical}" == "${source_root_canonical}/"* ]]; then
  echo "Frozen source and result roots must be disjoint." >&2
  exit 3
fi

mkdir -p "${RUN_ROOT}" "${SUMMARY_ROOT}" "${RECEIPT_ROOT}" \
  "${C2RN_RESULT_ROOT}/task_locks/${C2RN_RUN_MODE}"
SUMMARY_PATH="${SUMMARY_ROOT}/${TASK_ID}.json"
RECEIPT_PATH="${RECEIPT_ROOT}/${TASK_ID}.json"
LOCK_PATH="${C2RN_RESULT_ROOT}/task_locks/${C2RN_RUN_MODE}/${TASK_ID}.lock"
if [[ -e "${SUMMARY_PATH}" || -e "${RECEIPT_PATH}" ]]; then
  echo "Refusing duplicate ${C2RN_RUN_MODE} task ${TASK_ID}: terminal path exists." >&2
  exit 17
fi
if ! mkdir "${LOCK_PATH}"; then
  echo "Refusing concurrent duplicate ${C2RN_RUN_MODE} task ${TASK_ID}." >&2
  exit 17
fi
cleanup_lock() {
  rmdir "${LOCK_PATH}" 2>/dev/null || true
}
trap cleanup_lock EXIT

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

cd "${C2RN_SOURCE_ROOT}"
SELECTED_CONFIG="${CONFIGS[TASK_ID]}"
SELECTED_CONFIG_SHA="$(sha256sum -- "${SELECTED_CONFIG}")"
SELECTED_CONFIG_SHA="${SELECTED_CONFIG_SHA%% *}"

"${PYTHON_BIN}" - \
  "${STUDY_CONFIG}" "${SELECTED_CONFIG}" "${TASK_ID}" "${C2RN_DATASET_ROOT}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import torch

study_path, config_path, task_arg, dataset_arg = sys.argv[1:]
task = int(task_arg)
study = json.loads(Path(study_path).read_text(encoding="utf-8"))
config = json.loads(Path(config_path).read_text(encoding="utf-8"))
dataset_root = Path(dataset_arg)
contract = study.get("scientific_contract", {})
optimizer = str(contract.get("optimizer", "")).lower()
expected = []
for beta_tier, tier in study.get("beta_tiers", {}).items():
    for scheme in ("baseline", "ours", "legacy"):
        beta_row = tier.get("schemes", {}).get(scheme, {})
        for sigma in contract.get("noise_sigmas", []):
            expected.append(
                (
                    beta_tier,
                    scheme,
                    float(sigma),
                    float(beta_row.get("injected_beta_B", -1.0)),
                    float(beta_row.get("base_beta", -1.0)),
                )
            )
if task < 0 or task >= len(expected):
    raise SystemExit(f"Task {task} is outside the study's {len(expected)} cases.")
beta_tier, scheme, sigma, injected_beta, base_beta = expected[task]
if study.get("study_id") != config.get("study_id"):
    raise SystemExit("Study/config identity mismatch.")
if config.get("reporting", {}).get("evidence_class") != study.get("evidence_class"):
    raise SystemExit("Evidence-class mismatch.")
if optimizer not in {"adam", "sgd"}:
    raise SystemExit(f"Unsupported optimizer contract: {optimizer!r}.")
if str(config.get("optimizer", {}).get("name", "")).lower() != optimizer:
    raise SystemExit("Exact-config optimizer does not match the study contract.")
if not str(config.get("arm_id", "")).startswith(f"conv2_{scheme}_{optimizer}_"):
    raise SystemExit("Array/config scheme mapping mismatch.")
eqprop = config.get("eqprop", {})
if (
    eqprop.get("beta_tier") != beta_tier
    or float(eqprop.get("endpoint_read_noise_std", -1.0)) != sigma
    or float(eqprop.get("injected_beta_B", -1.0)) != injected_beta
    or float(config.get("beta", -1.0)) != base_beta
):
    raise SystemExit("Array/config beta or read-noise mapping mismatch.")
declared_rates = study.get("schemes", {}).get(scheme, {}).get(
    "learning_rates_by_parameter"
)
if config.get("learning_rates_by_parameter") != declared_rates:
    raise SystemExit("Exact-config learning rates do not match the study contract.")
model = config.get("model_base", {})
if (
    config.get("training_algorithm") != "EP"
    or config.get("runtime_dtype") != "float64"
    or int(model.get("num_iterations_inference", -1)) != 8
    or int(model.get("num_iterations_training", -1)) != 8
):
    raise SystemExit("Training algorithm, dtype, or T/K changed.")
if config.get("evaluation", {}).get("official_test", {}).get("policy") != "disabled":
    raise SystemExit("Official-test policy is not disabled.")
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
if "V100" not in device_name.upper() or device_bytes < 15 * 1024**3:
    raise SystemExit(
        f"Expected V100 with at least 15 GiB, got {device_name!r}/{device_bytes}."
    )
print(
    f"C2RN_ENVIRONMENT_PASS task={task} python={sys.executable} "
    f"torch={torch.__version__} device={device_name!r} bytes={device_bytes}",
    flush=True,
)
PY

echo "C2RN_SOURCE_PASS task=${TASK_ID} archive_sha256=${archive_sha} study_config_sha256=${study_config_sha} config_set_sha256=${config_set_sha}"

"${PYTHON_BIN}" -m experiments.exact_run \
  "${CONFIGS[@]}" \
  --output-root "${RUN_ROOT}" \
  --device cuda \
  --dataset-root "${C2RN_DATASET_ROOT}" \
  --study-id "${STUDY_ID}" \
  --evidence-class "${EVIDENCE_CLASS}" \
  --target jean-zay \
  --summary-json "${SUMMARY_PATH}" \
  --skip-terminal-official-test \
  "${MODE_ARGS[@]}"

"${PYTHON_BIN}" - \
  "${SUMMARY_PATH}" "${SELECTED_CONFIG}" "${SELECTED_CONFIG_SHA}" \
  "${STUDY_CONFIG}" "${C2RN_RUN_MODE}" "${TASK_ID}" "${EXPECTED_EPOCHS}" \
  "${EXPECTED_NOISE_DRAWS}" "${RUN_ROOT}" "${RECEIPT_PATH}" \
  "${C2RN_ENVIRONMENT_ID}" "${EXPERIMENT_SOURCE_COMMIT}" \
  "${EXPERIMENT_SOURCE_ARCHIVE_SHA256}" "${C2RN_CONFIG_SET_SHA256}" \
  "${PARENT_JOB_ID}" "${SLURM_JOB_ID}" <<'PY'
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from experiments.reporting import validate_run

(
    summary_arg,
    selected_config_arg,
    selected_config_sha,
    study_arg,
    run_mode,
    task_arg,
    expected_epochs_arg,
    expected_draws_arg,
    run_root_arg,
    receipt_arg,
    environment_id,
    source_commit,
    source_archive_sha256,
    config_set_sha256,
    parent_job_id,
    job_id,
) = sys.argv[1:]
task = int(task_arg)
expected_epochs = int(expected_epochs_arg)
expected_draws = int(expected_draws_arg)
summary_path = Path(summary_arg)
selected_config_path = Path(selected_config_arg)
study = json.loads(Path(study_arg).read_text(encoding="utf-8"))
selected_config = json.loads(selected_config_path.read_text(encoding="utf-8"))
rows = json.loads(summary_path.read_text(encoding="utf-8"))
if len(rows) != 1 or rows[0].get("status") != "complete":
    raise SystemExit(f"Task summary is not semantically complete: {rows!r}")
row = rows[0]
if row.get("config_sha256") != selected_config_sha:
    raise SystemExit(f"Task-summary config hash mismatch: {row!r}")
if int(row.get("epochs", -1)) != expected_epochs:
    raise SystemExit(f"Executed epoch count mismatch: {row!r}")
if bool(row.get("smoke")) != (run_mode == "canary"):
    raise SystemExit(f"Smoke/production mode mismatch: {row!r}")
run_dir = Path(row["output_dir"]).resolve()
run_root = Path(run_root_arg).resolve()
if not run_dir.is_relative_to(run_root):
    raise SystemExit(f"Run directory escapes the declared root: {run_dir}")
validation_errors = validate_run(run_dir)
if validation_errors:
    raise SystemExit(f"Canonical run validation failed: {validation_errors!r}")

manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
run_config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
if status.get("state") != "complete" or result.get("completion", {}).get("criteria_met") is not True:
    raise SystemExit("Canonical completion state is not complete.")
if manifest.get("study_id") != study.get("study_id"):
    raise SystemExit("Manifest study ID mismatch.")
if manifest.get("evidence_class") != study.get("evidence_class"):
    raise SystemExit("Manifest evidence class mismatch.")
if manifest.get("dataset", {}).get("official_test_read") is not False:
    raise SystemExit("Manifest does not prove official_test_read=false.")
if int(metrics.get("official_test_evaluations", -1)) != 0:
    raise SystemExit("Official test was evaluated.")
if metrics.get("runtime_dtype") != "float64" or metrics.get("training_algorithm") != "EP":
    raise SystemExit("Runtime dtype or training algorithm mismatch.")
if int(metrics.get("eqprop_endpoint_read_noise_draw_count", -1)) != expected_draws:
    raise SystemExit("Endpoint read-noise draw count mismatch.")

contract = study["scientific_contract"]
architecture = run_config.get("architecture", {})
provenance = run_config.get("dataset", {}).get("provenance", {})
optimizer_name = str(contract.get("optimizer", ""))
if selected_config.get("optimizer", {}).get("name") != optimizer_name:
    raise SystemExit("Selected-config optimizer mismatch.")
if metrics.get("optimizer", {}).get("name") != optimizer_name:
    raise SystemExit("Runtime optimizer mismatch.")
arm_id = str(selected_config.get("arm_id", ""))
scheme_matches = [
    scheme
    for scheme in ("baseline", "ours", "legacy")
    if arm_id.startswith(f"conv2_{scheme}_")
]
if len(scheme_matches) != 1:
    raise SystemExit("Could not resolve the amplification scheme from arm_id.")
declared_rates = study["schemes"][scheme_matches[0]][
    "learning_rates_by_parameter"
]
if selected_config.get("learning_rates_by_parameter") != declared_rates:
    raise SystemExit("Selected-config learning-rate vector mismatch.")
if selected_config.get("optimizer", {}).get("learning_rate") != [
    declared_rates[name] for name in selected_config.get("parameter_order", [])
]:
    raise SystemExit("Optimizer learning-rate order mismatch.")
if architecture.get("initial_parameter_state_sha256") != contract[
    "expected_initial_parameter_state_sha256"
]:
    raise SystemExit("Initial parameter-state hash mismatch.")
if provenance.get("train_indices_sha256") != contract["expected_train_indices_sha256"]:
    raise SystemExit("Training split hash mismatch.")
if provenance.get("validation_indices_sha256") != contract[
    "expected_validation_indices_sha256"
]:
    raise SystemExit("Validation split hash mismatch.")
expected_orders = contract["expected_train_batch_order_sha256"]
observed_orders = provenance.get("train_batch_order_sha256", [])
expected_prefix = expected_orders[:expected_epochs]
if observed_orders != expected_prefix:
    raise SystemExit("Minibatch-order hash mismatch.")

eqprop = metrics.get("eqprop", {})
selected_eqprop = selected_config.get("eqprop", {})
if (
    eqprop.get("variant") != "centered"
    or eqprop.get("nudging_mode") != "current"
    or eqprop.get("normalize_current_scale") is not True
    or eqprop.get("input_read_noise") is not False
    or float(eqprop.get("endpoint_read_noise_std", -1.0))
    != float(selected_eqprop["endpoint_read_noise_std"])
    or int(eqprop.get("endpoint_read_noise_seed", -1))
    != int(selected_eqprop["endpoint_read_noise_seed"])
):
    raise SystemExit("Runtime EqProp/read-noise contract mismatch.")

receipt = {
    "schema_version": "conv2-read-noise-jean-zay-task-receipt/v1",
    "recorded_at": datetime.now(timezone.utc).isoformat(),
    "study_id": study["study_id"],
    "evidence_class": study["evidence_class"],
    "run_mode": run_mode,
    "task_index": task,
    "job_id": job_id,
    "parent_job_id": parent_job_id,
    "environment_id": environment_id,
    "optimizer": optimizer_name,
    "source_commit": source_commit,
    "source_archive_sha256": source_archive_sha256,
    "config_set_sha256": config_set_sha256,
    "selected_config": str(selected_config_path),
    "selected_config_sha256": selected_config_sha,
    "run_dir": str(run_dir),
    "manifest_sha256": hashlib.sha256((run_dir / "manifest.json").read_bytes()).hexdigest(),
    "status_sha256": hashlib.sha256((run_dir / "status.json").read_bytes()).hexdigest(),
    "metrics_sha256": hashlib.sha256((run_dir / "metrics.jsonl").read_bytes()).hexdigest(),
    "result_sha256": hashlib.sha256((run_dir / "result.json").read_bytes()).hexdigest(),
    "semantic_pass": True,
    "official_test_read": False,
}
receipt_path = Path(receipt_arg)
receipt_path.parent.mkdir(parents=True, exist_ok=True)
receipt_path.write_text(
    json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
    encoding="utf-8",
)
print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False), flush=True)
PY

echo "C2RN_SEMANTIC_PASS mode=${C2RN_RUN_MODE} task=${TASK_ID} summary=${SUMMARY_PATH} receipt=${RECEIPT_PATH}"
