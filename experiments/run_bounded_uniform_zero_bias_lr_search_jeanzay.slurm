#!/bin/bash -l
#SBATCH --job-name=pd-bu-zb-lr-c3
#SBATCH --account=fmu@v100
#SBATCH --partition=gpu_p13
#SBATCH --qos=qos_gpu-t4
#SBATCH --constraint=v100-32g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=24:00:00
#SBATCH --array=0-2%1
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err

set -euo pipefail

: "${PDBLR_SOURCE_ROOT:?Set PDBLR_SOURCE_ROOT to the frozen staged source.}"
: "${PDBLR_SOURCE_ARCHIVE:?Set PDBLR_SOURCE_ARCHIVE to the frozen source archive.}"
: "${PDBLR_RESULT_ROOT:?Set PDBLR_RESULT_ROOT to the Jean Zay result root.}"
: "${PDBLR_DATASET_ROOT:?Set PDBLR_DATASET_ROOT to the Jean Zay MNIST root.}"
: "${PDBLR_ENVIRONMENT_ID:?Set the recorded Jean Zay environment identity.}"
: "${PDBLR_STUDY_CONFIG_SHA256:?Set the frozen study-config SHA-256.}"
: "${EXPERIMENT_SOURCE_COMMIT:?Set the frozen source parent commit.}"
: "${EXPERIMENT_SOURCE_ARCHIVE_SHA256:?Set the frozen source archive SHA-256.}"
: "${SLURM_ARRAY_TASK_ID:?Submit this wrapper as the declared three-task array.}"

STUDY_ID="perfectdiode-bounded-uniform-zero-bias-lr-search-conv123-seed0-20260810-v1"
STUDY_CONFIG_RELATIVE="configs/conv/perfectdiode_bounded_uniform_zero_bias_lr_search_conv123_seed0_20260810_v1.json"
STUDY_CONFIG="${PDBLR_SOURCE_ROOT}/${STUDY_CONFIG_RELATIVE}"
TARGET="jean-zay"
MODULE_ID="pytorch-gpu/py3/2.5.0"

if [[ ! "${PDBLR_ENVIRONMENT_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._:@/+%-]*$ ]]; then
  echo "PDBLR_ENVIRONMENT_ID must be a nonempty, path-free identity token." >&2
  exit 3
fi
if [[ ! "${EXPERIMENT_SOURCE_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "EXPERIMENT_SOURCE_COMMIT must be a lowercase forty-character Git identity." >&2
  exit 3
fi
for digest_name in EXPERIMENT_SOURCE_ARCHIVE_SHA256 PDBLR_STUDY_CONFIG_SHA256; do
  digest="${!digest_name}"
  if [[ ! "${digest}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "${digest_name} must be a lowercase hexadecimal SHA-256." >&2
    exit 3
  fi
done
if [[ ! "${SLURM_ARRAY_TASK_ID}" =~ ^[0-9]+$ ]]; then
  echo "SLURM_ARRAY_TASK_ID must be a non-negative integer." >&2
  exit 3
fi
SLURM_ARRAY_TASK_ID=$((10#${SLURM_ARRAY_TASK_ID}))
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID > 2 )); then
  echo "Array task ${SLURM_ARRAY_TASK_ID} is outside [0,2]." >&2
  exit 3
fi
PAIR_START=$((12 + 2 * SLURM_ARRAY_TASK_ID))
PAIR_END=$((PAIR_START + 1))
OUTPUT_ROOT="${PDBLR_RESULT_ROOT}/shards/jean-zay"
RECEIPT_ROOT="${PDBLR_RESULT_ROOT}/transport_receipts/jean-zay/task_${SLURM_ARRAY_TASK_ID}"

if [[ ! -d "${PDBLR_SOURCE_ROOT}" ]]; then
  echo "Frozen source directory is missing: ${PDBLR_SOURCE_ROOT}." >&2
  exit 3
fi
if [[ ! -s "${PDBLR_SOURCE_ARCHIVE}" ]]; then
  echo "Frozen source archive is missing or empty: ${PDBLR_SOURCE_ARCHIVE}." >&2
  exit 3
fi
if [[ ! -s "${STUDY_CONFIG}" ]]; then
  echo "Frozen study config is missing or empty: ${STUDY_CONFIG}." >&2
  exit 3
fi
if [[ ! -d "${PDBLR_DATASET_ROOT}" ]]; then
  echo "MNIST dataset root is missing: ${PDBLR_DATASET_ROOT}." >&2
  exit 3
fi

actual_archive_sha256="$(sha256sum -- "${PDBLR_SOURCE_ARCHIVE}")"
actual_archive_sha256="${actual_archive_sha256%% *}"
if [[ "${actual_archive_sha256}" != "${EXPERIMENT_SOURCE_ARCHIVE_SHA256}" ]]; then
  echo "Source archive SHA-256 mismatch: expected ${EXPERIMENT_SOURCE_ARCHIVE_SHA256}, got ${actual_archive_sha256}." >&2
  exit 3
fi
actual_config_sha256="$(sha256sum -- "${STUDY_CONFIG}")"
actual_config_sha256="${actual_config_sha256%% *}"
if [[ "${actual_config_sha256}" != "${PDBLR_STUDY_CONFIG_SHA256}" ]]; then
  echo "Study-config SHA-256 mismatch: expected ${PDBLR_STUDY_CONFIG_SHA256}, got ${actual_config_sha256}." >&2
  exit 3
fi

if git -C "${PDBLR_SOURCE_ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  actual_source_commit="$(git -C "${PDBLR_SOURCE_ROOT}" rev-parse HEAD)"
  if [[ "${actual_source_commit}" != "${EXPERIMENT_SOURCE_COMMIT}" ]]; then
    echo "Frozen source commit mismatch: expected ${EXPERIMENT_SOURCE_COMMIT}, got ${actual_source_commit}." >&2
    exit 3
  fi
  commit_guard="git_head_and_archive"
else
  commit_guard="environment_identity_and_archive"
fi

source_root_canonical="$(realpath -e -- "${PDBLR_SOURCE_ROOT}")"
result_root_canonical="$(realpath -m -- "${PDBLR_RESULT_ROOT}")"
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
  echo "The ${MODULE_ID} module did not provide an executable Python." >&2
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

mkdir -p "${OUTPUT_ROOT}" "${RECEIPT_ROOT}"
cd "${PDBLR_SOURCE_ROOT}"

"${PYTHON_BIN}" - \
  "${STUDY_CONFIG}" \
  "${PDBLR_DATASET_ROOT}" \
  "${STUDY_ID}" \
  "${PDBLR_ENVIRONMENT_ID}" \
  "${MODULE_ID}" <<'PY'
import hashlib
import hashlib
import json
import sys
from pathlib import Path

import torch

study_path = Path(sys.argv[1])
dataset_root = Path(sys.argv[2])
expected_study_id = sys.argv[3]
environment_id = sys.argv[4]
module_id = sys.argv[5]
study = json.loads(study_path.read_text(encoding="utf-8"))
if study.get("study_id") != expected_study_id:
    raise SystemExit(f"Unexpected study identity: {study.get('study_id')!r}")
if study.get("dataset", {}).get("official_test_read") is not False:
    raise SystemExit("Study config does not keep the official test unread.")

sources = (
    (
        "source_train_images_sha256",
        dataset_root / "MNIST" / "raw" / "train-images-idx3-ubyte",
    ),
    (
        "source_train_labels_sha256",
        dataset_root / "MNIST" / "raw" / "train-labels-idx1-ubyte",
    ),
)
for field, path in sources:
    if not path.is_file() or path.stat().st_size <= 0:
        raise SystemExit(f"Missing or empty MNIST source file: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = study["dataset"].get(field)
    if digest != expected:
        raise SystemExit(
            f"MNIST source SHA-256 mismatch for {path}: expected {expected}, got {digest}"
        )

if not torch.cuda.is_available():
    raise SystemExit("The selected Jean Zay Python module cannot access CUDA.")
device_name = torch.cuda.get_device_name(0)
device_bytes = int(torch.cuda.get_device_properties(0).total_memory)
if "V100" not in device_name.upper() or device_bytes < 30 * 1024**3:
    raise SystemExit(
        f"Expected a V100-32G allocation, got {device_name!r} with {device_bytes} bytes."
    )
print(
    "PDBLR_ENVIRONMENT_PASS "
    f"environment_id={environment_id} module={module_id} python={sys.executable} "
    f"torch={torch.__version__} cuda_device={device_name!r} cuda_bytes={device_bytes}",
    flush=True,
)
PY

echo "PDBLR_SOURCE_PASS environment_id=${PDBLR_ENVIRONMENT_ID} module=${MODULE_ID} commit=${EXPERIMENT_SOURCE_COMMIT} commit_guard=${commit_guard} archive_sha256=${EXPERIMENT_SOURCE_ARCHIVE_SHA256} config_sha256=${PDBLR_STUDY_CONFIG_SHA256} task=${SLURM_ARRAY_TASK_ID} pair=${PAIR_START}-${PAIR_END}"

for (( index=PAIR_START; index<=PAIR_END; index++ )); do
  "${PYTHON_BIN}" -m experiments.run_conv12_bounded_rho run-surface \
    --study "${STUDY_CONFIG}" \
    --surface-index "${index}" \
    --device cuda \
    --target "${TARGET}" \
    --dataset-root "${PDBLR_DATASET_ROOT}" \
    --output-root "${OUTPUT_ROOT}"

  receipt_path="${RECEIPT_ROOT}/surface_${index}.json"
  "${PYTHON_BIN}" - \
    "${STUDY_CONFIG}" \
    "${OUTPUT_ROOT}" \
    "${receipt_path}" \
    "${index}" \
    "${TARGET}" \
    "${PDBLR_ENVIRONMENT_ID}" \
    "${EXPERIMENT_SOURCE_COMMIT}" \
    "${EXPERIMENT_SOURCE_ARCHIVE_SHA256}" \
    "${PDBLR_STUDY_CONFIG_SHA256}" <<'PY'
import json
import sys
from itertools import product
from pathlib import Path

from experiments.run_conv12_bounded_rho import (
    validate_conv3_selected_post_training_tk_evidence,
)

(
    study_arg,
    output_root_arg,
    receipt_arg,
    index_arg,
    target,
    environment_id,
    source_commit,
    source_archive_sha256,
    study_config_sha256,
) = sys.argv[1:]
study_path = Path(study_arg)
output_root = Path(output_root_arg)
receipt_path = Path(receipt_arg)
surface_index = int(index_arg)
study = json.loads(study_path.read_text(encoding="utf-8"))

surfaces = []
for initializer, architecture, scheme, optimizer in product(
    study["scope"]["initializers"],
    study["scope"]["architectures"],
    study["scope"]["schemes"],
    study["scope"]["optimizers"],
):
    surfaces.append(
        {
            "initializer": initializer,
            "architecture": architecture,
            "scheme": scheme,
            "optimizer": optimizer,
            "surface_id": (
                f"{initializer}__{architecture}__{scheme}__{optimizer.lower()}"
            ),
        }
    )
surface = surfaces[surface_index]
selection_path = output_root / "surfaces" / surface["surface_id"] / "selection.json"
if not selection_path.is_file() or selection_path.stat().st_size <= 0:
    raise SystemExit(f"Missing or empty terminal selection: {selection_path}")
selection = json.loads(selection_path.read_text(encoding="utf-8"))
for key in ("initializer", "architecture", "scheme", "optimizer"):
    if selection.get(key) != surface[key]:
        raise SystemExit(f"Selection {key} mismatch: {selection!r}")
if selection.get("official_test_read") is not False:
    raise SystemExit(f"Selection does not prove official_test_read=false: {selection!r}")

asset_record_path = (
    output_root
    / "assets"
    / surface["initializer"]
    / surface["architecture"]
    / "asset.json"
)
if not asset_record_path.is_file() or asset_record_path.stat().st_size <= 0:
    raise SystemExit(f"Missing shared-initialization record: {asset_record_path}")
asset_record = json.loads(asset_record_path.read_text(encoding="utf-8"))
asset_checkpoint = Path(asset_record["checkpoint"])
if not asset_checkpoint.is_file() or asset_checkpoint.stat().st_size <= 0:
    raise SystemExit(f"Missing shared-initialization checkpoint: {asset_checkpoint}")
asset_checkpoint_sha256 = hashlib.sha256(asset_checkpoint.read_bytes()).hexdigest()
if asset_checkpoint_sha256 != asset_record.get("checkpoint_sha256"):
    raise SystemExit(
        "Shared-initialization checkpoint SHA-256 mismatch: "
        f"record={asset_record.get('checkpoint_sha256')}, "
        f"observed={asset_checkpoint_sha256}"
    )

terminal_unresolved = {
    "unresolved_after_boundary_expansion",
    "unresolved_dead_gradient",
    "unresolved_fixed_tk_gradient_mismatch",
    "unresolved_no_pass",
    "unresolved_no_safe_center",
    "unresolved_no_safe_completed_core_candidate",
    "unresolved_post_training_tk",
    "unresolved_probe",
    "unresolved_safety",
    "unresolved_tk_gradient_viability",
    "unresolved_tk_residual",
}
status = selection.get("status")
selected = selection.get("selected")
conv3_post_training_tk_validation = None
if surface["architecture"] == "conv3" and status in {
    "complete",
    "unresolved_post_training_tk",
}:
    operating_point_gate_path = (
        output_root
        / "fixed_tk"
        / surface["initializer"]
        / surface["architecture"]
        / surface["scheme"]
        / "result.json"
    )
    conv3_post_training_tk_validation = (
        validate_conv3_selected_post_training_tk_evidence(
            study_path,
            selection_path,
            operating_point_gate_path,
        )
    )
    expected_post_pass = status == "complete"
    if (
        conv3_post_training_tk_validation.get("post_training_tk_passed")
        is not expected_post_pass
    ):
        raise SystemExit(
            "Conv3 selection status disagrees with strict post-training T/K "
            f"validation: {conv3_post_training_tk_validation!r}"
        )
if status == "complete":
    if not isinstance(selected, dict):
        raise SystemExit(f"Complete selection lacks a selected cell: {selection!r}")
    if surface["architecture"] == "conv3":
        post_training_tk = selection.get("post_training_tk")
        post_epochs = (
            post_training_tk.get("epochs", [])
            if isinstance(post_training_tk, dict)
            else []
        )
        if not (
            isinstance(post_training_tk, dict)
            and post_training_tk.get("status") == "complete"
            and post_training_tk.get("passed") is True
            and len(post_epochs) == 3
            and all(epoch.get("passed") is True for epoch in post_epochs)
        ):
            raise SystemExit(
                "Complete Conv3 selection lacks three passing post-training "
                f"T/K epoch receipts: {selection!r}"
            )
    if selected.get("status") != "complete" or selected.get("selection_eligible") is not True:
        raise SystemExit(f"Selected cell is not complete and eligible: {selected!r}")
    run_dir = (
        Path(conv3_post_training_tk_validation["selected_run_dir"])
        if conv3_post_training_tk_validation is not None
        else Path(selected["path"])
    )
    required = {
        "manifest": run_dir / "manifest.json",
        "status": run_dir / "status.json",
        "metrics": run_dir / "metrics.jsonl",
        "result": run_dir / "result.json",
    }
    for name, path in required.items():
        if not path.is_file() or path.stat().st_size <= 0:
            raise SystemExit(f"Missing or empty selected-cell {name}: {path}")
    run_status = json.loads(required["status"].read_text(encoding="utf-8"))
    run_result = json.loads(required["result"].read_text(encoding="utf-8"))
    if run_status.get("state") != "complete":
        raise SystemExit(f"Selected-cell status is not complete: {run_status!r}")
    if run_result.get("completion", {}).get("criteria_met") is not True:
        raise SystemExit(f"Selected-cell completion criteria failed: {run_result!r}")
    if run_result.get("completion", {}).get("official_test_read") is not False:
        raise SystemExit(f"Selected cell read the official test: {run_result!r}")
    safety_artifacts = [
        artifact
        for artifact in run_result.get("artifacts", [])
        if artifact.get("path") == "artifacts/safety_diagnostics.json"
    ]
    if len(safety_artifacts) != 1:
        raise SystemExit(
            "Selected-cell result must index exactly one safety_diagnostics.json: "
            f"{run_result!r}"
        )
    safety_path = run_dir / safety_artifacts[0]["path"]
    safety_sha256 = hashlib.sha256(safety_path.read_bytes()).hexdigest()
    if safety_sha256 != safety_artifacts[0].get("sha256"):
        raise SystemExit(
            f"Selected-cell safety-diagnostics hash mismatch: {safety_path}"
        )
    selected_run_dir = str(run_dir)
elif status in terminal_unresolved:
    if selected is not None:
        raise SystemExit(f"Unresolved selection unexpectedly retained a cell: {selection!r}")
    selected_run_dir = None
else:
    raise SystemExit(f"Surface selection is not terminal: {selection!r}")

receipt = {
    "schema_version": "perfectdiode-conv123-bounded-uniform-zero-bias-rho-transport-receipt/v1",
    "study_id": study["study_id"],
    "surface_index": surface_index,
    "surface": surface,
    "target": target,
    "environment_id": environment_id,
    "source_commit": source_commit,
    "source_archive_sha256": source_archive_sha256,
    "study_config_sha256": study_config_sha256,
    "selection_path": str(selection_path),
    "selection_status": status,
    "selected_run_dir": selected_run_dir,
    "shared_initialization_checkpoint": str(asset_checkpoint),
    "shared_initialization_checkpoint_sha256": asset_checkpoint_sha256,
    "conv3_post_training_tk_validation": conv3_post_training_tk_validation,
    "semantic_status": "pass",
    "official_test_read": False,
}
receipt_path.parent.mkdir(parents=True, exist_ok=True)
receipt_path.write_text(
    json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
    encoding="utf-8",
)
print(
    f"PDBLR_SEMANTIC_PASS target={target} index={surface_index} "
    f"status={status} environment_id={environment_id} receipt={receipt_path}",
    flush=True,
)
PY
done
