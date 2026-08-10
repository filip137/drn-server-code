#!/usr/bin/env bash

set -euo pipefail

: "${PDBLR_SOURCE_ROOT:?Set PDBLR_SOURCE_ROOT to the frozen staged source.}"
: "${PDBLR_SOURCE_ARCHIVE:?Set PDBLR_SOURCE_ARCHIVE to the frozen source archive.}"
: "${PDBLR_RESULT_ROOT:?Set PDBLR_RESULT_ROOT to the target result root.}"
: "${PDBLR_DATASET_ROOT:?Set PDBLR_DATASET_ROOT to the target MNIST root.}"
: "${PDBLR_PYTHON:?Set PDBLR_PYTHON to the target Python executable.}"
: "${PDBLR_TARGET:?Set PDBLR_TARGET to the recorded launcher target.}"
: "${PDBLR_START_INDEX:?Set the first inclusive surface index.}"
: "${PDBLR_END_INDEX:?Set the last inclusive surface index.}"
: "${PDBLR_ENVIRONMENT_ID:?Set the recorded target environment identity.}"
: "${PDBLR_STUDY_CONFIG_SHA256:?Set the frozen study-config SHA-256.}"
: "${EXPERIMENT_SOURCE_COMMIT:?Set the frozen source parent commit.}"
: "${EXPERIMENT_SOURCE_ARCHIVE_SHA256:?Set the frozen source archive SHA-256.}"

STUDY_ID="perfectdiode-bounded-uniform-zero-bias-lr-search-conv123-seed0-20260810-v1"
STUDY_CONFIG_RELATIVE="configs/conv/perfectdiode_bounded_uniform_zero_bias_lr_search_conv123_seed0_20260810_v1.json"
STUDY_CONFIG="${PDBLR_SOURCE_ROOT}/${STUDY_CONFIG_RELATIVE}"
OUTPUT_ROOT="${PDBLR_RESULT_ROOT}/shards/${PDBLR_TARGET}"
RECEIPT_ROOT="${PDBLR_RESULT_ROOT}/transport_receipts/${PDBLR_TARGET}"

if [[ ! "${PDBLR_TARGET}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "PDBLR_TARGET must be a path-safe target token." >&2
  exit 3
fi
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
if [[ ! "${PDBLR_START_INDEX}" =~ ^[0-9]+$ || ! "${PDBLR_END_INDEX}" =~ ^[0-9]+$ ]]; then
  echo "Surface indices must be non-negative integers." >&2
  exit 3
fi
START_INDEX=$((10#${PDBLR_START_INDEX}))
END_INDEX=$((10#${PDBLR_END_INDEX}))
if (( START_INDEX < 0 || END_INDEX >= 18 || START_INDEX > END_INDEX )); then
  echo "Invalid inclusive surface interval [${START_INDEX},${END_INDEX}]." >&2
  exit 3
fi

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
if [[ ! -x "${PDBLR_PYTHON}" ]]; then
  echo "Python is missing or non-executable: ${PDBLR_PYTHON}." >&2
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

"${PDBLR_PYTHON}" - \
  "${STUDY_CONFIG}" \
  "${PDBLR_DATASET_ROOT}" \
  "${STUDY_ID}" \
  "${PDBLR_ENVIRONMENT_ID}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import torch

study_path = Path(sys.argv[1])
dataset_root = Path(sys.argv[2])
expected_study_id = sys.argv[3]
environment_id = sys.argv[4]
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
    raise SystemExit("The selected Python environment cannot access CUDA.")
print(
    "PDBLR_ENVIRONMENT_PASS "
    f"environment_id={environment_id} python={sys.executable} "
    f"torch={torch.__version__} cuda_device={torch.cuda.get_device_name(0)!r}",
    flush=True,
)
PY

echo "PDBLR_SOURCE_PASS environment_id=${PDBLR_ENVIRONMENT_ID} commit=${EXPERIMENT_SOURCE_COMMIT} commit_guard=${commit_guard} archive_sha256=${EXPERIMENT_SOURCE_ARCHIVE_SHA256} config_sha256=${PDBLR_STUDY_CONFIG_SHA256}"

for (( index=START_INDEX; index<=END_INDEX; index++ )); do
  "${PDBLR_PYTHON}" -m experiments.run_conv12_bounded_rho run-surface \
    --study "${STUDY_CONFIG}" \
    --surface-index "${index}" \
    --device cuda \
    --target "${PDBLR_TARGET}" \
    --dataset-root "${PDBLR_DATASET_ROOT}" \
    --output-root "${OUTPUT_ROOT}"

  receipt_path="${RECEIPT_ROOT}/surface_${index}.json"
  "${PDBLR_PYTHON}" - \
    "${STUDY_CONFIG}" \
    "${OUTPUT_ROOT}" \
    "${receipt_path}" \
    "${index}" \
    "${PDBLR_TARGET}" \
    "${PDBLR_ENVIRONMENT_ID}" \
    "${EXPERIMENT_SOURCE_COMMIT}" \
    "${EXPERIMENT_SOURCE_ARCHIVE_SHA256}" \
    "${PDBLR_STUDY_CONFIG_SHA256}" <<'PY'
import json
import sys
from itertools import product
from pathlib import Path

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
if status == "complete":
    if not isinstance(selected, dict):
        raise SystemExit(f"Complete selection lacks a selected cell: {selection!r}")
    if selected.get("status") != "complete" or selected.get("selection_eligible") is not True:
        raise SystemExit(f"Selected cell is not complete and eligible: {selected!r}")
    run_dir = Path(selected["path"])
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
