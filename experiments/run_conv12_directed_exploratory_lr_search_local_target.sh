#!/usr/bin/env bash
set -euo pipefail

: "${C12LR_SOURCE_ROOT:?Set C12LR_SOURCE_ROOT to the frozen staged source.}"
: "${C12LR_SOURCE_ARCHIVE:?Set C12LR_SOURCE_ARCHIVE to the frozen source archive.}"
: "${C12LR_RESULT_ROOT:?Set C12LR_RESULT_ROOT to the target result root.}"
: "${C12LR_DATASET_ROOT:?Set C12LR_DATASET_ROOT to the target MNIST root.}"
: "${C12LR_ENVIRONMENT_ID:?Set C12LR_ENVIRONMENT_ID.}"
: "${C12LR_STUDY_CONFIG_SHA256:?Set C12LR_STUDY_CONFIG_SHA256.}"
: "${C12LR_SURFACE_INDICES:?Set C12LR_SURFACE_INDICES to comma-separated indices.}"
: "${C12LR_TARGET:?Set C12LR_TARGET to the recorded target name.}"
: "${C12LR_PYTHON_BIN:?Set C12LR_PYTHON_BIN to the target Python executable.}"
: "${EXPERIMENT_SOURCE_COMMIT:?Set EXPERIMENT_SOURCE_COMMIT.}"
: "${EXPERIMENT_SOURCE_ARCHIVE_SHA256:?Set EXPERIMENT_SOURCE_ARCHIVE_SHA256.}"

C12LR_STUDY_ID="${C12LR_STUDY_ID:-perfectdiode-conv12-directed-exploratory-lr-search-seed0-20260812-v1}"
C12LR_STUDY_CONFIG_RELATIVE="${C12LR_STUDY_CONFIG_RELATIVE:-configs/conv/perfectdiode_conv12_directed_exploratory_lr_search_seed0_20260812_v1.json}"
STUDY_CONFIG="${C12LR_SOURCE_ROOT}/${C12LR_STUDY_CONFIG_RELATIVE}"

if [[ ! "${EXPERIMENT_SOURCE_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "EXPERIMENT_SOURCE_COMMIT must be a lowercase 40-character SHA." >&2
  exit 3
fi
for digest_name in EXPERIMENT_SOURCE_ARCHIVE_SHA256 C12LR_STUDY_CONFIG_SHA256; do
  digest="${!digest_name}"
  if [[ ! "${digest}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "${digest_name} must be a lowercase SHA-256." >&2
    exit 3
  fi
done
if [[ ! -d "${C12LR_SOURCE_ROOT}" || ! -s "${C12LR_SOURCE_ARCHIVE}" ]]; then
  echo "Frozen source directory or archive is missing." >&2
  exit 3
fi
if [[ ! -s "${STUDY_CONFIG}" || ! -d "${C12LR_DATASET_ROOT}" ]]; then
  echo "Study config or MNIST dataset root is missing." >&2
  exit 3
fi
if [[ ! -x "${C12LR_PYTHON_BIN}" ]]; then
  echo "Configured Python executable is missing: ${C12LR_PYTHON_BIN}." >&2
  exit 3
fi

archive_sha="$(sha256sum -- "${C12LR_SOURCE_ARCHIVE}")"
archive_sha="${archive_sha%% *}"
config_sha="$(sha256sum -- "${STUDY_CONFIG}")"
config_sha="${config_sha%% *}"
if [[ "${archive_sha}" != "${EXPERIMENT_SOURCE_ARCHIVE_SHA256}" ]]; then
  echo "Frozen source archive SHA mismatch." >&2
  exit 3
fi
if [[ "${config_sha}" != "${C12LR_STUDY_CONFIG_SHA256}" ]]; then
  echo "Directed study config SHA mismatch." >&2
  exit 3
fi

source_root_canonical="$(realpath -e -- "${C12LR_SOURCE_ROOT}")"
result_root_canonical="$(realpath -m -- "${C12LR_RESULT_ROOT}")"
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
export TF_CPP_MIN_LOG_LEVEL=2
export MPLCONFIGDIR="${C12LR_RESULT_ROOT}/.matplotlib"

mkdir -p "${C12LR_RESULT_ROOT}/transport_receipts" "${MPLCONFIGDIR}"
cd "${C12LR_SOURCE_ROOT}"

"${C12LR_PYTHON_BIN}" - "${STUDY_CONFIG}" "${C12LR_DATASET_ROOT}" "${C12LR_STUDY_ID}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import torch

study_path = Path(sys.argv[1])
dataset_root = Path(sys.argv[2])
expected_study_id = sys.argv[3]
study = json.loads(study_path.read_text(encoding="utf-8"))
if study.get("study_id") != expected_study_id:
    raise SystemExit("Unexpected directed study identity.")
parent = json.loads(
    (Path.cwd() / study["parent"]["study_config"]).read_text(encoding="utf-8")
)
for field, path in (
    ("source_train_images_sha256", dataset_root / "MNIST/raw/train-images-idx3-ubyte"),
    ("source_train_labels_sha256", dataset_root / "MNIST/raw/train-labels-idx1-ubyte"),
):
    expected = parent["dataset"][field]
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit(f"MNIST source guard failed for {path}.")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable on the target.")
name = torch.cuda.get_device_name(0)
memory = int(torch.cuda.get_device_properties(0).total_memory)
if memory < 9 * 1024**3:
    raise SystemExit(f"Expected at least 9 GiB GPU memory, got {name!r}/{memory}.")
print(
    f"C12LR_ENVIRONMENT_PASS python={sys.executable} torch={torch.__version__} "
    f"device={name!r} bytes={memory}",
    flush=True,
)
PY

IFS=',' read -r -a TASKS <<< "${C12LR_SURFACE_INDICES}"
if (( ${#TASKS[@]} == 0 )); then
  echo "No surface indices were supplied." >&2
  exit 3
fi

for raw_task in "${TASKS[@]}"; do
  if [[ ! "${raw_task}" =~ ^([0-9]|1[01])$ ]]; then
    echo "Invalid surface index: ${raw_task}." >&2
    exit 3
  fi
  task=$((10#${raw_task}))
  task_root="${C12LR_RESULT_ROOT}/shards/${C12LR_TARGET}-task_${task}"
  receipt_path="${C12LR_RESULT_ROOT}/transport_receipts/${C12LR_TARGET}-task_${task}.json"
  echo "C12LR_SURFACE_START target=${C12LR_TARGET} task=${task} root=${task_root}"
  "${C12LR_PYTHON_BIN}" -m experiments.run_conv12_directed_exploratory_lr_search run-surface \
    --study "${STUDY_CONFIG}" \
    --surface-index "${task}" \
    --device cuda \
    --target "${C12LR_TARGET}" \
    --dataset-root "${C12LR_DATASET_ROOT}" \
    --output-root "${task_root}"

  "${C12LR_PYTHON_BIN}" - \
    "${STUDY_CONFIG}" "${task_root}" "${receipt_path}" "${task}" \
    "${C12LR_TARGET}" "${C12LR_ENVIRONMENT_ID}" "${EXPERIMENT_SOURCE_COMMIT}" \
    "${EXPERIMENT_SOURCE_ARCHIVE_SHA256}" "${C12LR_STUDY_CONFIG_SHA256}" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

from experiments.reporting import sha256_file, validate_run
from experiments.run_conv12_directed_exploratory_lr_search import (
    surface_cells,
    surface_specs,
)

(
    study_arg,
    task_root_arg,
    receipt_arg,
    task_arg,
    target,
    environment_id,
    source_commit,
    source_archive_sha256,
    study_config_sha256,
) = sys.argv[1:]
study_path = Path(study_arg)
task_root = Path(task_root_arg)
receipt_path = Path(receipt_arg)
task = int(task_arg)
study = json.loads(study_path.read_text(encoding="utf-8"))
surface = surface_specs(study)[task]
expected_count = len(surface_cells(study, surface))
architecture = surface["architecture"]
summary_path = task_root / "surfaces" / surface["surface_id"] / "summary.json"
asset_path = task_root / "assets/bounded_uniform" / architecture / "asset.json"
if not summary_path.is_file() or not asset_path.is_file():
    raise SystemExit("Missing terminal surface summary or initializer asset record.")
summary = json.loads(summary_path.read_text(encoding="utf-8"))
asset = json.loads(asset_path.read_text(encoding="utf-8"))
expected_initializer = study["parent"]["initializer_checkpoint_sha256_by_architecture"][architecture]
asset_checkpoint = Path(asset.get("checkpoint", ""))
if (
    summary.get("surface") != surface
    or summary.get("official_test_read") is not False
    or summary.get("expected_cells") != expected_count
    or asset.get("checkpoint_sha256") != expected_initializer
    or not asset_checkpoint.is_file()
    or sha256_file(asset_checkpoint) != expected_initializer
):
    raise SystemExit("Terminal surface or initializer contract mismatch.")
summary_status = summary.get("status")
candidates = summary.get("candidates")
validated = 0
if summary_status == "complete":
    if (
        summary.get("complete") is not True
        or summary.get("terminal_cells") != expected_count
        or not isinstance(candidates, list)
        or len(candidates) != expected_count
    ):
        raise SystemExit("Incomplete directed surface.")
    task_root_resolved = task_root.resolve()
    for candidate in candidates:
        run_dir = Path(str(candidate.get("path", ""))).resolve()
        if not run_dir.is_relative_to(task_root_resolved):
            raise SystemExit(f"Cell path escapes its task root: {run_dir}.")
        errors = validate_run(run_dir)
        if errors:
            raise SystemExit(f"Canonical validation failed for {run_dir}: {errors!r}")
        cell = json.loads((run_dir / "cell.json").read_text(encoding="utf-8"))
        rates = cell.get("learning_rates_by_parameter")
        bias_rates = {
            name: value
            for name, value in (rates.items() if isinstance(rates, dict) else ())
            if str(name).startswith("Bias_")
        }
        if not bias_rates or any(float(value) != 0.0 for value in bias_rates.values()):
            raise SystemExit(f"Cell does not retain exact-zero bias rates: {run_dir}.")
        validated += 1
elif summary_status == "unresolved_probe":
    if summary.get("terminal_cells") != 0 or candidates:
        raise SystemExit("Malformed unresolved-probe surface.")
else:
    raise SystemExit(f"Unsupported terminal surface status: {summary_status!r}.")
receipt = {
    "schema_version": "perfectdiode-conv12-directed-exploratory-lr-transport/v1",
    "study_id": study["study_id"],
    "surface": surface,
    "task_id": task,
    "target": target,
    "environment_id": environment_id,
    "source_commit": source_commit,
    "source_archive_sha256": source_archive_sha256,
    "study_config_sha256": study_config_sha256,
    "initializer_checkpoint_sha256": expected_initializer,
    "summary_status": summary_status,
    "expected_cells": expected_count,
    "terminal_cells": summary["terminal_cells"],
    "validated_cell_bundles": validated,
    "status_counts": summary["status_counts"],
    "best_observation": summary.get("best_observation"),
    "summary_path": str(summary_path),
    "summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
    "official_test_read": False,
    "semantic_pass": True,
}
receipt_path.parent.mkdir(parents=True, exist_ok=True)
temporary = receipt_path.with_suffix(receipt_path.suffix + f".tmp-{os.getpid()}")
temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(receipt_path)
print(
    f"C12LR_SEMANTIC_PASS target={target} task={task} "
    f"surface={surface['surface_id']} outcome={summary_status} validated_cells={validated}",
    flush=True,
)
PY
done

echo "C12LR_LAUNCHER_SEMANTIC_PASS target=${C12LR_TARGET} surfaces=${C12LR_SURFACE_INDICES}"
