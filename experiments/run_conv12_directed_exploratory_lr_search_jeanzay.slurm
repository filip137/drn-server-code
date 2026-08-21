#!/bin/bash -l
#SBATCH --job-name=pd-c12-lr-directed
#SBATCH --account=fmu@v100
#SBATCH --partition=gpu_p13
#SBATCH --qos=qos_gpu-t3
#SBATCH --constraint=v100-16g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --array=0-11%12
#SBATCH --requeue
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err

set -euo pipefail

: "${C12LR_SOURCE_ROOT:?Set C12LR_SOURCE_ROOT to the frozen staged source.}"
: "${C12LR_SOURCE_ARCHIVE:?Set C12LR_SOURCE_ARCHIVE to the frozen source archive.}"
: "${C12LR_RESULT_ROOT:?Set C12LR_RESULT_ROOT to the Jean Zay result root.}"
: "${C12LR_DATASET_ROOT:?Set C12LR_DATASET_ROOT to the Jean Zay MNIST root.}"
: "${C12LR_ENVIRONMENT_ID:?Set C12LR_ENVIRONMENT_ID.}"
: "${C12LR_STUDY_CONFIG_SHA256:?Set C12LR_STUDY_CONFIG_SHA256.}"
: "${EXPERIMENT_SOURCE_COMMIT:?Set EXPERIMENT_SOURCE_COMMIT.}"
: "${EXPERIMENT_SOURCE_ARCHIVE_SHA256:?Set EXPERIMENT_SOURCE_ARCHIVE_SHA256.}"
: "${SLURM_ARRAY_TASK_ID:?Submit this wrapper as the declared array.}"

C12LR_STUDY_ID="${C12LR_STUDY_ID:-perfectdiode-conv12-directed-exploratory-lr-search-seed0-20260812-v1}"
C12LR_STUDY_CONFIG_RELATIVE="${C12LR_STUDY_CONFIG_RELATIVE:-configs/conv/perfectdiode_conv12_directed_exploratory_lr_search_seed0_20260812_v1.json}"
STUDY_CONFIG="${C12LR_SOURCE_ROOT}/${C12LR_STUDY_CONFIG_RELATIVE}"
TASK_ID=$((10#${SLURM_ARRAY_TASK_ID}))
TASK_ROOT="${C12LR_RESULT_ROOT}/shards/jean-zay-task_${TASK_ID}"
RECEIPT_PATH="${C12LR_RESULT_ROOT}/transport_receipts/jean-zay-task_${TASK_ID}.json"
MODULE_ID="pytorch-gpu/py3/2.5.0"

if (( TASK_ID < 0 || TASK_ID > 11 )); then
  echo "Array task ${TASK_ID} is outside [0,11]." >&2
  exit 3
fi
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

mkdir -p "${TASK_ROOT}" "$(dirname -- "${RECEIPT_PATH}")"
cd "${C12LR_SOURCE_ROOT}"

"${PYTHON_BIN}" - "${STUDY_CONFIG}" "${C12LR_DATASET_ROOT}" "${C12LR_STUDY_ID}" "${TASK_ID}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import torch

study_path = Path(sys.argv[1])
dataset_root = Path(sys.argv[2])
expected_study_id = sys.argv[3]
task = int(sys.argv[4])
study = json.loads(study_path.read_text(encoding="utf-8"))
if study.get("study_id") != expected_study_id:
    raise SystemExit("Unexpected directed study identity.")
if study.get("evidence_class") != "ordinary_mnist_exploratory":
    raise SystemExit("Unexpected evidence class.")
if task < 0 or task >= len(study.get("surfaces", [])):
    raise SystemExit("Task is outside the declared surface list.")
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
    raise SystemExit("CUDA is unavailable in the allocated task.")
name = torch.cuda.get_device_name(0)
memory = int(torch.cuda.get_device_properties(0).total_memory)
if "V100" not in name.upper() or memory < 15 * 1024**3:
    raise SystemExit(f"Expected V100 with at least 15 GiB, got {name!r}/{memory}.")
print(
    f"C12LR_ENVIRONMENT_PASS task={task} python={sys.executable} "
    f"torch={torch.__version__} device={name!r} bytes={memory}",
    flush=True,
)
PY

echo "C12LR_SOURCE_PASS task=${TASK_ID} archive_sha256=${archive_sha} config_sha256=${config_sha}"

"${PYTHON_BIN}" -m experiments.run_conv12_directed_exploratory_lr_search run-surface \
  --study "${STUDY_CONFIG}" \
  --surface-index "${TASK_ID}" \
  --device cuda \
  --target jean-zay \
  --dataset-root "${C12LR_DATASET_ROOT}" \
  --output-root "${TASK_ROOT}"

"${PYTHON_BIN}" - \
  "${STUDY_CONFIG}" "${TASK_ROOT}" "${RECEIPT_PATH}" "${TASK_ID}" \
  "${C12LR_ENVIRONMENT_ID}" "${EXPERIMENT_SOURCE_COMMIT}" \
  "${EXPERIMENT_SOURCE_ARCHIVE_SHA256}" "${C12LR_STUDY_CONFIG_SHA256}" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

from experiments.reporting import sha256_file, validate_run
from experiments.run_conv12_directed_exploratory_lr_search import (
    EXPECTED_TK,
    surface_cells,
    surface_specs,
)

(
    study_arg,
    task_root_arg,
    receipt_arg,
    task_arg,
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
expected_cells = surface_cells(study, surface)
expected_by_index = {row["index"]: row for row in expected_cells}
expected_count = len(expected_cells)
architecture = surface["architecture"]
summary_path = task_root / "surfaces" / surface["surface_id"] / "summary.json"
asset_path = task_root / "assets/bounded_uniform" / architecture / "asset.json"
if not summary_path.is_file() or not asset_path.is_file():
    raise SystemExit("Missing terminal surface summary or initializer asset record.")
summary = json.loads(summary_path.read_text(encoding="utf-8"))
asset = json.loads(asset_path.read_text(encoding="utf-8"))
if summary.get("surface") != surface:
    raise SystemExit("Terminal summary surface mismatch.")
if summary.get("official_test_read") is not False:
    raise SystemExit("Terminal summary does not prove official_test_read=false.")
if summary.get("expected_cells") != expected_count:
    raise SystemExit("Unexpected directed-cell contract in terminal summary.")
expected_initializer = study["parent"]["initializer_checkpoint_sha256_by_architecture"][architecture]
if asset.get("checkpoint_sha256") != expected_initializer:
    raise SystemExit("Initializer checkpoint SHA mismatch in terminal shard.")
asset_checkpoint = Path(asset.get("checkpoint", ""))
if not asset_checkpoint.is_file() or sha256_file(asset_checkpoint) != expected_initializer:
    raise SystemExit("Initializer checkpoint is missing or does not match its asset record.")

summary_status = summary.get("status")
candidates = summary.get("candidates")
if not isinstance(candidates, list):
    raise SystemExit("Terminal summary candidates must be a list.")
validated_cell_bundles = 0
if summary_status == "complete":
    if summary.get("complete") is not True or summary.get("terminal_cells") != expected_count:
        raise SystemExit("Incomplete directed surface.")
    if len(candidates) != expected_count or {row.get("index") for row in candidates} != set(expected_by_index):
        raise SystemExit("Complete surface does not contain exactly the declared sparse cells.")
    task_root_resolved = task_root.resolve()
    expected_t, expected_k = EXPECTED_TK[architecture]
    for candidate in candidates:
        run_dir = Path(str(candidate.get("path", ""))).resolve()
        if not run_dir.is_relative_to(task_root_resolved):
            raise SystemExit(f"Cell path escapes its task root: {run_dir}")
        validation_errors = validate_run(run_dir)
        if validation_errors:
            raise SystemExit(f"Canonical cell validation failed for {run_dir}: {validation_errors!r}")
        cell = json.loads((run_dir / "cell.json").read_text(encoding="utf-8"))
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        run_status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
        expected = expected_by_index.get(candidate.get("index"))
        if expected is None or cell.get("index") != candidate.get("index"):
            raise SystemExit(f"Cell and summary identity disagree for {run_dir}.")
        if float(cell.get("rho_conv")) != float(expected["rho_conv"]) or float(cell.get("rho_dense")) != float(expected["rho_dense"]):
            raise SystemExit(f"Sparse coordinate mismatch for {run_dir}.")
        if cell.get("status") != candidate.get("status") or cell.get("optimizer") != surface["optimizer"]:
            raise SystemExit(f"Cell status/optimizer mismatch for {run_dir}.")
        signature = cell.get("signature")
        if not isinstance(signature, dict) or signature.get("evidence_class") != study["evidence_class"]:
            raise SystemExit(f"Cell signature evidence class mismatch for {run_dir}.")
        rates = cell.get("learning_rates_by_parameter")
        bias_rates = {
            name: value
            for name, value in (rates.items() if isinstance(rates, dict) else ())
            if str(name).startswith("Bias_")
        }
        if not bias_rates or any(float(value) != 0.0 for value in bias_rates.values()):
            raise SystemExit(f"Cell does not retain exact-zero bias rates: {run_dir}.")
        if manifest.get("evidence_class") != study["evidence_class"] or manifest.get("study_id") != study["study_id"]:
            raise SystemExit(f"Cell manifest study/evidence mismatch for {run_dir}.")
        if manifest.get("dataset", {}).get("official_test_read") is not False:
            raise SystemExit(f"Cell manifest does not prove official_test_read=false: {run_dir}.")
        configuration = manifest.get("configuration", {})
        model = configuration.get("resolved", {}).get("model_base", {})
        if (
            configuration.get("optimizer") != surface["optimizer"]
            or float(model.get("weight_min", float("nan"))) != 1e-5
            or float(model.get("weight_max", float("nan"))) != 1e-4
            or int(model.get("num_iterations_inference", -1)) != expected_t
            or int(model.get("num_iterations_training", -1)) != expected_k
        ):
            raise SystemExit(f"Cell scientific contract mismatch for {run_dir}.")
        cell_status = cell.get("status")
        if cell_status == "complete":
            result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
            if run_status.get("state") != "complete" or result.get("evidence_class") != study["evidence_class"]:
                raise SystemExit(f"Successful cell reporting mismatch for {run_dir}.")
            if result.get("dataset", {}).get("official_test_read") is not False:
                raise SystemExit(f"Successful cell read the official test split: {run_dir}.")
        elif cell_status in {"candidate_rejected_nonfinite", "candidate_rejected_safety"}:
            failure = cell.get("safety_failure")
            failure_kind = failure.get("kind") if isinstance(failure, dict) else None
            if run_status.get("state") != "failed" or failure_kind not in {
                "nonfinite_diagnostic",
                "nonfinite_training_value",
            }:
                raise SystemExit(f"Rejected cell is not a declared non-finite terminal: {run_dir}.")
        else:
            raise SystemExit(f"Unexpected terminal cell status {cell_status!r}: {run_dir}.")
        validated_cell_bundles += 1
elif summary_status == "unresolved_probe":
    probe_path = task_root / "surfaces" / surface["surface_id"] / "rho" / "probe.json"
    if (
        summary.get("terminal_cells") != 0
        or candidates
        or not probe_path.is_file()
        or json.loads(probe_path.read_text(encoding="utf-8")).get("status") == "complete"
    ):
        raise SystemExit("Malformed unresolved-probe surface.")
else:
    raise SystemExit(f"Surface did not reach a supported terminal outcome: {summary_status!r}")

receipt = {
    "schema_version": "perfectdiode-conv12-directed-exploratory-lr-transport/v1",
    "study_id": study["study_id"],
    "surface": surface,
    "task_id": task,
    "target": "jean-zay",
    "environment_id": environment_id,
    "source_commit": source_commit,
    "source_archive_sha256": source_archive_sha256,
    "study_config_sha256": study_config_sha256,
    "initializer_checkpoint_sha256": asset["checkpoint_sha256"],
    "summary_status": summary_status,
    "expected_cells": expected_count,
    "terminal_cells": summary["terminal_cells"],
    "validated_cell_bundles": validated_cell_bundles,
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
    f"C12LR_SEMANTIC_PASS task={task} surface={surface['surface_id']} "
    f"outcome={summary_status} validated_cells={validated_cell_bundles}",
    flush=True,
)
PY
