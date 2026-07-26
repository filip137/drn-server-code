#!/bin/bash -l
#SBATCH --job-name=conv-lr-v5-pack
#SBATCH --output=slurm/conv-lr-v5/%x-%A_%a.out
#SBATCH --error=slurm/conv-lr-v5/%x-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=30
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --account=fmu@v100
#SBATCH --qos=qos_gpu-t3
#SBATCH --partition=gpu_p13
#SBATCH --constraint=v100-32g

set -euo pipefail

# One Slurm array task consumes one entry from an immutable pack manifest.  The
# pack manifest is separate from the immutable candidate-stage manifest: it
# records only execution packing and the memory preflight that justified it.
# Each child still executes the canonical candidate manifest by entry index.

fail_value() {
  local expected="$1"
  local provided="$2"
  printf 'Expected %s.\n' "${expected}" >&2
  printf 'Provided value: %s\n' "${provided}" >&2
  exit 2
}

require_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    fail_value "environment variable ${name} to be set" "${!name-<unset>}"
  fi
}

for name in \
  MNIST_CONV_REPO_ROOT \
  MNIST_CONV_PYTHON \
  MNIST_CONV_LR_STUDY \
  MNIST_CONV_LR_MANIFEST \
  MNIST_CONV_LR_PACK_MANIFEST \
  MNIST_CONV_LR_STAGE \
  MNIST_CONV_DATASET_ROOT \
  MNIST_CONV_DEVICE \
  SLURM_ARRAY_TASK_ID
do
  require_env "${name}"
done

if [[ "${MNIST_CONV_LR_STAGE}" != "candidates" ]]; then
  fail_value "MNIST_CONV_LR_STAGE to be candidates for the v5 packed launcher" "${MNIST_CONV_LR_STAGE}"
fi
if [[ "${MNIST_CONV_DEVICE}" != "cuda" && "${MNIST_CONV_DEVICE}" != "cuda:0" ]]; then
  fail_value "MNIST_CONV_DEVICE to identify the single allocated CUDA device (cuda or cuda:0)" "${MNIST_CONV_DEVICE}"
fi

runtime_account="${SLURM_JOB_ACCOUNT:-${MNIST_CONV_SLURM_ACCOUNT:-}}"
if [[ "${runtime_account}" != "fmu@v100" ]]; then
  fail_value "the Slurm account to be fmu@v100 (SLURM_JOB_ACCOUNT, or MNIST_CONV_SLURM_ACCOUNT when Slurm does not export it)" "${runtime_account:-<unset>}"
fi

runtime_constraint="${SLURM_CONSTRAINT:-${SLURM_JOB_CONSTRAINTS:-${MNIST_CONV_SLURM_CONSTRAINT:-}}}"
if [[ "${runtime_constraint}" != "v100-32g" ]]; then
  fail_value "the Slurm constraint to be v100-32g (SLURM_CONSTRAINT/SLURM_JOB_CONSTRAINTS, or MNIST_CONV_SLURM_CONSTRAINT when Slurm does not export it)" "${runtime_constraint:-<unset>}"
fi

runtime_gpu_count="${SLURM_GPUS_ON_NODE:-${MNIST_CONV_SLURM_GPUS:-}}"
if [[ "${runtime_gpu_count}" != "1" ]]; then
  fail_value "exactly one allocated GPU (SLURM_GPUS_ON_NODE, or MNIST_CONV_SLURM_GPUS when Slurm does not export it)" "${runtime_gpu_count:-<unset>}"
fi

require_env CUDA_VISIBLE_DEVICES
IFS=',' read -r -a visible_gpus <<< "${CUDA_VISIBLE_DEVICES}"
if [[ "${#visible_gpus[@]}" -ne 1 || -z "${visible_gpus[0]}" ]]; then
  fail_value "CUDA_VISIBLE_DEVICES to contain exactly one device" "${CUDA_VISIBLE_DEVICES}"
fi

idrenv_project="${MNIST_CONV_IDRENV_PROJECT:-fmu}"
if [[ "${idrenv_project}" != "fmu" ]]; then
  fail_value "MNIST_CONV_IDRENV_PROJECT to be fmu" "${idrenv_project}"
fi
if [[ -x /gpfslocalsup/bin/idrenv ]]; then
  eval "$(/gpfslocalsup/bin/idrenv -d "${idrenv_project}")"
elif command -v idrenv >/dev/null 2>&1; then
  eval "$(idrenv -d "${idrenv_project}")"
fi

if [[ -n "${MNIST_CONV_MODULE:-}" ]]; then
  module purge
  module load "${MNIST_CONV_MODULE}"
fi

if [[ ! -x "${MNIST_CONV_PYTHON}" ]] && ! command -v "${MNIST_CONV_PYTHON}" >/dev/null 2>&1; then
  fail_value "MNIST_CONV_PYTHON to name an executable Python interpreter" "${MNIST_CONV_PYTHON}"
fi
if [[ ! -d "${MNIST_CONV_REPO_ROOT}" ]]; then
  fail_value "MNIST_CONV_REPO_ROOT to be an existing directory" "${MNIST_CONV_REPO_ROOT}"
fi
if [[ ! -d "${MNIST_CONV_LR_STUDY}" ]]; then
  fail_value "MNIST_CONV_LR_STUDY to be an existing study directory" "${MNIST_CONV_LR_STUDY}"
fi
if [[ ! -f "${MNIST_CONV_LR_MANIFEST}" ]]; then
  fail_value "MNIST_CONV_LR_MANIFEST to be an existing candidate-stage manifest" "${MNIST_CONV_LR_MANIFEST}"
fi
if [[ ! -f "${MNIST_CONV_LR_PACK_MANIFEST}" ]]; then
  fail_value "MNIST_CONV_LR_PACK_MANIFEST to be an existing immutable pack manifest" "${MNIST_CONV_LR_PACK_MANIFEST}"
fi

project_results_root="/lustre/fsn1/projects/rech/fmu/${USER:?}/server_code/results"
resolved_paths="$("${MNIST_CONV_PYTHON}" -c '
from pathlib import Path
import sys

study = Path(sys.argv[1]).resolve()
stage = Path(sys.argv[2]).resolve()
pack = Path(sys.argv[3]).resolve()
results = Path(sys.argv[4]).resolve()
try:
    study.relative_to(results)
except ValueError as exc:
    raise SystemExit(
        f"Expected the v5 study to be under the fmu project results root {results}. "
        f"Provided value: {study}."
    ) from exc
for label, path in (("candidate-stage manifest", stage), ("pack manifest", pack)):
    try:
        path.relative_to(study)
    except ValueError as exc:
        raise SystemExit(
            f"Expected the {label} to be contained by the v5 study {study}. "
            f"Provided value: {path}."
        ) from exc
print(study)
print(stage)
print(pack)
' "${MNIST_CONV_LR_STUDY}" "${MNIST_CONV_LR_MANIFEST}" "${MNIST_CONV_LR_PACK_MANIFEST}" "${project_results_root}")"
mapfile -t canonical_paths <<< "${resolved_paths}"
MNIST_CONV_LR_STUDY="${canonical_paths[0]}"
MNIST_CONV_LR_MANIFEST="${canonical_paths[1]}"
MNIST_CONV_LR_PACK_MANIFEST="${canonical_paths[2]}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  fail_value "nvidia-smi to be available for the V100-32GB allocation check" "not found"
fi
gpu_record="$(nvidia-smi --id="${visible_gpus[0]}" --query-gpu=name,memory.total --format=csv,noheader,nounits)"
if [[ "${gpu_record}" == *$'\n'* ]]; then
  fail_value "the allocation to expose one GPU record" "${gpu_record}"
fi
IFS=',' read -r gpu_name gpu_memory_mib <<< "${gpu_record}"
gpu_name="${gpu_name#${gpu_name%%[![:space:]]*}}"
gpu_name="${gpu_name%${gpu_name##*[![:space:]]}}"
gpu_memory_mib="${gpu_memory_mib//[[:space:]]/}"
if [[ "${gpu_name}" != *V100* ]]; then
  fail_value "the allocated GPU model to be a V100" "${gpu_name}"
fi
if [[ ! "${gpu_memory_mib}" =~ ^[0-9]+$ ]] || (( gpu_memory_mib < 32000 || gpu_memory_mib > 33000 )); then
  fail_value "the allocated V100 to report approximately 32 GiB (32000-33000 MiB)" "${gpu_memory_mib:-<unset>} MiB"
fi

pack_record="$("${MNIST_CONV_PYTHON}" -c '
import hashlib
import json
import math
import sys
from pathlib import Path

pack_path = Path(sys.argv[1])
stage_path = Path(sys.argv[2])
try:
    pack_index = int(sys.argv[3])
except ValueError as exc:
    raise SystemExit(
        "Expected SLURM_ARRAY_TASK_ID to be a non-negative integer. "
        f"Provided value: {sys.argv[3]!r}."
    ) from exc
if pack_index < 0:
    raise SystemExit(
        "Expected SLURM_ARRAY_TASK_ID to be a non-negative integer. "
        f"Provided value: {pack_index!r}."
    )

def read_object(path, label):
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(
            f"Expected the {label} at {path} to be valid JSON. Provided value: {exc}."
        ) from exc
    if not isinstance(value, dict):
        raise SystemExit(
            f"Expected the {label} at {path} to be a JSON object. Provided value: {value!r}."
        )
    return value

pack_manifest = read_object(pack_path, "v5 pack manifest")
stage_manifest = read_object(stage_path, "candidate-stage manifest")
required_root = {
    "schema_version", "study_id", "stage_name", "stage_manifest_sha256",
    "gpu_contract", "packs",
}
missing = sorted(required_root - set(pack_manifest))
if missing:
    raise SystemExit(
        f"Expected the v5 pack manifest to contain keys {sorted(required_root)!r}. "
        f"Provided value: missing {missing!r}."
    )
if pack_manifest["schema_version"] != "mnist-conv-lr-pack-manifest/v1":
    raise SystemExit(
        "Expected the pack-manifest schema to be mnist-conv-lr-pack-manifest/v1. "
        f"Provided value: {pack_manifest['schema_version']!r}."
    )
if pack_manifest["stage_name"] != "candidates" or stage_manifest.get("stage_name") != "candidates":
    raise SystemExit(
        "Expected both manifests to describe the candidates stage. "
        f"Provided value: pack={pack_manifest['stage_name']!r}, "
        f"stage={stage_manifest.get('stage_name')!r}."
    )
if pack_manifest["study_id"] != stage_manifest.get("study_id"):
    raise SystemExit(
        "Expected the pack and candidate-stage manifests to have the same study_id. "
        f"Provided value: pack={pack_manifest['study_id']!r}, "
        f"stage={stage_manifest.get('study_id')!r}."
    )
stage_sha = hashlib.sha256(stage_path.read_bytes()).hexdigest()
if pack_manifest["stage_manifest_sha256"] != stage_sha:
    raise SystemExit(
        "Expected stage_manifest_sha256 to match the current candidate-stage manifest. "
        f"Provided value: declared={pack_manifest['stage_manifest_sha256']!r}, "
        f"observed={stage_sha!r}."
    )

expected_gpu = {
    "account": "fmu@v100",
    "constraint": "v100-32g",
    "gpus": 1,
    "capacity_mib": 32768,
    "headroom_fraction": 0.1,
}
if pack_manifest["gpu_contract"] != expected_gpu:
    raise SystemExit(
        f"Expected gpu_contract to be {expected_gpu!r}. "
        f"Provided value: {pack_manifest['gpu_contract']!r}."
    )

entries = stage_manifest.get("entries")
if not isinstance(entries, list) or not entries:
    raise SystemExit(
        "Expected the candidate-stage manifest entries to be a non-empty list. "
        f"Provided value: {entries!r}."
    )
for expected_index, entry in enumerate(entries):
    if not isinstance(entry, dict) or entry.get("entry_index") != expected_index:
        raise SystemExit(
            "Expected candidate-stage entry indices to be contiguous from zero. "
            f"Provided value at index {expected_index}: {entry!r}."
        )

packs = pack_manifest["packs"]
if not isinstance(packs, list) or not packs:
    raise SystemExit(
        f"Expected packs to be a non-empty list. Provided value: {packs!r}."
    )
seen = []
for expected_index, pack in enumerate(packs):
    required_pack = {
        "pack_index", "architecture", "entry_indices",
        "measured_combined_peak_gpu_memory_mib",
    }
    if not isinstance(pack, dict) or not required_pack.issubset(pack):
        raise SystemExit(
            f"Expected packs[{expected_index}] to contain keys {sorted(required_pack)!r}. "
            f"Provided value: {pack!r}."
        )
    if pack["pack_index"] != expected_index:
        raise SystemExit(
            f"Expected packs[{expected_index}].pack_index to be {expected_index}. "
            f"Provided value: {pack['pack_index']!r}."
        )
    architecture = pack["architecture"]
    indices = pack["entry_indices"]
    allowed_sizes = {"conv1": {2, 3}, "conv2": {2, 3}}
    if architecture not in allowed_sizes:
        raise SystemExit(
            "Expected pack architecture to be conv1 or conv2. "
            f"Provided value: {architecture!r}."
        )
    if (
        not isinstance(indices, list)
        or len(indices) not in allowed_sizes[architecture]
        or any(type(value) is not int for value in indices)
        or len(indices) != len(set(indices))
    ):
        raise SystemExit(
            f"Expected {architecture} entry_indices to contain "
            f"{sorted(allowed_sizes[architecture])!r} unique integers. "
            f"Provided value: {indices!r}."
        )
    for entry_index in indices:
        if not 0 <= entry_index < len(entries):
            raise SystemExit(
                f"Expected packed entry index in [0, {len(entries) - 1}]. "
                f"Provided value: {entry_index!r}."
            )
        row_id = entries[entry_index].get("payload", {}).get("row_id")
        if not isinstance(row_id, str) or not row_id.startswith(architecture + "_"):
            raise SystemExit(
                f"Expected entry {entry_index} row_id to belong to {architecture}. "
                f"Provided value: {row_id!r}."
            )
    measured = pack["measured_combined_peak_gpu_memory_mib"]
    if isinstance(measured, bool) or not isinstance(measured, (int, float)) or not math.isfinite(measured) or measured <= 0:
        raise SystemExit(
            "Expected measured_combined_peak_gpu_memory_mib to be a positive finite number. "
            f"Provided value: {measured!r}."
        )
    required = float(measured) * 1.1
    if required > 32768:
        raise SystemExit(
            "Expected measured combined peak GPU memory plus 10% headroom to be at most 32768 MiB. "
            f"Provided value: measured={measured!r}, required={required!r}."
        )
    seen.extend(indices)

if sorted(seen) != list(range(len(entries))):
    raise SystemExit(
        "Expected the pack manifest to cover every candidate-stage entry exactly once. "
        f"Provided value: packed={sorted(seen)!r}, expected={list(range(len(entries)))!r}."
    )
if pack_index >= len(packs):
    raise SystemExit(
        f"Expected SLURM_ARRAY_TASK_ID in [0, {len(packs) - 1}]. "
        f"Provided value: {pack_index!r}."
    )

selected = packs[pack_index]
print(selected["architecture"] + "\t" + str(selected["measured_combined_peak_gpu_memory_mib"]) + "\t" + ",".join(map(str, selected["entry_indices"])))
' "${MNIST_CONV_LR_PACK_MANIFEST}" "${MNIST_CONV_LR_MANIFEST}" "${SLURM_ARRAY_TASK_ID}")"

IFS=$'\t' read -r pack_architecture measured_peak_mib packed_indices <<< "${pack_record}"
IFS=',' read -r -a entry_indices <<< "${packed_indices}"

cd "${MNIST_CONV_REPO_ROOT}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${MNIST_CONV_REPO_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

array_job_id="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}"
log_dir="${MNIST_CONV_LR_STUDY}/slurm/candidates/packs/${array_job_id}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${log_dir}"

printf '[v5-pack] job=%s array_task=%s architecture=%s entries=%s\n' \
  "${SLURM_JOB_ID:-manual}" "${SLURM_ARRAY_TASK_ID}" "${pack_architecture}" "${packed_indices}"
printf '[v5-pack] account=%s constraint=%s gpu=%s memory_mib=%s measured_pack_peak_mib=%s\n' \
  "${runtime_account}" "${runtime_constraint}" "${gpu_name}" "${gpu_memory_mib}" "${measured_peak_mib}"
printf '[v5-pack] study=%s stage_manifest=%s pack_manifest=%s\n' \
  "${MNIST_CONV_LR_STUDY}" "${MNIST_CONV_LR_MANIFEST}" "${MNIST_CONV_LR_PACK_MANIFEST}"

pids=()
labels=()
terminate_children() {
  trap - HUP INT TERM
  local pid
  for pid in "${pids[@]:-}"; do
    kill "${pid}" 2>/dev/null || true
  done
  wait 2>/dev/null || true
  exit 130
}
trap terminate_children HUP INT TERM

for entry_index in "${entry_indices[@]}"; do
  entry_log="${log_dir}/entry_${entry_index}.log"
  (
    set -o pipefail
    "${MNIST_CONV_PYTHON}" -m experiments.mnist_conv lr-study \
      --stage candidates \
      --study "${MNIST_CONV_LR_STUDY}" \
      --manifest "${MNIST_CONV_LR_MANIFEST}" \
      --entry-index "${entry_index}" \
      --data-root "${MNIST_CONV_DATASET_ROOT}" \
      --device "${MNIST_CONV_DEVICE}" \
      2>&1 | sed -u "s/^/[entry ${entry_index}] /" | tee "${entry_log}"
  ) &
  pids+=("$!")
  labels+=("${entry_index}")
done

failures=0
for position in "${!pids[@]}"; do
  if wait "${pids[${position}]}"; then
    printf '[v5-pack] entry=%s status=complete\n' "${labels[${position}]}"
  else
    status=$?
    printf '[v5-pack] entry=%s status=failed exit=%s\n' "${labels[${position}]}" "${status}" >&2
    failures=$((failures + 1))
  fi
done
trap - HUP INT TERM

if (( failures > 0 )); then
  fail_value "every packed candidate entry to complete successfully" "${failures} of ${#entry_indices[@]} entries failed"
fi

printf '[v5-pack] status=complete architecture=%s entries=%s\n' "${pack_architecture}" "${packed_indices}"
