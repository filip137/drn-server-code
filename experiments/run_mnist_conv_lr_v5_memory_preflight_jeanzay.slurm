#!/bin/bash -l
#SBATCH --job-name=conv-lr-v5-mem
#SBATCH --output=slurm/conv-lr-v5/%x-%j.out
#SBATCH --error=slurm/conv-lr-v5/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=30
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --account=fmu@v100
#SBATCH --qos=qos_gpu-t3
#SBATCH --partition=gpu_p13
#SBATCH --constraint=v100-32g

set -euo pipefail

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
  MNIST_CONV_LR_MEMORY_PREFLIGHT \
  MNIST_CONV_LR_MEMORY_PREFLIGHT_STEPS \
  MNIST_CONV_LR_STAGE \
  MNIST_CONV_DATASET_ROOT \
  MNIST_CONV_DEVICE
do
  require_env "${name}"
done

if [[ "${MNIST_CONV_LR_STAGE}" != "candidates" ]]; then
  fail_value "MNIST_CONV_LR_STAGE to be candidates for the v5 memory preflight" "${MNIST_CONV_LR_STAGE}"
fi
if [[ "${MNIST_CONV_DEVICE}" != "cuda" && "${MNIST_CONV_DEVICE}" != "cuda:0" ]]; then
  fail_value "MNIST_CONV_DEVICE to identify the single allocated CUDA device (cuda or cuda:0)" "${MNIST_CONV_DEVICE}"
fi
if [[ ! "${MNIST_CONV_LR_MEMORY_PREFLIGHT_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
  fail_value "MNIST_CONV_LR_MEMORY_PREFLIGHT_STEPS to be a positive integer" "${MNIST_CONV_LR_MEMORY_PREFLIGHT_STEPS}"
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
if [[ ! -d "${MNIST_CONV_DATASET_ROOT}" ]]; then
  fail_value "MNIST_CONV_DATASET_ROOT to be an existing train-data directory" "${MNIST_CONV_DATASET_ROOT}"
fi

project_results_root="/lustre/fsn1/projects/rech/fmu/${USER:?}/server_code/results"
resolved_paths="$("${MNIST_CONV_PYTHON}" -c '
import json
import sys
from pathlib import Path

study = Path(sys.argv[1]).resolve()
stage_supplied = Path(sys.argv[2]).absolute()
output_supplied = Path(sys.argv[3]).absolute()
for label, path in (("candidate-stage manifest", stage_supplied), ("memory-preflight output", output_supplied)):
    if path.is_symlink():
        raise SystemExit(
            f"Expected the {label} not to be a symlink. Provided value: {path}."
        )
stage_path = stage_supplied.resolve()
output_path = output_supplied.resolve()
results = Path(sys.argv[4]).resolve()
try:
    study.relative_to(results)
except ValueError as exc:
    raise SystemExit(
        f"Expected the v5 study to be under the fmu project results root {results}. "
        f"Provided value: {study}."
    ) from exc
for label, path in (("candidate-stage manifest", stage_path), ("memory-preflight output", output_path)):
    try:
        path.relative_to(study)
    except ValueError as exc:
        raise SystemExit(
            f"Expected the {label} to be contained by the v5 study {study}. "
            f"Provided value: {path}."
        ) from exc
if stage_path == output_path:
    raise SystemExit(
        "Expected the memory-preflight output to differ from the candidate-stage manifest. "
        f"Provided value: {output_path}."
    )
contract_path = study / "study.resolved.json"
try:
    contract = json.loads(contract_path.read_text())
    stage = json.loads(stage_path.read_text())
except (OSError, json.JSONDecodeError) as exc:
    raise SystemExit(
        f"Expected the v5 contract and candidate manifest to be readable valid JSON. "
        f"Provided value: {exc}."
    ) from exc
schema_version = contract.get("schema_version") if isinstance(contract, dict) else contract
if not isinstance(contract, dict) or schema_version != "mnist-conv-lr-study/v5":
    raise SystemExit(
        "Expected study.resolved.json to declare mnist-conv-lr-study/v5. "
        f"Provided value: {schema_version!r}."
    )
official_test = contract.get("dataset", {}).get("official_test")
if official_test != {"enabled": False, "read_allowed": False}:
    raise SystemExit(
        "Expected the v5 study to prohibit official MNIST test reads. "
        f"Provided value: {official_test!r}."
    )
stage_name = stage.get("stage_name") if isinstance(stage, dict) else stage
if not isinstance(stage, dict) or stage_name != "candidates":
    raise SystemExit(
        "Expected the immutable stage manifest to describe candidates. "
        f"Provided value: {stage_name!r}."
    )
expected_study_id = study.name.rsplit("--", 1)[-1]
stage_study_id = stage.get("study_id")
if not expected_study_id.startswith("lrstudy_") or stage_study_id != expected_study_id:
    raise SystemExit(
        "Expected the candidate manifest study_id to match the content-addressed study directory. "
        f"Provided value: stage={stage_study_id!r}, directory={expected_study_id!r}."
    )
print(study)
print(stage_path)
print(output_path)
' "${MNIST_CONV_LR_STUDY}" "${MNIST_CONV_LR_MANIFEST}" "${MNIST_CONV_LR_MEMORY_PREFLIGHT}" "${project_results_root}")"
mapfile -t canonical_paths <<< "${resolved_paths}"
MNIST_CONV_LR_STUDY="${canonical_paths[0]}"
MNIST_CONV_LR_MANIFEST="${canonical_paths[1]}"
MNIST_CONV_LR_MEMORY_PREFLIGHT="${canonical_paths[2]}"

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

cd "${MNIST_CONV_REPO_ROOT}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${MNIST_CONV_REPO_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

printf '[v5-memory-preflight] job=%s account=%s constraint=%s gpu=%s memory_mib=%s\n' \
  "${SLURM_JOB_ID:-manual}" "${runtime_account}" "${runtime_constraint}" "${gpu_name}" "${gpu_memory_mib}"
printf '[v5-memory-preflight] study=%s candidate_manifest=%s output=%s steps=%s\n' \
  "${MNIST_CONV_LR_STUDY}" "${MNIST_CONV_LR_MANIFEST}" "${MNIST_CONV_LR_MEMORY_PREFLIGHT}" "${MNIST_CONV_LR_MEMORY_PREFLIGHT_STEPS}"

# The Python helper calls build_loader_bundle(download=False), which constructs
# only MNIST's train split.  The 5,000-example validation and official test
# splits are not iterated during this GPU preflight.
exec "${MNIST_CONV_PYTHON}" -m experiments.run_mnist_conv_lr_v5_memory_preflight \
  --study "${MNIST_CONV_LR_STUDY}" \
  --candidate-manifest "${MNIST_CONV_LR_MANIFEST}" \
  --data-root "${MNIST_CONV_DATASET_ROOT}" \
  --output "${MNIST_CONV_LR_MEMORY_PREFLIGHT}" \
  --device "${MNIST_CONV_DEVICE}" \
  --steps-per-runtime "${MNIST_CONV_LR_MEMORY_PREFLIGHT_STEPS}"
