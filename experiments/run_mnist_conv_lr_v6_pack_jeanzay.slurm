#!/bin/bash -l
#SBATCH --job-name=conv-lr-v6-pack
#SBATCH --output=slurm/conv-lr-v6/%x-%j.out
#SBATCH --error=slurm/conv-lr-v6/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --account=fmu@v100
#SBATCH --partition=gpu_p13
#SBATCH --qos=qos_gpu-t3
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
  MNIST_CONV_LR_PACK_MANIFEST \
  MNIST_CONV_LR_STAGE \
  MNIST_CONV_DATASET_ROOT \
  MNIST_CONV_DEVICE \
  MNIST_CONV_R3_ALLOCATION \
  MNIST_CONV_V6_R3_AUTHORIZATION \
  MNIST_CONV_IDRENV_PROJECT \
  SLURM_JOB_ID \
  SLURM_JOB_ACCOUNT \
  SLURM_JOB_PARTITION \
  SLURM_JOB_QOS \
  SLURM_CPUS_PER_TASK \
  SLURM_GPUS_ON_NODE
do
  require_env "${name}"
done
authorization_expected="AD010913993R3:fmu@v100:gpu_p13:qos_gpu-t3:v100-32g:gpu1"
if [[ "${MNIST_CONV_V6_R3_AUTHORIZATION}" != "${authorization_expected}" ]]; then
  fail_value "MNIST_CONV_V6_R3_AUTHORIZATION to match the frozen one-GPU R3 contract" "${MNIST_CONV_V6_R3_AUTHORIZATION}"
fi

case "${MNIST_CONV_LR_STAGE}" in
  audit|probe|baseline_candidates|confirmations) ;;
  *) fail_value "MNIST_CONV_LR_STAGE to be audit, probe, baseline_candidates, or confirmations" "${MNIST_CONV_LR_STAGE}" ;;
esac
v5_parent_expected="/lustre/fsn1/projects/rech/fmu/${USER:?}/server_code/results/lrstudy_5da3a7452c00d42325bfe930d80f037791125d3ce7b5ddd79be13ff8b96b49e2/lr_studies/conv1-conv2-hard-sigmoid-ordinary-mnist-architecture-relative-median-study--lrstudy_5da3a7452c00d42325bfe930d80f037791125d3ce7b5ddd79be13ff8b96b49e2"
if [[ "${MNIST_CONV_LR_STAGE}" == "audit" ]]; then
  require_env MNIST_CONV_LR_V5_STUDY
  if [[ "${MNIST_CONV_LR_V5_STUDY}" != "${v5_parent_expected}" ]]; then
    fail_value "MNIST_CONV_LR_V5_STUDY to be the frozen content-addressed v5 R3 parent" "${MNIST_CONV_LR_V5_STUDY}"
  fi
  if [[ ! -d "${MNIST_CONV_LR_V5_STUDY}" ]]; then
    fail_value "the frozen content-addressed v5 R3 parent to exist" "${MNIST_CONV_LR_V5_STUDY}"
  fi
elif [[ -n "${MNIST_CONV_LR_V5_STUDY:-}" ]]; then
  fail_value "MNIST_CONV_LR_V5_STUDY only for the audit stage" "${MNIST_CONV_LR_V5_STUDY}"
fi
if [[ "${MNIST_CONV_R3_ALLOCATION}" != "AD010913993R3" || "${MNIST_CONV_R3_ALLOCATION}" != *R3 ]]; then
  fail_value "the allocation to be AD010913993R3 and end in R3" "${MNIST_CONV_R3_ALLOCATION}"
fi
if [[ "${MNIST_CONV_IDRENV_PROJECT}" != "fmu" ]]; then
  fail_value "MNIST_CONV_IDRENV_PROJECT to be fmu" "${MNIST_CONV_IDRENV_PROJECT}"
fi
execution_override_mode="${MNIST_CONV_EXECUTION_OVERRIDE_MODE:-}"
execution_allocation="${MNIST_CONV_R3_ALLOCATION}"
execution_project="${MNIST_CONV_IDRENV_PROJECT}"
execution_account_expected="fmu@v100"
if [[ -n "${execution_override_mode}" ]]; then
  for name in \
    MNIST_CONV_EXECUTION_OVERRIDE_AUTHORIZATION \
    MNIST_CONV_EXECUTION_ALLOCATION_OVERRIDE \
    MNIST_CONV_EXECUTION_PROJECT_OVERRIDE \
    MNIST_CONV_EXECUTION_ACCOUNT_OVERRIDE
  do
    require_env "${name}"
  done
  override_mode_expected="user-approved-umg-v100-20260721"
  override_authorization_expected="AD011016471R1:umg@v100:gpu_p13:qos_gpu-t3:v100-32g:gpu1:user-approved-20260721"
  if [[ "${execution_override_mode}" != "${override_mode_expected}" ]]; then
    fail_value "MNIST_CONV_EXECUTION_OVERRIDE_MODE to identify the user-approved UMG V100 override" "${execution_override_mode}"
  fi
  if [[ "${MNIST_CONV_EXECUTION_OVERRIDE_AUTHORIZATION}" != "${override_authorization_expected}" ]]; then
    fail_value "MNIST_CONV_EXECUTION_OVERRIDE_AUTHORIZATION to match the user-approved UMG V100 contract" "${MNIST_CONV_EXECUTION_OVERRIDE_AUTHORIZATION}"
  fi
  if [[ "${MNIST_CONV_EXECUTION_ALLOCATION_OVERRIDE}" != "AD011016471R1" ]]; then
    fail_value "MNIST_CONV_EXECUTION_ALLOCATION_OVERRIDE to be AD011016471R1" "${MNIST_CONV_EXECUTION_ALLOCATION_OVERRIDE}"
  fi
  if [[ "${MNIST_CONV_EXECUTION_PROJECT_OVERRIDE}" != "umg" ]]; then
    fail_value "MNIST_CONV_EXECUTION_PROJECT_OVERRIDE to be umg" "${MNIST_CONV_EXECUTION_PROJECT_OVERRIDE}"
  fi
  if [[ "${MNIST_CONV_EXECUTION_ACCOUNT_OVERRIDE}" != "umg@v100" ]]; then
    fail_value "MNIST_CONV_EXECUTION_ACCOUNT_OVERRIDE to be umg@v100" "${MNIST_CONV_EXECUTION_ACCOUNT_OVERRIDE}"
  fi
  execution_allocation="${MNIST_CONV_EXECUTION_ALLOCATION_OVERRIDE}"
  execution_project="${MNIST_CONV_EXECUTION_PROJECT_OVERRIDE}"
  execution_account_expected="${MNIST_CONV_EXECUTION_ACCOUNT_OVERRIDE}"
fi
runtime_account="${SLURM_JOB_ACCOUNT}"
if [[ "${runtime_account}" != "${execution_account_expected}" ]]; then
  fail_value "the live Slurm account to match the selected execution contract" "${runtime_account:-<unset>}"
fi
runtime_partition="${SLURM_JOB_PARTITION}"
if [[ "${runtime_partition}" != "gpu_p13" ]]; then
  fail_value "the Slurm partition to be gpu_p13" "${runtime_partition:-<unset>}"
fi
runtime_qos="${SLURM_JOB_QOS}"
if [[ "${runtime_qos}" != "qos_gpu-t3" ]]; then
  fail_value "the Slurm QOS to be qos_gpu-t3" "${runtime_qos:-<unset>}"
fi
if ! command -v scontrol >/dev/null 2>&1; then
  fail_value "scontrol to verify the live Slurm job constraint" "not found"
fi
runtime_job_record="$(scontrol show job "${SLURM_JOB_ID}" -o)"
if [[ " ${runtime_job_record} " != *" Features=v100-32g "* ]]; then
  fail_value "the live Slurm job constraint to be v100-32g" "${runtime_job_record}"
fi
runtime_gpu_count="${SLURM_GPUS_ON_NODE}"
if [[ "${runtime_gpu_count}" != "1" ]]; then
  fail_value "exactly one allocated GPU" "${runtime_gpu_count:-<unset>}"
fi
if [[ "${SLURM_CPUS_PER_TASK:-}" != "16" ]]; then
  fail_value "exactly 16 physical CPUs per task" "${SLURM_CPUS_PER_TASK:-<unset>}"
fi
if [[ "${MNIST_CONV_DEVICE}" != "cuda" && "${MNIST_CONV_DEVICE}" != "cuda:0" ]]; then
  fail_value "MNIST_CONV_DEVICE to identify the one CUDA device" "${MNIST_CONV_DEVICE}"
fi

require_env CUDA_VISIBLE_DEVICES
IFS=',' read -r -a visible_gpus <<< "${CUDA_VISIBLE_DEVICES}"
if [[ "${#visible_gpus[@]}" -ne 1 || -z "${visible_gpus[0]}" ]]; then
  fail_value "CUDA_VISIBLE_DEVICES to contain exactly one device" "${CUDA_VISIBLE_DEVICES}"
fi

if [[ -x /gpfslocalsup/bin/idrenv ]]; then
  eval "$(/gpfslocalsup/bin/idrenv -d "${execution_project}")"
elif command -v idrenv >/dev/null 2>&1; then
  eval "$(idrenv -d "${execution_project}")"
else
  fail_value "idrenv to initialize the selected execution project" "not found"
fi
if [[ -n "${MNIST_CONV_MODULE:-}" ]]; then
  module purge
  module load "${MNIST_CONV_MODULE}"
fi

for path_name in MNIST_CONV_REPO_ROOT MNIST_CONV_LR_STUDY MNIST_CONV_DATASET_ROOT; do
  if [[ ! -d "${!path_name}" ]]; then
    fail_value "${path_name} to be an existing directory" "${!path_name}"
  fi
done
for path_name in MNIST_CONV_LR_MANIFEST MNIST_CONV_LR_PACK_MANIFEST; do
  if [[ ! -f "${!path_name}" ]]; then
    fail_value "${path_name} to be an existing file" "${!path_name}"
  fi
done
if [[ -n "${MNIST_CONV_LR_BENCHMARK:-}" && ! -f "${MNIST_CONV_LR_BENCHMARK}" ]]; then
  fail_value "MNIST_CONV_LR_BENCHMARK to be an existing file when supplied" "${MNIST_CONV_LR_BENCHMARK}"
fi
if [[ ! -x "${MNIST_CONV_PYTHON}" ]] && ! command -v "${MNIST_CONV_PYTHON}" >/dev/null 2>&1; then
  fail_value "MNIST_CONV_PYTHON to name an executable interpreter" "${MNIST_CONV_PYTHON}"
fi

project_results_root="/lustre/fsn1/projects/rech/fmu/${USER:?}/server_code/results"
resolved_paths="$("${MNIST_CONV_PYTHON}" -c '
from pathlib import Path
import json
import sys

study = Path(sys.argv[1]).resolve()
manifest = Path(sys.argv[2]).resolve()
pack = Path(sys.argv[3]).resolve()
benchmark = None if not sys.argv[4] else Path(sys.argv[4]).resolve()
results = Path(sys.argv[5]).resolve()
try:
    study.relative_to(results)
except ValueError as exc:
    raise SystemExit(
        f"Expected the v6 study under the fmu R3 project results root {results}. "
        f"Provided value: {study}."
    ) from exc
stage_contract = json.loads(manifest.read_text())
study_id = stage_contract.get("study_id")
if not isinstance(study_id, str) or not study_id.startswith("lrstudy_"):
    raise SystemExit(
        f"Expected the v6 stage manifest to contain a content-addressed study_id. Provided value: {study_id!r}."
    )
expected_wrapper = results / study_id
if study.parent.name != "lr_studies" or study.parent.parent != expected_wrapper or not study.name.endswith("--" + study_id):
    raise SystemExit(
        "Expected the v6 study at <fmu-results>/<study-id>/lr_studies/<name>--<study-id>. "
        f"Provided value: {study}."
    )
for label, path in (("stage manifest", manifest), ("pack manifest", pack), ("benchmark", benchmark)):
    if path is None:
        continue
    try:
        path.relative_to(study)
    except ValueError as exc:
        raise SystemExit(
            f"Expected the {label} inside v6 study {study}. Provided value: {path}."
        ) from exc
print(study)
print(manifest)
print(pack)
print("" if benchmark is None else benchmark)
' "${MNIST_CONV_LR_STUDY}" "${MNIST_CONV_LR_MANIFEST}" "${MNIST_CONV_LR_PACK_MANIFEST}" "${MNIST_CONV_LR_BENCHMARK:-}" "${project_results_root}")"
mapfile -t canonical_paths <<< "${resolved_paths}"
MNIST_CONV_LR_STUDY="${canonical_paths[0]}"
MNIST_CONV_LR_MANIFEST="${canonical_paths[1]}"
MNIST_CONV_LR_PACK_MANIFEST="${canonical_paths[2]}"
MNIST_CONV_LR_BENCHMARK="${canonical_paths[3]:-}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  fail_value "nvidia-smi to verify the V100-32GB allocation" "not found"
fi
gpu_record="$(nvidia-smi --id="${visible_gpus[0]}" --query-gpu=name,memory.total --format=csv,noheader,nounits)"
IFS=',' read -r gpu_name gpu_memory_mib <<< "${gpu_record}"
gpu_name="${gpu_name#${gpu_name%%[![:space:]]*}}"
gpu_name="${gpu_name%${gpu_name##*[![:space:]]}}"
gpu_memory_mib="${gpu_memory_mib//[[:space:]]/}"
if [[ "${gpu_name}" != *V100* ]]; then
  fail_value "the allocated GPU model to be a V100" "${gpu_name}"
fi
if [[ ! "${gpu_memory_mib}" =~ ^[0-9]+$ ]] || (( gpu_memory_mib < 32000 || gpu_memory_mib > 33000 )); then
  fail_value "the V100 to report approximately 32768 MiB" "${gpu_memory_mib:-<unset>} MiB"
fi

cd "${MNIST_CONV_REPO_ROOT}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${MNIST_CONV_REPO_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

job_id="${SLURM_JOB_ID:-manual}"
log_dir="${MNIST_CONV_LR_STUDY}/slurm/${MNIST_CONV_LR_STAGE}/waves/${job_id}"
mkdir -p "${log_dir}"
if [[ -n "${execution_override_mode}" ]]; then
  "${MNIST_CONV_PYTHON}" -c '
from pathlib import Path
import sys
from experiments.mnist_conv.io import atomic_write_json

output = Path(sys.argv[1])
atomic_write_json(
    output,
    {
        "schema_version": "mnist-conv-lr-execution-override/v1",
        "study_id": sys.argv[2],
        "stage": sys.argv[3],
        "slurm_job_id": sys.argv[4],
        "user_authorized": True,
        "authorization_date": "2026-07-21",
        "reason": "replace pending fmu R3 job with the user-selected UMG V100 account",
        "frozen_study_execution": {
            "allocation": "AD010913993R3",
            "project": "fmu",
            "account": "fmu@v100",
        },
        "actual_execution": {
            "allocation": sys.argv[5],
            "project": sys.argv[6],
            "account": sys.argv[7],
            "partition": "gpu_p13",
            "qos": "qos_gpu-t3",
            "constraint": "v100-32g",
            "gpus": 1,
        },
    },
    canonical=True,
)
' "${log_dir}/execution_override.v1.json" "${MNIST_CONV_LR_STUDY##*--}" "${MNIST_CONV_LR_STAGE}" "${job_id}" "${execution_allocation}" "${execution_project}" "${runtime_account}"
fi
printf '[v6-pack] frozen_allocation=%s frozen_project=%s execution_override=%s actual_allocation=%s actual_project=%s account=%s stage=%s gpu=%s memory_mib=%s\n' \
  "${MNIST_CONV_R3_ALLOCATION}" "${MNIST_CONV_IDRENV_PROJECT}" "${execution_override_mode:-none}" "${execution_allocation}" "${execution_project}" "${runtime_account}" "${MNIST_CONV_LR_STAGE}" "${gpu_name}" "${gpu_memory_mib}"

pack_args=(
  --study "${MNIST_CONV_LR_STUDY}"
  --manifest "${MNIST_CONV_LR_MANIFEST}"
  --pack-manifest "${MNIST_CONV_LR_PACK_MANIFEST}"
  --stage "${MNIST_CONV_LR_STAGE}"
  --data-root "${MNIST_CONV_DATASET_ROOT}"
  --device "${MNIST_CONV_DEVICE}"
  --python "${MNIST_CONV_PYTHON}"
  --log-dir "${log_dir}"
)
if [[ -n "${MNIST_CONV_LR_BENCHMARK:-}" ]]; then
  pack_args+=(--benchmark "${MNIST_CONV_LR_BENCHMARK}")
fi
exec "${MNIST_CONV_PYTHON}" -m experiments.run_mnist_conv_lr_v6_pack "${pack_args[@]}"
