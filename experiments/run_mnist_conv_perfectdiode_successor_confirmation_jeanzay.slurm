#!/bin/bash -l
#SBATCH --job-name=pd-successor
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --time=08:00:00
#SBATCH --account=umg@v100
#SBATCH --partition=gpu_p13
#SBATCH --qos=qos_gpu-t3
#SBATCH --constraint=v100-32g
#SBATCH --mem=64000M

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

sha256_file() {
  local path="$1"
  if ! command -v sha256sum >/dev/null 2>&1; then
    fail_value "sha256sum to verify immutable launch inputs" "not found"
  fi
  sha256sum "${path}" | awk '{print $1}'
}

require_hash() {
  local label="$1"
  local path="$2"
  local expected="$3"
  local observed
  observed="$(sha256_file "${path}")"
  if [[ "${observed}" != "${expected}" ]]; then
    fail_value "${label} SHA-256 to remain ${expected}" "${observed}"
  fi
}

for name in \
  PD_SUCCESSOR_REPO_ROOT \
  PD_SUCCESSOR_PYTHON \
  PD_SUCCESSOR_BUNDLE_DIR \
  PD_SUCCESSOR_OUTPUT_ROOT \
  PD_SUCCESSOR_DATA_ROOT \
  PD_SUCCESSOR_SOURCE_ARCHIVE \
  PD_SUCCESSOR_ENVIRONMENT_CONTRACT \
  PD_SUCCESSOR_BUNDLE_SHA256 \
  PD_SUCCESSOR_SOURCE_ARCHIVE_SHA256 \
  PD_SUCCESSOR_ENVIRONMENT_CONTRACT_SHA256 \
  PD_SUCCESSOR_RUNTIME_CLI_SHA256 \
  PD_SUCCESSOR_RUNTIME_MODULE_SHA256 \
  PD_SUCCESSOR_WRAPPER_SHA256 \
  PD_SUCCESSOR_MODE \
  PD_SUCCESSOR_ARRAY_KIND \
  PD_SUCCESSOR_CANARY_PACK_INDEX \
  PD_SUCCESSOR_PRODUCTION_PACK_COUNT \
  PD_SUCCESSOR_LOGICAL_ENTRY_COUNT \
  PD_SUCCESSOR_RUNS_PER_GPU \
  PD_SUCCESSOR_CONCURRENT_WITHIN_PACK \
  PD_SUCCESSOR_ALLOCATION \
  PD_SUCCESSOR_PROJECT \
  PD_SUCCESSOR_ACCOUNT \
  PD_SUCCESSOR_PARTITION \
  PD_SUCCESSOR_QOS \
  PD_SUCCESSOR_CONSTRAINT \
  PD_SUCCESSOR_MODULE \
  PD_SUCCESSOR_AUTHORIZATION
do
  require_env "${name}"
done

authorization_expected="AD011016471R1:umg@v100:gpu_p13:qos_gpu-t3:v100-32g:gpu1:cpu16:mem64000M:nomultithread:runs2:concurrent-processes:allocation-user-approved-20260727"
if [[ "${PD_SUCCESSOR_AUTHORIZATION}" != "${authorization_expected}" ]]; then
  fail_value "PD_SUCCESSOR_AUTHORIZATION to match the user-approved allocation-only contract" "${PD_SUCCESSOR_AUTHORIZATION}"
fi
if [[ "${PD_SUCCESSOR_ALLOCATION}" != "AD011016471R1" ]]; then
  fail_value "PD_SUCCESSOR_ALLOCATION to be AD011016471R1" "${PD_SUCCESSOR_ALLOCATION}"
fi
if [[ "${PD_SUCCESSOR_PROJECT}" != "umg" ]]; then
  fail_value "PD_SUCCESSOR_PROJECT to be umg" "${PD_SUCCESSOR_PROJECT}"
fi
if [[ "${PD_SUCCESSOR_ACCOUNT}" != "umg@v100" ]]; then
  fail_value "PD_SUCCESSOR_ACCOUNT to be umg@v100" "${PD_SUCCESSOR_ACCOUNT}"
fi
if [[ "${PD_SUCCESSOR_PARTITION}" != "gpu_p13" ]]; then
  fail_value "PD_SUCCESSOR_PARTITION to be gpu_p13" "${PD_SUCCESSOR_PARTITION}"
fi
if [[ "${PD_SUCCESSOR_QOS}" != "qos_gpu-t3" ]]; then
  fail_value "PD_SUCCESSOR_QOS to be qos_gpu-t3" "${PD_SUCCESSOR_QOS}"
fi
if [[ "${PD_SUCCESSOR_CONSTRAINT}" != "v100-32g" ]]; then
  fail_value "PD_SUCCESSOR_CONSTRAINT to be v100-32g" "${PD_SUCCESSOR_CONSTRAINT}"
fi
if [[ "${PD_SUCCESSOR_MODULE}" != "pytorch-gpu/py3/2.5.0" ]]; then
  fail_value "PD_SUCCESSOR_MODULE to be pytorch-gpu/py3/2.5.0" "${PD_SUCCESSOR_MODULE}"
fi
if [[ "${PD_SUCCESSOR_PYTHON}" != /* ]]; then
  fail_value "PD_SUCCESSOR_PYTHON to be an absolute interpreter path" "${PD_SUCCESSOR_PYTHON}"
fi
if [[ "${PD_SUCCESSOR_CANARY_PACK_INDEX}" != "4" ]]; then
  fail_value "PD_SUCCESSOR_CANARY_PACK_INDEX to be the frozen worst-case Conv2 pack 4" "${PD_SUCCESSOR_CANARY_PACK_INDEX}"
fi
if [[ "${PD_SUCCESSOR_PRODUCTION_PACK_COUNT}:${PD_SUCCESSOR_LOGICAL_ENTRY_COUNT}:${PD_SUCCESSOR_RUNS_PER_GPU}:${PD_SUCCESSOR_CONCURRENT_WITHIN_PACK}" != "6:12:2:true" ]]; then
  fail_value \
    "the exact six-pack, 12-entry, two-concurrent-processes-per-GPU contract" \
    "${PD_SUCCESSOR_PRODUCTION_PACK_COUNT}:${PD_SUCCESSOR_LOGICAL_ENTRY_COUNT}:${PD_SUCCESSOR_RUNS_PER_GPU}:${PD_SUCCESSOR_CONCURRENT_WITHIN_PACK}"
fi

case "${PD_SUCCESSOR_ARRAY_KIND}:${PD_SUCCESSOR_MODE}" in
  canary:smoke-pack|production:run-pack) ;;
  *)
    fail_value \
      "PD_SUCCESSOR_ARRAY_KIND:PD_SUCCESSOR_MODE to be canary:smoke-pack or production:run-pack" \
      "${PD_SUCCESSOR_ARRAY_KIND}:${PD_SUCCESSOR_MODE}"
    ;;
esac

runtime_cli="${PD_SUCCESSOR_REPO_ROOT}/experiments/run_mnist_conv_perfectdiode_successor_confirmation.py"
runtime_module="${PD_SUCCESSOR_REPO_ROOT}/experiments/mnist_conv/perfectdiode_successor_confirmation.py"
wrapper_path="$(realpath "$0")"
for path_name in PD_SUCCESSOR_REPO_ROOT PD_SUCCESSOR_BUNDLE_DIR PD_SUCCESSOR_OUTPUT_ROOT PD_SUCCESSOR_DATA_ROOT; do
  if [[ ! -d "${!path_name}" ]]; then
    fail_value "${path_name} to be an existing directory" "${!path_name}"
  fi
done
for path_name in PD_SUCCESSOR_SOURCE_ARCHIVE PD_SUCCESSOR_ENVIRONMENT_CONTRACT; do
  if [[ ! -f "${!path_name}" ]]; then
    fail_value "${path_name} to be an existing file" "${!path_name}"
  fi
done
if [[ ! -f "${runtime_cli}" ]]; then
  fail_value "the successor runtime CLI to exist" "${runtime_cli}"
fi
if [[ ! -f "${runtime_module}" ]]; then
  fail_value "the successor runtime module to exist" "${runtime_module}"
fi
if [[ ! -f "${wrapper_path}" ]]; then
  fail_value "the resolved successor Slurm wrapper to exist" "${wrapper_path}"
fi
require_hash "source archive" "${PD_SUCCESSOR_SOURCE_ARCHIVE}" "${PD_SUCCESSOR_SOURCE_ARCHIVE_SHA256}"
require_hash "environment contract" "${PD_SUCCESSOR_ENVIRONMENT_CONTRACT}" "${PD_SUCCESSOR_ENVIRONMENT_CONTRACT_SHA256}"
require_hash "successor runtime CLI" "${runtime_cli}" "${PD_SUCCESSOR_RUNTIME_CLI_SHA256}"
require_hash "successor runtime module" "${runtime_module}" "${PD_SUCCESSOR_RUNTIME_MODULE_SHA256}"
require_hash "successor Slurm wrapper" "${wrapper_path}" "${PD_SUCCESSOR_WRAPPER_SHA256}"

if [[ -x /gpfslocalsup/bin/idrenv ]]; then
  eval "$(/gpfslocalsup/bin/idrenv -d "${PD_SUCCESSOR_PROJECT}")"
elif command -v idrenv >/dev/null 2>&1; then
  eval "$(idrenv -d "${PD_SUCCESSOR_PROJECT}")"
else
  fail_value "idrenv to initialize project umg" "not found"
fi
if command -v module >/dev/null 2>&1; then
  module purge
  module load "${PD_SUCCESSOR_MODULE}"
else
  module_home="/lustre/fshomisc/sup/spack_soft/environment-modules/4.3.1/gcc-11.3.1-wf7m7j6whgecysm2fm5n73sm4jg7txup"
  module_command="${module_home}/bin/modulecmd"
  if [[ ! -x "${module_command}" ]]; then
    fail_value "module or the reviewed Jean Zay modulecmd backend" "${module_command}"
  fi
  export MODULESHOME="${module_home}"
  export MODULEPATH="/lustre/fshomisc/sup/hpe/pub/module-rh/modulefiles:/lustre/fshomisc/sup/hpe/pub/modules-idris-env4/modulefiles/linux-rhel9-skylake_avx512"
  eval "$("${module_command}" bash purge)"
  eval "$("${module_command}" bash load "${PD_SUCCESSOR_MODULE}")"
fi
observed_python="$(command -v python || true)"
if [[ "${observed_python}" != "${PD_SUCCESSOR_PYTHON}" ]]; then
  fail_value "the loaded module Python to equal PD_SUCCESSOR_PYTHON" "${observed_python:-<unset>}"
fi
if [[ ! -x "${PD_SUCCESSOR_PYTHON}" ]]; then
  fail_value "PD_SUCCESSOR_PYTHON to name an executable Python interpreter" "${PD_SUCCESSOR_PYTHON}"
fi

observed_bundle_sha256="$("${PD_SUCCESSOR_PYTHON}" -c '
from pathlib import Path
import hashlib
import sys

root = Path(sys.argv[1]).resolve()
digest = hashlib.sha256()
files = []
for candidate in root.rglob("*"):
    if candidate.is_symlink():
        raise SystemExit(
            f"Expected the immutable bundle tree not to contain symlinks. "
            f"Provided value: {candidate}."
        )
    if candidate.is_file():
        files.append(candidate)
    elif not candidate.is_dir():
        raise SystemExit(
            f"Expected the immutable bundle tree to contain only files/directories. "
            f"Provided value: {candidate}."
        )
for candidate in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
    relative = candidate.relative_to(root).as_posix().encode("utf-8")
    file_digest = hashlib.sha256(candidate.read_bytes()).digest()
    digest.update(len(relative).to_bytes(8, "big"))
    digest.update(relative)
    digest.update(candidate.stat().st_size.to_bytes(8, "big"))
    digest.update(file_digest)
print(digest.hexdigest())
' "${PD_SUCCESSOR_BUNDLE_DIR}")"
if [[ "${observed_bundle_sha256}" != "${PD_SUCCESSOR_BUNDLE_SHA256}" ]]; then
  fail_value "the immutable bundle SHA-256 to remain ${PD_SUCCESSOR_BUNDLE_SHA256}" "${observed_bundle_sha256}"
fi

runtime_preflight=(
  "${PD_SUCCESSOR_PYTHON}"
  "${runtime_cli}"
  preflight
  --bundle-dir "${PD_SUCCESSOR_BUNDLE_DIR}"
  --source-archive "${PD_SUCCESSOR_SOURCE_ARCHIVE}"
  --environment-contract "${PD_SUCCESSOR_ENVIRONMENT_CONTRACT}"
)

if (( $# > 1 )); then
  fail_value "zero arguments for a Slurm worker or exactly --preflight" "$*"
fi
if (( $# == 1 )); then
  if [[ "$1" != "--preflight" ]]; then
    fail_value "the only wrapper argument to be --preflight" "$1"
  fi
  # The reviewed idrenv/module environment above is intentionally identical
  # to the worker environment. This branch performs no scheduler/device query
  # and no write; the public runtime preflight is likewise read-only.
  exec "${runtime_preflight[@]}"
fi

for name in \
  PD_SUCCESSOR_LAUNCH_AUTHORIZATION_RECEIPT \
  PD_SUCCESSOR_LAUNCH_AUTHORIZATION_RECEIPT_SHA256 \
  PD_SUCCESSOR_SCHEDULED_PREFLIGHT_RECEIPT \
  PD_SUCCESSOR_SCHEDULED_PREFLIGHT_RECEIPT_SHA256 \
  PD_SUCCESSOR_PREFLIGHT_RECEIPT \
  PD_SUCCESSOR_PREFLIGHT_RECEIPT_SHA256
do
  require_env "${name}"
done
if [[ ! -f "${PD_SUCCESSOR_LAUNCH_AUTHORIZATION_RECEIPT}" ]]; then
  fail_value "PD_SUCCESSOR_LAUNCH_AUTHORIZATION_RECEIPT to be an existing file" "${PD_SUCCESSOR_LAUNCH_AUTHORIZATION_RECEIPT}"
fi
if [[ ! -f "${PD_SUCCESSOR_SCHEDULED_PREFLIGHT_RECEIPT}" ]]; then
  fail_value "PD_SUCCESSOR_SCHEDULED_PREFLIGHT_RECEIPT to be an existing file" "${PD_SUCCESSOR_SCHEDULED_PREFLIGHT_RECEIPT}"
fi
if [[ ! -f "${PD_SUCCESSOR_PREFLIGHT_RECEIPT}" ]]; then
  fail_value "PD_SUCCESSOR_PREFLIGHT_RECEIPT to be an existing file" "${PD_SUCCESSOR_PREFLIGHT_RECEIPT}"
fi
require_hash \
  "launch-authorization receipt" \
  "${PD_SUCCESSOR_LAUNCH_AUTHORIZATION_RECEIPT}" \
  "${PD_SUCCESSOR_LAUNCH_AUTHORIZATION_RECEIPT_SHA256}"
require_hash \
  "scheduled-run preflight receipt" \
  "${PD_SUCCESSOR_SCHEDULED_PREFLIGHT_RECEIPT}" \
  "${PD_SUCCESSOR_SCHEDULED_PREFLIGHT_RECEIPT_SHA256}"
require_hash \
  "successor preflight receipt" \
  "${PD_SUCCESSOR_PREFLIGHT_RECEIPT}" \
  "${PD_SUCCESSOR_PREFLIGHT_RECEIPT_SHA256}"
if [[ "${PD_SUCCESSOR_ARRAY_KIND}" == "production" ]]; then
  require_env "PD_SUCCESSOR_CANARY_RECEIPT"
  require_env "PD_SUCCESSOR_CANARY_RECEIPT_SHA256"
  if [[ ! -f "${PD_SUCCESSOR_CANARY_RECEIPT}" ]]; then
    fail_value "PD_SUCCESSOR_CANARY_RECEIPT to be an existing file" "${PD_SUCCESSOR_CANARY_RECEIPT}"
  fi
  require_hash \
    "successor canary-gate receipt" \
    "${PD_SUCCESSOR_CANARY_RECEIPT}" \
    "${PD_SUCCESSOR_CANARY_RECEIPT_SHA256}"
fi

for name in \
  SLURM_JOB_ID \
  SLURM_ARRAY_JOB_ID \
  SLURM_ARRAY_TASK_ID \
  SLURM_ARRAY_TASK_COUNT \
  SLURM_ARRAY_TASK_MIN \
  SLURM_ARRAY_TASK_MAX \
  SLURM_ARRAY_TASK_STEP \
  SLURM_CPUS_PER_TASK \
  CUDA_VISIBLE_DEVICES
do
  require_env "${name}"
done

for name in SLURM_JOB_ID SLURM_ARRAY_JOB_ID SLURM_ARRAY_TASK_ID SLURM_ARRAY_TASK_COUNT SLURM_ARRAY_TASK_MIN SLURM_ARRAY_TASK_MAX SLURM_ARRAY_TASK_STEP; do
  if [[ ! "${!name}" =~ ^[0-9]+$ ]]; then
    fail_value "${name} to contain only decimal digits" "${!name}"
  fi
done
if [[ "${SLURM_CPUS_PER_TASK}" != "16" ]]; then
  fail_value "exactly 16 physical CPUs per task" "${SLURM_CPUS_PER_TASK}"
fi
if [[ "${SLURM_ARRAY_TASK_STEP}" != "1" ]]; then
  fail_value "the Slurm array task step to be one" "${SLURM_ARRAY_TASK_STEP}"
fi

case "${PD_SUCCESSOR_ARRAY_KIND}" in
  canary)
    if [[ "${SLURM_ARRAY_TASK_ID}:${SLURM_ARRAY_TASK_COUNT}:${SLURM_ARRAY_TASK_MIN}:${SLURM_ARRAY_TASK_MAX}" != "0:1:0:0" ]]; then
      fail_value \
        "the canary allocation to be exact array 0-0" \
        "${SLURM_ARRAY_TASK_ID}:${SLURM_ARRAY_TASK_COUNT}:${SLURM_ARRAY_TASK_MIN}:${SLURM_ARRAY_TASK_MAX}"
    fi
    pack_index="${PD_SUCCESSOR_CANARY_PACK_INDEX}"
    ;;
  production)
    if [[ "${SLURM_ARRAY_TASK_COUNT}:${SLURM_ARRAY_TASK_MIN}:${SLURM_ARRAY_TASK_MAX}" != "6:0:5" ]]; then
      fail_value \
        "the production allocation to be exact six-pack array 0-5" \
        "${SLURM_ARRAY_TASK_COUNT}:${SLURM_ARRAY_TASK_MIN}:${SLURM_ARRAY_TASK_MAX}"
    fi
    if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID > 5 )); then
      fail_value "the production pack index to be from 0 through 5" "${SLURM_ARRAY_TASK_ID}"
    fi
    pack_index="${SLURM_ARRAY_TASK_ID}"
    ;;
esac

if ! command -v scontrol >/dev/null 2>&1; then
  fail_value "scontrol to verify the live Slurm resource contract" "not found"
fi
runtime_job_record="$(scontrol show job "${SLURM_JOB_ID}" -o)"
job_field() {
  local field="$1"
  awk -v prefix="${field}=" '
    {
      for (field_index = 1; field_index <= NF; field_index++) {
        if (index($field_index, prefix) == 1) {
          sub(prefix, "", $field_index)
          print $field_index
          exit
        }
      }
    }
  ' <<< "${runtime_job_record}"
}

live_account="$(job_field Account)"
live_partition="$(job_field Partition)"
live_qos="$(job_field QOS)"
live_features="$(job_field Features)"
live_threads_per_core="$(job_field ThreadsPerCore)"
live_thread_spec="$(job_field 'ReqB:S:C:T')"
live_nodes="$(job_field NumNodes)"
live_tasks="$(job_field NumTasks)"
live_cpus="$(job_field NumCPUs)"
live_cpus_per_task="$(job_field CPUs/Task)"
live_tres_per_node="$(job_field TresPerNode)"
live_req_tres="$(job_field ReqTRES)"
live_memory="$(job_field MinMemoryNode)"
live_array_job_id="$(job_field ArrayJobId)"
live_array_task_id="$(job_field ArrayTaskId)"

if [[ "${live_account}" != "${PD_SUCCESSOR_ACCOUNT}" ]]; then
  fail_value "the live Slurm account to be umg@v100" "${live_account:-<unset>}"
fi
if [[ "${live_partition}" != "${PD_SUCCESSOR_PARTITION}" ]]; then
  fail_value "the live Slurm partition to be gpu_p13" "${live_partition:-<unset>}"
fi
if [[ "${live_qos}" != "${PD_SUCCESSOR_QOS}" ]]; then
  fail_value "the live Slurm QOS to be qos_gpu-t3" "${live_qos:-<unset>}"
fi
if [[ "${live_features}" != *"v100-32g"* ]]; then
  fail_value "the live Slurm job constraint to include v100-32g" "${live_features:-<unset>}"
fi
if [[ -n "${live_threads_per_core}" ]]; then
  live_thread_count="${live_threads_per_core}"
elif [[ "${live_thread_spec}" =~ :([0-9]+)$ ]]; then
  live_thread_count="${BASH_REMATCH[1]}"
else
  live_thread_count=""
fi
if [[ -z "${live_cpus_per_task}" && "${live_req_tres}" =~ (^|,)cpu=([0-9]+)($|,) ]]; then
  live_cpus_per_task="${BASH_REMATCH[2]}"
fi
if [[ "${live_thread_count}:${live_nodes}:${live_tasks}:${live_cpus}:${live_cpus_per_task}" != "1:1:1:16:16" ]]; then
  fail_value \
    "the live job to use one task on one node, 16 physical CPUs, 16 CPUs/task, and one thread/core" \
    "${live_thread_count}:${live_nodes}:${live_tasks}:${live_cpus}:${live_cpus_per_task}"
fi
if [[ ",${live_tres_per_node}," != *",gres/gpu:1,"* && ",${live_req_tres}," != *",gres/gpu=1,"* ]]; then
  fail_value "the live Slurm job to request exactly one GPU per node" "${live_tres_per_node:-<unset>}"
fi
if [[ -z "${live_memory}" && "${live_req_tres}" =~ (^|,)mem=([^,]+)($|,) ]]; then
  live_memory="${BASH_REMATCH[2]}"
fi
case "${live_memory}" in
  64000M) live_memory_mb="64000" ;;
  *G)
    live_memory_mb="$(awk -v value="${live_memory%G}" 'BEGIN { printf "%.0f", value * 1024 }')"
    ;;
  *) live_memory_mb="" ;;
esac
if [[ "${live_memory_mb}" != "64000" ]]; then
  fail_value "the live Slurm job memory to be exactly 64000 MiB" "${live_memory:-<unset>}"
fi
if [[ "${live_array_job_id}:${live_array_task_id}" != "${SLURM_ARRAY_JOB_ID}:${SLURM_ARRAY_TASK_ID}" ]]; then
  fail_value \
    "scontrol array job/task identity to match the live task environment" \
    "${live_array_job_id}:${live_array_task_id}"
fi

for optional_name in SLURM_JOB_ACCOUNT SLURM_JOB_PARTITION SLURM_JOB_QOS; do
  if [[ -n "${!optional_name:-}" ]]; then
    case "${optional_name}" in
      SLURM_JOB_ACCOUNT) optional_expected="${live_account}" ;;
      SLURM_JOB_PARTITION) optional_expected="${live_partition}" ;;
      SLURM_JOB_QOS) optional_expected="${live_qos}" ;;
    esac
    if [[ "${!optional_name}" != "${optional_expected}" ]]; then
      fail_value "${optional_name} to agree with scontrol" "${!optional_name}"
    fi
  fi
done
if [[ -n "${SLURM_GPUS_ON_NODE:-}" && "${SLURM_GPUS_ON_NODE}" != "1" ]]; then
  fail_value "SLURM_GPUS_ON_NODE, when set, to be one" "${SLURM_GPUS_ON_NODE}"
fi
if [[ -n "${SLURM_MEM_PER_NODE:-}" && "${SLURM_MEM_PER_NODE}" != "64000" ]]; then
  fail_value "SLURM_MEM_PER_NODE, when set, to be 64000" "${SLURM_MEM_PER_NODE}"
fi
export SLURM_JOB_ACCOUNT="${live_account}"
export SLURM_JOB_PARTITION="${live_partition}"
export SLURM_JOB_QOS="${live_qos}"
export SLURM_GPUS_ON_NODE="1"
export SLURM_MEM_PER_NODE="64000"

IFS=',' read -r -a visible_gpus <<< "${CUDA_VISIBLE_DEVICES}"
if [[ "${#visible_gpus[@]}" -ne 1 || -z "${visible_gpus[0]}" ]]; then
  fail_value "CUDA_VISIBLE_DEVICES to contain exactly one device" "${CUDA_VISIBLE_DEVICES}"
fi

expected_source_prefix="/lustre/fswork/projects/rech/umg/${USER:?}/"
expected_result_prefix="/lustre/fsn1/projects/rech/umg/${USER}/server_code/results/"
expected_data_root="/lustre/fsn1/projects/rech/umg/${USER}/datasets/mnist"
resolved_repo="$(realpath "${PD_SUCCESSOR_REPO_ROOT}")"
resolved_source_archive="$(realpath "${PD_SUCCESSOR_SOURCE_ARCHIVE}")"
resolved_environment_contract="$(realpath "${PD_SUCCESSOR_ENVIRONMENT_CONTRACT}")"
resolved_bundle="$(realpath "${PD_SUCCESSOR_BUNDLE_DIR}")"
resolved_output="$(realpath "${PD_SUCCESSOR_OUTPUT_ROOT}")"
resolved_data="$(realpath "${PD_SUCCESSOR_DATA_ROOT}")"
for source_path in "${resolved_repo}" "${resolved_source_archive}" "${resolved_environment_contract}"; do
  if [[ "${source_path}/" != "${expected_source_prefix}"* ]]; then
    fail_value "source inputs to be under ${expected_source_prefix}" "${source_path}"
  fi
done
for result_path in "${resolved_bundle}" "${resolved_output}"; do
  if [[ "${result_path}/" != "${expected_result_prefix}"* ]]; then
    fail_value "bundle and output roots to be under ${expected_result_prefix}" "${result_path}"
  fi
done
if [[ "${resolved_data}" != "${expected_data_root}" ]]; then
  fail_value "the MNIST data root to be ${expected_data_root}" "${resolved_data}"
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  fail_value "nvidia-smi to verify the V100-32GB allocation" "not found"
fi
gpu_record="$(nvidia-smi --id="${visible_gpus[0]}" --query-gpu=name,memory.total --format=csv,noheader,nounits)"
if [[ "${gpu_record}" == *$'\n'* ]]; then
  fail_value "the allocation to expose exactly one GPU record" "${gpu_record}"
fi
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

cd "${PD_SUCCESSOR_REPO_ROOT}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PD_SUCCESSOR_REPO_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

runtime_args=(
  "${PD_SUCCESSOR_PYTHON}"
  "${runtime_cli}"
  "${PD_SUCCESSOR_MODE}"
  --bundle-dir "${PD_SUCCESSOR_BUNDLE_DIR}"
  --pack-index "${pack_index}"
  --output-root "${PD_SUCCESSOR_OUTPUT_ROOT}"
  --data-root "${PD_SUCCESSOR_DATA_ROOT}"
  --source-archive "${PD_SUCCESSOR_SOURCE_ARCHIVE}"
  --environment-contract "${PD_SUCCESSOR_ENVIRONMENT_CONTRACT}"
  --device cuda
)
exec "${runtime_args[@]}"
