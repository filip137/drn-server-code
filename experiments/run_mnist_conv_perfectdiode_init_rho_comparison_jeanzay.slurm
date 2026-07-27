#!/bin/bash -l
#SBATCH --job-name=pd-init-rho
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --time=03:00:00
#SBATCH --account=fmu@v100
#SBATCH --partition=gpu_p13
#SBATCH --qos=qos_gpu-t3
#SBATCH --constraint=v100-16g

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
  sha256sum "$1" | awk '{print $1}'
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
  PD_INIT_RHO_SOURCE_ROOT \
  PD_INIT_RHO_PYTHON \
  PD_INIT_RHO_ROOT \
  PD_INIT_RHO_DATA_ROOT \
  PD_INIT_RHO_SOURCE_ARCHIVE \
  PD_INIT_RHO_ENVIRONMENT_CONTRACT \
  PD_INIT_RHO_SOURCE_ARCHIVE_SHA256 \
  PD_INIT_RHO_ENVIRONMENT_CONTRACT_SHA256 \
  PD_INIT_RHO_MANIFEST_SHA256 \
  PD_INIT_RHO_RUNTIME_CLI_SHA256 \
  PD_INIT_RHO_RUNTIME_MODULE_SHA256 \
  PD_INIT_RHO_WRAPPER_SHA256 \
  PD_INIT_RHO_STAGE \
  PD_INIT_RHO_ARRAY_KIND \
  PD_INIT_RHO_CANARY_TASK_INDEX \
  PD_INIT_RHO_AUTHORIZATION
do
  require_env "${name}"
done

authorization_expected="AD010913993R3:fmu@v100:gpu_p13:qos_gpu-t3:v100-16g:gpu1:cpu16:nomultithread:init-rho-approved-20260727"
if [[ "${PD_INIT_RHO_AUTHORIZATION}" != "${authorization_expected}" ]]; then
  fail_value \
    "PD_INIT_RHO_AUTHORIZATION to match the approved allocation contract" \
    "${PD_INIT_RHO_AUTHORIZATION}"
fi
if [[ "${PD_INIT_RHO_STAGE}" != "gate" && "${PD_INIT_RHO_STAGE}" != "surface" ]]; then
  fail_value "PD_INIT_RHO_STAGE to be gate or surface" "${PD_INIT_RHO_STAGE}"
fi
if [[ "${PD_INIT_RHO_ARRAY_KIND}" != "canary" && "${PD_INIT_RHO_ARRAY_KIND}" != "production" ]]; then
  fail_value \
    "PD_INIT_RHO_ARRAY_KIND to be canary or production" \
    "${PD_INIT_RHO_ARRAY_KIND}"
fi
if [[ ! "${PD_INIT_RHO_CANARY_TASK_INDEX}" =~ ^([0-9]|1[01])$ ]]; then
  fail_value \
    "PD_INIT_RHO_CANARY_TASK_INDEX to be an integer from 0 through 11" \
    "${PD_INIT_RHO_CANARY_TASK_INDEX}"
fi

runtime_cli="${PD_INIT_RHO_SOURCE_ROOT}/experiments/run_mnist_conv_perfectdiode_init_rho_comparison.py"
runtime_module="${PD_INIT_RHO_SOURCE_ROOT}/experiments/mnist_conv/perfectdiode_init_rho_comparison.py"
wrapper_path="$(realpath "$0")"
manifest_path="${PD_INIT_RHO_ROOT}/manifest.json"

for path_name in PD_INIT_RHO_SOURCE_ROOT PD_INIT_RHO_ROOT PD_INIT_RHO_DATA_ROOT; do
  if [[ ! -d "${!path_name}" ]]; then
    fail_value "${path_name} to be an existing directory" "${!path_name}"
  fi
done
for path_name in PD_INIT_RHO_SOURCE_ARCHIVE PD_INIT_RHO_ENVIRONMENT_CONTRACT; do
  if [[ ! -f "${!path_name}" ]]; then
    fail_value "${path_name} to be an existing file" "${!path_name}"
  fi
done
for required_path in "${runtime_cli}" "${runtime_module}" "${wrapper_path}" "${manifest_path}"; do
  if [[ ! -f "${required_path}" ]]; then
    fail_value "a required immutable launch file to exist" "${required_path}"
  fi
done

require_hash \
  "source archive" \
  "${PD_INIT_RHO_SOURCE_ARCHIVE}" \
  "${PD_INIT_RHO_SOURCE_ARCHIVE_SHA256}"
require_hash \
  "environment contract" \
  "${PD_INIT_RHO_ENVIRONMENT_CONTRACT}" \
  "${PD_INIT_RHO_ENVIRONMENT_CONTRACT_SHA256}"
require_hash \
  "manifest" \
  "${manifest_path}" \
  "${PD_INIT_RHO_MANIFEST_SHA256}"
require_hash \
  "runtime CLI" \
  "${runtime_cli}" \
  "${PD_INIT_RHO_RUNTIME_CLI_SHA256}"
require_hash \
  "runtime module" \
  "${runtime_module}" \
  "${PD_INIT_RHO_RUNTIME_MODULE_SHA256}"
require_hash \
  "Slurm wrapper" \
  "${wrapper_path}" \
  "${PD_INIT_RHO_WRAPPER_SHA256}"

if [[ -x /gpfslocalsup/bin/idrenv ]]; then
  eval "$(/gpfslocalsup/bin/idrenv -d fmu)"
elif command -v idrenv >/dev/null 2>&1; then
  eval "$(idrenv -d fmu)"
else
  fail_value "idrenv to initialize project fmu" "not found"
fi
if ! command -v module >/dev/null 2>&1; then
  fail_value "the Jean Zay module command" "not found"
fi
module purge
module load pytorch-gpu/py3/2.5.0
observed_python="$(command -v python || true)"
if [[ "${observed_python}" != "${PD_INIT_RHO_PYTHON}" ]]; then
  fail_value \
    "the loaded module Python to equal PD_INIT_RHO_PYTHON" \
    "${observed_python:-<unset>}"
fi

runtime=(
  "${PD_INIT_RHO_PYTHON}"
  "${runtime_cli}"
)
runtime_preflight=(
  "${runtime[@]}"
  preflight
  --root "${PD_INIT_RHO_ROOT}"
  --source-archive "${PD_INIT_RHO_SOURCE_ARCHIVE}"
  --environment-contract "${PD_INIT_RHO_ENVIRONMENT_CONTRACT}"
)

if (( $# > 1 )); then
  fail_value "zero arguments for a worker or exactly --preflight" "$*"
fi
if (( $# == 1 )); then
  if [[ "$1" != "--preflight" ]]; then
    fail_value "the only wrapper argument to be --preflight" "$1"
  fi
  exec "${runtime_preflight[@]}"
fi

for name in \
  PD_INIT_RHO_SCHEDULED_PREFLIGHT_RECEIPT \
  PD_INIT_RHO_SCHEDULED_PREFLIGHT_RECEIPT_SHA256 \
  PD_INIT_RHO_EXPERIMENT_PREFLIGHT_RECEIPT \
  PD_INIT_RHO_EXPERIMENT_PREFLIGHT_RECEIPT_SHA256
do
  require_env "${name}"
done
for receipt_name in \
  PD_INIT_RHO_SCHEDULED_PREFLIGHT_RECEIPT \
  PD_INIT_RHO_EXPERIMENT_PREFLIGHT_RECEIPT
do
  if [[ ! -f "${!receipt_name}" ]]; then
    fail_value "${receipt_name} to be an existing file" "${!receipt_name}"
  fi
done
require_hash \
  "scheduled-run preflight receipt" \
  "${PD_INIT_RHO_SCHEDULED_PREFLIGHT_RECEIPT}" \
  "${PD_INIT_RHO_SCHEDULED_PREFLIGHT_RECEIPT_SHA256}"
require_hash \
  "experiment preflight receipt" \
  "${PD_INIT_RHO_EXPERIMENT_PREFLIGHT_RECEIPT}" \
  "${PD_INIT_RHO_EXPERIMENT_PREFLIGHT_RECEIPT_SHA256}"

if [[ "${PD_INIT_RHO_ARRAY_KIND}" == "production" ]]; then
  require_env "PD_INIT_RHO_CANARY_RECEIPT"
  require_env "PD_INIT_RHO_CANARY_RECEIPT_SHA256"
  if [[ ! -f "${PD_INIT_RHO_CANARY_RECEIPT}" ]]; then
    fail_value \
      "PD_INIT_RHO_CANARY_RECEIPT to be an existing file" \
      "${PD_INIT_RHO_CANARY_RECEIPT}"
  fi
  require_hash \
    "live-canary receipt" \
    "${PD_INIT_RHO_CANARY_RECEIPT}" \
    "${PD_INIT_RHO_CANARY_RECEIPT_SHA256}"
fi
if [[ "${PD_INIT_RHO_STAGE}" == "surface" ]]; then
  require_env "PD_INIT_RHO_TK_GATE_RECEIPT"
  require_env "PD_INIT_RHO_TK_GATE_RECEIPT_SHA256"
  if [[ ! -f "${PD_INIT_RHO_TK_GATE_RECEIPT}" ]]; then
    fail_value \
      "PD_INIT_RHO_TK_GATE_RECEIPT to be an existing file" \
      "${PD_INIT_RHO_TK_GATE_RECEIPT}"
  fi
  require_hash \
    "fixed-T/K gate receipt" \
    "${PD_INIT_RHO_TK_GATE_RECEIPT}" \
    "${PD_INIT_RHO_TK_GATE_RECEIPT_SHA256}"
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
for name in \
  SLURM_JOB_ID \
  SLURM_ARRAY_JOB_ID \
  SLURM_ARRAY_TASK_ID \
  SLURM_ARRAY_TASK_COUNT \
  SLURM_ARRAY_TASK_MIN \
  SLURM_ARRAY_TASK_MAX \
  SLURM_ARRAY_TASK_STEP
do
  if [[ ! "${!name}" =~ ^[0-9]+$ ]]; then
    fail_value "${name} to contain only decimal digits" "${!name}"
  fi
done
if [[ "${SLURM_CPUS_PER_TASK}" != "16" ]]; then
  fail_value "exactly 16 physical CPUs per task" "${SLURM_CPUS_PER_TASK}"
fi
if [[ "${SLURM_ARRAY_TASK_STEP}" != "1" ]]; then
  fail_value "the Slurm array step to be one" "${SLURM_ARRAY_TASK_STEP}"
fi

if [[ "${PD_INIT_RHO_ARRAY_KIND}" == "canary" ]]; then
  if [[ "${SLURM_ARRAY_TASK_ID}:${SLURM_ARRAY_TASK_COUNT}:${SLURM_ARRAY_TASK_MIN}:${SLURM_ARRAY_TASK_MAX}" != "0:1:0:0" ]]; then
    fail_value \
      "the live canary allocation to be exact array 0-0" \
      "${SLURM_ARRAY_TASK_ID}:${SLURM_ARRAY_TASK_COUNT}:${SLURM_ARRAY_TASK_MIN}:${SLURM_ARRAY_TASK_MAX}"
  fi
  task_index="${PD_INIT_RHO_CANARY_TASK_INDEX}"
else
  if [[ "${PD_INIT_RHO_STAGE}" == "gate" ]]; then
    production_contract="12:0:11"
  else
    production_contract="22:0:21"
  fi
  if [[ "${SLURM_ARRAY_TASK_COUNT}:${SLURM_ARRAY_TASK_MIN}:${SLURM_ARRAY_TASK_MAX}" != "${production_contract}" ]]; then
    fail_value \
      "the production allocation to match ${production_contract}" \
      "${SLURM_ARRAY_TASK_COUNT}:${SLURM_ARRAY_TASK_MIN}:${SLURM_ARRAY_TASK_MAX}"
  fi
  task_index="${SLURM_ARRAY_TASK_ID}"
fi

if ! command -v scontrol >/dev/null 2>&1; then
  fail_value "scontrol to verify the live allocation" "not found"
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
live_threads="$(job_field ThreadsPerCore)"
live_thread_spec="$(job_field 'ReqB:S:C:T')"
live_nodes="$(job_field NumNodes)"
live_tasks="$(job_field NumTasks)"
live_cpus="$(job_field NumCPUs)"
live_cpus_per_task="$(job_field CPUs/Task)"
live_tres_per_node="$(job_field TresPerNode)"
live_req_tres="$(job_field ReqTRES)"
live_array_job_id="$(job_field ArrayJobId)"
live_array_task_id="$(job_field ArrayTaskId)"

if [[ "${live_account}:${live_partition}:${live_qos}" != "fmu@v100:gpu_p13:qos_gpu-t3" ]]; then
  fail_value \
    "the live account/partition/QOS to match the frozen contract" \
    "${live_account}:${live_partition}:${live_qos}"
fi
if [[ "${live_features}" != *"v100-16g"* ]]; then
  fail_value "the live feature set to include v100-16g" "${live_features}"
fi
if [[ -z "${live_threads}" && "${live_thread_spec}" =~ :([0-9]+)$ ]]; then
  live_threads="${BASH_REMATCH[1]}"
fi
if [[ -z "${live_cpus_per_task}" && "${live_req_tres}" =~ (^|,)cpu=([0-9]+)($|,) ]]; then
  live_cpus_per_task="${BASH_REMATCH[2]}"
fi
if [[ "${live_threads}:${live_nodes}:${live_tasks}:${live_cpus}:${live_cpus_per_task}" != "1:1:1:16:16" ]]; then
  fail_value \
    "one task, one node, 16 physical CPUs, and one thread per core" \
    "${live_threads}:${live_nodes}:${live_tasks}:${live_cpus}:${live_cpus_per_task}"
fi
if [[ ",${live_tres_per_node}," != *",gres/gpu:1,"* && ",${live_req_tres}," != *",gres/gpu=1,"* ]]; then
  fail_value \
    "the live request to contain exactly one GPU" \
    "${live_tres_per_node:-<unset>}/${live_req_tres:-<unset>}"
fi
if [[ -n "${live_array_job_id}" || -n "${live_array_task_id}" ]]; then
  if [[ "${live_array_job_id}:${live_array_task_id}" != "${SLURM_ARRAY_JOB_ID}:${SLURM_ARRAY_TASK_ID}" ]]; then
    fail_value \
      "scontrol array identity to match the live task environment" \
      "${live_array_job_id}:${live_array_task_id}"
  fi
elif [[ "${PD_INIT_RHO_ARRAY_KIND}" != "canary" || "${SLURM_JOB_ID}" != "${SLURM_ARRAY_JOB_ID}" ]]; then
  fail_value \
    "singleton 0-0 parent identity only for a canary" \
    "${SLURM_JOB_ID}:${SLURM_ARRAY_JOB_ID}"
fi

IFS=',' read -r -a visible_gpus <<< "${CUDA_VISIBLE_DEVICES}"
if [[ "${#visible_gpus[@]}" -ne 1 || -z "${visible_gpus[0]}" ]]; then
  fail_value \
    "CUDA_VISIBLE_DEVICES to contain exactly one device" \
    "${CUDA_VISIBLE_DEVICES}"
fi
gpu_record="$(nvidia-smi --id="${visible_gpus[0]}" --query-gpu=name,memory.total --format=csv,noheader,nounits)"
IFS=',' read -r gpu_name gpu_memory_mib <<< "${gpu_record}"
gpu_name="${gpu_name#${gpu_name%%[![:space:]]*}}"
gpu_memory_mib="${gpu_memory_mib//[[:space:]]/}"
if [[ "${gpu_name}" != *V100* ]]; then
  fail_value "the live GPU to be a V100" "${gpu_name}"
fi
if [[ ! "${gpu_memory_mib}" =~ ^[0-9]+$ ]] || (( gpu_memory_mib < 16000 || gpu_memory_mib > 17000 )); then
  fail_value \
    "the V100-16g allocation to report 16000-17000 MiB" \
    "${gpu_memory_mib:-<unset>}"
fi

expected_source="/lustre/fswork/projects/rech/umg/${USER:?}/server_code/init-rho-20260727-v1/source"
expected_archive="/lustre/fswork/projects/rech/umg/${USER}/server_code/init-rho-20260727-v1/source.tar"
expected_data="/lustre/fsn1/projects/rech/umg/${USER}/datasets/mnist"
expected_result="/lustre/fsn1/projects/rech/fmu/${USER}/server_code/results/perfectdiode-conv12-initialization-rho-comparison-20260727-v1"
if [[ "$(realpath "${PD_INIT_RHO_SOURCE_ROOT}")" != "${expected_source}" ]]; then
  fail_value "the exact frozen source root" "$(realpath "${PD_INIT_RHO_SOURCE_ROOT}")"
fi
if [[ "$(realpath "${PD_INIT_RHO_SOURCE_ARCHIVE}")" != "${expected_archive}" ]]; then
  fail_value "the exact frozen source archive" "$(realpath "${PD_INIT_RHO_SOURCE_ARCHIVE}")"
fi
if [[ "$(realpath "${PD_INIT_RHO_DATA_ROOT}")" != "${expected_data}" ]]; then
  fail_value "the exact ordinary-MNIST data root" "$(realpath "${PD_INIT_RHO_DATA_ROOT}")"
fi
if [[ "$(realpath "${PD_INIT_RHO_ROOT}")" != "${expected_result}" ]]; then
  fail_value "the exact run-specific result root" "$(realpath "${PD_INIT_RHO_ROOT}")"
fi

cd "${PD_INIT_RHO_SOURCE_ROOT}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PD_INIT_RHO_SOURCE_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

if [[ "${PD_INIT_RHO_STAGE}" == "gate" ]]; then
  worker_command="run-gate"
else
  worker_command="run-surface"
fi
runtime_worker=(
  "${runtime[@]}"
  "${worker_command}"
  --root "${PD_INIT_RHO_ROOT}"
  --index "${task_index}"
  --data-root "${PD_INIT_RHO_DATA_ROOT}"
  --device cuda
)
exec "${runtime_worker[@]}"
