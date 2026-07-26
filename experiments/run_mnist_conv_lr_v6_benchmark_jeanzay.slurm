#!/bin/bash -l
#SBATCH --job-name=conv-lr-v6-benchmark
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
  MNIST_CONV_LR_BENCHMARK \
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

if [[ "${MNIST_CONV_R3_ALLOCATION}" != "AD010913993R3" || "${MNIST_CONV_R3_ALLOCATION}" != *R3 ]]; then
  fail_value "the allocation to be AD010913993R3 and end in R3" "${MNIST_CONV_R3_ALLOCATION}"
fi
if [[ "${MNIST_CONV_IDRENV_PROJECT}" != "fmu" ]]; then
  fail_value "MNIST_CONV_IDRENV_PROJECT to be fmu" "${MNIST_CONV_IDRENV_PROJECT}"
fi
runtime_account="${SLURM_JOB_ACCOUNT}"
if [[ "${runtime_account}" != "fmu@v100" || "${runtime_account,,}" == *umg* ]]; then
  fail_value "the Slurm account to be fmu@v100 with no historical umg account" "${runtime_account:-<unset>}"
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
  eval "$(/gpfslocalsup/bin/idrenv -d fmu)"
elif command -v idrenv >/dev/null 2>&1; then
  eval "$(idrenv -d fmu)"
else
  fail_value "idrenv to initialize project fmu" "not found"
fi
if [[ -n "${MNIST_CONV_MODULE:-}" ]]; then
  module purge
  module load "${MNIST_CONV_MODULE}"
fi

if [[ ! -d "${MNIST_CONV_REPO_ROOT}" ]]; then
  fail_value "MNIST_CONV_REPO_ROOT to be an existing directory" "${MNIST_CONV_REPO_ROOT}"
fi
if [[ ! -d "${MNIST_CONV_LR_STUDY}" ]]; then
  fail_value "MNIST_CONV_LR_STUDY to be an existing directory" "${MNIST_CONV_LR_STUDY}"
fi
if [[ ! -f "${MNIST_CONV_LR_MANIFEST}" ]]; then
  fail_value "MNIST_CONV_LR_MANIFEST to be an existing file" "${MNIST_CONV_LR_MANIFEST}"
fi
if [[ ! -d "${MNIST_CONV_DATASET_ROOT}" ]]; then
  fail_value "MNIST_CONV_DATASET_ROOT to contain staged MNIST train data" "${MNIST_CONV_DATASET_ROOT}"
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
output = Path(sys.argv[3]).resolve()
results = Path(sys.argv[4]).resolve()
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
for label, path in (("baseline manifest", manifest), ("benchmark output", output)):
    try:
        path.relative_to(study)
    except ValueError as exc:
        raise SystemExit(
            f"Expected the {label} inside v6 study {study}. Provided value: {path}."
        ) from exc
print(study)
print(manifest)
print(output)
' "${MNIST_CONV_LR_STUDY}" "${MNIST_CONV_LR_MANIFEST}" "${MNIST_CONV_LR_BENCHMARK}" "${project_results_root}")"
mapfile -t canonical_paths <<< "${resolved_paths}"
MNIST_CONV_LR_STUDY="${canonical_paths[0]}"
MNIST_CONV_LR_MANIFEST="${canonical_paths[1]}"
MNIST_CONV_LR_BENCHMARK="${canonical_paths[2]}"

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
benchmark_scratch_base="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
if [[ "${benchmark_scratch_base}" != /* ]]; then
  fail_value "the job scratch base to be an absolute path" "${benchmark_scratch_base}"
fi
benchmark_scratch="${benchmark_scratch_base%/}/mnist-conv-lr-v6-benchmark-${SLURM_JOB_ID:-manual}"
mkdir -p "${benchmark_scratch}"

printf '[v6-benchmark] allocation=%s project=%s account=%s gpu=%s memory_mib=%s\n' \
  "${MNIST_CONV_R3_ALLOCATION}" "${MNIST_CONV_IDRENV_PROJECT}" "${runtime_account}" "${gpu_name}" "${gpu_memory_mib}"
printf '[v6-benchmark] concurrency=1,2,4,8,12,16 warmup=32 measured=256\n'
printf '[v6-benchmark] scratch=%s\n' "${benchmark_scratch}"

set +e
"${MNIST_CONV_PYTHON}" -m experiments.run_mnist_conv_lr_v6_concurrency_benchmark \
  --study "${MNIST_CONV_LR_STUDY}" \
  --candidate-manifest "${MNIST_CONV_LR_MANIFEST}" \
  --data-root "${MNIST_CONV_DATASET_ROOT}" \
  --output "${MNIST_CONV_LR_BENCHMARK}" \
  --scratch-root "${benchmark_scratch}" \
  --device "${MNIST_CONV_DEVICE}"
benchmark_status=$?
set -e

if (( benchmark_status != 0 )); then
  evidence_dir="${MNIST_CONV_LR_STUDY}/slurm/concurrency-benchmark-v6"
  evidence_archive="${evidence_dir}/failure-evidence-${SLURM_JOB_ID}.tar.gz"
  mkdir -p "${evidence_dir}"
  if tar -C "${benchmark_scratch}" -czf "${evidence_archive}" .; then
    sha256sum "${evidence_archive}"
    printf '[v6-benchmark] preserved failure evidence=%s\n' "${evidence_archive}" >&2
  else
    printf '[v6-benchmark] failed to preserve scratch evidence from %s\n' "${benchmark_scratch}" >&2
  fi
fi
exit "${benchmark_status}"
