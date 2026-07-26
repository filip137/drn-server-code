#!/bin/bash -l
#SBATCH --job-name=conv2-opt-boundary-bench
#SBATCH --output=slurm/conv2-opt-boundary/%x-%j.out
#SBATCH --error=slurm/conv2-opt-boundary/%x-%j.err
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
  MNIST_CONV_BOUNDARY_REPO_ROOT \
  MNIST_CONV_BOUNDARY_PYTHON \
  MNIST_CONV_BOUNDARY_STUDY_CONFIG \
  MNIST_CONV_BOUNDARY_STUDY_DIR \
  MNIST_CONV_BOUNDARY_MANIFEST \
  MNIST_CONV_BOUNDARY_REUSE_ROOT \
  MNIST_CONV_BOUNDARY_DATA_ROOT \
  MNIST_CONV_BOUNDARY_BENCHMARK_OUTPUT \
  MNIST_CONV_BOUNDARY_AUTHORIZATION \
  SLURM_JOB_ID \
  SLURM_JOB_ACCOUNT \
  SLURM_JOB_PARTITION \
  SLURM_JOB_QOS \
  SLURM_CPUS_PER_TASK \
  SLURM_GPUS_ON_NODE \
  JOBSCRATCH
do
  require_env "${name}"
done

authorization_expected="AD010913993R3:fmu@v100:gpu_p13:qos_gpu-t3:v100-32g:gpu1:conv2-optimizer-boundary-v1"
if [[ "${MNIST_CONV_BOUNDARY_AUTHORIZATION}" != "${authorization_expected}" ]]; then
  fail_value "MNIST_CONV_BOUNDARY_AUTHORIZATION to match the frozen R3 V100 contract" "${MNIST_CONV_BOUNDARY_AUTHORIZATION}"
fi
if [[ "${SLURM_JOB_ACCOUNT}" != "fmu@v100" ]]; then
  fail_value "the live Slurm account to be fmu@v100" "${SLURM_JOB_ACCOUNT}"
fi
if [[ "${SLURM_JOB_PARTITION}" != "gpu_p13" ]]; then
  fail_value "the live Slurm partition to be gpu_p13" "${SLURM_JOB_PARTITION}"
fi
if [[ "${SLURM_JOB_QOS}" != "qos_gpu-t3" ]]; then
  fail_value "the live Slurm QOS to be qos_gpu-t3" "${SLURM_JOB_QOS}"
fi
if [[ "${SLURM_CPUS_PER_TASK}" != "16" ]]; then
  fail_value "exactly 16 physical CPUs" "${SLURM_CPUS_PER_TASK}"
fi
if [[ "${SLURM_GPUS_ON_NODE}" != "1" ]]; then
  fail_value "exactly one allocated GPU" "${SLURM_GPUS_ON_NODE}"
fi
if ! command -v scontrol >/dev/null 2>&1; then
  fail_value "scontrol to verify the live job constraint" "not found"
fi
job_record="$(scontrol show job "${SLURM_JOB_ID}" -o)"
if [[ " ${job_record} " != *" Features=v100-32g "* ]]; then
  fail_value "the live job constraint to include v100-32g" "${job_record}"
fi

if [[ -x /gpfslocalsup/bin/idrenv ]]; then
  eval "$(/gpfslocalsup/bin/idrenv -d fmu)"
elif command -v idrenv >/dev/null 2>&1; then
  eval "$(idrenv -d fmu)"
else
  fail_value "idrenv to initialize project fmu" "not found"
fi
if [[ -n "${MNIST_CONV_BOUNDARY_MODULE:-}" ]]; then
  module purge
  module load "${MNIST_CONV_BOUNDARY_MODULE}"
fi

for path_name in \
  MNIST_CONV_BOUNDARY_REPO_ROOT \
  MNIST_CONV_BOUNDARY_STUDY_DIR \
  MNIST_CONV_BOUNDARY_REUSE_ROOT \
  MNIST_CONV_BOUNDARY_DATA_ROOT \
  JOBSCRATCH
do
  if [[ ! -d "${!path_name}" ]]; then
    fail_value "${path_name} to be an existing directory" "${!path_name}"
  fi
done
for path_name in \
  MNIST_CONV_BOUNDARY_STUDY_CONFIG \
  MNIST_CONV_BOUNDARY_MANIFEST
do
  if [[ ! -f "${!path_name}" ]]; then
    fail_value "${path_name} to be an existing file" "${!path_name}"
  fi
done
if [[ -e "${MNIST_CONV_BOUNDARY_BENCHMARK_OUTPUT}" ]]; then
  fail_value "benchmark output to be absent before immutable publication" "${MNIST_CONV_BOUNDARY_BENCHMARK_OUTPUT}"
fi

project_results="/lustre/fsn1/projects/rech/fmu/${USER:?}/server_code/results"
"${MNIST_CONV_BOUNDARY_PYTHON}" -c '
from pathlib import Path
import sys

study = Path(sys.argv[1]).resolve()
results = Path(sys.argv[2]).resolve()
output = Path(sys.argv[3]).resolve()
try:
    study.relative_to(results)
    output.relative_to(study)
except ValueError as exc:
    raise SystemExit(
        f"Expected study/output under {results}. "
        f"Provided value: study={study}, output={output}."
    ) from exc
' "${MNIST_CONV_BOUNDARY_STUDY_DIR}" "${project_results}" \
  "${MNIST_CONV_BOUNDARY_BENCHMARK_OUTPUT}"

gpu_record="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits)"
if [[ "$(printf '%s\n' "${gpu_record}" | wc -l)" -ne 1 ]]; then
  fail_value "exactly one visible GPU" "${gpu_record}"
fi
if [[ "${gpu_record}" != *"V100"* || "${gpu_record}" != *"32768"* ]]; then
  fail_value "one V100 with 32768 MiB" "${gpu_record}"
fi

export KMP_DISABLE_SHM=1
export KMP_SHM_DISABLE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

benchmark_scratch="${JOBSCRATCH}/conv2-opt-boundary-benchmark-${SLURM_JOB_ID}"
cd "${MNIST_CONV_BOUNDARY_REPO_ROOT}"
"${MNIST_CONV_BOUNDARY_PYTHON}" \
  -m experiments.run_mnist_conv_lr_optimizer_boundary \
  benchmark \
  --study-config "${MNIST_CONV_BOUNDARY_STUDY_CONFIG}" \
  --study-dir "${MNIST_CONV_BOUNDARY_STUDY_DIR}" \
  --manifest "${MNIST_CONV_BOUNDARY_MANIFEST}" \
  --reuse-root "${MNIST_CONV_BOUNDARY_REUSE_ROOT}" \
  --data-root "${MNIST_CONV_BOUNDARY_DATA_ROOT}" \
  --output "${MNIST_CONV_BOUNDARY_BENCHMARK_OUTPUT}" \
  --scratch-dir "${benchmark_scratch}" \
  --python "${MNIST_CONV_BOUNDARY_PYTHON}" \
  --device cuda
