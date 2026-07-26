#!/bin/bash -l
#SBATCH --job-name=conv2-opt-boundary
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
  MNIST_CONV_BOUNDARY_BENCHMARK \
  MNIST_CONV_BOUNDARY_REUSE_ROOT \
  MNIST_CONV_BOUNDARY_DATA_ROOT \
  MNIST_CONV_BOUNDARY_SCHEME \
  MNIST_CONV_BOUNDARY_STAGE \
  MNIST_CONV_BOUNDARY_CONCURRENCY \
  MNIST_CONV_BOUNDARY_LOG_DIR \
  MNIST_CONV_BOUNDARY_PACK_OUTPUT \
  MNIST_CONV_BOUNDARY_AUTHORIZATION \
  SLURM_JOB_ID \
  SLURM_JOB_ACCOUNT \
  SLURM_JOB_PARTITION \
  SLURM_JOB_QOS \
  SLURM_CPUS_PER_TASK \
  SLURM_GPUS_ON_NODE
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

case "${MNIST_CONV_BOUNDARY_SCHEME}" in
  baseline|ours|legacy) ;;
  *) fail_value "scheme to be baseline, ours, or legacy" "${MNIST_CONV_BOUNDARY_SCHEME}" ;;
esac
case "${MNIST_CONV_BOUNDARY_STAGE}" in
  main_grid|upper_sentinel|bias_capped_confirmation) ;;
  *) fail_value "stage to be main_grid, upper_sentinel, or bias_capped_confirmation" "${MNIST_CONV_BOUNDARY_STAGE}" ;;
esac
case "${MNIST_CONV_BOUNDARY_CONCURRENCY}" in
  1|2|4|8|12|16) ;;
  *) fail_value "concurrency to be one of 1,2,4,8,12,16" "${MNIST_CONV_BOUNDARY_CONCURRENCY}" ;;
esac

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
  MNIST_CONV_BOUNDARY_DATA_ROOT
do
  if [[ ! -d "${!path_name}" ]]; then
    fail_value "${path_name} to be an existing directory" "${!path_name}"
  fi
done
for path_name in \
  MNIST_CONV_BOUNDARY_STUDY_CONFIG \
  MNIST_CONV_BOUNDARY_MANIFEST \
  MNIST_CONV_BOUNDARY_BENCHMARK
do
  if [[ ! -f "${!path_name}" ]]; then
    fail_value "${path_name} to be an existing file" "${!path_name}"
  fi
done
if [[ ! -x "${MNIST_CONV_BOUNDARY_PYTHON}" ]] && ! command -v "${MNIST_CONV_BOUNDARY_PYTHON}" >/dev/null 2>&1; then
  fail_value "MNIST_CONV_BOUNDARY_PYTHON to name an executable interpreter" "${MNIST_CONV_BOUNDARY_PYTHON}"
fi

project_results="/lustre/fsn1/projects/rech/fmu/${USER:?}/server_code/results"
"${MNIST_CONV_BOUNDARY_PYTHON}" -c '
from pathlib import Path
import sys

study = Path(sys.argv[1]).resolve()
results = Path(sys.argv[2]).resolve()
try:
    study.relative_to(results)
except ValueError as exc:
    raise SystemExit(
        f"Expected the boundary study under {results}. Provided value: {study}."
    ) from exc
for label, raw in (
    ("benchmark", sys.argv[3]),
    ("pack output", sys.argv[4]),
    ("log directory", sys.argv[5]),
):
    path = Path(raw).resolve()
    try:
        path.relative_to(study)
    except ValueError as exc:
        raise SystemExit(
            f"Expected {label} under {study}. Provided value: {path}."
        ) from exc
' "${MNIST_CONV_BOUNDARY_STUDY_DIR}" "${project_results}" \
  "${MNIST_CONV_BOUNDARY_BENCHMARK}" \
  "${MNIST_CONV_BOUNDARY_PACK_OUTPUT}" \
  "${MNIST_CONV_BOUNDARY_LOG_DIR}"

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

cd "${MNIST_CONV_BOUNDARY_REPO_ROOT}"
"${MNIST_CONV_BOUNDARY_PYTHON}" \
  -m experiments.run_mnist_conv_lr_optimizer_boundary \
  run-pack \
  --study-config "${MNIST_CONV_BOUNDARY_STUDY_CONFIG}" \
  --study-dir "${MNIST_CONV_BOUNDARY_STUDY_DIR}" \
  --manifest "${MNIST_CONV_BOUNDARY_MANIFEST}" \
  --benchmark "${MNIST_CONV_BOUNDARY_BENCHMARK}" \
  --reuse-root "${MNIST_CONV_BOUNDARY_REUSE_ROOT}" \
  --data-root "${MNIST_CONV_BOUNDARY_DATA_ROOT}" \
  --scheme "${MNIST_CONV_BOUNDARY_SCHEME}" \
  --stage "${MNIST_CONV_BOUNDARY_STAGE}" \
  --concurrency "${MNIST_CONV_BOUNDARY_CONCURRENCY}" \
  --python "${MNIST_CONV_BOUNDARY_PYTHON}" \
  --device cuda \
  --log-dir "${MNIST_CONV_BOUNDARY_LOG_DIR}" \
  --output "${MNIST_CONV_BOUNDARY_PACK_OUTPUT}"
