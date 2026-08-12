#!/bin/bash -l
#SBATCH --job-name=eqprop-beta-tk
#SBATCH --account=fmu@v100
#SBATCH --partition=gpu_p13
#SBATCH --qos=qos_gpu-dev
#SBATCH --constraint=v100-32g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/perfectdiode-conv123-all-amplification-adam-eqprop-bptt-beta-tk-displacement-seed0-20260810-v1/slurm/%x-%j.out
#SBATCH --error=/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/perfectdiode-conv123-all-amplification-adam-eqprop-bptt-beta-tk-displacement-seed0-20260810-v1/slurm/%x-%j.err

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

: "${EQ_ANALYSIS_ROOT:?Set the immutable analysis overlay root.}"
: "${EQ_RUNTIME_SOURCE_ROOT:?Set the clean exact runtime source root.}"
: "${EQ_SOURCE_STUDY_ROOT:?Set the immutable source-checkpoint study root.}"
: "${EQ_MECHANISM_ARTIFACT:?Set the initialization-hash provenance artifact.}"
: "${EQ_CONFIG:?Set the frozen analysis config path.}"
: "${EQ_OUTPUT_ROOT:?Set the production result root.}"
: "${EQ_DATASET_ROOT:?Set the existing ordinary-MNIST root.}"
: "${EQ_RUNNER_SHA256:?Set the frozen analysis-runner SHA-256.}"
: "${EQ_BASE_RUNNER_SHA256:?Set the immutable base-runner SHA-256.}"
: "${EQ_CONFIG_SHA256:?Set the frozen config SHA-256.}"
: "${EQ_MECHANISM_SHA256:?Set the provenance-artifact SHA-256.}"
: "${EQ_WRAPPER_SHA256:?Set the frozen Slurm-wrapper SHA-256.}"
: "${EQ_RUNTIME_HEAD:?Set the required exact runtime Git HEAD.}"

runner="${EQ_ANALYSIS_ROOT}/experiments/analyze_conv_eqprop_bptt_beta_tk_displacement.py"
base_runner="${EQ_ANALYSIS_ROOT}/experiments/analyze_conv_eqprop_bptt_checkpoint_gradients.py"
wrapper="${EQ_ANALYSIS_ROOT}/experiments/run_conv_eqprop_bptt_beta_tk_displacement_jeanzay.slurm"
for path in "${runner}" "${base_runner}" "${wrapper}" "${EQ_CONFIG}" "${EQ_MECHANISM_ARTIFACT}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Missing frozen input: ${path}" >&2
    exit 2
  fi
done
for path in "${EQ_RUNTIME_SOURCE_ROOT}" "${EQ_SOURCE_STUDY_ROOT}" "${EQ_DATASET_ROOT}"; do
  if [[ ! -d "${path}" ]]; then
    echo "Missing frozen directory: ${path}" >&2
    exit 2
  fi
done

check_sha256() {
  local path="$1"
  local expected="$2"
  local observed
  observed="$(sha256sum "${path}" | awk '{print $1}')"
  if [[ "${observed}" != "${expected}" ]]; then
    echo "SHA-256 mismatch for ${path}: ${observed} != ${expected}" >&2
    exit 2
  fi
}
check_sha256 "${runner}" "${EQ_RUNNER_SHA256}"
check_sha256 "${base_runner}" "${EQ_BASE_RUNNER_SHA256}"
check_sha256 "${EQ_CONFIG}" "${EQ_CONFIG_SHA256}"
check_sha256 "${EQ_MECHANISM_ARTIFACT}" "${EQ_MECHANISM_SHA256}"
check_sha256 "${wrapper}" "${EQ_WRAPPER_SHA256}"

export MODULESHOME="/lustre/fshomisc/sup/spack_soft/environment-modules/4.3.1/gcc-11.3.1-wf7m7j6whgecysm2fm5n73sm4jg7txup"
export MODULEPATH="/lustre/fshomisc/sup/hpe/pub/module-rh/modulefiles:/lustre/fshomisc/sup/hpe/pub/modules-idris-env4/modulefiles/linux-rhel9-skylake_avx512"
MODULECMD="${MODULESHOME}/bin/modulecmd"
eval "$("${MODULECMD}" bash purge)"
eval "$("${MODULECMD}" bash load git/2.39.1)"
eval "$("${MODULECMD}" bash load pytorch-gpu/py3/2.5.0)"

if [[ "$(git -C "${EQ_RUNTIME_SOURCE_ROOT}" rev-parse HEAD)" != "${EQ_RUNTIME_HEAD}" ]]; then
  echo "Exact runtime HEAD mismatch." >&2
  exit 2
fi
if [[ -n "$(git -C "${EQ_RUNTIME_SOURCE_ROOT}" status --porcelain)" ]]; then
  echo "Exact runtime source is dirty." >&2
  exit 2
fi

export KMP_DISABLE_SHM=1
export KMP_SHM_DISABLE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib-eqprop-beta-tk"
mkdir -p "${MPLCONFIGDIR}" "${EQ_OUTPUT_ROOT}" "${EQ_OUTPUT_ROOT}/submission"
scontrol show job -dd "${SLURM_JOB_ID}" > "${EQ_OUTPUT_ROOT}/submission/${SLURM_JOB_ID}.scontrol.txt"
{
  echo "job_id=${SLURM_JOB_ID}"
  echo "analysis_root=${EQ_ANALYSIS_ROOT}"
  echo "runtime_source_root=${EQ_RUNTIME_SOURCE_ROOT}"
  echo "source_study_root=${EQ_SOURCE_STUDY_ROOT}"
  echo "mechanism_artifact=${EQ_MECHANISM_ARTIFACT}"
  echo "config=${EQ_CONFIG}"
  echo "output_root=${EQ_OUTPUT_ROOT}"
  echo "dataset_root=${EQ_DATASET_ROOT}"
  echo "runner_sha256=${EQ_RUNNER_SHA256}"
  echo "base_runner_sha256=${EQ_BASE_RUNNER_SHA256}"
  echo "config_sha256=${EQ_CONFIG_SHA256}"
  echo "mechanism_sha256=${EQ_MECHANISM_SHA256}"
  echo "wrapper_sha256=${EQ_WRAPPER_SHA256}"
  echo "runtime_head=${EQ_RUNTIME_HEAD}"
} > "${EQ_OUTPUT_ROOT}/submission/${SLURM_JOB_ID}.contract.txt"

cd "${EQ_ANALYSIS_ROOT}"
srun python "${runner}" \
  --config "${EQ_CONFIG}" \
  --runtime-source-root "${EQ_RUNTIME_SOURCE_ROOT}" \
  --source-study-root "${EQ_SOURCE_STUDY_ROOT}" \
  --mechanism-initialization-hash-artifact "${EQ_MECHANISM_ARTIFACT}" \
  --output-root "${EQ_OUTPUT_ROOT}" \
  --dataset-root "${EQ_DATASET_ROOT}" \
  --device cuda \
  --target jean-zay \
  --run-id analysis

PYTHONPATH="${EQ_RUNTIME_SOURCE_ROOT}/labs:${EQ_RUNTIME_SOURCE_ROOT}" \
  python -m experiments.reporting validate-run "${EQ_OUTPUT_ROOT}/analysis"
