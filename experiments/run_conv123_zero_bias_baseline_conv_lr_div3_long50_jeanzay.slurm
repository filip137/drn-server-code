#!/bin/bash -l
#SBATCH --job-name=pd-c123-zb-div3-e50
#SBATCH --account=fmu@v100
#SBATCH --partition=gpu_p13
#SBATCH --qos=qos_gpu-t3
#SBATCH --constraint=v100-16g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=10:00:00
#SBATCH --array=0-2%3
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err

set -euo pipefail

: "${LR_DIV3_LONG50_SOURCE_ROOT:?Set the frozen staged source directory.}"
: "${LR_DIV3_LONG50_RESULT_ROOT:?Set the study result directory.}"
: "${LR_DIV3_LONG50_CONFIG_SET_SHA256:?Set the ordered config-set SHA-256.}"
: "${EXPERIMENT_SOURCE_COMMIT:?Set the frozen source commit.}"
: "${EXPERIMENT_SOURCE_ARCHIVE_SHA256:?Set the staged source archive SHA-256.}"
: "${SLURM_ARRAY_TASK_ID:?Submit this wrapper as a three-task Slurm array.}"

STUDY_ID="perfectdiode-conv123-zero-bias-baseline-conv-lr-div3-long50-ordinary-mnist-seed0-20260808-v1"
EVIDENCE_CLASS="ordinary_mnist_lr_duration_counterfactual"
CONFIG_ROOT="${LR_DIV3_LONG50_SOURCE_ROOT}/configs/conv/perfectdiode_conv123_zero_bias_baseline_conv_lr_div3_long50_ordinary_mnist_seed0_20260808_v1"
STUDY_MANIFEST="${CONFIG_ROOT}/study_manifest.json"
CONFIGS=(
  "${CONFIG_ROOT}/conv1/00_baseline_sgd_conv_lr_div3_long50.json"
  "${CONFIG_ROOT}/conv2/00_baseline_sgd_conv_lr_div3_long50.json"
  "${CONFIG_ROOT}/conv3/00_baseline_sgd_conv_lr_div3_long50.json"
)

if [[ "${#CONFIGS[@]}" -ne 3 ]]; then
  echo "Expected exactly three frozen Conv-LR /3 long50 configs." >&2
  exit 3
fi
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ${#CONFIGS[@]} )); then
  echo "Array task ${SLURM_ARRAY_TASK_ID} is outside [0,2]." >&2
  exit 3
fi
if [[ ! -d "${LR_DIV3_LONG50_SOURCE_ROOT}" ]]; then
  echo "Missing frozen source directory: ${LR_DIV3_LONG50_SOURCE_ROOT}" >&2
  exit 3
fi
if [[ ! -s "${STUDY_MANIFEST}" ]]; then
  echo "Missing or empty study manifest: ${STUDY_MANIFEST}" >&2
  exit 3
fi
for config in "${CONFIGS[@]}"; do
  if [[ ! -s "${config}" ]]; then
    echo "Missing or empty exact config: ${config}" >&2
    exit 3
  fi
done

actual_config_set_sha256="$({
  for config in "${CONFIGS[@]}"; do
    sha256sum "${config}" | awk '{print $1}'
  done
} | sha256sum | awk '{print $1}')"
if [[ "${actual_config_set_sha256}" != "${LR_DIV3_LONG50_CONFIG_SET_SHA256}" ]]; then
  echo "Config-set SHA-256 mismatch: expected ${LR_DIV3_LONG50_CONFIG_SET_SHA256}, got ${actual_config_set_sha256}." >&2
  exit 3
fi

LR_DIV3_LONG50_RUN_MODE="${LR_DIV3_LONG50_RUN_MODE:-production}"
RUNNER_MODE_ARGS=()
case "${LR_DIV3_LONG50_RUN_MODE}" in
  production)
    RUN_ROOT="${LR_DIV3_LONG50_RESULT_ROOT}/runs"
    SUMMARY_ROOT="${LR_DIV3_LONG50_RESULT_ROOT}/task_summaries/production"
    ;;
  smoke)
    RUN_ROOT="${LR_DIV3_LONG50_RESULT_ROOT}/remote_canary"
    SUMMARY_ROOT="${LR_DIV3_LONG50_RESULT_ROOT}/task_summaries/remote_canary"
    RUNNER_MODE_ARGS+=(--smoke)
    ;;
  *)
    echo "LR_DIV3_LONG50_RUN_MODE must be production or smoke; got ${LR_DIV3_LONG50_RUN_MODE}." >&2
    exit 3
    ;;
esac

export MODULESHOME="/lustre/fshomisc/sup/spack_soft/environment-modules/4.3.1/gcc-11.3.1-wf7m7j6whgecysm2fm5n73sm4jg7txup"
export MODULEPATH="/lustre/fshomisc/sup/hpe/pub/module-rh/modulefiles:/lustre/fshomisc/sup/hpe/pub/modules-idris-env4/modulefiles/linux-rhel9-skylake_avx512"
MODULECMD="${MODULESHOME}/bin/modulecmd"
eval "$("${MODULECMD}" bash purge)"
eval "$("${MODULECMD}" bash load pytorch-gpu/py3/2.5.0)"

export KMP_DISABLE_SHM=1
export KMP_SHM_DISABLE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

mkdir -p "${RUN_ROOT}" "${SUMMARY_ROOT}"
cd "${LR_DIV3_LONG50_SOURCE_ROOT}"
SUMMARY_PATH="${SUMMARY_ROOT}/${SLURM_ARRAY_TASK_ID}.json"
SELECTED_CONFIG="${CONFIGS[SLURM_ARRAY_TASK_ID]}"

python -m experiments.exact_run \
  "${CONFIGS[@]}" \
  --output-root "${RUN_ROOT}" \
  --device cuda \
  --study-id "${STUDY_ID}" \
  --evidence-class "${EVIDENCE_CLASS}" \
  --target jean-zay \
  --summary-json "${SUMMARY_PATH}" \
  --gradient-trace-samples-per-epoch 5 \
  --checkpoint-every-epoch \
  --skip-terminal-official-test \
  "${RUNNER_MODE_ARGS[@]}"

python -m experiments.validate_zero_bias_exact_summary \
  --summary-json "${SUMMARY_PATH}" \
  --study-manifest "${STUDY_MANIFEST}" \
  --config "${SELECTED_CONFIG}" \
  --config-set-sha256 "${LR_DIV3_LONG50_CONFIG_SET_SHA256}" \
  --run-mode "${LR_DIV3_LONG50_RUN_MODE}"
