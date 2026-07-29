#!/bin/bash -l
#SBATCH --account=fmu@v100
#SBATCH --partition=gpu_p13
#SBATCH --qos=qos_gpu-t3
#SBATCH --constraint=v100-16g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread

set -euo pipefail

: "${PAPER_ARCH:?Set PAPER_ARCH to conv1, conv2, or conv3.}"
: "${PAPER_MODE:?Set PAPER_MODE to smoke or full.}"
: "${PAPER_SOURCE_ROOT:?Set PAPER_SOURCE_ROOT to the frozen staged source.}"
: "${PAPER_RESULT_ROOT:?Set PAPER_RESULT_ROOT to the study result directory.}"
: "${EXPERIMENT_SOURCE_COMMIT:?Set the frozen source commit.}"
: "${EXPERIMENT_SOURCE_ARCHIVE_SHA256:?Set the staged source archive SHA-256.}"
: "${SLURM_ARRAY_TASK_ID:?Submit this wrapper as a Slurm array.}"

case "${PAPER_ARCH}" in
  conv1|conv2|conv3)
    ;;
  *)
    echo "Unsupported PAPER_ARCH=${PAPER_ARCH}" >&2
    exit 2
    ;;
esac
case "${PAPER_MODE}" in
  smoke|full)
    ;;
  *)
    echo "Unsupported PAPER_MODE=${PAPER_MODE}" >&2
    exit 2
    ;;
esac

CONFIG_ROOT="${PAPER_SOURCE_ROOT}/configs/conv/paper_medium_affine_perfectdiode_wide_seed0_20260729_v1"
CONFIG_DIR="${CONFIG_ROOT}/${PAPER_ARCH}"
mapfile -t CONFIGS < <(find "${CONFIG_DIR}" -maxdepth 1 -type f -name '*.json' | sort)
if [[ "${#CONFIGS[@]}" -ne 6 ]]; then
  echo "Expected six exact configs in ${CONFIG_DIR}; found ${#CONFIGS[@]}." >&2
  exit 3
fi
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ${#CONFIGS[@]} )); then
  echo "Array task ${SLURM_ARRAY_TASK_ID} is outside [0,5]." >&2
  exit 3
fi

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

cd "${PAPER_SOURCE_ROOT}"

RUN_ROOT="${PAPER_RESULT_ROOT}/${PAPER_ARCH}"
SUMMARY_DIR="${PAPER_RESULT_ROOT}/task_summaries/${PAPER_MODE}/${PAPER_ARCH}"
mkdir -p "${RUN_ROOT}" "${SUMMARY_DIR}"
SUMMARY_PATH="${SUMMARY_DIR}/${SLURM_ARRAY_TASK_ID}.json"

MODE_ARGS=()
if [[ "${PAPER_MODE}" == "smoke" ]]; then
  MODE_ARGS+=(--smoke)
fi

python -m experiments.exact_run \
  "${CONFIGS[@]}" \
  --output-root "${RUN_ROOT}" \
  --device cuda \
  --study-id perfectdiode-paper-medium-affine-wide-seed0-20260729-v1 \
  --evidence-class paper_medium_affine \
  --target jean-zay \
  --summary-json "${SUMMARY_PATH}" \
  "${MODE_ARGS[@]}"

python - "${SUMMARY_PATH}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
rows = json.loads(path.read_text(encoding="utf-8"))
if len(rows) != 1 or rows[0].get("status") != "complete":
    raise SystemExit(f"Task summary is not semantically complete: {rows!r}")
PY
