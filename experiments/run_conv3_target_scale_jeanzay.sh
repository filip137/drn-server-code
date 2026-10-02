#!/bin/bash -l
# One frozen Conv3 target-amplitude case per array task.
set -euo pipefail
: "${SLURM_ARRAY_TASK_ID:?Submit as array 0-5.}"
: "${EXPERIMENT_SOURCE_COMMIT:?Set the staged source commit.}"
: "${EXPERIMENT_SOURCE_ARCHIVE_SHA256:?Set the staged source archive digest.}"
RESULT_ROOT="${1:?Supply the absolute study result root.}"
SOURCE_ROOT="${2:?Supply the absolute frozen source root; Slurm relocates this script.}"
MODE="${3:-production}"
[[ "$SOURCE_ROOT" = /* && -d "$SOURCE_ROOT" ]] || { echo "Invalid frozen source root: $SOURCE_ROOT" >&2; exit 2; }
cd "${SOURCE_ROOT}"
module purge
module load pytorch-gpu/py3/2.5.0
export KMP_DISABLE_SHM=1 KMP_SHM_DISABLE=1 OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
CONFIGS=(configs/conv/target_scale_conv3_20260929/*.json)
[[ "${#CONFIGS[@]}" -eq 6 ]] || { echo "Expected six configs under $SOURCE_ROOT" >&2; exit 2; }
(( SLURM_ARRAY_TASK_ID >= 0 && SLURM_ARRAY_TASK_ID < 6 ))
EXTRA=()
case "$MODE" in
    production) ;;
    smoke) EXTRA+=(--smoke) ;;
    *) echo "Unsupported launch mode: $MODE" >&2; exit 2 ;;
esac
mkdir -p "${RESULT_ROOT}/task_summaries"
# Pass the full list: exact_run performs the Slurm index selection once.
python -m experiments.exact_run "${CONFIGS[@]}" \
    --output-root "${RESULT_ROOT}/runs" --device cuda \
    --study-id conv3-target-scale-20260929-v1 \
    --evidence-class ordinary_mnist_target_scale_pilot --target jean-zay \
    --checkpoint-every-epoch --epoch-chunk-size 30 \
    "${EXTRA[@]}" \
    --summary-json "${RESULT_ROOT}/task_summaries/${SLURM_ARRAY_TASK_ID}.json" \
    --skip-terminal-official-test
