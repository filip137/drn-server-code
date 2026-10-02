#!/bin/bash -l
# Resources are supplied by experiments.launch; this wrapper runs one array cell.
set -euo pipefail
: "${SLURM_ARRAY_TASK_ID:?Submit as array 0-5.}"
: "${EXPERIMENT_SOURCE_COMMIT:?Set the staged source commit.}"
: "${EXPERIMENT_SOURCE_ARCHIVE_SHA256:?Set the staged source archive digest.}"
RESULT_ROOT="${1:?Supply the absolute study result root.}"
SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${SOURCE_ROOT}"

module purge
module load pytorch-gpu/py3/2.5.0
export KMP_DISABLE_SHM=1 KMP_SHM_DISABLE=1 OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
CONFIGS=(configs/conv/target_scale_conv2_20260928/*.json)
[[ "${#CONFIGS[@]}" -eq 6 ]]
(( SLURM_ARRAY_TASK_ID >= 0 && SLURM_ARRAY_TASK_ID < 6 ))
mkdir -p "${RESULT_ROOT}/task_summaries"
python -m experiments.exact_run "${CONFIGS[@]}" \
    --output-root "${RESULT_ROOT}/runs" --device cuda \
    --study-id conv2-target-scale-20260928-v1 \
    --evidence-class ordinary_mnist_target_scale_pilot --target jean-zay \
    --summary-json "${RESULT_ROOT}/task_summaries/${SLURM_ARRAY_TASK_ID}.json" \
    --skip-terminal-official-test
