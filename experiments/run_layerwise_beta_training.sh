#!/usr/bin/env bash
# Direct independent workers; configs are explicit arguments, never a discovery queue.
# Usage: SOURCE OUTPUT DATA PYTHON CONFIG...
set -euo pipefail
[[ $# -ge 5 ]] || exit 2
source_dir=$(realpath "$1"); output_dir=$(realpath "$2"); data_dir=$(realpath "$3")
python_bin=$4; shift 4
configs=("$@"); target=${EXPERIMENT_TARGET:-$(hostname -s)}
expected_gpu=${EXPECTED_GPU_MODEL:-RTX 5090}
mkdir "$output_dir/launcher-lock"
echo $$ > "$output_dir/launcher.pid"
date -u +%FT%TZ > "$output_dir/started_at"
trap 'rc=$?; echo "$rc" > "$output_dir/exit_code"; date -u +%FT%TZ > "$output_dir/finished_at"' EXIT
export KMP_DISABLE_SHM=1 KMP_SHM_DISABLE=1 OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 MPLCONFIGDIR="$output_dir/mpl-cache"
export GIT_CEILING_DIRECTORIES=$(dirname "$source_dir")
cd "$source_dir"
export EXPERIMENT_SOURCE_COMMIT=$(cat SOURCE_COMMIT)
sha256sum -c --quiet LAYERWISE_SHA256SUMS
[[ $(nvidia-smi --query-gpu=name --format=csv,noheader) == *"$expected_gpu"* ]] || exit 74
gpu_pids=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader,nounits | awk '!/nvidia-cuda-mps-server/ {print $1}')
used_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
[[ -z "$gpu_pids" ]] && (( used_mib < 256 )) || exit 75
echo smoking > "$output_dir/phase"
pids=()
for config in "${configs[@]}"; do
    name=$(basename "$config" .json)
    timeout --signal=TERM --kill-after=30s 10m "$python_bin" -m experiments.exact_run "$config" \
        --output-root "$output_dir/smokes" --device cuda --dataset-root "$data_dir" \
        --study-id eqprop-layerwise-beta-training-20260919-v1 --target "$target" \
        --gradient-trace-samples-per-epoch 8 --smoke > "$output_dir/$name.smoke.log" 2>&1 &
    pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
"$python_bin" - "$output_dir" "${#configs[@]}" <<'PY'
import json, math, sys
from pathlib import Path
from experiments.reporting import validate_run
root=Path(sys.argv[1]); runs=list((root/'smokes/smoke').glob('*/result.json'))
assert len(runs)==int(sys.argv[2])
for p in runs:
    assert not validate_run(p.parent)
    m=json.loads((p.parent/'metrics.json').read_text())
    assert m['official_test_evaluations']==0 and math.isfinite(m['final_train_loss'])
(root/'smoke-validation.json').write_text(json.dumps({'validated':len(runs),'official_test_read':False})+'\n')
PY
echo running > "$output_dir/phase"
worker() {
    local config=$1 name code=0
    name=$(basename "$config" .json)
    date -u +%FT%TZ > "$output_dir/$name.started_at"
    timeout --signal=TERM --kill-after=30s 4h "$python_bin" -m experiments.exact_run "$config" \
        --output-root "$output_dir/training" --device cuda --dataset-root "$data_dir" \
        --study-id eqprop-layerwise-beta-training-20260919-v1 --target "$target" \
        --gradient-trace-samples-per-epoch 8 > "$output_dir/$name.log" 2>&1 || code=$?
    echo "$code" > "$output_dir/$name.exit_code"
    date -u +%FT%TZ > "$output_dir/$name.finished_at"
    return "$code"
}
pids=()
for config in "${configs[@]}"; do worker "$config" & pids+=("$!"); done
overall=0
for pid in "${pids[@]}"; do wait "$pid" || overall=1; done
echo terminal > "$output_dir/phase"
exit "$overall"
