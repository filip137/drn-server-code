#!/usr/bin/env bash
# One isolated two-worker study on an otherwise idle RTX5090.
# Usage: SOURCE OUTPUT DATA PYTHON
set -euo pipefail
[[ $# == 4 ]] || exit 2
source_dir=$(realpath "$1")
output_dir=$(realpath "$2")
data_dir=$(realpath "$3")
python_bin=$4
target=$(hostname -s)
mkdir "$output_dir/launcher-lock"
echo $$ > "$output_dir/launcher.pid"
date -u +%FT%TZ > "$output_dir/started_at"
trap 'code=$?; echo "$code" > "$output_dir/exit_code"; date -u +%FT%TZ > "$output_dir/finished_at"' EXIT
export KMP_DISABLE_SHM=1 KMP_SHM_DISABLE=1 OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 MPLCONFIGDIR="$output_dir/mpl-cache"
cd "$source_dir"
sha256sum -c STABILITY_SHA256SUMS > "$output_dir/source-verification.log"
echo waiting-for-free-gpu > "$output_dir/phase"
deadline=$((SECONDS + 43200))
while true; do
    gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader)
    [[ "$gpu_name" == *RTX\ 5090* ]] || exit 74
    # An idle MPS daemon is not a training job. Require near-empty VRAM too,
    # so clients hidden behind MPS cannot be mistaken for an available GPU.
    gpu_pids=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader,nounits | awk '!/nvidia-cuda-mps-server/ {print $1}')
    used_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    date -u +%FT%TZ > "$output_dir/last_resource_check"
    nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader > "$output_dir/gpu_last.csv"
    [[ -z "$gpu_pids" ]] && (( used_mib < 256 )) && break
    (( SECONDS < deadline )) || exit 75
    sleep 30
done
echo smoking > "$output_dir/phase"
mapfile -t configs < <(find configs/stability -maxdepth 1 -name '*.json' | sort)
[[ ${#configs[@]} == 9 ]] || exit 2
# Two same-runner smokes concurrently, followed by the remaining smoke cases.
smoke_worker() {
    local lane=$1
    for ((i=lane; i<9; i+=2)); do
        timeout --signal=TERM --kill-after=30s 10m "$python_bin" -m experiments.exact_run "${configs[$i]}" \
            --output-root "$output_dir/remote-smokes" --device cuda \
            --dataset-root "$data_dir" --study-id eqprop-beta-training-stability-20260918-v1 \
            --target "$target" --gradient-trace-samples-per-epoch 8 --smoke
    done
}
smoke_worker 0 > "$output_dir/smoke-lane0.log" 2>&1 &
smoke0=$!
smoke_worker 1 > "$output_dir/smoke-lane1.log" 2>&1 &
smoke1=$!
wait "$smoke0"
wait "$smoke1"
"$python_bin" - "$output_dir" <<'PY'
import json, math, sys
from pathlib import Path
from experiments.reporting import validate_run
root = Path(sys.argv[1])
runs = sorted((root / 'remote-smokes/smoke').glob('*/result.json'))
assert len(runs) == 9, len(runs)
for result in runs:
    errors = validate_run(result.parent)
    assert not errors, (result, errors)
    metrics = json.loads((result.parent / 'metrics.json').read_text())
    assert metrics['official_test_evaluations'] == 0
    assert math.isfinite(metrics['final_train_loss'])
    assert math.isfinite(metrics['final_validation_accuracy'])
    assert (result.parent / 'gradient_trace.jsonl').stat().st_size > 0
(root / 'smoke-validation.json').write_text(json.dumps({'validated': 9, 'official_test_read': False}, indent=2)+'\n')
PY
echo running > "$output_dir/phase"
date -u +%FT%TZ > "$output_dir/production_started_at"
worker() {
    local lane=$1 overall=0 code=0
    for ((i=lane; i<9; i+=2)); do
        case_name=$(basename "${configs[$i]}" .json)
        date -u +%FT%TZ > "$output_dir/$case_name.started_at"
        code=0
        timeout --signal=TERM --kill-after=30s 4h "$python_bin" -m experiments.exact_run "${configs[$i]}" \
            --output-root "$output_dir/training" --device cuda \
            --dataset-root "$data_dir" --study-id eqprop-beta-training-stability-20260918-v1 \
            --target "$target" --gradient-trace-samples-per-epoch 8 \
            > "$output_dir/$case_name.log" 2>&1 || code=$?
        echo "$code" > "$output_dir/$case_name.exit_code"
        date -u +%FT%TZ > "$output_dir/$case_name.finished_at"
        (( code == 0 )) || overall=1
    done
    return "$overall"
}
worker 0 &
lane0=$!
worker 1 &
lane1=$!
code0=0; code1=0
wait "$lane0" || code0=$?
wait "$lane1" || code1=$?
echo terminal > "$output_dir/phase"
(( code0 == 0 && code1 == 0 ))
