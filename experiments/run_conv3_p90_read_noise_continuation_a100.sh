#!/usr/bin/env bash
# One A100 allocation per ten-epoch chunk. SOURCE CONFIGS OUTPUT DATA CHUNK [canary].
set -euo pipefail
(( $# == 5 || $# == 6 )) || exit 2
chunk=${5:?Expected chunk 0,1,2}
mode=${6:-production}
[[ $mode == production || $mode == canary ]] || exit 2
if [[ $mode == canary && $chunk == all ]]; then
  for phase in 0 1 2; do
    bash "$0" "$1" "$2" "$3" "$4" "$phase" canary
  done
  exit 0
fi
[[ $chunk =~ ^[012]$ ]] || exit 2
source_dir=$1
config_dir=$2
output_root=$3
data_dir=$4
task=${SLURM_ARRAY_TASK_ID:?Expected a Slurm array task index}
[[ $task =~ ^[0-9]+$ ]] && (( task < 18 )) || exit 2
mapfile -t configs < "$config_dir/config-names.txt"
(( ${#configs[@]} == 18 )) || exit 2
case_dir="$output_root/task_$task"
segment="$case_dir/segments/chunk_$chunk"
mkdir -p "$segment"
mkdir "$segment/worker_started"
printf '%s\n' "$$" > "$segment/worker_started/pid"
date --iso-8601=seconds > "$segment/worker_started/started_at"
finish() {
  code=$?
  date --iso-8601=seconds > "$segment/worker_started/finished_at"
  printf '%s\n' "$code" > "$segment/worker_started/exit_code"
}
trap finish EXIT
export MODULESHOME=/lustre/fshomisc/sup/spack_soft/environment-modules/4.3.1/gcc-11.3.1-wf7m7j6whgecysm2fm5n73sm4jg7txup
export MODULEPATH=/lustre/fshomisc/sup/hpe/pub/module-rh/modulefiles:/lustre/fshomisc/sup/hpe/pub/modules-idris-env4/modulefiles/linux-rhel9-skylake_avx512
module_command="$MODULESHOME/bin/modulecmd"
eval "$("$module_command" bash purge)"
eval "$("$module_command" bash load arch/a100)"
eval "$("$module_command" bash load pytorch-gpu/py3/2.5.0)"
export KMP_DISABLE_SHM=1 KMP_SHM_DISABLE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2
export MPLCONFIGDIR="/tmp/mpl-p90-a100-${SLURM_JOB_ID}"
export GIT_CEILING_DIRECTORIES
GIT_CEILING_DIRECTORIES=$(dirname "$source_dir")
cd "$source_dir"
export EXPERIMENT_SOURCE_COMMIT EXPERIMENT_SOURCE_ARCHIVE_SHA256
EXPERIMENT_SOURCE_COMMIT=$(cat SOURCE_COMMIT)
EXPERIMENT_SOURCE_ARCHIVE_SHA256=$(cat SOURCE_ARCHIVE_SHA256)
sha256sum --check --quiet LAYERWISE_SHA256SUMS
python - "$segment/gpu.json" <<'PYGPU'
import json,sys,torch
assert torch.cuda.device_count()==1
p=torch.cuda.get_device_properties(0)
assert 'A100' in p.name, p.name
with open(sys.argv[1],'w') as f:
    json.dump(dict(name=p.name,total_memory=p.total_memory,torch=torch.__version__,cuda=torch.version.cuda),f,indent=2)
PYGPU
config="$config_dir/${configs[$task]}"
[[ $mode != canary ]] || config="$config_dir/${configs[11]}"
extra=()
chunk_epochs=10
if [[ $mode == canary ]]; then
  extra+=(--smoke)
  chunk_epochs=1
fi
# A retry cannot silently skip or repeat an epoch block.
python - "$case_dir/runs" "$config" "$chunk" "$chunk_epochs" "$mode" <<'PYCHECK'
import hashlib,json,sys
from pathlib import Path
root,config,chunk,width,mode=sys.argv[1:]
config=Path(config);sha=hashlib.sha256(config.read_bytes()).hexdigest()
run=Path(root)/('smoke' if mode=='canary' else '')/f'000_{config.stem}_{sha[:8]}'
checkpoint=run/'continuation.json'
if int(chunk)==0:
    assert not checkpoint.exists() and not (run/'manifest.json').exists(), 'Duplicate first chunk'
else:
    assert json.loads(checkpoint.read_text())['completed_epochs']==int(chunk)*int(width), 'Wrong continuation epoch'
assert not (run/'result.json').exists(), 'Run already complete'
PYCHECK
set +e
timeout --signal=TERM --kill-after=20s 5340s \
  python -m experiments.exact_run "$config" --index 0 "${extra[@]}" \
  --epoch-chunk-size "$chunk_epochs" \
  --output-root "$case_dir/runs" --device cuda --dataset-root "$data_dir" \
  --target jean-zay --gradient-trace-samples-per-epoch 8 \
  --summary-json "$segment/summary.json"
code=$?
set -e
expected=75
[[ $chunk != 2 ]] || expected=0
if [[ $code != "$expected" ]]; then
  (( code != 0 )) || code=1
  exit "$code"
fi
# Native75 is a successful planned pause; Slurm afterok may release the next chunk.
python - "$segment/summary.json" "$chunk" "$chunk_epochs" <<'PYSUMMARY'
import json,sys
rows=json.load(open(sys.argv[1]));chunk=int(sys.argv[2]);width=int(sys.argv[3])
assert len(rows)==1
row=rows[0]
assert row['status']==('complete' if chunk==2 else 'paused'),row
if chunk<2: assert row['completed_epochs']==(chunk+1)*width
else: assert row['epochs']==3*width
PYSUMMARY
exit 0
