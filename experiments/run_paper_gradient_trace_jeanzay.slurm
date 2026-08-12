#!/bin/bash -l
#SBATCH --job-name=pd-grad-trace
#SBATCH --account=umg@v100
#SBATCH --partition=gpu_p13
#SBATCH --qos=qos_gpu-t3
#SBATCH --constraint=v100-16g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=03:00:00

set -euo pipefail

: "${TRACE_MODE:?Set TRACE_MODE to canary or production.}"
: "${TRACE_SOURCE_ROOT:?Set TRACE_SOURCE_ROOT to the frozen staged source.}"
: "${TRACE_RESULT_ROOT:?Set TRACE_RESULT_ROOT to the study result directory.}"
: "${EXPERIMENT_SOURCE_COMMIT:?Set the frozen source commit.}"
: "${EXPERIMENT_SOURCE_ARCHIVE_SHA256:?Set the staged source archive SHA-256.}"
: "${SLURM_ARRAY_TASK_ID:?Submit this wrapper as a Slurm array.}"

case "${TRACE_MODE}" in
  canary|production)
    ;;
  *)
    echo "Unsupported TRACE_MODE=${TRACE_MODE}" >&2
    exit 2
    ;;
esac

CONFIG_ROOT="${TRACE_SOURCE_ROOT}/configs/conv/paper_medium_affine_perfectdiode_wide_seed0_20260729_v1"
CONFIGS=(
  "${CONFIG_ROOT}/conv1/00_baseline_sgd_seed0.json"
  "${CONFIG_ROOT}/conv1/01_baseline_adam_seed0.json"
  "${CONFIG_ROOT}/conv1/02_ours_sgd_seed0.json"
  "${CONFIG_ROOT}/conv1/03_ours_adam_seed0.json"
  "${CONFIG_ROOT}/conv1/04_legacy_sgd_seed0.json"
  "${CONFIG_ROOT}/conv1/05_legacy_adam_seed0.json"
  "${CONFIG_ROOT}/conv2/00_baseline_sgd_seed0.json"
  "${CONFIG_ROOT}/conv2/02_ours_sgd_seed0.json"
  "${CONFIG_ROOT}/conv2/04_legacy_sgd_seed0.json"
  "${CONFIG_ROOT}/conv3/00_baseline_sgd_seed0.json"
  "${CONFIG_ROOT}/conv3/02_ours_sgd_seed0.json"
  "${CONFIG_ROOT}/conv3/04_legacy_sgd_seed0.json"
)

if [[ "${#CONFIGS[@]}" -ne 12 ]]; then
  echo "Expected exactly twelve frozen diagnostic configs." >&2
  exit 3
fi
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ${#CONFIGS[@]} )); then
  echo "Array task ${SLURM_ARRAY_TASK_ID} is outside [0,11]." >&2
  exit 3
fi
for config in "${CONFIGS[@]}"; do
  if [[ ! -s "${config}" ]]; then
    echo "Missing or empty exact config: ${config}" >&2
    exit 3
  fi
done

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

cd "${TRACE_SOURCE_ROOT}"

CONFIG="${CONFIGS[SLURM_ARRAY_TASK_ID]}"
ARCHITECTURE="$(basename "$(dirname "${CONFIG}")")"
RUN_ROOT="${TRACE_RESULT_ROOT}/runs/${ARCHITECTURE}"
SUMMARY_DIR="${TRACE_RESULT_ROOT}/task_summaries/${TRACE_MODE}"
mkdir -p "${RUN_ROOT}" "${SUMMARY_DIR}"
SUMMARY_PATH="${SUMMARY_DIR}/${SLURM_ARRAY_TASK_ID}.json"

MODE_ARGS=()
EXPECTED_EPOCHS=10
EXPECTED_TRACE_STEPS=50
EXPECTED_CHECKPOINTS=11
if [[ "${TRACE_MODE}" == "canary" ]]; then
  MODE_ARGS+=(--smoke)
  EXPECTED_EPOCHS=1
  EXPECTED_TRACE_STEPS=1
  EXPECTED_CHECKPOINTS=2
else
  MODE_ARGS+=(--epoch-override 10)
fi

python -m experiments.exact_run \
  "${CONFIGS[@]}" \
  --output-root "${RUN_ROOT}" \
  --device cuda \
  --study-id perfectdiode-paper-gradient-trace-limited-20260731-v1 \
  --evidence-class diagnostic_medium_affine_gradient_trace \
  --target jean-zay \
  --summary-json "${SUMMARY_PATH}" \
  --gradient-trace-samples-per-epoch 5 \
  --checkpoint-every-epoch \
  --skip-terminal-official-test \
  "${MODE_ARGS[@]}"

python - "${SUMMARY_PATH}" "${EXPECTED_EPOCHS}" "${EXPECTED_TRACE_STEPS}" "${EXPECTED_CHECKPOINTS}" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
expected_epochs = int(sys.argv[2])
expected_trace_steps = int(sys.argv[3])
expected_checkpoints = int(sys.argv[4])
rows = json.loads(summary_path.read_text(encoding="utf-8"))
if len(rows) != 1 or rows[0].get("status") != "complete":
    raise SystemExit(f"Task summary is not semantically complete: {rows!r}")
row = rows[0]
if int(row.get("epochs", -1)) != expected_epochs:
    raise SystemExit(f"Unexpected executed epoch count: {row!r}")
run_dir = Path(row["output_dir"])
trace_path = run_dir / "gradient_trace.jsonl"
metadata_path = run_dir / "gradient_trace_metadata.json"
checkpoint_index_path = run_dir / "epoch_checkpoint_index.jsonl"
for path in (trace_path, metadata_path, checkpoint_index_path):
    if not path.is_file() or path.stat().st_size <= 0:
        raise SystemExit(f"Missing or empty diagnostic artifact: {path}")
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
if metadata.get("status") != "complete":
    raise SystemExit(f"Gradient trace metadata is incomplete: {metadata!r}")
if int(metadata.get("recorded_steps", -1)) != expected_trace_steps:
    raise SystemExit(f"Unexpected trace step count: {metadata!r}")
checkpoints = [
    json.loads(line)
    for line in checkpoint_index_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
if len(checkpoints) != expected_checkpoints:
    raise SystemExit(f"Unexpected diagnostic checkpoint count: {checkpoints!r}")
if [int(item["epoch"]) for item in checkpoints] != list(range(expected_checkpoints)):
    raise SystemExit(f"Unexpected diagnostic checkpoint epochs: {checkpoints!r}")
for item in checkpoints:
    for key in ("model_path", "optimizer_path"):
        path = run_dir / item[key]
        if not path.is_file() or path.stat().st_size <= 0:
            raise SystemExit(f"Missing or empty checkpoint: {path}")
PY

RUN_DIR="$(python - "${SUMMARY_PATH}" <<'PY'
import json
import sys
from pathlib import Path

rows = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(rows[0]["output_dir"])
PY
)"
python -m experiments.reporting validate-run "${RUN_DIR}"
