#!/bin/bash -l
#SBATCH --job-name=pd-c23-zero-bias
#SBATCH --account=umg@v100
#SBATCH --partition=gpu_p13
#SBATCH --qos=qos_gpu-t3
#SBATCH --constraint=v100-16g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=06:00:00
#SBATCH --array=0-1%2
#SBATCH --output=%x-%A_%a.out

set -euo pipefail

: "${BIAS_SOURCE_ROOT:?Set BIAS_SOURCE_ROOT to the frozen staged source.}"
: "${BIAS_CONFIG_ROOT:?Set BIAS_CONFIG_ROOT to the staged exact-config directory.}"
: "${BIAS_RESULT_ROOT:?Set BIAS_RESULT_ROOT to the study result directory.}"
: "${BIAS_CONFIG_SHA256_CONV2:?Set the Conv2 config SHA-256.}"
: "${BIAS_CONFIG_SHA256_CONV3:?Set the Conv3 config SHA-256.}"
: "${EXPERIMENT_SOURCE_COMMIT:?Set the frozen source commit.}"
: "${EXPERIMENT_SOURCE_ARCHIVE_SHA256:?Set the staged source archive SHA-256.}"
: "${SLURM_ARRAY_TASK_ID:?Submit this wrapper as a two-task Slurm array.}"

STUDY_ID="perfectdiode-conv23-legacy-zero-bias-medium-affine-seed0-20260731-v1"
CONFIGS=(
  "${BIAS_CONFIG_ROOT}/conv2_legacy_sgd_bias_zero_seed0.json"
  "${BIAS_CONFIG_ROOT}/conv3_legacy_sgd_bias_zero_seed0.json"
)
EXPECTED_HASHES=(
  "${BIAS_CONFIG_SHA256_CONV2}"
  "${BIAS_CONFIG_SHA256_CONV3}"
)
EXPECTED_ARMS=(
  "conv2_legacy_sgd_bias_zero_seed0"
  "conv3_legacy_sgd_bias_zero_seed0"
)
BIAS_RUN_MODE="${BIAS_RUN_MODE:-production}"
RUNNER_MODE_ARGS=()
case "${BIAS_RUN_MODE}" in
  production)
    EXPECTED_EPOCHS=30
    EXPECTED_TRACE_STEPS=150
    EXPECTED_CHECKPOINTS=31
    ;;
  smoke)
    EXPECTED_EPOCHS=1
    EXPECTED_TRACE_STEPS=1
    EXPECTED_CHECKPOINTS=2
    RUNNER_MODE_ARGS+=(--smoke)
    ;;
  *)
    echo "BIAS_RUN_MODE must be 'production' or 'smoke'; got ${BIAS_RUN_MODE}." >&2
    exit 3
    ;;
esac

if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ${#CONFIGS[@]} )); then
  echo "Array task ${SLURM_ARRAY_TASK_ID} is outside [0,1]." >&2
  exit 3
fi
if [[ ! -d "${BIAS_SOURCE_ROOT}" ]]; then
  echo "Missing frozen source directory: ${BIAS_SOURCE_ROOT}" >&2
  exit 3
fi
for index in "${!CONFIGS[@]}"; do
  config="${CONFIGS[index]}"
  if [[ ! -s "${config}" ]]; then
    echo "Missing or empty exact config: ${config}" >&2
    exit 3
  fi
  actual_hash="$(sha256sum "${config}" | awk '{print $1}')"
  if [[ "${actual_hash}" != "${EXPECTED_HASHES[index]}" ]]; then
    echo "Config ${index} SHA-256 mismatch: expected ${EXPECTED_HASHES[index]}, got ${actual_hash}" >&2
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

mkdir -p "${BIAS_RESULT_ROOT}/task_summaries"
cd "${BIAS_SOURCE_ROOT}"
SUMMARY_PATH="${BIAS_RESULT_ROOT}/task_summaries/${SLURM_ARRAY_TASK_ID}.json"

python -m experiments.exact_run \
  "${CONFIGS[@]}" \
  --output-root "${BIAS_RESULT_ROOT}/runs" \
  --device cuda \
  --study-id "${STUDY_ID}" \
  --evidence-class diagnostic_medium_affine_bias_ablation \
  --target jean-zay \
  --summary-json "${SUMMARY_PATH}" \
  --gradient-trace-samples-per-epoch 5 \
  --checkpoint-every-epoch \
  --skip-terminal-official-test \
  "${RUNNER_MODE_ARGS[@]}"

python - \
  "${SUMMARY_PATH}" \
  "${EXPECTED_EPOCHS}" \
  "${EXPECTED_TRACE_STEPS}" \
  "${EXPECTED_CHECKPOINTS}" \
  "${EXPECTED_HASHES[SLURM_ARRAY_TASK_ID]}" \
  "${EXPECTED_ARMS[SLURM_ARRAY_TASK_ID]}" \
  "${STUDY_ID}" <<'PY'
import json
import sys
from pathlib import Path

import torch

(
    summary_arg,
    expected_epochs_arg,
    expected_trace_steps_arg,
    expected_checkpoints_arg,
    expected_config_sha256,
    expected_arm_id,
    expected_study_id,
) = sys.argv[1:]
summary_path = Path(summary_arg)
expected_epochs = int(expected_epochs_arg)
expected_trace_steps = int(expected_trace_steps_arg)
expected_checkpoints = int(expected_checkpoints_arg)

rows = json.loads(summary_path.read_text(encoding="utf-8"))
if len(rows) != 1 or rows[0].get("status") != "complete":
    raise SystemExit(f"Task summary is not semantically complete: {rows!r}")
row = rows[0]
if int(row.get("epochs", -1)) != expected_epochs:
    raise SystemExit(f"Unexpected executed epoch count: {row!r}")
if row.get("config_sha256") != expected_config_sha256:
    raise SystemExit(f"Unexpected config hash in task summary: {row!r}")

run_dir = Path(row["output_dir"])
required = {
    "trace": run_dir / "gradient_trace.jsonl",
    "trace_metadata": run_dir / "gradient_trace_metadata.json",
    "checkpoint_index": run_dir / "epoch_checkpoint_index.jsonl",
    "manifest": run_dir / "manifest.json",
    "metrics": run_dir / "metrics.json",
    "result": run_dir / "result.json",
}
for name, path in required.items():
    if not path.is_file() or path.stat().st_size <= 0:
        raise SystemExit(f"Missing or empty {name}: {path}")

metadata = json.loads(required["trace_metadata"].read_text(encoding="utf-8"))
if metadata.get("status") != "complete":
    raise SystemExit(f"Gradient trace metadata is incomplete: {metadata!r}")
if int(metadata.get("recorded_steps", -1)) != expected_trace_steps:
    raise SystemExit(f"Unexpected trace step count: {metadata!r}")

checkpoints = [
    json.loads(line)
    for line in required["checkpoint_index"].read_text(encoding="utf-8").splitlines()
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
    model_path = run_dir / item["model_path"]
    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    bias_count = 0
    for spec, state in zip(payload["schema"], payload["states"]):
        name = str(spec["name"]).strip()
        if name.startswith("Bias_"):
            bias_count += 1
            if int(torch.count_nonzero(state).item()) != 0:
                raise SystemExit(f"Nonzero frozen bias in {model_path}: {name}")
    expected_bias_count = 2 if expected_arm_id.startswith("conv2_") else 3
    if bias_count != expected_bias_count:
        raise SystemExit(
            f"Expected {expected_bias_count} frozen biases in {model_path}; found {bias_count}"
        )

manifest = json.loads(required["manifest"].read_text(encoding="utf-8"))
if manifest.get("study_id") != expected_study_id:
    raise SystemExit(f"Unexpected study ID: {manifest!r}")
if manifest.get("arm_id") != expected_arm_id:
    raise SystemExit(f"Unexpected arm ID: {manifest!r}")
resolved = manifest.get("configuration", {}).get("resolved", {})
for name, rate in resolved.get("learning_rates_by_parameter", {}).items():
    if name.startswith("Bias_") and float(rate) != 0.0:
        raise SystemExit(f"Nonzero configured bias LR: {name}={rate}")
if manifest.get("dataset", {}).get("official_test_read") is not False:
    raise SystemExit(f"Official test was not kept disabled: {manifest!r}")
metrics = json.loads(required["metrics"].read_text(encoding="utf-8"))
if int(metrics.get("official_test_evaluations", 0)) != 0:
    raise SystemExit(f"Official test was evaluated: {metrics!r}")
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
