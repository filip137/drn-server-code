#!/bin/bash -l
#SBATCH --job-name=pd-c123-zbias
#SBATCH --account=fmu@v100
#SBATCH --partition=gpu_p13
#SBATCH --qos=qos_gpu-t3
#SBATCH --constraint=v100-16g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=06:00:00

set -euo pipefail

: "${ZERO_BIAS_SOURCE_ROOT:?Set ZERO_BIAS_SOURCE_ROOT to the frozen staged source.}"
: "${ZERO_BIAS_RESULT_ROOT:?Set ZERO_BIAS_RESULT_ROOT to the study result directory.}"
: "${ZERO_BIAS_CONFIG_SET_SHA256:?Set ZERO_BIAS_CONFIG_SET_SHA256 to the ordered config-set digest.}"
: "${EXPERIMENT_SOURCE_COMMIT:?Set EXPERIMENT_SOURCE_COMMIT to the source commit.}"
: "${EXPERIMENT_SOURCE_ARCHIVE_SHA256:?Set EXPERIMENT_SOURCE_ARCHIVE_SHA256 to the staged archive digest.}"
: "${SLURM_ARRAY_TASK_ID:?Submit this wrapper as a Slurm array.}"

STUDY_ID="perfectdiode-conv123-zero-bias-ordinary-mnist-seed0-20260805-v1"
CONFIG_ROOT="${ZERO_BIAS_SOURCE_ROOT}/configs/conv/perfectdiode_conv123_zero_bias_ordinary_mnist_seed0_20260805_v1"
CONFIGS=(
  "${CONFIG_ROOT}/conv1/00_baseline_sgd_bias_zero_seed0.json"
  "${CONFIG_ROOT}/conv1/01_baseline_adam_bias_zero_seed0.json"
  "${CONFIG_ROOT}/conv1/02_ours_sgd_bias_zero_seed0.json"
  "${CONFIG_ROOT}/conv1/03_ours_adam_bias_zero_seed0.json"
  "${CONFIG_ROOT}/conv1/04_legacy_sgd_bias_zero_seed0.json"
  "${CONFIG_ROOT}/conv1/05_legacy_adam_bias_zero_seed0.json"
  "${CONFIG_ROOT}/conv2/00_baseline_sgd_bias_zero_seed0.json"
  "${CONFIG_ROOT}/conv2/01_baseline_adam_bias_zero_seed0.json"
  "${CONFIG_ROOT}/conv2/02_ours_sgd_bias_zero_seed0.json"
  "${CONFIG_ROOT}/conv2/03_ours_adam_bias_zero_seed0.json"
  "${CONFIG_ROOT}/conv2/04_legacy_sgd_bias_zero_seed0.json"
  "${CONFIG_ROOT}/conv2/05_legacy_adam_bias_zero_seed0.json"
  "${CONFIG_ROOT}/conv3/00_baseline_sgd_bias_zero_seed0.json"
  "${CONFIG_ROOT}/conv3/01_baseline_adam_bias_zero_seed0.json"
  "${CONFIG_ROOT}/conv3/02_ours_sgd_bias_zero_seed0.json"
  "${CONFIG_ROOT}/conv3/03_ours_adam_bias_zero_seed0.json"
  "${CONFIG_ROOT}/conv3/04_legacy_sgd_bias_zero_seed0.json"
  "${CONFIG_ROOT}/conv3/05_legacy_adam_bias_zero_seed0.json"
)

if [[ "${#CONFIGS[@]}" -ne 18 ]]; then
  echo "Expected exactly eighteen frozen configs." >&2
  exit 3
fi
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ${#CONFIGS[@]} )); then
  echo "Array task ${SLURM_ARRAY_TASK_ID} is outside [0,17]." >&2
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
if [[ "${actual_config_set_sha256}" != "${ZERO_BIAS_CONFIG_SET_SHA256}" ]]; then
  echo "Config-set SHA-256 mismatch: expected ${ZERO_BIAS_CONFIG_SET_SHA256}, got ${actual_config_set_sha256}." >&2
  exit 3
fi

ZERO_BIAS_RUN_MODE="${ZERO_BIAS_RUN_MODE:-production}"
case "${ZERO_BIAS_RUN_MODE}" in
  production)
    MODE_ARGS=()
    RUN_ROOT="${ZERO_BIAS_RESULT_ROOT}/runs"
    SUMMARY_ROOT="${ZERO_BIAS_RESULT_ROOT}/task_summaries/production"
    if (( SLURM_ARRAY_TASK_ID < 6 )); then
      EXPECTED_EPOCHS=10
    else
      EXPECTED_EPOCHS=30
    fi
    ;;
  smoke)
    MODE_ARGS=(--smoke)
    RUN_ROOT="${ZERO_BIAS_RESULT_ROOT}/remote_canary"
    SUMMARY_ROOT="${ZERO_BIAS_RESULT_ROOT}/task_summaries/remote_canary"
    EXPECTED_EPOCHS=1
    ;;
  *)
    echo "ZERO_BIAS_RUN_MODE must be production or smoke; got ${ZERO_BIAS_RUN_MODE}." >&2
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
cd "${ZERO_BIAS_SOURCE_ROOT}"
SUMMARY_PATH="${SUMMARY_ROOT}/${SLURM_ARRAY_TASK_ID}.json"

python -m experiments.exact_run \
  "${CONFIGS[@]}" \
  --output-root "${RUN_ROOT}" \
  --device cuda \
  --study-id "${STUDY_ID}" \
  --evidence-class ordinary_mnist_bias_ablation \
  --target jean-zay \
  --summary-json "${SUMMARY_PATH}" \
  --skip-terminal-official-test \
  "${MODE_ARGS[@]}"

python - \
  "${SUMMARY_PATH}" \
  "${EXPECTED_EPOCHS}" \
  "${CONFIGS[SLURM_ARRAY_TASK_ID]}" \
  "${STUDY_ID}" <<'PY'
import json
import sys
from pathlib import Path

import torch

summary_path = Path(sys.argv[1])
expected_epochs = int(sys.argv[2])
config_path = Path(sys.argv[3])
expected_study_id = sys.argv[4]
config = json.loads(config_path.read_text(encoding="utf-8"))
rows = json.loads(summary_path.read_text(encoding="utf-8"))
if len(rows) != 1 or rows[0].get("status") != "complete":
    raise SystemExit(f"Task summary is not semantically complete: {rows!r}")
row = rows[0]
if int(row.get("epochs", -1)) != expected_epochs:
    raise SystemExit(f"Unexpected executed epoch count: {row!r}")
if row.get("config_sha256") is None:
    raise SystemExit(f"Task summary lacks a config digest: {row!r}")

run_dir = Path(row["output_dir"])
required = {
    "manifest": run_dir / "manifest.json",
    "status": run_dir / "status.json",
    "metrics": run_dir / "metrics.json",
    "result": run_dir / "result.json",
    "best_model": run_dir / "best_model.pt",
    "final_model": run_dir / "final_model.pt",
}
for name, path in required.items():
    if not path.is_file() or path.stat().st_size <= 0:
        raise SystemExit(f"Missing or empty {name}: {path}")

manifest = json.loads(required["manifest"].read_text(encoding="utf-8"))
status = json.loads(required["status"].read_text(encoding="utf-8"))
metrics = json.loads(required["metrics"].read_text(encoding="utf-8"))
result = json.loads(required["result"].read_text(encoding="utf-8"))
if manifest.get("study_id") != expected_study_id:
    raise SystemExit(f"Unexpected study ID: {manifest!r}")
if manifest.get("arm_id") != config["arm_id"]:
    raise SystemExit(f"Unexpected arm ID: {manifest!r}")
if status.get("state") != "complete":
    raise SystemExit(f"Run status is not complete: {status!r}")
if result.get("completion", {}).get("criteria_met") is not True:
    raise SystemExit(f"Completion criteria were not met: {result!r}")

dataset = manifest.get("dataset", {})
if dataset.get("variant") != "ordinary":
    raise SystemExit(f"Expected ordinary MNIST, got: {dataset!r}")
if dataset.get("factory") != "labs.datasets.MnistTrainValidationDataset":
    raise SystemExit(f"Unexpected dataset factory: {dataset!r}")
if dataset.get("official_test_read") is not False:
    raise SystemExit(f"Official MNIST test was not kept unread: {dataset!r}")
if int(metrics.get("official_test_evaluations", 0)) != 0:
    raise SystemExit(f"Official MNIST test was evaluated: {metrics!r}")

resolved = manifest.get("configuration", {}).get("resolved", {})
params = resolved.get("datasets", {}).get("mnist", {}).get("params", {})
if params.get("affine_config", "missing") is not None:
    raise SystemExit(f"Affine corruption was not disabled: {params!r}")
bias_names = [name for name in config["parameter_order"] if name.startswith("Bias_")]
for name in bias_names:
    rate = resolved.get("learning_rates_by_parameter", {}).get(name)
    if float(rate) != 0.0:
        raise SystemExit(f"Nonzero configured bias LR: {name}={rate}")

for checkpoint_name in ("best_model", "final_model"):
    checkpoint_path = required[checkpoint_name]
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    seen = []
    for spec, state in zip(payload["schema"], payload["states"]):
        name = str(spec["name"]).strip()
        if name.startswith("Bias_"):
            seen.append(name)
            if int(torch.count_nonzero(state).item()) != 0:
                raise SystemExit(f"Nonzero frozen bias in {checkpoint_path}: {name}")
    if sorted(seen) != sorted(bias_names):
        raise SystemExit(
            f"Checkpoint bias inventory mismatch for {checkpoint_path}: "
            f"expected {bias_names}, got {seen}"
        )
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
