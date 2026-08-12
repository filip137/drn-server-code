#!/bin/bash -l
#SBATCH --job-name=pd-c123-zb-lr30
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
#SBATCH --array=0-17%6
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err

set -euo pipefail

: "${LR_COUNTERFACTUAL_SOURCE_ROOT:?Set the frozen staged source directory.}"
: "${LR_COUNTERFACTUAL_RESULT_ROOT:?Set the study result directory.}"
: "${LR_COUNTERFACTUAL_CONFIG_SET_SHA256:?Set the ordered config-set SHA-256.}"
: "${EXPERIMENT_SOURCE_COMMIT:?Set the frozen source commit.}"
: "${EXPERIMENT_SOURCE_ARCHIVE_SHA256:?Set the staged source archive SHA-256.}"
: "${SLURM_ARRAY_TASK_ID:?Submit this wrapper as an eighteen-task Slurm array.}"

STUDY_ID="perfectdiode-conv123-zero-bias-lr-counterfactual-ordinary-mnist-seed0-20260807-v1"
EVIDENCE_CLASS="ordinary_mnist_lr_counterfactual"
CONFIG_ROOT="${LR_COUNTERFACTUAL_SOURCE_ROOT}/configs/conv/perfectdiode_conv123_zero_bias_lr_counterfactual_ordinary_mnist_seed0_20260807_v1"
STUDY_MANIFEST="${CONFIG_ROOT}/study_manifest.json"
CONFIGS=(
  "${CONFIG_ROOT}/conv1/00_baseline_sgd_current.json"
  "${CONFIG_ROOT}/conv1/01_baseline_sgd_conv3x_stress.json"
  "${CONFIG_ROOT}/conv1/02_baseline_sgd_ours_proposal_exact.json"
  "${CONFIG_ROOT}/conv1/03_baseline_sgd_legacy_proposal_exact.json"
  "${CONFIG_ROOT}/conv1/04_ours_sgd_current.json"
  "${CONFIG_ROOT}/conv1/05_legacy_sgd_current.json"
  "${CONFIG_ROOT}/conv2/00_baseline_sgd_current.json"
  "${CONFIG_ROOT}/conv2/01_baseline_sgd_conv3x_stress.json"
  "${CONFIG_ROOT}/conv2/02_baseline_sgd_ours_proposal_exact.json"
  "${CONFIG_ROOT}/conv2/03_baseline_sgd_legacy_proposal_exact.json"
  "${CONFIG_ROOT}/conv2/04_ours_sgd_current.json"
  "${CONFIG_ROOT}/conv2/05_legacy_sgd_current.json"
  "${CONFIG_ROOT}/conv3/00_baseline_sgd_current.json"
  "${CONFIG_ROOT}/conv3/01_baseline_sgd_conv3x_stress.json"
  "${CONFIG_ROOT}/conv3/02_baseline_sgd_ours_proposal_exact.json"
  "${CONFIG_ROOT}/conv3/03_baseline_sgd_legacy_proposal_exact.json"
  "${CONFIG_ROOT}/conv3/04_ours_sgd_current.json"
  "${CONFIG_ROOT}/conv3/05_legacy_sgd_current.json"
)
EXPECTED_ARMS=(
  "conv1_baseline_sgd_bias_zero_lr_current_seed0"
  "conv1_baseline_sgd_bias_zero_conv3x_stress_seed0"
  "conv1_baseline_sgd_bias_zero_ours_proposal_exact_seed0"
  "conv1_baseline_sgd_bias_zero_legacy_proposal_exact_seed0"
  "conv1_ours_sgd_bias_zero_lr_current_seed0"
  "conv1_legacy_sgd_bias_zero_lr_current_seed0"
  "conv2_baseline_sgd_bias_zero_lr_current_seed0"
  "conv2_baseline_sgd_bias_zero_conv3x_stress_seed0"
  "conv2_baseline_sgd_bias_zero_ours_proposal_exact_seed0"
  "conv2_baseline_sgd_bias_zero_legacy_proposal_exact_seed0"
  "conv2_ours_sgd_bias_zero_lr_current_seed0"
  "conv2_legacy_sgd_bias_zero_lr_current_seed0"
  "conv3_baseline_sgd_bias_zero_lr_current_seed0"
  "conv3_baseline_sgd_bias_zero_conv3x_stress_seed0"
  "conv3_baseline_sgd_bias_zero_ours_proposal_exact_seed0"
  "conv3_baseline_sgd_bias_zero_legacy_proposal_exact_seed0"
  "conv3_ours_sgd_bias_zero_lr_current_seed0"
  "conv3_legacy_sgd_bias_zero_lr_current_seed0"
)

if [[ "${#CONFIGS[@]}" -ne 18 || "${#EXPECTED_ARMS[@]}" -ne 18 ]]; then
  echo "Expected exactly eighteen frozen LR-counterfactual configs and arms." >&2
  exit 3
fi
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ${#CONFIGS[@]} )); then
  echo "Array task ${SLURM_ARRAY_TASK_ID} is outside [0,17]." >&2
  exit 3
fi
if [[ ! -d "${LR_COUNTERFACTUAL_SOURCE_ROOT}" ]]; then
  echo "Missing frozen source directory: ${LR_COUNTERFACTUAL_SOURCE_ROOT}" >&2
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
if [[ "${actual_config_set_sha256}" != "${LR_COUNTERFACTUAL_CONFIG_SET_SHA256}" ]]; then
  echo "Config-set SHA-256 mismatch: expected ${LR_COUNTERFACTUAL_CONFIG_SET_SHA256}, got ${actual_config_set_sha256}." >&2
  exit 3
fi

LR_COUNTERFACTUAL_RUN_MODE="${LR_COUNTERFACTUAL_RUN_MODE:-production}"
RUNNER_MODE_ARGS=()
case "${LR_COUNTERFACTUAL_RUN_MODE}" in
  production)
    RUN_ROOT="${LR_COUNTERFACTUAL_RESULT_ROOT}/runs"
    SUMMARY_ROOT="${LR_COUNTERFACTUAL_RESULT_ROOT}/task_summaries/production"
    ;;
  smoke)
    RUN_ROOT="${LR_COUNTERFACTUAL_RESULT_ROOT}/remote_canary"
    SUMMARY_ROOT="${LR_COUNTERFACTUAL_RESULT_ROOT}/task_summaries/remote_canary"
    RUNNER_MODE_ARGS+=(--smoke)
    ;;
  *)
    echo "LR_COUNTERFACTUAL_RUN_MODE must be production or smoke; got ${LR_COUNTERFACTUAL_RUN_MODE}." >&2
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
cd "${LR_COUNTERFACTUAL_SOURCE_ROOT}"
SUMMARY_PATH="${SUMMARY_ROOT}/${SLURM_ARRAY_TASK_ID}.json"

python -m experiments.exact_run \
  "${CONFIGS[@]}" \
  --output-root "${RUN_ROOT}" \
  --device cuda \
  --study-id "${STUDY_ID}" \
  --evidence-class ordinary_mnist_lr_counterfactual \
  --target jean-zay \
  --summary-json "${SUMMARY_PATH}" \
  --gradient-trace-samples-per-epoch 5 \
  --checkpoint-every-epoch \
  --skip-terminal-official-test \
  "${RUNNER_MODE_ARGS[@]}"

SELECTED_CONFIG="${CONFIGS[SLURM_ARRAY_TASK_ID]}"
EXPECTED_CONFIG_SHA256="$(sha256sum "${SELECTED_CONFIG}" | awk '{print $1}')"

RUN_DIR="$(python - \
  "${SUMMARY_PATH}" \
  "${SELECTED_CONFIG}" \
  "${STUDY_MANIFEST}" \
  "${EXPECTED_CONFIG_SHA256}" \
  "${EXPECTED_ARMS[SLURM_ARRAY_TASK_ID]}" \
  "${STUDY_ID}" \
  "${EVIDENCE_CLASS}" \
  "${LR_COUNTERFACTUAL_RUN_MODE}" \
  "${LR_COUNTERFACTUAL_CONFIG_SET_SHA256}" <<'PY'
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

(
    summary_arg,
    selected_config_arg,
    study_manifest_arg,
    expected_config_sha256,
    expected_arm_id,
    expected_study_id,
    expected_evidence_class,
    run_mode,
    expected_config_set_sha256,
) = sys.argv[1:]


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


selected_config_path = Path(selected_config_arg)
selected_config = load_json(selected_config_path)
study_manifest = load_json(Path(study_manifest_arg))
configured_epochs = int(selected_config["lab"]["epochs"])
if configured_epochs <= 0:
    raise SystemExit(f"Configured epoch count must be positive: {configured_epochs}")
if run_mode == "production":
    expected_epochs = configured_epochs
    expected_trace_per_epoch = 5
elif run_mode == "smoke":
    expected_epochs = 1
    expected_trace_per_epoch = 1
else:
    raise SystemExit(f"Unexpected run mode: {run_mode!r}")
expected_trace_steps = expected_epochs * expected_trace_per_epoch
expected_checkpoint_count = expected_epochs + 1

parameter_order = selected_config.get("parameter_order")
if not isinstance(parameter_order, list) or not parameter_order:
    raise SystemExit(f"Invalid parameter order: {parameter_order!r}")
if len(parameter_order) != len(set(parameter_order)):
    raise SystemExit(f"Duplicate parameter names: {parameter_order!r}")
bias_names = [name for name in parameter_order if name.startswith("Bias_")]
if not bias_names:
    raise SystemExit(f"No Bias_* parameters in selected config: {parameter_order!r}")
rates = selected_config.get("learning_rates_by_parameter", {})
if set(rates) != set(parameter_order):
    raise SystemExit(f"Learning-rate inventory differs from parameter order: {rates!r}")
for name in bias_names:
    if float(rates[name]) != 0.0:
        raise SystemExit(f"Nonzero configured bias learning rate: {name}={rates[name]}")

reporting = selected_config.get("reporting", {})
if selected_config.get("arm_id") != expected_arm_id or reporting.get("arm_id") != expected_arm_id:
    raise SystemExit(f"Selected config arm mismatch: {selected_config.get('arm_id')!r}")
if selected_config.get("study_id") != expected_study_id or reporting.get("study_id") != expected_study_id:
    raise SystemExit(f"Selected config study mismatch: {selected_config.get('study_id')!r}")
if reporting.get("evidence_class") != expected_evidence_class:
    raise SystemExit(f"Selected config evidence-class mismatch: {reporting!r}")
mnist = selected_config.get("datasets", {}).get("mnist", {})
if mnist.get("factory") != "labs.datasets.MnistTrainValidationDataset":
    raise SystemExit(f"Unexpected dataset factory: {mnist!r}")
if mnist.get("params", {}).get("affine_config", "missing") is not None:
    raise SystemExit(f"Affine corruption was not disabled: {mnist!r}")
if selected_config.get("evaluation", {}).get("official_test", {}).get("policy") != "disabled":
    raise SystemExit("Selected config does not disable the official test.")

if study_manifest.get("study_id") != expected_study_id:
    raise SystemExit(f"Study-manifest ID mismatch: {study_manifest.get('study_id')!r}")
if study_manifest.get("evidence_class") != expected_evidence_class:
    raise SystemExit(f"Study-manifest evidence class mismatch: {study_manifest!r}")
if study_manifest.get("ordered_config_set_sha256") != expected_config_set_sha256:
    raise SystemExit(f"Study-manifest config-set digest mismatch: {study_manifest!r}")

rows = load_json(Path(summary_arg))
if len(rows) != 1 or rows[0].get("status") != "complete":
    raise SystemExit(f"Task summary is not semantically complete: {rows!r}")
row = rows[0]
expected_diagnostics = {
    "gradient_trace_samples_per_epoch": 5,
    "checkpoint_every_epoch": True,
    "skip_terminal_official_test": True,
}
if row.get("config_sha256") != expected_config_sha256:
    raise SystemExit(f"Task-summary config digest mismatch: {row!r}")
if int(row.get("configured_epochs", -1)) != configured_epochs or int(row.get("epochs", -1)) != expected_epochs:
    raise SystemExit(f"Task-summary epoch contract mismatch: {row!r}")
if bool(row.get("smoke")) != (run_mode == "smoke"):
    raise SystemExit(f"Task-summary smoke mode mismatch: {row!r}")
if row.get("diagnostics") != expected_diagnostics:
    raise SystemExit(f"Task-summary diagnostic contract mismatch: {row!r}")

run_dir = Path(row["output_dir"])
required = {
    "trace": run_dir / "gradient_trace.jsonl",
    "trace_metadata": run_dir / "gradient_trace_metadata.json",
    "checkpoint_index": run_dir / "epoch_checkpoint_index.jsonl",
    "manifest": run_dir / "manifest.json",
    "status": run_dir / "status.json",
    "metrics": run_dir / "metrics.json",
    "metrics_jsonl": run_dir / "metrics.jsonl",
    "result": run_dir / "result.json",
    "best_model": run_dir / "best_model.pt",
    "final_model": run_dir / "final_model.pt",
}
for name, path in required.items():
    if not path.is_file() or path.stat().st_size <= 0:
        raise SystemExit(f"Missing or empty {name}: {path}")

metadata = load_json(required["trace_metadata"])
if metadata.get("status") != "complete":
    raise SystemExit(f"Gradient trace metadata is incomplete: {metadata!r}")
if metadata.get("parameter_names") != parameter_order:
    raise SystemExit(f"Gradient trace parameter inventory mismatch: {metadata!r}")
if int(metadata.get("epoch_count", -1)) != expected_epochs:
    raise SystemExit(f"Gradient trace epoch count mismatch: {metadata!r}")
if int(metadata.get("samples_per_epoch", -1)) != 5:
    raise SystemExit(f"Gradient trace requested sample count mismatch: {metadata!r}")
if int(metadata.get("recorded_steps", -1)) != expected_trace_steps:
    raise SystemExit(f"Gradient trace step count mismatch: {metadata!r}")
if int(metadata.get("recorded_rows", -1)) != expected_trace_steps * len(parameter_order):
    raise SystemExit(f"Gradient trace row count mismatch: {metadata!r}")

sampled = metadata.get("sampled_source_batches", [])
sampled_counts = Counter(int(item["epoch"]) for item in sampled)
if sampled_counts != Counter({epoch: expected_trace_per_epoch for epoch in range(1, expected_epochs + 1)}):
    raise SystemExit(f"Gradient trace metadata lacks per-epoch coverage: {sampled_counts!r}")
metadata_pairs = [(int(item["epoch"]), int(item["batch"])) for item in sampled]
if len(metadata_pairs) != len(set(metadata_pairs)):
    raise SystemExit(f"Duplicate sampled source transitions: {metadata_pairs!r}")

trace_rows = [
    json.loads(line)
    for line in required["trace"].read_text(encoding="utf-8").splitlines()
    if line.strip()
]
if len(trace_rows) != expected_trace_steps * len(parameter_order):
    raise SystemExit(f"Unexpected gradient trace row count: {len(trace_rows)}")
transition_parameters = defaultdict(list)
for trace_row in trace_rows:
    key = (
        int(trace_row["epoch"]),
        int(trace_row["batch"]),
        int(trace_row["global_step"]),
    )
    name = trace_row["parameter_name"]
    transition_parameters[key].append(name)
    if float(trace_row["learning_rate"]) != float(rates[name]):
        raise SystemExit(f"Traced learning rate differs from config: {trace_row!r}")
for key, names in transition_parameters.items():
    if names != parameter_order:
        raise SystemExit(f"Trace parameter inventory mismatch at {key}: {names!r}")
trace_pairs = [(epoch, batch) for epoch, batch, _ in transition_parameters]
if Counter(trace_pairs) != Counter(metadata_pairs):
    raise SystemExit("Trace transitions differ from trace metadata.")
trace_counts = Counter(epoch for epoch, _ in trace_pairs)
if trace_counts != Counter({epoch: expected_trace_per_epoch for epoch in range(1, expected_epochs + 1)}):
    raise SystemExit(f"Gradient trace lacks per-epoch coverage: {trace_counts!r}")
if metadata.get("trace_sha256") != sha256_file(required["trace"]):
    raise SystemExit("Gradient trace SHA-256 mismatch.")
if int(metadata.get("trace_size_bytes", -1)) != required["trace"].stat().st_size:
    raise SystemExit("Gradient trace size mismatch.")

checkpoint_rows = [
    json.loads(line)
    for line in required["checkpoint_index"].read_text(encoding="utf-8").splitlines()
    if line.strip()
]
if len(checkpoint_rows) != expected_checkpoint_count:
    raise SystemExit(f"Unexpected checkpoint count: {len(checkpoint_rows)}")
if [int(item["epoch"]) for item in checkpoint_rows] != list(range(expected_checkpoint_count)):
    raise SystemExit(f"Unexpected checkpoint epochs: {checkpoint_rows!r}")


def verify_biases(path: Path) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    schema = payload.get("schema", [])
    states = payload.get("states", [])
    if len(schema) != len(states):
        raise SystemExit(f"Checkpoint schema/state mismatch: {path}")
    seen = []
    for spec, state in zip(schema, states):
        name = str(spec.get("name", "")).strip()
        if name.startswith("Bias_"):
            seen.append(name)
            if int(torch.count_nonzero(state).item()) != 0:
                raise SystemExit(f"Nonzero frozen bias in {path}: {name}")
    if seen != bias_names:
        raise SystemExit(f"Checkpoint bias inventory mismatch in {path}: expected {bias_names}, got {seen}")


for item in checkpoint_rows:
    for prefix in ("model", "optimizer"):
        relative = Path(item[f"{prefix}_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise SystemExit(f"Unsafe checkpoint path: {relative}")
        path = run_dir / relative
        if not path.is_file() or path.stat().st_size <= 0:
            raise SystemExit(f"Missing or empty checkpoint: {path}")
        if path.stat().st_size != int(item[f"{prefix}_size_bytes"]):
            raise SystemExit(f"Checkpoint size mismatch: {path}")
        if sha256_file(path) != item[f"{prefix}_sha256"]:
            raise SystemExit(f"Checkpoint SHA-256 mismatch: {path}")
    verify_biases(run_dir / item["model_path"])
verify_biases(required["best_model"])
verify_biases(required["final_model"])

manifest = load_json(required["manifest"])
status = load_json(required["status"])
metrics = load_json(required["metrics"])
result = load_json(required["result"])
for payload_name, payload in (("manifest", manifest), ("status", status), ("result", result)):
    if payload.get("study_id") != expected_study_id or payload.get("arm_id") != expected_arm_id:
        raise SystemExit(f"{payload_name} study/arm mismatch: {payload!r}")
if manifest.get("evidence_class") != expected_evidence_class:
    raise SystemExit(f"Manifest evidence-class mismatch: {manifest!r}")
configuration = manifest.get("configuration", {})
if configuration.get("sha256") != expected_config_sha256 or configuration.get("resolved") != selected_config:
    raise SystemExit("Manifest does not contain the exact selected config.")
if int(configuration.get("configured_epochs", -1)) != configured_epochs or int(configuration.get("epochs", -1)) != expected_epochs:
    raise SystemExit(f"Manifest epoch contract mismatch: {configuration!r}")
if configuration.get("diagnostic_overrides") != expected_diagnostics:
    raise SystemExit(f"Manifest diagnostic contract mismatch: {configuration!r}")
dataset = manifest.get("dataset", {})
if dataset.get("variant") != "ordinary" or dataset.get("factory") != "labs.datasets.MnistTrainValidationDataset":
    raise SystemExit(f"Expected ordinary MNIST: {dataset!r}")
if dataset.get("evaluation_split") != "validation" or dataset.get("official_test_read") is not False:
    raise SystemExit(f"Official-test/dataset split contract failed: {dataset!r}")
if dataset.get("params", {}).get("affine_config", "missing") is not None:
    raise SystemExit(f"Affine corruption was not disabled at runtime: {dataset!r}")
if status.get("state") != "complete":
    raise SystemExit(f"Run status is not complete: {status!r}")
completion = result.get("completion", {})
if completion.get("criteria_met") is not True or completion.get("diagnostic_overrides") != expected_diagnostics:
    raise SystemExit(f"Completion criteria were not met: {completion!r}")
if int(metrics.get("official_test_evaluations", -1)) != 0 or int(metrics.get("official_test_examples", -1)) != 0:
    raise SystemExit(f"Official test was evaluated: {metrics!r}")
for key in ("official_test_accuracy", "official_test_loss", "official_test_checkpoint"):
    if metrics.get(key, "missing") is not None:
        raise SystemExit(f"Official-test field {key} is populated: {metrics!r}")
if metrics.get("checkpoint_every_epoch") is not True or int(metrics.get("epoch_checkpoint_count", -1)) != expected_checkpoint_count:
    raise SystemExit(f"Metrics checkpoint contract mismatch: {metrics!r}")
if int(metrics.get("gradient_trace_samples_per_epoch", -1)) != 5:
    raise SystemExit(f"Metrics gradient-trace contract mismatch: {metrics!r}")

metric_rows = [
    json.loads(line)
    for line in required["metrics_jsonl"].read_text(encoding="utf-8").splitlines()
    if line.strip()
]
if [int(item.get("epoch", -1)) for item in metric_rows] != list(range(1, expected_epochs + 1)):
    raise SystemExit(f"Per-epoch metrics coverage mismatch: {metric_rows!r}")

print(run_dir)
PY
)"

python -m experiments.reporting validate-run "${RUN_DIR}"
