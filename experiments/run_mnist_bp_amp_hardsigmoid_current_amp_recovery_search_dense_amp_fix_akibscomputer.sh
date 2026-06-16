#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-/home/filip/miniconda3/envs/py312/bin/python}"
TRAIN="/home/filip/server_code/experiments/train_mnist_bp_amplification_sweep.py"

INIT_ROOT="${INIT_ROOT:-/home/filip/server_code/results/mnist_bp_amplification_sweep_hardsigmoid_voff15_legacy}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/filip/server_code/results/mnist_bp_amplification_sweep_hardsigmoid_voff15_current_amp_recovery_search_dense_amp_fix}"
LOG_DIR="${OUTPUT_ROOT}/logs"

EPOCHS="${EPOCHS:-10}"
LR_DECAY="${LR_DECAY:-0.99}"
SEEDS_STR="${SEEDS:-0}"
LRS_STR="${LRS:-0.0012 0.0020 0.0030 0.0040}"
ITERS_STR="${ITERS:-8 16}"
V_OFF="${V_OFF:-1.5}"
G_ON="${G_ON:-100.0}"
G_OFF="${G_OFF:-0.0}"

mkdir -p "${LOG_DIR}"

export KMP_DISABLE_SHM=1
export KMP_SHM_DISABLE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-mnist-amp-current-recovery-search}"

read -r -a seeds <<<"${SEEDS_STR}"
read -r -a lrs <<<"${LRS_STR}"
read -r -a iterations <<<"${ITERS_STR}"

run_names=(
  mnist_bp_amp_v1_c2
  mnist_bp_amp_v1_c4
)

lr_label() {
  local lr="$1"
  printf '%s' "${lr}" | sed 's/\./p/g; s/-/m/g'
}

echo "$(date) launching fixed-dynamics current-amplification recovery search from ${INIT_ROOT}" | tee -a "${LOG_DIR}/queue.log"
echo "$(date) output=${OUTPUT_ROOT} epochs=${EPOCHS} seeds=${SEEDS_STR} lrs=${LRS_STR} iterations=${ITERS_STR}" | tee -a "${LOG_DIR}/queue.log"

pids=()
status=0

for num_iterations in "${iterations[@]}"; do
  for lr in "${lrs[@]}"; do
    label="$(lr_label "${lr}")"
    lr_output_root="${OUTPUT_ROOT}/iter_${num_iterations}/lr_${label}"
    mkdir -p "${lr_output_root}"
    for run_name in "${run_names[@]}"; do
      for seed in "${seeds[@]}"; do
        log_path="${LOG_DIR}/iter_${num_iterations}_lr_${label}_${run_name}_seed_${seed}.log"
        echo "$(date) iter=${num_iterations} lr=${lr} run=${run_name} seed=${seed} -> ${log_path}" | tee -a "${LOG_DIR}/queue.log"
        "${PYTHON}" "${TRAIN}" \
          --output-root "${lr_output_root}" \
          --dataset-root /home/filip/server_code/model/resistive/data \
          --device cuda \
          --model-key mnist_bp_amp \
          --epochs "${EPOCHS}" \
          --batch-size 4 \
          --num-iterations "${num_iterations}" \
          --learning-rate "${lr}" "${lr}" "${lr}" \
          --beta 1.0 \
          --lr-decay "${LR_DECAY}" \
          --weight-min 0.0 \
          --weight-max 100.0 \
          --weight-init-mode kaiming_uniform \
          --weight-gains 1.0 1.0 \
          --input-gain 100.0 \
          --non-linearity hard_sigmoid \
          --hard-sigmoid-g-on "${G_ON}" \
          --hard-sigmoid-g-off "${G_OFF}" \
          --hard-sigmoid-v-min "-${V_OFF}" \
          --hard-sigmoid-v-max "${V_OFF}" \
          --normalize-mean 0.1307 \
          --normalize-std 0.3081 \
          --normalize-scale 0.3 \
          --init-checkpoint-template "${INIT_ROOT}/{run_name}/{seed_dir}/best_model.pt" \
          --no-download \
          --run-name "${run_name}" \
          --seeds "${seed}" \
          >"${log_path}" 2>&1 &
        pids+=("$!")
      done
    done
  done
done

for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

"${PYTHON}" - "${OUTPUT_ROOT}" >"${LOG_DIR}/collect_summary.log" 2>&1 <<'PY' || status=1
import csv
import json
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for metrics_path in sorted(root.glob("iter_*/lr_*/mnist_bp_amp_v1_c*/seed_*/metrics.json")):
    metrics = json.loads(metrics_path.read_text())
    parts = metrics_path.relative_to(root).parts
    iteration = int(parts[0].split("_", 1)[1])
    lr_label = parts[1].split("_", 1)[1]
    config = json.loads(Path(metrics["config_path"]).read_text())
    lr = config["optimizer"]["learning_rate"][0]
    rows.append({
        "iteration": iteration,
        "lr_label": lr_label,
        "lr": lr,
        "run_name": metrics["run_dir"].split("/")[-2],
        "seed": metrics["seed"],
        "voltage_amp": metrics["voltage_amp"],
        "current_amp": metrics["current_amp"],
        "best_test_accuracy": metrics.get("best_test_accuracy"),
        "final_test_accuracy": metrics.get("final_test_accuracy"),
        "best_epoch": metrics.get("best_epoch"),
        "final_train_loss": metrics.get("final_train_loss"),
        "final_test_loss": metrics.get("final_test_loss"),
        "checkpoint_path": metrics.get("checkpoint_path"),
        "weights_best_path": metrics.get("weights_best_path"),
        "weights_final_path": metrics.get("weights_final_path"),
    })

fieldnames = [
    "iteration",
    "lr_label",
    "lr",
    "run_name",
    "seed",
    "voltage_amp",
    "current_amp",
    "best_test_accuracy",
    "final_test_accuracy",
    "best_epoch",
    "final_train_loss",
    "final_test_loss",
    "checkpoint_path",
    "weights_best_path",
    "weights_final_path",
]
summary_path = root / "summary.csv"
with summary_path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

groups = {}
for row in rows:
    groups.setdefault((row["iteration"], row["lr"], row["run_name"]), []).append(row)

by_setting = []
for (iteration, lr, run_name), group in sorted(groups.items()):
    best = [float(r["best_test_accuracy"]) for r in group if r["best_test_accuracy"] is not None]
    final = [float(r["final_test_accuracy"]) for r in group if r["final_test_accuracy"] is not None]
    by_setting.append({
        "iteration": iteration,
        "lr": lr,
        "run_name": run_name,
        "num_seeds": len(group),
        "mean_best_test_accuracy": statistics.fmean(best) if best else "",
        "mean_final_test_accuracy": statistics.fmean(final) if final else "",
        "max_best_test_accuracy": max(best) if best else "",
    })

summary_by_setting_path = root / "summary_by_setting.csv"
with summary_by_setting_path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=[
        "iteration",
        "lr",
        "run_name",
        "num_seeds",
        "mean_best_test_accuracy",
        "mean_final_test_accuracy",
        "max_best_test_accuracy",
    ])
    writer.writeheader()
    writer.writerows(by_setting)

print(f"[summary] wrote {summary_path} rows={len(rows)}")
print(f"[summary] wrote {summary_by_setting_path} rows={len(by_setting)}")
for row in by_setting:
    print(row)
PY

if [ "${status}" -eq 0 ]; then
  echo "$(date) fixed-dynamics current-amplification recovery search finished successfully" | tee -a "${LOG_DIR}/queue.log"
else
  echo "$(date) fixed-dynamics current-amplification recovery search finished with errors" | tee -a "${LOG_DIR}/queue.log"
fi

exit "${status}"
