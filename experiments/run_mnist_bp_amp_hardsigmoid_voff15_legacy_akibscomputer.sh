#!/usr/bin/env bash
set -euo pipefail

PYTHON="/home/filip/miniconda3/envs/py312/bin/python"
TRAIN="/home/filip/server_code/experiments/train_mnist_bp_amplification_sweep.py"

OUTPUT_ROOT="/home/filip/server_code/results/mnist_bp_amplification_sweep_hardsigmoid_voff15_legacy"
LOG_DIR="${OUTPUT_ROOT}/logs"

LR="0.006"
V_OFF="1.5"
G_ON="100.0"
G_OFF="0.0"

mkdir -p "${LOG_DIR}"

export KMP_DISABLE_SHM=1
export KMP_SHM_DISABLE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="/tmp/matplotlib-mnist-amp-hardsigmoid-voff15"

echo "$(date) launching MNIST BP amplification sweep with hard_sigmoid v_off=${V_OFF}" | tee -a "${LOG_DIR}/queue.log"

run_names=(
  mnist_bp_amp_v1_c1
  mnist_bp_amp_v2_c1
  mnist_bp_amp_v4_c1
  mnist_bp_amp_v1_c2
  mnist_bp_amp_v1_c4
)

seeds=(0 1 2)

common_args=(
  --output-root "${OUTPUT_ROOT}"
  --dataset-root /home/filip/server_code/model/resistive/data
  --device cuda
  --model-key mnist_bp_amp
  --epochs 10
  --batch-size 4
  --num-iterations 4
  --learning-rate "${LR}" "${LR}" "${LR}"
  --beta 1.0
  --lr-decay 0.99
  --weight-min 0.0
  --weight-max 100.0
  --weight-init-mode kaiming_uniform
  --weight-gains 1.0 1.0
  --input-gain 100.0
  --non-linearity hard_sigmoid
  --hard-sigmoid-g-on "${G_ON}"
  --hard-sigmoid-g-off "${G_OFF}"
  --hard-sigmoid-v-min "-${V_OFF}"
  --hard-sigmoid-v-max "${V_OFF}"
  --normalize-mean 0.1307
  --normalize-std 0.3081
  --normalize-scale 0.3
  --no-download
)

pids=()
status=0

for run_name in "${run_names[@]}"; do
  for seed in "${seeds[@]}"; do
    log_path="${LOG_DIR}/${run_name}_seed_${seed}.log"
    echo "$(date) run=${run_name} seed=${seed} lr=${LR} hard_sigmoid_g_on=${G_ON} hard_sigmoid_g_off=${G_OFF} v_off=${V_OFF} -> ${log_path}" | tee -a "${LOG_DIR}/queue.log"
    "${PYTHON}" "${TRAIN}" \
      "${common_args[@]}" \
      --run-name "${run_name}" \
      --seeds "${seed}" >"${log_path}" 2>&1 &
    pids+=("$!")
  done
done

for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

"${PYTHON}" "${TRAIN}" \
  --output-root "${OUTPUT_ROOT}" \
  --seeds "${seeds[@]}" \
  --summary-only >"${LOG_DIR}/summary_runner.log" 2>&1 || status=1

"${PYTHON}" - "${OUTPUT_ROOT}" >"${LOG_DIR}/summary_by_amp.log" 2>&1 <<'PY' || status=1
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

output_root = Path(sys.argv[1])
summary_path = output_root / "summary.csv"
rows = list(csv.DictReader(summary_path.open()))
if not rows:
    raise SystemExit(f"No rows found in {summary_path}")

groups = defaultdict(list)
for row in rows:
    key = (row["run_name"], row["voltage_amp"], row["current_amp"])
    groups[key].append(row)

fieldnames = [
    "run_name",
    "voltage_amp",
    "current_amp",
    "num_seeds",
    "mean_best_test_accuracy",
    "std_best_test_accuracy",
    "mean_final_test_accuracy",
    "std_final_test_accuracy",
    "mean_final_train_loss",
    "mean_final_test_loss",
]

def mean(values):
    return statistics.fmean(values) if values else ""

def stdev(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0

out_rows = []
for key, group in sorted(groups.items()):
    run_name, voltage_amp, current_amp = key
    best = [float(row["best_test_accuracy"]) for row in group if row["best_test_accuracy"]]
    final = [float(row["final_test_accuracy"]) for row in group if row["final_test_accuracy"]]
    train_loss = [float(row["final_train_loss"]) for row in group if row["final_train_loss"]]
    test_loss = [float(row["final_test_loss"]) for row in group if row["final_test_loss"]]
    out_rows.append(
        {
            "run_name": run_name,
            "voltage_amp": voltage_amp,
            "current_amp": current_amp,
            "num_seeds": len(group),
            "mean_best_test_accuracy": mean(best),
            "std_best_test_accuracy": stdev(best),
            "mean_final_test_accuracy": mean(final),
            "std_final_test_accuracy": stdev(final),
            "mean_final_train_loss": mean(train_loss),
            "mean_final_test_loss": mean(test_loss),
        }
    )

out_path = output_root / "summary_by_amp.csv"
with out_path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(out_rows)

print(f"[summary] wrote {out_path}")
for row in out_rows:
    print(row)
PY

if [ "${status}" -eq 0 ]; then
  echo "$(date) MNIST BP hard_sigmoid v_off=${V_OFF} amplification sweep finished successfully" | tee -a "${LOG_DIR}/queue.log"
else
  echo "$(date) MNIST BP hard_sigmoid v_off=${V_OFF} amplification sweep finished with errors" | tee -a "${LOG_DIR}/queue.log"
fi

exit "${status}"
