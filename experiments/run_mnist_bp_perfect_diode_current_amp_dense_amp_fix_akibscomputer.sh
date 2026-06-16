#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-/home/filip/miniconda3/envs/py312/bin/python}"
TRAIN="/home/filip/server_code/experiments/train_mnist_bp_amplification_sweep.py"

OUTPUT_ROOT="${OUTPUT_ROOT:-/home/filip/server_code/results/mnist_bp_amplification_sweep_perfect_diode_current_amp_dense_amp_fix_10epoch}"
LOG_DIR="${OUTPUT_ROOT}/logs"

SEEDS_STR="${SEEDS:-0 1 2}"
EPOCHS="${EPOCHS:-10}"
NUM_ITERATIONS="${NUM_ITERATIONS:-4}"

mkdir -p "${LOG_DIR}"

export KMP_DISABLE_SHM=1
export KMP_SHM_DISABLE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-mnist-perfect-diode-current-amp}"

read -r -a seeds <<<"${SEEDS_STR}"
run_names=(
  mnist_bp_amp_v1_c2
  mnist_bp_amp_v1_c4
)

echo "$(date) launching perfect-diode current-amplification rerun" | tee -a "${LOG_DIR}/queue.log"
echo "$(date) output=${OUTPUT_ROOT} epochs=${EPOCHS} seeds=${SEEDS_STR} iterations=${NUM_ITERATIONS}" | tee -a "${LOG_DIR}/queue.log"

pids=()
status=0
for run_name in "${run_names[@]}"; do
  for seed in "${seeds[@]}"; do
    log_path="${LOG_DIR}/${run_name}_seed_${seed}.log"
    echo "$(date) run=${run_name} seed=${seed} -> ${log_path}" | tee -a "${LOG_DIR}/queue.log"
    "${PYTHON}" "${TRAIN}" \
      --output-root "${OUTPUT_ROOT}" \
      --dataset-root /home/filip/server_code/model/resistive/data \
      --device cuda \
      --model-key mnist_bp_amp \
      --epochs "${EPOCHS}" \
      --batch-size 4 \
      --num-iterations "${NUM_ITERATIONS}" \
      --learning-rate 0.006 0.006 0.006 \
      --beta 1.0 \
      --lr-decay 0.99 \
      --weight-min 0.0 \
      --weight-max 100.0 \
      --weight-init-mode kaiming_uniform \
      --weight-gains 1.0 1.0 \
      --input-gain 100.0 \
      --non-linearity perfect_diode \
      --normalize-mean 0.1307 \
      --normalize-std 0.3081 \
      --normalize-scale 0.3 \
      --no-download \
      --run-name "${run_name}" \
      --seeds "${seed}" \
      >"${log_path}" 2>&1 &
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
rows = list(csv.DictReader((output_root / "summary.csv").open()))
groups = defaultdict(list)
for row in rows:
    groups[(row["run_name"], row["voltage_amp"], row["current_amp"])].append(row)

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
for (run_name, voltage_amp, current_amp), group in sorted(groups.items()):
    best = [float(row["best_test_accuracy"]) for row in group if row["best_test_accuracy"]]
    final = [float(row["final_test_accuracy"]) for row in group if row["final_test_accuracy"]]
    train_loss = [float(row["final_train_loss"]) for row in group if row["final_train_loss"]]
    test_loss = [float(row["final_test_loss"]) for row in group if row["final_test_loss"]]
    out_rows.append({
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
    })

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
  echo "$(date) perfect-diode current-amplification rerun finished successfully" | tee -a "${LOG_DIR}/queue.log"
else
  echo "$(date) perfect-diode current-amplification rerun finished with errors" | tee -a "${LOG_DIR}/queue.log"
fi

exit "${status}"
