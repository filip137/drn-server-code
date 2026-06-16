#!/usr/bin/env bash
set -euo pipefail

PYTHON="/home/filip/miniconda3/envs/py312/bin/python"
TRAIN="/home/filip/server_code/experiments/train_mnist_bp_amplification_sweep.py"

OUTPUT_ROOT="/home/filip/server_code/results/mnist_bp_v1c1_bounded_bs4_lower_lr"
LOG_DIR="${OUTPUT_ROOT}/logs"

mkdir -p "${LOG_DIR}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="/tmp/matplotlib-mnist-v1c1-bounded-bs4-lower-lr"

echo "$(date) launching MNIST BP v1/c1 bounded_uniform batch-size-4 lower-LR controls" | tee -a "${LOG_DIR}/queue.log"

common_args=(
  --dataset-root /home/filip/datasets/mnist
  --device cuda
  --model-key mnist
  --epochs 10
  --batch-size 4
  --num-iterations 4
  --beta 1.0
  --lr-decay 1.0
  --weight-min 0.0
  --weight-max 0.00018
  --weight-init-mode bounded_uniform
  --weight-gains 1.0 1.0
  --input-gain 50.0
  --normalize-mean 0.13066
  --normalize-std 0.308108
  --normalize-scale 1.0
  --run-name mnist_bp_amp_v1_c1
  --seeds 0
  --no-download
)

lrs=(
  "0.00000002"
  "0.00000005"
  "0.00000010"
  "0.00000020"
)

pids=()
status=0

for lr in "${lrs[@]}"; do
  lr_label="$(printf '%s' "${lr}" | sed 's/0*$//; s/\.$//; s/\./p/g')"
  case_root="${OUTPUT_ROOT}/lr_${lr_label}"
  case_log="${LOG_DIR}/lr_${lr_label}.log"
  echo "$(date) lr=${lr} -> ${case_log}" | tee -a "${LOG_DIR}/queue.log"
  "${PYTHON}" "${TRAIN}" \
    --output-root "${case_root}" \
    --learning-rate "${lr}" "${lr}" "${lr}" \
    "${common_args[@]}" >"${case_log}" 2>&1 &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

for lr in "${lrs[@]}"; do
  lr_label="$(printf '%s' "${lr}" | sed 's/0*$//; s/\.$//; s/\./p/g')"
  "${PYTHON}" "${TRAIN}" \
    --output-root "${OUTPUT_ROOT}/lr_${lr_label}" \
    --seeds 0 \
    --run-name mnist_bp_amp_v1_c1 \
    --summary-only >"${LOG_DIR}/summary_lr_${lr_label}.log" 2>&1 || status=1
done

"${PYTHON}" - "${OUTPUT_ROOT}" "${lrs[@]}" >"${LOG_DIR}/summary_controls.log" 2>&1 <<'PY' || status=1
import csv
import json
import sys
from pathlib import Path

output_root = Path(sys.argv[1])
lrs = sys.argv[2:]
rows = []
for lr in lrs:
    lr_label = lr.rstrip("0").rstrip(".").replace(".", "p")
    run_dir = output_root / f"lr_{lr_label}" / "mnist_bp_amp_v1_c1" / "seed_0"
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"missing {metrics_path}")
    metrics = json.loads(metrics_path.read_text())
    rows.append(
        {
            "lr": lr,
            "run_name": "mnist_bp_amp_v1_c1",
            "seed": 0,
            "voltage_amp": 1.0,
            "current_amp": 1.0,
            "batch_size": 4,
            "weight_init_mode": "bounded_uniform",
            "weight_gains": "[1.0, 1.0]",
            "best_test_accuracy": metrics.get("best_test_accuracy"),
            "final_test_accuracy": metrics.get("final_test_accuracy"),
            "best_epoch": metrics.get("best_epoch"),
            "final_train_loss": metrics.get("final_train_loss"),
            "final_test_loss": metrics.get("final_test_loss"),
            "checkpoint_path": metrics.get("checkpoint_path"),
            "weights_best_path": metrics.get("weights_best_path"),
            "weights_final_path": metrics.get("weights_final_path"),
            "run_dir": str(run_dir),
        }
    )

summary_path = output_root / "summary.csv"
with summary_path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(f"[summary] wrote {summary_path}")
for row in rows:
    print(row)
PY

if [ "${status}" -eq 0 ]; then
  echo "$(date) MNIST BP v1/c1 bounded_uniform batch-size-4 lower-LR controls finished successfully" | tee -a "${LOG_DIR}/queue.log"
else
  echo "$(date) MNIST BP v1/c1 bounded_uniform batch-size-4 lower-LR controls finished with errors" | tee -a "${LOG_DIR}/queue.log"
fi

exit "${status}"
