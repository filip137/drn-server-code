#!/usr/bin/env bash
set -euo pipefail

PYTHON="/home/filip/miniconda3/envs/py312/bin/python"
TRAIN="/home/filip/server_code/experiments/train_mnist_bp_amplification_sweep.py"

OUTPUT_ROOT="/home/filip/server_code/results/mnist_bp_amp_bounded_bs4_best_lr"
LOG_DIR="${OUTPUT_ROOT}/logs"
LR="0.00000010"

mkdir -p "${LOG_DIR}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="/tmp/matplotlib-mnist-amp-bounded-bs4-best-lr"

echo "$(date) launching MNIST BP amplification sweep with bounded_uniform batch-size-4 best LR" | tee -a "${LOG_DIR}/queue.log"

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
  --dataset-root /home/filip/datasets/mnist
  --device cuda
  --model-key mnist
  --epochs 10
  --batch-size 4
  --num-iterations 4
  --learning-rate "${LR}" "${LR}" "${LR}"
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
  --no-download
)

pids=()
status=0

for run_name in "${run_names[@]}"; do
  for seed in "${seeds[@]}"; do
    log_path="${LOG_DIR}/${run_name}_seed_${seed}.log"
    echo "$(date) lr=${LR} run=${run_name} seed=${seed} -> ${log_path}" | tee -a "${LOG_DIR}/queue.log"
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

"${PYTHON}" - "${OUTPUT_ROOT}" "${LR}" >"${LOG_DIR}/summary_all.log" 2>&1 <<'PY' || status=1
import csv
import json
import sys
from pathlib import Path

output_root = Path(sys.argv[1])
lr = sys.argv[2]
run_specs = [
    ("mnist_bp_amp_v1_c1", 1.0, 1.0),
    ("mnist_bp_amp_v2_c1", 2.0, 1.0),
    ("mnist_bp_amp_v4_c1", 4.0, 1.0),
    ("mnist_bp_amp_v1_c2", 1.0, 2.0),
    ("mnist_bp_amp_v1_c4", 1.0, 4.0),
]
seeds = [0, 1, 2]
rows = []
for run_name, voltage_amp, current_amp in run_specs:
    for seed in seeds:
        run_dir = output_root / run_name / f"seed_{seed}"
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError(f"missing {metrics_path}")
        metrics = json.loads(metrics_path.read_text())
        rows.append(
            {
                "lr": lr,
                "run_name": run_name,
                "seed": seed,
                "voltage_amp": voltage_amp,
                "current_amp": current_amp,
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
  echo "$(date) MNIST BP amplification bounded_uniform batch-size-4 best-LR sweep finished successfully" | tee -a "${LOG_DIR}/queue.log"
else
  echo "$(date) MNIST BP amplification bounded_uniform batch-size-4 best-LR sweep finished with errors" | tee -a "${LOG_DIR}/queue.log"
fi

exit "${status}"
