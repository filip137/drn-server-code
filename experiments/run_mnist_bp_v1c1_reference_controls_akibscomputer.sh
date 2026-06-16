#!/usr/bin/env bash
set -euo pipefail

PYTHON="/home/filip/miniconda3/envs/py312/bin/python"
TRAIN="/home/filip/server_code/experiments/train_mnist_bp_amplification_sweep.py"

OUTPUT_ROOT="/home/filip/server_code/results/mnist_bp_v1c1_reference_controls"
LOG_DIR="${OUTPUT_ROOT}/logs"

mkdir -p "${LOG_DIR}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="/tmp/matplotlib-mnist-v1c1-reference-controls"

echo "$(date) launching MNIST BP v1/c1 reference controls" | tee -a "${LOG_DIR}/queue.log"

common_args=(
  --dataset-root /home/filip/datasets/mnist
  --device cuda
  --model-key mnist
  --epochs 10
  --batch-size 256
  --num-iterations 4
  --learning-rate 0.00000125 0.00000125 0.00000125
  --beta 1.0
  --lr-decay 1.0
  --weight-min 0.0
  --weight-max 0.00018
  --input-gain 50.0
  --normalize-mean 0.13066
  --normalize-std 0.308108
  --normalize-scale 1.0
  --run-name mnist_bp_amp_v1_c1
  --no-download
)

pids=()
status=0

bounded_root="${OUTPUT_ROOT}/bounded_uniform_wg1_seed0"
bounded_log="${LOG_DIR}/bounded_uniform_wg1_seed0.log"
echo "$(date) bounded_uniform_wg1 seed=0 -> ${bounded_log}" | tee -a "${LOG_DIR}/queue.log"
"${PYTHON}" "${TRAIN}" \
  --output-root "${bounded_root}" \
  --seeds 0 \
  --weight-init-mode bounded_uniform \
  --weight-gains 1.0 1.0 \
  "${common_args[@]}" >"${bounded_log}" 2>&1 &
pids+=("$!")

rescaled_root="${OUTPUT_ROOT}/kaiming_uniform_rescaled_seed1"
rescaled_log="${LOG_DIR}/kaiming_uniform_rescaled_seed1.log"
echo "$(date) kaiming_uniform_rescaled seed=1 -> ${rescaled_log}" | tee -a "${LOG_DIR}/queue.log"
"${PYTHON}" "${TRAIN}" \
  --output-root "${rescaled_root}" \
  --seeds 1 \
  --weight-init-mode kaiming_uniform \
  --weight-gains 0.007127636354360399 0.0018000000000000002 \
  "${common_args[@]}" >"${rescaled_log}" 2>&1 &
pids+=("$!")

for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

"${PYTHON}" "${TRAIN}" --output-root "${bounded_root}" --seeds 0 --summary-only >"${LOG_DIR}/summary_bounded_uniform_wg1_seed0.log" 2>&1 || status=1
"${PYTHON}" "${TRAIN}" --output-root "${rescaled_root}" --seeds 1 --summary-only >"${LOG_DIR}/summary_kaiming_uniform_rescaled_seed1.log" 2>&1 || status=1

"${PYTHON}" - <<'PY' "${OUTPUT_ROOT}" >"${LOG_DIR}/summary_controls.log" 2>&1 || status=1
import csv
import json
import sys
from pathlib import Path

output_root = Path(sys.argv[1])
cases = [
    ("bounded_uniform_wg1_seed0", "bounded_uniform", "[1.0, 1.0]", 0),
    (
        "kaiming_uniform_rescaled_seed1",
        "kaiming_uniform",
        "[0.007127636354360399, 0.0018000000000000002]",
        1,
    ),
]
rows = []
for case, init_mode, gains, seed in cases:
    run_dir = output_root / case / "mnist_bp_amp_v1_c1" / f"seed_{seed}"
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"missing {metrics_path}")
    metrics = json.loads(metrics_path.read_text())
    rows.append(
        {
            "case": case,
            "seed": seed,
            "weight_init_mode": init_mode,
            "weight_gains": gains,
            "best_test_accuracy": metrics.get("best_test_accuracy"),
            "final_test_accuracy": metrics.get("final_test_accuracy"),
            "best_epoch": metrics.get("best_epoch"),
            "final_train_loss": metrics.get("final_train_loss"),
            "final_test_loss": metrics.get("final_test_loss"),
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
  echo "$(date) MNIST BP v1/c1 reference controls finished successfully" | tee -a "${LOG_DIR}/queue.log"
else
  echo "$(date) MNIST BP v1/c1 reference controls finished with errors" | tee -a "${LOG_DIR}/queue.log"
fi

exit "${status}"
