#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 <budget_gpu_hours> <poll_seconds> <job_id> [<job_id> ...]" >&2
  exit 2
fi

BUDGET_GPU_HOURS="$1"
POLL_SECONDS="$2"
shift 2
JOB_IDS=("$@")
JOB_CSV="$(IFS=,; echo "${JOB_IDS[*]}")"

LOG_DIR="${LOG_DIR:-${SCRATCH:-/tmp}/server_code/results/jeanzay_gpu_budget_monitor}"
mkdir -p "${LOG_DIR}"
LOG_PATH="${LOG_PATH:-${LOG_DIR}/gpu_budget_${JOB_CSV//,/}_$(date +%Y%m%dT%H%M%S).log}"

echo "[budget-monitor] start=$(date --iso-8601=seconds)" | tee -a "${LOG_PATH}"
echo "[budget-monitor] budget_gpu_hours=${BUDGET_GPU_HOURS} poll_seconds=${POLL_SECONDS} jobs=${JOB_CSV}" | tee -a "${LOG_PATH}"

gpu_hours_for_jobs() {
  sacct -X -j "${JOB_CSV}" --format=JobIDRaw,State,ElapsedRaw,AllocTRES -P -n 2>/dev/null \
    | python -c '
import re
import sys

total = 0.0
seen = set()
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    parts = line.split("|")
    if len(parts) < 4:
        continue
    jobid, state, elapsed_raw, tres = parts[:4]
    if jobid in seen:
        continue
    seen.add(jobid)
    match = re.search(r"gres/gpu=(\d+)", tres)
    if not match:
        continue
    try:
        elapsed = float(elapsed_raw or 0.0)
    except ValueError:
        elapsed = 0.0
    total += elapsed * int(match.group(1)) / 3600.0
print(f"{total:.6f}")
'
}

while true; do
  used="$(gpu_hours_for_jobs)"
  now="$(date --iso-8601=seconds)"
  echo "[budget-monitor] ${now} used_gpu_hours=${used}" | tee -a "${LOG_PATH}"
  if python - "${used}" "${BUDGET_GPU_HOURS}" <<'PY'
import sys
used = float(sys.argv[1])
budget = float(sys.argv[2])
raise SystemExit(0 if used >= budget else 1)
PY
  then
    echo "[budget-monitor] budget exceeded; cancelling ${JOB_CSV}" | tee -a "${LOG_PATH}"
    scancel "${JOB_IDS[@]}" || true
    exit 0
  fi
  if ! squeue -h -j "${JOB_CSV}" >/tmp/budget_monitor_squeue.$$ 2>/dev/null || [[ ! -s /tmp/budget_monitor_squeue.$$ ]]; then
    rm -f /tmp/budget_monitor_squeue.$$
    echo "[budget-monitor] no target jobs remain in queue; exiting" | tee -a "${LOG_PATH}"
    exit 0
  fi
  rm -f /tmp/budget_monitor_squeue.$$
  sleep "${POLL_SECONDS}"
done
