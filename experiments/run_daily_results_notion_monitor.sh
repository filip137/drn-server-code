#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/filip/server_code}"
PYTHON_BIN="${PYTHON_BIN:-/home/filip/miniconda3/envs/py312/bin/python}"
CODEX_BIN="${CODEX_BIN:-/home/filip/.local/bin/codex}"
RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/results}"
REPORT_DIR="${REPORT_DIR:-${RESULTS_ROOT}/daily_result_reports}"
NOTION_PAGE_ID="${NOTION_PAGE_ID:-3795e8e9-c9f4-8103-a58c-e268951a58a2}"
SINCE_HOURS="${SINCE_HOURS:-24}"
TIMEZONE="${TIMEZONE:-Europe/Paris}"
RUN_HOUR="${RUN_HOUR:-7}"
RUN_MINUTE="${RUN_MINUTE:-0}"

mkdir -p "${REPORT_DIR}"

seconds_until_next_run() {
  TZ="${TIMEZONE}" "${PYTHON_BIN}" - "${RUN_HOUR}" "${RUN_MINUTE}" "${TIMEZONE}" <<'PY'
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

hour = int(sys.argv[1])
minute = int(sys.argv[2])
tz = ZoneInfo(sys.argv[3])
now = datetime.now(tz)
target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
if target <= now:
    target += timedelta(days=1)
print(max(1, int((target - now).total_seconds())))
PY
}

run_once() {
  local stamp
  local report_md
  local report_csv
  local prompt_file
  local codex_log
  local codex_last

  stamp="$(TZ="${TIMEZONE}" date +%Y%m%d_%H%M%S)"
  report_md="${REPORT_DIR}/recent_results_${stamp}.md"
  report_csv="${REPORT_DIR}/recent_results_${stamp}.csv"
  prompt_file="${REPORT_DIR}/notion_prompt_${stamp}.md"
  codex_log="${REPORT_DIR}/codex_notion_${stamp}.log"
  codex_last="${REPORT_DIR}/codex_notion_${stamp}.txt"

  "${PYTHON_BIN}" "${REPO_ROOT}/experiments/summarize_recent_results.py" \
    --results-root "${RESULTS_ROOT}" \
    --since-hours "${SINCE_HOURS}" \
    --timezone "${TIMEZONE}" \
    --report-dir "${REPORT_DIR}" \
    --markdown-output "${report_md}" \
    --csv-output "${report_csv}"

  cat > "${prompt_file}" <<PROMPT
You are the unattended daily results monitor for the bidirectional amplification paper.

Append the contents of this Markdown report to the end of the existing Notion page:
${NOTION_PAGE_ID}

Report file:
${report_md}

Requirements:
- Use the Notion connector; do not use a browser.
- Insert the report at the end of the page without replacing existing content.
- Keep the report content concise and preserve the run paths exactly.
- If the report says no completed metrics were modified, still append that short status.
- Reply with the Notion page URL and the local report path.
PROMPT

  "${CODEX_BIN}" exec \
    -C "${REPO_ROOT}" \
    --sandbox workspace-write \
    --output-last-message "${codex_last}" \
    - < "${prompt_file}" > "${codex_log}" 2>&1
}

usage() {
  cat <<USAGE
Usage: $0 [--once|--loop]

Environment overrides:
  REPO_ROOT=${REPO_ROOT}
  PYTHON_BIN=${PYTHON_BIN}
  CODEX_BIN=${CODEX_BIN}
  RESULTS_ROOT=${RESULTS_ROOT}
  REPORT_DIR=${REPORT_DIR}
  NOTION_PAGE_ID=${NOTION_PAGE_ID}
  SINCE_HOURS=${SINCE_HOURS}
  TIMEZONE=${TIMEZONE}
  RUN_HOUR=${RUN_HOUR}
  RUN_MINUTE=${RUN_MINUTE}
USAGE
}

mode="${1:---loop}"
case "${mode}" in
  --once)
    run_once
    ;;
  --loop)
    while true; do
      sleep_seconds="$(seconds_until_next_run)"
      echo "[$(TZ="${TIMEZONE}" date --iso-8601=seconds)] sleeping ${sleep_seconds}s until next daily results monitor run"
      sleep "${sleep_seconds}"
      echo "[$(TZ="${TIMEZONE}" date --iso-8601=seconds)] running daily results monitor"
      if ! run_once; then
        echo "[$(TZ="${TIMEZONE}" date --iso-8601=seconds)] monitor run failed; see ${REPORT_DIR}/codex_notion_*.log" >&2
      fi
      sleep 60
    done
    ;;
  -h|--help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
