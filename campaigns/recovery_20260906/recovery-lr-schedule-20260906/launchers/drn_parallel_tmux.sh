#!/usr/bin/env bash
set -u
cd /home/filip/server_code/.codex/worktrees/recovery-lr-schedule-20260906
/home/filip/miniconda3/envs/py312/bin/python handoff_drn.py > launchers/drn_parallel.log 2>&1
run_exit=$?
printf "%s\n" "$run_exit" > launchers/drn_parallel.exit
exit "$run_exit"
