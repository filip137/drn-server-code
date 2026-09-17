#!/usr/bin/env bash
set -u
cd /home/filip/server_code/.codex/worktrees/recovery-lr-schedule-20260906
/home/filip/miniconda3/envs/py312/bin/python run_lr_boundary.py > launchers/drn_lr_boundary.log 2>&1
run_exit=$?
printf "%s\n" "$run_exit" > launchers/drn_lr_boundary.exit
exit "$run_exit"
