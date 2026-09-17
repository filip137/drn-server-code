#!/usr/bin/env bash
set -u
cd /home/filip/server_code/.codex/worktrees/recovery-lr-schedule-20260906
/home/filip/miniconda3/envs/py312/bin/python confirm.py drn --workers 3 --plan plan.with_lr_boundary.json --additional-screen-status drn_lr_boundary/status.json --mnist-root /home/filip/datasets/mnist > launchers/drn_confirmation_boundary.log 2>&1
run_exit=$?
printf "%s\n" "$run_exit" > launchers/drn_confirmation_boundary.exit
exit "$run_exit"
