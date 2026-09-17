#!/usr/bin/env bash
set -u
cd /home/filip/server_code/.codex/worktrees/recovery-lr-schedule-20260906
/home/filip/miniconda3/envs/py312/bin/python confirm.py drn --mnist-root /home/filip/datasets/mnist > launchers/drn_confirmation.log 2>&1
run_exit=$?
printf "%s\n" "$run_exit" > launchers/drn_confirmation.exit
exit "$run_exit"
