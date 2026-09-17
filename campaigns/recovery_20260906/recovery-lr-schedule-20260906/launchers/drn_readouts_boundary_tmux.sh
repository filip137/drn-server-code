#!/usr/bin/env bash
set -u
cd /home/filip/server_code/.codex/worktrees/recovery-lr-schedule-20260906
/home/filip/miniconda3/envs/py312/bin/python finish_readouts.py drn --plan plan.with_lr_boundary.json --mnist-root /home/filip/datasets/mnist > launchers/drn_readouts_boundary.log 2>&1
run_exit=$?
printf "%s\n" "$run_exit" > launchers/drn_readouts_boundary.exit
exit "$run_exit"
