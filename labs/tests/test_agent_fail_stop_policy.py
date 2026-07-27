from __future__ import annotations

import os
from pathlib import Path
import subprocess
import time


REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT_POLICY = REPO_ROOT / "AGENTS.md"
EXPERIMENT_POLICY = REPO_ROOT / "experiments" / "AGENTS.md"
RUN_SKILL = REPO_ROOT / "skills" / "run-experiment-pipeline" / "SKILL.md"
SCHEDULED_SKILL = (
    REPO_ROOT / "skills" / "scheduled-run-preflight" / "SKILL.md"
)
HOURLY_REFRESH = (
    REPO_ROOT
    / "skills"
    / "run-experiment-pipeline"
    / "scripts"
    / "run_hourly_current_experiments_refresh.sh"
)
STAGED_JEANZAY_VALIDATION = (
    REPO_ROOT
    / "experiments"
    / "validate_mnist_conv_perfectdiode_successor_staged_jeanzay.sh"
)
SUCCESSOR_SUPERVISOR = (
    REPO_ROOT
    / "experiments"
    / "supervise_mnist_conv_perfectdiode_successor_confirmation_jeanzay.py"
)


def test_agent_policy_scopes_terminal_stop_to_armed_long_runs() -> None:
    root = ROOT_POLICY.read_text(encoding="utf-8")
    nested = EXPERIMENT_POLICY.read_text(encoding="utf-8")
    run_skill = RUN_SKILL.read_text(encoding="utf-8")
    scheduled = SCHEDULED_SKILL.read_text(encoding="utf-8")
    normalized_root = " ".join(root.split())
    normalized_nested = " ".join(nested.split())
    normalized_run_skill = " ".join(run_skill.split())
    normalized_scheduled = " ".join(scheduled.split())

    for required in (
        "only to an **armed long-run simulation attempt**",
        "LONG-RUN ATTEMPT ARMED",
        "first such failure is terminal for that long-run attempt",
        "Only a new user message sent after the long-run failure report",
        "new immutable attempt ID and receipt paths",
        "first-write-wins failure report",
        "exit nonzero",
        "hard deadline",
    ):
        assert required in normalized_root
    for required in (
        "only after a long-run attempt is explicitly armed",
        "Ordinary implementation work, tests, environment setup",
        "Catch-and-continue and automatic corrective retries are prohibited",
        "Default CLI behavior must be one bounded reconciliation pass",
        "production scheduler side effect",
    ):
        assert required in normalized_nested
    for text in (normalized_run_skill, normalized_scheduled):
        assert "armed" in text
        assert "long-run" in text or "long-running" in text
    assert "Short preflight and smoke execution completed before `start`" in (
        scheduled
    )
    assert "Correct the root cause and repeat the full procedure" not in (
        scheduled
    )
    assert "entire user-requested experiment workflow" not in root
    assert "Apply the root Fail-Stop-Report policy to every" not in nested


def test_short_development_failures_remain_recoverable() -> None:
    root = ROOT_POLICY.read_text(encoding="utf-8")
    normalized = " ".join(root.split())
    for recoverable in (
        "Planning, code edits, unit/integration tests, linting",
        "dependency and environment setup",
        "plan-only/dry-run commands",
        "short interactive probes",
        "bounded developer smoke tests",
        "may be diagnosed, fixed, and retested",
        "Offline collection and analysis",
        "diagnose the cause before rerunning",
        "reasonable diagnosis-backed attempts",
    ):
        assert recoverable in normalized
    assert "must never be represented as passed long-run launch gates" in (
        normalized
    )
    assert "independent scientific outcome" in root


def test_non_simulation_hourly_tracker_loop_remains_available() -> None:
    script = HOURLY_REFRESH.read_text(encoding="utf-8")
    run_skill = RUN_SKILL.read_text(encoding="utf-8")
    assert 'case "${1:---loop}" in' in script
    assert "--loop)" in script
    assert "while true" in script
    assert "CURRENT_EXPERIMENTS_REFRESH_INTERVAL_SECONDS" in script
    assert "Stopping without retry" not in script
    assert "conservative hourly fallback" in run_skill


def test_long_run_supervisor_requires_explicit_arm_marker() -> None:
    source = SUCCESSOR_SUPERVISOR.read_text(encoding="utf-8")
    assert '"--arm-long-run"' in source
    assert "if not args.arm_long_run" in source
    assert "LONG-RUN ATTEMPT ARMED" in source


def test_staged_jeanzay_validation_is_one_bounded_canonical_gate() -> None:
    assert STAGED_JEANZAY_VALIDATION.stat().st_mode & 0o111
    completed = subprocess.run(
        ["bash", "-n", str(STAGED_JEANZAY_VALIDATION)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    missing_arguments = subprocess.run(
        [str(STAGED_JEANZAY_VALIDATION)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert missing_arguments.returncode == 2
    assert "Expected --repo-root to be supplied" in missing_arguments.stderr
    script = STAGED_JEANZAY_VALIDATION.read_text(encoding="utf-8")
    assert "set -euo pipefail" in script
    assert 'timeout --kill-after=30s "${outer_timeout_seconds}s"' in script
    assert 'timeout --kill-after=30s "${remaining}s"' in script
    assert "pytorch-gpu/py3/2.5.0" in script
    assert (
        "/lustre/fshomisc/sup/hpe/pub/miniforge/24.9.0/envs/"
        "pytorch-gpu-2.5.0+py3.12.7/bin/python"
    ) in script
    assert "validate_experiment_plan.py" in script
    assert "validate_current_experiments.py" in script
    assert "official_canary_verifier_path_template" in script
    assert "--environment-contract" in script
    assert "test_mnist_conv_perfectdiode_successor_runtime.py" in script
    assert "test_mnist_conv_perfectdiode_successor_jeanzay.py" in script
    assert ".codex" not in script
    assert "sbatch" not in script
    assert "pytest -k" not in script
    assert "while true" not in script


def test_staged_jeanzay_bootstrap_obeys_whole_script_deadline(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_idrenv = fake_bin / "idrenv"
    fake_idrenv.write_text(
        "#!/bin/sh\nsleep 60\n",
        encoding="utf-8",
    )
    fake_idrenv.chmod(0o700)
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    approved_plan = tmp_path / "plan.md"
    approved_plan.write_text("plan\n", encoding="utf-8")
    environment_contract = tmp_path / "environment.json"
    environment_contract.write_text("{}\n", encoding="utf-8")

    started = time.monotonic()
    completed = subprocess.run(
        [
            "bash",
            str(STAGED_JEANZAY_VALIDATION),
            "--repo-root",
            str(repo_root),
            "--approved-plan",
            str(approved_plan),
            "--environment-contract",
            str(environment_contract),
            "--timeout-seconds",
            "1",
        ],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "PATH": f"{fake_bin}:/usr/bin:/bin",
        },
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    elapsed = time.monotonic() - started

    assert completed.returncode == 124
    assert elapsed < 5
    assert "whole-script hard deadline reached" in completed.stderr
