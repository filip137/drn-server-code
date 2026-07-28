from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
PREFLIGHT = (
    REPO_ROOT
    / "skills"
    / "scheduled-run-preflight"
    / "scripts"
    / "preflight_scheduled_runner.py"
)
SKILL = (
    REPO_ROOT
    / "skills"
    / "scheduled-run-preflight"
    / "SKILL.md"
)


def _run_preflight(runner: Path, receipt: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(PREFLIGHT),
            "--runner",
            str(runner),
            "--cwd",
            str(REPO_ROOT),
            "--receipt",
            str(receipt),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_preflight_writes_receipt_only_after_success(tmp_path):
    marker = tmp_path / "actual-ran"
    runner = tmp_path / "runner.sh"
    runner.write_text(
        "#!/bin/bash\n"
        "set -u\n"
        'if [[ "${1:-}" == "--preflight" ]]; then\n'
        "  exit 0\n"
        "fi\n"
        f"touch {marker}\n"
    )
    receipt = tmp_path / "receipt.json"

    completed = _run_preflight(runner, receipt)

    assert completed.returncode == 0
    assert receipt.is_file()
    assert not marker.exists()
    payload = json.loads(receipt.read_text())
    assert payload["schema_version"] == "scheduled-run-preflight-receipt/v1"
    assert payload["status"] == "passed"
    assert payload["runner"] == str(runner)
    assert len(payload["runner_sha256"]) == 64
    assert payload["preflight_command"] == [
        "/bin/bash",
        "-euo",
        "pipefail",
        str(runner),
        "--preflight",
    ]


def test_preflight_failure_is_fail_closed(tmp_path):
    runner = tmp_path / "runner.sh"
    runner.write_text(
        "#!/bin/bash\n"
        'if [[ "${1:-}" == "--preflight" ]]; then\n'
        "  echo parser-failure >&2\n"
        "  exit 2\n"
        "fi\n"
    )
    receipt = tmp_path / "receipt.json"

    completed = _run_preflight(runner, receipt)

    assert completed.returncode != 0
    assert "Expected runner --preflight to exit with code 0" in completed.stderr
    assert "parser-failure" in completed.stderr
    assert not receipt.exists()


def test_shell_preflight_cannot_mask_failed_command_with_later_exit_zero(tmp_path):
    runner = tmp_path / "runner.sh"
    runner.write_text(
        "#!/bin/bash\n"
        'if [[ "${1:-}" == "--preflight" ]]; then\n'
        "  false\n"
        "  exit 0\n"
        "fi\n"
    )
    receipt = tmp_path / "receipt.json"

    completed = _run_preflight(runner, receipt)

    assert completed.returncode != 0
    assert "Expected runner --preflight to exit with code 0" in completed.stderr
    assert not receipt.exists()


def test_skill_requires_durable_launch_deadlines_and_failure_report():
    text = SKILL.read_text(encoding="utf-8")
    normalized = " ".join(text.split())
    assert "run exactly three focused local runner checks" in normalized
    assert "Broader tests are diagnostic escalation only" in text
    assert "Never run every available suite" in text
    assert "one shared environment bootstrap" in text
    assert "resolve Python only after module loading" in text
    assert "workstation-only locations" in text
    assert "durable launch-attempt clock" in text
    assert "1,200 seconds" in text
    assert "900 seconds" in text
    assert "does not arm the attempt" in normalized
    assert "not a simulation" in normalized
    assert "do not cancel it" in normalized
    assert "new user message" in normalized
    assert "observable progress at least every 60 seconds" in normalized
    assert "first-write-wins durable JSON failure report" in normalized
    assert "new immutable attempt" in normalized
    assert "machine-readable failure marker" in normalized
