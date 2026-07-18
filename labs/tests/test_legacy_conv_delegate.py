from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_RUNNER = REPO_ROOT / "experiments" / "train_mnist_bp_conv_amplification_sweep.py"
DEPRECATION_MARKER = "MNIST_CONV_LEGACY_LAUNCHER_DEPRECATED"


def test_legacy_conv_scientific_flags_fail_closed():
    result = subprocess.run(
        [sys.executable, str(LEGACY_RUNNER), "--epochs", "10", "--num-iterations", "4"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    lines = result.stderr.splitlines()
    assert lines[0].startswith("Expected format:")
    assert "delegates only the canonical" in result.stderr
    assert "--num-iterations" in result.stderr


def _legacy_launchers() -> list[Path]:
    launchers: list[Path] = []
    experiments = REPO_ROOT / "experiments"
    for pattern in (
        "run_mnist_bp_conv*.sh",
        "run_mnist_bp_conv*.slurm",
        "submit_mnist_bp_conv*.sh",
        "submit_mnist_bp_conv*.slurm",
    ):
        launchers.extend(experiments.glob(pattern))
    return sorted(launchers)


def test_every_historical_launcher_fails_before_setup_allocation_or_work():
    launchers = _legacy_launchers()
    assert launchers, "expected retained historical Conv launchers"

    for path in launchers:
        lines = path.read_text().splitlines()
        assert lines[0].startswith("#!"), path
        if path.suffix == ".slurm":
            assert lines[1] == "#SBATCH --mnist-conv-legacy-launcher-is-deprecated", path
            assert lines[2] == f"# {DEPRECATION_MARKER}", path
            message_lines = lines[3:6]
            assert lines[6].strip() == "exit 2", path
        else:
            assert lines[1] == f"# {DEPRECATION_MARKER}", path
            message_lines = lines[2:5]
            assert lines[5].strip() == "exit 2", path
        assert "Expected format:" in message_lines[0], path
        assert "Provided deprecated MNIST Conv launcher invocation:" in message_lines[1], path
        assert "no setup, allocation, or work was performed" in message_lines[2], path


def test_every_historical_launcher_exits_with_migration_message():
    for path in _legacy_launchers():
        result = subprocess.run(
            ["bash", str(path)],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
            timeout=5,
        )
        assert result.returncode == 2
        assert "Deprecated MNIST Conv launcher" in result.stderr
        assert "python -m experiments.mnist_conv" in result.stderr
