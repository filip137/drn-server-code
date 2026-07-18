from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_percentile_plotter_fails_when_nothing_was_plotted(tmp_path):
    script = REPO_ROOT / "labs" / "tools" / "plot_percentile_error_vs_iters.py"
    result = subprocess.run(
        [sys.executable, str(script), str(tmp_path), "--percentiles", "90"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "Expected at least one percentile error summary" in result.stderr
    assert f"Provided root: {tmp_path}" in result.stderr
