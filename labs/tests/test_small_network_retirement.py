from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SHIM = REPO_ROOT / "labs" / "small_network.py"
CORE_SHIM = REPO_ROOT / "labs" / "small_network_core.py"


def test_small_network_command_is_an_informative_error_shim(tmp_path):
    config = tmp_path / "legacy.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SHIM),
            "--mode",
            "digits_validate",
            "--config",
            str(config),
            "--weights",
            "model.pt",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    lines = result.stderr.splitlines()
    assert lines[0].startswith("Expected format:")
    assert "archive/small-network-v1" in result.stderr
    assert "python -m experiments.mnist_conv run" in result.stderr
    assert "mode='digits_validate'" in result.stderr
    assert f"config='{config}'" in result.stderr


def test_small_network_core_import_fails_with_migration_message():
    result = subprocess.run(
        [sys.executable, "-c", "import labs.small_network_core"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "archive/small-network-v1" in result.stderr
    assert "experiments.mnist_conv" in result.stderr


def test_active_code_does_not_import_or_execute_small_network():
    excluded = {SHIM.resolve(), CORE_SHIM.resolve(), Path(__file__).resolve()}
    import_violations: list[str] = []
    execution_violations: list[str] = []

    for root_name in ("experiments", "labs", "model", "plotting_functions", "training"):
        for path in (REPO_ROOT / root_name).rglob("*.py"):
            if path.resolve() in excluded:
                continue
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in {"small_network", "small_network_core"}:
                            import_violations.append(f"{path}:{node.lineno}")
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    imported_names = {alias.name for alias in node.names}
                    if (
                        module in {"small_network", "small_network_core"}
                        or module.endswith(".small_network")
                        or module.endswith(".small_network_core")
                        or (
                            module in {"labs", ""}
                            and imported_names & {"small_network", "small_network_core"}
                        )
                    ):
                        import_violations.append(f"{path}:{node.lineno}")
                elif isinstance(node, ast.Call):
                    target = node.func
                    is_process_call = (
                        isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id in {"os", "subprocess"}
                        and target.attr
                        in {"system", "run", "Popen", "call", "check_call", "check_output"}
                    )
                    if is_process_call and any(
                        isinstance(arg, ast.Constant)
                        and isinstance(arg.value, str)
                        and ("small_network.py" in arg.value or "small_network_core" in arg.value)
                        for arg in node.args
                    ):
                        execution_violations.append(f"{path}:{node.lineno}")

    for root_name in ("experiments", "labs"):
        for pattern in ("*.sh", "*.slurm"):
            for path in (REPO_ROOT / root_name).rglob(pattern):
                for line_number, line in enumerate(path.read_text().splitlines(), start=1):
                    if "small_network.py" in line or "-m small_network" in line:
                        execution_violations.append(f"{path}:{line_number}")

    assert import_violations == []
    assert execution_violations == []
