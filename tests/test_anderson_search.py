from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from labs.tools import anderson_search


def _base_config() -> dict:
    path = anderson_search._repo_root() / "examples" / "small_drn" / "base.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_prepare_config_deep_copies_and_mutates_nested_versioned_fields() -> None:
    base = _base_config()
    original = deepcopy(base)

    prepared = anderson_search._prepare_config(
        base,
        m=5,
        omega=0.625,
        reg=2e-7,
        batch_size=13,
        linspace_samples=17,
    )

    assert base == original
    assert prepared["solver"]["minimizer_impl"] == "anderson"
    assert prepared["solver"]["anderson"] == {
        **original["solver"]["anderson"],
        "memory": 5,
        "omega": 0.625,
        "regularization": 2e-7,
    }
    assert prepared["data"]["batch_size"] == 13
    assert prepared["modes"]["linspace"]["samples"] == 17
    assert "anderson_m" not in prepared
    assert "linspace_samples" not in prepared


def test_load_base_config_uses_strict_small_drn_schema(
    tmp_path: Path,
) -> None:
    payload = _base_config()
    config_path = tmp_path / "base.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    assert anderson_search._load_base_config(config_path) == payload

    payload["legacy_output_dir"] = "runs"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="contain only the keys"):
        anderson_search._load_base_config(config_path)


def test_run_linspace_uses_public_ebl_command_and_repo_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls = []

    def run(command, **options):
        calls.append((command, options))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(anderson_search.subprocess, "run", run)
    config_path = tmp_path / "step.json"
    output_dir = tmp_path / "runs"
    weights_path = tmp_path / "weights.pt"
    env = {"OMP_NUM_THREADS": "1"}

    result = anderson_search._run_linspace(
        config_path=config_path,
        output_dir=output_dir,
        weights_path=weights_path,
        env=env,
    )

    assert result.returncode == 0
    assert calls == [
        (
            [
                sys.executable,
                "-m",
                "ebl",
                "linspace",
                "--config",
                str(config_path),
                "--output-dir",
                str(output_dir),
                "--weights",
                str(weights_path),
            ],
            {
                "text": True,
                "capture_output": True,
                "env": env,
                "cwd": anderson_search._repo_root(),
            },
        )
    ]
    assert anderson_search._compare_script_path() == (
        Path(anderson_search.__file__).resolve().with_name("compare_npz.sh")
    )


def test_resolve_states_artifact_uses_the_single_completed_result(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "step"
    run_dir = output_root / "run-001"
    states_path = run_dir / "artifacts" / "settled_states.npz"
    states_path.parent.mkdir(parents=True)
    states_path.write_bytes(b"states")
    (run_dir / "result.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "artifacts": [
                    {
                        "kind": "solver_iterations",
                        "path": "artifacts/iterations.npz",
                    },
                    {
                        "kind": "states",
                        "path": "artifacts/settled_states.npz",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    assert anderson_search._resolve_states_artifact(output_root) == (
        run_dir.resolve(),
        states_path.resolve(),
    )

    second_run = output_root / "run-002"
    second_run.mkdir()
    (second_run / "result.json").write_text(
        json.dumps({"status": "complete", "artifacts": []}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exactly one completed run"):
        anderson_search._resolve_states_artifact(output_root)
