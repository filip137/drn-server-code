from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

import pytest

from experiments.launch import (
    build_launch_command,
    build_status_command,
    launch,
    launch_after_local_canary,
    load_targets,
    main,
    run_local_canary,
    status,
)


def test_load_targets(tmp_path: Path) -> None:
    path = tmp_path / "targets.json"
    path.write_text(
        json.dumps({"cpu": {"kind": "local", "workdir": "."}}),
        encoding="utf-8",
    )

    assert load_targets(path)["cpu"]["kind"] == "local"


def test_load_targets_rejects_non_boolean_canary_requirement(tmp_path: Path) -> None:
    path = tmp_path / "targets.json"
    path.write_text(
        json.dumps(
            {
                "cluster": {
                    "kind": "slurm",
                    "workdir": ".",
                    "require_local_canary": "yes",
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="require_local_canary"):
        load_targets(path)


def test_load_targets_rejects_invalid_ssh_command(tmp_path: Path) -> None:
    path = tmp_path / "targets.json"
    path.write_text(
        json.dumps(
            {
                "cluster": {
                    "kind": "slurm",
                    "workdir": ".",
                    "ssh": "ssh -F config",
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="ssh to be a non-empty string list"):
        load_targets(path)


def test_build_local_tmux_command() -> None:
    command = build_launch_command(
        {"kind": "tmux", "session": "main", "workdir": "/repo"},
        ["python", "train.py", "--seed", "0"],
        name="smoke",
        log="/results/smoke.log",
    )

    assert command[:8] == [
        "tmux",
        "new-window",
        "-dP",
        "-F",
        "#{window_id}",
        "-t",
        "main",
        "-n",
    ]
    assert command[8] == "smoke"
    assert "python train.py --seed 0" in command[-1]
    assert "/results/smoke.log.exitcode" in command[-1]


def test_build_ssh_tmux_command_quotes_once() -> None:
    command = build_launch_command(
        {"kind": "ssh-tmux", "host": "trex", "workdir": "/repo with space"},
        ["python", "train.py", "--label", "a b"],
        name="trial",
        log="/tmp/trial.log",
    )

    assert command[:2] == ["ssh", "trex"]
    assert "tmux new-session -d -s trial" in command[2]
    assert "'/repo with space'" in command[2]
    assert "'a b'" in command[2]


def test_build_slurm_command() -> None:
    command = build_launch_command(
        {
            "kind": "slurm",
            "host": "jean-zay",
            "workdir": "/repo",
            "sbatch": ["--account=fmu@v100", "--gres=gpu:1"],
        },
        ["python", "train.py"],
        name="trial",
        log="/results/%x-%j.log",
        slurm_args=["--array=0-5%2", "--time=02:00:00"],
    )

    assert command[:2] == ["ssh", "jean-zay"]
    assert command[2].startswith("mkdir -p /results && ")
    assert "sbatch --parsable" in command[2]
    assert "--account=fmu@v100" in command[2]
    assert "--array=0-5%2" in command[2]
    assert "--time=02:00:00" in command[2]
    assert "'--wrap=python train.py'" in command[2]


def test_build_slurm_command_uses_configured_ssh_command() -> None:
    command = build_launch_command(
        {
            "kind": "slurm",
            "host": "jean-zay",
            "workdir": "/repo",
            "ssh": ["ssh", "-F", "/home/user/.ssh/config"],
        },
        ["true"],
        name="trial",
        log=None,
    )

    assert command[:4] == ["ssh", "-F", "/home/user/.ssh/config", "jean-zay"]


def test_slurm_args_are_rejected_for_non_slurm_target() -> None:
    with pytest.raises(ValueError, match="only for a Slurm target"):
        build_launch_command(
            {"kind": "local", "workdir": "."},
            ["true"],
            name="trial",
            log=None,
            slurm_args=["--array=0-1"],
        )


def test_detached_targets_require_a_log() -> None:
    with pytest.raises(ValueError, match="Expected --log"):
        build_launch_command(
            {"kind": "tmux", "session": "main", "workdir": "/repo"},
            ["true"],
            name="trial",
            log=None,
        )


def test_local_launch_propagates_exit_code(tmp_path: Path) -> None:
    result = launch(
        "local",
        {"kind": "local", "workdir": str(tmp_path)},
        [sys.executable, "-c", "raise SystemExit(7)"],
        name="failure",
    )

    assert result["state"] == "finished"
    assert result["returncode"] == 7


def test_local_launch_writes_log_and_exit_code(tmp_path: Path) -> None:
    result = launch(
        "local",
        {"kind": "local", "workdir": str(tmp_path)},
        [sys.executable, "-c", "print('hello')"],
        name="logged",
        log="logs/run.log",
    )

    assert result["returncode"] == 0
    assert (tmp_path / "logs/run.log").read_text(encoding="utf-8") == "hello\n"
    assert (tmp_path / "logs/run.log.exitcode").read_text(encoding="utf-8") == "0\n"


def test_dry_run_has_no_side_effect(tmp_path: Path) -> None:
    marker = tmp_path / "marker"
    result = launch(
        "local",
        {"kind": "local", "workdir": str(tmp_path)},
        [sys.executable, "-c", f"open({str(marker)!r}, 'w').close()"],
        name="planned",
        dry_run=True,
    )

    assert result["state"] == "planned"
    assert not marker.exists()


def test_local_canary_passes_before_production(tmp_path: Path) -> None:
    canary_marker = tmp_path / "canary.json"
    production_marker = tmp_path / "production"

    result = launch_after_local_canary(
        "local",
        {"kind": "local", "workdir": str(tmp_path)},
        [
            sys.executable,
            "-c",
            f"open({str(production_marker)!r}, 'w').write('production')",
        ],
        name="gated",
        canary_argv=[
            sys.executable,
            "-c",
            f"open({str(canary_marker)!r}, 'w').write('canary')",
        ],
        canary_workdir=str(tmp_path),
        canary_required_files=[str(canary_marker)],
    )

    assert result["state"] == "finished"
    assert result["local_canary"]["state"] == "passed"
    assert result["local_canary"]["required_artifacts"][0]["bytes"] == 6
    assert production_marker.read_text(encoding="utf-8") == "production"


def test_failed_local_canary_blocks_production(tmp_path: Path) -> None:
    production_marker = tmp_path / "production"

    result = launch_after_local_canary(
        "local",
        {"kind": "local", "workdir": str(tmp_path)},
        [
            sys.executable,
            "-c",
            f"open({str(production_marker)!r}, 'w').close()",
        ],
        name="gated",
        canary_argv=[sys.executable, "-c", "raise SystemExit(9)"],
        canary_workdir=str(tmp_path),
    )

    assert result["state"] == "blocked"
    assert result["stage"] == "local-canary"
    assert result["returncode"] == 9
    assert result["production"] is None
    assert not production_marker.exists()


def test_missing_local_canary_artifact_blocks_production(tmp_path: Path) -> None:
    production_marker = tmp_path / "production"

    result = launch_after_local_canary(
        "local",
        {"kind": "local", "workdir": str(tmp_path)},
        [
            sys.executable,
            "-c",
            f"open({str(production_marker)!r}, 'w').close()",
        ],
        name="gated",
        canary_argv=[sys.executable, "-c", "print('canary passed')"],
        canary_workdir=str(tmp_path),
        canary_required_files=["missing.json"],
    )

    assert result["state"] == "blocked"
    assert result["local_canary"]["state"] == "failed"
    assert "missing or empty" in result["local_canary"]["error"]
    assert not production_marker.exists()


def test_local_canary_dry_run_plans_both_steps_without_executing(
    tmp_path: Path,
) -> None:
    canary_marker = tmp_path / "canary"
    production_marker = tmp_path / "production"

    result = launch_after_local_canary(
        "local",
        {"kind": "local", "workdir": str(tmp_path)},
        [
            sys.executable,
            "-c",
            f"open({str(production_marker)!r}, 'w').close()",
        ],
        name="gated",
        canary_argv=[
            sys.executable,
            "-c",
            f"open({str(canary_marker)!r}, 'w').close()",
        ],
        canary_workdir=str(tmp_path),
        dry_run=True,
    )

    assert result["state"] == "planned"
    assert result["local_canary"]["state"] == "planned"
    assert result["production"]["state"] == "planned"
    assert not canary_marker.exists()
    assert not production_marker.exists()


def test_run_local_canary_writes_log(tmp_path: Path) -> None:
    result = run_local_canary(
        [sys.executable, "-c", "print('canary output')"],
        workdir=str(tmp_path),
        log="logs/canary.log",
    )

    assert result["state"] == "passed"
    assert (tmp_path / "logs/canary.log").read_text(encoding="utf-8") == (
        "canary output\n"
    )


def test_status_commands() -> None:
    assert build_status_command(
        {"kind": "tmux", "workdir": ".", "session": "main"}, "@3"
    ) == [
        "tmux",
        "list-panes",
        "-t",
        "@3",
        "-F",
        "#{pane_dead} #{pane_dead_status}",
    ]
    slurm = build_status_command(
        {"kind": "slurm", "host": "jean-zay", "workdir": "/repo"}, "123"
    )
    assert slurm[:2] == ["ssh", "jean-zay"]
    assert "sacct -j 123" in slurm[2]


def test_status_uses_saved_tmux_exit_code(tmp_path: Path) -> None:
    (tmp_path / "run.log.exitcode").write_text("4\n", encoding="utf-8")

    result = status(
        "main",
        {"kind": "tmux", "workdir": str(tmp_path), "session": "main"},
        "@9",
        log="run.log",
    )

    assert result["state"] == "failed"
    assert result["detail"] == "exitcode=4"


def test_status_distinguishes_missing_tmux_from_transport_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(*args: object, **kwargs: object) -> object:
        return type(
            "Completed",
            (),
            {"returncode": 1, "stdout": "", "stderr": "can't find window: @9"},
        )()

    monkeypatch.setattr("experiments.launch.subprocess.run", missing)
    target = {"kind": "tmux", "workdir": ".", "session": "main"}
    assert status("main", target, "@9")["state"] == "not-running"

    def broken(*args: object, **kwargs: object) -> object:
        return type(
            "Completed",
            (),
            {"returncode": 255, "stdout": "", "stderr": "connection refused"},
        )()

    monkeypatch.setattr("experiments.launch.subprocess.run", broken)
    with pytest.raises(RuntimeError, match="connection refused"):
        status("main", target, "@9")


def test_slurm_launch_accepts_banner_before_job_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def submitted(*args: object, **kwargs: object) -> object:
        return type(
            "Completed",
            (),
            {"returncode": 0, "stdout": "Welcome\n12345;cluster\n", "stderr": ""},
        )()

    monkeypatch.setattr("experiments.launch.subprocess.run", submitted)
    result = launch(
        "jean-zay",
        {"kind": "slurm", "host": "jean-zay", "workdir": "/repo"},
        ["true"],
        name="trial",
    )

    assert result["handle"] == "12345"


def test_cli_accepts_options_after_target(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(
        [
            "run",
            "local",
            "--name",
            "example",
            "--dry-run",
            "--",
            sys.executable,
            "-c",
            "print(1)",
        ]
    ) == 0

    assert json.loads(capsys.readouterr().out)["state"] == "planned"


def test_cli_accepts_slurm_array_argument(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    targets = tmp_path / "targets.json"
    targets.write_text(
        json.dumps(
            {
                "cluster": {
                    "kind": "slurm",
                    "host": "cluster",
                    "workdir": "/repo",
                }
            }
        ),
        encoding="utf-8",
    )

    assert main(
        [
            "--targets",
            str(targets),
            "run",
            "cluster",
            "--name",
            "batch",
            "--slurm-arg=--array=0-3",
            "--dry-run",
            "--",
            "python",
            "-m",
            "experiments.exact_run",
        ]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["slurm_args"] == ["--array=0-3"]


def test_cli_required_local_canary_cannot_be_omitted(
    tmp_path: Path,
) -> None:
    targets = tmp_path / "targets.json"
    targets.write_text(
        json.dumps(
            {
                "cluster": {
                    "kind": "slurm",
                    "host": "cluster",
                    "workdir": "/repo",
                    "require_local_canary": True,
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="requires --local-canary-command"):
        main(
            [
                "--targets",
                str(targets),
                "run",
                "cluster",
                "--name",
                "batch",
                "--dry-run",
                "--",
                "true",
            ]
        )


def test_cli_local_canary_runs_before_local_production(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    targets = tmp_path / "targets.json"
    targets.write_text(
        json.dumps({"local-test": {"kind": "local", "workdir": str(tmp_path)}}),
        encoding="utf-8",
    )
    canary_marker = tmp_path / "canary"
    production_marker = tmp_path / "production"
    canary_command = (
        f"{shlex.quote(sys.executable)} -c "
        + shlex.quote(f"open({str(canary_marker)!r}, 'w').write('ok')")
    )

    assert main(
        [
            "--targets",
            str(targets),
            "run",
            "local-test",
            "--name",
            "gated",
            "--local-canary-command",
            canary_command,
            "--local-canary-workdir",
            str(tmp_path),
            "--local-canary-require",
            str(canary_marker),
            "--",
            sys.executable,
            "-c",
            f"open({str(production_marker)!r}, 'w').write('ok')",
        ]
    ) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["local_canary"]["state"] == "passed"
    assert payload["production"]["state"] == "finished"
    assert production_marker.read_text(encoding="utf-8") == "ok"
