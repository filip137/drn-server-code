from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import fcntl
import json
import os
import signal
import shutil
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path

import pytest

import experiments.submit_mnist_conv_perfectdiode_successor_confirmation_jeanzay as submitter
import experiments.supervise_mnist_conv_perfectdiode_successor_confirmation_jeanzay as supervisor
import experiments.verify_jeanzay_canary as official_canary
import experiments.verify_mnist_conv_perfectdiode_successor_canary as canary
from experiments.mnist_conv.io import (
    atomic_create_json,
    atomic_write_json,
    read_json,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
ENVIRONMENT_CONTRACT = (
    REPO_ROOT
    / "configs"
    / "conv"
    / "perfectdiode_successor_jeanzay_umg_v100_environment_v1.json"
)


def _runtime_preflight() -> dict[str, object]:
    return {
        "schema_version": (
            "mnist-conv-perfectdiode-successor-preflight/v1"
        ),
        "status": "passed",
        "passed": True,
        "official_test_read": False,
        "bundle_id": "pdconfirm_" + "1" * 64,
        "bundle_manifest_sha256": "2" * 64,
        "config_sha256": "3" * 64,
        "config_file_sha256": "5" * 64,
        "approved_plan_sha256": "4" * 64,
        "expected_entry_count": 12,
        "checks": {
            "approved_plan": True,
            "launch_authorized": True,
            "selection_override_approved": True,
            "config_valid": True,
            "manifest_valid": True,
            "official_test_prohibited": True,
            "expected_entry_count": True,
            "source_archive_valid": True,
            "environment_contract_valid": True,
            "worker_launcher_valid": True,
            "runtime_module_valid": True,
            "implementation_ready": True,
        },
    }


def _fixture_repo(tmp_path: Path) -> dict[str, Path]:
    repo = tmp_path / "repo"
    experiments = repo / "experiments"
    runtime_package = experiments / "mnist_conv"
    skill_scripts = (
        repo
        / "skills"
        / "scheduled-run-preflight"
        / "scripts"
    )
    plan_skill_scripts = (
        repo
        / "skills"
        / "run-experiment-pipeline"
        / "scripts"
    )
    experiments.mkdir(parents=True)
    runtime_package.mkdir()
    skill_scripts.mkdir(parents=True)
    plan_skill_scripts.mkdir(parents=True)
    wrapper = experiments / submitter.WRAPPER_NAME
    shutil.copy2(
        REPO_ROOT / "experiments" / submitter.WRAPPER_NAME,
        wrapper,
    )
    wrapper.chmod(0o700)
    scheduled = skill_scripts / "preflight_scheduled_runner.py"
    shutil.copy2(
        REPO_ROOT
        / "skills"
        / "scheduled-run-preflight"
        / "scripts"
        / "preflight_scheduled_runner.py",
        scheduled,
    )
    runtime = experiments / submitter.RUNTIME_CLI_NAME
    runtime.write_text(
        "#!/usr/bin/env python3\n"
        "import json\n"
        f"value = {repr(_runtime_preflight())}\n"
        "print('PD_SUCCESSOR_RESULT_JSON=' + "
        "json.dumps(value, sort_keys=True, separators=(',', ':')))\n",
        encoding="utf-8",
    )
    runtime.chmod(0o700)
    runtime_module = runtime_package / "perfectdiode_successor_confirmation.py"
    runtime_module.write_text(
        '"""Dummy staged successor runtime module."""\n',
        encoding="utf-8",
    )
    for name in (
        submitter.SUBMITTER_NAME,
        submitter.SUPERVISOR_NAME,
        submitter.SEMANTIC_CANARY_VERIFIER_NAME,
        "verify_jeanzay_canary.py",
    ):
        shutil.copy2(REPO_ROOT / "experiments" / name, experiments / name)
    shutil.copy2(
        REPO_ROOT / submitter.EXPERIMENT_PLAN_VALIDATOR_RELATIVE,
        repo / submitter.EXPERIMENT_PLAN_VALIDATOR_RELATIVE,
    )
    tracker_validator = (
        plan_skill_scripts / "validate_current_experiments.py"
    )
    shutil.copy2(
        REPO_ROOT / supervisor.TRACKER_VALIDATOR_RELATIVE,
        tracker_validator,
    )
    tracker = repo / supervisor.TRACKER_RELATIVE
    tracker.parent.mkdir(parents=True)
    tracker.write_text(
        "# Current Experiments\n\n"
        f"<!-- experiment-id: {supervisor.EXPERIMENT_ID} -->\n"
        "### Perfect-diode successor\n\n"
        "- **Testing:** Exact approved Conv1/Conv2 confirmation.\n"
        "- **Where:** Immutable Jean Zay staging paths are assigned.\n"
        "- **Status:** `preflighting` — payload canary gates are active.\n",
        encoding="utf-8",
    )
    tracker_gate_receipt = tmp_path / "tracker-canary-gate.json"
    output = tmp_path / "output"
    supervisor_state = tmp_path / "supervisor-state.json"
    subprocess.run(
        [
            sys.executable,
            str(tracker_validator),
            str(tracker),
            "--require-experiment-id",
            supervisor.EXPERIMENT_ID,
            "--write-gate-receipt",
            str(tracker_gate_receipt),
            "--gate-stage",
            "canary",
            "--gate-state",
            str(supervisor_state),
            "--gate-output-root",
            str(output),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    bundle = tmp_path / "bundle"
    production_output = tmp_path / "production-output"
    data = tmp_path / "mnist"
    for path in (bundle, output, production_output, data):
        path.mkdir()
    (bundle / "immutable.txt").write_text("bundle\n", encoding="utf-8")
    archive = tmp_path / "source.tar.gz"
    with tarfile.open(archive, mode="w:gz") as handle:
        handle.add(
            runtime,
            arcname=f"experiments/{submitter.RUNTIME_CLI_NAME}",
        )
        handle.add(
            runtime_module,
            arcname=(
                "experiments/mnist_conv/"
                "perfectdiode_successor_confirmation.py"
            ),
        )
    environment = tmp_path / ENVIRONMENT_CONTRACT.name
    shutil.copy2(ENVIRONMENT_CONTRACT, environment)
    environment_value = read_json(environment)
    environment_value["resources"]["python_executable"] = sys.executable
    atomic_write_json(environment, environment_value, canonical=True)
    return {
        "repo": repo,
        "wrapper": wrapper,
        "runtime": runtime,
        "runtime_module": runtime_module,
        "bundle": bundle,
        "output": output,
        "production_output": production_output,
        "data": data,
        "archive": archive,
        "environment": environment,
        "tracker": tracker,
        "tracker_validator": tracker_validator,
        "tracker_gate_receipt": tracker_gate_receipt,
        "supervisor_state": supervisor_state,
    }


def _args(
    paths: dict[str, Path],
    *,
    kind: str = "canary",
    submit: bool = False,
    test_only: bool = False,
) -> argparse.Namespace:
    return argparse.Namespace(
        kind=kind,
        bundle_dir=str(paths["bundle"]),
        output_root=str(paths["output"]),
        data_root=str(paths["data"]),
        source_archive=str(paths["archive"]),
        environment_contract=str(paths["environment"]),
        launch_authorization_receipt=None,
        launch_authorization_receipt_sha256=None,
        scheduled_preflight_receipt=None,
        preflight_receipt=None,
        canary_receipt=str(
            paths["output"].parent / "canary-gate.json"
        ),
        canary_receipt_sha256=None,
        official_canary_receipt=str(
            paths["output"].parent / "official-canary-gate.json"
        ),
        supervisor_state=str(
            paths["supervisor_state"]
        ),
        repo_root=str(paths["repo"]),
        python=sys.executable,
        official_verifier=str(
            paths["repo"] / "experiments" / "verify_jeanzay_canary.py"
        ),
        remote_user="testuser",
        canary_pack_index=4,
        tracker=str(paths["tracker"]),
        tracker_validator=str(paths["tracker_validator"]),
        tracker_gate_receipt=str(paths["tracker_gate_receipt"]),
        tracker_gate_receipt_sha256=submitter.sha256_file(
            paths["tracker_gate_receipt"]
        ),
        command_timeout_seconds=30.0,
        submit=submit,
        test_only=test_only,
    )


def _tree_snapshot(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): submitter.sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_official_canary_verifier_is_repo_local_and_hash_bound(
    tmp_path: Path,
) -> None:
    paths = _fixture_repo(tmp_path)
    args = _args(paths)
    args.state = str(paths["supervisor_state"])
    verifier = Path(args.official_verifier).resolve()
    assert verifier == (
        paths["repo"] / "experiments" / "verify_jeanzay_canary.py"
    ).resolve()
    assert ".codex" not in verifier.parts
    assert submitter.sha256_file(verifier) == (
        canary.OFFICIAL_CANARY_VERIFIER_SHA256
    )


def test_default_official_verifier_comes_from_environment_contract() -> None:
    contract = submitter.load_environment_contract(ENVIRONMENT_CONTRACT)
    assert submitter.expected_official_canary_verifier_path(
        contract,
        remote_user="testuser",
    ) == Path(
        "/lustre/fswork/projects/rech/umg/testuser/server_code/"
        "launch_tools/verify_canary.py"
    )
    submitter_args = submitter._parser().parse_args(
        [
            "--kind",
            "canary",
            "--bundle-dir",
            "/tmp/bundle",
            "--output-root",
            "/tmp/output",
            "--data-root",
            "/tmp/data",
            "--source-archive",
            "/tmp/source.tar.gz",
            "--environment-contract",
            "/tmp/environment.json",
        ]
    )
    assert submitter_args.official_verifier is None


def test_prepare_submission_resolves_default_verifier_with_storage_enforced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _fixture_repo(tmp_path)
    launch_tools = tmp_path / "testuser" / "repo" / "launch_tools"
    launch_tools.mkdir(parents=True)
    staged_verifier = launch_tools / "verify_canary.py"
    shutil.copy2(
        REPO_ROOT / "experiments" / "verify_jeanzay_canary.py",
        staged_verifier,
    )
    contract = submitter.load_environment_contract(ENVIRONMENT_CONTRACT)
    contract["resources"]["python_executable"] = submitter.PYTHON_EXECUTABLE
    contract["storage"] = {
        "source_root_template": str(tmp_path),
        "data_root_template": str(paths["data"]),
        "result_root_template": str(tmp_path),
    }
    contract["verification"][
        "official_canary_verifier_path_template"
    ] = str(tmp_path / "{user}" / "repo" / "launch_tools" / "verify_canary.py")
    monkeypatch.setattr(
        submitter,
        "load_environment_contract",
        lambda _path: contract,
    )
    args = _args(paths)
    args.python = submitter.PYTHON_EXECUTABLE
    args.official_verifier = None
    _command, metadata = submitter.prepare_submission(
        args,
        enforce_storage_roots=True,
    )
    assert metadata["official_canary_verifier"] == str(
        staged_verifier.resolve()
    )
    assert metadata["hashes"][
        "official_canary_verifier_sha256"
    ] == submitter.sha256_file(staged_verifier)


def test_canary_receipts_are_first_write_wins(
    tmp_path: Path,
) -> None:
    official_path = tmp_path / "official.json"
    original = {"status": "original"}
    official_canary.write_receipt_exclusive(official_path, original)
    original_bytes = official_path.read_bytes()
    with pytest.raises(ValueError, match="not to exist"):
        official_canary.write_receipt_exclusive(
            official_path,
            {"status": "replacement"},
        )
    assert official_path.read_bytes() == original_bytes

    semantic_path = tmp_path / "semantic.json"
    atomic_create_json(semantic_path, original, canonical=True)
    semantic_bytes = semantic_path.read_bytes()
    with pytest.raises(FileExistsError, match="not to exist"):
        atomic_create_json(
            semantic_path,
            {"status": "replacement"},
            canonical=True,
        )
    assert semantic_path.read_bytes() == semantic_bytes


@pytest.mark.parametrize(
    "receipt_argument",
    ["official_canary_receipt", "canary_receipt"],
)
def test_new_supervisor_attempt_rejects_preexisting_canary_receipt(
    tmp_path: Path,
    receipt_argument: str,
) -> None:
    official_path = tmp_path / "official.json"
    semantic_path = tmp_path / "semantic.json"
    selected = (
        official_path
        if receipt_argument == "official_canary_receipt"
        else semantic_path
    )
    selected.write_text("prior evidence\n", encoding="utf-8")
    value = supervisor.SuccessorJeanZaySupervisor(
        args=argparse.Namespace(
            state=str(tmp_path / "state.json"),
            official_canary_receipt=str(official_path),
            canary_receipt=str(semantic_path),
        ),
    )
    with pytest.raises(ValueError, match="new-attempt"):
        value._require_fresh_canary_receipt_paths()
    assert selected.read_text(encoding="utf-8") == "prior evidence\n"


def test_remote_tracker_gate_uses_canonical_hash_bound_receipt(
    tmp_path: Path,
) -> None:
    paths = _fixture_repo(tmp_path)
    args = _args(paths)
    args.state = str(paths["supervisor_state"])
    # A staged Markdown copy is not live authority after the canonical
    # workflow has emitted its immutable receipt.
    paths["tracker"].write_text(
        paths["tracker"].read_text(encoding="utf-8").replace(
            "`preflighting`",
            "`blocked`",
        ),
        encoding="utf-8",
    )
    value = supervisor.SuccessorJeanZaySupervisor(args=args)
    gate = value._validate_tracker(required_status="preflighting")
    assert gate["status"] == "preflighting"
    assert gate["receipt_sha256"] == args.tracker_gate_receipt_sha256
    assert gate["validator_sha256"] == supervisor.TRACKER_VALIDATOR_SHA256
    assert gate["tracker_sha256"] != submitter.sha256_file(paths["tracker"])


@pytest.mark.parametrize(
    ("changed_argument", "expected_fragment"),
    [
        ("state", "state_path"),
        ("output_root", "output_root"),
    ],
)
def test_tracker_gate_receipt_is_bound_to_exact_attempt(
    tmp_path: Path,
    changed_argument: str,
    expected_fragment: str,
) -> None:
    paths = _fixture_repo(tmp_path)
    args = _args(paths)
    args.state = str(paths["supervisor_state"])
    setattr(args, changed_argument, str(tmp_path / f"other-{changed_argument}"))
    value = supervisor.SuccessorJeanZaySupervisor(args=args)
    with pytest.raises(RuntimeError, match=expected_fragment):
        value._validate_tracker(required_status="preflighting")


def test_stale_tracker_gate_receipt_is_rejected_before_submission(
    tmp_path: Path,
) -> None:
    paths = _fixture_repo(tmp_path)
    stale_path = tmp_path / "stale-tracker-gate.json"
    stale = read_json(paths["tracker_gate_receipt"])
    stale["generated_at_utc"] = (
        datetime.now(timezone.utc)
        - timedelta(seconds=supervisor.TRACKER_GATE_MAX_AGE_SECONDS + 1)
    ).isoformat()
    atomic_write_json(stale_path, stale, canonical=True)
    args = _args(paths)
    args.state = str(paths["supervisor_state"])
    args.tracker_gate_receipt = str(stale_path)
    args.tracker_gate_receipt_sha256 = submitter.sha256_file(stale_path)
    value = supervisor.SuccessorJeanZaySupervisor(args=args)
    with pytest.raises(RuntimeError, match="fresh"):
        value._validate_tracker(required_status="preflighting")


def test_canonical_validator_emits_production_launch_ready_receipt(
    tmp_path: Path,
) -> None:
    paths = _fixture_repo(tmp_path)
    paths["tracker"].write_text(
        paths["tracker"].read_text(encoding="utf-8").replace(
            "`preflighting` — payload canary gates are active.",
            "`launch-ready` — canary passed; production is authorized.",
        ),
        encoding="utf-8",
    )
    production_receipt = tmp_path / "tracker-production-gate.json"
    subprocess.run(
        [
            sys.executable,
            str(paths["tracker_validator"]),
            str(paths["tracker"]),
            "--require-experiment-id",
            supervisor.EXPERIMENT_ID,
            "--require-launch-ready",
            "--write-gate-receipt",
            str(production_receipt),
            "--gate-stage",
            "production",
            "--gate-state",
            str(paths["supervisor_state"]),
            "--gate-output-root",
            str(paths["output"]),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    args = _args(paths)
    args.state = str(paths["supervisor_state"])
    args.tracker_gate_receipt = str(production_receipt)
    args.tracker_gate_receipt_sha256 = submitter.sha256_file(
        production_receipt
    )
    value = supervisor.SuccessorJeanZaySupervisor(args=args)
    gate = value._validate_tracker(required_status="launch-ready")
    assert gate["gate_stage"] == "production"
    assert gate["status"] == "launch-ready"


def test_tracker_receipt_publication_is_atomic_first_write_wins(
    tmp_path: Path,
) -> None:
    paths = _fixture_repo(tmp_path)
    destination = tmp_path / "contended-tracker-gate.json"
    command = [
        sys.executable,
        str(paths["tracker_validator"]),
        str(paths["tracker"]),
        "--require-experiment-id",
        supervisor.EXPERIMENT_ID,
        "--write-gate-receipt",
        str(destination),
        "--gate-stage",
        "canary",
        "--gate-state",
        str(paths["supervisor_state"]),
        "--gate-output-root",
        str(paths["output"]),
    ]

    def invoke() -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _index: invoke(), range(2)))
    assert sorted(result.returncode for result in results) == [0, 1]
    receipt = read_json(destination)
    assert receipt["experiment_id"] == supervisor.EXPERIMENT_ID
    assert receipt["state_path"] == str(paths["supervisor_state"])
    assert not list(destination.parent.glob(f".{destination.name}.*.tmp"))


def test_canary_to_production_uses_distinct_receipt_and_new_invocation(
    tmp_path: Path,
) -> None:
    paths = _fixture_repo(tmp_path)
    canary_args = _args(paths)
    canary_args.state = str(paths["supervisor_state"])
    canary_supervisor = supervisor.SuccessorJeanZaySupervisor(
        args=canary_args
    )
    assert canary_supervisor._validate_tracker(
        required_status="preflighting"
    )["gate_stage"] == "canary"

    paths["tracker"].write_text(
        paths["tracker"].read_text(encoding="utf-8").replace(
            "`preflighting` — payload canary gates are active.",
            "`launch-ready` — canary passed; production is authorized.",
        ),
        encoding="utf-8",
    )
    production_receipt = tmp_path / "tracker-production-handoff.json"
    subprocess.run(
        [
            sys.executable,
            str(paths["tracker_validator"]),
            str(paths["tracker"]),
            "--require-experiment-id",
            supervisor.EXPERIMENT_ID,
            "--require-launch-ready",
            "--write-gate-receipt",
            str(production_receipt),
            "--gate-stage",
            "production",
            "--gate-state",
            str(paths["supervisor_state"]),
            "--gate-output-root",
            str(paths["output"]),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    production_args = _args(paths)
    production_args.state = str(paths["supervisor_state"])
    production_args.tracker_gate_receipt = str(production_receipt)
    production_args.tracker_gate_receipt_sha256 = submitter.sha256_file(
        production_receipt
    )
    production_supervisor = supervisor.SuccessorJeanZaySupervisor(
        args=production_args
    )
    resumed_gate = production_supervisor._validate_tracker(
        required_status=None
    )
    assert resumed_gate["gate_stage"] == "production"
    assert production_supervisor._validate_tracker(
        required_status="launch-ready"
    )["status"] == "launch-ready"
    assert production_receipt != paths["tracker_gate_receipt"]
    assert production_args.tracker_gate_receipt_sha256 != (
        canary_args.tracker_gate_receipt_sha256
    )


def _enable_fake_jeanzay_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_bin = tmp_path / "fake-jeanzay-bin"
    fake_bin.mkdir(exist_ok=True)
    for name, body in (
        ("idrenv", "#!/bin/sh\nprintf ':'\n"),
        ("module", "#!/bin/sh\nexit 0\n"),
    ):
        path = fake_bin / name
        path.write_text(body, encoding="utf-8")
        path.chmod(0o700)
    monkeypatch.setenv(
        "PATH",
        os.pathsep.join(
            [
                str(Path(sys.executable).parent),
                str(fake_bin),
                os.environ.get("PATH", ""),
            ]
        ),
    )


def _allocation(**unused: object) -> dict[str, object]:
    return {
        "verified": True,
        "allocation_id": "AD011016471R1",
        "idrenv_project": "umg",
        "slurm_account": "umg@v100",
        "partition": "gpu_p13",
        "qos": "qos_gpu-t3",
        "constraint": "v100-32g",
    }


def _tracker_gate() -> dict[str, object]:
    return {
        "status": "preflighting",
        "experiment_id": supervisor.EXPERIMENT_ID,
    }


def _artifact(base: Path, path: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(base).as_posix(),
        "sha256": submitter.sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _write_launch_authorization(
    paths: dict[str, Path],
) -> tuple[Path, str]:
    args = _args(paths)
    _command, metadata = submitter.prepare_submission(
        args, enforce_storage_roots=False
    )
    plan = (
        paths["repo"]
        / "docs"
        / "experiment_plans"
        / "perfectdiode-conv12-best-observed-confirmation-20260727-v1.md"
    )
    plan.parent.mkdir(parents=True)
    plan.write_text("# Approved plan\n", encoding="utf-8")
    hashes = metadata["hashes"]
    receipt = paths["output"].parent / "launch-authorization.json"
    value = {
        "schema_version": (
            "perfectdiode-successor-launch-authorization/v1"
        ),
        "status": "approved",
        "launch_authorized": True,
        "approved_plan": {
            "path": str(plan),
            "sha256": submitter.sha256_file(plan),
        },
        "bundle": {
            "bundle_id": _runtime_preflight()["bundle_id"],
            "manifest_sha256": _runtime_preflight()[
                "bundle_manifest_sha256"
            ],
            "config_sha256": _runtime_preflight()["config_sha256"],
            "config_file_sha256": _runtime_preflight()[
                "config_file_sha256"
            ],
            "bundle_tree_sha256": hashes["bundle_sha256"],
        },
        "execution": {
            "source_archive_sha256": hashes["source_archive_sha256"],
            "environment_contract_sha256": hashes[
                "environment_contract_sha256"
            ],
            "runtime_cli_sha256": hashes["runtime_cli_sha256"],
            "runtime_module_sha256": hashes["runtime_module_sha256"],
            "submitter_sha256": hashes["submitter_sha256"],
            "supervisor_sha256": hashes["supervisor_sha256"],
            "semantic_canary_verifier_sha256": hashes[
                "semantic_canary_verifier_sha256"
            ],
            "experiment_plan_validator_sha256": hashes[
                "experiment_plan_validator_sha256"
            ],
            "wrapper_sha256": hashes["wrapper_sha256"],
            "scheduled_preflight_script_sha256": hashes[
                "scheduled_preflight_script_sha256"
            ],
            "official_canary_verifier_sha256": hashes[
                "official_canary_verifier_sha256"
            ],
        },
    }
    atomic_write_json(receipt, value, canonical=True)
    return receipt, submitter.sha256_file(receipt)


def test_environment_contract_and_exact_array_resources(
    tmp_path: Path,
) -> None:
    contract = submitter.load_environment_contract(ENVIRONMENT_CONTRACT)
    assert contract["allocation"]["dossier"] == "AD011016471R1"
    assert contract["walltime"] == {
        "canary": "02:00:00",
        "production": "08:00:00",
    }
    assert contract["arrays"] == {
        "canary": "0-0",
        "canary_task_count": 1,
        "canary_pack_index": 4,
        "canary_entry_indices": [8, 9],
        "production": "0-5%6",
        "production_task_count": 6,
        "logical_entry_count": 12,
        "runs_per_gpu": 2,
        "concurrent_within_pack": True,
        "fixed_entry_packs": [
            [0, 1],
            [2, 3],
            [4, 5],
            [6, 7],
            [8, 9],
            [10, 11],
        ],
    }
    paths = _fixture_repo(tmp_path)
    wrapper_text = (
        REPO_ROOT / "experiments" / submitter.WRAPPER_NAME
    ).read_text(encoding="utf-8")
    assert "#SBATCH --mem=" not in wrapper_text
    assert "#SBATCH --mem-per-cpu=" not in wrapper_text
    assert "#SBATCH --mem-per-gpu=" not in wrapper_text
    canary_command, canary_metadata = submitter.prepare_submission(
        _args(paths), enforce_storage_roots=False
    )
    production_command, production_metadata = submitter.prepare_submission(
        _args(paths, kind="production"), enforce_storage_roots=False
    )
    assert canary_metadata["output_root"] == production_metadata["output_root"]
    assert canary_metadata["output_root"] == str(paths["output"].resolve())
    assert "--array=0-0" in canary_command
    assert "--time=02:00:00" in canary_command
    assert "--array=0-5%6" in production_command
    assert "--time=08:00:00" in production_command
    for command in (canary_command, production_command):
        assert "--account=umg@v100" in command
        assert "--partition=gpu_p13" in command
        assert "--qos=qos_gpu-t3" in command
        assert "--constraint=v100-32g" in command
        assert "--gres=gpu:1" in command
        assert "--cpus-per-task=16" in command
        assert not any(
            item.startswith(("--mem=", "--mem-per-cpu=", "--mem-per-gpu="))
            for item in command
        )
        assert "--hint=nomultithread" in command
        assert (
            f"--output={paths['output'].resolve()}/slurm/%x-%A_%a.out"
            in command
        )
        assert (
            f"--error={paths['output'].resolve()}/slurm/%x-%A_%a.err"
            in command
        )
        export_arg = next(
            item for item in command if item.startswith("--export=ALL,")
        )
        assert "PD_SUCCESSOR_RUNTIME_MODULE_SHA256=" in export_arg
        assert "PD_SUCCESSOR_RUNS_PER_GPU=2" in export_arg
        assert "PD_SUCCESSOR_CONCURRENT_WITHIN_PACK=true" in export_arg
    assert canary_metadata["canary_pack_index"] == 4
    assert canary_metadata["canary_entry_indices"] == [8, 9]
    assert production_metadata["task_count"] == 6
    assert production_metadata["logical_entry_count"] == 12
    assert production_metadata["fixed_entry_packs"] == [
        [0, 1],
        [2, 3],
        [4, 5],
        [6, 7],
        [8, 9],
        [10, 11],
    ]


def test_submitter_rejects_an_arbitrary_file_as_source_archive(
    tmp_path: Path,
) -> None:
    paths = _fixture_repo(tmp_path)
    args = _args(paths)
    args.source_archive = "/etc/hosts"
    with pytest.raises(ValueError, match="tar.gz or .tgz"):
        submitter.prepare_submission(args, enforce_storage_roots=False)


def test_wrapper_preflight_is_side_effect_free_and_skill_receipt_binds_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _fixture_repo(tmp_path)
    _enable_fake_jeanzay_environment(tmp_path, monkeypatch)
    _command, metadata = submitter.prepare_submission(
        _args(paths), enforce_storage_roots=False
    )
    before = _tree_snapshot(tmp_path)
    result = submitter.run_wrapper_preflight(metadata)
    after = _tree_snapshot(tmp_path)
    assert before == after
    assert result["runtime_preflight"]["checks"]["approved_plan"] is True

    receipt_path = tmp_path / "scheduled-receipt.json"
    path, digest = submitter.ensure_scheduled_preflight_receipt(
        receipt_path,
        metadata=metadata,
        runtime_preflight=result["runtime_preflight"],
        create=True,
    )
    receipt = read_json(path)
    assert receipt["schema_version"] == (
        "scheduled-run-preflight-receipt/v1"
    )
    assert receipt["runner"] == str(paths["wrapper"])
    assert receipt["runner_sha256"] == metadata["hashes"]["wrapper_sha256"]
    assert digest == submitter.sha256_file(path)


def test_runtime_preflight_rejects_missing_approval_or_manifest_gate() -> None:
    value = _runtime_preflight()
    value["checks"]["approved_plan"] = False  # type: ignore[index]
    with pytest.raises(RuntimeError, match="approved-plan"):
        submitter._parse_runtime_preflight(
            submitter.RESULT_JSON_MARKER + json.dumps(value)
        )
    value = _runtime_preflight()
    value["checks"]["manifest_valid"] = False  # type: ignore[index]
    with pytest.raises(RuntimeError, match="immutable config/manifest"):
        submitter._parse_runtime_preflight(
            submitter.RESULT_JSON_MARKER + json.dumps(value)
        )


def test_production_live_operation_cannot_bypass_preflight_or_canary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _fixture_repo(tmp_path)
    _enable_fake_jeanzay_environment(tmp_path, monkeypatch)
    args = _args(paths, kind="production", submit=True)
    allocation_called = False

    def forbidden_allocation(**unused: object) -> dict[str, object]:
        nonlocal allocation_called
        allocation_called = True
        raise AssertionError("allocation audit must not run after gate failure")

    with pytest.raises(RuntimeError, match="Standalone submitter --submit"):
        submitter.dispatch(
            args,
            allocation_verifier=forbidden_allocation,
            enforce_storage_roots=False,
        )
    assert allocation_called is False
    assert not any(path.name.endswith("receipt.json") for path in tmp_path.rglob("*"))

    args.submit = False
    args.test_only = True
    with pytest.raises(RuntimeError, match="fail-closed gate inputs"):
        submitter.dispatch(
            args,
            allocation_verifier=forbidden_allocation,
            enforce_storage_roots=False,
        )
    args.test_only = False
    dry = submitter.dispatch(args, enforce_storage_roots=False)
    assert dry["status"] == "dry_run_blocked"
    assert "missing --launch-authorization-receipt" in dry["blockers"]
    assert "missing --launch-authorization-receipt-sha256" in dry["blockers"]
    assert "missing --scheduled-preflight-receipt" in dry["blockers"]
    assert "missing --canary-receipt-sha256" in dry["blockers"]


def test_launch_authorization_rejects_a_hash_only_placeholder_plan(
    tmp_path: Path,
) -> None:
    paths = _fixture_repo(tmp_path)
    authorization, authorization_sha = _write_launch_authorization(paths)
    _command, metadata = submitter.prepare_submission(
        _args(paths), enforce_storage_roots=False
    )
    with pytest.raises(ValueError, match="fenced JSON contract"):
        submitter.validate_launch_authorization_receipt(
            authorization,
            expected_sha256=authorization_sha,
            metadata=metadata,
            runtime_preflight=_runtime_preflight(),
        )


def test_canary_test_only_runs_both_preflights_then_sbatch_test_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _fixture_repo(tmp_path)
    _enable_fake_jeanzay_environment(tmp_path, monkeypatch)
    args = _args(paths, test_only=True)
    authorization, authorization_sha = _write_launch_authorization(paths)
    args.launch_authorization_receipt = str(authorization)
    args.launch_authorization_receipt_sha256 = authorization_sha
    authorization_value = read_json(authorization)
    monkeypatch.setattr(
        submitter,
        "validate_launch_authorization_receipt",
        lambda path, **unused: {
            "receipt_path": str(Path(path).resolve()),
            "receipt_sha256": submitter.sha256_file(path),
            "approved_plan": authorization_value["approved_plan"],
            "receipt": authorization_value,
        },
    )
    args.scheduled_preflight_receipt = str(tmp_path / "scheduled.json")
    args.preflight_receipt = str(tmp_path / "runtime-preflight.json")
    observed: list[list[str]] = []

    def runner(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        observed.append(list(command))
        if command[0] == "sbatch":
            assert "--test-only" in command
            return subprocess.CompletedProcess(
                command, 0, stdout="test-only accepted\n", stderr=""
            )
        return subprocess.run(command, **kwargs)  # type: ignore[arg-type]

    result = submitter.dispatch(
        args,
        runner=runner,
        allocation_verifier=_allocation,
        enforce_storage_roots=False,
    )
    assert result["status"] == "test_only_passed"
    assert result["submitted"] is False
    assert read_json(args.scheduled_preflight_receipt)["schema_version"] == (
        "scheduled-run-preflight-receipt/v1"
    )
    runtime_receipt = read_json(args.preflight_receipt)
    assert runtime_receipt["schema_version"] == (
        "perfectdiode-successor-preflight-receipt/v1"
    )
    assert runtime_receipt["runtime_preflight"]["schema_version"] == (
        "mnist-conv-perfectdiode-successor-preflight/v1"
    )
    binding = runtime_receipt["input_binding"]
    assert binding["scheduled_preflight_receipt_sha256"] == (
        submitter.sha256_file(args.scheduled_preflight_receipt)
    )
    joined = "\n".join(result["command"])
    assert "PD_SUCCESSOR_SCHEDULED_PREFLIGHT_RECEIPT=" in joined
    assert "PD_SUCCESSOR_PREFLIGHT_RECEIPT=" in joined
    assert any(command[0] == "sbatch" for command in observed)


def test_smoke_semantic_receipt_requires_six_tk_rows_and_full_epoch(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    manifest = bundle / "manifest.json"
    config = bundle / "config.json"
    manifest.write_text("{}\n", encoding="utf-8")
    config.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "canary"
    bundle_id = "pdconfirm_" + "1" * 64
    smoke = (
        output
        / "smoke"
        / bundle_id
        / "jobs"
        / "123_456_0"
    )
    smoke.mkdir(parents=True)
    controls = tmp_path / "controls"
    controls.mkdir()
    path_hash_pairs = {
        "source_archive": "source_archive_sha256",
        "environment_contract": "environment_contract_sha256",
        "runtime_cli": "runtime_cli_sha256",
        "runtime_module": "runtime_module_sha256",
        "submitter": "submitter_sha256",
        "supervisor": "supervisor_sha256",
        "experiment_plan_validator": (
            "experiment_plan_validator_sha256"
        ),
        "wrapper": "wrapper_sha256",
        "scheduled_preflight_script": (
            "scheduled_preflight_script_sha256"
        ),
        "official_canary_verifier": (
            "official_canary_verifier_sha256"
        ),
    }
    control_paths: dict[str, str] = {}
    control_hashes: dict[str, str] = {}
    for path_key, hash_key in path_hash_pairs.items():
        path = controls / path_key
        path.write_text(path_key + "\n", encoding="utf-8")
        control_paths[path_key] = str(path)
        control_hashes[hash_key] = submitter.sha256_file(path)
    control_paths["semantic_canary_verifier"] = str(
        REPO_ROOT
        / "experiments"
        / submitter.SEMANTIC_CANARY_VERIFIER_NAME
    )
    control_hashes["semantic_canary_verifier_sha256"] = (
        submitter.sha256_file(
            control_paths["semantic_canary_verifier"]
        )
    )
    plan = controls / "approved-plan.md"
    plan.write_text("approved\n", encoding="utf-8")
    authorization = tmp_path / "launch-authorization.json"
    authorization_payload = {"status": "approved"}
    atomic_write_json(authorization, authorization_payload, canonical=True)
    input_binding = {
        "bundle_id": bundle_id,
        "bundle_manifest_sha256": "2" * 64,
        "config_sha256": "3" * 64,
        "config_file_sha256": "4" * 64,
        "bundle_dir": str(bundle),
        **control_paths,
        "hashes": control_hashes,
        "approved_plan": {
            "path": str(plan),
            "sha256": submitter.sha256_file(plan),
        },
        "launch_authorization_receipt_path": str(authorization),
        "launch_authorization_receipt_sha256": (
            submitter.sha256_file(authorization)
        ),
    }
    preflight = tmp_path / "runtime-preflight.json"
    preflight_payload = {
        "input_binding": input_binding,
        "input_binding_sha256": submitter._canonical_json_sha256(
            input_binding
        ),
    }
    atomic_write_json(preflight, preflight_payload, canonical=True)
    gate_binding = {
        **input_binding,
        "preflight_receipt_path": str(preflight),
        "preflight_receipt_sha256": submitter.sha256_file(preflight),
    }
    rows = []
    generated = []
    for index in range(6):
        result = smoke / "tk_security" / f"row-{index}" / "result.json"
        result.parent.mkdir(parents=True)
        result.write_text(f'{{"row":{index}}}\n', encoding="utf-8")
        generated.append(result)
        architecture = "conv1" if index < 3 else "conv2"
        t = 4 if index < 3 else 6
        rows.append(
            {
                "row_id": f"row-{index}",
                "architecture": architecture,
                "scheme": ("baseline", "ours", "legacy")[index % 3],
                "T": t,
                "K": t,
                "reference_T": 64,
                "reference_K": 64,
                "security_passed": True,
                "fresh_replay": True,
                "result_artifact": _artifact(smoke, result),
            }
        )
    child_specs = (
        {
            "entry_index": 8,
            "entry_id": "pdconfirm_conv2_ours_sgd_seed0",
            "optimizer": "sgd",
            "elapsed_seconds": 100.0,
            "peak_allocated": 1_000_000_000,
            "peak_reserved": 1_100_000_000,
            "process_elapsed_seconds": 105.0,
        },
        {
            "entry_index": 9,
            "entry_id": "pdconfirm_conv2_ours_adam_seed0",
            "optimizer": "adam",
            "elapsed_seconds": 120.0,
            "peak_allocated": 1_200_000_000,
            "peak_reserved": 1_300_000_000,
            "process_elapsed_seconds": 125.0,
        },
    )
    training_canaries = []
    child_processes = []
    timing_children = []
    for spec in child_specs:
        index = spec["entry_index"]
        training_dir = smoke / "training_canaries" / f"entry-{index}"
        training_dir.mkdir(parents=True)
        training_outputs = []
        for name in (
            "best_validation.pt",
            "final.pt",
            "result.json",
            "run_spec.json",
            "step_log.csv",
            "validation.json",
        ):
            path = training_dir / name
            path.write_bytes((name + "\n").encode("utf-8"))
            training_outputs.append(path)
        generated.extend(training_outputs)
        benchmark = {
            "elapsed_seconds": spec["elapsed_seconds"],
            "successful_steps_per_second": (
                3438.0 / spec["elapsed_seconds"]
            ),
            "cuda_peak_memory_allocated_bytes": spec["peak_allocated"],
            "cuda_peak_memory_reserved_bytes": spec["peak_reserved"],
        }
        training_canaries.append(
            {
                "entry_index": index,
                "entry_id": spec["entry_id"],
                "architecture": "conv2",
                "scheme": "ours",
                "optimizer": spec["optimizer"],
                "mode": "successor_canary",
                "status": "complete",
                "training_completed": True,
                "completed_steps": 3438,
                "expected_steps": 3438,
                "epochs_completed": 1,
                "expected_epochs": 1,
                "official_test_read": False,
                "fresh_restart_from_shared_initialization": True,
                "raw_learning_rates_by_parameter": {"weight": 0.001},
                "benchmark": benchmark,
                "output_artifacts": [
                    _artifact(smoke, path)
                    for path in sorted(
                        training_outputs,
                        key=lambda item: item.relative_to(smoke).as_posix(),
                    )
                ],
            }
        )
        child_dir = smoke / "children" / f"entry-{index}"
        child_dir.mkdir(parents=True)
        child_receipt = child_dir / "receipt.json"
        atomic_write_json(
            child_receipt,
            {"entry_index": index, "training_canary": training_canaries[-1]},
            canonical=True,
        )
        child_stderr = child_dir / "stderr.log"
        child_stderr.write_text("", encoding="utf-8")
        terminal = {
            "status": "passed",
            "entry_index": index,
            "entry_id": spec["entry_id"],
            "receipt_path": str(child_receipt),
            "receipt_sha256": submitter.sha256_file(child_receipt),
            "completed_steps": 3438,
            "benchmark": benchmark,
        }
        child_stdout = child_dir / "stdout.log"
        child_stdout.write_text(
            "PD_SUCCESSOR_RESULT_JSON="
            + json.dumps(terminal, sort_keys=True, separators=(",", ":"))
            + "\n",
            encoding="utf-8",
        )
        generated.extend((child_receipt, child_stdout, child_stderr))
        child_processes.append(
            {
                "label": f"canary-entry-{index}",
                "entry_index": index,
                "entry_id": spec["entry_id"],
                "command": [
                    sys.executable,
                    "runtime.py",
                    "run-canary-entry",
                    "--entry-index",
                    str(index),
                ],
                "returncode": 0,
                "process_elapsed_seconds": spec[
                    "process_elapsed_seconds"
                ],
                "terminal_result": terminal,
                "stdout_log": _artifact(smoke, child_stdout),
                "stderr_log": _artifact(smoke, child_stderr),
                "receipt_artifact": _artifact(smoke, child_receipt),
            }
        )
        timing_children.append(
            {
                "entry_index": index,
                "entry_id": spec["entry_id"],
                "completed_steps": 3438,
                "elapsed_seconds": spec["elapsed_seconds"],
                "successful_steps_per_second": (
                    3438.0 / spec["elapsed_seconds"]
                ),
                "cuda_peak_memory_allocated_bytes": spec[
                    "peak_allocated"
                ],
                "cuda_peak_memory_reserved_bytes": spec["peak_reserved"],
                "projected_20_epoch_duration_seconds": (
                    spec["elapsed_seconds"] * 20.0
                ),
            }
        )
    combined_peak = sum(
        int(item["cuda_peak_memory_allocated_bytes"])
        for item in timing_children
    )
    combined_reserved_peak = sum(
        int(item["cuda_peak_memory_reserved_bytes"])
        for item in timing_children
    )
    concurrent_wall = 130.0
    timing_memory_evidence = {
        "thresholds": {
            "gpu_memory_capacity_bytes": 34_359_738_368,
            "maximum_combined_peak_fraction": 0.8,
            "maximum_combined_peak_reserved_bytes": 27_487_790_694,
            "maximum_projected_child_duration_seconds": 28_800,
        },
        "children": timing_children,
        "combined_peak_memory_allocated_bytes": combined_peak,
        "combined_peak_memory_reserved_bytes": combined_reserved_peak,
        "combined_peak_memory_fraction": (
            combined_reserved_peak / 34_359_738_368
        ),
        "concurrent_training_wall_elapsed_seconds": concurrent_wall,
        "aggregate_completed_steps": 6876,
        "aggregate_successful_steps_per_second": 6876.0 / concurrent_wall,
        "maximum_projected_20_epoch_duration_seconds": 2400.0,
        "checks": {
            "child_evidence_positive": True,
            "combined_peak_within_limit": True,
            "projected_duration_below_limit": True,
        },
        "passed": True,
    }
    body = {
        "schema_version": (
            "mnist-conv-perfectdiode-successor-smoke-receipt/v1"
        ),
        "status": "passed",
        "passed": True,
        "official_test_read": False,
        "bundle_id": gate_binding["bundle_id"],
        "bundle_manifest_sha256": gate_binding[
            "bundle_manifest_sha256"
        ],
        "config_sha256": gate_binding["config_sha256"],
        "config_file_sha256": gate_binding["config_file_sha256"],
        "expected_tk_security_count": 6,
        "slurm_identity": {
            "job_id": "456",
            "array_job_id": "123",
            "array_task_id": "0",
            "account": "umg@v100",
            "partition": "gpu_p13",
            "qos": "qos_gpu-t3",
            "constraint": "v100-32g",
        },
        "pack_index": 4,
        "entry_indices": [8, 9],
        "runs_per_gpu": 2,
        "concurrent": True,
        "tk_security": rows,
        "training_canaries": training_canaries,
        "child_processes": child_processes,
        "timing_memory_evidence": timing_memory_evidence,
        "execution_source": {"source_archive_sha256": "5" * 64},
        "preflight_receipt": {
            "source_path": str(preflight),
            "source_file_sha256": submitter.sha256_file(preflight),
            "payload_sha256": submitter._canonical_json_sha256(
                preflight_payload
            ),
            "payload": preflight_payload,
            "launch_authorization": {
                "source_path": str(authorization),
                "source_file_sha256": submitter.sha256_file(
                    authorization
                ),
                "payload_sha256": submitter._canonical_json_sha256(
                    authorization_payload
                ),
                "payload": authorization_payload,
            },
        },
        "input_artifacts": [
            _artifact(bundle, manifest),
            _artifact(bundle, config),
        ],
        "output_artifacts": [
            _artifact(smoke, path)
            for path in sorted(
                generated, key=lambda item: item.relative_to(smoke).as_posix()
            )
        ],
    }
    receipt_value = {
        **body,
        "smoke_id": "pdconfirmsmoke_"
        + submitter._canonical_json_sha256(body),
    }
    receipt = smoke / "receipt.json"
    atomic_write_json(receipt, receipt_value, canonical=True)
    validated = canary.validate_smoke_receipt(
        receipt,
        output_root=output,
        gate_binding=gate_binding,
        expected_array_job_id="123",
    )
    assert len(validated["tk_row_ids"]) == 6
    assert validated["training_canary_entry_indices"] == [8, 9]
    assert validated["training_canary_child_count"] == 2
    assert validated["training_canary_total_steps_completed"] == 6876

    missing_child = json.loads(json.dumps(receipt_value))
    missing_child["child_processes"] = missing_child[
        "child_processes"
    ][:-1]
    missing_child_body = {
        key: item
        for key, item in missing_child.items()
        if key != "smoke_id"
    }
    missing_child["smoke_id"] = (
        "pdconfirmsmoke_"
        + submitter._canonical_json_sha256(missing_child_body)
    )
    atomic_write_json(receipt, missing_child, canonical=True)
    with pytest.raises(ValueError, match="exactly entries 8 and 9"):
        canary.validate_smoke_receipt(
            receipt,
            output_root=output,
            gate_binding=gate_binding,
            expected_array_job_id="123",
        )

    failed_safety = json.loads(json.dumps(receipt_value))
    failed_safety["timing_memory_evidence"]["checks"][
        "combined_peak_within_limit"
    ] = False
    failed_safety["timing_memory_evidence"]["passed"] = False
    failed_safety_body = {
        key: item
        for key, item in failed_safety.items()
        if key != "smoke_id"
    }
    failed_safety["smoke_id"] = (
        "pdconfirmsmoke_"
        + submitter._canonical_json_sha256(failed_safety_body)
    )
    atomic_write_json(receipt, failed_safety, canonical=True)
    with pytest.raises(RuntimeError, match="timing/memory evidence"):
        canary.validate_smoke_receipt(
            receipt,
            output_root=output,
            gate_binding=gate_binding,
            expected_array_job_id="123",
        )

    broken = dict(receipt_value)
    broken["tk_security"] = rows[:-1]
    broken_body = {
        key: item for key, item in broken.items() if key != "smoke_id"
    }
    broken["smoke_id"] = (
        "pdconfirmsmoke_"
        + submitter._canonical_json_sha256(broken_body)
    )
    atomic_write_json(receipt, broken, canonical=True)
    with pytest.raises(ValueError, match="exactly six"):
        canary.validate_smoke_receipt(
            receipt,
            output_root=output,
            gate_binding=gate_binding,
            expected_array_job_id="123",
        )


def test_canary_gate_receipt_must_match_exact_current_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        canary, "_validate_live_gate_files", lambda unused: []
    )
    binding = {"bundle_id": "bundle", "hashes": {"wrapper": "a" * 64}}
    stdout = tmp_path / "canary.out"
    stderr = tmp_path / "canary.err"
    stdout.write_text("{}\n", encoding="utf-8")
    stderr.write_text("", encoding="utf-8")
    value = {
        "schema_version": "perfectdiode-successor-canary-gate/v1",
        "status": "passed",
        "gate_passed": True,
        "gate_binding": binding,
        "gate_binding_sha256": submitter._canonical_json_sha256(binding),
        "official_gate": {"receipt": {"gate_passed": True}},
        "successor_semantics": {
            "training_canary_entry_indices": [8, 9],
            "training_canary_child_count": 2,
            "training_canary_total_steps_completed": 6876,
            "tk_row_ids": [f"row-{index}" for index in range(6)],
            "timing_memory_thresholds": {
                "gpu_memory_capacity_bytes": 34_359_738_368,
                "maximum_combined_peak_fraction": 0.8,
                "maximum_combined_peak_reserved_bytes": 27_487_790_694,
                "maximum_projected_child_duration_seconds": 28_800,
            },
            "timing_memory_evidence": {"passed": True},
        },
        "canary_logs": {
            "stdout": {
                "path": str(stdout),
                "sha256": submitter.sha256_file(stdout),
                "bytes": stdout.stat().st_size,
            },
            "stderr": {
                "path": str(stderr),
                "sha256": submitter.sha256_file(stderr),
                "bytes": stderr.stat().st_size,
            },
            "terminal_result": {},
        },
    }
    path = tmp_path / "canary.json"
    atomic_write_json(path, value, canonical=True)
    assert canary.validate_canary_gate_receipt(
        path, expected_gate_binding=binding
    )["gate_passed"] is True
    with pytest.raises(RuntimeError, match="exact current"):
        canary.validate_canary_gate_receipt(
            path,
            expected_gate_binding={**binding, "bundle_id": "changed"},
        )


def _accounting_row(job: str, state: str, exit_code: str = "0:0") -> str:
    return (
        f"{job}|{job}|{state}|{exit_code}|umg@v100|gpu_p13|"
        "qos_gpu-t3|v100-32g|\n"
    )


def _scontrol_submission_row(
    *,
    job_id: str,
    kind: str,
    repo_root: Path,
    wrapper: Path,
    output_root: Path,
    nodes: str = "1-1",
    num_cpus: str = "16",
    cpus_per_task: str | None = "16",
    requested_cpus: str | None = "16",
    time_limit: str | None = None,
    stdout_path: str | None = None,
) -> str:
    array = "0" if kind == "canary" else "0-5%6"
    throttle = "" if kind == "canary" else " ArrayTaskThrottle=6"
    walltime = time_limit or (
        "02:00:00" if kind == "canary" else "08:00:00"
    )
    name = f"pd-successor-{kind}"
    stdout_value = stdout_path or str(
        output_root / "slurm" / f"{name}-{job_id}_%a.out"
    )
    stderr_value = str(
        output_root / "slurm" / f"{name}-{job_id}_%a.err"
    )
    cpus_per_task_field = (
        ""
        if cpus_per_task is None
        else f" CPUs/Task={cpus_per_task}"
    )
    req_tres = (
        "gres/gpu=1"
        if requested_cpus is None
        else f"cpu={requested_cpus},gres/gpu=1"
    )
    return (
        f"JobId={job_id} ArrayJobId={job_id} ArrayTaskId={array}"
        f"{throttle} Account=umg@v100 Partition=gpu_p13 "
        "QOS=qos_gpu-t3 Features=v100-32g "
        f"NumNodes={nodes} NumTasks=1 NumCPUs={num_cpus}"
        f"{cpus_per_task_field} "
        "ReqB:S:C:T=0:0:*:1 "
        f"ReqTRES={req_tres} "
        f"TimeLimit={walltime} WorkDir={repo_root} Command={wrapper} "
        f"StdOut={stdout_value} StdErr={stderr_value}\n"
    )


def test_strict_array_accounting_accepts_exact_rows_and_rejects_drift() -> None:
    def complete_runner(
        command: list[str], **unused: object
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                _accounting_row("123_0", "COMPLETED")
                + _accounting_row("123_1", "COMPLETED")
            ),
            stderr="",
        )

    result = supervisor.query_successor_array_state(
        "123", 2, runner=complete_runner
    )
    assert result["status"] == "complete"
    assert result["completion_basis"] == (
        "all_indexed_tasks_without_parent_row"
    )

    def duplicate_runner(
        command: list[str], **unused: object
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                _accounting_row("123_0", "COMPLETED")
                + _accounting_row("123_0", "COMPLETED")
            ),
            stderr="",
        )

    with pytest.raises(RuntimeError, match="duplicate"):
        supervisor.query_successor_array_state(
            "123", 2, runner=duplicate_runner
        )

    def wrong_account_runner(
        command: list[str], **unused: object
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                "123_0|123_0|COMPLETED|0:0|fmu@v100|gpu_p13|"
                "qos_gpu-t3|v100-32g|\n"
            ),
            stderr="",
        )

    with pytest.raises(RuntimeError, match="scheduler contract"):
        supervisor.query_successor_array_state(
            "123", 1, runner=wrong_account_runner
        )


def test_supervisor_ambiguous_submission_never_retries(
    tmp_path: Path,
) -> None:
    args = argparse.Namespace(
        enable_submit=True,
        state=str(tmp_path / "state.json"),
    )

    def ambiguous(
        command: list[str], **unused: object
    ) -> subprocess.CompletedProcess[str]:
        if "--test-only" in command:
            return subprocess.CompletedProcess(
                command, 0, stdout="accepted\n", stderr=""
            )
        raise KeyboardInterrupt

    value = supervisor.SuccessorJeanZaySupervisor(
        args=args,
        runner=ambiguous,
        allocation_verifier=_allocation,
        tracker_validator=_tracker_gate,
    )
    value.state_path = tmp_path / "state.json"
    value.state = {
        "binding": {"gate_binding_sha256": "a" * 64},
        "stages": {
            "canary": {"status": "ready"},
            "production": {"status": "blocked"},
        }
    }
    value.gate_binding = {"bundle_id": "bundle"}
    with pytest.raises(KeyboardInterrupt):
        value._submit_once("canary", ["sbatch", "wrapper.slurm"])
    assert value.state["stages"]["canary"]["submission_status"] == "ambiguous"

    called = False

    def forbidden(
        command: list[str], **unused: object
    ) -> subprocess.CompletedProcess[str]:
        nonlocal called
        called = True
        raise AssertionError(command)

    value.runner = forbidden
    with pytest.raises(RuntimeError, match="duplicate submission"):
        value._submit_once("canary", ["sbatch", "wrapper.slurm"])
    assert called is False


def test_failed_sbatch_test_only_is_terminal_and_never_retried(
    tmp_path: Path,
) -> None:
    calls: list[list[str]] = []

    def rejected(
        command: list[str], **unused: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(list(command))
        raise subprocess.CalledProcessError(
            2,
            command,
            output="",
            stderr="invalid request",
        )

    value = supervisor.SuccessorJeanZaySupervisor(
        args=argparse.Namespace(
            enable_submit=True,
            state=str(tmp_path / "state.json"),
        ),
        runner=rejected,
        allocation_verifier=_allocation,
        tracker_validator=_tracker_gate,
    )
    value.state = {
        "status": "active",
        "binding": {"gate_binding_sha256": "a" * 64},
        "stages": {
            "canary": {"status": "ready"},
            "production": {"status": "blocked"},
        },
    }
    value.gate_binding = {"bundle_id": "bundle"}
    command = ["sbatch", "--array=0-0", "worker.slurm"]
    with pytest.raises(subprocess.CalledProcessError):
        value._submit_once("canary", command)
    assert len(calls) == 1
    assert value.state["stages"]["canary"]["status"] == (
        "failed_sbatch_test_only"
    )

    with pytest.raises(RuntimeError, match="failed stage"):
        value._submit_once("canary", command)
    assert len(calls) == 1


def test_blocked_tracker_prevents_allocation_audit_and_sbatch(
    tmp_path: Path,
) -> None:
    runner_called = False
    allocation_called = False

    def forbidden_runner(
        command: list[str], **unused: object
    ) -> subprocess.CompletedProcess[str]:
        nonlocal runner_called
        runner_called = True
        raise AssertionError(command)

    def forbidden_allocation(**unused: object) -> dict[str, object]:
        nonlocal allocation_called
        allocation_called = True
        raise AssertionError("allocation audit must not run")

    value = supervisor.SuccessorJeanZaySupervisor(
        args=argparse.Namespace(
            enable_submit=True,
            state=str(tmp_path / "state.json"),
        ),
        runner=forbidden_runner,
        allocation_verifier=forbidden_allocation,
        tracker_validator=lambda: {
            "status": "blocked",
            "experiment_id": supervisor.EXPERIMENT_ID,
        },
    )
    value.state = {
        "status": "active",
        "binding": {"gate_binding_sha256": "a" * 64},
        "stages": {
            "canary": {"status": "ready"},
            "production": {"status": "blocked"},
        },
    }
    value.gate_binding = {"bundle_id": "bundle"}
    with pytest.raises(ValueError, match="preflighting"):
        value._submit_once(
            "canary",
            ["sbatch", "--array=0-0", "worker.slurm"],
        )
    assert runner_called is False
    assert allocation_called is False


def test_tracker_is_rechecked_after_test_only_before_live_sbatch(
    tmp_path: Path,
) -> None:
    observed: list[list[str]] = []
    statuses = iter(("preflighting", "blocked"))

    def runner(
        command: list[str], **unused: object
    ) -> subprocess.CompletedProcess[str]:
        observed.append(list(command))
        if "--test-only" not in command:
            raise AssertionError("live sbatch must remain unreachable")
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="accepted\n",
            stderr="",
        )

    value = supervisor.SuccessorJeanZaySupervisor(
        args=argparse.Namespace(
            enable_submit=True,
            state=str(tmp_path / "state.json"),
        ),
        runner=runner,
        allocation_verifier=_allocation,
        tracker_validator=lambda: {
            "status": next(statuses),
            "experiment_id": supervisor.EXPERIMENT_ID,
        },
    )
    value.state = {
        "status": "active",
        "binding": {"gate_binding_sha256": "a" * 64},
        "stages": {
            "canary": {"status": "ready"},
            "production": {"status": "blocked_on_canary"},
        },
    }
    value.gate_binding = {"bundle_id": "bundle"}
    with pytest.raises(ValueError, match="preflighting"):
        value._submit_once(
            "canary",
            ["sbatch", "--array=0-0", "worker.slurm"],
        )
    assert len(observed) == 1
    assert "--test-only" in observed[0]
    assert value.state["stages"]["canary"].get(
        "submission_status"
    ) is None


def test_failure_report_is_first_write_wins_and_complete(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "state.json"
    report_path = tmp_path / "failure.json"
    state = {
        "status": "active",
        "binding": {
            "preflight_receipt_path": str(tmp_path / "preflight.json"),
            "preflight_receipt_sha256": "a" * 64,
        },
        "last_tracker_gate": {
            "receipt_path": str(tmp_path / "tracker-canary.json"),
            "receipt_sha256": "b" * 64,
            "tracker_path": str(tmp_path / "current-experiments.md"),
            "tracker_sha256": "c" * 64,
        },
        "stages": {
            "canary": {
                "status": "failed_accounting_validation",
                "job_id": "311193",
                "last_array_state": {"status": "failed"},
                "command": ["sbatch", "worker.slurm"],
                "live_submission_tracker_gate": {
                    "receipt_path": str(
                        tmp_path / "tracker-canary-live.json"
                    ),
                    "receipt_sha256": "d" * 64,
                },
            },
            "production": {"status": "blocked_on_canary"},
        },
    }
    first = supervisor.write_failure_report(
        report_path,
        error=RuntimeError("allocation mismatch"),
        state_path=state_path,
        state=state,
    )
    first_hash = submitter.sha256_file(report_path)
    second = supervisor.write_failure_report(
        report_path,
        error=RuntimeError("must not replace first failure"),
        state_path=state_path,
        state=state,
    )
    assert second == first
    assert submitter.sha256_file(report_path) == first_hash
    assert first["stage"] == "canary"
    assert first["launched_job_count"] == 1
    assert first["launched_jobs"][0]["job_id"] == "311193"
    assert first["completed_stages"] == []
    assert first["last_valid_artifacts"]["preflight_receipt_sha256"] == (
        "a" * 64
    )
    assert first["last_valid_artifacts"][
        "last_tracker_gate.receipt_sha256"
    ] == "b" * 64
    assert first["last_valid_artifacts"][
        "canary.live_submission_tracker_gate.receipt_sha256"
    ] == "d" * 64
    assert first["retry_authorized"] is False
    assert "wait for a new user message" in first["smallest_next_action"]


def test_failure_report_is_first_write_wins_under_concurrency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "failure.json"
    state_path = tmp_path / "state.json"
    barrier = threading.Barrier(2)
    real_atomic_write_json = supervisor.atomic_write_json

    def synchronized_write(
        path: str | Path,
        value: object,
        *,
        canonical: bool = False,
    ) -> Path:
        barrier.wait(timeout=5)
        return real_atomic_write_json(path, value, canonical=canonical)

    monkeypatch.setattr(
        supervisor,
        "atomic_write_json",
        synchronized_write,
    )

    def write(message: str) -> dict[str, object]:
        return supervisor.write_failure_report(
            destination,
            error=RuntimeError(message),
            state_path=state_path,
            state=None,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(write, "first contender"),
            executor.submit(write, "second contender"),
        ]
        reports = [future.result(timeout=10) for future in futures]
    persisted = read_json(destination)
    assert reports[0] == reports[1] == persisted
    assert persisted["failure"]["message"] in {
        "first contender",
        "second contender",
    }


def test_timeout_failure_report_preserves_command_and_streams(
    tmp_path: Path,
) -> None:
    error = subprocess.TimeoutExpired(
        ["ssh", "jean-zay"],
        17,
        output=b"partial stdout\xff",
        stderr=b"partial stderr",
    )
    report = supervisor.write_failure_report(
        tmp_path / "failure.json",
        error=error,
        state_path=tmp_path / "state.json",
        state=None,
    )
    assert report["failure"]["error_type"] == "TimeoutExpired"
    assert report["failure"]["command"] == ["ssh", "jean-zay"]
    assert report["failure"]["timeout_seconds"] == 17
    assert report["failure"]["stdout"].startswith("partial stdout")
    assert report["failure"]["stderr"] == "partial stderr"
    json.dumps(report, allow_nan=False)


def test_ambiguous_submission_report_never_claims_zero_jobs(
    tmp_path: Path,
) -> None:
    report = supervisor.write_failure_report(
        tmp_path / "failure.json",
        error=subprocess.TimeoutExpired(["sbatch", "worker.slurm"], 30),
        state_path=tmp_path / "state.json",
        state={
            "status": "active",
            "binding": {},
            "stages": {
                "canary": {
                    "status": "ready",
                    "submission_status": "ambiguous",
                    "command": ["sbatch", "worker.slurm"],
                },
                "production": {"status": "blocked_on_canary"},
            },
        },
        scheduler_readback={
            "attempted": False,
            "jobs": {},
            "ambiguous_submissions": [
                {
                    "stage": "canary",
                    "reason": "sbatch may have created an unknown job",
                }
            ],
        },
    )
    assert report["launched_job_count"] is None
    assert report["launched_job_count_status"] == (
        "unknown_due_to_ambiguous_submission"
    )


def test_dangling_failure_report_symlink_is_rejected(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "failure.json"
    target = tmp_path / "elsewhere.json"
    report_path.symlink_to(target)
    with pytest.raises(ValueError, match="non-symlink"):
        supervisor.write_failure_report(
            report_path,
            error=RuntimeError("must not follow"),
            state_path=tmp_path / "state.json",
            state=None,
        )
    assert not target.exists()


def test_record_failure_reads_live_scheduler_state_and_worker_log(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "output"
    log_root = output_root / "slurm"
    log_root.mkdir(parents=True)
    stderr_log = log_root / "pd-successor-canary-311705_0.err"
    stderr_log.write_text(
        "archive check failed: [Errno 28] No space left on device\n",
        encoding="utf-8",
    )
    tracker_gate = tmp_path / "tracker-canary-gate.json"
    tracker_gate.write_text('{"status":"preflighting"}\n', encoding="utf-8")

    def runner(
        command: list[str], **unused: object
    ) -> subprocess.CompletedProcess[str]:
        assert command[:5] == ["sacct", "-n", "-X", "-j", "311705"]
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="311705|311705|FAILED|1:0|\n",
            stderr="",
        )

    value = supervisor.SuccessorJeanZaySupervisor(
        args=argparse.Namespace(
            state=str(tmp_path / "state.json"),
            output_root=str(output_root),
            tracker_gate_receipt=str(tracker_gate),
        ),
        runner=runner,
    )
    value.state = {
        "status": "active",
        "binding": {},
        "stages": {
            "canary": {
                "status": "failed_accounting_validation",
                "job_id": "311705",
            },
            "production": {"status": "blocked_on_canary"},
        },
    }
    value._note_failure_operation(
        name="scheduler_accounting",
        stage="canary",
        command=["sacct", "-j", "311705"],
    )
    report = value.record_failure(
        RuntimeError("canary task entered FAILED")
    )
    readback = report["launched_jobs"][0]["failure_readback"]
    assert readback["rows"][0]["state"] == "FAILED"
    assert report["failure_operation"]["name"] == "scheduler_accounting"
    assert report["failure"]["command"] == [
        "sacct",
        "-j",
        "311705",
    ]
    assert any(
        "No space left on device" in item.get("tail", "")
        for item in report["diagnostic_evidence"]["slurm_logs"]
    )
    observed_tracker = report["files_or_external_state_changed"][
        "observed_attempt_paths"
    ]["tracker_gate_receipt"]
    assert observed_tracker["path"] == str(tracker_gate)
    assert observed_tracker["sha256"] == submitter.sha256_file(tracker_gate)
    assert read_json(value.state_path)["status"] == "failed"


def test_record_failure_recovers_existing_state_before_reporting(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "state.json"
    atomic_write_json(
        state_path,
        {
            "status": "active",
            "binding": {},
            "stages": {
                "canary": {
                    "status": "waiting",
                    "job_id": "8123",
                },
                "production": {"status": "blocked_on_canary"},
            },
        },
        canonical=True,
    )

    value = supervisor.SuccessorJeanZaySupervisor(
        args=argparse.Namespace(
            state=str(state_path),
            output_root=str(tmp_path / "output"),
        ),
        runner=lambda command, **unused: subprocess.CompletedProcess(
            command,
            0,
            stdout="8123|8123|RUNNING|0:0|\n",
            stderr="",
        ),
    )
    report = value.record_failure(TimeoutError("preflight timed out"))
    assert report["launched_job_count"] == 1
    assert report["launched_jobs"][0]["job_id"] == "8123"
    assert read_json(state_path)["status"] == "failed"


def test_existing_failure_report_blocks_restart_before_any_command(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "state.json"
    failure_path = tmp_path / "state-failure.json"
    supervisor.write_failure_report(
        failure_path,
        error=RuntimeError("first failure"),
        state_path=state_path,
        state=None,
    )
    called = False

    def forbidden(
        command: list[str], **unused: object
    ) -> subprocess.CompletedProcess[str]:
        nonlocal called
        called = True
        raise AssertionError(command)

    value = supervisor.SuccessorJeanZaySupervisor(
        args=argparse.Namespace(
            state=str(state_path),
        ),
        runner=forbidden,
    )
    with pytest.raises(RuntimeError, match="new immutable attempt path"):
        value.initialize()
    assert called is False


def test_failed_state_without_report_still_blocks_restart(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "state.json"
    atomic_write_json(
        state_path,
        {
            "schema_version": supervisor.STATE_SCHEMA_VERSION,
            "status": "failed",
            "binding": {},
            "stages": {
                "canary": {"status": "failed"},
                "production": {"status": "blocked_on_canary"},
            },
        },
        canonical=True,
    )
    value = supervisor.SuccessorJeanZaySupervisor(
        args=argparse.Namespace(state=str(state_path)),
    )
    with pytest.raises(RuntimeError, match="never to resume"):
        value.initialize()
    assert not value.failure_report_path.exists()


@pytest.mark.parametrize("occupied_kind", ["directory", "dangling_symlink"])
def test_invalid_occupied_failure_report_path_blocks_before_any_command(
    tmp_path: Path,
    occupied_kind: str,
) -> None:
    state_path = tmp_path / "state.json"
    failure_path = tmp_path / "state-failure.json"
    if occupied_kind == "directory":
        failure_path.mkdir()
    else:
        failure_path.symlink_to(tmp_path / "missing-failure.json")
    runner_called = False
    allocation_called = False

    def forbidden_runner(
        command: list[str], **unused: object
    ) -> subprocess.CompletedProcess[str]:
        nonlocal runner_called
        runner_called = True
        raise AssertionError(command)

    def forbidden_allocation(**unused: object) -> dict[str, object]:
        nonlocal allocation_called
        allocation_called = True
        raise AssertionError("allocation audit must not run")

    value = supervisor.SuccessorJeanZaySupervisor(
        args=argparse.Namespace(state=str(state_path)),
        runner=forbidden_runner,
        allocation_verifier=forbidden_allocation,
    )
    with pytest.raises(ValueError, match="non-symlink regular JSON file"):
        value.initialize()
    assert runner_called is False
    assert allocation_called is False


@pytest.mark.parametrize("occupied_kind", ["directory", "dangling_symlink"])
def test_public_main_rejects_invalid_failure_path_before_arming_or_commands(
    tmp_path: Path,
    occupied_kind: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    state_path = tmp_path / "state.json"
    failure_path = tmp_path / "state-failure.json"
    if occupied_kind == "directory":
        failure_path.mkdir()
    else:
        failure_path.symlink_to(tmp_path / "missing-failure.json")
    runner_called = False
    initialize_called = False

    def forbidden_runner(
        command: list[str], **unused: object
    ) -> subprocess.CompletedProcess[str]:
        nonlocal runner_called
        runner_called = True
        raise AssertionError(command)

    def forbidden_initialize(
        unused_self: supervisor.SuccessorJeanZaySupervisor,
    ) -> None:
        nonlocal initialize_called
        initialize_called = True
        raise AssertionError("initialize must not run")

    monkeypatch.setattr(supervisor, "_run_process_group", forbidden_runner)
    monkeypatch.setattr(
        supervisor.SuccessorJeanZaySupervisor,
        "initialize",
        forbidden_initialize,
    )
    required = [
        "--state",
        str(state_path),
        "--bundle-dir",
        str(tmp_path / "bundle"),
        "--output-root",
        str(tmp_path / "output"),
        "--data-root",
        str(tmp_path / "data"),
        "--source-archive",
        str(tmp_path / "source.tar.gz"),
        "--environment-contract",
        str(tmp_path / "environment.json"),
        "--launch-authorization-receipt",
        str(tmp_path / "authorization.json"),
        "--launch-authorization-receipt-sha256",
        "a" * 64,
        "--scheduled-preflight-receipt",
        str(tmp_path / "scheduled.json"),
        "--preflight-receipt",
        str(tmp_path / "preflight.json"),
        "--tracker-gate-receipt",
        str(tmp_path / "tracker-gate.json"),
        "--tracker-gate-receipt-sha256",
        "b" * 64,
        "--official-canary-receipt",
        str(tmp_path / "official.json"),
        "--canary-receipt",
        str(tmp_path / "canary.json"),
        "--arm-long-run",
    ]

    with pytest.raises(ValueError, match="non-symlink regular JSON file"):
        supervisor.main(required)

    assert "LONG-RUN ATTEMPT ARMED" not in capsys.readouterr().err
    assert runner_called is False
    assert initialize_called is False
    if occupied_kind == "directory":
        assert failure_path.is_dir()
    else:
        assert failure_path.is_symlink()


def test_lock_contender_does_not_poison_active_attempt(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    state_path = tmp_path / "state.json"
    lock_path = state_path.with_name(state_path.name + ".lock")
    required = [
        "--state",
        str(state_path),
        "--bundle-dir",
        str(tmp_path / "bundle"),
        "--output-root",
        str(tmp_path / "output"),
        "--data-root",
        str(tmp_path / "data"),
        "--source-archive",
        str(tmp_path / "source.tar.gz"),
        "--environment-contract",
        str(tmp_path / "environment.json"),
        "--launch-authorization-receipt",
        str(tmp_path / "authorization.json"),
        "--launch-authorization-receipt-sha256",
        "a" * 64,
        "--scheduled-preflight-receipt",
        str(tmp_path / "scheduled.json"),
        "--preflight-receipt",
        str(tmp_path / "preflight.json"),
        "--tracker-gate-receipt",
        str(tmp_path / "tracker-gate.json"),
        "--tracker-gate-receipt-sha256",
        "b" * 64,
        "--official-canary-receipt",
        str(tmp_path / "official.json"),
        "--canary-receipt",
        str(tmp_path / "canary.json"),
        "--arm-long-run",
    ]
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="exactly one active"):
            supervisor.main(required)
    assert "LONG-RUN ATTEMPT ARMED" not in capsys.readouterr().err
    assert not state_path.exists()
    assert not (tmp_path / "state-failure.json").exists()


def test_supervisor_commands_receive_a_hard_timeout(
    tmp_path: Path,
) -> None:
    observed: dict[str, object] = {}

    def runner(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        observed.update(kwargs)
        return subprocess.CompletedProcess(command, 0, "", "")

    value = supervisor.SuccessorJeanZaySupervisor(
        args=argparse.Namespace(
            state=str(tmp_path / "state.json"),
            command_timeout_seconds=17,
        ),
        runner=runner,
    )
    value.runner(["probe"], check=True)
    assert observed["timeout"] == 17.0


def test_follow_deadline_limits_each_external_command(
    tmp_path: Path,
) -> None:
    now = [10.0]
    observed_timeouts: list[float] = []

    def runner(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        timeout = float(kwargs["timeout"])
        observed_timeouts.append(timeout)
        now[0] += timeout
        return subprocess.CompletedProcess(command, 0, "", "")

    value = supervisor.SuccessorJeanZaySupervisor(
        args=argparse.Namespace(
            state=str(tmp_path / "state.json"),
            command_timeout_seconds=60,
        ),
        runner=runner,
        clock=lambda: now[0],
    )
    value.state = {
        "status": "active",
        "binding": {},
        "stages": {
            "canary": {"status": "waiting"},
            "production": {"status": "blocked_on_canary"},
        },
    }
    value.gate_binding = {"bundle_id": "bundle"}

    def advance() -> dict[str, object]:
        value.runner(["slow-command"], check=True)
        return {"status": "waiting", "stage": "canary"}

    value.advance_once = advance  # type: ignore[method-assign]
    with pytest.raises(TimeoutError, match="terminal progress"):
        value.run(
            poll_seconds=1,
            once=False,
            max_follow_seconds=2,
        )
    assert observed_timeouts == [2.0]


def test_supervisor_rejects_command_timeout_above_observability_interval(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="observability interval"):
        supervisor.SuccessorJeanZaySupervisor(
            args=argparse.Namespace(
                state=str(tmp_path / "state.json"),
                command_timeout_seconds=(
                    supervisor.MAX_ARMED_COMMAND_TIMEOUT_SECONDS + 1
                ),
            ),
        )


@pytest.mark.skipif(
    not hasattr(os, "killpg"),
    reason="requires POSIX process groups",
)
def test_supervisor_timeout_kills_term_ignoring_descendant_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descendant_pid_path = tmp_path / "descendant.pid"
    descendant_ready_path = tmp_path / "descendant.ready"
    descendant_code = (
        "import pathlib, signal, sys, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "pathlib.Path(sys.argv[1]).write_text('ready', encoding='utf-8')\n"
        "time.sleep(60)\n"
    )
    leader_code = (
        "import pathlib, signal, subprocess, sys, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "pid_path = pathlib.Path(sys.argv[1])\n"
        "ready_path = pathlib.Path(sys.argv[2])\n"
        "child = subprocess.Popen("
        "[sys.executable, '-c', sys.argv[3], str(ready_path)])\n"
        "deadline = time.monotonic() + 5\n"
        "while not ready_path.is_file() and time.monotonic() < deadline:\n"
        "    time.sleep(0.01)\n"
        "pid_path.write_text(str(child.pid), encoding='utf-8')\n"
        "time.sleep(60)\n"
    )
    monkeypatch.setattr(
        supervisor,
        "PROCESS_GROUP_TERMINATE_SECONDS",
        0.2,
    )
    value = supervisor.SuccessorJeanZaySupervisor(
        args=argparse.Namespace(
            state=str(tmp_path / "state.json"),
            command_timeout_seconds=1,
        ),
    )
    def process_is_live(process_id: int) -> bool:
        try:
            fields = Path(
                f"/proc/{process_id}/stat"
            ).read_text(encoding="utf-8").split()
        except FileNotFoundError:
            return False
        return len(fields) > 2 and fields[2] != "Z"
    descendant_pid: int | None = None
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            value.runner(
                [
                    sys.executable,
                    "-c",
                    leader_code,
                    str(descendant_pid_path),
                    str(descendant_ready_path),
                    descendant_code,
                ],
                check=True,
                text=True,
                capture_output=True,
            )
        descendant_pid = int(
            descendant_pid_path.read_text(encoding="utf-8")
        )
        deadline = time.monotonic() + 5
        while (
            process_is_live(descendant_pid)
            and time.monotonic() < deadline
        ):
            time.sleep(0.02)
        assert not process_is_live(descendant_pid)
    finally:
        if descendant_pid is not None and process_is_live(descendant_pid):
            os.kill(descendant_pid, signal.SIGKILL)


@pytest.mark.skipif(
    not hasattr(os, "killpg"),
    reason="requires POSIX process groups",
)
def test_supervisor_timeout_cleanup_is_bounded_when_escaped_child_keeps_pipe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    escaped_pid_path = tmp_path / "escaped.pid"
    escaped_code = (
        "import pathlib, signal, sys, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "pathlib.Path(sys.argv[1]).write_text(str(__import__('os').getpid()))\n"
        "time.sleep(60)\n"
    )
    leader_code = (
        "import pathlib, subprocess, sys, time\n"
        "pid_path = pathlib.Path(sys.argv[1])\n"
        "subprocess.Popen("
        "[sys.executable, '-c', sys.argv[2], str(pid_path)], "
        "start_new_session=True)\n"
        "deadline = time.monotonic() + 5\n"
        "while not pid_path.is_file() and time.monotonic() < deadline:\n"
        "    time.sleep(0.01)\n"
        "time.sleep(60)\n"
    )
    monkeypatch.setattr(
        supervisor,
        "PROCESS_GROUP_TERMINATE_SECONDS",
        0.1,
    )
    value = supervisor.SuccessorJeanZaySupervisor(
        args=argparse.Namespace(
            state=str(tmp_path / "state.json"),
            command_timeout_seconds=1,
        ),
    )
    started = time.monotonic()
    escaped_pid: int | None = None
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            value.runner(
                [
                    sys.executable,
                    "-c",
                    leader_code,
                    str(escaped_pid_path),
                    escaped_code,
                ],
                timeout=0.2,
                check=True,
                text=True,
                capture_output=True,
            )
        assert time.monotonic() - started < 2
        escaped_pid = int(escaped_pid_path.read_text(encoding="utf-8"))
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            try:
                fields = Path(
                    f"/proc/{escaped_pid}/stat"
                ).read_text(encoding="utf-8").split()
            except FileNotFoundError:
                break
            if len(fields) > 2 and fields[2] == "Z":
                break
            time.sleep(0.02)
        else:
            pytest.fail("escaped descendant remained live after cleanup")
    finally:
        if escaped_pid is not None:
            try:
                os.kill(escaped_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.skipif(
    not hasattr(os, "killpg"),
    reason="requires POSIX process groups",
)
def test_supervisor_rejects_and_kills_successful_background_descendant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child_pid_path = tmp_path / "background.pid"
    child_code = (
        "import os, pathlib, signal, sys, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "pathlib.Path(sys.argv[1]).write_text(str(os.getpid()))\n"
        "time.sleep(60)\n"
    )
    leader_code = (
        "import pathlib, subprocess, sys, time\n"
        "pid_path = pathlib.Path(sys.argv[1])\n"
        "subprocess.Popen("
        "[sys.executable, '-c', sys.argv[2], str(pid_path)], "
        "stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, "
        "stderr=subprocess.DEVNULL)\n"
        "deadline = time.monotonic() + 5\n"
        "while not pid_path.is_file() and time.monotonic() < deadline:\n"
        "    time.sleep(0.01)\n"
    )
    monkeypatch.setattr(
        supervisor,
        "PROCESS_GROUP_TERMINATE_SECONDS",
        0.1,
    )
    value = supervisor.SuccessorJeanZaySupervisor(
        args=argparse.Namespace(
            state=str(tmp_path / "state.json"),
            command_timeout_seconds=2,
        ),
    )
    child_pid: int | None = None
    try:
        with pytest.raises(RuntimeError, match="background descendants"):
            value.runner(
                [
                    sys.executable,
                    "-c",
                    leader_code,
                    str(child_pid_path),
                    child_code,
                ],
                check=True,
                text=True,
                capture_output=True,
            )
        child_pid = int(child_pid_path.read_text(encoding="utf-8"))
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            try:
                fields = Path(
                    f"/proc/{child_pid}/stat"
                ).read_text(encoding="utf-8").split()
            except FileNotFoundError:
                break
            if len(fields) > 2 and fields[2] == "Z":
                break
            time.sleep(0.02)
        else:
            pytest.fail("background descendant remained live after cleanup")
    finally:
        if child_pid is not None:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_supervisor_defaults_to_one_pass_and_bounds_explicit_follow() -> None:
    parser = supervisor._parser()
    required = [
        "--state",
        "/tmp/state.json",
        "--bundle-dir",
        "/tmp/bundle",
        "--output-root",
        "/tmp/output",
        "--data-root",
        "/tmp/data",
        "--source-archive",
        "/tmp/source.tar.gz",
        "--environment-contract",
        "/tmp/environment.json",
        "--launch-authorization-receipt",
        "/tmp/authorization.json",
        "--launch-authorization-receipt-sha256",
        "a" * 64,
        "--scheduled-preflight-receipt",
        "/tmp/scheduled.json",
        "--preflight-receipt",
        "/tmp/preflight.json",
        "--tracker-gate-receipt",
        "/tmp/tracker-gate.json",
        "--tracker-gate-receipt-sha256",
        "b" * 64,
        "--official-canary-receipt",
        "/tmp/official.json",
        "--canary-receipt",
        "/tmp/canary.json",
    ]
    assert parser.parse_args(required).once is True
    assert parser.parse_args([*required, "--follow"]).once is False
    assert parser.parse_args(required).max_follow_seconds == (
        supervisor.DEFAULT_MAX_FOLLOW_SECONDS
    )
    assert parser.parse_args(required).official_verifier is None
    assert parser.parse_args(required).arm_long_run is False


def test_supervisor_arm_marker_follows_lock_and_reports_actual_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = tmp_path / "state.json"
    events: list[str] = []

    class MarkerStream:
        def __init__(self) -> None:
            self.text = ""

        def write(self, value: str) -> int:
            if "LONG-RUN ATTEMPT ARMED" in value:
                assert "lock" in events
                events.append("arm")
            self.text += value
            return len(value)

        def flush(self) -> None:
            pass

    class FakeSupervisor:
        def __init__(self, *, args: argparse.Namespace):
            self.args = args
            self.failure_report_path = state.with_name("state-failure.json")

        def _reject_terminal_attempt(self) -> None:
            assert "lock" in events
            events.append("precheck")

        def initialize(self) -> None:
            events.append("initialize")

        def run(self, **kwargs: object) -> dict[str, object]:
            events.append("run")
            assert kwargs["max_follow_seconds"] == 123.0
            return {"status": "waiting"}

    real_flock = supervisor.fcntl.flock

    def observed_flock(descriptor: int, operation: int) -> None:
        real_flock(descriptor, operation)
        events.append("lock")

    stream = MarkerStream()
    monkeypatch.setattr(supervisor.fcntl, "flock", observed_flock)
    monkeypatch.setattr(
        supervisor,
        "SuccessorJeanZaySupervisor",
        FakeSupervisor,
    )
    monkeypatch.setattr(supervisor.sys, "stderr", stream)
    required = [
        "--state",
        str(state),
        "--bundle-dir",
        str(tmp_path / "bundle"),
        "--output-root",
        str(tmp_path / "output"),
        "--data-root",
        str(tmp_path / "data"),
        "--source-archive",
        str(tmp_path / "source.tar.gz"),
        "--environment-contract",
        str(tmp_path / "environment.json"),
        "--launch-authorization-receipt",
        str(tmp_path / "authorization.json"),
        "--launch-authorization-receipt-sha256",
        "a" * 64,
        "--scheduled-preflight-receipt",
        str(tmp_path / "scheduled.json"),
        "--preflight-receipt",
        str(tmp_path / "preflight.json"),
        "--tracker-gate-receipt",
        str(tmp_path / "tracker-gate.json"),
        "--tracker-gate-receipt-sha256",
        "b" * 64,
        "--official-canary-receipt",
        str(tmp_path / "official.json"),
        "--canary-receipt",
        str(tmp_path / "canary.json"),
        "--max-follow-seconds",
        "123",
        "--arm-long-run",
    ]

    assert supervisor.main(required) == 0
    assert events == ["lock", "precheck", "arm", "initialize", "run"]
    assert "hard_deadline_seconds=123" in stream.text


def test_supervisor_rejects_follow_beyond_declared_hard_deadline(
    tmp_path: Path,
) -> None:
    value = supervisor.SuccessorJeanZaySupervisor(
        args=argparse.Namespace(
            state=str(tmp_path / "state.json"),
            command_timeout_seconds=1,
        ),
    )
    with pytest.raises(ValueError, match="no greater than"):
        value.run(
            poll_seconds=1,
            once=False,
            max_follow_seconds=(
                supervisor.LONG_RUN_HARD_DEADLINE_SECONDS + 1
            ),
        )


def test_supervisor_requires_explicit_arm_before_state_mutation(
    tmp_path: Path,
) -> None:
    state = tmp_path / "state.json"
    required = [
        "--state",
        str(state),
        "--bundle-dir",
        str(tmp_path / "bundle"),
        "--output-root",
        str(tmp_path / "output"),
        "--data-root",
        str(tmp_path / "data"),
        "--source-archive",
        str(tmp_path / "source.tar.gz"),
        "--environment-contract",
        str(tmp_path / "environment.json"),
        "--launch-authorization-receipt",
        str(tmp_path / "authorization.json"),
        "--launch-authorization-receipt-sha256",
        "a" * 64,
        "--scheduled-preflight-receipt",
        str(tmp_path / "scheduled.json"),
        "--preflight-receipt",
        str(tmp_path / "preflight.json"),
        "--tracker-gate-receipt",
        str(tmp_path / "tracker-gate.json"),
        "--tracker-gate-receipt-sha256",
        "b" * 64,
        "--official-canary-receipt",
        str(tmp_path / "official.json"),
        "--canary-receipt",
        str(tmp_path / "canary.json"),
    ]

    with pytest.raises(ValueError, match="--arm-long-run"):
        supervisor.main(required)

    assert not state.exists()
    assert not state.with_name("state-failure.json").exists()


def test_approved_plan_launcher_covers_every_required_supervisor_argument() -> None:
    metadata: dict[str, object] = {
        "resources": {"python_executable": "/opt/reviewed/bin/python"},
        "supervisor": "/staging/repo/experiments/supervisor.py",
        "supervisor_state": "/results/attempt/state.json",
        "bundle_dir": "/results/bundle",
        "output_root": "/results/attempt/output",
        "data_root": "/datasets/mnist",
        "source_archive": "/staging/source.tar.gz",
        "environment_contract": "/staging/environment.json",
        "launch_authorization_receipt": "/results/attempt/authorization.json",
        "scheduled_preflight_receipt": "/results/attempt/scheduled.json",
        "preflight_receipt": "/results/attempt/preflight.json",
        "official_canary_receipt": "/results/attempt/official-canary.json",
        "successor_canary_receipt": "/results/attempt/canary.json",
        "repo_root": "/staging/repo",
        "remote_user": "testuser",
        "official_canary_verifier": (
            "/staging/repo/launch_tools/verify_canary.py"
        ),
        "canary_pack_index": 4,
    }
    launcher = submitter._expected_plan_supervisor_launcher(metadata)
    actualized = [
        (
            "/results/attempt/tracker-canary.json"
            if value
            == submitter.TRACKER_GATE_RECEIPT_PATH_PLACEHOLDER
            else "a" * 64
            if value
            == submitter.TRACKER_GATE_RECEIPT_SHA256_PLACEHOLDER
            else "b" * 64
            if value
            == submitter.LAUNCH_AUTHORIZATION_SHA256_PLACEHOLDER
            else value
        )
        for value in launcher
    ]
    parsed = supervisor._parser().parse_args(actualized[2:])
    assert parsed.state == metadata["supervisor_state"]
    assert parsed.tracker_gate_receipt == (
        "/results/attempt/tracker-canary.json"
    )
    assert parsed.tracker_gate_receipt_sha256 == "a" * 64
    assert parsed.arm_long_run is True
    assert parsed.enable_submit is True


def test_explicit_follow_stops_at_hard_deadline(
    tmp_path: Path,
) -> None:
    now = [0.0]
    sleeps: list[float] = []
    value = supervisor.SuccessorJeanZaySupervisor(
        args=argparse.Namespace(state=str(tmp_path / "state.json")),
        clock=lambda: now[0],
        sleeper=lambda seconds: (
            sleeps.append(seconds),
            now.__setitem__(0, now[0] + seconds),
        )[-1],
    )
    value.state = {
        "status": "active",
        "binding": {},
        "stages": {
            "canary": {"status": "waiting"},
            "production": {"status": "blocked_on_canary"},
        },
    }
    value.gate_binding = {"bundle_id": "bundle"}
    value.advance_once = lambda: {"status": "waiting", "stage": "canary"}  # type: ignore[method-assign]
    with pytest.raises(TimeoutError, match="terminal progress"):
        value.run(
            poll_seconds=1,
            once=False,
            max_follow_seconds=2,
        )
    assert sleeps == [1.0, 1.0]


def test_follow_stops_at_canary_stage_boundary() -> None:
    value = supervisor.SuccessorJeanZaySupervisor(
        args=argparse.Namespace(state="/tmp/stage-boundary-state.json"),
    )
    calls = 0

    def advance() -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {"status": "stage_complete", "stage": "canary"}

    value.advance_once = advance  # type: ignore[method-assign]
    result = value.run(
        poll_seconds=1,
        once=False,
        max_follow_seconds=10,
    )
    assert result == {"status": "stage_complete", "stage": "canary"}
    assert calls == 1


def test_supervisor_runs_fresh_test_only_before_both_submissions(
    tmp_path: Path,
) -> None:
    args = argparse.Namespace(
        enable_submit=True,
        state=str(tmp_path / "state.json"),
    )
    observed: list[list[str]] = []
    job_ids = iter(("7001\n", "7002\n"))

    def runner(
        command: list[str], **unused: object
    ) -> subprocess.CompletedProcess[str]:
        observed.append(list(command))
        if command[0] == "scontrol":
            job_id = command[3]
            kind = "canary" if job_id == "7001" else "production"
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=_scontrol_submission_row(
                    job_id=job_id,
                    kind=kind,
                    repo_root=tmp_path / "repo",
                    wrapper=tmp_path / "repo" / "worker.slurm",
                    output_root=tmp_path / "output",
                ),
                stderr="",
            )
        if "--test-only" in command:
            return subprocess.CompletedProcess(
                command, 0, stdout="test accepted\n", stderr=""
            )
        return subprocess.CompletedProcess(
            command, 0, stdout=next(job_ids), stderr=""
        )

    value = supervisor.SuccessorJeanZaySupervisor(
        args=args,
        runner=runner,
        allocation_verifier=_allocation,
        tracker_validator=(
            lambda statuses=iter(
                (
                    "preflighting",
                    "preflighting",
                    "launch-ready",
                    "launch-ready",
                )
            ): {
                "status": next(statuses),
                "experiment_id": supervisor.EXPERIMENT_ID,
            }
        ),
    )
    value.state_path = tmp_path / "state.json"
    value.state = {
        "binding": {"gate_binding_sha256": "b" * 64},
        "stages": {
            "canary": {"status": "ready"},
            "production": {"status": "ready"},
        },
    }
    value.gate_binding = {
        "bundle_id": "bundle",
        "repo_root": str(tmp_path / "repo"),
        "wrapper": str(tmp_path / "repo" / "worker.slurm"),
        "launch_contract": {
            "resources": {"account": "umg@v100"},
            "output_root": str(tmp_path / "output"),
        },
    }
    canary_command = ["sbatch", "--array=0-0", "worker.slurm"]
    production_command = [
        "sbatch",
        "--array=0-5%6",
        "worker.slurm",
    ]
    assert value._submit_once("canary", canary_command) == "7001"
    assert value._submit_once("production", production_command) == "7002"
    assert [command for command in observed if command[0] == "sbatch"] == [
        ["sbatch", "--test-only", "--array=0-0", "worker.slurm"],
        canary_command,
        [
            "sbatch",
            "--test-only",
            "--array=0-5%6",
            "worker.slurm",
        ],
        production_command,
    ]
    assert [command for command in observed if command[0] == "scontrol"] == [
        ["scontrol", "show", "job", "7001", "-o"],
        ["scontrol", "show", "job", "7002", "-o"],
    ]
    for stage in ("canary", "production"):
        record = value.state["stages"][stage]
        assert len(record["test_only_attempts"]) == 1
        assert record["last_test_only"]["allocation_audit"]["verified"] is True


def test_supervisor_initialization_creates_shared_slurm_log_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _fixture_repo(tmp_path)
    _enable_fake_jeanzay_environment(tmp_path, monkeypatch)
    authorization, authorization_sha = _write_launch_authorization(paths)
    args = _args(paths)
    args.state = str(tmp_path / "supervisor-state.json")
    args.output_root = str(paths["output"])
    args.scheduled_preflight_receipt = str(tmp_path / "scheduled.json")
    args.preflight_receipt = str(tmp_path / "preflight.json")
    args.launch_authorization_receipt = str(authorization)
    args.launch_authorization_receipt_sha256 = authorization_sha
    args.enable_submit = False

    original_prepare = submitter.prepare_submission
    monkeypatch.setattr(
        supervisor,
        "prepare_submission",
        lambda value: original_prepare(
            value,
            enforce_storage_roots=False,
        ),
    )
    authorization_payload = read_json(authorization)
    monkeypatch.setattr(
        supervisor,
        "validate_launch_authorization_receipt",
        lambda path, **unused: {
            "receipt_path": str(Path(path).resolve()),
            "receipt_sha256": submitter.sha256_file(path),
            "approved_plan": authorization_payload["approved_plan"],
            "receipt": authorization_payload,
        },
    )
    value = supervisor.SuccessorJeanZaySupervisor(
        args=args,
        allocation_verifier=_allocation,
    )
    value.initialize()
    log_dir = paths["output"] / "slurm"
    assert log_dir.is_dir()
    assert value.state["binding"]["output_root"] == str(
        paths["output"].resolve()
    )
    assert "tracker_gate" not in value.state["binding"]
    assert value.state["last_tracker_gate"]["status"] == "preflighting"
    for command in (
        value.state["binding"]["canary_command"],
        value.state["binding"]["production_command"],
    ):
        assert any(
            item.startswith(f"--output={log_dir}/")
            for item in command
        )
        assert any(
            item.startswith(f"--error={log_dir}/")
            for item in command
        )


def test_immediate_scontrol_readback_binds_pending_resources_and_paths(
    tmp_path: Path,
) -> None:
    repo = (tmp_path / "staged-repo").resolve()
    wrapper = repo / "experiments" / "worker.slurm"
    output = (tmp_path / "output").resolve()
    metadata = {
        "resources": {
            "host_memory_policy": (
                "jean_zay_site_managed_no_explicit_slurm_request"
            )
        },
        "repo_root": str(repo),
        "wrapper": str(wrapper),
        "output_root": str(output),
    }

    def accepted(
        command: list[str], **unused: object
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=_scontrol_submission_row(
                job_id="8123",
                kind="production",
                repo_root=repo,
                wrapper=wrapper,
                output_root=output,
                nodes="1-1",
            ),
            stderr="",
        )

    result = submitter.verify_submitted_job_contract(
        job_id="8123",
        kind="production",
        metadata=metadata,
        runner=accepted,
    )
    assert result["array_indices"] == list(range(6))
    assert result["array_throttle"] == 6
    assert result["time_limit"] == "08:00:00"
    assert result["work_dir"] == str(repo)
    assert result["command_path"] == str(wrapper)

    running_result = submitter.verify_submitted_job_contract(
        job_id="8123",
        kind="production",
        metadata=metadata,
        runner=lambda command, **unused: subprocess.CompletedProcess(
            command,
            0,
            stdout=_scontrol_submission_row(
                job_id="8123",
                kind="production",
                repo_root=repo,
                wrapper=wrapper,
                output_root=output,
                nodes="1",
                num_cpus="32",
            ),
            stderr="",
        ),
    )
    assert running_result["array_indices"] == list(range(6))

    drift_rows = {
        "allocated_cpus": _scontrol_submission_row(
            job_id="8123",
            kind="production",
            repo_root=repo,
            wrapper=wrapper,
            output_root=output,
            num_cpus="64",
        ),
        "requested_cpus": _scontrol_submission_row(
            job_id="8123",
            kind="production",
            repo_root=repo,
            wrapper=wrapper,
            output_root=output,
            requested_cpus="32",
        ),
        "missing_requested_cpus": _scontrol_submission_row(
            job_id="8123",
            kind="production",
            repo_root=repo,
            wrapper=wrapper,
            output_root=output,
            requested_cpus=None,
        ),
        "cpus_per_task": _scontrol_submission_row(
            job_id="8123",
            kind="production",
            repo_root=repo,
            wrapper=wrapper,
            output_root=output,
            cpus_per_task="32",
        ),
        "missing_cpus_per_task": _scontrol_submission_row(
            job_id="8123",
            kind="production",
            repo_root=repo,
            wrapper=wrapper,
            output_root=output,
            cpus_per_task=None,
        ),
        "walltime": _scontrol_submission_row(
            job_id="8123",
            kind="production",
            repo_root=repo,
            wrapper=wrapper,
            output_root=output,
            time_limit="05:59:59",
        ),
        "workdir": _scontrol_submission_row(
            job_id="8123",
            kind="production",
            repo_root=tmp_path / "wrong-repo",
            wrapper=wrapper,
            output_root=output,
        ),
        "command": _scontrol_submission_row(
            job_id="8123",
            kind="production",
            repo_root=repo,
            wrapper=tmp_path / "wrong.slurm",
            output_root=output,
        ),
        "stdout": _scontrol_submission_row(
            job_id="8123",
            kind="production",
            repo_root=repo,
            wrapper=wrapper,
            output_root=output,
            stdout_path=str(tmp_path / "wrong.out"),
        ),
    }
    for label, row in drift_rows.items():
        with pytest.raises(RuntimeError, match="scontrol readback"):
            submitter.verify_submitted_job_contract(
                job_id="8123",
                kind="production",
                metadata=metadata,
                runner=lambda command, row=row, **unused: (
                    subprocess.CompletedProcess(
                        command, 0, stdout=row, stderr=""
                    )
                ),
            )


def test_supervisor_resume_reaudits_submitted_unverified_job(
    tmp_path: Path,
) -> None:
    repo = (tmp_path / "repo").resolve()
    wrapper = repo / "worker.slurm"
    output = (tmp_path / "output").resolve()
    observed: list[list[str]] = []

    def runner(
        command: list[str], **unused: object
    ) -> subprocess.CompletedProcess[str]:
        observed.append(list(command))
        if command[0] != "scontrol":
            raise AssertionError(f"unexpected duplicate submission: {command}")
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=_scontrol_submission_row(
                job_id="9001",
                kind="canary",
                repo_root=repo,
                wrapper=wrapper,
                output_root=output,
            ),
            stderr="",
        )

    value = supervisor.SuccessorJeanZaySupervisor(
        args=argparse.Namespace(
            enable_submit=True,
            state=str(tmp_path / "state.json"),
        ),
        runner=runner,
        allocation_verifier=_allocation,
        tracker_validator=_tracker_gate,
    )
    value.state = {
        "binding": {"gate_binding_sha256": "a" * 64},
        "stages": {
            "canary": {
                "status": "waiting",
                "job_id": "9001",
                "submission_status": "submitted_unverified",
            },
            "production": {"status": "blocked"},
        },
    }
    value.gate_binding = {
        "repo_root": str(repo),
        "wrapper": str(wrapper),
        "launch_contract": {
            "resources": {
                "host_memory_policy": (
                    "jean_zay_site_managed_no_explicit_slurm_request"
                )
            },
            "output_root": str(output),
        },
    }
    assert value._submit_once("canary", ["sbatch", "worker.slurm"]) == "9001"
    record = value.state["stages"]["canary"]
    assert record["submission_status"] == "submitted"
    assert record["post_submit_readback"]["array_indices"] == [0]
    assert observed == [["scontrol", "show", "job", "9001", "-o"]]


def test_canary_log_contract_accepts_one_terminal_marker_and_fails_closed(
    tmp_path: Path,
) -> None:
    receipt = tmp_path / "receipt.json"
    atomic_write_json(receipt, {"status": "passed"}, canonical=True)
    stdout = tmp_path / "canary.out"
    stderr = tmp_path / "canary.err"
    terminal = {
        "status": "passed",
        "receipt_path": str(receipt.resolve()),
        "receipt_sha256": submitter.sha256_file(receipt),
    }
    stdout.write_text(
        "ordinary progress\n"
        + submitter.RESULT_JSON_MARKER
        + json.dumps(terminal, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    stderr.write_text("non-fatal warning\n", encoding="utf-8")
    logs = canary.inspect_canary_logs(
        stdout_log=stdout,
        stderr_log=stderr,
        smoke_receipt=receipt,
    )
    assert logs["terminal_result"] == terminal

    stdout.write_text(
        submitter.RESULT_JSON_MARKER
        + json.dumps(terminal)
        + "\n"
        + submitter.RESULT_JSON_MARKER
        + json.dumps(terminal)
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="exactly one terminal"):
        canary.inspect_canary_logs(
            stdout_log=stdout,
            stderr_log=stderr,
            smoke_receipt=receipt,
        )

    stdout.write_text(
        submitter.RESULT_JSON_MARKER + json.dumps(terminal) + "\n",
        encoding="utf-8",
    )
    stderr.write_text("RuntimeError: worker failed\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="fatal diagnostic"):
        canary.inspect_canary_logs(
            stdout_log=stdout,
            stderr_log=stderr,
            smoke_receipt=receipt,
        )


def test_accounting_regression_matrix_and_scontrol_fallback() -> None:
    singleton_parent_raw = (
        "123|123_0|COMPLETED|0:0|umg@v100|gpu_p13|"
        "qos_gpu-t3|v100-32g|\n"
    )
    singleton = supervisor.query_successor_array_state(
        "123",
        1,
        runner=lambda command, **unused: subprocess.CompletedProcess(
            command, 0, stdout=singleton_parent_raw, stderr=""
        ),
    )
    assert singleton["status"] == "complete"
    assert singleton["task_states"] == {"0": "COMPLETED"}

    cases = {
        "missing": (
            _accounting_row("123", "COMPLETED")
            + _accounting_row("123_0", "COMPLETED"),
            "missing indices",
            2,
        ),
        "failed": (
            _accounting_row("123_0", "FAILED", "1:0"),
            "terminal failure",
            1,
        ),
        "mixed_compressed": (
            _accounting_row("123_[0-1]", "RUNNING")
            + _accounting_row("123_[2-3]", "PENDING"),
            "consistent states",
            4,
        ),
        "unexpected": (
            _accounting_row("123_2", "RUNNING"),
            "unexpected indices",
            2,
        ),
        "compressed_failed": (
            _accounting_row("123_[0-3]", "FAILED", "1:0"),
            "compressed successor",
            4,
        ),
    }
    for stdout, message, count in cases.values():
        with pytest.raises(RuntimeError, match=message):
            supervisor.query_successor_array_state(
                "123",
                count,
                runner=lambda command, stdout=stdout, **unused: (
                    subprocess.CompletedProcess(
                        command, 0, stdout=stdout, stderr=""
                    )
                ),
            )

    calls = 0

    def fallback(
        command: list[str], **unused: object
    ) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        calls += 1
        if command[0] == "sacct":
            raise subprocess.CalledProcessError(1, command)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                "JobId=123_0 ArrayTaskId=0 JobState=COMPLETED "
                "ExitCode=0:0 Account=umg@v100 Partition=gpu_p13 "
                "QOS=qos_gpu-t3 Features=v100-32g\n"
            ),
            stderr="",
        )

    fallback_result = supervisor.query_successor_array_state(
        "123", 1, runner=fallback
    )
    assert calls == 2
    assert fallback_result["status"] == "complete"
    assert fallback_result["accounting_source"] == "scontrol_fallback"


def test_completion_binds_full_preflight_and_canary_receipt_payloads(
    tmp_path: Path,
) -> None:
    output = (tmp_path / "output").resolve()
    entry_dir = output / "entries" / "entry-0"
    entry_dir.mkdir(parents=True)
    artifact = entry_dir / "result.json"
    artifact.write_text('{"status":"complete"}\n', encoding="utf-8")
    best = entry_dir / "best_validation.pt"
    final = entry_dir / "final.pt"
    best.write_bytes(b"best\n")
    final.write_bytes(b"final\n")
    preflight = tmp_path / "preflight.json"
    authorization = tmp_path / "authorization.json"
    canary_receipt = tmp_path / "canary.json"
    atomic_write_json(preflight, {"preflight": "passed"}, canonical=True)
    atomic_write_json(
        authorization, {"launch_authorized": True}, canonical=True
    )
    atomic_write_json(
        canary_receipt, {"gate_passed": True}, canonical=True
    )
    preflight_payload = read_json(preflight)
    authorization_payload = read_json(authorization)
    canary_payload = read_json(canary_receipt)
    expected_identity = {
        "bundle_id": "bundle",
        "bundle_manifest_sha256": "1" * 64,
        "config_sha256": "2" * 64,
        "config_file_sha256": "3" * 64,
        "preflight_receipt_path": str(preflight.resolve()),
        "preflight_receipt_sha256": submitter.sha256_file(preflight),
        "launch_authorization_receipt_path": str(authorization.resolve()),
        "launch_authorization_receipt_sha256": submitter.sha256_file(
            authorization
        ),
        "canary_gate_receipt_path": str(canary_receipt.resolve()),
        "canary_gate_receipt_sha256": submitter.sha256_file(canary_receipt),
    }
    embedded_preflight = {
        "source_path": str(preflight.resolve()),
        "source_file_sha256": submitter.sha256_file(preflight),
        "payload_sha256": submitter._canonical_json_sha256(
            preflight_payload
        ),
        "payload": preflight_payload,
        "launch_authorization": {
            "source_path": str(authorization.resolve()),
            "source_file_sha256": submitter.sha256_file(authorization),
            "payload_sha256": submitter._canonical_json_sha256(
                authorization_payload
            ),
            "payload": authorization_payload,
        },
    }
    embedded_canary = {
        "source_path": str(canary_receipt.resolve()),
        "source_file_sha256": submitter.sha256_file(canary_receipt),
        "payload_sha256": submitter._canonical_json_sha256(canary_payload),
        "payload": canary_payload,
    }
    completion = {
        "schema_version": supervisor.COMPLETION_SCHEMA_VERSION,
        "state": "complete",
        "official_test_read": False,
        "bundle_id": "bundle",
        "bundle_manifest_sha256": "1" * 64,
        "config_sha256": "2" * 64,
        "config_file_sha256": "3" * 64,
        "entry_index": 0,
        "entry_id": "entry-0",
        "entry_payload_sha256": "4" * 64,
        "execution_source": {"source_archive_sha256": "5" * 64},
        "preflight_receipt": embedded_preflight,
        "canary_gate_receipt": embedded_canary,
        "outputs": [
            {
                "path": str(artifact),
                "sha256": submitter.sha256_file(artifact),
                "bytes": artifact.stat().st_size,
            }
        ],
    }
    completion_path = entry_dir / "completion.json"
    atomic_write_json(completion_path, completion, canonical=True)
    validated = supervisor._validate_entry_completion(
        completion_path,
        output_root=output,
        status_entry={"entry_index": 0, "entry_id": "entry-0"},
        expected_identity=expected_identity,
    )
    assert validated["outputs"][0]["path"] == str(artifact)
    status_entries = []
    for index in range(12):
        current_id = "entry-0" if index == 0 else f"entry-{index}"
        current_dir = output / "entries" / current_id
        status_entries.append(
            {
                "entry_index": index,
                "entry_id": current_id,
                "status": "complete" if index == 0 else "pending",
                "mode": "successor_confirmation",
                "optimizer_steps_completed": 34380 if index == 0 else 0,
                "completion_path": str(current_dir / "completion.json"),
                "result_path": str(current_dir / "result.json"),
                "checkpoint_paths": [
                    str(current_dir / "best_validation.pt"),
                    str(current_dir / "final.pt"),
                ],
            }
        )
    partial = {
        "schema_version": supervisor.STATUS_SCHEMA_VERSION,
        "status": "partial",
        "official_test_read": False,
        "bundle_id": "bundle",
        "bundle_manifest_sha256": "1" * 64,
        "config_sha256": "2" * 64,
        "config_file_sha256": "3" * 64,
        "expected_entry_count": 12,
        "completed_entry_count": 1,
        "pending_entry_count": 11,
        "entries": status_entries,
        "errors": [],
    }
    partial_validated = supervisor.validate_partial_production_status(
        partial,
        output_root=output,
        expected_identity=expected_identity,
    )
    assert len(partial_validated["completion_markers"]) == 1

    completion["canary_gate_receipt"] = {
        **embedded_canary,
        "source_file_sha256": "f" * 64,
    }
    atomic_write_json(completion_path, completion, canonical=True)
    with pytest.raises(RuntimeError, match="exact verified canary-gate"):
        supervisor.validate_partial_production_status(
            partial,
            output_root=output,
            expected_identity=expected_identity,
        )


def test_supervisor_status_consumes_one_terminal_machine_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    gate_receipt = tmp_path / "canary-gate.json"
    atomic_write_json(gate_receipt, {"gate_passed": True}, canonical=True)
    payload = {"schema_version": supervisor.STATUS_SCHEMA_VERSION}
    stdout = submitter.RESULT_JSON_MARKER + json.dumps(payload) + "\n"

    value = supervisor.SuccessorJeanZaySupervisor(
        args=argparse.Namespace(
            state=str(tmp_path / "state.json"),
            python=sys.executable,
            repo_root=str(tmp_path),
            bundle_dir=str(tmp_path / "bundle"),
            output_root=str(output),
        ),
        runner=lambda command, **unused: subprocess.CompletedProcess(
            command, 0, stdout=stdout, stderr=""
        ),
    )
    value.state_path = tmp_path / "state.json"
    value.state = {
        "binding": {},
        "stages": {
            "canary": {
                "gate_receipt_path": str(gate_receipt),
                "gate_receipt_sha256": submitter.sha256_file(gate_receipt),
            },
            "production": {},
        },
    }
    value.gate_binding = {"bundle_id": "bundle"}
    monkeypatch.setattr(
        supervisor,
        "validate_canary_gate_receipt",
        lambda path, *, expected_gate_binding: read_json(path),
    )
    monkeypatch.setattr(
        supervisor,
        "validate_production_status",
        lambda observed, **unused: {
            "status": observed,
            "completion_markers": [],
        },
    )
    monkeypatch.setattr(
        value,
        "_persist_first_production_artifact",
        lambda **unused: {},
    )
    assert value._run_production_status() == payload

    value.runner = lambda command, **unused: subprocess.CompletedProcess(
        command, 0, stdout=stdout + stdout, stderr=""
    )
    with pytest.raises(RuntimeError, match="exactly one terminal"):
        value._run_production_status()


def test_all_finished_status_preserves_identity_for_first_artifact_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = {
        "bundle_id": "bundle",
        "bundle_manifest_sha256": "1" * 64,
        "config_sha256": "2" * 64,
        "config_file_sha256": "3" * 64,
    }
    entries = [
        {
            "entry_index": index,
            "entry_id": f"entry-{index}",
            "status": "complete",
            "mode": "successor_confirmation",
            "optimizer_steps_completed": (
                34380 if index < 6 else 68760
            ),
            "completion_path": f"entries/entry-{index}/completion.json",
            "result_path": f"entries/entry-{index}/result.json",
            "checkpoint_paths": [
                f"entries/entry-{index}/best_validation.pt",
                f"entries/entry-{index}/final.pt",
            ],
        }
        for index in range(12)
    ]
    status = {
        "schema_version": supervisor.STATUS_SCHEMA_VERSION,
        "status": "complete",
        "official_test_read": False,
        **identity,
        "expected_entry_count": 12,
        "completed_entry_count": 12,
        "pending_entry_count": 0,
        "entries": entries,
        "errors": [],
    }
    monkeypatch.setattr(
        supervisor,
        "_artifact_under_root",
        lambda raw_path, root, label: root / str(raw_path),
    )
    monkeypatch.setattr(
        supervisor,
        "_validate_entry_completion",
        lambda path, **unused: {
            "completion_path": str(path),
            "completion_sha256": "4" * 64,
            "outputs": [],
        },
    )
    validated = supervisor.validate_production_status(
        status,
        output_root=tmp_path / "output",
        expected_identity=identity,
    )
    assert [
        (row["entry_index"], row["entry_id"])
        for row in validated["completion_markers"]
    ] == [(index, f"entry-{index}") for index in range(12)]

    value = supervisor.SuccessorJeanZaySupervisor(
        args=argparse.Namespace(
            state=str(tmp_path / "state.json"),
        ),
    )
    value.state = {
        "binding": {},
        "stages": {"canary": {}, "production": {}},
    }
    value.gate_binding = {"bundle_id": "bundle"}
    evidence = value._persist_first_production_artifact(
        value=status,
        completions=validated["completion_markers"],
    )
    assert evidence["entry_index"] == 0
    assert evidence["entry_id"] == "entry-0"
    assert value.state["stages"]["production"][
        "first_production_artifact_verified"
    ] == evidence


def test_waiting_production_persists_first_artifact_and_invalid_is_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = supervisor.SuccessorJeanZaySupervisor(
        args=argparse.Namespace(
            state=str(tmp_path / "state.json"),
            output_root=str(tmp_path / "output"),
            enable_submit=True,
        ),
    )
    value.state = {
        "binding": {},
        "stages": {
            "canary": {"status": "complete"},
            "production": {
                "status": "waiting",
                "job_id": "42",
                "submission_status": "submitted",
            },
        },
    }
    value.gate_binding = {"bundle_id": "bundle"}
    completion = (
        tmp_path
        / "output"
        / "entries"
        / "entry-0"
        / "completion.json"
    )
    completion.parent.mkdir(parents=True)
    completion.write_text("{}\n", encoding="utf-8")
    first = {
        "entry_index": 0,
        "entry_id": "entry-0",
        "completion_path": str(completion),
        "completion_sha256": submitter.sha256_file(completion),
        "outputs": [],
    }
    monkeypatch.setattr(
        value,
        "_fetch_production_status",
        lambda: {"status": "partial"},
    )
    monkeypatch.setattr(
        value,
        "_current_completion_identity",
        lambda: {"bundle_id": "bundle"},
    )
    monkeypatch.setattr(
        supervisor,
        "validate_partial_production_status",
        lambda observed, **unused: {
            "status": observed,
            "completion_markers": [first],
        },
    )
    evidence = value._inspect_first_production_artifact()
    assert evidence["entry_id"] == "entry-0"
    assert value.state["stages"]["production"][
        "first_production_artifact_verified"
    ] == evidence
    assert submitter.sha256_file(evidence["status_path"]) == evidence[
        "status_sha256"
    ]

    value.state["stages"]["production"].pop(
        "first_production_artifact_verified"
    )
    monkeypatch.setattr(
        value,
        "_revalidate_before_production",
        lambda: None,
    )
    monkeypatch.setattr(
        value,
        "_submit_once",
        lambda stage, command: "42",
    )
    monkeypatch.setattr(
        value,
        "_poll_stage",
        lambda *unused, **kwargs: {"status": "waiting"},
    )
    monkeypatch.setattr(
        value,
        "_inspect_first_production_artifact",
        lambda: (_ for _ in ()).throw(RuntimeError("invalid partial output")),
    )
    with pytest.raises(RuntimeError, match="invalid partial output"):
        value._advance_production()
    assert value.state["stages"]["production"]["status"] == (
        "failed_first_artifact_validation"
    )
    with pytest.raises(RuntimeError, match="authoritative reconciliation"):
        value._advance_production()


def test_wrapper_can_derive_optional_slurm_fields_and_hashes_canary_gate() -> None:
    wrapper = (
        REPO_ROOT
        / "experiments"
        / submitter.WRAPPER_NAME
    ).read_text(encoding="utf-8")
    assert (
        "for optional_name in SLURM_JOB_ACCOUNT SLURM_JOB_PARTITION "
        "SLURM_JOB_QOS"
    ) in wrapper
    assert '${!optional_name:-}' in wrapper
    assert 'export SLURM_JOB_ACCOUNT="${live_account}"' in wrapper
    assert 'require_env "PD_SUCCESSOR_CANARY_RECEIPT"' in wrapper
    assert (
        '"successor canary-gate receipt" \\\n'
        '    "${PD_SUCCESSOR_CANARY_RECEIPT}" \\\n'
        '    "${PD_SUCCESSOR_CANARY_RECEIPT_SHA256}"'
    ) in wrapper
    assert "canary:smoke-pack|production:run-pack" in wrapper
    assert '--pack-index "${pack_index}"' in wrapper
    assert (
        '"${SLURM_ARRAY_TASK_COUNT}:${SLURM_ARRAY_TASK_MIN}:'
        '${SLURM_ARRAY_TASK_MAX}" != "6:0:5"'
    ) in wrapper
    assert (
        "PD_SUCCESSOR_PRODUCTION_PACK_COUNT}:"
        '${PD_SUCCESSOR_LOGICAL_ENTRY_COUNT}:'
        '${PD_SUCCESSOR_RUNS_PER_GPU}:'
        '${PD_SUCCESSOR_CONCURRENT_WITHIN_PACK}" != "6:12:2:true"'
    ) in wrapper
    assert (
        '"${live_thread_count}:${live_nodes}:${live_tasks}:'
        '${live_cpus_per_task}:${live_requested_cpus}" != '
        '"1:1:1:16:16"'
    ) in wrapper
    assert (
        '[[ "${live_cpus}" != "16" && "${live_cpus}" != "32" ]]'
        in wrapper
    )
    assert (
        '[[ -z "${live_cpus_per_task}" && "${live_req_tres}"'
        not in wrapper
    )
    assert '  JOBSCRATCH' in wrapper
    assert (
        'expected_jobscratch_prefix="/lustre/fsn1/jobscratch_hpe/'
        '${USER}_"'
    ) in wrapper
    assert '! -w "${JOBSCRATCH}"' in wrapper
    assert 'export TMPDIR="${resolved_jobscratch}"' in wrapper
    assert wrapper.index('exec "${runtime_preflight[@]}"') < wrapper.index(
        'expected_jobscratch_prefix="/lustre/fsn1/jobscratch_hpe/'
    )
    assert wrapper.index('export TMPDIR="${resolved_jobscratch}"') < (
        wrapper.index('exec "${runtime_args[@]}"')
    )
