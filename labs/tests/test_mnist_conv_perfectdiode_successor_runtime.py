from __future__ import annotations

import copy
import json
import sys
import tarfile
import tempfile
from pathlib import Path

import pytest

import experiments.mnist_conv.perfectdiode_successor_confirmation as successor
import experiments.run_mnist_conv_perfectdiode_successor_confirmation as successor_cli
from experiments.mnist_conv.identity import code_fingerprint, sha256_file, sha256_json
from experiments.mnist_conv.io import atomic_write_json, read_json


REPO_ROOT = Path(__file__).resolve().parents[2]
RICH_CONFIG = (
    REPO_ROOT
    / "configs"
    / "conv"
    / "perfectdiode_conv12_best_observed_confirmation_20260727_v1.json"
)


def _mini_checkout(root: Path) -> None:
    package = root / "experiments" / "mnist_conv"
    package.mkdir(parents=True)
    (package / "identity.py").write_text("IDENTITY = 1\n", encoding="utf-8")
    (package / "perfectdiode_successor_confirmation.py").write_text(
        "RUNTIME = 1\n", encoding="utf-8"
    )
    launcher = (
        root
        / "experiments"
        / "run_mnist_conv_perfectdiode_successor_confirmation.py"
    )
    launcher.write_text("LAUNCHER = 1\n", encoding="utf-8")


def _archive_checkout(checkout: Path, archive: Path) -> None:
    with tarfile.open(archive, mode="w:gz") as handle:
        handle.add(checkout, arcname="source")


def test_source_archive_is_safe_and_checkout_needs_no_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staged = tmp_path / "staged"
    _mini_checkout(staged)
    archive = tmp_path / "source.tar.gz"
    _archive_checkout(staged, archive)
    environment = tmp_path / "environment.json"
    environment.write_text("{}\n", encoding="utf-8")
    fingerprint = code_fingerprint(staged)
    assert successor._source_archive_fingerprint(archive) == fingerprint
    assert not (staged / ".git").exists()

    launcher = (
        staged
        / "experiments"
        / "run_mnist_conv_perfectdiode_successor_confirmation.py"
    )
    runtime = (
        staged
        / "experiments"
        / "mnist_conv"
        / "perfectdiode_successor_confirmation.py"
    )
    monkeypatch.setattr(successor, "REPO_ROOT", staged)
    monkeypatch.setattr(successor, "WORKER_LAUNCHER", launcher)
    monkeypatch.setattr(successor, "RUNTIME_MODULE", runtime)
    manifest = {
        "execution_source": {
            "source_commit": "a" * 40,
            "source_archive_sha256": sha256_file(archive),
            "effective_code_fingerprint": fingerprint,
            "environment_contract_sha256": sha256_file(environment),
            "worker_launcher_sha256": sha256_file(launcher),
            "runtime_module_sha256": sha256_file(runtime),
        }
    }
    checks = successor._validate_external_execution_inputs(
        manifest,
        source_archive=archive,
        environment_contract=environment,
    )
    assert checks
    assert all(checks.values())
    assert "source_checkout_commit_valid" not in checks

    not_an_archive = tmp_path / "not-an-archive"
    not_an_archive.write_text("not a tar file\n", encoding="utf-8")
    with pytest.raises(
        successor.PerfectDiodeSuccessorError,
        match="readable tar or compressed-tar archive",
    ):
        successor._source_archive_fingerprint(not_an_archive)


def test_source_archive_fingerprint_honors_job_tmpdir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged = tmp_path / "staged"
    _mini_checkout(staged)
    archive = tmp_path / "source.tar.gz"
    _archive_checkout(staged, archive)
    jobscratch = tmp_path / "jobscratch"
    jobscratch.mkdir()
    monkeypatch.setenv("TMPDIR", str(jobscratch))
    monkeypatch.setattr(tempfile, "tempdir", None)
    created: list[Path] = []
    real_mkdtemp = tempfile.mkdtemp

    def recording_mkdtemp(*args: object, **kwargs: object) -> str:
        path = Path(real_mkdtemp(*args, **kwargs))
        created.append(path)
        return str(path)

    monkeypatch.setattr(successor.tempfile, "mkdtemp", recording_mkdtemp)
    assert successor._source_archive_fingerprint(archive) == code_fingerprint(
        staged
    )
    assert len(created) == 1
    assert created[0].parent == jobscratch
    assert not created[0].exists()


def test_bundle_config_copy_preserves_source_bytes_and_hash_domain(
    tmp_path: Path,
) -> None:
    value = {"z": [1, 2], "a": {"enabled": False}}
    source = tmp_path / "pretty.json"
    source.write_text(json.dumps(value, indent=4) + "\n", encoding="utf-8")
    destination = tmp_path / "bundle-config.json"
    successor._copy_config_verbatim(source, destination, value)
    assert destination.read_bytes() == source.read_bytes()
    assert sha256_file(destination) == sha256_file(source)
    assert sha256_file(source) != sha256_json(value)


def test_pending_implementation_state_blocks_production() -> None:
    value = successor.load_config(RICH_CONFIG)
    value["status"]["plan"] = "approved"
    value["status"]["implementation"] = (
        "implemented_preflight_pending_approval"
    )
    value["status"]["launch_authorized"] = True
    value["execution"]["implementation_state"] = (
        "implemented_preflight_pending_approval"
    )
    value["selection_override"].update(
        {
            "state": "approved",
            "approved": True,
            "approved_by": "Filip",
            "approved_at": "2026-07-27T16:00:00+02:00",
        }
    )
    with pytest.raises(
        successor.PerfectDiodeSuccessorError,
        match="reviewed_ready_for_launch",
    ):
        successor.validate_config(value, require_production_approval=True)
    value["status"]["implementation"] = "reviewed_ready_for_launch"
    value["execution"]["implementation_state"] = "reviewed_ready_for_launch"
    successor.validate_config(value, require_production_approval=True)


def test_successor_adapter_extends_and_restores_frozen_v1_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frozen = successor._hparam_runtime.training_contract
    root = tmp_path / "bundle"
    root.mkdir()
    observed: list[tuple[str, str, int, int]] = []

    monkeypatch.setattr(
        successor._hparam_runtime,
        "_study_and_spec",
        lambda _study: ({}, object()),
    )

    def execute(
        _data: object,
        _spec: object,
        _root: Path,
        _output: Path,
        _payload: dict[str, object],
        *,
        mode: str,
        **_kwargs: object,
    ) -> dict[str, object]:
        contract = successor._hparam_runtime.training_contract(
            architecture=str(_payload["architecture"]),
            mode=mode,
        )
        observed.append(
            (
                str(_payload["architecture"]),
                mode,
                contract.epochs,
                contract.total_steps,
            )
        )
        return {"status": "complete"}

    monkeypatch.setattr(
        successor._hparam_runtime,
        "_execute_training_entry",
        execute,
    )
    common = {
        "study": {},
        "shard_dir": root,
        "data_root": tmp_path / "data",
        "device": "cuda",
    }
    successor.execute_successor_stage_entry(
        **common,
        stage="successor_confirm",
        payload={"entry_id": "conv1", "architecture": "conv1"},
        output_dir=tmp_path / "conv1-output",
    )
    successor.execute_successor_stage_entry(
        **common,
        stage="successor_confirm",
        payload={"entry_id": "conv2", "architecture": "conv2"},
        output_dir=tmp_path / "conv2-output",
    )
    successor.execute_successor_stage_entry(
        **common,
        stage="successor_canary",
        payload={"entry_id": "canary", "architecture": "conv2"},
        output_dir=tmp_path / "canary-output",
    )
    assert observed == [
        ("conv1", "successor_confirmation", 10, 34_380),
        ("conv2", "successor_confirmation", 20, 68_760),
        ("conv2", "successor_canary", 1, 3_438),
    ]
    assert successor._hparam_runtime.training_contract is frozen

    def fail(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(
        successor._hparam_runtime,
        "_execute_training_entry",
        fail,
    )
    with pytest.raises(RuntimeError, match="synthetic failure"):
        successor.execute_successor_stage_entry(
            **common,
            stage="successor_confirm",
            payload={"entry_id": "failure", "architecture": "conv2"},
            output_dir=tmp_path / "failure-output",
        )
    assert successor._hparam_runtime.training_contract is frozen


def test_cli_emits_one_unambiguous_machine_result_line(
    capsys: pytest.CaptureFixture[str],
) -> None:
    value = {"status": "passed", "nested": {"value": 1}}
    successor_cli._print(value)
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1
    assert lines[0].startswith(successor_cli.RESULT_MARKER)
    assert json.loads(lines[0][len(successor_cli.RESULT_MARKER) :]) == value


def test_fixed_pack_contract_covers_entries_once_in_optimizer_pairs() -> None:
    assert successor.FIXED_ENTRY_PACKS == (
        (0, 1),
        (2, 3),
        (4, 5),
        (6, 7),
        (8, 9),
        (10, 11),
    )
    assert [
        index for pack in successor.FIXED_ENTRY_PACKS for index in pack
    ] == list(range(12))
    config = successor.load_config(RICH_CONFIG)
    for pack_index, (first, second) in enumerate(
        successor.FIXED_ENTRY_PACKS
    ):
        left = config["entries"][first]
        right = config["entries"][second]
        assert (left["architecture"], left["scheme"], left["row_id"]) == (
            right["architecture"],
            right["scheme"],
            right["row_id"],
        )
        assert (left["optimizer"], right["optimizer"]) == ("sgd", "adam")
        assert successor.fixed_pack_entries(pack_index) == (first, second)
    with pytest.raises(successor.PerfectDiodeSuccessorError):
        successor.fixed_pack_entries(True)
    with pytest.raises(successor.PerfectDiodeSuccessorError):
        successor.fixed_pack_entries(6)


def test_concurrent_children_start_both_before_wait_and_bind_logs(
    tmp_path: Path,
) -> None:
    started: list[int] = []
    waited: list[int] = []

    class FakeProcess:
        def __init__(self, index: int):
            self.index = index

        def wait(self) -> int:
            assert started == [0, 1]
            waited.append(self.index)
            return 0

        def poll(self) -> int:
            return 0

    def factory(
        command: list[str],
        *,
        stdout: object,
        **_kwargs: object,
    ) -> FakeProcess:
        index = len(started)
        started.append(index)
        terminal = {"status": "complete", "entry_index": index}
        stdout.write(  # type: ignore[attr-defined]
            successor.RESULT_JSON_MARKER
            + json.dumps(terminal, separators=(",", ":"))
            + "\n"
        )
        return FakeProcess(index)

    records, elapsed = successor._run_concurrent_children(
        [
            {"label": "entry_00", "entry_index": 0, "command": ["child", "0"]},
            {"label": "entry_01", "entry_index": 1, "command": ["child", "1"]},
        ],
        log_root=tmp_path / "logs",
        artifact_root=tmp_path,
        popen_factory=factory,
    )
    assert started == [0, 1]
    assert waited == [0, 1]
    assert elapsed > 0
    assert [record["terminal_result"]["entry_index"] for record in records] == [
        0,
        1,
    ]
    assert all(record["returncode"] == 0 for record in records)


def test_concurrent_children_wait_for_sibling_then_fail(
    tmp_path: Path,
) -> None:
    started: list[int] = []
    waited: list[int] = []

    class FakeProcess:
        def __init__(self, index: int):
            self.index = index

        def wait(self) -> int:
            assert started == [0, 1]
            waited.append(self.index)
            return 1 if self.index == 0 else 0

        def poll(self) -> int:
            return 0

    def factory(
        _command: list[str], *, stdout: object, **_kwargs: object
    ) -> FakeProcess:
        index = len(started)
        started.append(index)
        stdout.write("{}\n")  # type: ignore[attr-defined]
        return FakeProcess(index)

    with pytest.raises(RuntimeError, match="both concurrent packed children"):
        successor._run_concurrent_children(
            [
                {
                    "label": "entry_00",
                    "entry_index": 0,
                    "command": ["child", "0"],
                },
                {
                    "label": "entry_01",
                    "entry_index": 1,
                    "command": ["child", "1"],
                },
            ],
            log_root=tmp_path / "logs",
            artifact_root=tmp_path,
            popen_factory=factory,
        )
    assert waited == [0, 1]


def _admission_record(
    entry_index: int,
    *,
    elapsed: float = 600.0,
    steps_per_second: float = 6.0,
    allocated: int = 4 * 1024**3,
    reserved: int = 5 * 1024**3,
) -> dict[str, object]:
    return {
        "entry_index": entry_index,
        "entry_id": f"entry-{entry_index}",
        "completed_steps": 3_438,
        "benchmark": {
            "elapsed_seconds": elapsed,
            "successful_steps_per_second": steps_per_second,
            "cuda_peak_memory_allocated_bytes": allocated,
            "cuda_peak_memory_reserved_bytes": reserved,
        },
    }


def test_paired_canary_admission_enforces_memory_and_duration() -> None:
    records = [_admission_record(8), _admission_record(9)]
    evidence = successor._paired_canary_admission(
        records, concurrent_wall_elapsed_seconds=700.0
    )
    assert evidence["passed"] is True
    assert evidence["aggregate_completed_steps"] == 6_876
    assert evidence["combined_peak_memory_reserved_bytes"] == 10 * 1024**3
    with pytest.raises(
        successor.PerfectDiodeSuccessorError,
        match="paired canary admission",
    ):
        successor._paired_canary_admission(
            [
                _admission_record(8, reserved=14 * 1024**3),
                _admission_record(9, reserved=14 * 1024**3),
            ],
            concurrent_wall_elapsed_seconds=700.0,
        )
    with pytest.raises(
        successor.PerfectDiodeSuccessorError,
        match="paired canary admission",
    ):
        successor._paired_canary_admission(
            [
                _admission_record(8, steps_per_second=2.0),
                _admission_record(9, steps_per_second=2.0),
            ],
            concurrent_wall_elapsed_seconds=700.0,
        )


def test_canary_receipt_environment_is_required_and_hash_verified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embedded_preflight = {
        "source_path": str((tmp_path / "preflight.json").resolve()),
        "source_file_sha256": "1" * 64,
        "payload": {"input_binding": {"bundle_id": "bundle-1"}},
    }
    expected_gate = successor._expected_canary_gate_binding(
        embedded_preflight
    )
    payload = {
        "schema_version": successor.CANARY_GATE_SCHEMA_VERSION,
        "status": "passed",
        "gate_passed": True,
        "gate_binding": expected_gate,
        "gate_binding_sha256": sha256_json(expected_gate),
    }
    receipt = tmp_path / "successor-canary.json"
    atomic_write_json(receipt, payload, canonical=True)
    monkeypatch.setattr(
        successor,
        "_validate_live_canary_gate_receipt",
        lambda path, *, expected_gate_binding: read_json(path),
    )

    with pytest.raises(
        successor.PerfectDiodeSuccessorError,
        match="both environment variables",
    ):
        successor._canary_gate_receipt_from_environment(
            preflight_receipt=embedded_preflight,
            required=True,
        )

    monkeypatch.setenv("PD_SUCCESSOR_CANARY_RECEIPT", str(receipt))
    monkeypatch.setenv(
        "PD_SUCCESSOR_CANARY_RECEIPT_SHA256", sha256_file(receipt)
    )
    binding = successor._canary_gate_receipt_from_environment(
        preflight_receipt=embedded_preflight,
        required=True,
    )
    assert binding is not None
    assert binding["payload"] == payload

    atomic_write_json(
        receipt,
        {**payload, "status": "tampered"},
        canonical=True,
    )
    with pytest.raises(successor.PerfectDiodeSuccessorError):
        successor._canary_gate_receipt_from_environment(
            preflight_receipt=embedded_preflight,
            required=True,
        )


@pytest.mark.parametrize("tampered_receipt", [False, True])
def test_production_worker_rejects_missing_or_tampered_canary_before_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tampered_receipt: bool,
) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    atomic_write_json(bundle / "config.json", {"config": 1}, canonical=True)
    atomic_write_json(bundle / "manifest.json", {"manifest": 1}, canonical=True)
    output = tmp_path / "output"
    data = tmp_path / "data"
    data.mkdir()
    source_archive = tmp_path / "source.tar.gz"
    source_archive.write_bytes(b"archive")
    environment = tmp_path / "environment.json"
    atomic_write_json(environment, {}, canonical=True)
    manifest = {"entries": []}
    embedded_preflight = {
        "source_path": str((tmp_path / "preflight.json").resolve()),
        "source_file_sha256": "1" * 64,
        "payload": {"input_binding": {"bundle_id": "bundle-1"}},
    }
    monkeypatch.setattr(
        successor,
        "validate_input_bundle",
        lambda _path: copy.deepcopy(manifest),
    )
    monkeypatch.setattr(successor, "validate_config", lambda *_a, **_kw: {})
    monkeypatch.setattr(
        successor,
        "preflight_bundle",
        lambda **_kwargs: {"passed": True},
    )
    monkeypatch.setattr(
        successor,
        "_preflight_receipt_from_environment",
        lambda **_kwargs: copy.deepcopy(embedded_preflight),
    )
    executor_called = False

    def forbidden_executor(**_kwargs: object) -> dict[str, object]:
        nonlocal executor_called
        executor_called = True
        return {}

    if tampered_receipt:
        canary = tmp_path / "canary.json"
        atomic_write_json(canary, {"status": "passed"}, canonical=True)
        monkeypatch.setenv("PD_SUCCESSOR_CANARY_RECEIPT", str(canary))
        monkeypatch.setenv(
            "PD_SUCCESSOR_CANARY_RECEIPT_SHA256", "f" * 64
        )
    with pytest.raises(successor.PerfectDiodeSuccessorError):
        successor.run_indexed_entry(
            bundle_dir=bundle,
            entry_index=0,
            output_root=output,
            data_root=data,
            source_archive=source_archive,
            environment_contract=environment,
            executor=forbidden_executor,
        )
    assert executor_called is False
    assert not output.exists()


def _receipt_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, object]:
    repo = tmp_path / "repo"
    _mini_checkout(repo)
    wrapper = (
        repo
        / "experiments"
        / "run_mnist_conv_perfectdiode_successor_confirmation_jeanzay.slurm"
    )
    wrapper.write_text("#!/bin/bash\n", encoding="utf-8")
    scheduled_script = (
        repo
        / "skills"
        / "scheduled-run-preflight"
        / "scripts"
        / "preflight_scheduled_runner.py"
    )
    scheduled_script.parent.mkdir(parents=True)
    scheduled_script.write_text("SCRIPT = 1\n", encoding="utf-8")
    submitter = repo / "experiments" / "submit.py"
    supervisor = repo / "experiments" / "supervise.py"
    semantic_verifier = repo / "experiments" / "verify.py"
    plan_validator = repo / "skills" / "run-experiment-pipeline" / "validate.py"
    official_verifier = repo / "official-canary-verifier.py"
    plan_validator.parent.mkdir(parents=True)
    for path in (
        submitter,
        supervisor,
        semantic_verifier,
        plan_validator,
        official_verifier,
    ):
        path.write_text(f"{path.name!r}\n", encoding="utf-8")
    launcher = (
        repo
        / "experiments"
        / "run_mnist_conv_perfectdiode_successor_confirmation.py"
    )
    runtime_module = (
        repo
        / "experiments"
        / "mnist_conv"
        / "perfectdiode_successor_confirmation.py"
    )
    monkeypatch.setattr(successor, "REPO_ROOT", repo)
    monkeypatch.setattr(successor, "WORKER_LAUNCHER", launcher)
    monkeypatch.setattr(successor, "RUNTIME_MODULE", runtime_module)
    monkeypatch.setattr(successor, "JEAN_ZAY_WRAPPER", wrapper)
    monkeypatch.setattr(successor, "SCHEDULED_PREFLIGHT_SCRIPT", scheduled_script)
    monkeypatch.setattr(successor, "JEAN_ZAY_SUBMITTER", submitter)
    monkeypatch.setattr(successor, "JEAN_ZAY_SUPERVISOR", supervisor)
    monkeypatch.setattr(successor, "SEMANTIC_CANARY_VERIFIER", semantic_verifier)
    monkeypatch.setattr(successor, "EXPERIMENT_PLAN_VALIDATOR", plan_validator)
    python_executable = str(Path(sys.executable).resolve())
    monkeypatch.setattr(successor, "JEAN_ZAY_PYTHON", python_executable)
    monkeypatch.setenv("PD_SUCCESSOR_PYTHON", python_executable)

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    atomic_write_json(bundle / "manifest.json", {"immutable": True}, canonical=True)
    source_archive = tmp_path / "source.tar.gz"
    _archive_checkout(repo, source_archive)
    environment = tmp_path / "environment.json"
    atomic_write_json(environment, {"environment": 1}, canonical=True)
    data = tmp_path / "data"
    data.mkdir()
    output = tmp_path / "output"
    output.mkdir()
    manifest = {
        "bundle_id": "pdconfirmbundle_" + "1" * 64,
        "config_sha256": "2" * 64,
        "config_file_sha256": "5" * 64,
        "execution_source": {
            "source_commit": "a" * 40,
            "source_archive_sha256": sha256_file(source_archive),
            "effective_code_fingerprint": code_fingerprint(repo),
            "environment_contract_sha256": sha256_file(environment),
            "worker_launcher_sha256": sha256_file(launcher),
            "runtime_module_sha256": sha256_file(runtime_module),
        },
    }
    runtime_preflight = {
        "schema_version": successor.PREFLIGHT_SCHEMA_VERSION,
        "status": "passed",
        "passed": True,
        "official_test_read": False,
        "bundle_id": manifest["bundle_id"],
        "bundle_manifest_sha256": sha256_file(bundle / "manifest.json"),
        "config_sha256": manifest["config_sha256"],
        "config_file_sha256": manifest["config_file_sha256"],
        "external_launch_authorization_required": True,
        "expected_entry_count": 12,
        "checks": {"all_gates": True},
    }
    hashes = {
        "bundle_sha256": successor._directory_tree_sha256(bundle),
        "source_archive_sha256": sha256_file(source_archive),
        "environment_contract_sha256": sha256_file(environment),
        "runtime_cli_sha256": sha256_file(launcher),
        "runtime_module_sha256": sha256_file(runtime_module),
        "submitter_sha256": sha256_file(submitter),
        "supervisor_sha256": sha256_file(supervisor),
        "semantic_canary_verifier_sha256": sha256_file(semantic_verifier),
        "experiment_plan_validator_sha256": sha256_file(plan_validator),
        "wrapper_sha256": sha256_file(wrapper),
        "scheduled_preflight_script_sha256": sha256_file(scheduled_script),
        "official_canary_verifier_sha256": sha256_file(official_verifier),
    }
    plan = tmp_path / "approved-plan.md"
    plan.write_text("# Approved\n", encoding="utf-8")
    approved_plan = {"path": str(plan), "sha256": sha256_file(plan)}
    authorization = {
        "schema_version": successor.LAUNCH_AUTHORIZATION_SCHEMA_VERSION,
        "status": "approved",
        "launch_authorized": True,
        "approved_plan": approved_plan,
        "bundle": {
            "bundle_id": manifest["bundle_id"],
            "manifest_sha256": sha256_file(bundle / "manifest.json"),
            "config_sha256": manifest["config_sha256"],
            "config_file_sha256": manifest["config_file_sha256"],
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
    authorization_path = tmp_path / "launch-authorization.json"
    atomic_write_json(authorization_path, authorization, canonical=True)
    authorization_sha = sha256_file(authorization_path)
    scheduled = {
        "schema_version": successor.SCHEDULED_PREFLIGHT_RECEIPT_SCHEMA_VERSION,
        "status": "passed",
        "runner": str(wrapper.resolve()),
        "runner_sha256": hashes["wrapper_sha256"],
    }
    scheduled_path = tmp_path / "scheduled.json"
    atomic_write_json(scheduled_path, scheduled, canonical=True)
    scheduled_sha = sha256_file(scheduled_path)
    input_binding = {
        "schema_version": successor.GATE_INPUT_SCHEMA_VERSION,
        "bundle_id": manifest["bundle_id"],
        "bundle_manifest_sha256": sha256_file(bundle / "manifest.json"),
        "config_sha256": manifest["config_sha256"],
        "config_file_sha256": manifest["config_file_sha256"],
        "repo_root": str(repo.resolve()),
        "bundle_dir": str(bundle.resolve()),
        "data_root": str(data.resolve()),
        "source_archive": str(source_archive.resolve()),
        "environment_contract": str(environment.resolve()),
        "runtime_cli": str(launcher.resolve()),
        "runtime_module": str(runtime_module.resolve()),
        "submitter": str(submitter.resolve()),
        "supervisor": str(supervisor.resolve()),
        "semantic_canary_verifier": str(semantic_verifier.resolve()),
        "experiment_plan_validator": str(plan_validator.resolve()),
        "wrapper": str(wrapper.resolve()),
        "scheduled_preflight_script": str(scheduled_script.resolve()),
        "official_canary_verifier": str(official_verifier.resolve()),
        "canary_pack_index": 4,
        "canary_entry_indices": [8, 9],
        "hashes": hashes,
        "runtime_preflight_sha256": sha256_json(runtime_preflight),
        "scheduled_preflight_receipt_path": str(scheduled_path.resolve()),
        "scheduled_preflight_receipt_sha256": scheduled_sha,
        "launch_authorization_receipt_path": str(
            authorization_path.resolve()
        ),
        "launch_authorization_receipt_sha256": authorization_sha,
        "approved_plan": approved_plan,
    }
    placeholder_path = "__PD_SUCCESSOR_PREFLIGHT_RECEIPT_PATH__"
    placeholder_sha = "__PD_SUCCESSOR_PREFLIGHT_RECEIPT_SHA256__"
    canary_placeholder_path = "__PD_SUCCESSOR_CANARY_RECEIPT_PATH__"
    canary_placeholder_sha = "__PD_SUCCESSOR_CANARY_RECEIPT_SHA256__"
    common_inputs = {
        key: input_binding[key]
        for key in (
            "repo_root",
            "bundle_dir",
            "data_root",
            "source_archive",
            "environment_contract",
            "runtime_cli",
            "runtime_module",
            "submitter",
            "supervisor",
            "semantic_canary_verifier",
            "experiment_plan_validator",
            "wrapper",
            "scheduled_preflight_script",
            "official_canary_verifier",
            "hashes",
            "canary_pack_index",
            "canary_entry_indices",
        )
    }
    common_inputs.update(
        {
            "scheduled_preflight_script": str(scheduled_script.resolve()),
            "official_canary_verifier": str(official_verifier.resolve()),
            "supervisor_state": str((tmp_path / "supervisor-state.json").resolve()),
            "official_canary_receipt": str(
                (tmp_path / "official-canary.json").resolve()
            ),
            "successor_canary_receipt": str(
                (tmp_path / "successor-canary.json").resolve()
            ),
            "preflight_receipt": str((tmp_path / "preflight.json").resolve()),
            "remote_user": "testuser",
            "launch_authorization_receipt_path": input_binding[
                "launch_authorization_receipt_path"
            ],
            "launch_authorization_receipt_sha256": input_binding[
                "launch_authorization_receipt_sha256"
            ],
            "scheduled_preflight_receipt_path": input_binding[
                "scheduled_preflight_receipt_path"
            ],
            "scheduled_preflight_receipt_sha256": input_binding[
                "scheduled_preflight_receipt_sha256"
            ],
        }
    )
    resources = {
        "allocation_id": "AD011016471R1",
        "project": "umg",
        "account": "umg@v100",
        "partition": "gpu_p13",
        "qos": "qos_gpu-t3",
        "constraint": "v100-32g",
        "nodes": 1,
        "tasks": 1,
        "gpus_per_task": 1,
        "cpus_per_task": 16,
        "host_memory_policy": (
            "jean_zay_site_managed_no_explicit_slurm_request"
        ),
        "hint": "nomultithread",
        "module": "pytorch-gpu/py3/2.5.0",
        "python_executable": python_executable,
    }

    def command_template(
        array: str, walltime: str, *, include_canary_gate: bool
    ) -> list[str]:
        placeholders = f"PATH={placeholder_path},SHA={placeholder_sha}"
        if include_canary_gate:
            placeholders += (
                f",CANARY_PATH={canary_placeholder_path},"
                f"CANARY_SHA={canary_placeholder_sha}"
            )
        return [
            "sbatch",
            f"--array={array}",
            f"--time={walltime}",
            "--account=umg@v100",
            "--partition=gpu_p13",
            "--qos=qos_gpu-t3",
            "--constraint=v100-32g",
            "--gres=gpu:1",
            "--cpus-per-task=16",
            "--hint=nomultithread",
            f"--export={placeholders}",
        ]

    commands = {}
    for kind, array, task_count, walltime, output_root in (
        ("canary", "0-0", 1, "02:00:00", output),
        ("production", "0-5%6", 6, "08:00:00", output),
    ):
        template = command_template(
            array,
            walltime,
            include_canary_gate=kind == "production",
        )
        commands[kind] = {
            "kind": kind,
            "array": array,
            "task_count": task_count,
            "walltime": walltime,
            "output_root": str(output_root.resolve()),
            "template": template,
            "template_sha256": sha256_json(template),
        }
    launch_contract = {
        "schema_version": "perfectdiode-successor-launch-contract/v1",
        "self_reference_placeholders": {
            "preflight_receipt_path": placeholder_path,
            "preflight_receipt_sha256": placeholder_sha,
            "canary_receipt_path": canary_placeholder_path,
            "canary_receipt_sha256": canary_placeholder_sha,
        },
        "python_executable": python_executable,
        "output_root": str(output.resolve()),
        "common_inputs": common_inputs,
        "resources": resources,
        "commands": commands,
    }
    input_binding["launch_contract"] = launch_contract
    input_binding["launch_contract_sha256"] = sha256_json(launch_contract)
    receipt = {
        "schema_version": successor.CUSTOM_PREFLIGHT_RECEIPT_SCHEMA_VERSION,
        "status": "passed",
        "passed": True,
        "runtime_preflight": runtime_preflight,
        "runtime_preflight_sha256": sha256_json(runtime_preflight),
        "input_binding": input_binding,
        "input_binding_sha256": sha256_json(input_binding),
        "scheduled_run_preflight": {
            "path": str(scheduled_path.resolve()),
            "sha256": scheduled_sha,
            "schema_version": successor.SCHEDULED_PREFLIGHT_RECEIPT_SCHEMA_VERSION,
            "status": "passed",
            "runner": str(wrapper.resolve()),
            "runner_sha256": hashes["wrapper_sha256"],
        },
    }
    receipt_path = tmp_path / "preflight.json"
    atomic_write_json(receipt_path, receipt, canonical=True)
    monkeypatch.setenv("PD_SUCCESSOR_PREFLIGHT_RECEIPT", str(receipt_path))
    monkeypatch.setenv(
        "PD_SUCCESSOR_PREFLIGHT_RECEIPT_SHA256", sha256_file(receipt_path)
    )
    monkeypatch.setenv(
        "PD_SUCCESSOR_LAUNCH_AUTHORIZATION_RECEIPT",
        str(authorization_path),
    )
    monkeypatch.setenv(
        "PD_SUCCESSOR_LAUNCH_AUTHORIZATION_RECEIPT_SHA256",
        authorization_sha,
    )
    monkeypatch.setattr(
        successor,
        "preflight_bundle",
        lambda **_kwargs: copy.deepcopy(runtime_preflight),
    )
    return {
        "manifest": manifest,
        "bundle": bundle,
        "source_archive": source_archive,
        "environment": environment,
        "data": data,
        "canary_output": output,
        "receipt_path": receipt_path,
        "authorization_path": authorization_path,
        "scheduled_path": scheduled_path,
        "plan": plan,
    }


def test_custom_receipt_is_live_verified_then_self_contained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _receipt_fixture(tmp_path, monkeypatch)
    embedded = successor._preflight_receipt_from_environment(
        manifest=fixture["manifest"],
        bundle_root=fixture["bundle"],
        required=True,
        source_archive=fixture["source_archive"],
        environment_contract=fixture["environment"],
        data_root=fixture["data"],
        output_root=fixture["canary_output"],
        expected_kind="canary",
    )
    assert embedded is not None
    for key in ("receipt_path", "authorization_path", "scheduled_path", "plan"):
        fixture[key].unlink()
    successor._validate_embedded_preflight_binding(
        embedded,
        manifest=fixture["manifest"],
        bundle_root=fixture["bundle"],
        path="embedded",
    )
    tampered = copy.deepcopy(embedded)
    tampered["payload"]["status"] = "blocked"
    with pytest.raises(
        successor.PerfectDiodeSuccessorError, match="payload_sha256"
    ):
        successor._validate_embedded_preflight_binding(
            tampered,
            manifest=fixture["manifest"],
            bundle_root=fixture["bundle"],
            path="embedded",
        )


def test_entry_lock_rejects_a_duplicate_writer(tmp_path: Path) -> None:
    with successor._entry_execution_lock(tmp_path, "entry"):
        with pytest.raises(
            successor.PerfectDiodeSuccessorError,
            match="no concurrent execution",
        ):
            with successor._entry_execution_lock(tmp_path, "entry"):
                pass


def test_training_canary_requires_full_epoch_and_all_six_outputs(
    tmp_path: Path,
) -> None:
    smoke = tmp_path / "smoke"
    output = smoke / "training_canary"
    output.mkdir(parents=True)
    entry = {
        "entry_index": 9,
        "entry_id": "entry-9",
        "cell_id": "cell-9",
        "row_id": "conv2_ours_v4_c1",
        "architecture": "conv2",
        "scheme": "ours",
        "optimizer": "adam",
        "runtime_rho_conv": 0.027,
        "runtime_rho_dense": 0.03,
        "raw_learning_rates_by_parameter": {"layer": 0.1},
    }
    run_spec = {
        "official_test_read": False,
        "training": {
            "mode": "successor_canary",
            "epochs": 1,
            "steps_per_epoch": 3_438,
            "total_steps": 3_438,
            "restart_from_shared_initialization": True,
            "continue_from_canary_or_candidate": False,
        },
    }
    validation = {
        "official_test_read": False,
        "records": [{"step": 3_438}],
    }
    atomic_write_json(output / "run_spec.json", run_spec, canonical=True)
    atomic_write_json(output / "validation.json", validation, canonical=True)
    (output / "step_log.csv").write_text("step\n3438\n", encoding="utf-8")
    (output / "best_validation.pt").write_bytes(b"best")
    (output / "final.pt").write_bytes(b"final")
    artifact_names = (
        "run_spec.json",
        "validation.json",
        "step_log.csv",
        "best_validation.pt",
        "final.pt",
    )
    result = {
        "official_test_read": False,
        "mode": "successor_canary",
        "status": "complete",
        "training_completed": True,
        "completed_steps": 3_438,
        "expected_steps": 3_438,
        "epochs_completed": 1,
        "expected_epochs": 1,
        "entry_id": entry["entry_id"],
        "cell_id": entry["cell_id"],
        "row_id": entry["row_id"],
        "architecture": entry["architecture"],
        "scheme": entry["scheme"],
        "optimizer": entry["optimizer"],
        "rho_conv": entry["runtime_rho_conv"],
        "rho_dense": entry["runtime_rho_dense"],
        "raw_learning_rates_by_parameter": entry[
            "raw_learning_rates_by_parameter"
        ],
        "artifacts": [
            {
                "path": name,
                "sha256": sha256_file(output / name),
                "bytes": (output / name).stat().st_size,
            }
            for name in artifact_names
        ],
    }
    atomic_write_json(output / "result.json", result, canonical=True)
    record = successor._validate_training_canary_outputs(
        smoke_root=smoke,
        entry=entry,
        output_dir=output,
    )
    assert record["completed_steps"] == 3_438
    assert len(record["output_artifacts"]) == 6
    (output / "step_log.csv").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(successor.PerfectDiodeSuccessorError):
        successor._validate_training_canary_outputs(
            smoke_root=smoke,
            entry=entry,
            output_dir=output,
        )


def test_smoke_passes_all_execution_inputs_to_receipt_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name, value in (
        ("SLURM_JOB_ID", "101"),
        ("SLURM_ARRAY_JOB_ID", "100"),
        ("SLURM_ARRAY_TASK_ID", "0"),
        ("SLURM_JOB_ACCOUNT", "umg@v100"),
        ("SLURM_JOB_PARTITION", "gpu_p13"),
        ("SLURM_JOB_QOS", "qos_gpu-t3"),
        ("PD_SUCCESSOR_CONSTRAINT", "v100-32g"),
    ):
        monkeypatch.setenv(name, value)
    bundle = tmp_path / "bundle"
    output = tmp_path / "output"
    data = tmp_path / "data"
    bundle.mkdir()
    output.mkdir()
    data.mkdir()
    for name, value in (
        ("manifest.json", {"manifest": 1}),
        ("config.json", {"config": 1}),
        ("study.resolved.json", {"study": 1}),
    ):
        atomic_write_json(bundle / name, value, canonical=True)
    entry = {
        "entry_index": 9,
        "entry_id": "entry-9",
        "row_id": "conv2_ours_v4_c1",
        "architecture": "conv2",
        "scheme": "ours",
        "optimizer": "adam",
        "asset_entry_id": "conv2-assets",
        "runtime_rho_conv": 0.027,
        "runtime_rho_dense": 0.03,
        "cell_id": "cell-9",
        "probe_result": "probe.json",
        "raw_learning_rates_by_parameter": {"layer": 0.1},
    }
    manifest = {
        "bundle_id": "bundle-1",
        "config_sha256": "1" * 64,
        "config_file_sha256": "2" * 64,
        "entries": [{}, {}, {}, {}, {}, {}, {}, {}, {}, entry],
        "execution_source": {
            "source_commit": "a" * 40,
            "source_archive_sha256": "b" * 64,
            "environment_contract_sha256": "c" * 64,
        },
    }
    monkeypatch.setattr(
        successor, "validate_input_bundle", lambda _path: copy.deepcopy(manifest)
    )
    monkeypatch.setattr(successor, "validate_config", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        successor,
        "preflight_bundle",
        lambda **_kwargs: {"passed": True},
    )
    captured: dict[str, object] = {}

    def receipt_gate(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"embedded": True}

    monkeypatch.setattr(
        successor, "_preflight_receipt_from_environment", receipt_gate
    )
    monkeypatch.setattr(
        successor,
        "_validate_training_canary_outputs",
        lambda **_kwargs: {
            "entry_index": 9,
            "entry_id": "entry-9",
            "mode": "successor_canary",
            "status": "complete",
            "training_completed": True,
            "completed_steps": 3_438,
            "expected_steps": 3_438,
            "epochs_completed": 1,
            "expected_epochs": 1,
            "official_test_read": False,
            "fresh_restart_from_shared_initialization": True,
            "raw_learning_rates_by_parameter": {"layer": 0.1},
            "output_artifacts": [],
        },
    )
    monkeypatch.setattr(
        successor,
        "_validate_smoke_receipt",
        lambda **kwargs: read_json(kwargs["smoke_dir"] / "receipt.json"),
    )
    stages: list[str] = []

    def executor(**kwargs: object) -> dict[str, object]:
        stage = str(kwargs["stage"])
        stages.append(stage)
        result_dir = Path(kwargs["output_dir"])
        result_dir.mkdir(parents=True, exist_ok=True)
        if stage == "fixed_tk_gradient_security":
            architecture = kwargs["payload"]["architecture"]
            result = {
                "status": "complete",
                "security_passed": True,
                "official_test_read": False,
                "operational": {
                    "T": 4 if architecture == "conv1" else 6,
                    "K": 4 if architecture == "conv1" else 6,
                },
                "reference": {"T": 64, "K": 64},
            }
            atomic_write_json(result_dir / "result.json", result, canonical=True)
            return result
        return {"status": "complete"}

    source_archive = tmp_path / "source.tar.gz"
    environment = tmp_path / "environment.json"
    source_archive.write_text("source\n", encoding="utf-8")
    environment.write_text("{}\n", encoding="utf-8")
    result = successor.run_smoke(
        bundle_dir=bundle,
        output_root=output,
        data_root=data,
        source_archive=source_archive,
        environment_contract=environment,
        stage_executor=executor,
    )
    assert result["training_canary_steps"] == 3_438
    assert stages == ["fixed_tk_gradient_security"] * 6 + ["successor_canary"]
    assert captured["source_archive"] == source_archive
    assert captured["environment_contract"] == environment
    assert captured["data_root"] == data
    assert captured["required"] is True
    first_receipt = Path(result["receipt_path"])
    assert first_receipt.parent.name == "100_101_0"

    monkeypatch.setenv("SLURM_JOB_ID", "201")
    monkeypatch.setenv("SLURM_ARRAY_JOB_ID", "200")
    second = successor.run_smoke(
        bundle_dir=bundle,
        output_root=output,
        data_root=data,
        source_archive=source_archive,
        environment_contract=environment,
        stage_executor=executor,
    )
    second_receipt = Path(second["receipt_path"])
    assert second_receipt.parent.name == "200_201_0"
    assert second_receipt != first_receipt
    assert stages == (
        ["fixed_tk_gradient_security"] * 6 + ["successor_canary"]
    ) * 2
