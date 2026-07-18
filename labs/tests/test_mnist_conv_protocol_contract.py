from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from experiments.mnist_conv import protocol
from experiments.mnist_conv.backend import ExecutionContext
from experiments.mnist_conv.collection import _frozen_science_reasons
from experiments.mnist_conv.layout import ResultLayout
from experiments.mnist_conv.protocol import (
    ProtocolExecutionError,
    ensure_run_executable,
    final_protocol_approval_reason,
    protocol_contract_fingerprint,
)
from experiments.mnist_conv.runner import execute_run
from experiments.mnist_conv.specs import RunSpec


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_EXAMPLE = REPO_ROOT / "configs/conv/run_v1.diagnostic.example.json"
PROVENANCE = {
    "git_revision": "1" * 40,
    "dirty_source_digest": "2" * 64,
    "effective_code_fingerprint": "3" * 64,
}
TEST_PROTOCOL_ID = "approved-test-protocol"


def _final_value() -> dict:
    value = json.loads(RUN_EXAMPLE.read_text())
    value["category"] = "final"
    value["protocol_id"] = TEST_PROTOCOL_ID
    return value


def _approved(monkeypatch: pytest.MonkeyPatch, spec: RunSpec) -> str:
    fingerprint = protocol_contract_fingerprint(spec)
    monkeypatch.setattr(
        protocol,
        "APPROVED_FINAL_PROTOCOL_CONTRACTS",
        {TEST_PROTOCOL_ID: frozenset({fingerprint})},
    )
    return fingerprint


def test_contract_fingerprint_excludes_display_paths_but_keeps_stochastic_identity() -> None:
    value = _final_value()
    value["run"]["initialization"]["checkpoint"] = {
        "path": "imports/initial.pt",
        "sha256": "4" * 64,
        "format": "drn.function.parameters/v1",
        "source_run_id": None,
        "role": "initialization",
    }
    baseline = protocol_contract_fingerprint(RunSpec.from_dict(value))

    renamed = copy.deepcopy(value)
    renamed["label"] = "another-display-label"
    renamed["run"]["initialization"]["checkpoint"]["path"] = "relocated/initial.pt"
    assert protocol_contract_fingerprint(RunSpec.from_dict(renamed)) == baseline

    replicated = copy.deepcopy(value)
    replicated["replicate_id"] = "exact-replication-2"
    assert protocol_contract_fingerprint(RunSpec.from_dict(replicated)) != baseline

    changed_checkpoint = copy.deepcopy(value)
    changed_checkpoint["run"]["initialization"]["checkpoint"]["sha256"] = "5" * 64
    assert protocol_contract_fingerprint(RunSpec.from_dict(changed_checkpoint)) != baseline


def test_reused_protocol_id_cannot_execute_changed_science(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    approved = RunSpec.from_dict(_final_value())
    approved_fingerprint = _approved(monkeypatch, approved)
    assert final_protocol_approval_reason(approved) is None
    ensure_run_executable(approved)

    changed_value = _final_value()
    changed_value["run"]["training"]["learning_rate"][0] *= 2
    changed = RunSpec.from_dict(changed_value)
    assert protocol_contract_fingerprint(changed) != approved_fingerprint
    assert final_protocol_approval_reason(changed) == "protocol_contract_mismatch"
    with pytest.raises(ProtocolExecutionError, match="protocol_contract_mismatch"):
        ensure_run_executable(changed)

    results = tmp_path / "results"
    with pytest.raises(ProtocolExecutionError, match="protocol_contract_mismatch"):
        execute_run(
            changed,
            ExecutionContext(tmp_path / "data", device="cpu"),
            ResultLayout(results),
            PROVENANCE,
            backend=pytest.fail,
        )
    assert not results.exists()


def test_collector_uses_the_same_exact_contract_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    approved = RunSpec.from_dict(_final_value())
    _approved(monkeypatch, approved)
    approved_reasons = _frozen_science_reasons(approved)
    assert "protocol_not_approved" not in approved_reasons
    assert "protocol_contract_mismatch" not in approved_reasons

    changed_value = _final_value()
    changed_value["run"]["solver"]["training_iterations"] += 1
    changed = RunSpec.from_dict(changed_value)
    changed_reasons = _frozen_science_reasons(changed)
    assert "protocol_not_approved" not in changed_reasons
    assert "protocol_contract_mismatch" in changed_reasons

    unknown_value = _final_value()
    unknown_value["protocol_id"] = "unknown-protocol"
    unknown = RunSpec.from_dict(unknown_value)
    assert "protocol_not_approved" in _frozen_science_reasons(unknown)
