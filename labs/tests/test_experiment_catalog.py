from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiments import experiment_catalog as catalog_module


REPO_ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = REPO_ROOT / "experiments" / "experiment_catalog.json"


def test_repository_catalog_is_valid() -> None:
    catalog = catalog_module.load_catalog(CATALOG_PATH)
    warnings = catalog_module.validate_catalog(catalog, REPO_ROOT)

    states = {
        entry["experiment_id"]: catalog_module.plan_contract(entry, REPO_ROOT)["state"]
        for entry in catalog["entries"]
    }
    assert (
        states["perfectdiode-conv2-high-rho-corner-20260727-v1"]
        == "legacy_receipt_contract"
    )
    assert (
        states["conv3-perfectdiode-tk-ordinary-mnist-gain360-seed0-v1"]
        == "missing"
    )
    assert (
        states["perfectdiode-conv3-lr-ordinary-mnist-t8k8-20260727-v5"]
        == "legacy_receipt_contract"
    )
    assert any(
        "conv3-perfectdiode-tk-ordinary-mnist-gain360-seed0-v1" in warning
        for warning in warnings
    )
    assert any(
        "perfectdiode-conv3-lr-ordinary-mnist-t8k8-20260727-v5" in warning
        and "legacy receipt contract" in warning
        for warning in warnings
    )

    legacy_entry = next(
        entry
        for entry in catalog["entries"]
        if entry["experiment_id"]
        == "perfectdiode-conv3-lr-ordinary-mnist-t8k8-20260727-v5"
    )
    assert (
        catalog_module.next_step(legacy_entry, REPO_ROOT)
        == "repair_catalog_or_plan"
    )

def test_resolve_exact_alias_ambiguity_and_not_found() -> None:
    catalog = catalog_module.load_catalog(CATALOG_PATH)
    catalog_module.validate_catalog(catalog, REPO_ROOT)

    exact = catalog_module.resolve(
        catalog,
        "conv3-perfectdiode-tk-ordinary-mnist-gain360-seed0-v1",
        REPO_ROOT,
    )
    alias = catalog_module.resolve(
        catalog,
        "perfect diode conv3 tk",
        REPO_ROOT,
    )
    ambiguous = catalog_module.resolve(
        catalog,
        "conv3 perfect diode",
        REPO_ROOT,
    )
    missing = catalog_module.resolve(
        catalog,
        "limited conductance range conv2 perfect_diode",
        REPO_ROOT,
    )
    underspecified = catalog_module.resolve(catalog, "high", REPO_ROOT)

    assert exact["status"] == "resolved"
    assert exact["match"] == "exact"
    assert alias["status"] == "resolved"
    assert (
        alias["selected"]["experiment_id"]
        == "conv3-perfectdiode-tk-ordinary-mnist-gain360-seed0-v1"
    )
    assert ambiguous["status"] == "ambiguous"
    assert {
        "conv3-perfectdiode-tk-ordinary-mnist-gain360-seed0-v1",
        "perfectdiode-conv3-lr-ordinary-mnist-t8k8-20260727-v5",
    } <= {
        candidate["experiment_id"] for candidate in ambiguous["candidates"]
    }
    assert "selected" not in ambiguous
    assert missing["status"] == "not_found"
    assert "selected" not in missing
    assert all(
        "limited" in suggestion["unmatched_tokens"]
        for suggestion in missing["suggestions"]
    )
    assert underspecified["status"] == "not_found"
    assert "at least two" in underspecified["reason"]


def test_structured_resolution_derives_explicit_parent_subset() -> None:
    catalog = catalog_module.load_catalog(CATALOG_PATH)

    outcome = catalog_module.resolve_structured_request(
        catalog,
        {
            "experiment": "Conv1 best-rho continuation",
            "type": "continuation",
            "parent": (
                "perfectdiode-conv12-best-observed-confirmation-20260727-v1"
            ),
            "cases_subset": [
                "pdconfirm_conv1_baseline_sgd_seed0",
                "pdconfirm_conv1_baseline_adam_seed0",
                "pdconfirm_conv1_ours_sgd_seed0",
                "pdconfirm_conv1_ours_adam_seed0",
                "pdconfirm_conv1_legacy_sgd_seed0",
                "pdconfirm_conv1_legacy_adam_seed0",
            ],
            "scientific_changes": "none",
            "keep_fixed": "everything scientific",
            "operational_changes": {
                "preferred_compute": ["local", "Trex", "Akib"],
                "distribution": "two jobs per host",
            },
            "unresolved_scientific_choices": "none",
        },
        REPO_ROOT,
    )

    assert outcome["status"] == "derived"
    assert outcome["resolution_kind"] == (
        "derived_study_and_execution_attempt"
    )
    assert outcome["parent"]["experiment_id"] == (
        "perfectdiode-conv12-best-observed-confirmation-20260727-v1"
    )
    assert outcome["selection"] == {
        "mode": "subset",
        "cases": [
            "pdconfirm_conv1_baseline_sgd_seed0",
            "pdconfirm_conv1_baseline_adam_seed0",
            "pdconfirm_conv1_ours_sgd_seed0",
            "pdconfirm_conv1_ours_adam_seed0",
            "pdconfirm_conv1_legacy_sgd_seed0",
            "pdconfirm_conv1_legacy_adam_seed0",
        ],
    }
    assert outcome["difference_classification"]["scientific"] == {}
    assert outcome["difference_classification"]["operational"] == {
        "preferred_compute": ["local", "Trex", "Akib"],
        "distribution": "two jobs per host",
    }
    assert outcome["unresolved_scientific_choices"] == []
    assert outcome["next_step"] == "define_derived_study"
    assert outcome["request_metadata"] == {
        "experiment": "Conv1 best-rho continuation",
        "type": "continuation",
        "keep_fixed": "everything scientific",
    }
    # The public result is suitable for a later CLI or parser integration.
    json.dumps(outcome, allow_nan=False)


def test_structured_resolution_classifies_changes_conservatively() -> None:
    outcome = catalog_module.resolve_structured_request(
        _minimal_catalog(),
        {
            "parent": "test-experiment-v1",
            "cases": "all parent cases",
            "changes": {
                "epochs": 10,
                "preferred compute": "local",
                "mystery knob": 4,
            },
            "purpose": "Check a generic parser extension.",
        },
        Path("/unused"),
    )

    assert outcome["status"] == "derived"
    assert outcome["resolution_kind"] == (
        "derived_study_and_execution_attempt"
    )
    assert outcome["difference_classification"] == {
        "scientific": {"epochs": 10},
        "operational": {"preferred compute": "local"},
        "unclassified": {"mystery knob": 4},
    }
    assert outcome["unrecognized_request_fields"] == {
        "purpose": "Check a generic parser extension."
    }
    assert outcome["unresolved_scientific_choices"] == [
        "classify change field 'mystery knob' as scientific or operational",
        "classify structured request field 'purpose' before study definition",
    ]
    assert outcome["next_step"] == "resolve_scientific_choices"


def test_structured_operational_change_reuses_parent_science() -> None:
    outcome = catalog_module.resolve_structured_request(
        _minimal_catalog(),
        {
            "parent": "test route",
            "operational_changes": {"host": "Trex"},
        },
        Path("/unused"),
    )

    assert outcome["status"] == "resolved"
    assert outcome["resolution_kind"] == "execution_attempt"
    assert outcome["selection"] == {"mode": "all", "cases": None}
    assert outcome["next_step"] == "define_execution_attempt"


def test_structured_resolution_preserves_ambiguous_parent() -> None:
    catalog = catalog_module.load_catalog(CATALOG_PATH)
    outcome = catalog_module.resolve_structured_request(
        catalog,
        {
            "parent": "conv3 perfect diode",
            "cases_subset": ["one case"],
            "operational_changes": {"host": "local"},
        },
        REPO_ROOT,
    )

    assert outcome["status"] == "ambiguous"
    assert "parent" not in outcome
    assert "selected" not in outcome
    assert outcome["resolution_kind"] == "parent_resolution"
    assert len(outcome["candidates"]) >= 2


def test_structured_resolution_rejects_two_case_keys() -> None:
    with pytest.raises(
        catalog_module.CatalogError,
        match="^Expected only one of 'cases' or 'cases_subset'",
    ):
        catalog_module.resolve_structured_request(
            _minimal_catalog(),
            {
                "parent": "test-experiment-v1",
                "cases": ["case-a"],
                "cases_subset": ["case-b"],
            },
            Path("/unused"),
        )


def _minimal_catalog() -> dict:
    return {
        "schema_version": "experiment-catalog/v1",
        "entries": [
            {
                "experiment_id": "test-experiment-v1",
                "title": "Test experiment",
                "summary": "A small validation fixture.",
                "lifecycle": "current",
                "aliases": ["test route"],
                "scope": {
                    "architectures": ["conv1"],
                    "nonlinearities": ["perfect_diode"],
                    "datasets": ["ordinary_mnist"],
                    "tasks": ["diagnostic"],
                },
                "authority": {
                    "protocols": ["docs/protocol.md"],
                    "config": "configs/config.json",
                    "plan": None,
                },
                "execution": {
                    "launcher": "experiments/launcher.py",
                },
            }
        ],
    }


def _write_fixture_files(repo_root: Path) -> None:
    for relative_path, content in (
        ("docs/protocol.md", "# Protocol\n"),
        ("configs/config.json", "{}\n"),
        ("experiments/launcher.py", "# launcher\n"),
    ):
        path = repo_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def test_check_rejects_duplicate_alias_with_expected_first_error(
    tmp_path: Path,
) -> None:
    _write_fixture_files(tmp_path)
    malformed = _minimal_catalog()
    duplicate = deepcopy(malformed["entries"][0])
    duplicate["experiment_id"] = "test-experiment-v2"
    malformed["entries"].append(duplicate)

    with pytest.raises(
        catalog_module.CatalogError,
        match="^Expected globally unique normalized experiment IDs and aliases; got",
    ):
        catalog_module.validate_catalog(malformed, tmp_path)


def test_check_rejects_missing_authority_file(tmp_path: Path) -> None:
    _write_fixture_files(tmp_path)
    malformed = _minimal_catalog()
    malformed["entries"][0]["authority"]["config"] = "configs/missing.json"

    with pytest.raises(
        catalog_module.CatalogError,
        match="^Expected an existing file for configs/missing.json; got",
    ):
        catalog_module.validate_catalog(malformed, tmp_path)


def test_check_rejects_non_string_lifecycle_with_structured_error(
    tmp_path: Path,
) -> None:
    _write_fixture_files(tmp_path)
    malformed = _minimal_catalog()
    malformed["entries"][0]["lifecycle"] = []

    with pytest.raises(
        catalog_module.CatalogError,
        match="^Expected one of .* for entries\\[0\\]\\.lifecycle; got \\[\\]",
    ):
        catalog_module.validate_catalog(malformed, tmp_path)


def test_check_does_not_trust_declared_plan_approval(tmp_path: Path) -> None:
    _write_fixture_files(tmp_path)
    malformed = _minimal_catalog()
    plan_path = tmp_path / "docs" / "experiment_plans" / "test-experiment-v1.md"
    plan_path.parent.mkdir(parents=True)
    plan_path.write_text(
        "# Incomplete plan\n\n```json\n"
        + json.dumps(
            {
                "experiment_id": "test-experiment-v1",
                "approval": {"status": "approved"},
            }
        )
        + "\n```\n",
        encoding="utf-8",
    )
    malformed["entries"][0]["authority"]["plan"] = (
        "docs/experiment_plans/test-experiment-v1.md"
    )

    with pytest.raises(
        catalog_module.CatalogError,
        match="^Expected a valid linked experiment plan at",
    ):
        catalog_module.validate_catalog(malformed, tmp_path)


def test_cli_check_accepts_catalog_override(tmp_path: Path, capsys) -> None:
    _write_fixture_files(tmp_path)
    catalog_path = tmp_path / "experiments" / "experiment_catalog.json"
    catalog_path.write_text(
        json.dumps(_minimal_catalog(), indent=2) + "\n",
        encoding="utf-8",
    )

    exit_code = catalog_module.main(["--catalog", str(catalog_path), "check"])
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output["status"] == "ok"
    assert output["plan_contract_states"] == {"missing": 1}
