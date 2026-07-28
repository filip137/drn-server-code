#!/usr/bin/env python3
"""Read-only experiment catalog for routing launch requests."""

from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = Path(__file__).with_name("experiment_catalog.json")
PLAN_VALIDATOR = (
    REPO_ROOT
    / "skills"
    / "run-experiment-pipeline"
    / "scripts"
    / "validate_experiment_plan.py"
)
ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
JSON_BLOCK_RE = re.compile(r"```json[ \t]*\n(.*?)\n```", re.DOTALL)
LIFECYCLES = {"current", "draft", "deprecated"}
STOP_WORDS = {
    "a",
    "an",
    "experiment",
    "experiments",
    "for",
    "launch",
    "on",
    "please",
    "run",
    "the",
    "use",
    "using",
    "with",
}
LEGACY_PREFLIGHT_KEYS = {
    "smoke_required",
    "tk_reference_required",
    "scheduled_run_preflight_required",
    "receipt_path",
}
SCIENTIFIC_CHANGE_FIELDS = {
    "amplification",
    "amplification_scheme",
    "architecture",
    "architectures",
    "batch_size",
    "budget",
    "cases",
    "checkpoint_policy",
    "data_split",
    "dataset",
    "decision_rule",
    "epochs",
    "evidence_scope",
    "gradient_steps",
    "initialization",
    "input_gain",
    "k",
    "learning_rate",
    "learning_rates",
    "loss",
    "model",
    "nonlinearity",
    "official_test_policy",
    "optimizer",
    "optimizers",
    "rho",
    "rho_values",
    "seed",
    "seeds",
    "steps",
    "sweep",
    "t",
    "tk",
    "training_evaluation_budget",
}
OPERATIONAL_CHANGE_FIELDS = {
    "adapt_environment",
    "collection_destination",
    "compute",
    "compute_distribution",
    "concurrency",
    "cuda",
    "device",
    "distribution",
    "distribution_preference",
    "environment",
    "executor",
    "host",
    "hosts",
    "lane",
    "lanes",
    "launch_after_checks",
    "output_directory",
    "output_path",
    "preferred_compute",
    "python",
    "runner",
    "staging_path",
    "target",
    "targets",
    "tmux",
}
ALL_CASES_VALUES = {
    "all",
    "all cases",
    "all parent cases",
    "inherit",
    "inherit from parent",
}


class CatalogError(ValueError):
    """Raised when the catalog violates its small v1 contract."""


def _require(condition: bool, expected: str, provided: Any) -> None:
    if not condition:
        raise CatalogError(f"Expected {expected}; got {provided!r}")


def _require_keys(
    value: Any,
    *,
    required: set[str],
    context: str,
) -> Mapping[str, Any]:
    _require(isinstance(value, dict), f"an object for {context}", type(value).__name__)
    missing = required - set(value)
    unknown = set(value) - required
    _require(not missing, f"all required keys in {context}", sorted(missing))
    _require(not unknown, f"only documented keys in {context}", sorted(unknown))
    return value


def _require_text(value: Any, context: str) -> str:
    _require(
        isinstance(value, str) and bool(value.strip()),
        f"non-empty text for {context}",
        value,
    )
    return value


def _require_string_list(
    value: Any,
    context: str,
    *,
    allow_empty: bool = False,
) -> list[str]:
    _require(
        isinstance(value, list)
        and (allow_empty or bool(value))
        and all(isinstance(item, str) and bool(item.strip()) for item in value),
        f"{'a' if allow_empty else 'a non-empty'} list of strings for {context}",
        value,
    )
    normalized = [_normalize(item) for item in value]
    _require(
        len(normalized) == len(set(normalized)),
        f"unique normalized strings for {context}",
        value,
    )
    return value


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def _field_key(value: str) -> str:
    return _normalize(value).replace(" ", "_")


def _tokens(value: str) -> set[str]:
    return {
        token
        for token in _normalize(value).split()
        if token and token not in STOP_WORDS
    }


def _repo_relative_path(value: Any, context: str) -> Path:
    text = _require_text(value, context)
    path = Path(text)
    _require(not path.is_absolute(), f"a repository-relative path for {context}", text)
    _require(".." not in path.parts, f"a path without '..' for {context}", text)
    return path


def _load_plan(path: Path) -> Mapping[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CatalogError(f"Expected a readable plan at {path}; got {exc}") from exc
    blocks = JSON_BLOCK_RE.findall(text)
    _require(len(blocks) == 1, f"one fenced JSON contract in {path}", len(blocks))
    try:
        value = json.loads(blocks[0])
    except json.JSONDecodeError as exc:
        raise CatalogError(f"Expected valid plan JSON in {path}; got {exc}") from exc
    _require(isinstance(value, dict), f"a JSON object in {path}", type(value).__name__)
    return value


def _validate_linked_plan(
    plan_path: Path,
    *,
    repo_root: Path,
    require_approved: bool,
) -> str:
    try:
        spec = importlib.util.spec_from_file_location(
            "_experiment_catalog_plan_validator",
            PLAN_VALIDATOR,
        )
        _require(
            spec is not None and spec.loader is not None,
            f"an importable plan validator at {PLAN_VALIDATOR}",
            spec,
        )
        validator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(validator)
    except (OSError, ImportError) as exc:
        raise CatalogError(
            f"Expected an importable plan validator at {PLAN_VALIDATOR}; got {exc}"
        ) from exc

    validator.REPO_ROOT = repo_root
    try:
        plan = validator.load_plan(plan_path)
        validator.validate_plan(
            plan_path,
            plan,
            require_approved=require_approved,
            verify_files=require_approved,
            allow_legacy_receipt_contract_for_catalog=True,
        )
        return (
            "legacy_receipt_contract"
            if _has_exact_legacy_receipt_contract(plan)
            else "canonical_receipt_contract"
        )
    except validator.PlanError as exc:
        raise CatalogError(
            f"Expected a valid linked experiment plan at {plan_path}; got {exc}"
        ) from exc


def _has_exact_legacy_receipt_contract(plan: Mapping[str, Any]) -> bool:
    execution = plan.get("execution")
    if not isinstance(execution, dict):
        return False
    preflight = execution.get("preflight")
    return (
        isinstance(preflight, dict)
        and set(preflight) == LEGACY_PREFLIGHT_KEYS
    )


def _is_request_contract_path(value: str) -> bool:
    return Path(value).parts[:2] == ("docs", "experiment_requests")


def _validate_approved_request_contract(path: Path) -> None:
    from experiments import experiment_request

    try:
        request = experiment_request.load_request(path)
        validation = experiment_request.validate_request(
            request,
            require_review=True,
        )
    except (OSError, ValueError, TypeError) as exc:
        raise CatalogError(
            f"Expected a valid approved request contract at {path}; got {exc}"
        ) from exc
    if validation.get("status") != "valid":
        raise CatalogError(
            "Expected an approved request contract with valid user fields and "
            f"review at {path}; got {validation!r}"
        )


def load_catalog(path: Path = DEFAULT_CATALOG) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise CatalogError(f"Expected a readable catalog at {path}; got {exc}") from exc
    except json.JSONDecodeError as exc:
        raise CatalogError(f"Expected valid catalog JSON in {path}; got {exc}") from exc
    _require(isinstance(value, dict), "a JSON object for the catalog", type(value).__name__)
    return value


def plan_contract(entry: Mapping[str, Any], repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    plan_value = entry["authority"]["plan"]
    if plan_value is None:
        return {"state": "missing", "plan": None}
    plan_path = repo_root / Path(plan_value)
    if _is_request_contract_path(plan_value):
        _validate_approved_request_contract(plan_path)
        return {"state": "approved", "plan": plan_value}
    plan = _load_plan(plan_path)
    approval = plan.get("approval")
    state = approval.get("status") if isinstance(approval, dict) else None
    if state == "approved" and _has_exact_legacy_receipt_contract(plan):
        state = "legacy_receipt_contract"
    return {"state": state or "invalid", "plan": plan_value}


def next_step(entry: Mapping[str, Any], repo_root: Path = REPO_ROOT) -> str:
    lifecycle = entry["lifecycle"]
    contract_state = plan_contract(entry, repo_root)["state"]
    if lifecycle == "deprecated":
        return "do_not_launch"
    if lifecycle == "draft":
        return "complete_draft_and_freeze_plan"
    if contract_state == "missing":
        return "freeze_approved_plan"
    if contract_state == "draft":
        return "obtain_plan_approval"
    if contract_state == "approved":
        return "check_tracker_and_preflight"
    return "repair_catalog_or_plan"


def validate_catalog(
    catalog: Mapping[str, Any],
    repo_root: Path = REPO_ROOT,
) -> list[str]:
    _require_keys(
        catalog,
        required={"schema_version", "entries"},
        context="catalog",
    )
    _require(
        catalog["schema_version"] == "experiment-catalog/v1",
        "schema_version 'experiment-catalog/v1'",
        catalog["schema_version"],
    )
    entries = catalog["entries"]
    _require(isinstance(entries, list) and bool(entries), "a non-empty entries list", entries)

    seen_ids: set[str] = set()
    resolution_keys: dict[str, str] = {}
    warnings: list[str] = []

    for index, raw_entry in enumerate(entries):
        context = f"entries[{index}]"
        entry = _require_keys(
            raw_entry,
            required={
                "experiment_id",
                "title",
                "summary",
                "lifecycle",
                "aliases",
                "scope",
                "authority",
                "execution",
            },
            context=context,
        )
        experiment_id = _require_text(entry["experiment_id"], f"{context}.experiment_id")
        _require(
            ID_RE.fullmatch(experiment_id) is not None,
            f"a filesystem-safe identifier for {context}.experiment_id",
            experiment_id,
        )
        _require(experiment_id not in seen_ids, "unique experiment IDs", experiment_id)
        seen_ids.add(experiment_id)
        _require_text(entry["title"], f"{context}.title")
        _require_text(entry["summary"], f"{context}.summary")
        _require(
            isinstance(entry["lifecycle"], str)
            and entry["lifecycle"] in LIFECYCLES,
            f"one of {sorted(LIFECYCLES)} for {context}.lifecycle",
            entry["lifecycle"],
        )
        aliases = _require_string_list(entry["aliases"], f"{context}.aliases")

        for raw_key in [experiment_id, *aliases]:
            key = _normalize(raw_key)
            _require(bool(key), f"a searchable identifier or alias for {context}", raw_key)
            previous = resolution_keys.get(key)
            _require(
                previous is None,
                "globally unique normalized experiment IDs and aliases",
                {"value": raw_key, "owners": [previous, experiment_id]},
            )
            resolution_keys[key] = experiment_id

        scope = _require_keys(
            entry["scope"],
            required={"architectures", "nonlinearities", "datasets", "tasks"},
            context=f"{context}.scope",
        )
        for key in ("architectures", "nonlinearities", "datasets", "tasks"):
            _require_string_list(scope[key], f"{context}.scope.{key}")

        authority = _require_keys(
            entry["authority"],
            required={"protocols", "config", "plan"},
            context=f"{context}.authority",
        )
        protocols = _require_string_list(
            authority["protocols"],
            f"{context}.authority.protocols",
        )
        required_paths = [
            _repo_relative_path(path, f"{context}.authority.protocols")
            for path in protocols
        ]
        required_paths.append(
            _repo_relative_path(authority["config"], f"{context}.authority.config")
        )

        plan_value = authority["plan"]
        _require(
            plan_value is None or isinstance(plan_value, str),
            f"null or a repository-relative path for {context}.authority.plan",
            plan_value,
        )
        plan_path: Path | None = None
        if plan_value is not None:
            plan_path = _repo_relative_path(plan_value, f"{context}.authority.plan")
            required_paths.append(plan_path)

        execution = _require_keys(
            entry["execution"],
            required={"launcher"},
            context=f"{context}.execution",
        )
        required_paths.append(
            _repo_relative_path(execution["launcher"], f"{context}.execution.launcher")
        )

        for relative_path in required_paths:
            full_path = repo_root / relative_path
            _require(
                full_path.is_file(),
                f"an existing file for {relative_path}",
                str(full_path),
            )

        if plan_path is None:
            warnings.append(
                f"{experiment_id}: no approved plan is linked; resolution is planning-only"
            )
        elif _is_request_contract_path(plan_value):
            _validate_approved_request_contract(repo_root / plan_path)
        else:
            plan = _load_plan(repo_root / plan_path)
            _require(
                plan.get("experiment_id") == experiment_id,
                f"plan experiment_id {experiment_id!r}",
                plan.get("experiment_id"),
            )
            approval = plan.get("approval")
            approval_status = (
                approval.get("status") if isinstance(approval, dict) else None
            )
            _require(
                isinstance(approval, dict)
                and isinstance(approval_status, str)
                and approval_status in {"draft", "approved"},
                "plan approval.status 'draft' or 'approved'",
                approval,
            )
            receipt_contract_state = _validate_linked_plan(
                repo_root / plan_path,
                repo_root=repo_root,
                require_approved=approval_status == "approved",
            )
            if receipt_contract_state == "legacy_receipt_contract":
                warnings.append(
                    f"{experiment_id}: linked plan uses the legacy receipt "
                    "contract; exact full-plan launch remains blocked until "
                    "the catalog or plan is repaired, but an independently "
                    "validated simplified continuation may still derive from "
                    "this immutable parent"
                )
            if approval_status != "approved":
                warnings.append(
                    f"{experiment_id}: linked plan is not approved; launch remains blocked"
                )

    return warnings


def _entry_tokens(entry: Mapping[str, Any]) -> set[str]:
    values = [
        entry["experiment_id"],
        entry["title"],
        entry["summary"],
        *entry["aliases"],
        *entry["scope"]["architectures"],
        *entry["scope"]["nonlinearities"],
        *entry["scope"]["datasets"],
        *entry["scope"]["tasks"],
    ]
    return set().union(*(_tokens(value) for value in values))


def _candidate(entry: Mapping[str, Any], repo_root: Path) -> dict[str, Any]:
    return {
        "experiment_id": entry["experiment_id"],
        "title": entry["title"],
        "lifecycle": entry["lifecycle"],
        "plan_contract": plan_contract(entry, repo_root),
        "next_step": next_step(entry, repo_root),
    }


def public_entry(entry: Mapping[str, Any], repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    rendered = deepcopy(dict(entry))
    rendered["plan_contract"] = plan_contract(entry, repo_root)
    rendered["next_step"] = next_step(entry, repo_root)
    return rendered


def resolve(
    catalog: Mapping[str, Any],
    query: str,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    normalized_query = _normalize(query)
    _require(bool(normalized_query), "a non-empty resolution query", query)
    entries = catalog["entries"]

    exact_matches = [
        entry
        for entry in entries
        if normalized_query
        in {_normalize(entry["experiment_id"]), *map(_normalize, entry["aliases"])}
    ]
    if exact_matches:
        entry = exact_matches[0]
        return {
            "status": "resolved",
            "query": query,
            "match": "exact",
            "selected": public_entry(entry, repo_root),
        }

    query_tokens = _tokens(query)
    _require(bool(query_tokens), "at least one searchable query token", query)
    matches = (
        [entry for entry in entries if query_tokens <= _entry_tokens(entry)]
        if len(query_tokens) >= 2
        else []
    )
    if len(matches) == 1:
        return {
            "status": "resolved",
            "query": query,
            "match": "all_query_tokens",
            "selected": public_entry(matches[0], repo_root),
        }
    if len(matches) > 1:
        return {
            "status": "ambiguous",
            "query": query,
            "match": "all_query_tokens",
            "candidates": [_candidate(entry, repo_root) for entry in matches],
        }

    suggestions: list[tuple[float, int, str, Mapping[str, Any], set[str]]] = []
    for entry in entries:
        matched = query_tokens & _entry_tokens(entry)
        if matched:
            suggestions.append(
                (
                    len(matched) / len(query_tokens),
                    len(matched),
                    entry["experiment_id"],
                    entry,
                    matched,
                )
            )
    suggestions.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return {
        "status": "not_found",
        "query": query,
        "reason": (
            "A descriptive query needs at least two searchable tokens."
            if len(query_tokens) < 2
            else "No entry matched every searchable query token."
        ),
        "suggestions": [
            {
                **_candidate(entry, repo_root),
                "matched_tokens": sorted(matched),
                "unmatched_tokens": sorted(query_tokens - matched),
            }
            for _score, _count, _identifier, entry, matched in suggestions[:3]
        ],
    }


def _request_mapping(
    value: Any,
    *,
    context: str,
    allow_empty: bool = True,
) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"an object for {context}", value)
    _require(
        allow_empty or bool(value),
        f"a non-empty object for {context}",
        value,
    )
    rendered: dict[str, Any] = {}
    normalized_keys: set[str] = set()
    for raw_key, item in value.items():
        key = _require_text(raw_key, f"{context} field name")
        normalized = _field_key(key)
        _require(
            normalized not in normalized_keys,
            f"unique normalized field names for {context}",
            list(value),
        )
        normalized_keys.add(normalized)
        try:
            serialized = json.dumps(item, allow_nan=False)
            rendered[key] = json.loads(serialized)
        except (TypeError, ValueError) as exc:
            raise CatalogError(
                f"Expected JSON-serializable values for {context}; "
                f"got {item!r}"
            ) from exc
    return rendered


def _explicit_change_mapping(
    value: Any,
    *,
    context: str,
    field_prefix: str,
) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, str):
        text = _require_text(value, context)
        if _normalize(text) in {
            "inherit",
            "inherit from parent",
            "no change",
            "no changes",
            "none",
        }:
            return {}
        return {f"{field_prefix}_description": text}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        _require(
            bool(value)
            and all(isinstance(item, str) and bool(item.strip()) for item in value),
            f"a mapping, non-empty text, or list of non-empty strings for {context}",
            value,
        )
        return {
            f"{field_prefix}_{index}": item
            for index, item in enumerate(value, start=1)
        }
    return _request_mapping(value, context=context)


def _unresolved_choices(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = _require_text(value, "unresolved_scientific_choices")
        return [] if _normalize(text) == "none" else [text]
    _require(
        isinstance(value, list)
        and all(
            isinstance(item, str) and bool(item.strip())
            for item in value
        ),
        "non-empty text, 'none', or a list of non-empty strings for "
        "unresolved_scientific_choices",
        value,
    )
    return list(value)


def _structured_cases(value: Any) -> dict[str, Any]:
    if value is None:
        return {"mode": "all", "cases": None}
    if isinstance(value, str):
        text = _require_text(value, "structured request cases")
        if _normalize(text) in ALL_CASES_VALUES:
            return {"mode": "all", "cases": None}
        cases = [text]
    else:
        _require(
            isinstance(value, Sequence)
            and not isinstance(value, (str, bytes))
            and bool(value),
            "a non-empty case string, case list, or 'all parent cases'",
            value,
        )
        cases = [
            _require_text(item, "structured request case")
            for item in value
        ]
    normalized = [_normalize(item) for item in cases]
    _require(
        len(normalized) == len(set(normalized)),
        "unique normalized structured request cases",
        cases,
    )
    return {"mode": "subset", "cases": cases}


def _merge_classified_changes(
    target: dict[str, Any],
    additions: Mapping[str, Any],
    *,
    classification: str,
    ownership: dict[str, str],
) -> None:
    for raw_key, value in additions.items():
        normalized = _field_key(raw_key)
        previous = ownership.get(normalized)
        _require(
            previous is None or previous == classification,
            "each change field to have exactly one scientific or operational classification",
            {
                "field": raw_key,
                "classifications": [previous, classification],
            },
        )
        ownership[normalized] = classification
        target[raw_key] = deepcopy(value)


def resolve_structured_request(
    catalog: Mapping[str, Any],
    request: Mapping[str, Any],
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Resolve a parent plus compositional differences without inventing science.

    ``request`` is deliberately independent of any Markdown intake parser. Its
    small mapping contract is:

    - ``parent`` (required): catalog ID, alias, or descriptive parent query;
    - ``cases`` / ``cases_subset``: a case string/list, or
      ``"all parent cases"``;
    - ``changes``: fields to classify using the catalog's conservative field
      vocabulary;
    - ``scientific_changes`` / ``operational_changes``: explicitly classified
      fields, including domain-specific fields outside that vocabulary; and
    - ``unresolved_scientific_choices``: user-owned choices still needing an
      answer.

    Unknown generic change fields are never guessed to be operational. They
    are returned as unclassified changes and unresolved scientific choices.
    """

    value = _request_mapping(
        request,
        context="structured request",
        allow_empty=False,
    )
    allowed_keys = {
        "experiment",
        "type",
        "parent",
        "cases",
        "cases_subset",
        "changes",
        "scientific_changes",
        "keep_fixed",
        "operational_changes",
        "unresolved_scientific_choices",
    }
    missing = {"parent"} - set(value)
    unknown = set(value) - allowed_keys
    _require(not missing, "a structured request containing 'parent'", missing)

    parent_query = _require_text(value["parent"], "structured request parent")
    parent_outcome = resolve(catalog, parent_query, repo_root)
    if parent_outcome["status"] != "resolved":
        # Preserve the ordinary resolver's fail-closed ambiguity and
        # not-found diagnostics. The structured fields must never make an
        # uncertain parent look exact.
        return {
            **parent_outcome,
            "resolution_kind": "parent_resolution",
            "structured_request": deepcopy(value),
        }

    _require(
        not ("cases" in value and "cases_subset" in value),
        "only one of 'cases' or 'cases_subset' in a structured request",
        {
            "cases": value.get("cases"),
            "cases_subset": value.get("cases_subset"),
        },
    )
    selection = _structured_cases(
        value.get("cases", value.get("cases_subset"))
    )
    generic_changes = _request_mapping(
        value.get("changes", {}),
        context="structured request changes",
    )
    explicit_scientific = _explicit_change_mapping(
        value.get("scientific_changes", {}),
        context="structured request scientific_changes",
        field_prefix="scientific_change",
    )
    explicit_operational = _explicit_change_mapping(
        value.get("operational_changes", {}),
        context="structured request operational_changes",
        field_prefix="operational_change",
    )
    explicit_unresolved = _unresolved_choices(
        value.get("unresolved_scientific_choices", [])
    )

    scientific: dict[str, Any] = {}
    operational: dict[str, Any] = {}
    unclassified: dict[str, Any] = {}
    ownership: dict[str, str] = {}
    _merge_classified_changes(
        scientific,
        explicit_scientific,
        classification="scientific",
        ownership=ownership,
    )
    _merge_classified_changes(
        operational,
        explicit_operational,
        classification="operational",
        ownership=ownership,
    )
    for raw_key, item in generic_changes.items():
        normalized = _field_key(raw_key)
        if normalized in SCIENTIFIC_CHANGE_FIELDS:
            _merge_classified_changes(
                scientific,
                {raw_key: item},
                classification="scientific",
                ownership=ownership,
            )
        elif normalized in OPERATIONAL_CHANGE_FIELDS:
            _merge_classified_changes(
                operational,
                {raw_key: item},
                classification="operational",
                ownership=ownership,
            )
        else:
            unclassified[raw_key] = deepcopy(item)

    unresolved = list(explicit_unresolved)
    unresolved.extend(
        f"classify change field {raw_key!r} as scientific or operational"
        for raw_key in unclassified
    )
    unresolved.extend(
        f"classify structured request field {raw_key!r} before study definition"
        for raw_key in sorted(unknown)
    )
    normalized_unresolved: set[str] = set()
    unique_unresolved: list[str] = []
    for item in unresolved:
        normalized = _normalize(item)
        if normalized not in normalized_unresolved:
            normalized_unresolved.add(normalized)
            unique_unresolved.append(item)

    has_study_derivation = selection["mode"] == "subset" or bool(scientific)
    has_execution_derivation = bool(operational)
    if has_study_derivation and has_execution_derivation:
        derivation_kind = "derived_study_and_execution_attempt"
    elif has_study_derivation:
        derivation_kind = "derived_study"
    elif has_execution_derivation:
        derivation_kind = "execution_attempt"
    else:
        derivation_kind = "reuse_parent"

    status = "derived" if has_study_derivation else "resolved"
    if unique_unresolved:
        structured_next_step = "resolve_scientific_choices"
    elif has_study_derivation:
        structured_next_step = "define_derived_study"
    elif has_execution_derivation:
        structured_next_step = "define_execution_attempt"
    else:
        structured_next_step = parent_outcome["selected"]["next_step"]

    return {
        "status": status,
        "query": parent_query,
        "match": parent_outcome["match"],
        "resolution_kind": derivation_kind,
        "parent": parent_outcome["selected"],
        "selection": selection,
        "difference_classification": {
            "scientific": scientific,
            "operational": operational,
            "unclassified": unclassified,
        },
        "request_metadata": {
            key: deepcopy(value[key])
            for key in ("experiment", "type", "keep_fixed")
            if key in value
        },
        "unrecognized_request_fields": {
            key: deepcopy(value[key])
            for key in sorted(unknown)
        },
        "unresolved_scientific_choices": unique_unresolved,
        "next_step": structured_next_step,
    }


def _catalog_repo_root(path: Path) -> Path:
    resolved = path.resolve()
    if resolved.parent.name == "experiments":
        return resolved.parent.parent
    return resolved.parent


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="List, inspect, resolve, and validate known experiment workflows."
    )
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list")
    show = commands.add_parser("show")
    show.add_argument("experiment_id")
    resolve_command = commands.add_parser("resolve")
    resolve_command.add_argument("query", nargs="+")
    commands.add_parser("check")
    return parser


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    catalog_path = args.catalog.resolve()
    repo_root = _catalog_repo_root(catalog_path)
    try:
        catalog = load_catalog(catalog_path)
        warnings = validate_catalog(catalog, repo_root)
    except CatalogError as exc:
        _print(
            {
                "status": "invalid",
                "catalog_path": str(catalog_path),
                "error": str(exc),
            }
        )
        return 1

    if args.command == "check":
        states: dict[str, int] = {}
        for entry in catalog["entries"]:
            state = plan_contract(entry, repo_root)["state"]
            states[state] = states.get(state, 0) + 1
        _print(
            {
                "status": "ok",
                "schema_version": catalog["schema_version"],
                "catalog_path": str(catalog_path),
                "entry_count": len(catalog["entries"]),
                "plan_contract_states": states,
                "warnings": warnings,
            }
        )
        return 0

    if args.command == "list":
        _print(
            {
                "status": "ok",
                "schema_version": catalog["schema_version"],
                "entries": [
                    _candidate(entry, repo_root) for entry in catalog["entries"]
                ],
            }
        )
        return 0

    if args.command == "show":
        entry = next(
            (
                item
                for item in catalog["entries"]
                if item["experiment_id"] == args.experiment_id
            ),
            None,
        )
        if entry is None:
            _print(
                {
                    "status": "not_found",
                    "experiment_id": args.experiment_id,
                }
            )
            return 2
        _print({"status": "ok", "entry": public_entry(entry, repo_root)})
        return 0

    if args.command == "resolve":
        try:
            outcome = resolve(catalog, " ".join(args.query), repo_root)
        except CatalogError as exc:
            _print({"status": "invalid_query", "error": str(exc)})
            return 2
        _print(outcome)
        return {"resolved": 0, "ambiguous": 3, "not_found": 4}[outcome["status"]]

    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
