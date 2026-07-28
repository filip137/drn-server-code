#!/usr/bin/env python3
"""Audit repository-level experiment control-plane compatibility.

This command is deliberately read-only.  It complements unit tests by checking
the real plan corpus, catalog, tracker, and launch-authority worktree together.
Use ``--strict`` to turn blocking findings into a nonzero exit status.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
from types import ModuleType
from typing import Any


REPORT_SCHEMA = "experiment-control-plane-audit/v1"
AUTHORITY_PREFIXES = (
    "configs/dispatch/",
    "docs/experiment_plans/",
    "experiments/experiment_catalog.",
    "skills/run-experiment-pipeline/",
    "skills/scheduled-run-preflight/",
)
AUTHORITY_PATHS = {
    "AGENTS.md",
    "docs/AGENTS.md",
    "docs/current_experiments.md",
    "docs/experiment_launch_request.md",
    "docs/local_dispatch.md",
    "experiments/AGENTS.md",
    "experiments/local_dispatch.py",
}
TRACKER_MARKER_RE = re.compile(
    r"(?m)^<!-- experiment-id: (?P<experiment_id>[a-z0-9][a-z0-9._-]*) -->$"
)


class AuditError(RuntimeError):
    """Raised when the audit itself cannot inspect a required authority."""


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AuditError(
            f"Expected an importable Python module at {path}; "
            f"provided value: {spec!r}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _is_authority_path(path: str) -> bool:
    return path in AUTHORITY_PATHS or any(
        path.startswith(prefix) for prefix in AUTHORITY_PREFIXES
    )


def _git_worktree(repo_root: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "git",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise AuditError(
            "Expected git status to succeed for the repository; "
            f"provided value: exit={completed.returncode}, "
            f"stderr={completed.stderr.strip()!r}"
        )

    records: list[dict[str, str]] = []
    for line in completed.stdout.splitlines():
        if len(line) < 4:
            continue
        status = line[:2]
        path = line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        records.append({"status": status, "path": path})
    authority_records = [
        record for record in records if _is_authority_path(record["path"])
    ]
    return {
        "changed_path_count": len(records),
        "authority_changed_path_count": len(authority_records),
        "authority_changes": authority_records,
    }


def _plan_corpus(repo_root: Path) -> dict[str, Any]:
    validator_path = (
        repo_root
        / "skills"
        / "run-experiment-pipeline"
        / "scripts"
        / "validate_experiment_plan.py"
    )
    validator = _load_module("_control_plane_plan_validator", validator_path)
    validator.REPO_ROOT = repo_root

    plan_dir = repo_root / "docs" / "experiment_plans"
    plans = sorted(plan_dir.glob("*.md"))
    records: list[dict[str, Any]] = []
    for plan_path in plans:
        plan: dict[str, Any] | None = None
        record: dict[str, Any] = {
            "path": str(plan_path.relative_to(repo_root)),
            "experiment_id": plan_path.stem,
            "declared_schema": None,
            "approval_status": None,
            "canonical_status": "failed",
            "canonical_error": None,
            "legacy_compatibility_status": "not_available",
            "legacy_compatibility_error": None,
        }
        try:
            plan = validator.load_plan(plan_path)
            record["declared_schema"] = plan.get("schema_version")
            approval = plan.get("approval")
            if isinstance(approval, dict):
                record["approval_status"] = approval.get("status")
            validator.validate_plan(
                plan_path,
                plan,
                require_approved=False,
                verify_files=True,
            )
            record["canonical_status"] = "passed"
        except Exception as exc:  # validators expose their own typed errors
            record["canonical_error"] = str(exc)

        if plan is None:
            records.append(record)
            continue
        try:
            validator.validate_plan(
                plan_path,
                plan,
                require_approved=False,
                verify_files=True,
                allow_legacy_receipt_contract_for_catalog=True,
            )
            record["legacy_compatibility_status"] = "passed"
        except TypeError:
            record["legacy_compatibility_status"] = "not_available"
        except Exception as exc:
            record["legacy_compatibility_status"] = "failed"
            record["legacy_compatibility_error"] = str(exc)
        records.append(record)

    return {
        "plan_count": len(records),
        "canonical_pass_count": sum(
            record["canonical_status"] == "passed" for record in records
        ),
        "legacy_compatibility_pass_count": sum(
            record["legacy_compatibility_status"] == "passed"
            for record in records
        ),
        "approval_states": dict(
            sorted(
                Counter(
                    str(record["approval_status"]) for record in records
                ).items()
            )
        ),
        "records": records,
    }


def _catalog(repo_root: Path) -> dict[str, Any]:
    module = _load_module(
        "_control_plane_catalog",
        repo_root / "experiments" / "experiment_catalog.py",
    )
    module.REPO_ROOT = repo_root
    catalog_path = repo_root / "experiments" / "experiment_catalog.json"
    value = module.load_catalog(catalog_path)
    warnings = module.validate_catalog(value, repo_root)
    entries = value["entries"]
    states = Counter(
        module.plan_contract(entry, repo_root)["state"] for entry in entries
    )
    return {
        "status": "passed",
        "entry_count": len(entries),
        "experiment_ids": [entry["experiment_id"] for entry in entries],
        "plan_contract_states": dict(sorted(states.items())),
        "warnings": warnings,
    }


def _tracker(repo_root: Path) -> dict[str, Any]:
    module = _load_module(
        "_control_plane_tracker_validator",
        repo_root
        / "skills"
        / "run-experiment-pipeline"
        / "scripts"
        / "validate_current_experiments.py",
    )
    tracker_path = repo_root / "docs" / "current_experiments.md"
    text = tracker_path.read_text(encoding="utf-8")
    entries = module.validate_tracker(text)
    marker_ids = TRACKER_MARKER_RE.findall(text)
    return {
        "status": "passed",
        "entry_count": len(entries),
        "experiment_ids": marker_ids,
        "states": dict(
            sorted(Counter(entry.status for entry in entries.values()).items())
        ),
    }


def audit(repo_root: Path) -> dict[str, Any]:
    root = repo_root.expanduser().resolve()
    required = (
        root / "AGENTS.md",
        root / "docs" / "experiment_plans",
        root / "experiments" / "experiment_catalog.json",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise AuditError(
            "Expected a server_code repository with experiment authorities; "
            f"provided value: missing={missing!r}, repo_root={str(root)!r}"
        )

    plans = _plan_corpus(root)
    catalog = _catalog(root)
    tracker = _tracker(root)
    worktree = _git_worktree(root)

    catalog_ids = set(catalog["experiment_ids"])
    tracker_ids = set(tracker["experiment_ids"])
    tracker_without_catalog = sorted(tracker_ids - catalog_ids)
    canonical_approved = {
        record["experiment_id"]
        for record in plans["records"]
        if record["canonical_status"] == "passed"
        and record["approval_status"] == "approved"
    }
    catalog_canonical_approved = sorted(catalog_ids & canonical_approved)

    checks = [
        {
            "id": "declared-plan-schema-corpus",
            "status": (
                "passed"
                if plans["canonical_pass_count"] == plans["plan_count"]
                else "failed"
            ),
            "blocking": True,
            "summary": (
                f"{plans['canonical_pass_count']}/{plans['plan_count']} real "
                "plans pass the current canonical validator with file hashes."
            ),
        },
        {
            "id": "catalog-integrity",
            "status": catalog["status"],
            "blocking": True,
            "summary": (
                f"{catalog['entry_count']} catalog entries parsed; "
                f"states={catalog['plan_contract_states']}."
            ),
        },
        {
            "id": "tracker-integrity",
            "status": tracker["status"],
            "blocking": True,
            "summary": (
                f"{tracker['entry_count']} live tracker entries parsed; "
                f"states={tracker['states']}."
            ),
        },
        {
            "id": "tracker-catalog-identity",
            "status": "failed" if tracker_without_catalog else "passed",
            "blocking": True,
            "summary": (
                "Every tracker experiment ID resolves exactly through the "
                f"catalog; missing={tracker_without_catalog}."
            ),
        },
        {
            "id": "canonical-approved-catalog-route",
            "status": "passed" if catalog_canonical_approved else "failed",
            "blocking": True,
            "summary": (
                "At least one catalog route binds an approved plan accepted "
                f"by the canonical validator; routes={catalog_canonical_approved}."
            ),
        },
        {
            "id": "authority-worktree-release-boundary",
            "status": (
                "failed"
                if worktree["authority_changed_path_count"]
                else "passed"
            ),
            "blocking": True,
            "summary": (
                f"{worktree['authority_changed_path_count']} launch-authority "
                "paths are modified or untracked."
            ),
        },
    ]
    blockers = [
        check["id"]
        for check in checks
        if check["blocking"] and check["status"] != "passed"
    ]
    return {
        "schema_version": REPORT_SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(root),
        "status": "blocked" if blockers else "passed",
        "blockers": blockers,
        "checks": checks,
        "details": {
            "plans": plans,
            "catalog": catalog,
            "tracker": tracker,
            "worktree": worktree,
            "tracker_without_catalog": tracker_without_catalog,
            "catalog_canonical_approved": catalog_canonical_approved,
        },
    }


def _human_report(report: dict[str, Any]) -> str:
    lines = [
        f"Experiment control-plane audit: {report['status']}",
        f"Repository: {report['repo_root']}",
    ]
    for check in report["checks"]:
        lines.append(
            f"[{check['status'].upper()}] {check['id']}: {check['summary']}"
        )
    if report["blockers"]:
        lines.append("Blocking checks: " + ", ".join(report["blockers"]))
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero when any blocking repository-level check fails.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = audit(args.repo_root)
    except (AuditError, OSError, ValueError) as exc:
        print(f"Experiment control-plane audit failed: {exc}", file=sys.stderr)
        return 2
    if args.as_json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(_human_report(report))
    return 1 if args.strict and report["blockers"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
