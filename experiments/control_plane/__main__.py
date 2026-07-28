"""CLI for the side-by-side experiment control-plane contracts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

from .attempt import load_attempt, validate_attempt
from .common import ContractError
from .coverage import validate_attempt_set
from .legacy import publish_legacy_split, split_legacy_plan
from .study import load_study, validate_study


def _emit(value: Any, *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(value, indent=2, sort_keys=True))
        return
    if isinstance(value, dict):
        identity = (
            value.get("attempt_id")
            or value.get("study_id")
            or value.get("schema_version")
        )
        print(f"{value.get('status', 'ok')}: {identity}")
    else:
        print(value)


def _add_root(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--repo-root",
        default=".",
        help="repository root used to resolve contract bindings",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m experiments.control_plane",
        description=(
            "Validate separate scientific-study and execution-attempt "
            "contracts."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    study = commands.add_parser(
        "validate-study",
        help="validate an experiment-study/v1 JSON contract",
    )
    study.add_argument("path")
    _add_root(study)
    study.add_argument("--verify-bindings", action="store_true")
    study.add_argument("--require-approved", action="store_true")
    study.add_argument("--json", action="store_true")

    attempt = commands.add_parser(
        "validate-attempt",
        help="validate an experiment-execution-attempt/v1 JSON contract",
    )
    attempt.add_argument("path")
    _add_root(attempt)
    attempt.add_argument("--verify-files", action="store_true")
    attempt.add_argument("--require-authorized", action="store_true")
    attempt.add_argument("--json", action="store_true")

    attempt_set = commands.add_parser(
        "validate-attempt-set",
        help="validate exact non-overlapping attempt coverage of one study",
    )
    attempt_set.add_argument("study_path")
    attempt_set.add_argument("attempt_paths", nargs="+")
    _add_root(attempt_set)
    attempt_set.add_argument("--verify-files", action="store_true")
    attempt_set.add_argument("--require-authorized", action="store_true")
    attempt_set.add_argument("--json", action="store_true")

    split = commands.add_parser(
        "split-legacy-plan",
        help=(
            "split a legacy mixed Markdown plan into a review-required "
            "study and non-authoritative execution residue"
        ),
    )
    split.add_argument("path")
    _add_root(split)
    split.add_argument("--study-output", required=True)
    split.add_argument("--residue-output", required=True)
    split.add_argument("--json", action="store_true")

    corpus = commands.add_parser(
        "check-legacy-corpus",
        help="dry-run the deterministic split over every legacy plan",
    )
    _add_root(corpus)
    corpus.add_argument(
        "--plan-dir",
        default="docs/experiment_plans",
        help="plan directory relative to the repository root",
    )
    corpus.add_argument("--json", action="store_true")
    return parser


def _validate_study(args: argparse.Namespace) -> dict[str, Any]:
    return validate_study(
        load_study(args.path),
        repo_root=args.repo_root,
        verify_bindings=args.verify_bindings,
        require_approved=args.require_approved,
    )


def _validate_attempt(args: argparse.Namespace) -> dict[str, Any]:
    return validate_attempt(
        load_attempt(args.path),
        repo_root=args.repo_root,
        verify_files=args.verify_files,
        require_authorized=args.require_authorized,
    )


def _validate_attempt_set(args: argparse.Namespace) -> dict[str, Any]:
    return validate_attempt_set(
        study_path=args.study_path,
        attempt_paths=args.attempt_paths,
        repo_root=args.repo_root,
        verify_files=args.verify_files,
        require_authorized=args.require_authorized,
    )


def _split(args: argparse.Namespace) -> dict[str, Any]:
    study, residue = split_legacy_plan(
        args.path,
        repo_root=args.repo_root,
    )
    publish_legacy_split(
        study_document=study,
        residue_document=residue,
        study_output=args.study_output,
        residue_output=args.residue_output,
    )
    return {
        "schema_version": "experiment-legacy-split-result/v1",
        "status": "passed",
        "study_id": study["study"]["study_id"],
        "study_approval_status": study["approval"]["status"],
        "study_output": str(Path(args.study_output).expanduser().resolve()),
        "residue_output": str(
            Path(args.residue_output).expanduser().resolve()
        ),
        "launch_authority_created": False,
        "next_action": (
            "Review and explicitly approve the scientific study, then "
            "create and separately authorize an execution attempt."
        ),
    }


def _check_corpus(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    root = Path(args.repo_root).expanduser().resolve()
    relative_dir = Path(args.plan_dir)
    plan_dir = relative_dir if relative_dir.is_absolute() else root / relative_dir
    records: list[dict[str, Any]] = []
    for plan_path in sorted(plan_dir.glob("*.md")):
        try:
            study, residue = split_legacy_plan(plan_path, repo_root=root)
            records.append(
                {
                    "path": str(plan_path.relative_to(root)),
                    "status": "passed",
                    "study_id": study["study"]["study_id"],
                    "approval_status": study["approval"]["status"],
                    "study_spec_sha256": residue["study_spec_sha256"],
                    "unclassified_fixed_by": residue[
                        "operational_residue"
                    ]["fixed_by"]["unclassified_auxiliary_or_operational"],
                }
            )
        except ContractError as exc:
            records.append(
                {
                    "path": str(plan_path),
                    "status": "failed",
                    "error": exc.as_dict(),
                }
            )
    passed = sum(record["status"] == "passed" for record in records)
    report = {
        "schema_version": "experiment-legacy-corpus-check/v1",
        "status": "passed" if passed == len(records) and records else "failed",
        "plan_count": len(records),
        "passed_count": passed,
        "failed_count": len(records) - passed,
        "launch_authority_created": False,
        "records": records,
    }
    return report, 0 if report["status"] == "passed" else 1


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "validate-study":
            result = _validate_study(args)
            code = 0
        elif args.command == "validate-attempt":
            result = _validate_attempt(args)
            code = 0
        elif args.command == "validate-attempt-set":
            result = _validate_attempt_set(args)
            code = 0
        elif args.command == "split-legacy-plan":
            result = _split(args)
            code = 0
        else:
            result, code = _check_corpus(args)
        _emit(result, json_output=args.json)
        return code
    except ContractError as exc:
        print(json.dumps(exc.as_dict(), indent=2, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
