#!/usr/bin/env python3
"""Run a provenance-linked second rho expansion for bounded Conv2 surfaces."""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
from argparse import Namespace
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.rho_search import (
    _float_token,
    _git_state,
    _write_json,
    run as run_rho_search,
)
from experiments.run_conv12_bounded_rho import (
    _candidate_record,
    _maximum_safe_accuracy,
    _rho_args,
    _rho_index,
    _same_rho,
    _selection_edge,
    load_study as load_base_study,
    select_candidates,
    surface_specs as base_surface_specs,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STUDY = (
    REPO_ROOT
    / "configs"
    / "conv"
    / "perfectdiode_conv2_bounded_rho_continuation_wave2_20260730_v1.json"
)
SCHEMA = "perfectdiode-conv2-bounded-rho-continuation/v1"
TERMINAL_STATUSES = {
    "complete_continuation_range_bounded",
    "complete_continuation_bracketed",
    "complete_parent_bracketed_no_expansion",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolved_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def load_study(
    path: str | Path = DEFAULT_STUDY,
) -> tuple[Path, dict[str, Any], Path, dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    study = json.loads(source.read_text(encoding="utf-8"))
    if study.get("schema_version") != SCHEMA:
        raise ValueError(f"Unexpected continuation schema: {study.get('schema_version')!r}.")
    scope = study["scope"]
    expected_scope = {
        "architectures": ["conv2"],
        "schemes": ["baseline", "ours"],
        "optimizers": ["SGD", "Adam"],
        "initializers": ["bounded_uniform", "bounded_kaiming_uniform"],
        "excluded": ["legacy", "conv1", "conv3"],
    }
    for key, expected in expected_scope.items():
        if scope.get(key) != expected:
            raise ValueError(
                f"Expected scope.{key}={expected!r}; provided {scope.get(key)!r}."
            )
    continuation = study["continuation"]
    if int(continuation["wave"]) != 2 or float(continuation["factor"]) != 3.0:
        raise ValueError("The Conv2 continuation must be the factor-three second wave.")
    if continuation.get("bound_occupancy") != "report_only":
        raise ValueError("Bound occupancy must remain report-only.")
    if continuation.get("projection_efficiency") != "report_only":
        raise ValueError("Projection efficiency must remain report-only.")
    if study["dataset"].get("official_test_read") is not False:
        raise ValueError("The continuation must not read the official MNIST test split.")

    base_path = _resolved_path(study["parent"]["study_config"])
    if _sha256_file(base_path) != study["parent"]["study_config_sha256"]:
        raise RuntimeError("The parent bounded-study config hash does not match.")
    _loaded_base_path, base = load_base_study(base_path)
    if base["study_id"] != study["parent"]["study_id"]:
        raise ValueError("The continuation names a different parent study.")

    expected_ids = [
        surface["surface_id"]
        for surface in base_surface_specs(base)
        if surface["architecture"] == "conv2"
    ]
    surfaces = study["surfaces"]
    if [surface["surface_id"] for surface in surfaces] != expected_ids:
        raise ValueError("Continuation surfaces must cover all eight parent Conv2 rows.")
    if [int(surface["index"]) for surface in surfaces] != list(range(8)):
        raise ValueError("Continuation surface indices must be contiguous from zero.")
    expanded = [surface for surface in surfaces if surface["expand"]]
    if len(expanded) != 7 or sum(int(surface["expected_new_cells"]) for surface in expanded) != 35:
        raise ValueError("The declared continuation must contain seven surfaces and 35 cells.")
    return source, study, base_path, base


def surface_specs(study: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [dict(surface) for surface in study["surfaces"]]


def _contains(values: Sequence[float], requested: float) -> bool:
    return any(_same_rho(value, requested) for value in values)


def continuation_axes_and_pairs(
    parent_selection: Mapping[str, Any],
    expand: Mapping[str, str],
    factor: float,
) -> tuple[dict[str, list[float]], list[tuple[float, float]]]:
    previous = parent_selection["expansion"]["expanded_axes"]
    old_conv = sorted(float(value) for value in previous["rho_conv"])
    old_dense = sorted(float(value) for value in previous["rho_dense"])
    new_conv = list(old_conv)
    new_dense = list(old_dense)
    for axis, values in (("rho_conv", new_conv), ("rho_dense", new_dense)):
        edge = expand.get(axis)
        if edge is None:
            continue
        if edge not in {"lower", "upper"}:
            raise ValueError(f"Unsupported {axis} continuation edge: {edge!r}.")
        value = min(values) / factor if edge == "lower" else max(values) * factor
        values.append(float(value))
        values.sort()
    pairs = [
        (rho_conv, rho_dense)
        for rho_conv, rho_dense in itertools.product(new_conv, new_dense)
        if not (
            _contains(old_conv, rho_conv) and _contains(old_dense, rho_dense)
        )
    ]
    return {"rho_conv": new_conv, "rho_dense": new_dense}, pairs


def _parent_surface(
    parent_root: Path,
    surface: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    surface_dir = parent_root / "surfaces" / surface["surface_id"]
    selection_path = surface_dir / "selection.json"
    if _sha256_file(selection_path) != surface["parent_selection_sha256"]:
        raise RuntimeError(
            f"Parent selection hash mismatch for {surface['surface_id']}."
        )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    expected_expand = dict(surface["expand"])
    range_status = selection["rho_range_status"]
    if expected_expand:
        if range_status.get("classification") != "bounded":
            raise RuntimeError(
                f"Expected a bounded parent selection for {surface['surface_id']}."
            )
        observed = {
            axis: range_status.get(f"{axis}_edge")
            for axis in ("rho_conv", "rho_dense")
            if range_status.get(f"{axis}_edge") is not None
        }
        if observed != expected_expand:
            raise RuntimeError(
                f"Continuation edge mismatch for {surface['surface_id']}: "
                f"expected={expected_expand!r}, observed={observed!r}."
            )
    elif range_status.get("classification") != "unbounded":
        raise RuntimeError(
            f"Only a bracketed parent surface may skip expansion: {surface['surface_id']}."
        )
    return surface_dir, selection


def _search_contract(args: Namespace) -> dict[str, Any]:
    dry = copy.copy(args)
    dry.dry_run = True
    planned = run_rho_search(dry)
    for key in ("cells", "selected_indices", "mode"):
        planned.pop(key, None)
    return planned


def _seed_imported_probe(
    args: Namespace,
    parent_probe_path: Path,
    rho_root: Path,
    *,
    surface_id: str,
) -> dict[str, Any]:
    contract = _search_contract(args)
    parent_probe = json.loads(parent_probe_path.read_text(encoding="utf-8"))
    parent_signature = parent_probe.get("search_signature")
    if not isinstance(parent_signature, Mapping):
        raise RuntimeError(f"Parent probe has no search signature: {parent_probe_path}.")
    for key in (
        "source_config_sha256",
        "optimizer",
        "bias_policy",
        "probe_batches",
        "stability_tolerance",
        "split_seed",
        "shuffle_seed",
        "validation_batch_size",
    ):
        if parent_signature.get(key) != contract.get(key):
            raise RuntimeError(
                f"Imported probe contract mismatch for {surface_id}: {key}."
            )
    if parent_probe.get("probe_stable") is not True:
        raise RuntimeError(f"Cannot import an unstable probe for {surface_id}.")

    imported_probe = copy.deepcopy(parent_probe)
    imported_probe["import_provenance"] = {
        "path": str(parent_probe_path),
        "sha256": _sha256_file(parent_probe_path),
        "parent_search_signature_sha256": hashlib.sha256(
            json.dumps(
                parent_signature,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest(),
        "reason": "user_directed_conv2_second_rho_expansion",
    }
    imported_probe["search_signature"] = contract
    rho_root.mkdir(parents=True, exist_ok=True)
    resolved_path = rho_root / "resolved.json"
    probe_path = rho_root / "probe.json"
    for path, payload in (
        (resolved_path, contract),
        (probe_path, imported_probe),
    ):
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing != payload:
                raise RuntimeError(f"Existing continuation artifact mismatch: {path}.")
        else:
            _write_json(path, payload)
    return imported_probe


def _args(
    base: Mapping[str, Any],
    surface: Mapping[str, Any],
    source_config: Path,
    rho_root: Path,
    axes: Mapping[str, Sequence[float]],
    *,
    device: str,
    index: int | None,
    smoke: bool,
    study_id: str,
) -> Namespace:
    parent_surface = {
        "index": int(surface["parent_surface_index"]),
        "initializer": surface["initializer"],
        "architecture": "conv2",
        "scheme": surface["scheme"],
        "optimizer": surface["optimizer"],
        "surface_id": surface["surface_id"],
    }
    args = _rho_args(
        base,
        parent_surface,
        source_config,
        rho_root,
        axes["rho_conv"],
        axes["rho_dense"],
        device=device,
        index=index,
        probe_only=index is None,
        smoke=smoke,
    )
    # The imported production probe is also used by the one-batch smoke. Keep
    # its measurement contract unchanged while shortening only candidate work.
    args.probe_batches = list(base["rho_search"]["probe_batches"])
    args.stability_tolerance = float(
        base["rho_search"]["probe_stability_tolerance"]
    )
    args.study_id = study_id + ("-smoke" if smoke else "")
    args.target = surface["target"]
    args.smoke = smoke
    args.reporting_command = {
        "module": "experiments.continue_conv2_bounded_rho",
        "surface_index": int(surface["index"]),
        "cell_index": index,
        "continuation_wave": 2,
    }
    return args


def _cell_dir(
    rho_root: Path,
    axes: Mapping[str, Sequence[float]],
    rho_conv: float,
    rho_dense: float,
) -> tuple[int, Path]:
    index = _rho_index(
        axes["rho_conv"],
        axes["rho_dense"],
        rho_conv,
        rho_dense,
    )
    conv_index, dense_index = divmod(index, len(axes["rho_dense"]))
    resolved_conv = float(axes["rho_conv"][conv_index])
    resolved_dense = float(axes["rho_dense"][dense_index])
    name = (
        f"{index:03d}_rc_{_float_token(resolved_conv)}_"
        f"rd_{_float_token(resolved_dense)}"
    )
    return index, rho_root / "cells" / name


def _materialize(
    study_path: Path,
    study: Mapping[str, Any],
    parent_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    resolved = {
        "schema_version": "perfectdiode-conv2-bounded-rho-continuation-resolved/v1",
        "study_id": study["study_id"],
        "study_config": str(study_path),
        "study_config_sha256": _sha256_file(study_path),
        "parent_study_id": study["parent"]["study_id"],
        "parent_root": str(parent_root),
        "code": _git_state(),
        "continuation_wave": 2,
        "protocol_deviation": study["continuation"]["protocol_deviation"],
        "surface_count": len(study["surfaces"]),
        "new_cell_budget": sum(
            int(surface["expected_new_cells"]) for surface in study["surfaces"]
        ),
        "official_test_read": False,
    }
    path = output_root / "study.resolved.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != resolved:
            raise RuntimeError(f"Existing resolved continuation mismatch: {path}.")
    else:
        _write_json(path, resolved)
    return resolved


def run_surface(
    study_path: Path,
    study: Mapping[str, Any],
    base: Mapping[str, Any],
    parent_root: Path,
    output_root: Path,
    surface: Mapping[str, Any],
    *,
    device: str,
    smoke: bool,
) -> dict[str, Any]:
    _materialize(study_path, study, parent_root, output_root)
    parent_surface_dir, parent_selection = _parent_surface(parent_root, surface)
    axes, pairs = continuation_axes_and_pairs(
        parent_selection,
        surface["expand"],
        float(study["continuation"]["factor"]),
    )
    if len(pairs) != int(surface["expected_new_cells"]):
        raise RuntimeError(
            f"Unexpected new-cell count for {surface['surface_id']}: {len(pairs)}."
        )
    if not pairs:
        result = {
            "schema_version": "perfectdiode-conv2-bounded-rho-continuation-surface/v1",
            **dict(surface),
            "status": "complete_parent_bracketed_no_expansion",
            "parent_status": parent_selection["status"],
            "parent_selected": parent_selection["selected"],
            "selected": parent_selection["selected"],
            "accuracy_gate_met": bool(
                parent_selection["selected"].get("selection_eligible", False)
            ),
            "rho_range_status": parent_selection["rho_range_status"],
            "continuation": {
                "wave": 2,
                "triggered": False,
                "reason": "parent_range_already_bracketed",
                "new_cell_count": 0,
            },
            "official_test_read": False,
        }
        if not smoke:
            path = output_root / "surfaces" / surface["surface_id"] / "continuation.json"
            _write_json(path, result)
        return result

    surface_root = (
        output_root / "smoke" / surface["surface_id"]
        if smoke
        else output_root / "surfaces" / surface["surface_id"]
    )
    result_path = surface_root / "continuation.json"
    if result_path.exists() and not smoke:
        return json.loads(result_path.read_text(encoding="utf-8"))
    rho_root = surface_root / "rho"
    source_config = parent_surface_dir / "source_config.json"
    parent_probe = parent_surface_dir / "rho" / "probe.json"
    probe_args = _args(
        base,
        surface,
        source_config,
        rho_root,
        axes,
        device=device,
        index=None,
        smoke=smoke,
        study_id=study["study_id"],
    )
    _seed_imported_probe(
        probe_args,
        parent_probe,
        rho_root,
        surface_id=surface["surface_id"],
    )
    run_rho_search(probe_args)

    run_pairs = pairs[:1] if smoke else pairs
    new_candidates = []
    for rho_conv, rho_dense in run_pairs:
        index, cell_dir = _cell_dir(rho_root, axes, rho_conv, rho_dense)
        cell_args = _args(
            base,
            surface,
            source_config,
            rho_root,
            axes,
            device=device,
            index=index,
            smoke=smoke,
            study_id=study["study_id"],
        )
        run_rho_search(cell_args)
        cell = json.loads((cell_dir / "cell.json").read_text(encoding="utf-8"))
        new_candidates.append(_candidate_record(cell_dir, cell))

    if smoke:
        result = {
            "schema_version": "perfectdiode-conv2-bounded-rho-continuation-smoke/v1",
            "status": "complete",
            "surface": dict(surface),
            "new_pair": list(run_pairs[0]),
            "rho_cell": new_candidates[0],
            "imported_probe": str(rho_root / "probe.json"),
            "official_test_read": False,
        }
        _write_json(result_path, result)
        return result

    candidates = [dict(candidate) for candidate in parent_selection["candidates"]]
    candidates.extend(new_candidates)
    selection = select_candidates(
        candidates,
        float(base["rho_search"]["inclusive_loss_plateau"]),
        select_best_safe_below_accuracy=True,
    )
    if selection["selected"] is None:
        raise RuntimeError(
            f"Continuation produced no selectable candidate: {surface['surface_id']}."
        )
    conv_edge = _selection_edge(selection, "rho_conv", axes["rho_conv"])
    dense_edge = _selection_edge(selection, "rho_dense", axes["rho_dense"])
    bounded = conv_edge is not None or dense_edge is not None
    final_maximum = _maximum_safe_accuracy(candidates)
    floor = float(base["rho_search"]["suspicious_validation_accuracy_floor"])
    result = {
        "schema_version": "perfectdiode-conv2-bounded-rho-continuation-surface/v1",
        **dict(surface),
        "status": (
            "complete_continuation_range_bounded"
            if bounded
            else "complete_continuation_bracketed"
        ),
        "parent": {
            "study_id": study["parent"]["study_id"],
            "selection_path": str(parent_surface_dir / "selection.json"),
            "selection_sha256": surface["parent_selection_sha256"],
            "status": parent_selection["status"],
            "selected": parent_selection["selected"],
            "expanded_axes": parent_selection["expansion"]["expanded_axes"],
        },
        "continuation": {
            "wave": 2,
            "triggered": True,
            "factor": float(study["continuation"]["factor"]),
            "expanded_edges": dict(surface["expand"]),
            "expanded_axes": axes,
            "new_cell_count": len(new_candidates),
            "protocol_deviation": study["continuation"]["protocol_deviation"],
        },
        "new_candidates": new_candidates,
        "candidates": sorted(
            candidates,
            key=lambda item: (item["rho_conv"], item["rho_dense"]),
        ),
        "selection": selection,
        "selected": selection["selected"],
        "accuracy_gate_met": bool(selection["accuracy_gate_met"]),
        "rho_range_status": {
            "classification": "bounded" if bounded else "unbounded",
            "bracketed": not bounded,
            "rho_conv_edge": conv_edge,
            "rho_dense_edge": dense_edge,
        },
        "suspicious_accuracy_status": {
            "floor": floor,
            "final_maximum_safe_accuracy": final_maximum,
            "remains_below_floor": bool(
                final_maximum is not None and final_maximum < floor
            ),
        },
        "official_test_read": False,
    }
    _write_json(result_path, result)
    return result


def collect(study: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    records = []
    for surface in surface_specs(study):
        path = output_root / "surfaces" / surface["surface_id"] / "continuation.json"
        if path.exists():
            record = json.loads(path.read_text(encoding="utf-8"))
            records.append(
                {
                    **dict(surface),
                    "status": record["status"],
                    "selected": record.get("selected"),
                    "accuracy_gate_met": record.get("accuracy_gate_met"),
                    "rho_range_status": record.get("rho_range_status"),
                    "suspicious_accuracy_status": record.get(
                        "suspicious_accuracy_status"
                    ),
                    "new_cell_count": int(
                        record.get("continuation", {}).get("new_cell_count", 0)
                    ),
                    "path": str(path),
                }
            )
        else:
            records.append(
                {
                    **dict(surface),
                    "status": "pending",
                    "selected": None,
                    "accuracy_gate_met": None,
                    "rho_range_status": None,
                    "suspicious_accuracy_status": None,
                    "new_cell_count": 0,
                    "path": str(path),
                }
            )
    counts: dict[str, int] = {}
    for record in records:
        counts[record["status"]] = counts.get(record["status"], 0) + 1
    complete = all(record["status"] in TERMINAL_STATUSES for record in records)
    summary = {
        "schema_version": "perfectdiode-conv2-bounded-rho-continuation-summary/v1",
        "study_id": study["study_id"],
        "parent_study_id": study["parent"]["study_id"],
        "status": "complete" if complete else "partial",
        "execution_status": (
            "terminal" if counts.get("pending", 0) == 0 else "in_progress"
        ),
        "counts": counts,
        "completed_new_cells": sum(record["new_cell_count"] for record in records),
        "protocol_deviation": study["continuation"]["protocol_deviation"],
        "surfaces": records,
        "official_test_read": False,
    }
    _write_json(output_root / "summary.json", summary)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("plan", "smoke", "run-surface", "run-target", "status", "collect"),
    )
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--parent-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--surface-index", type=int, default=0)
    parser.add_argument("--target", choices=("main", "akib", "trex"))
    parser.add_argument("--device")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    study_path, study, _base_path, base = load_study(args.study)
    parent_root = (
        args.parent_root.expanduser().resolve()
        if args.parent_root is not None
        else _resolved_path(study["parent"]["local_result_root"])
    )
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else _resolved_path(study["execution"]["result_root"])
    )
    device = args.device or study["execution"]["device"]
    surfaces = surface_specs(study)
    if args.surface_index < 0 or args.surface_index >= len(surfaces):
        raise ValueError(
            f"Expected --surface-index in [0,{len(surfaces) - 1}]."
        )

    if args.command == "plan":
        planned_surfaces = []
        for surface in surfaces:
            _parent_dir, parent_selection = _parent_surface(parent_root, surface)
            axes, pairs = continuation_axes_and_pairs(
                parent_selection,
                surface["expand"],
                float(study["continuation"]["factor"]),
            )
            if len(pairs) != int(surface["expected_new_cells"]):
                raise RuntimeError(
                    f"Unexpected planned cell count for {surface['surface_id']}."
                )
            previous_axes = parent_selection["expansion"]["expanded_axes"]
            planned_surfaces.append(
                {
                    **dict(surface),
                    "parent_selected": parent_selection["selected"],
                    "added_rho_conv": [
                        value
                        for value in axes["rho_conv"]
                        if not _contains(previous_axes["rho_conv"], value)
                    ],
                    "added_rho_dense": [
                        value
                        for value in axes["rho_dense"]
                        if not _contains(previous_axes["rho_dense"], value)
                    ],
                    "new_pairs": [list(pair) for pair in pairs],
                }
            )
        result = {
            "study_id": study["study_id"],
            "parent_root": str(parent_root),
            "output_root": str(output_root),
            "surface_count": len(surfaces),
            "expanded_surface_count": sum(bool(surface["expand"]) for surface in surfaces),
            "new_cell_budget": sum(
                int(surface["expected_new_cells"]) for surface in surfaces
            ),
            "candidate_epochs": int(base["rho_search"]["candidate_epochs"]),
            "canary_steps": int(base["rho_search"]["canary_steps"]),
            "surfaces": planned_surfaces,
            "protocol_deviation": study["continuation"]["protocol_deviation"],
            "official_test_read": False,
        }
    elif args.command == "smoke":
        surface = surfaces[args.surface_index]
        if args.target is not None and args.target != surface["target"]:
            raise ValueError("Smoke target does not match the recorded surface target.")
        result = run_surface(
            study_path,
            study,
            base,
            parent_root,
            output_root,
            surface,
            device=device,
            smoke=True,
        )
    elif args.command == "run-surface":
        surface = surfaces[args.surface_index]
        if args.target is not None and args.target != surface["target"]:
            raise ValueError("Run target does not match the recorded surface target.")
        result = run_surface(
            study_path,
            study,
            base,
            parent_root,
            output_root,
            surface,
            device=device,
            smoke=False,
        )
    elif args.command == "run-target":
        if args.target is None:
            raise ValueError("run-target requires --target.")
        completed = []
        for surface in surfaces:
            if surface["target"] == args.target:
                completed.append(
                    run_surface(
                        study_path,
                        study,
                        base,
                        parent_root,
                        output_root,
                        surface,
                        device=device,
                        smoke=False,
                    )
                )
        result = {
            "status": "complete",
            "target": args.target,
            "surface_count": len(completed),
            "surfaces": completed,
        }
    elif args.command in {"status", "collect"}:
        result = collect(study, output_root)
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
