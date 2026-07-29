#!/usr/bin/env python3
"""Continue below-accuracy bounded-rho surfaces from an exact prior runner.

This controller preserves completed core cells.  It selects the lowest-loss
safety-clean candidate when no candidate reached the configured accuracy
threshold, adds the single permitted factor-of-three expansion when the
selected rho is bounded or core accuracy is suspiciously low, and writes a
terminal selection that reports whether the expanded rho range is still
bounded or bracketed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


RESELECTABLE_STATUSES = {
    "unresolved_no_passing_core_candidate",
    "unresolved_no_safe_completed_core_candidate",
}
TERMINAL_BELOW_ACCURACY_STATUSES = {
    "complete_below_accuracy_range_bounded",
    "complete_below_accuracy_bracketed",
}
DEFAULT_SUSPICIOUS_ACCURACY_FLOOR = 0.80


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit(root: Path) -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _load_runner(runner_root: Path):
    runner_root = runner_root.expanduser().resolve()
    for name in tuple(sys.modules):
        if name == "experiments" or name.startswith("experiments."):
            del sys.modules[name]
    sys.path.insert(0, str(runner_root))
    return importlib.import_module("experiments.run_conv12_bounded_rho")


def _select_best_safe(
    candidates: Sequence[Mapping[str, Any]],
    plateau_fraction: float,
) -> dict[str, Any]:
    eligible = [
        dict(candidate)
        for candidate in candidates
        if candidate.get("status") == "complete"
        and candidate.get("final_validation_loss") is not None
        and candidate.get("final_validation_accuracy") is not None
    ]
    if not eligible:
        return {
            "eligible": [],
            "plateau": [],
            "selected": None,
            "selection_basis": "none",
            "accuracy_gate_met": False,
        }
    best_loss = min(float(candidate["final_validation_loss"]) for candidate in eligible)
    plateau = [
        candidate
        for candidate in eligible
        if float(candidate["final_validation_loss"])
        <= best_loss * (1.0 + float(plateau_fraction))
    ]

    def key(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
        efficiency = candidate.get("median_projection_efficiency")
        return (
            -float(candidate["final_validation_accuracy"]),
            -float(efficiency) if efficiency is not None else math.inf,
            max(float(candidate["rho_conv"]), float(candidate["rho_dense"])),
            float(candidate["rho_conv"]) + float(candidate["rho_dense"]),
            float(candidate["rho_conv"]),
            float(candidate["rho_dense"]),
            str(candidate["cell_id"]),
        )

    return {
        "eligible": eligible,
        "minimum_final_validation_loss": best_loss,
        "plateau": plateau,
        "selected": min(plateau, key=key),
        "selection_basis": "best_safe_below_accuracy",
        "accuracy_gate_met": False,
    }


def _confined_edge(
    plateau: Sequence[Mapping[str, Any]],
    axis: str,
    values: Sequence[float],
) -> str | None:
    observed = [float(candidate[axis]) for candidate in plateau]
    if observed and all(_same_rho(value, min(values)) for value in observed):
        return "lower"
    if observed and all(_same_rho(value, max(values)) for value in observed):
        return "upper"
    return None


def _same_rho(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-15)


def _selected_edge(
    selection: Mapping[str, Any],
    axis: str,
    values: Sequence[float],
) -> str | None:
    selected = float(selection["selected"][axis])
    if _same_rho(selected, min(values)):
        return "lower"
    if _same_rho(selected, max(values)):
        return "upper"
    return None


def _maximum_safe_accuracy(candidates: Sequence[Mapping[str, Any]]) -> float | None:
    accuracies = [
        float(candidate["final_validation_accuracy"])
        for candidate in candidates
        if candidate.get("status") == "complete"
        and candidate.get("final_validation_accuracy") is not None
    ]
    return max(accuracies) if accuracies else None


def _accuracy_trend_edge(
    candidates: Sequence[Mapping[str, Any]],
    axis: str,
    values: Sequence[float],
) -> str | None:
    edge_accuracy: dict[str, float] = {}
    for edge, value in (("lower", min(values)), ("upper", max(values))):
        accuracies = [
            float(candidate["final_validation_accuracy"])
            for candidate in candidates
            if candidate.get("status") == "complete"
            and candidate.get("final_validation_accuracy") is not None
            and _same_rho(float(candidate[axis]), float(value))
        ]
        if accuracies:
            edge_accuracy[edge] = max(accuracies)
    if not edge_accuracy:
        return None
    if len(edge_accuracy) == 1:
        return next(iter(edge_accuracy))
    if edge_accuracy["upper"] > edge_accuracy["lower"]:
        return "upper"
    return "lower"


def _expanded_value(values: Sequence[float], edge: str) -> float:
    if edge == "lower":
        return min(values) / 3.0
    if edge == "upper":
        return max(values) * 3.0
    raise ValueError(f"Unexpected edge: {edge!r}.")


def _repair_terminal_range_status(
    runner,
    *,
    original: Mapping[str, Any],
    selection_path: Path,
    surface_id: str,
) -> dict[str, Any]:
    expanded = original.get("expansion", {}).get("expanded_axes", {})
    selected = original.get("selected")
    if (
        not isinstance(selected, Mapping)
        or not isinstance(expanded, Mapping)
        or not expanded.get("rho_conv")
        or not expanded.get("rho_dense")
    ):
        return {
            "surface_id": surface_id,
            "status": "skipped_terminal_without_expanded_axes",
        }
    conv_edge = _selected_edge(
        {"selected": selected}, "rho_conv", expanded["rho_conv"]
    )
    dense_edge = _selected_edge(
        {"selected": selected}, "rho_dense", expanded["rho_dense"]
    )
    bounded = conv_edge is not None or dense_edge is not None
    repaired = dict(original)
    repaired["status"] = (
        "complete_below_accuracy_range_bounded"
        if bounded
        else "complete_below_accuracy_bracketed"
    )
    repaired["rho_range_status"] = {
        "classification": "bounded" if bounded else "unbounded",
        "bracketed": not bounded,
        "rho_conv_edge": conv_edge,
        "rho_dense_edge": dense_edge,
    }
    repaired_expansion = dict(repaired["expansion"])
    repaired_expansion["plateau_on_outer_rho_conv_edge"] = conv_edge
    repaired_expansion["plateau_on_outer_rho_dense_edge"] = dense_edge
    repaired["expansion"] = repaired_expansion
    repaired["range_classification_repair"] = {
        "reason": "rho_factor_floating_point_tolerance",
        "controller_commit": _git_commit(Path(__file__).resolve().parents[1]),
        "training_cells_reused": True,
    }
    backup = selection_path.with_name("selection.pre_range_tolerance_repair.json")
    if not backup.exists():
        shutil.copy2(selection_path, backup)
    runner._write_json(selection_path, repaired)
    return {
        "surface_id": surface_id,
        "status": repaired["status"],
        "selected": repaired["selected"],
        "rho_range_status": repaired["rho_range_status"],
        "new_cell_count": 0,
        "range_classification_repaired": True,
    }


def reselect_surface(
    runner,
    *,
    study: Mapping[str, Any],
    output_root: Path,
    surface: Mapping[str, Any],
    device: str,
    controller_path: Path,
) -> dict[str, Any]:
    surface_dir = output_root / "surfaces" / surface["surface_id"]
    selection_path = surface_dir / "selection.json"
    if not selection_path.is_file():
        return {
            "surface_id": surface["surface_id"],
            "status": "skipped_missing_selection",
        }
    original = json.loads(selection_path.read_text(encoding="utf-8"))
    if original.get("status") in TERMINAL_BELOW_ACCURACY_STATUSES:
        return _repair_terminal_range_status(
            runner,
            original=original,
            selection_path=selection_path,
            surface_id=surface["surface_id"],
        )
    if original.get("status") not in RESELECTABLE_STATUSES:
        return {
            "surface_id": surface["surface_id"],
            "status": "skipped_not_below_accuracy_unresolved",
            "existing_status": original.get("status"),
        }

    candidates = [dict(candidate) for candidate in original.get("candidates", ())]
    selection = _select_best_safe(
        candidates,
        float(study["rho_search"]["inclusive_loss_plateau"]),
    )
    if selection["selected"] is None:
        return {
            "surface_id": surface["surface_id"],
            "status": "unresolved_no_safe_completed_core_candidate",
        }

    core_conv = sorted({float(candidate["rho_conv"]) for candidate in candidates})
    core_dense = sorted({float(candidate["rho_dense"]) for candidate in candidates})
    if len(core_conv) != 3 or len(core_dense) != 3:
        raise RuntimeError(
            f"Expected a 3x3 core for {surface['surface_id']}, got "
            f"{core_conv!r} x {core_dense!r}."
        )
    conv_edge = _selected_edge(selection, "rho_conv", core_conv)
    dense_edge = _selected_edge(selection, "rho_dense", core_dense)
    core_maximum_safe_accuracy = _maximum_safe_accuracy(candidates)
    configured_suspicious_floor = study["rho_search"].get(
        "suspicious_validation_accuracy_floor"
    )
    suspicious_floor = float(
        DEFAULT_SUSPICIOUS_ACCURACY_FLOOR
        if configured_suspicious_floor is None
        else configured_suspicious_floor
    )
    suspicious_accuracy_triggered = bool(
        core_maximum_safe_accuracy is not None
        and core_maximum_safe_accuracy < suspicious_floor
    )
    if suspicious_accuracy_triggered:
        if conv_edge is None:
            conv_edge = _accuracy_trend_edge(candidates, "rho_conv", core_conv)
        if dense_edge is None:
            dense_edge = _accuracy_trend_edge(candidates, "rho_dense", core_dense)
    expanded_conv = list(core_conv)
    expanded_dense = list(core_dense)
    new_pairs: set[tuple[float, float]] = set()
    if conv_edge is not None:
        new_conv = _expanded_value(core_conv, conv_edge)
        expanded_conv.append(new_conv)
        expanded_conv.sort()
        new_pairs.update((new_conv, value) for value in expanded_dense)
    if dense_edge is not None:
        new_dense = _expanded_value(core_dense, dense_edge)
        expanded_dense.append(new_dense)
        expanded_dense.sort()
        new_pairs.update((value, new_dense) for value in expanded_conv)

    rho_root = surface_dir / "rho"
    source_config = surface_dir / "source_config.json"
    rho_conv_axis, rho_dense_axis = runner._rho_axes(study)
    candidate_by_pair = {
        (float(candidate["rho_conv"]), float(candidate["rho_dense"])): candidate
        for candidate in candidates
    }
    for rho_conv, rho_dense in sorted(new_pairs):
        cell_dir, cell = runner._run_rho_cell(
            study,
            surface,
            source_config,
            rho_root,
            rho_conv_axis,
            rho_dense_axis,
            rho_conv,
            rho_dense,
            device=device,
            canary_only=False,
        )
        candidate_by_pair[(rho_conv, rho_dense)] = runner._candidate_record(
            cell_dir, cell
        )

    final_selection = _select_best_safe(
        list(candidate_by_pair.values()),
        float(study["rho_search"]["inclusive_loss_plateau"]),
    )
    outer_conv_edge = _selected_edge(
        final_selection, "rho_conv", expanded_conv
    )
    outer_dense_edge = _selected_edge(
        final_selection, "rho_dense", expanded_dense
    )
    range_bounded = outer_conv_edge is not None or outer_dense_edge is not None
    final_maximum_safe_accuracy = _maximum_safe_accuracy(
        list(candidate_by_pair.values())
    )
    backup = surface_dir / "selection.pre_below_accuracy_policy.json"
    if not backup.exists():
        shutil.copy2(selection_path, backup)
    result = {
        **original,
        "status": (
            "complete_below_accuracy_range_bounded"
            if range_bounded
            else "complete_below_accuracy_bracketed"
        ),
        "core_axes": {"rho_conv": core_conv, "rho_dense": core_dense},
        "expansion": {
            "triggered": bool(new_pairs),
            "trigger": (
                "suspicious_low_accuracy"
                if suspicious_accuracy_triggered
                else "selected_boundary"
                if new_pairs
                else None
            ),
            "rho_conv_edge": conv_edge,
            "rho_dense_edge": dense_edge,
            "expanded_axes": {
                "rho_conv": expanded_conv,
                "rho_dense": expanded_dense,
            },
            "new_cell_count": len(new_pairs),
            "plateau_on_outer_rho_conv_edge": outer_conv_edge,
            "plateau_on_outer_rho_dense_edge": outer_dense_edge,
        },
        "rho_range_status": {
            "classification": "bounded" if range_bounded else "unbounded",
            "bracketed": not range_bounded,
            "rho_conv_edge": outer_conv_edge,
            "rho_dense_edge": outer_dense_edge,
        },
        "suspicious_accuracy_status": {
            "floor": suspicious_floor,
            "floor_source": (
                "compatibility_controller_default"
                if configured_suspicious_floor is None
                else "study_config"
            ),
            "triggered": suspicious_accuracy_triggered,
            "core_maximum_safe_accuracy": core_maximum_safe_accuracy,
            "final_maximum_safe_accuracy": final_maximum_safe_accuracy,
            "remains_below_floor": bool(
                final_maximum_safe_accuracy is not None
                and final_maximum_safe_accuracy < suspicious_floor
            ),
            "direction_policy": "better_safety_clean_core_edge_per_axis",
        },
        "candidates": sorted(
            candidate_by_pair.values(),
            key=lambda item: (item["rho_conv"], item["rho_dense"]),
        ),
        "selection": final_selection,
        "selected": final_selection["selected"],
        "below_accuracy_policy": {
            "minimum_validation_accuracy": float(
                study["rho_search"]["minimum_validation_accuracy"]
            ),
            "directive": (
                "select_best_safe_expand_if_range_bounded_or_suspiciously_low"
            ),
            "controller": str(controller_path),
            "controller_sha256": _sha256_file(controller_path),
            "controller_commit": _git_commit(controller_path.parents[1]),
            "training_runner_commit": runner._git_state().get("commit"),
        },
    }
    runner._write_json(selection_path, result)
    return {
        "surface_id": surface["surface_id"],
        "status": result["status"],
        "selected": result["selected"],
        "rho_range_status": result["rho_range_status"],
        "new_cell_count": len(new_pairs),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner-root", type=Path, required=True)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--surface-index", type=int, action="append")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    runner = _load_runner(args.runner_root)
    _study_path, study = runner.load_study(args.study)
    surfaces = runner.surface_specs(study)
    indices = (
        list(args.surface_index)
        if args.surface_index is not None
        else list(range(len(surfaces)))
    )
    controller_path = Path(__file__).resolve()
    results = [
        reselect_surface(
            runner,
            study=study,
            output_root=args.output_root.expanduser().resolve(),
            surface=surfaces[index],
            device=args.device,
            controller_path=controller_path,
        )
        for index in indices
    ]
    print(json.dumps({"surfaces": results}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
