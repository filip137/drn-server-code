#!/usr/bin/env python3
"""Build a canonical sweep JSON from a complete run and calibration records.

This is a configuration conversion tool.  It never launches training and it
never selects or derives training hyperparameters.  In particular, learning
rates, T/K, minimizer settings, epoch count, and checkpoint policy come only
from the complete base run JSON.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from experiments.mnist_conv.io import atomic_write_json, read_json
from experiments.mnist_conv.specs import RunSpec, SpecValidationError, SweepSpec


CALIBRATION_RECORDS_SCHEMA = "mnist-conv-calibration-records/v1"
RECORD_KEYS = {
    "case_id",
    "status",
    "dataset_name",
    "architecture_profile",
    "non_linearity",
    "voltage_amp",
    "current_amp",
    "input_gain",
    "calibration",
}
REQUIRED_CASES = {
    "mnist_bp_amp_v1_c1": (1.0, 1.0),
    "mnist_bp_amp_v4_c1": (4.0, 1.0),
    "mnist_bp_amp_v4_c0p25": (4.0, 0.25),
}


class ConversionError(ValueError):
    """A fail-closed calibration-to-sweep conversion error."""


def _error(expected: str, provided: Any, path: str) -> ConversionError:
    return ConversionError(f"Expected {path} to be {expected}. Provided value: {provided!r}.")


def _read_json(path: Path, description: str) -> Any:
    try:
        return read_json(path)
    except (OSError, ValueError) as exc:
        raise ConversionError(
            f"Expected {description} at {path} to contain valid JSON. Provided error: {exc}."
        ) from exc


def _integer(value: Any, path: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise _error(f"an integer >= {minimum}", value, path)
    return value


def _finite(value: Any, path: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _error("a finite number", value, path)
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        raise _error("a positive finite number" if positive else "a finite number", value, path)
    return result


def _records_from_path(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    value = _read_json(source, "calibration records")
    if not isinstance(value, dict) or set(value) != {"schema_version", "records"}:
        raise _error(
            "an object with exactly 'schema_version' and 'records'",
            value,
            "calibration records",
        )
    if value["schema_version"] != CALIBRATION_RECORDS_SCHEMA:
        raise _error(CALIBRATION_RECORDS_SCHEMA, value["schema_version"], "calibration records schema_version")
    records = value["records"]
    if not isinstance(records, list) or not records:
        raise _error("a non-empty list", records, "calibration records.records")
    for index, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != RECORD_KEYS:
            raise _error(
                f"an object with exactly keys {sorted(RECORD_KEYS)}",
                record,
                f"calibration records.records[{index}]",
            )
    return records


def _validate_fixed_protocol(record: dict[str, Any], base: RunSpec, index: int) -> None:
    path = f"calibration records.records[{index}]"
    base_run = base.data["run"]
    expected_pairs = {
        "dataset_name": base_run["dataset"]["name"],
        "architecture_profile": base_run["architecture"]["profile"],
        "non_linearity": base_run["model"]["non_linearity"],
    }
    for key, expected in expected_pairs.items():
        if record[key] != expected:
            raise _error(repr(expected), record[key], f"{path}.{key}")
    if record["status"] != "complete":
        raise _error("exactly 'complete'", record["status"], f"{path}.status")


def _validate_calibration(record: dict[str, Any], base: RunSpec, index: int) -> dict[str, Any]:
    path = f"calibration records.records[{index}].calibration"
    calibration = record["calibration"]
    if not isinstance(calibration, dict):
        raise _error("a complete calibration object", calibration, path)
    base_calibration = base.data["run"]["calibration"]
    if set(calibration) != set(base_calibration):
        raise _error(
            f"an object with exactly keys {sorted(base_calibration)}",
            calibration,
            path,
        )
    expected_common = {
        "scope": "first_hidden_layer",
        "sample_count": 256,
        "batch_size": 64,
        "model_seed": 0,
        "affine_seed": 1729,
        "settling_iterations": 64,
        "adaptive_equilibrium": False,
    }
    for key, expected in expected_common.items():
        if calibration.get(key) != expected:
            raise _error(repr(expected), calibration.get(key), f"{path}.{key}")
    layer_measurements = calibration.get("layer_measurements")
    depth = len(base.data["run"]["architecture"]["channels"])
    if not isinstance(layer_measurements, list) or len(layer_measurements) != depth:
        raise _error(
            f"one explicit measurement for each of {depth} hidden layers",
            layer_measurements,
            f"{path}.layer_measurements",
        )
    non_linearity = base.data["run"]["model"]["non_linearity"]
    if non_linearity == "hard_sigmoid":
        if calibration.get("kind") != "hard_sigmoid_saturation":
            raise _error("'hard_sigmoid_saturation'", calibration.get("kind"), f"{path}.kind")
        if _finite(calibration.get("v_off"), f"{path}.v_off") != 4.0:
            raise _error("the frozen v_off=4.0", calibration.get("v_off"), f"{path}.v_off")
        if _finite(calibration.get("g_on"), f"{path}.g_on") != 100.0:
            raise _error("the frozen g_on=100", calibration.get("g_on"), f"{path}.g_on")
        if _finite(calibration.get("g_off"), f"{path}.g_off") != 0.0:
            raise _error("the frozen g_off=0", calibration.get("g_off"), f"{path}.g_off")
        model_params = base.data["run"]["model"]["hard_sigmoid_param"]
        for key in ("v_off", "g_on", "g_off"):
            if float(calibration[key]) != float(model_params[key]):
                raise _error(
                    f"the base model value {model_params[key]!r}",
                    calibration[key],
                    f"{path}.{key}",
                )
        target_key = "target_initial_saturation"
        measured_key = "measured_initial_saturation"
        layer_value_key = "measured_saturation"
    else:
        if calibration.get("kind") != "perfect_diode_clamped_occupancy":
            raise _error("'perfect_diode_clamped_occupancy'", calibration.get("kind"), f"{path}.kind")
        if _finite(calibration.get("clamp_epsilon"), f"{path}.clamp_epsilon", positive=True) != 1e-8:
            raise _error("the frozen clamp epsilon 1e-8", calibration.get("clamp_epsilon"), f"{path}.clamp_epsilon")
        target_key = "target_initial_occupancy"
        measured_key = "measured_initial_occupancy"
        layer_value_key = "measured_clamped_occupancy"
    if _finite(calibration.get(target_key), f"{path}.{target_key}") != 0.3:
        raise _error("the frozen 30% target", calibration.get(target_key), f"{path}.{target_key}")
    first_measurement = _finite(calibration.get(measured_key), f"{path}.{measured_key}")
    for offset, layer in enumerate(layer_measurements, start=1):
        expected_keys = {"layer_index", layer_value_key}
        if not isinstance(layer, dict) or set(layer) != expected_keys:
            raise _error(
                f"an object with exactly keys {sorted(expected_keys)}",
                layer,
                f"{path}.layer_measurements[{offset - 1}]",
            )
        if layer["layer_index"] != offset:
            raise _error(
                f"the one-based layer index {offset}",
                layer["layer_index"],
                f"{path}.layer_measurements[{offset - 1}].layer_index",
            )
        value = _finite(layer[layer_value_key], f"{path}.layer_measurements[{offset - 1}].{layer_value_key}")
        if not 0.0 <= value <= 1.0:
            raise _error("a fraction in [0, 1]", value, f"{path}.layer_measurements[{offset - 1}].{layer_value_key}")
        if offset == 1 and value != first_measurement:
            raise _error(
                f"the first-hidden measurement {first_measurement}",
                value,
                f"{path}.layer_measurements[0].{layer_value_key}",
            )
    return calibration


def build_sweep(
    *,
    base_run_path: str | Path,
    calibration_records_path: str | Path,
    name: str,
    seeds: list[int],
) -> SweepSpec:
    base = RunSpec.from_path(base_run_path)
    if not isinstance(name, str) or not name.strip():
        raise _error("a non-empty string", name, "sweep name")
    if not seeds or len(seeds) != len(set(seeds)):
        raise _error("a non-empty list of unique seeds in listed order", seeds, "seeds")
    normalized_seeds = [_integer(seed, f"seeds[{index}]") for index, seed in enumerate(seeds)]
    if len(normalized_seeds) == 1 and normalized_seeds[0] != base.data["seed"]:
        raise _error(
            "either the base-run seed or at least two explicit axis values",
            normalized_seeds,
            "single-seed coverage",
        )

    records = _records_from_path(calibration_records_path)
    by_case: dict[str, dict[str, Any]] = {}
    calibration_ids: set[str] = set()
    cases: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        _validate_fixed_protocol(record, base, index)
        case_id = record["case_id"]
        if not isinstance(case_id, str):
            raise _error("a canonical case-id string", case_id, f"calibration records.records[{index}].case_id")
        if case_id not in REQUIRED_CASES:
            raise _error(f"one of {list(REQUIRED_CASES)}", case_id, f"calibration records.records[{index}].case_id")
        if case_id in by_case:
            raise _error("a unique case id", case_id, f"calibration records.records[{index}].case_id")
        expected_amp = REQUIRED_CASES[case_id]
        amp = (
            _finite(record["voltage_amp"], f"calibration records.records[{index}].voltage_amp", positive=True),
            _finite(record["current_amp"], f"calibration records.records[{index}].current_amp", positive=True),
        )
        if amp != expected_amp:
            raise _error(
                f"the frozen amplification pair {expected_amp!r}",
                amp,
                f"calibration records.records[{index}] amplification",
            )
        calibration = _validate_calibration(record, base, index)
        calibration_id = calibration.get("calibration_id")
        if not isinstance(calibration_id, str) or not calibration_id.strip():
            raise _error("a non-empty string", calibration_id, f"calibration records.records[{index}].calibration.calibration_id")
        if calibration_id in calibration_ids:
            raise _error("a unique calibration id", calibration_id, f"calibration records.records[{index}].calibration.calibration_id")
        calibration_ids.add(calibration_id)
        assignments = {
            "/run/model/voltage_amp": amp[0],
            "/run/model/current_amp": amp[1],
            "/run/model/input_gain": _finite(
                record["input_gain"],
                f"calibration records.records[{index}].input_gain",
                positive=True,
            ),
            "/run/calibration": calibration,
        }
        cases.append({"id": case_id, "set": assignments})
        by_case[case_id] = record

    missing = [case_id for case_id in REQUIRED_CASES if case_id not in by_case]
    if missing or len(by_case) != len(REQUIRED_CASES):
        raise _error(
            f"exactly the frozen cases {list(REQUIRED_CASES)}",
            {"present": list(by_case), "missing": missing},
            "calibration cases",
        )

    axes: list[dict[str, Any]] = []
    varying_fields = [
        "/run/model/voltage_amp",
        "/run/model/current_amp",
        "/run/model/input_gain",
        "/run/calibration",
    ]
    if len(normalized_seeds) > 1:
        axes.append({"path": "/seed", "values": normalized_seeds})
        varying_fields.append("/seed")

    sweep = {
        "schema_version": "mnist-conv-sweep/v1",
        "name": name.strip(),
        "base_run": base.to_dict(),
        "axes": axes,
        "cases": cases,
        "varying_fields": varying_fields,
        "collection": {
            "expected_seeds": normalized_seeds,
            "required_cases": [case["id"] for case in cases],
            "group_by": [
                "/run/architecture/profile",
                "/run/model/non_linearity",
                "/run/model/voltage_amp",
                "/run/model/current_amp",
            ],
        },
    }
    return SweepSpec.from_dict(sweep)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run", required=True, help="Complete canonical run JSON; all training settings are preserved.")
    parser.add_argument("--calibration-records", required=True, help="Strict canonical calibration-record JSON.")
    parser.add_argument("--name", required=True, help="Canonical sweep name.")
    parser.add_argument("--seeds", required=True, type=int, nargs="+", help="Explicit model-seed coverage in listed order.")
    parser.add_argument("--output-sweep", required=True, help="Destination canonical sweep JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        sweep = build_sweep(
            base_run_path=args.base_run,
            calibration_records_path=args.calibration_records,
            name=args.name,
            seeds=args.seeds,
        )
        output = Path(args.output_sweep).expanduser().resolve()
        value = sweep.to_dict()
        if output.exists():
            existing = _read_json(output, "existing output sweep")
            if existing != value:
                raise ConversionError(
                    f"Expected existing output {output} to be identical. Provided different content; choose a new output path."
                )
        else:
            atomic_write_json(output, value)
    except (ConversionError, SpecValidationError, OSError) as exc:
        print(str(exc), flush=True)
        return 2
    print(json.dumps({"output_sweep": str(output), "jobs": len(sweep.expand())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
