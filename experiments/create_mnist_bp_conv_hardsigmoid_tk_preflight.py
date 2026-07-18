#!/usr/bin/env python3
"""Create medium-affine hard-sigmoid source configs for Conv T/K diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    "mnist_bp_amp_v1_c1": (1.0, 1.0),
    "mnist_bp_amp_v4_c1": (4.0, 1.0),
    "mnist_bp_amp_v4_c0p25": (4.0, 0.25),
}
ARCHITECTURES = {
    1: {"channels": [64], "strides": [2], "paddings": [1]},
    2: {"channels": [64, 128], "strides": [2, 2], "paddings": [1, 1]},
    3: {"channels": [64, 128, 256], "strides": [2, 2, 1], "paddings": [1, 1, 1]},
}
AFFINE_CONFIG = {
    "enabled": True,
    "preset": "medium",
    "degrees": 25.0,
    "translate": [0.2, 0.2],
    "scale": [0.8, 1.2],
    "shear": 0.0,
    "seed": 1729,
    "interpolation": "bilinear",
    "fill": 0.0,
}
MANIFEST_COLUMNS = [
    "conv_depth",
    "run_name",
    "seed",
    "voltage_amp",
    "current_amp",
    "input_gain",
    "gain_source_T",
    "gain_source_K",
    "operational_T",
    "operational_K",
    "gain_source_dataset",
    "gain_source_measured_saturation",
    "target_saturation",
    "source_config_path",
]


def _load_json_object(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Expected {label} to be a readable JSON object. Provided value: {path}."
        ) from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected {label} to be a JSON object. Provided value: {value!r}.")
    return value


def _parse_int_list(value: str, label: str) -> list[int]:
    try:
        if value.strip().startswith("["):
            parsed = json.loads(value)
            if not isinstance(parsed, list):
                raise ValueError
            result = [int(part) for part in parsed]
        else:
            result = [int(part.strip()) for part in value.split(",") if part.strip()]
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Expected {label} to be a JSON or comma-separated integer list. "
            f"Provided value: {value!r}."
        ) from exc
    if not result:
        raise ValueError(
            f"Expected {label} to contain at least one integer. Provided value: {value!r}."
        )
    return result


def _load_gains(path: Path) -> dict[tuple[int, str], dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Expected --gain-csv to point to an existing CSV. Provided value: {path}."
        )
    rows: dict[tuple[int, str], dict] = {}
    with path.open(newline="") as handle:
        for raw in csv.DictReader(handle):
            try:
                depth = int(raw["conv_depth"])
                run_name = raw["run_name"]
                gain = float(raw["input_gain"])
                target = float(raw["target_saturation"])
                v_off = float(raw["v_off"])
                seed = int(raw["seed"])
                sample_count = int(raw["num_samples"])
                batch_size = int(raw["batch_size"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    "Expected every gain row to define conv_depth, run_name, input_gain, "
                    "target_saturation, v_off, seed, num_samples, and batch_size. "
                    f"Provided value: {raw!r}."
                ) from exc
            if depth not in ARCHITECTURES or run_name not in RUNS:
                continue
            expected_strides = ARCHITECTURES[depth]["strides"]
            expected_paddings = ARCHITECTURES[depth]["paddings"]
            provided_strides = _parse_int_list(raw.get("strides", ""), "strides")
            provided_paddings = _parse_int_list(raw.get("paddings", ""), "paddings")
            if provided_strides != expected_strides or provided_paddings != expected_paddings:
                raise ValueError(
                    f"Expected Conv{depth} strides/paddings to be "
                    f"{expected_strides}/{expected_paddings}. Provided value: "
                    f"{provided_strides}/{provided_paddings}."
                )
            expected_amp = RUNS[run_name]
            provided_amp = (float(raw["voltage_amp"]), float(raw["current_amp"]))
            if provided_amp != expected_amp:
                raise ValueError(
                    f"Expected {run_name} amplification to be {expected_amp}. "
                    f"Provided value: {provided_amp}."
                )
            if not math.isfinite(gain) or gain <= 0.0:
                raise ValueError(
                    f"Expected input_gain to be positive and finite. Provided value: {gain!r}."
                )
            if target != 0.3 or v_off != 4.0 or seed != 0 or sample_count != 256 or batch_size != 64:
                raise ValueError(
                    "Expected gain provenance target/v_off/seed/samples/batch to be "
                    f"0.3/4.0/0/256/64. Provided value: "
                    f"{target}/{v_off}/{seed}/{sample_count}/{batch_size}."
                )
            key = (depth, run_name)
            if key in rows:
                raise ValueError(f"Expected one gain row per architecture and scheme. Provided duplicate: {key}.")
            rows[key] = dict(raw)

    expected = {(depth, run_name) for depth in ARCHITECTURES for run_name in RUNS}
    missing = sorted(expected - set(rows))
    if missing:
        raise ValueError(
            f"Expected gain CSV coverage for all nine hard-sigmoid rows. Provided missing rows: {missing}."
        )
    return rows


def _conv_shapes(depth: int) -> tuple[list[list[int]], list[dict]]:
    arch = ARCHITECTURES[depth]
    height = width = 28
    shapes: list[list[int]] = [[2, height, width]]
    pipeline: list[dict] = []
    for channels, stride, padding in zip(arch["channels"], arch["strides"], arch["paddings"]):
        height = (height + 2 * padding - 3) // stride + 1
        width = (width + 2 * padding - 3) // stride + 1
        shapes.append([channels, height, width])
        pipeline.append(
            {"kernel": [3, 3], "stride": stride, "padding": padding, "mode": "convolution"}
        )
    shapes.append([20])
    return shapes, pipeline


def _source_config(
    *,
    depth: int,
    run_name: str,
    gain_row: dict,
    dataset_root: Path,
    minimizer: dict,
    gain_source_dataset: str,
) -> dict:
    voltage_amp, current_amp = RUNS[run_name]
    input_gain = float(gain_row["input_gain"])
    layer_shapes, conv_pipeline = _conv_shapes(depth)
    model_key = "mnist_bp_conv_amp"
    return {
        "lab": {"model_key": model_key, "dataset_key": "mnist", "epochs": 0},
        "input_mode": "train",
        "training_algorithm": "BP",
        "seed": 0,
        "batch_state_policy": "reset_each_batch",
        "lr": [1.44 / input_gain],
        "beta": 1.0,
        "lr_decay": 1.0,
        "max_batches": None,
        "max_test_batches": None,
        "datasets": {
            "mnist": {
                "factory": "labs.datasets.AffineMnistDataset",
                "params": {
                    "name": "mnist",
                    "batch_size": 64,
                    "root": str(dataset_root),
                    "train": True,
                    "download": False,
                    "normalize": True,
                    "normalize_mean": 0.1307,
                    "normalize_std": 0.3081,
                    "normalize_scale": 0.3,
                    "shuffle_seed": 0,
                    "affine_config": dict(AFFINE_CONFIG),
                },
            }
        },
        "model_base": {
            "weight_min": 0.0,
            "weight_max": 100.0,
            "weight_init_mode": "kaiming_uniform",
            "input_gain": input_gain,
            "voltage_amp": voltage_amp,
            "current_amp": current_amp,
            "trainable_amplification": False,
            "amplification_min": 1.0e-6,
            "amplification_max": None,
            "non_linearity": "hard_sigmoid",
            "quadratic_diode_param": {
                "diode_conductance": 1.0,
                "v_min": -1.0e6,
                "v_max": 1.0e6,
            },
            "exponential_diode_param": {"I_s": 1.0e-6, "V_t": 0.025, "V_off": 0.0},
            "hard_sigmoid_param": {"g_on": 100.0, "g_off": 0.0, "v_off": 4.0},
            "num_iterations_inference": 64,
            "num_iterations_training": 64,
            "minimizer": json.loads(json.dumps(minimizer)),
        },
        "model_overrides": {
            model_key: {
                "layer_shapes": layer_shapes,
                "conv_pipeline": conv_pipeline,
                "weight_gains": [1.0] * (depth + 1),
            }
        },
        "energy_minimizer": {"mode": "asynchronous"},
        "diagnostic_provenance": {
            "protocol": "conv-paper-hard-sigmoid-tk-20260718",
            "gain_csv_row": gain_row,
            "gain_source_dataset": gain_source_dataset,
            "verification_dataset": "deterministic_medium_affine_mnist",
            "affine_config": dict(AFFINE_CONFIG),
            "gain_verification_T": 64,
            "gain_verification_samples": 256,
            "gain_verification_batch_size": 64,
            "adaptive_equilibrium": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gain-csv", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument(
        "--gain-source-dataset",
        required=True,
        choices=(
            "ordinary_mnist_superseded_diagnostic",
            "deterministic_medium_affine_mnist_t64",
        ),
    )
    parser.add_argument(
        "--minimizer-config",
        default=str(REPO_ROOT / "labs" / "configs" / "mnist_minimizer_fixed_iterations.json"),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    gain_csv = Path(args.gain_csv).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    minimizer_path = Path(args.minimizer_config).expanduser().resolve()
    minimizer = _load_json_object(minimizer_path, "--minimizer-config")
    if minimizer.get("adaptive_equilibrium") is not False:
        raise ValueError(
            "Expected minimizer adaptive_equilibrium to be false. "
            f"Provided value: {minimizer.get('adaptive_equilibrium')!r}."
        )
    gains = _load_gains(gain_csv)

    manifest_rows = []
    for depth in sorted(ARCHITECTURES):
        for run_name in RUNS:
            gain_row = gains[(depth, run_name)]
            run_dir = output_root / f"conv{depth}" / "hard_sigmoid" / run_name / "seed_0"
            source_path = run_dir / "source_config.json"
            if source_path.exists() and not args.overwrite:
                raise FileExistsError(
                    f"Expected a new source-config path or --overwrite. Provided value: {source_path}."
                )
            run_dir.mkdir(parents=True, exist_ok=True)
            config = _source_config(
                depth=depth,
                run_name=run_name,
                gain_row=gain_row,
                dataset_root=dataset_root,
                minimizer=minimizer,
                gain_source_dataset=args.gain_source_dataset,
            )
            source_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
            voltage_amp, current_amp = RUNS[run_name]
            manifest_rows.append(
                {
                    "conv_depth": depth,
                    "run_name": run_name,
                    "seed": 0,
                    "voltage_amp": voltage_amp,
                    "current_amp": current_amp,
                    "input_gain": gain_row["input_gain"],
                    "gain_source_T": 64,
                    "gain_source_K": "",
                    "operational_T": gain_row.get("selected_T", ""),
                    "operational_K": gain_row.get("selected_K", ""),
                    "gain_source_dataset": args.gain_source_dataset,
                    "gain_source_measured_saturation": gain_row.get("measured_saturation", ""),
                    "target_saturation": 0.3,
                    "source_config_path": str(source_path),
                }
            )
            print(f"[preflight] wrote {source_path}", flush=True)

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "preflight_manifest.csv"
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(manifest_rows)
    metadata = {
        "protocol": "conv-paper-hard-sigmoid-tk-20260718",
        "gain_csv": str(gain_csv),
        "gain_source_dataset": args.gain_source_dataset,
        "verification_dataset": "deterministic_medium_affine_mnist",
        "affine_config": AFFINE_CONFIG,
        "num_rows": len(manifest_rows),
        "fixed_step": True,
        "calibration_reference_T": 64,
    }
    (output_root / "preflight_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(f"[preflight] rows={len(manifest_rows)} manifest={manifest_path}")


if __name__ == "__main__":
    main()
