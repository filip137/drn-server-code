#!/usr/bin/env python3
"""Freeze exact Adam configs for the Conv1/2/3 fixed-initializer wmax study."""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
STUDY_ID = (
    "perfectdiode-conv123-fixed-uniform-init-wmax-adam-"
    "10-30-30ep-seed0-20260814-v1"
)
EVIDENCE_CLASS = "ordinary_mnist_weight_ceiling_sensitivity_exploratory"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "configs" / "conv" / STUDY_ID
RESULT_ASSET_ROOT = Path("results") / STUDY_ID / "assets" / "bounded_uniform"


@dataclass(frozen=True)
class Architecture:
    name: str
    epochs: int
    iterations: int
    input_gain: float
    parameter_order: tuple[str, ...]
    initializer_source: str
    initializer_sha256: str
    walltime: str

    @property
    def bias_names(self) -> tuple[str, ...]:
        return tuple(name for name in self.parameter_order if name.startswith("Bias_"))


ARCHITECTURES = (
    Architecture(
        name="conv1",
        epochs=10,
        iterations=4,
        input_gain=40.0,
        parameter_order=("ConvWeight_0", "DenseWeight_0", "Bias_0"),
        initializer_source=(
            "results/perfectdiode-bounded-uniform-zero-bias-lr-search-conv123-"
            "seed0-20260810-v1/local_gate_frozen96_v2/conv1/assets/"
            "bounded_uniform/conv1/final_model.pt"
        ),
        initializer_sha256=(
            "235d7ae56a277e097e840a4eb11cf16e325ed2d3eb80c9055f7903493cb1f16e"
        ),
        walltime="00:45:00",
    ),
    Architecture(
        name="conv2",
        epochs=30,
        iterations=6,
        input_gain=100.0,
        parameter_order=(
            "ConvWeight_0",
            "ConvWeight_1",
            "DenseWeight_0",
            "Bias_0",
            "Bias_1",
        ),
        initializer_source=(
            "results/perfectdiode-bounded-uniform-zero-bias-lr-search-conv123-"
            "seed0-20260810-v1/local_gate_frozen96_v2/conv2/assets/"
            "bounded_uniform/conv2/final_model.pt"
        ),
        initializer_sha256=(
            "6469bc8f465e9abd2b4720231ad03d0116fc176ef4e437dc005027204c4d88d2"
        ),
        walltime="03:00:00",
    ),
    Architecture(
        name="conv3",
        epochs=30,
        iterations=8,
        input_gain=360.0,
        parameter_order=(
            "ConvWeight_0",
            "ConvWeight_1",
            "ConvWeight_2",
            "DenseWeight_0",
            "Bias_0",
            "Bias_1",
            "Bias_2",
        ),
        initializer_source=(
            "results/perfectdiode-bounded-uniform-zero-bias-lr-search-conv123-"
            "seed0-20260810-v1/local_gate_frozen96_v2/conv3/assets/"
            "bounded_uniform/conv3/final_model.pt"
        ),
        initializer_sha256=(
            "8ebe9dd916e1e9299c31053e2bb37b4f5121f8322f26af7746a92875e555438f"
        ),
        walltime="05:00:00",
    ),
)
ARCHITECTURE_BY_NAME = {architecture.name: architecture for architecture in ARCHITECTURES}


@dataclass(frozen=True)
class WmaxCase:
    tag: str
    value: float


WMAX_CASES = (
    WmaxCase("1em4", 1e-4),
    WmaxCase("5em4", 5e-4),
    WmaxCase("1em3", 1e-3),
)


@dataclass(frozen=True)
class Surface:
    architecture: str
    scheme: str
    source_config: str
    source_config_sha256: str
    rho_conv: float
    rho_dense: float
    validation_accuracy_percent: float
    range_status: str
    rates: tuple[float, ...]

    @property
    def key(self) -> str:
        return f"{self.architecture}_{self.scheme}_adam"


SURFACES = (
    Surface(
        "conv1",
        "baseline",
        "results/perfectdiode-bounded-uniform-zero-bias-lr-search-conv123-seed0-20260810-v1/shards/main-r1/surfaces/bounded_uniform__conv1__baseline__adam/rho/cells/078_rc_0p003_rd_0p03/config.used.json",
        "b26f6bc99e6abb6fe521f4b22546d6ec87fd86ffb5a3f6a878fe86c3e3a925d1",
        0.003,
        0.03,
        88.84,
        "bracketed",
        (1.8163761812439207e-7, 1.8257546674661046e-6, 0.0),
    ),
    Surface(
        "conv1",
        "ours",
        "results/perfectdiode-bounded-uniform-zero-bias-lr-search-conv123-seed0-20260810-v1/shards/main-r1/surfaces/bounded_uniform__conv1__ours__adam/rho/cells/067_rc_0p001_rd_0p01/config.used.json",
        "2ab0708e0c700dde9af600c411e7d3bd540790f0674dfae618d61d4ea01e7108",
        0.001,
        0.01,
        91.60,
        "bracketed",
        (6.054587142218743e-8, 6.08565526037015e-7, 0.0),
    ),
    Surface(
        "conv1",
        "legacy",
        "results/perfectdiode-bounded-uniform-zero-bias-lr-search-conv123-seed0-20260810-v1/shards/main-r1/surfaces/bounded_uniform__conv1__legacy__adam/rho/cells/087_rc_0p009_rd_0p01/config.used.json",
        "c18431d9b92317ece9c3f510bd63ac8a9c464215343aff16ba92d232821bf10b",
        0.009,
        0.01,
        95.42,
        "bracketed",
        (5.449128249811996e-7, 6.085548426894188e-7, 0.0),
    ),
    Surface(
        "conv2",
        "baseline",
        "results/perfectdiode-conv12-directed-exploratory-lr-search-seed0-20260812-v1/shards/jean-zay-task_9/surfaces/bounded_uniform__conv2__baseline__adam/rho/cells/021_rc_0p243_rd_0p00333333333333/config.used.json",
        "c9409b6f852bcde474b3b7e5c2a50f57bfd250575a63916bdbedb0322271ad14",
        0.243,
        0.0033333333333333335,
        84.54,
        "open_upper_rho_conv",
        (
            1.471266669772638e-5,
            1.4788493028149698e-5,
            2.038514659443057e-7,
            0.0,
            0.0,
        ),
    ),
    Surface(
        "conv2",
        "ours",
        "results/perfectdiode-conv12-directed-exploratory-lr-search-seed0-20260812-v1/shards/jean-zay-task_10/surfaces/bounded_uniform__conv2__ours__adam/rho/cells/010_rc_0p027_rd_0p01/config.used.json",
        "1eeab7ad2fcccca32c6d74efdaaf2f417f65beb650a23cf74606e93045621803",
        0.027,
        0.01,
        87.70,
        "bracketed",
        (
            1.6347391583547683e-6,
            1.6428324925911035e-6,
            6.094626207410665e-7,
            0.0,
            0.0,
        ),
    ),
    Surface(
        "conv2",
        "legacy",
        "results/perfectdiode-bounded-uniform-zero-bias-lr-search-conv123-seed0-20260810-v1/shards/akib-r1/surfaces/bounded_uniform__conv2__legacy__adam/rho/cells/087_rc_0p009_rd_0p01/config.used.json",
        "2d4c424c00418d05e19c5c466b67e5cab6483c6592f4689ead1d058283353c70",
        0.009,
        0.01,
        95.42,
        "bracketed",
        (
            5.449128384578525e-7,
            5.475686125795819e-7,
            6.092466384879761e-7,
            0.0,
            0.0,
        ),
    ),
    Surface(
        "conv3",
        "baseline",
        "results/perfectdiode-conv3-exploratory-lr-ladder-seed0-20260811-v1/shards/trex-task_1/surfaces/bounded_uniform__conv3__baseline__adam/rho/cells/045_rc_0p001_rd_0p00333333333333/config.used.json",
        "eb9ffaeb3ad706585cf87ae9356f6e4fecfeea25259507595eab1ee2d081f441",
        0.001,
        0.0033333333333333335,
        76.16,
        "locally_bracketed_by_accuracy",
        (
            6.054726133320255e-8,
            6.143696204539704e-8,
            6.14562270686436e-8,
            2.099421760206363e-7,
            0.0,
            0.0,
            0.0,
        ),
    ),
    Surface(
        "conv3",
        "ours",
        "results/perfectdiode-conv3-exploratory-lr-ladder-seed0-20260811-v1/shards/trex-task_3/surfaces/bounded_uniform__conv3__ours__adam/rho/cells/061_rc_0p009_rd_0p00333333333333/config.used.json",
        "5feb3c48bd6dd62a336b59cbde2c26553728501f664aa1721fd1a182bced7cea",
        0.009,
        0.0033333333333333335,
        80.90,
        "open_upper_rho_conv",
        (
            5.449156625672776e-7,
            5.489725044358455e-7,
            5.491381900061772e-7,
            2.039674013983286e-7,
            0.0,
            0.0,
            0.0,
        ),
    ),
    Surface(
        "conv3",
        "legacy",
        "results/perfectdiode-bounded-uniform-zero-bias-lr-search-conv3-occupancy-report-only-seed0-20260811-v1/shards/jean-zay-task_1/surfaces/bounded_uniform__conv3__legacy__adam/rho/cells/077_rc_0p003_rd_0p01/config.used.json",
        "22b2c0d50b97290f909666717768d5375f3ee54ae8a31b619d454a6bc4acffbb",
        0.003,
        0.01,
        88.66,
        "highest_observed_below_accuracy_gate_noncanonical",
        (
            1.816376567545556e-7,
            1.8256616170160948e-7,
            1.8261822032001863e-7,
            6.08744636058677e-7,
            0.0,
            0.0,
            0.0,
        ),
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _architecture(name: str) -> Architecture:
    try:
        return ARCHITECTURE_BY_NAME[name]
    except KeyError as exc:
        raise ValueError(f"Unknown architecture: {name}") from exc


def _load_source(surface: Surface) -> dict[str, Any]:
    architecture = _architecture(surface.architecture)
    path = REPO_ROOT / surface.source_config
    if _sha256(path) != surface.source_config_sha256:
        raise ValueError(f"Frozen source config digest mismatch: {path}")
    source = json.loads(path.read_text(encoding="utf-8"))
    model = source.get("model_base", {})
    if source.get("optimizer", {}).get("name") != "Adam":
        raise ValueError(f"Source optimizer is not Adam: {path}")
    if source.get("lr") != list(surface.rates):
        raise ValueError(f"Frozen LR vector does not match source config: {path}")
    source_order = source.get("parameter_order")
    if source_order not in (None, list(architecture.parameter_order)):
        raise ValueError(f"Unexpected source parameter order: {path}")
    rate_map = dict(zip(architecture.parameter_order, surface.rates, strict=True))
    if any(rate_map[name] != 0.0 for name in architecture.bias_names):
        raise ValueError(f"Expected exact-zero bias rates: {path}")
    if model.get("weight_init_mode") != "bounded_uniform":
        raise ValueError(f"Source initializer is not bounded_uniform: {path}")
    if float(model.get("weight_min")) != 1e-5 or float(model.get("weight_max")) != 1e-4:
        raise ValueError(f"Unexpected source weight interval: {path}")
    if int(model.get("num_iterations_inference", -1)) != architecture.iterations or int(
        model.get("num_iterations_training", -1)
    ) != architecture.iterations:
        raise ValueError(f"Unexpected accepted T/K for {architecture.name}: {path}")
    if float(model.get("input_gain")) != architecture.input_gain:
        raise ValueError(f"Unexpected input gain for {architecture.name}: {path}")
    if source.get("training_algorithm") != "BP" or int(source.get("seed", -1)) != 0:
        raise ValueError(f"Unexpected source algorithm or seed: {path}")
    for key in ("exponential_diode_param", "quadratic_diode_param", "hard_sigmoid_param"):
        if not isinstance(model.get(key), dict):
            raise ValueError(f"Missing explicit {key} in {path}")
    return source


def _arm_id(architecture: Architecture, wmax: WmaxCase, surface: Surface) -> str:
    return (
        f"{architecture.name}_wmax_{wmax.tag}_{surface.scheme}_adam_"
        "fixed_uniform_init_seed0"
    )


def _filename(index: int, wmax: WmaxCase, surface: Surface) -> str:
    return f"{index:02d}_wmax_{wmax.tag}_{surface.scheme}_adam.json"


def _config(
    architecture: Architecture,
    wmax: WmaxCase,
    surface: Surface,
) -> dict[str, Any]:
    config = deepcopy(_load_source(surface))
    arm_id = _arm_id(architecture, wmax, surface)
    checkpoint = RESULT_ASSET_ROOT / architecture.name / "final_model.pt"
    rate_map = dict(zip(architecture.parameter_order, surface.rates, strict=True))
    config["study_id"] = STUDY_ID
    config["arm_id"] = arm_id
    config["init_checkpoint_path"] = checkpoint.as_posix()
    config["parameter_order"] = list(architecture.parameter_order)
    config["learning_rates_by_parameter"] = rate_map
    config["lr"] = list(surface.rates)
    config["optimizer"]["learning_rate"] = list(surface.rates)
    config["lab"]["epochs"] = architecture.epochs
    config["model_base"]["weight_init_mode"] = "bounded_uniform"
    config["model_base"]["weight_min"] = 1e-5
    config["model_base"]["weight_max"] = wmax.value
    config["evaluation"] = {
        "checkpoint_selection": "maximum_validation_accuracy",
        "official_test": {"policy": "disabled"},
    }
    config["reporting"] = {
        "study_id": STUDY_ID,
        "arm_id": arm_id,
        "evidence_class": EVIDENCE_CLASS,
        "paper_facing": False,
    }
    config["weight_ceiling_sweep"] = {
        "canonical_lr_handoff": False,
        "source_transform_fields": ["lab.epochs", "model_base.weight_max"],
        "varied_field_across_cases": "model_base.weight_max",
        "fixed_raw_learning_rates": True,
        "fixed_initial_parameter_states": True,
        "initializer": "bounded_uniform",
        "initializer_support": [1e-5, 1e-4],
        "initializer_checkpoint_path": checkpoint.as_posix(),
        "initializer_checkpoint_sha256": architecture.initializer_sha256,
        "training_projection_interval": [1e-5, wmax.value],
        "rho_conv_provenance": surface.rho_conv,
        "rho_dense_provenance": surface.rho_dense,
        "source_config": surface.source_config,
        "source_config_sha256": surface.source_config_sha256,
        "source_final_validation_accuracy_percent": surface.validation_accuracy_percent,
        "source_range_status": surface.range_status,
        "scientific_gate_policy": (
            "user-waived repeated security and post-training T/K checks; "
            f"reuse accepted {architecture.name} T=K={architecture.iterations}"
        ),
    }
    return config


def prepare(output_root: Path) -> dict[str, Any]:
    output_root = Path(output_root).expanduser().resolve()
    for architecture in ARCHITECTURES:
        source = REPO_ROOT / architecture.initializer_source
        if _sha256(source) != architecture.initializer_sha256:
            raise ValueError(f"Frozen initializer digest mismatch: {source}")

    rows: list[dict[str, Any]] = []
    ordered_config_set_sha256_by_architecture: dict[str, str] = {}
    for architecture in ARCHITECTURES:
        architecture_rows: list[dict[str, Any]] = []
        architecture_surfaces = tuple(
            surface for surface in SURFACES if surface.architecture == architecture.name
        )
        if tuple(surface.scheme for surface in architecture_surfaces) != (
            "baseline",
            "ours",
            "legacy",
        ):
            raise ValueError(f"Unexpected surface order for {architecture.name}")
        index = 0
        for wmax in WMAX_CASES:
            for surface in architecture_surfaces:
                filename = _filename(index, wmax, surface)
                relative_config = Path(architecture.name) / filename
                path = output_root / relative_config
                _write_json(path, _config(architecture, wmax, surface))
                row = {
                    "index": index,
                    "architecture": architecture.name,
                    "scheme": surface.scheme,
                    "optimizer": "Adam",
                    "arm_id": _arm_id(architecture, wmax, surface),
                    "config": relative_config.as_posix(),
                    "config_sha256": _sha256(path),
                    "epochs": architecture.epochs,
                    "T": architecture.iterations,
                    "K": architecture.iterations,
                    "weight_min": 1e-5,
                    "weight_max": wmax.value,
                    "weight_max_tag": wmax.tag,
                    "learning_rates_by_parameter": dict(
                        zip(architecture.parameter_order, surface.rates, strict=True)
                    ),
                    "rho_conv_provenance": surface.rho_conv,
                    "rho_dense_provenance": surface.rho_dense,
                    "source_config": surface.source_config,
                    "source_config_sha256": surface.source_config_sha256,
                    "source_final_validation_accuracy_percent": (
                        surface.validation_accuracy_percent
                    ),
                    "source_range_status": surface.range_status,
                    "initializer_checkpoint_sha256": architecture.initializer_sha256,
                }
                architecture_rows.append(row)
                rows.append(row)
                index += 1
        ordered_config_set_sha256_by_architecture[architecture.name] = hashlib.sha256(
            "".join(f"{row['config_sha256']}\n" for row in architecture_rows).encode(
                "ascii"
            )
        ).hexdigest()

    manifest = {
        "schema": "perfectdiode-conv123-fixed-uniform-init-wmax-adam/v1",
        "study_id": STUDY_ID,
        "evidence_class": EVIDENCE_CLASS,
        "paper_facing": False,
        "canonical_lr_handoff": False,
        "scientific_role": "fixed-initial-state upper-weight-limit sensitivity",
        "selection_rule": (
            "highest observed final validation accuracy in the reviewed bounded-uniform "
            "surface; Conv3 legacy is explicitly below its historical accuracy gate"
        ),
        "run_count": len(rows),
        "run_count_by_architecture": {
            architecture.name: 9 for architecture in ARCHITECTURES
        },
        "ordered_config_set_sha256_by_architecture": (
            ordered_config_set_sha256_by_architecture
        ),
        "initializers": {
            architecture.name: {
                "mode": "bounded_uniform",
                "support": [1e-5, 1e-4],
                "checkpoint_source": architecture.initializer_source,
                "checkpoint_path": (
                    RESULT_ASSET_ROOT / architecture.name / "final_model.pt"
                ).as_posix(),
                "checkpoint_sha256": architecture.initializer_sha256,
                "shared_exactly_across_all_architecture_runs": True,
            }
            for architecture in ARCHITECTURES
        },
        "varied_field": "model_base.weight_max",
        "weight_max_values": [case.value for case in WMAX_CASES],
        "fixed_fields": {
            "dataset": "ordinary_mnist_train_validation",
            "optimizer": "Adam",
            "epochs_by_architecture": {
                architecture.name: architecture.epochs for architecture in ARCHITECTURES
            },
            "T_K_by_architecture": {
                architecture.name: architecture.iterations
                for architecture in ARCHITECTURES
            },
            "seed": 0,
            "bias_learning_rates": 0.0,
            "official_test_read": False,
        },
        "execution": {
            "target": "jean-zay",
            "dossier": "AD010913993R3",
            "account": "fmu@v100",
            "partition": "gpu_p13",
            "qos": "qos_gpu-t3",
            "constraint": "v100-16g",
            "array_by_architecture": {
                architecture.name: "0-8%9" for architecture in ARCHITECTURES
            },
            "time_limit_by_architecture": {
                architecture.name: architecture.walltime for architecture in ARCHITECTURES
            },
            "maximum_concurrent_gpus": 27,
            "expected_gpu_hours": 52,
            "maximum_gpu_hours": 78.75,
        },
        "protocol_deviation": (
            "User-authorized exploratory omission of repeated gradient security and "
            "post-training T/K checks; accepted T/K operating points are reused. "
            "Conv3 best-observed learning rates are noncanonical."
        ),
        "runs": rows,
    }
    _write_json(output_root / "study_manifest.json", manifest)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    print(json.dumps(prepare(args.output_root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
