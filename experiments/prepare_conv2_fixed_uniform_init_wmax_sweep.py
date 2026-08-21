#!/usr/bin/env python3
"""Freeze the exact configs for the Conv2 fixed-initializer wmax sweep."""

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
    "perfectdiode-conv2-fixed-uniform-init-wmax-sweep-sgd-adam-"
    "10ep-seed0-20260813-v1"
)
EVIDENCE_CLASS = "ordinary_mnist_weight_ceiling_sensitivity_exploratory"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "configs" / "conv" / STUDY_ID
RESULT_ASSET_ROOT = Path("results") / STUDY_ID / "assets" / "bounded_uniform" / "conv2"
INITIALIZER_SOURCE = (
    REPO_ROOT
    / "results/perfectdiode-bounded-uniform-zero-bias-lr-search-conv123-"
    "seed0-20260810-v1/shards/akib-r1/assets/bounded_uniform/conv2/final_model.pt"
)
INITIALIZER_SHA256 = "6469bc8f465e9abd2b4720231ad03d0116fc176ef4e437dc005027204c4d88d2"
PARAMETER_ORDER = (
    "ConvWeight_0",
    "ConvWeight_1",
    "DenseWeight_0",
    "Bias_0",
    "Bias_1",
)


@dataclass(frozen=True)
class WmaxCase:
    tag: str
    value: float


WMAX_CASES = (
    WmaxCase("1em4", 1e-4),
    WmaxCase("2em4", 2e-4),
    WmaxCase("5em4", 5e-4),
    WmaxCase("1em3", 1e-3),
)


@dataclass(frozen=True)
class Surface:
    scheme: str
    optimizer: str
    source_config: str
    source_config_sha256: str
    rho_conv: float
    rho_dense: float
    validation_accuracy_percent: float
    range_status: str
    rates: tuple[float, ...]

    @property
    def key(self) -> str:
        return f"{self.scheme}_{self.optimizer.lower()}"


SURFACES = (
    Surface(
        "baseline",
        "SGD",
        "results/perfectdiode-conv12-directed-exploratory-lr-search-seed0-20260812-v1/shards/akib-task_6/surfaces/bounded_uniform__conv2__baseline__sgd/rho/cells/006_rc_3p7037037037em05_rd_4p11522633745em05/config.used.json",
        "a29d5a3210578cb70c77fcbcab671c6c54a2c393503a5c0b40e00a39f632c608",
        3.7037037037037037e-5,
        4.11522633744856e-5,
        82.84,
        "bracketed",
        (7.758687523956016e-8, 3.11171011067116e-6, 8.590900207684691e-7, 0.0, 0.0),
    ),
    Surface(
        "baseline",
        "Adam",
        "results/perfectdiode-conv12-directed-exploratory-lr-search-seed0-20260812-v1/shards/jean-zay-task_9/surfaces/bounded_uniform__conv2__baseline__adam/rho/cells/021_rc_0p243_rd_0p00333333333333/config.used.json",
        "c9409b6f852bcde474b3b7e5c2a50f57bfd250575a63916bdbedb0322271ad14",
        0.243,
        0.0033333333333333335,
        84.54,
        "open_upper_rho_conv",
        (1.471266669772638e-5, 1.4788493028149698e-5, 2.038514659443057e-7, 0.0, 0.0),
    ),
    Surface(
        "ours",
        "SGD",
        "results/perfectdiode-conv12-directed-exploratory-lr-search-seed0-20260812-v1/shards/akib-task_7/surfaces/bounded_uniform__conv2__ours__sgd/rho/cells/030_rc_0p000333333333333_rd_0p000123456790123/config.used.json",
        "8484c204cf959e57752e8751845dc16045ff7fca0cb938adf10253d024de9907",
        0.0003333333333333333,
        0.0001234567901234568,
        85.72,
        "bracketed",
        (1.788103962589367e-7, 7.195822043592904e-6, 6.705709229851339e-7, 0.0, 0.0),
    ),
    Surface(
        "ours",
        "Adam",
        "results/perfectdiode-conv12-directed-exploratory-lr-search-seed0-20260812-v1/shards/jean-zay-task_10/surfaces/bounded_uniform__conv2__ours__adam/rho/cells/010_rc_0p027_rd_0p01/config.used.json",
        "1eeab7ad2fcccca32c6d74efdaaf2f417f65beb650a23cf74606e93045621803",
        0.027,
        0.01,
        87.70,
        "bracketed",
        (1.6347391583547683e-6, 1.6428324925911035e-6, 6.094626207410665e-7, 0.0, 0.0),
    ),
    Surface(
        "legacy",
        "SGD",
        "results/perfectdiode-bounded-uniform-zero-bias-lr-search-conv123-seed0-20260810-v1/shards/akib-r1/surfaces/bounded_uniform__conv2__legacy__sgd/rho/cells/076_rc_0p003_rd_0p00333333333333/config.used.json",
        "10e720c71514c8bf69273b8af3f7a0258037285d5931476294c60ee3544b0cd0",
        0.003,
        0.003333333333333333,
        93.34,
        "bracketed",
        (1.252548618599936e-7, 5.458013672386962e-6, 1.571918643043664e-6, 0.0, 0.0),
    ),
    Surface(
        "legacy",
        "Adam",
        "results/perfectdiode-bounded-uniform-zero-bias-lr-search-conv123-seed0-20260810-v1/shards/akib-r1/surfaces/bounded_uniform__conv2__legacy__adam/rho/cells/087_rc_0p009_rd_0p01/config.used.json",
        "2d4c424c00418d05e19c5c466b67e5cab6483c6592f4689ead1d058283353c70",
        0.009,
        0.01,
        95.42,
        "bracketed",
        (5.449128384578525e-7, 5.475686125795819e-7, 6.092466384879761e-7, 0.0, 0.0),
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


def _load_source(surface: Surface) -> dict[str, Any]:
    path = REPO_ROOT / surface.source_config
    if _sha256(path) != surface.source_config_sha256:
        raise ValueError(f"Frozen source config digest mismatch: {path}")
    source = json.loads(path.read_text(encoding="utf-8"))
    model = source.get("model_base", {})
    if source.get("optimizer", {}).get("name") != surface.optimizer:
        raise ValueError(f"Source optimizer mismatch: {path}")
    if source.get("lr") != list(surface.rates):
        raise ValueError(f"Frozen LR vector does not match source config: {path}")
    source_order = source.get("parameter_order")
    if source_order not in (None, list(PARAMETER_ORDER)):
        raise ValueError(f"Unexpected source parameter order: {path}")
    if any(surface.rates[index] != 0.0 for index in (3, 4)):
        raise ValueError(f"Expected exact-zero bias rates: {path}")
    if model.get("weight_init_mode") != "bounded_uniform":
        raise ValueError(f"Source initializer is not bounded_uniform: {path}")
    if float(model.get("weight_min")) != 1e-5 or float(model.get("weight_max")) != 1e-4:
        raise ValueError(f"Unexpected source weight interval: {path}")
    if int(model.get("num_iterations_inference", -1)) != 6 or int(
        model.get("num_iterations_training", -1)
    ) != 6:
        raise ValueError(f"Expected accepted Conv2 T=K=6: {path}")
    if source.get("training_algorithm") != "BP" or int(source.get("seed", -1)) != 0:
        raise ValueError(f"Unexpected source algorithm or seed: {path}")
    for key in ("exponential_diode_param", "quadratic_diode_param", "hard_sigmoid_param"):
        if not isinstance(model.get(key), dict):
            raise ValueError(f"Missing explicit {key} in {path}")
    return source


def _arm_id(wmax: WmaxCase, surface: Surface) -> str:
    return f"conv2_wmax_{wmax.tag}_{surface.scheme}_{surface.optimizer.lower()}_fixed_uniform_init_seed0"


def _filename(index: int, wmax: WmaxCase, surface: Surface) -> str:
    return f"{index:02d}_wmax_{wmax.tag}_{surface.scheme}_{surface.optimizer.lower()}.json"


def _config(wmax: WmaxCase, surface: Surface) -> dict[str, Any]:
    config = deepcopy(_load_source(surface))
    arm_id = _arm_id(wmax, surface)
    checkpoint = RESULT_ASSET_ROOT / "final_model.pt"
    config["study_id"] = STUDY_ID
    config["arm_id"] = arm_id
    config["init_checkpoint_path"] = checkpoint.as_posix()
    config["parameter_order"] = list(PARAMETER_ORDER)
    config["learning_rates_by_parameter"] = dict(zip(PARAMETER_ORDER, surface.rates))
    config["lr"] = list(surface.rates)
    config["optimizer"]["learning_rate"] = list(surface.rates)
    config["lab"]["epochs"] = 10
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
        "changed_scientific_fields": ["model_base.weight_max"],
        "fixed_raw_learning_rates": True,
        "fixed_initial_parameter_states": True,
        "initializer": "bounded_uniform",
        "initializer_support": [1e-5, 1e-4],
        "initializer_checkpoint_path": checkpoint.as_posix(),
        "initializer_checkpoint_sha256": INITIALIZER_SHA256,
        "training_projection_interval": [1e-5, wmax.value],
        "rho_conv_provenance": surface.rho_conv,
        "rho_dense_provenance": surface.rho_dense,
        "source_config": surface.source_config,
        "source_config_sha256": surface.source_config_sha256,
        "source_final_validation_accuracy_percent": surface.validation_accuracy_percent,
        "source_range_status": surface.range_status,
        "scientific_gate_policy": (
            "user-waived repeated security and post-training T/K checks; "
            "reuse accepted Conv2 T=K=6"
        ),
    }
    return config


def prepare(output_root: Path) -> dict[str, Any]:
    output_root = Path(output_root).expanduser().resolve()
    if _sha256(INITIALIZER_SOURCE) != INITIALIZER_SHA256:
        raise ValueError(f"Frozen initializer digest mismatch: {INITIALIZER_SOURCE}")

    rows: list[dict[str, Any]] = []
    index = 0
    for wmax in WMAX_CASES:
        for surface in SURFACES:
            filename = _filename(index, wmax, surface)
            path = output_root / filename
            _write_json(path, _config(wmax, surface))
            rows.append(
                {
                    "index": index,
                    "architecture": "conv2",
                    "scheme": surface.scheme,
                    "optimizer": surface.optimizer,
                    "arm_id": _arm_id(wmax, surface),
                    "config": filename,
                    "config_sha256": _sha256(path),
                    "weight_min": 1e-5,
                    "weight_max": wmax.value,
                    "weight_max_tag": wmax.tag,
                    "learning_rates_by_parameter": dict(zip(PARAMETER_ORDER, surface.rates)),
                    "rho_conv_provenance": surface.rho_conv,
                    "rho_dense_provenance": surface.rho_dense,
                    "source_config": surface.source_config,
                    "source_config_sha256": surface.source_config_sha256,
                    "source_final_validation_accuracy_percent": surface.validation_accuracy_percent,
                    "source_range_status": surface.range_status,
                    "initializer_checkpoint_sha256": INITIALIZER_SHA256,
                }
            )
            index += 1

    ordered_set = hashlib.sha256(
        "".join(f"{row['config_sha256']}\n" for row in rows).encode("ascii")
    ).hexdigest()
    manifest = {
        "schema": "perfectdiode-conv2-fixed-uniform-init-wmax-sweep/v1",
        "study_id": STUDY_ID,
        "evidence_class": EVIDENCE_CLASS,
        "paper_facing": False,
        "canonical_lr_handoff": False,
        "scientific_role": "fixed-initial-state upper-weight-limit sensitivity",
        "selection_rule": (
            "highest observed final validation accuracy in the combined parent-plus-"
            "directed bounded-uniform Conv2 surface"
        ),
        "run_count": len(rows),
        "ordered_config_set_sha256": ordered_set,
        "initializer": {
            "mode": "bounded_uniform",
            "support": [1e-5, 1e-4],
            "checkpoint_source": str(INITIALIZER_SOURCE.relative_to(REPO_ROOT)),
            "checkpoint_path": (RESULT_ASSET_ROOT / "final_model.pt").as_posix(),
            "checkpoint_sha256": INITIALIZER_SHA256,
            "shared_exactly_across_all_runs": True,
        },
        "varied_field": "model_base.weight_max",
        "weight_max_values": [case.value for case in WMAX_CASES],
        "fixed_fields": {
            "architecture": "conv2",
            "dataset": "ordinary_mnist_train_validation",
            "epochs": 10,
            "seed": 0,
            "T": 6,
            "K": 6,
            "bias_learning_rates": [0.0, 0.0],
            "official_test_read": False,
        },
        "execution": {
            "target": "jean-zay",
            "account": "umg@v100",
            "partition": "gpu_p13",
            "qos": "qos_gpu-t3",
            "constraint": "v100-16g",
            "array": "0-23%24",
            "time_limit": "01:00:00",
        },
        "protocol_deviation": (
            "User-authorized exploratory omission of repeated gradient security and "
            "post-training T/K checks; accepted Conv2 T=K=6 is reused."
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
