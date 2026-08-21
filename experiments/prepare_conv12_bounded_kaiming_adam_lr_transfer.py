#!/usr/bin/env python3
"""Freeze the six exact configs for the bounded-Kaiming Adam LR transfer."""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
STUDY_ID = "perfectdiode-conv12-bounded-kaiming-adam-lr-transfer-3ep-seed0-20260813-v1"
EVIDENCE_CLASS = "ordinary_mnist_initializer_transfer_exploratory"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "configs" / "conv" / STUDY_ID
RESULT_ASSET_ROOT = Path("results") / STUDY_ID / "assets" / "bounded_kaiming_uniform"
INITIALIZER_SHA256 = {
    "conv1": "d603b26c86a622263572705dd693987e5af24da78e38e629256097d603c2a646",
    "conv2": "8ecf6240d572619308203301f7c4b462f7224fa7b9913778588fdadd2594d21b",
}


@dataclass(frozen=True)
class Case:
    architecture: str
    scheme: str
    source_config: str
    source_config_sha256: str
    rho_conv: float
    rho_dense: float
    validation_accuracy_percent: float
    range_status: str
    rates: tuple[float, ...]
    parameter_order: tuple[str, ...]

    @property
    def arm_id(self) -> str:
        return f"{self.architecture}_{self.scheme}_adam_bounded_kaiming_raw_lr_transfer_seed0"

    @property
    def filename(self) -> str:
        return f"{self.architecture}_{self.scheme}_adam.json"


CASES = (
    Case(
        "conv1",
        "baseline",
        "results/perfectdiode-bounded-uniform-zero-bias-lr-search-conv123-seed0-20260810-v1/shards/main-r1/surfaces/bounded_uniform__conv1__baseline__adam/rho/cells/078_rc_0p003_rd_0p03/config.used.json",
        "b26f6bc99e6abb6fe521f4b22546d6ec87fd86ffb5a3f6a878fe86c3e3a925d1",
        0.003,
        0.03,
        88.84,
        "bracketed",
        (1.8163761812439207e-7, 1.8257546674661046e-6, 0.0),
        ("ConvWeight_0", "DenseWeight_0", "Bias_0"),
    ),
    Case(
        "conv1",
        "ours",
        "results/perfectdiode-bounded-uniform-zero-bias-lr-search-conv123-seed0-20260810-v1/shards/main-r1/surfaces/bounded_uniform__conv1__ours__adam/rho/cells/067_rc_0p001_rd_0p01/config.used.json",
        "2ab0708e0c700dde9af600c411e7d3bd540790f0674dfae618d61d4ea01e7108",
        0.001,
        0.01,
        91.6,
        "bracketed",
        (6.054587142218743e-8, 6.08565526037015e-7, 0.0),
        ("ConvWeight_0", "DenseWeight_0", "Bias_0"),
    ),
    Case(
        "conv1",
        "legacy",
        "results/perfectdiode-bounded-uniform-zero-bias-lr-search-conv123-seed0-20260810-v1/shards/main-r1/surfaces/bounded_uniform__conv1__legacy__adam/rho/cells/087_rc_0p009_rd_0p01/config.used.json",
        "c18431d9b92317ece9c3f510bd63ac8a9c464215343aff16ba92d232821bf10b",
        0.009,
        0.01,
        95.42,
        "bracketed",
        (5.449128249811996e-7, 6.085548426894188e-7, 0.0),
        ("ConvWeight_0", "DenseWeight_0", "Bias_0"),
    ),
    Case(
        "conv2",
        "baseline",
        "results/perfectdiode-conv12-directed-exploratory-lr-search-seed0-20260812-v1/shards/jean-zay-task_9/surfaces/bounded_uniform__conv2__baseline__adam/rho/cells/021_rc_0p243_rd_0p00333333333333/config.used.json",
        "c9409b6f852bcde474b3b7e5c2a50f57bfd250575a63916bdbedb0322271ad14",
        0.243,
        0.0033333333333333335,
        84.54,
        "open_upper_rho_conv",
        (1.471266669772638e-5, 1.4788493028149698e-5, 2.038514659443057e-7, 0.0, 0.0),
        ("ConvWeight_0", "ConvWeight_1", "DenseWeight_0", "Bias_0", "Bias_1"),
    ),
    Case(
        "conv2",
        "ours",
        "results/perfectdiode-conv12-directed-exploratory-lr-search-seed0-20260812-v1/shards/jean-zay-task_10/surfaces/bounded_uniform__conv2__ours__adam/rho/cells/010_rc_0p027_rd_0p01/config.used.json",
        "1eeab7ad2fcccca32c6d74efdaaf2f417f65beb650a23cf74606e93045621803",
        0.027,
        0.01,
        87.7,
        "bracketed",
        (1.6347391583547683e-6, 1.6428324925911035e-6, 6.094626207410665e-7, 0.0, 0.0),
        ("ConvWeight_0", "ConvWeight_1", "DenseWeight_0", "Bias_0", "Bias_1"),
    ),
    Case(
        "conv2",
        "legacy",
        "results/perfectdiode-bounded-uniform-zero-bias-lr-search-conv123-seed0-20260810-v1/shards/akib-r1/surfaces/bounded_uniform__conv2__legacy__adam/rho/cells/087_rc_0p009_rd_0p01/config.used.json",
        "2d4c424c00418d05e19c5c466b67e5cab6483c6592f4689ead1d058283353c70",
        0.009,
        0.01,
        95.42,
        "bracketed",
        (5.449128384578525e-7, 5.475686125795819e-7, 6.092466384879761e-7, 0.0, 0.0),
        ("ConvWeight_0", "ConvWeight_1", "DenseWeight_0", "Bias_0", "Bias_1"),
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


def _load_source(case: Case) -> dict[str, Any]:
    path = REPO_ROOT / case.source_config
    if _sha256(path) != case.source_config_sha256:
        raise ValueError(f"Frozen source config digest mismatch: {path}")
    source = json.loads(path.read_text(encoding="utf-8"))
    if source.get("optimizer", {}).get("name") != "Adam":
        raise ValueError(f"Source optimizer is not Adam: {path}")
    if source.get("model_base", {}).get("weight_init_mode") != "bounded_uniform":
        raise ValueError(f"Source initializer is not bounded_uniform: {path}")
    if source.get("lr") != list(case.rates):
        raise ValueError(f"Frozen LR vector does not match source config: {path}")
    return source


def _config(case: Case) -> dict[str, Any]:
    config = deepcopy(_load_source(case))
    checkpoint = RESULT_ASSET_ROOT / case.architecture / "final_model.pt"
    config["study_id"] = STUDY_ID
    config["arm_id"] = case.arm_id
    config["init_checkpoint_path"] = checkpoint.as_posix()
    config["parameter_order"] = list(case.parameter_order)
    config["learning_rates_by_parameter"] = dict(zip(case.parameter_order, case.rates))
    config["lr"] = list(case.rates)
    config["optimizer"]["learning_rate"] = list(case.rates)
    config["lab"]["epochs"] = 3
    config["model_base"]["weight_init_mode"] = "bounded_kaiming_uniform"
    config["evaluation"] = {
        "checkpoint_selection": "maximum_validation_accuracy",
        "official_test": {"policy": "disabled"},
    }
    config["reporting"] = {
        "study_id": STUDY_ID,
        "arm_id": case.arm_id,
        "evidence_class": EVIDENCE_CLASS,
        "paper_facing": False,
    }
    config["initializer_transfer"] = {
        "canonical_lr_handoff": False,
        "changed_scientific_fields": [
            "model_base.weight_init_mode",
            "init_checkpoint_path",
        ],
        "initializer": "bounded_kaiming_uniform",
        "initializer_gain": 1.0,
        "initializer_midpoint": 5.5e-5,
        "initializer_checkpoint_path": checkpoint.as_posix(),
        "initializer_checkpoint_sha256": INITIALIZER_SHA256[case.architecture],
        "protocol_override": "user-authorized exploratory cross-initializer raw-LR transfer",
        "rho_conv_provenance": case.rho_conv,
        "rho_dense_provenance": case.rho_dense,
        "source_config": case.source_config,
        "source_config_sha256": case.source_config_sha256,
        "source_final_validation_accuracy_percent": case.validation_accuracy_percent,
        "source_range_status": case.range_status,
    }
    return config


def prepare(output_root: Path) -> dict[str, Any]:
    output_root = Path(output_root).expanduser().resolve()
    rows = []
    for index, case in enumerate(CASES):
        path = output_root / case.filename
        _write_json(path, _config(case))
        rows.append(
            {
                "index": index,
                "architecture": case.architecture,
                "scheme": case.scheme,
                "optimizer": "Adam",
                "arm_id": case.arm_id,
                "config": case.filename,
                "config_sha256": _sha256(path),
                "rho_conv_provenance": case.rho_conv,
                "rho_dense_provenance": case.rho_dense,
                "source_final_validation_accuracy_percent": case.validation_accuracy_percent,
                "source_range_status": case.range_status,
                "initializer_checkpoint_sha256": INITIALIZER_SHA256[case.architecture],
            }
        )
    ordered_set = hashlib.sha256(
        "".join(f"{row['config_sha256']}\n" for row in rows).encode("ascii")
    ).hexdigest()
    manifest = {
        "schema": "perfectdiode-conv12-bounded-kaiming-adam-lr-transfer-study/v1",
        "study_id": STUDY_ID,
        "evidence_class": EVIDENCE_CLASS,
        "paper_facing": False,
        "canonical_lr_handoff": False,
        "protocol_override": "user-authorized exploratory cross-initializer raw-LR transfer",
        "selection_rule": "highest observed final validation accuracy in the combined parent-plus-directed bounded-uniform Adam surface",
        "run_count": len(rows),
        "ordered_config_set_sha256": ordered_set,
        "initializer": {
            "mode": "bounded_kaiming_uniform",
            "gain": 1.0,
            "weight_min": 1e-5,
            "weight_max": 1e-4,
            "midpoint": 5.5e-5,
            "checkpoint_sha256_by_architecture": INITIALIZER_SHA256,
        },
        "execution": {
            "dataset": "ordinary_mnist_train_validation",
            "epochs": 3,
            "optimizer": "Adam",
            "seed": 0,
            "official_test_read": False,
            "target": "jean-zay",
            "array": "0-5%6",
        },
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
