#!/usr/bin/env python3
"""Prepare the matched Conv1 bounded-uniform absolute-scale diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
STUDY_ID = "perfectdiode-conv1-bounded-uniform-scale1e4-adam-3ep-seed0-20260813-v1"
EVIDENCE_CLASS = "ordinary_mnist_absolute_weight_scale_exploratory"
DEFAULT_CONFIG_ROOT = REPO_ROOT / "configs" / "conv" / STUDY_ID
RESULT_ASSET = (
    Path("results")
    / STUDY_ID
    / "assets"
    / "bounded_uniform_scaled_1e4"
    / "conv1"
    / "initial_model.pt"
)
SOURCE_ASSET = Path(
    "results/perfectdiode-bounded-uniform-zero-bias-lr-search-conv123-seed0-20260810-v1/"
    "shards/main-r1/assets/bounded_uniform/conv1/final_model.pt"
)
SOURCE_ASSET_SHA256 = "235d7ae56a277e097e840a4eb11cf16e325ed2d3eb80c9055f7903493cb1f16e"
SCALE_FACTOR = 1.0e4
SOURCE_WEIGHT_MIN = 1.0e-5
SOURCE_WEIGHT_MAX = 1.0e-4
SCALED_WEIGHT_MIN = 1.0e-1
SCALED_WEIGHT_MAX = 1.0


@dataclass(frozen=True)
class Case:
    scheme: str
    source_config: str
    source_config_sha256: str
    rho_conv: float
    rho_dense: float
    source_final_validation_accuracy_percent: float
    rates: tuple[float, float, float]

    @property
    def filename(self) -> str:
        return f"conv1_{self.scheme}_adam.json"

    @property
    def arm_id(self) -> str:
        return f"conv1_{self.scheme}_adam_bounded_uniform_scale1e4_seed0"


CASES = (
    Case(
        scheme="baseline",
        source_config=(
            "results/perfectdiode-bounded-uniform-zero-bias-lr-search-conv123-seed0-"
            "20260810-v1/shards/main-r1/surfaces/bounded_uniform__conv1__baseline__adam/"
            "rho/cells/078_rc_0p003_rd_0p03/config.used.json"
        ),
        source_config_sha256=(
            "b26f6bc99e6abb6fe521f4b22546d6ec87fd86ffb5a3f6a878fe86c3e3a925d1"
        ),
        rho_conv=0.003,
        rho_dense=0.03,
        source_final_validation_accuracy_percent=88.84,
        rates=(1.8163761812439207e-7, 1.8257546674661046e-6, 0.0),
    ),
    Case(
        scheme="ours",
        source_config=(
            "results/perfectdiode-bounded-uniform-zero-bias-lr-search-conv123-seed0-"
            "20260810-v1/shards/main-r1/surfaces/bounded_uniform__conv1__ours__adam/"
            "rho/cells/067_rc_0p001_rd_0p01/config.used.json"
        ),
        source_config_sha256=(
            "2ab0708e0c700dde9af600c411e7d3bd540790f0674dfae618d61d4ea01e7108"
        ),
        rho_conv=0.001,
        rho_dense=0.01,
        source_final_validation_accuracy_percent=91.60,
        rates=(6.054587142218743e-8, 6.08565526037015e-7, 0.0),
    ),
    Case(
        scheme="legacy",
        source_config=(
            "results/perfectdiode-bounded-uniform-zero-bias-lr-search-conv123-seed0-"
            "20260810-v1/shards/main-r1/surfaces/bounded_uniform__conv1__legacy__adam/"
            "rho/cells/087_rc_0p009_rd_0p01/config.used.json"
        ),
        source_config_sha256=(
            "c18431d9b92317ece9c3f510bd63ac8a9c464215343aff16ba92d232821bf10b"
        ),
        rho_conv=0.009,
        rho_dense=0.01,
        source_final_validation_accuracy_percent=95.42,
        rates=(5.449128249811996e-7, 6.085548426894188e-7, 0.0),
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


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _scientific_parameter_kind(descriptor: dict[str, Any]) -> str:
    parameter_type = str(descriptor.get("type", ""))
    if parameter_type.endswith((".ConvWeight", ".DenseWeight")):
        return "weight"
    if parameter_type.endswith(".Bias"):
        return "bias"
    raise ValueError(f"Unexpected checkpoint parameter descriptor: {descriptor!r}")


def _scaled_checkpoint() -> dict[str, Any]:
    source_path = REPO_ROOT / SOURCE_ASSET
    if _sha256(source_path) != SOURCE_ASSET_SHA256:
        raise ValueError(f"Frozen source checkpoint digest mismatch: {source_path}")
    checkpoint = torch.load(source_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unexpected checkpoint payload: {source_path}")
    schema = checkpoint.get("schema")
    states = checkpoint.get("states")
    if not isinstance(schema, list) or not isinstance(states, list) or len(schema) != len(states):
        raise ValueError(f"Unexpected checkpoint schema/state layout: {source_path}")

    scaled_states = []
    weight_count = 0
    bias_count = 0
    source_observed_min = float("inf")
    source_observed_max = float("-inf")
    scaled_observed_min = float("inf")
    scaled_observed_max = float("-inf")
    for descriptor, state in zip(schema, states):
        if not isinstance(descriptor, dict) or not isinstance(state, torch.Tensor):
            raise ValueError(f"Unexpected checkpoint entry in {source_path}")
        kind = _scientific_parameter_kind(descriptor)
        if kind == "weight":
            source_min = float(state.min().item())
            source_max = float(state.max().item())
            if source_min < SOURCE_WEIGHT_MIN or source_max >= SOURCE_WEIGHT_MAX:
                raise ValueError(
                    f"Source weight outside [{SOURCE_WEIGHT_MIN},{SOURCE_WEIGHT_MAX}): "
                    f"{descriptor.get('name')!r} has [{source_min},{source_max}]"
                )
            scaled = state * SCALE_FACTOR
            scaled_min = float(scaled.min().item())
            scaled_max = float(scaled.max().item())
            if scaled_min < SCALED_WEIGHT_MIN or scaled_max >= SCALED_WEIGHT_MAX:
                raise ValueError(
                    f"Scaled weight outside [{SCALED_WEIGHT_MIN},{SCALED_WEIGHT_MAX}): "
                    f"{descriptor.get('name')!r} has [{scaled_min},{scaled_max}]"
                )
            scaled_states.append(scaled)
            weight_count += int(state.numel())
            source_observed_min = min(source_observed_min, source_min)
            source_observed_max = max(source_observed_max, source_max)
            scaled_observed_min = min(scaled_observed_min, scaled_min)
            scaled_observed_max = max(scaled_observed_max, scaled_max)
        else:
            if torch.count_nonzero(state).item() != 0:
                raise ValueError(f"Expected exact-zero source bias: {descriptor.get('name')!r}")
            scaled_states.append(state.clone())
            bias_count += int(state.numel())

    scaled_checkpoint = dict(checkpoint)
    scaled_checkpoint["states"] = scaled_states
    return {
        "checkpoint": scaled_checkpoint,
        "audit": {
            "source_checkpoint": SOURCE_ASSET.as_posix(),
            "source_checkpoint_sha256": SOURCE_ASSET_SHA256,
            "scale_factor": SCALE_FACTOR,
            "weight_count": weight_count,
            "bias_count": bias_count,
            "source_observed_weight_min": source_observed_min,
            "source_observed_weight_max": source_observed_max,
            "scaled_observed_weight_min": scaled_observed_min,
            "scaled_observed_weight_max": scaled_observed_max,
            "biases_remain_exact_zero": True,
        },
    }


def _materialize_scaled_checkpoint() -> tuple[str, dict[str, Any]]:
    payload = _scaled_checkpoint()
    output_path = REPO_ROOT / RESULT_ASSET
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload["checkpoint"], output_path)
    digest = _sha256(output_path)

    reloaded = torch.load(output_path, map_location="cpu", weights_only=True)
    expected_states = payload["checkpoint"]["states"]
    observed_states = reloaded.get("states") if isinstance(reloaded, dict) else None
    if not isinstance(observed_states, list) or len(observed_states) != len(expected_states):
        raise ValueError(f"Scaled checkpoint did not round-trip: {output_path}")
    if any(not torch.equal(expected, observed) for expected, observed in zip(expected_states, observed_states)):
        raise ValueError(f"Scaled checkpoint changed during serialization: {output_path}")
    return digest, payload["audit"]


def _config(case: Case, scaled_checkpoint_sha256: str) -> dict[str, Any]:
    source_path = REPO_ROOT / case.source_config
    if _sha256(source_path) != case.source_config_sha256:
        raise ValueError(f"Frozen source config digest mismatch: {source_path}")
    source = _load_json(source_path)
    if source.get("model_base", {}).get("weight_init_mode") != "bounded_uniform":
        raise ValueError(f"Source is not bounded_uniform: {source_path}")
    if source.get("model_base", {}).get("weight_min") != SOURCE_WEIGHT_MIN:
        raise ValueError(f"Unexpected source lower bound: {source_path}")
    if source.get("model_base", {}).get("weight_max") != SOURCE_WEIGHT_MAX:
        raise ValueError(f"Unexpected source upper bound: {source_path}")
    if source.get("optimizer", {}).get("name") != "Adam":
        raise ValueError(f"Source optimizer is not Adam: {source_path}")
    if source.get("lr") != list(case.rates):
        raise ValueError(f"Frozen LR vector does not match source config: {source_path}")

    parameter_order = ("ConvWeight_0", "DenseWeight_0", "Bias_0")
    scaled_rates = tuple(float(rate) * SCALE_FACTOR for rate in case.rates)
    config = deepcopy(source)
    config["study_id"] = STUDY_ID
    config["arm_id"] = case.arm_id
    config["init_checkpoint_path"] = RESULT_ASSET.as_posix()
    config["parameter_order"] = list(parameter_order)
    config["learning_rates_by_parameter"] = dict(zip(parameter_order, scaled_rates))
    config["lr"] = list(scaled_rates)
    config["optimizer"]["learning_rate"] = list(scaled_rates)
    config["lab"]["epochs"] = 3
    config["model_base"]["weight_init_mode"] = "bounded_uniform"
    config["model_base"]["weight_min"] = SCALED_WEIGHT_MIN
    config["model_base"]["weight_max"] = SCALED_WEIGHT_MAX
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
    config["absolute_scale_transfer"] = {
        "canonical_lr_handoff": False,
        "protocol_override": "user-authorized exploratory absolute-weight-scale repeat",
        "scale_factor": SCALE_FACTOR,
        "changed_scientific_fields": [
            "model_base.weight_min",
            "model_base.weight_max",
            "init_checkpoint_path",
            "lr",
            "optimizer.learning_rate",
        ],
        "quantile_mapping": "exact source checkpoint tensors multiplied by 1e4",
        "source_weight_interval": [SOURCE_WEIGHT_MIN, SOURCE_WEIGHT_MAX],
        "scaled_weight_interval": [SCALED_WEIGHT_MIN, SCALED_WEIGHT_MAX],
        "source_initializer_checkpoint": SOURCE_ASSET.as_posix(),
        "source_initializer_checkpoint_sha256": SOURCE_ASSET_SHA256,
        "scaled_initializer_checkpoint": RESULT_ASSET.as_posix(),
        "scaled_initializer_checkpoint_sha256": scaled_checkpoint_sha256,
        "source_learning_rates_by_parameter": dict(zip(parameter_order, case.rates)),
        "scaled_learning_rates_by_parameter": dict(zip(parameter_order, scaled_rates)),
        "rho_conv_provenance": case.rho_conv,
        "rho_dense_provenance": case.rho_dense,
        "source_config": case.source_config,
        "source_config_sha256": case.source_config_sha256,
        "source_final_validation_accuracy_percent": (
            case.source_final_validation_accuracy_percent
        ),
    }
    return config


def prepare(config_root: Path) -> dict[str, Any]:
    config_root = Path(config_root).expanduser().resolve()
    scaled_checkpoint_sha256, checkpoint_audit = _materialize_scaled_checkpoint()
    rows = []
    for index, case in enumerate(CASES):
        config_path = config_root / case.filename
        _write_json(config_path, _config(case, scaled_checkpoint_sha256))
        rows.append(
            {
                "index": index,
                "architecture": "conv1",
                "scheme": case.scheme,
                "optimizer": "Adam",
                "arm_id": case.arm_id,
                "config": case.filename,
                "config_sha256": _sha256(config_path),
                "source_config": case.source_config,
                "source_config_sha256": case.source_config_sha256,
                "source_final_validation_accuracy_percent": (
                    case.source_final_validation_accuracy_percent
                ),
                "rho_conv_provenance": case.rho_conv,
                "rho_dense_provenance": case.rho_dense,
                "scaled_initializer_checkpoint_sha256": scaled_checkpoint_sha256,
            }
        )
    ordered_config_set_sha256 = hashlib.sha256(
        "".join(f"{row['config_sha256']}\n" for row in rows).encode("ascii")
    ).hexdigest()
    manifest = {
        "schema": "perfectdiode-conv1-bounded-uniform-absolute-scale-study/v1",
        "study_id": STUDY_ID,
        "evidence_class": EVIDENCE_CLASS,
        "paper_facing": False,
        "canonical_lr_handoff": False,
        "protocol_override": "user-authorized exploratory absolute-weight-scale repeat",
        "scientific_question": (
            "Does increasing the Conv1 bounded-uniform absolute conductance scale by "
            "1e4, at fixed dynamic range and proportionally scaled raw Adam LRs, recover "
            "performance lost under the [1e-5,1e-4] contract?"
        ),
        "run_count": len(rows),
        "ordered_config_set_sha256": ordered_config_set_sha256,
        "initializer": {
            "mode": "bounded_uniform",
            "source_interval": [SOURCE_WEIGHT_MIN, SOURCE_WEIGHT_MAX],
            "scaled_interval": [SCALED_WEIGHT_MIN, SCALED_WEIGHT_MAX],
            "scale_factor": SCALE_FACTOR,
            "quantile_mapping": "exact source checkpoint tensors multiplied by 1e4",
            "source_checkpoint_sha256": SOURCE_ASSET_SHA256,
            "scaled_checkpoint_sha256": scaled_checkpoint_sha256,
            "checkpoint_audit": checkpoint_audit,
        },
        "execution": {
            "dataset": "ordinary_mnist_train_validation",
            "architecture": "conv1",
            "schemes": [case.scheme for case in CASES],
            "optimizer": "Adam",
            "epochs": 3,
            "seed": 0,
            "num_iterations_inference": 4,
            "num_iterations_training": 4,
            "official_test_read": False,
            "target": "main",
            "launcher": "local_tmux",
        },
        "runs": rows,
    }
    _write_json(config_root / "study_manifest.json", manifest)
    result_manifest = REPO_ROOT / "results" / STUDY_ID / "study_manifest.json"
    _write_json(result_manifest, manifest)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-root", type=Path, default=DEFAULT_CONFIG_ROOT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    print(json.dumps(prepare(args.config_root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
