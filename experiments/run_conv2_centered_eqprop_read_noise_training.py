#!/usr/bin/env python3
"""Materialize and run the Conv2 centered-float64 EqProp read-noise study."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments import exact_run


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STUDY = (
    ROOT
    / "configs/conv/perfectdiode_conv2_centered_float64_eqprop_read_noise_training_10ep_seed0_20260812_v1.json"
)
BETA_TIER_STUDY = (
    ROOT
    / "configs/conv/perfectdiode_conv2_centered_float64_eqprop_read_noise_beta_down_1to2decades_10ep_seed0_20260812_v1.json"
)
HIGH_NOISE_STUDY = (
    ROOT
    / "configs/conv/perfectdiode_conv2_centered_float64_eqprop_read_noise_one_decade_sigma_3em4_5em4_10ep_seed0_20260813_v1.json"
)
SGD_NOISE_STUDY = (
    ROOT
    / "configs/conv/perfectdiode_conv2_centered_float64_eqprop_read_noise_one_decade_sigma_5em4_sgd_10ep_seed0_20260816_v1.json"
)
SCHEMA = "perfectdiode-conv2-centered-float64-eqprop-read-noise-training-study/v1"
BETA_TIER_SCHEMA = (
    "perfectdiode-conv2-centered-float64-eqprop-read-noise-beta-tiers-training-study/v1"
)
HIGH_NOISE_SCHEMA = (
    "perfectdiode-conv2-centered-float64-eqprop-read-noise-focused-training-study/v1"
)
OPTIMIZER_COMPARISON_SCHEMA = (
    "perfectdiode-conv2-centered-float64-eqprop-read-noise-optimizer-comparison-study/v1"
)
SCHEMES = ("baseline", "ours", "legacy")
PARAMETER_ORDER = (
    "ConvWeight_0",
    "ConvWeight_1",
    "DenseWeight_0",
    "Bias_0",
    "Bias_1",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected {path} to contain a JSON object.")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _root_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def _sigma_label(value: float) -> str:
    if value == 0.0:
        return "0"
    exponent = int(round(math.log10(value)))
    if math.isclose(value, 10.0**exponent, rel_tol=1.0e-12, abs_tol=0.0):
        return f"1e{'m' if exponent < 0 else ''}{abs(exponent)}"
    return f"{value:.12g}".replace("-", "m").replace(".", "p")


def _beta_tiers(
    study: Mapping[str, Any],
) -> list[tuple[str | None, Mapping[str, Any]]]:
    """Return ordered beta tiers while preserving the original v1 contract."""
    if study.get("schema_version") == SCHEMA:
        return [(None, study["schemes"])]
    tiers = study.get("beta_tiers")
    if not isinstance(tiers, Mapping) or not tiers:
        raise ValueError("beta_tiers must be a non-empty ordered object.")
    return [(str(name), row["schemes"]) for name, row in tiers.items()]


def load_and_validate_study(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    study = _read_json(path)
    schema = study.get("schema_version")
    if schema not in (
        SCHEMA,
        BETA_TIER_SCHEMA,
        HIGH_NOISE_SCHEMA,
        OPTIMIZER_COMPARISON_SCHEMA,
    ):
        raise ValueError(f"Unexpected study schema in {path}.")
    contract = study.get("scientific_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("scientific_contract must be an object.")
    expected = {
        "architecture": "conv2",
        "dataset": "ordinary_mnist_deterministic_55000_train_5000_validation",
        "official_test_read": False,
        "epochs": 10,
        "T": 8,
        "K": 8,
        "runtime_dtype": "float64",
        "training_algorithm": "centered_frozen_current_eqprop",
        "input_read_noise": False,
    }
    if any(contract.get(key) != value for key, value in expected.items()):
        raise ValueError("The frozen Conv2 training contract changed.")
    optimizer = contract.get("optimizer")
    if optimizer not in {"Adam", "SGD"}:
        raise ValueError("The optimizer must be exactly Adam or SGD.")
    sigmas = [float(value) for value in contract.get("noise_sigmas", [])]
    expected_sigmas = (
        [3.0e-4, 5.0e-4]
        if schema == HIGH_NOISE_SCHEMA
        else [5.0e-4]
        if schema == OPTIMIZER_COMPARISON_SCHEMA
        else [0.0, 1e-6, 1e-5, 1e-4]
    )
    if sigmas != sorted(set(sigmas)) or sigmas != expected_sigmas:
        raise ValueError(f"Expected the exact noise grid {expected_sigmas!r}.")

    source = study["source"]
    handoff_path = _root_path(source["learning_rate_handoff"])
    if _sha256(handoff_path) != source["learning_rate_handoff_sha256"]:
        raise ValueError("The fixed LR handoff bytes changed.")
    handoff = _read_json(handoff_path)
    entries = {row["entry_id"]: row for row in handoff["entries"]}

    schemes = study.get("schemes")
    if not isinstance(schemes, Mapping) or tuple(schemes) != SCHEMES:
        raise ValueError(f"Expected schemes in order {SCHEMES}.")
    for scheme in SCHEMES:
        row = schemes[scheme]
        source_row = source["scheme_configs"][scheme]
        source_path = _root_path(source_row["path"])
        if _sha256(source_path) != source_row["sha256"]:
            raise ValueError(f"Source config bytes changed for {scheme}.")
        source_config = exact_run.load_exact_config(source_path)
        if source_config["optimizer"]["name"] != optimizer:
            raise ValueError(
                f"Expected the {optimizer} source config for {scheme}."
            )
        if list(source_config["parameter_order"]) != list(PARAMETER_ORDER):
            raise ValueError(f"Parameter order changed for {scheme}.")
        declared_rates = row["learning_rates_by_parameter"]
        if declared_rates != source_config["learning_rates_by_parameter"]:
            raise ValueError(f"Declared/source LR mismatch for {scheme}.")
        handoff_entry = entries[row["learning_rate_handoff_entry"]]
        if declared_rates != handoff_entry["learning_rates_by_parameter"]:
            raise ValueError(f"Declared/handoff LR mismatch for {scheme}.")
        factor = (float(row["voltage_amp"]) / float(row["current_amp"])) ** 2
        if not math.isclose(factor, float(row["amplification_factor"]), rel_tol=1e-15):
            raise ValueError(f"Amplification factor mismatch for {scheme}.")
        if not math.isclose(
            float(row["base_beta"]) * factor,
            float(row["injected_beta_B"]),
            rel_tol=1e-15,
        ):
            raise ValueError(f"Base/injected beta mismatch for {scheme}.")

    tiers = _beta_tiers(study)
    if schema in (
        BETA_TIER_SCHEMA,
        HIGH_NOISE_SCHEMA,
        OPTIMIZER_COMPARISON_SCHEMA,
    ):
        expected_names = (
            ("one_decade_lower",)
            if schema in (HIGH_NOISE_SCHEMA, OPTIMIZER_COMPARISON_SCHEMA)
            else ("one_decade_lower", "two_decades_lower")
        )
        if tuple(name for name, _ in tiers) != expected_names:
            raise ValueError(f"Expected beta tiers in order {expected_names}.")
        expected_scales = (
            (0.1,)
            if schema in (HIGH_NOISE_SCHEMA, OPTIMIZER_COMPARISON_SCHEMA)
            else (0.1, 0.01)
        )
        for (tier_name, tier_schemes), expected_scale in zip(
            tiers, expected_scales, strict=True
        ):
            if not isinstance(tier_schemes, Mapping) or tuple(tier_schemes) != SCHEMES:
                raise ValueError(f"Expected schemes in order {SCHEMES} for {tier_name}.")
            tier = study["beta_tiers"][tier_name]
            if not math.isclose(
                float(tier["relative_to_original_beta"]),
                expected_scale,
                rel_tol=1e-15,
            ):
                raise ValueError(f"Unexpected relative beta for {tier_name}.")
            for scheme in SCHEMES:
                row = tier_schemes[scheme]
                reference = schemes[scheme]
                factor = float(reference["amplification_factor"])
                if not math.isclose(
                    float(row["base_beta"]) * factor,
                    float(row["injected_beta_B"]),
                    rel_tol=1e-15,
                ):
                    raise ValueError(
                        f"Base/injected beta mismatch for {tier_name}/{scheme}."
                    )
                if not math.isclose(
                    float(row["injected_beta_B"]),
                    float(reference["injected_beta_B"]) * expected_scale,
                    rel_tol=1e-15,
                ):
                    raise ValueError(
                        f"Tier/reference injected beta mismatch for {tier_name}/{scheme}."
                    )

    expected_case_count = len(tiers) * len(SCHEMES) * len(sigmas)
    if int(study["execution"]["case_count"]) != expected_case_count:
        raise ValueError("Declared case count does not match the Cartesian grid.")
    study["_path"] = str(path)
    return study


def _resolved_case_config(
    study: Mapping[str, Any],
    scheme: str,
    sigma: float,
    beta_tier: str | None = None,
) -> dict[str, Any]:
    source_row = study["source"]["scheme_configs"][scheme]
    source_path = _root_path(source_row["path"])
    config = copy.deepcopy(_read_json(source_path))
    contract = study["scientific_contract"]
    scheme_row = study["schemes"][scheme]
    tier_rows = dict(_beta_tiers(study))
    if beta_tier not in tier_rows:
        raise ValueError(f"Unknown beta tier: {beta_tier!r}.")
    beta_row = tier_rows[beta_tier][scheme]
    tier_label = "" if beta_tier is None else f"_beta_{beta_tier}"
    optimizer_label = str(contract["optimizer"]).lower()
    arm_id = (
        f"conv2_{scheme}_{optimizer_label}{tier_label}_sigma_{_sigma_label(sigma)}"
    )

    dataset = config["datasets"]["mnist"]
    dataset["factory"] = "labs.datasets.MnistTrainValidationDataset"
    params = dataset["params"]
    params["affine_config"] = None
    params.pop("official_test_batch_size", None)
    params["batch_size"] = int(contract["batch_size"])
    params["validation_batch_size"] = int(contract["validation_batch_size"])
    params["split_seed"] = int(contract["model_seed"])
    params["shuffle_seed"] = int(contract["shuffle_seed"])

    config["schema_version"] = "mnist-conv-eqprop-read-noise-exact-run/v1"
    config["study_id"] = study["study_id"]
    config["arm_id"] = arm_id
    config["training_algorithm"] = "EP"
    config["runtime_dtype"] = "float64"
    config["beta"] = float(beta_row["base_beta"])
    config["lab"]["epochs"] = int(contract["epochs"])
    config["model_base"]["num_iterations_inference"] = int(contract["T"])
    config["model_base"]["num_iterations_training"] = int(contract["K"])
    config["evaluation"] = {
        "checkpoint_selection": "maximum_validation_accuracy",
        "epoch_split": "validation",
        "inclusion_rule": "Exploratory matched diagnostic; retain every finite run completing ten epochs and report failures rather than excluding by outcome.",
        "official_test": {"policy": "disabled"},
    }
    config["eqprop"] = {
        "variant": "centered",
        "nudging_mode": "current",
        "current_scale": "auto",
        "normalize_current_scale": True,
        "injected_beta_B": float(beta_row["injected_beta_B"]),
        "amplification_factor": float(scheme_row["amplification_factor"]),
        "endpoint_read_noise_std": float(sigma),
        "endpoint_read_noise_seed": int(contract["endpoint_read_noise_seed"]),
        "input_read_noise": False,
        "noise_applied_after_equilibrium_before_local_energy_gradient": True,
        "phase_noise_correlation": "independent",
        "layer_noise_correlation": "independent",
        "matched_standard_normal_draws_across_cases": True,
    }
    if beta_tier is not None:
        config["eqprop"]["beta_tier"] = beta_tier
    config["reporting"] = {
        "study_id": study["study_id"],
        "arm_id": arm_id,
        "evidence_class": study["evidence_class"],
        "paper_facing": False,
        "evidence_tier": study["evidence_tier"],
    }
    config["exploratory_source"] = {
        "study_config": str(Path(study["_path"]).relative_to(ROOT)),
        "source_config": source_row["path"],
        "source_config_sha256": source_row["sha256"],
        "learning_rate_handoff": study["source"]["learning_rate_handoff"],
        "learning_rate_handoff_sha256": study["source"][
            "learning_rate_handoff_sha256"
        ],
        "beta_status": beta_row.get("beta_status", scheme_row["beta_status"]),
        "dataset_change": "deterministic_medium_affine_source_template_to_ordinary_mnist_selection_split",
    }
    return config


def materialize_configs(study: Mapping[str, Any], directory: Path) -> list[Path]:
    directory = directory.expanduser().resolve()
    paths: list[Path] = []
    index = 0
    for beta_tier, _ in _beta_tiers(study):
        for scheme in SCHEMES:
            for sigma in study["scientific_contract"]["noise_sigmas"]:
                config = _resolved_case_config(
                    study, scheme, float(sigma), beta_tier=beta_tier
                )
                path = directory / f"{index:02d}_{config['arm_id']}.json"
                _write_json(path, config)
                exact_run.load_exact_config(path)
                paths.append(path)
                index += 1
    return paths


def plan(study: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    for beta_tier, tier_schemes in _beta_tiers(study):
        for scheme in SCHEMES:
            scheme_row = study["schemes"][scheme]
            beta_row = tier_schemes[scheme]
            for sigma in study["scientific_contract"]["noise_sigmas"]:
                rows.append(
                    {
                        "index": len(rows),
                        "beta_tier": beta_tier,
                        "scheme": scheme,
                        "sigma": float(sigma),
                        "injected_beta_B": beta_row["injected_beta_B"],
                        "base_beta": beta_row["base_beta"],
                        "learning_rates_by_parameter": scheme_row[
                            "learning_rates_by_parameter"
                        ],
                    }
                )
    return {
        "study_id": study["study_id"],
        "target": study["execution"]["target"],
        "epochs_per_case": study["scientific_contract"]["epochs"],
        "T": study["scientific_contract"]["T"],
        "K": study["scientific_contract"]["K"],
        "runtime_dtype": study["scientific_contract"]["runtime_dtype"],
        "optimizer": study["scientific_contract"]["optimizer"],
        "case_count": len(rows),
        "cases": rows,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("plan")
    materialize = subparsers.add_parser("materialize")
    materialize.add_argument("--directory", type=Path, required=True)
    for name in ("smoke", "run"):
        command = subparsers.add_parser(name)
        command.add_argument("--output-root", type=Path, required=True)
        command.add_argument("--dataset-root", type=Path, required=True)
        command.add_argument("--device", default="cuda")
        command.add_argument("--target", default="local")
        command.add_argument(
            "--indices",
            type=int,
            nargs="*",
            default=[0, 3, 7, 11] if name == "smoke" else None,
        )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    study = load_and_validate_study(args.study)
    if args.command == "plan":
        print(json.dumps(plan(study), indent=2, sort_keys=True, allow_nan=False))
        return 0
    if args.command == "materialize":
        paths = materialize_configs(study, args.directory)
        print(json.dumps([str(path) for path in paths], indent=2))
        return 0

    output_root = args.output_root.expanduser().resolve()
    configs = materialize_configs(study, output_root / "resolved_configs")
    if args.indices is not None:
        requested = list(args.indices)
        if len(requested) != len(set(requested)) or any(
            index < 0 or index >= len(configs) for index in requested
        ):
            raise ValueError(f"Invalid smoke indices: {requested!r}.")
        configs = [configs[index] for index in requested]
    result = exact_run.run(
        configs,
        output_root=output_root,
        device=args.device,
        smoke=args.command == "smoke",
        study_id=study["study_id"],
        evidence_class=study["evidence_class"],
        target=args.target,
        dataset_root=args.dataset_root,
        skip_terminal_official_test=True,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
