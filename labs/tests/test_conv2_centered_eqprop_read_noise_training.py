from __future__ import annotations

import json

import pytest

from experiments.run_conv2_centered_eqprop_read_noise_training import (
    BETA_TIER_STUDY,
    DEFAULT_STUDY,
    HIGH_NOISE_STUDY,
    SGD_NOISE_STUDY,
    load_and_validate_study,
    materialize_configs,
    plan,
)


def test_plan_is_exact_matched_twelve_case_grid():
    study = load_and_validate_study(DEFAULT_STUDY)
    resolved = plan(study)

    assert resolved["case_count"] == 12
    assert (resolved["T"], resolved["K"]) == (8, 8)
    assert resolved["runtime_dtype"] == "float64"
    assert [(row["scheme"], row["sigma"]) for row in resolved["cases"]] == [
        (scheme, sigma)
        for scheme in ("baseline", "ours", "legacy")
        for sigma in (0.0, 1.0e-6, 1.0e-5, 1.0e-4)
    ]


def test_materialized_configs_freeze_algorithm_beta_noise_and_dataset(tmp_path):
    study = load_and_validate_study(DEFAULT_STUDY)
    paths = materialize_configs(study, tmp_path / "configs")
    configs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]

    assert len(configs) == 12
    for config in configs:
        assert config["training_algorithm"] == "EP"
        assert config["runtime_dtype"] == "float64"
        assert config["lab"]["epochs"] == 10
        assert config["model_base"]["num_iterations_inference"] == 8
        assert config["model_base"]["num_iterations_training"] == 8
        assert config["datasets"]["mnist"]["factory"] == (
            "labs.datasets.MnistTrainValidationDataset"
        )
        assert config["datasets"]["mnist"]["params"]["affine_config"] is None
        assert config["evaluation"]["official_test"]["policy"] == "disabled"
        assert config["eqprop"]["variant"] == "centered"
        assert config["eqprop"]["nudging_mode"] == "current"
        assert config["eqprop"]["normalize_current_scale"] is True
        assert config["eqprop"]["input_read_noise"] is False

    by_scheme = {config["arm_id"].split("_")[1]: config for config in configs[::4]}
    assert by_scheme["baseline"]["beta"] == pytest.approx(1000.0)
    assert by_scheme["ours"]["beta"] == pytest.approx(6.25)
    assert by_scheme["legacy"]["beta"] == pytest.approx(0.001171875)
    assert by_scheme["baseline"]["eqprop"]["injected_beta_B"] == 1000.0
    assert by_scheme["ours"]["eqprop"]["injected_beta_B"] == 100.0
    assert by_scheme["legacy"]["eqprop"]["injected_beta_B"] == 0.3


def test_lower_beta_plan_is_exact_matched_twenty_four_case_grid():
    study = load_and_validate_study(BETA_TIER_STUDY)
    resolved = plan(study)

    assert resolved["case_count"] == 24
    assert [(row["beta_tier"], row["scheme"], row["sigma"]) for row in resolved["cases"]] == [
        (tier, scheme, sigma)
        for tier in ("one_decade_lower", "two_decades_lower")
        for scheme in ("baseline", "ours", "legacy")
        for sigma in (0.0, 1.0e-6, 1.0e-5, 1.0e-4)
    ]
    injected = {
        (row["beta_tier"], row["scheme"]): row["injected_beta_B"]
        for row in resolved["cases"]
    }
    assert injected[("one_decade_lower", "baseline")] == 100.0
    assert injected[("one_decade_lower", "ours")] == 10.0
    assert injected[("one_decade_lower", "legacy")] == 0.03
    assert injected[("two_decades_lower", "baseline")] == 10.0
    assert injected[("two_decades_lower", "ours")] == 1.0
    assert injected[("two_decades_lower", "legacy")] == 0.003


def test_lower_beta_materialization_freezes_base_and_injected_beta(tmp_path):
    study = load_and_validate_study(BETA_TIER_STUDY)
    paths = materialize_configs(study, tmp_path / "configs")
    configs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]

    assert len(configs) == 24
    expected = {
        ("one_decade_lower", "baseline"): (100.0, 100.0),
        ("one_decade_lower", "ours"): (0.625, 10.0),
        ("one_decade_lower", "legacy"): (0.0001171875, 0.03),
        ("two_decades_lower", "baseline"): (10.0, 10.0),
        ("two_decades_lower", "ours"): (0.0625, 1.0),
        ("two_decades_lower", "legacy"): (0.00001171875, 0.003),
    }
    for config in configs[::4]:
        key = (config["eqprop"]["beta_tier"], config["arm_id"].split("_")[1])
        base_beta, injected_beta = expected[key]
        assert config["beta"] == pytest.approx(base_beta)
        assert config["eqprop"]["injected_beta_B"] == pytest.approx(injected_beta)


def test_high_noise_extension_is_exact_six_case_parallel_surface():
    study = load_and_validate_study(HIGH_NOISE_STUDY)
    resolved = plan(study)

    assert resolved["case_count"] == 6
    assert [(row["beta_tier"], row["scheme"], row["sigma"]) for row in resolved["cases"]] == [
        ("one_decade_lower", scheme, sigma)
        for scheme in ("baseline", "ours", "legacy")
        for sigma in (3.0e-4, 5.0e-4)
    ]
    injected = {
        row["scheme"]: row["injected_beta_B"] for row in resolved["cases"]
    }
    assert injected == {"baseline": 100.0, "ours": 10.0, "legacy": 0.03}


def test_high_noise_materialization_preserves_matched_noise_seed_and_one_decade_beta(
    tmp_path,
):
    study = load_and_validate_study(HIGH_NOISE_STUDY)
    paths = materialize_configs(study, tmp_path / "configs")
    configs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]

    assert len(configs) == 6
    assert [config["eqprop"]["endpoint_read_noise_std"] for config in configs] == [
        3.0e-4,
        5.0e-4,
    ] * 3
    assert {config["eqprop"]["endpoint_read_noise_seed"] for config in configs} == {
        2026081201
    }
    assert {config["eqprop"]["beta_tier"] for config in configs} == {
        "one_decade_lower"
    }
    assert [config["beta"] for config in configs[::2]] == pytest.approx(
        [100.0, 0.625, 0.0001171875]
    )


def test_sgd_noise_comparison_is_exact_three_case_surface():
    study = load_and_validate_study(SGD_NOISE_STUDY)
    resolved = plan(study)

    assert resolved["optimizer"] == "SGD"
    assert resolved["case_count"] == 3
    assert [(row["scheme"], row["sigma"]) for row in resolved["cases"]] == [
        (scheme, 5.0e-4) for scheme in ("baseline", "ours", "legacy")
    ]
    assert [row["injected_beta_B"] for row in resolved["cases"]] == pytest.approx(
        [100.0, 10.0, 0.03]
    )


def test_sgd_noise_materialization_preserves_optimizer_tk_and_noise_seed(tmp_path):
    study = load_and_validate_study(SGD_NOISE_STUDY)
    paths = materialize_configs(study, tmp_path / "configs")
    configs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]

    assert len(configs) == 3
    assert [config["arm_id"] for config in configs] == [
        f"conv2_{scheme}_sgd_beta_one_decade_lower_sigma_0p0005"
        for scheme in ("baseline", "ours", "legacy")
    ]
    assert {config["optimizer"]["name"] for config in configs} == {"SGD"}
    assert {config["runtime_dtype"] for config in configs} == {"float64"}
    assert {config["model_base"]["num_iterations_inference"] for config in configs} == {8}
    assert {config["model_base"]["num_iterations_training"] for config in configs} == {8}
    assert {config["eqprop"]["endpoint_read_noise_std"] for config in configs} == {
        5.0e-4
    }
    assert {config["eqprop"]["endpoint_read_noise_seed"] for config in configs} == {
        2026081201
    }
    assert [config["lr"] for config in configs] == [
        [7.90864, 3.81476, 0.82063, 7.90864, 3.81476],
        [0.523358, 0.246043, 0.0520201, 0.523358, 0.246043],
        [0.00496733, 0.00367839, 0.00236044, 0.00496733, 0.000674726],
    ]


def test_study_rejects_unmatched_noise_grid(tmp_path):
    value = json.loads(DEFAULT_STUDY.read_text(encoding="utf-8"))
    value["scientific_contract"]["noise_sigmas"] = [0.0, 1.0e-4]
    path = tmp_path / "study.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="exact noise grid"):
        load_and_validate_study(path)
