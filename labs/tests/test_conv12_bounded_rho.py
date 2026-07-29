from __future__ import annotations

import torch

from experiments.run_conv12_bounded_rho import (
    _rho_axes,
    _rho_index,
    build_source_config,
    compare_fixed_tk_gradients,
    load_study,
    select_candidates,
    surface_specs,
)


def test_focused_study_has_exactly_sixteen_nonlegacy_surfaces() -> None:
    _path, study = load_study()
    surfaces = surface_specs(study)

    assert len(surfaces) == 16
    assert {surface["architecture"] for surface in surfaces} == {"conv1", "conv2"}
    assert {surface["scheme"] for surface in surfaces} == {"baseline", "ours"}
    assert {surface["optimizer"] for surface in surfaces} == {"SGD", "Adam"}
    assert all("legacy" not in surface["surface_id"] for surface in surfaces)


def test_source_config_freezes_bounded_perfect_diode_contract(tmp_path) -> None:
    _path, study = load_study()
    checkpoint = tmp_path / "initial.pt"
    config = build_source_config(
        study,
        initializer="bounded_kaiming_uniform",
        architecture="conv2",
        scheme="ours",
        optimizer="Adam",
        init_checkpoint_path=checkpoint,
    )

    assert config["model_base"]["weight_min"] == 1e-5
    assert config["model_base"]["weight_max"] == 1e-4
    assert config["model_base"]["weight_init_mode"] == "bounded_kaiming_uniform"
    assert config["model_base"]["num_iterations_inference"] == 6
    assert config["model_base"]["num_iterations_training"] == 6
    assert config["model_base"]["voltage_amp"] == 4.0
    assert config["model_base"]["current_amp"] == 1.0
    assert config["optimizer"]["name"] == "Adam"
    assert config["datasets"]["mnist"]["factory"].endswith(
        "MnistTrainValidationDataset"
    )
    assert config["init_checkpoint_path"] == str(checkpoint.resolve())
    for name in (
        "quadratic_diode_param",
        "exponential_diode_param",
        "hard_sigmoid_param",
    ):
        assert config["model_base"][name]


def test_rho_axis_resolves_all_center_attempts_and_expansion_edges() -> None:
    _path, study = load_study()
    conv, dense = _rho_axes(study)
    center = study["rho_search"]["center"]

    for attempt in range(6):
        index = _rho_index(
            conv,
            dense,
            center["rho_conv"] / (3.0**attempt),
            center["rho_dense"] / (3.0**attempt),
        )
        assert 0 <= index < 100
    assert _rho_index(conv, dense, center["rho_conv"] * 9, center["rho_dense"]) >= 0
    assert _rho_index(conv, dense, center["rho_conv"], center["rho_dense"] * 9) >= 0


def test_fixed_tk_comparison_uses_inclusive_layerwise_gates() -> None:
    contract = {
        "reference_T": 64,
        "reference_K": 64,
        "gradient_zero_epsilon": 1e-12,
        "relative_gradient_l2_norm_delta_maximum": 0.0,
        "absolute_zero_fraction_delta_maximum": 0.0,
        "gradient_vector_cosine_minimum": 1.0,
    }
    reference = {"ConvWeight_0": [torch.tensor([1.0, 0.0])]}
    operational = {"ConvWeight_0": [torch.tensor([1.0, 0.0])]}

    result = compare_fixed_tk_gradients(
        operational,
        reference,
        contract,
        operational_t=4,
        operational_k=4,
    )

    assert result["security_passed"] is True
    assert (
        result["parameter_diagnostics"][0]["relative_gradient_l2_norm_delta"]
        == 0.0
    )


def test_selector_uses_loss_plateau_then_accuracy_and_projection() -> None:
    candidates = [
        {
            "cell_id": "a",
            "selection_eligible": True,
            "final_validation_loss": 1.0,
            "final_validation_accuracy": 0.91,
            "median_projection_efficiency": 0.9,
            "rho_conv": 0.003,
            "rho_dense": 0.01,
        },
        {
            "cell_id": "b",
            "selection_eligible": True,
            "final_validation_loss": 1.02,
            "final_validation_accuracy": 0.92,
            "median_projection_efficiency": 0.1,
            "rho_conv": 0.001,
            "rho_dense": 0.01,
        },
    ]

    selection = select_candidates(candidates, 0.02)

    assert {item["cell_id"] for item in selection["plateau"]} == {"a", "b"}
    assert selection["selected"]["cell_id"] == "b"
