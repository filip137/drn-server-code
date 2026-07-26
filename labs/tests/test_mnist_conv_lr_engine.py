from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from experiments.mnist_conv.lr_engine import (
    _architecture_geometry,
    build_loader_bundle,
    build_model_runtime,
    canonical_parameter_name,
    parameter_state_diagnostics,
    parameter_tensor_digest,
    training_step,
)
from experiments.mnist_conv.lr_study_spec import LRStudySpec


REPO_ROOT = Path(__file__).resolve().parents[2]
STUDY_CONFIG = REPO_ROOT / "configs/conv/hardsigmoid_lr_study_sgd_bs16_v1.json"
V7_STUDY_CONFIG = (
    REPO_ROOT
    / "configs/conv/hardsigmoid_lr_conv3_scheme_two_rho_constant_sgd_bs16_v7.json"
)


class _DigestParameter:
    def __init__(self, name: str, values: list[float]):
        self.name = name
        self.state = torch.tensor(values, dtype=torch.float32)


@pytest.fixture(scope="module")
def frozen_study() -> dict:
    return LRStudySpec.from_file(STUDY_CONFIG).data


@pytest.fixture(scope="module")
def conv1_initializations(frozen_study: dict, tmp_path_factory):
    """Build the real Conv1 baseline and load its checkpoint into ours."""

    # This is the smallest study mapping needed by the numerical engine.  Its
    # model, solver, and optimizer values remain the exact frozen contract,
    # including the complete minimizer settings object.
    runtime_study = {
        key: copy.deepcopy(frozen_study[key])
        for key in ("model", "solver", "optimizer")
    }
    rows = {row["row_id"]: row for row in frozen_study["rows"]}
    baseline = build_model_runtime(
        runtime_study,
        rows["conv1_baseline_v1_c1"],
        device="cpu",
        learning_rate=1.0,
    )
    checkpoint = tmp_path_factory.mktemp("lr-engine") / "conv1-initialization.pt"
    baseline.save(checkpoint)
    ours = build_model_runtime(
        runtime_study,
        rows["conv1_ours_v4_c1"],
        device="cpu",
        initialization_checkpoint=checkpoint,
        learning_rate=1.0,
    )
    return baseline, ours, checkpoint


def test_frozen_conv_geometry_has_padding_one_and_expected_spatial_shapes(
    frozen_study: dict,
) -> None:
    architectures = frozen_study["model"]["architectures"]

    conv1_shapes, conv1_pipeline = _architecture_geometry(architectures["conv1"])
    conv2_shapes, conv2_pipeline = _architecture_geometry(architectures["conv2"])

    assert conv1_shapes == [(2, 28, 28), (64, 14, 14), (20,)]
    assert conv1_pipeline == [
        {"kernel": [3, 3], "stride": 2, "padding": 1, "mode": "convolution"}
    ]
    assert conv2_shapes == [
        (2, 28, 28),
        (64, 14, 14),
        (128, 7, 7),
        (20,),
    ]
    assert conv2_pipeline == [
        {"kernel": [3, 3], "stride": 2, "padding": 1, "mode": "convolution"},
        {"kernel": [3, 3], "stride": 2, "padding": 1, "mode": "convolution"},
    ]


def test_v7_conv3_geometry_has_three_padding_one_stages() -> None:
    architecture = LRStudySpec.from_path(V7_STUDY_CONFIG).data["model"][
        "architectures"
    ]["conv3"]

    shapes, pipeline = _architecture_geometry(architecture)

    assert shapes == [
        (2, 28, 28),
        (64, 14, 14),
        (128, 7, 7),
        (256, 7, 7),
        (20,),
    ]
    assert pipeline == [
        {"kernel": [3, 3], "stride": 2, "padding": 1, "mode": "convolution"},
        {"kernel": [3, 3], "stride": 2, "padding": 1, "mode": "convolution"},
        {"kernel": [3, 3], "stride": 1, "padding": 1, "mode": "convolution"},
    ]


def test_parameter_tensor_digest_is_deterministic_and_state_sensitive() -> None:
    parameters = [
        _DigestParameter("ConvWeight_0 ", [0.0, 1.0]),
        _DigestParameter("Bias_0", [2.0]),
    ]

    original = parameter_tensor_digest(parameters)
    assert original == parameter_tensor_digest(parameters)
    assert len(original) == 64

    parameters[0].state[0] = 0.25
    assert parameter_tensor_digest(parameters) != original
    parameters[0].state[0] = 0.0
    assert parameter_tensor_digest(parameters) == original


def test_loader_bundle_routes_only_through_mnist_train_split(
    frozen_study: dict, monkeypatch, tmp_path: Path
) -> None:
    import labs.datasets

    sentinel = object()
    calls = []

    def fake_train_validation_builder(**kwargs):
        calls.append(kwargs)
        return sentinel

    monkeypatch.setattr(
        labs.datasets,
        "build_mnist_train_validation_loaders",
        fake_train_validation_builder,
    )

    assert (
        build_loader_bundle(
            frozen_study,
            data_root=tmp_path,
            download=False,
            return_source_indices=True,
        )
        is sentinel
    )
    assert calls == [
        {
            "batch_size": 16,
            "root": str(tmp_path),
            "download": False,
            "normalize": True,
            "normalize_mean": 0.1307,
            "normalize_std": 0.3081,
            "normalize_scale": 0.3,
            "affine_config": frozen_study["dataset"]["affine"],
            "split_seed": 0,
            "shuffle_seed": 0,
            "validation_batch_size": 128,
            "return_source_indices": True,
            "num_workers": 0,
            "pin_memory": False,
        }
    ]


def test_real_conv1_runtime_has_frozen_parameters_and_bounds(
    conv1_initializations,
) -> None:
    baseline, _ours, checkpoint = conv1_initializations

    assert checkpoint.is_file()
    assert [canonical_parameter_name(param) for param in baseline.parameters] == [
        "ConvWeight_0",
        "DenseWeight_0",
        "Bias_0",
    ]
    assert [tuple(param.state.shape) for param in baseline.parameters] == [
        (64, 2, 3, 3),
        (64, 14, 14, 20),
        (64, 14, 14),
    ]
    assert [tuple(layer.state.shape[1:]) for layer in baseline.free_layers] == [
        (64, 14, 14),
        (20,),
    ]

    diagnostics = parameter_state_diagnostics(baseline.parameters)
    assert diagnostics["ConvWeight_0"]["lower_bound"] == 0.0
    assert diagnostics["ConvWeight_0"]["upper_bound"] == 100.0
    assert diagnostics["DenseWeight_0"]["lower_bound"] == 0.0
    assert diagnostics["DenseWeight_0"]["upper_bound"] == 100.0
    assert diagnostics["Bias_0"]["bounded_gate"] is False
    assert diagnostics["Bias_0"]["report_only"] is True


def test_shared_conv1_checkpoint_is_identical_across_amplification_schemes(
    conv1_initializations,
) -> None:
    baseline, ours, _checkpoint = conv1_initializations

    assert parameter_tensor_digest(baseline.parameters) == parameter_tensor_digest(
        ours.parameters
    )
    for baseline_param, ours_param in zip(baseline.parameters, ours.parameters):
        assert torch.equal(baseline_param.state, ours_param.state)
    assert baseline.row["scheme"] == "baseline"
    assert ours.row["scheme"] == "ours"
    assert baseline.energy_fn._voltage_amp == 1.0
    assert ours.energy_fn._voltage_amp == 4.0


def test_shadow_training_step_captures_all_states_and_restores_exactly(
    conv1_initializations,
) -> None:
    baseline, _ours, _checkpoint = conv1_initializations
    before = [param.state.detach().clone() for param in baseline.parameters]
    digest_before = parameter_tensor_digest(baseline.parameters)
    generator = torch.Generator().manual_seed(17)
    images = 0.1 * torch.randn((1, 1, 28, 28), generator=generator)
    labels = torch.tensor([3], dtype=torch.long)
    source_indices = torch.tensor([1234], dtype=torch.long)

    result = training_step(
        baseline,
        (images, labels, source_indices),
        learning_rate=1.0,
        restore=True,
    )

    assert result.sample_count == 1
    assert result.source_indices == (1234,)
    assert 0.0 <= result.accuracy <= 1.0
    assert result.transition.restored is True
    assert parameter_tensor_digest(baseline.parameters) == digest_before
    assert all(
        torch.equal(param.state, original)
        for param, original in zip(baseline.parameters, before)
    )

    transitions = result.transition.parameters
    assert [item.name for item in transitions] == [
        "ConvWeight_0",
        "DenseWeight_0",
        "Bias_0",
    ]
    assert any(
        not torch.equal(item.pre_update, item.post_sgd_pre_projection)
        for item in transitions
    )
    for index, (item, original) in enumerate(zip(transitions, before)):
        assert torch.equal(item.pre_update, original)
        assert item.pre_update.data_ptr() != baseline.parameters[index].state.data_ptr()
        if item.bounded_gate:
            assert bool((item.post_projection >= 0.0).all().item())
            assert bool((item.post_projection <= 100.0).all().item())
            assert item.proposed_bound_crossing_fraction is not None
            assert item.combined_bound_occupancy is not None
        else:
            assert item.report_only is True
            assert item.proposed_bound_crossing_fraction is None
