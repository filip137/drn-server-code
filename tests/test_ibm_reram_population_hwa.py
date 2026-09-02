from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from experiments.artifacts import sha256_file
from model.resistive.builders import ParameterBinding
from model.variable.parameter import DenseWeight
from training.ibm_reram_endpoint_model import (
    build_ibm_reram_accepted_endpoint_model,
)
from training.ibm_reram_population_hwa import (
    IbmReramPositiveGPopulationHwaConfig,
    IbmReramPositiveGPopulationHwaModifier,
    IbmReramPositiveGProgrammingErrorSampler,
    load_raw_active_accepted_endpoint_model,
)
from training.ibm_reram_raw_active_program_verify import RAW_ACTIVE_COORDINATE


CONDITION_KEY = "adaptive__lower_to_target__tau_step_0.5"


def _artifact() -> dict[str, object]:
    records = []
    for target, probabilities in (
        (0.0, [1e-6, 1.0]),
        (1.0, [1.0, 1e-6]),
    ):
        classes = {}
        for name in (
            "target_below_lower_bound",
            "target_inside_bounds",
            "target_above_upper_bound",
        ):
            active = name == "target_inside_bounds"
            endpoint = {
                "count": 10,
                "probabilities": [0.0, 1.0],
                "values": [target, target],
            }
            classes[name] = {
                "probability": 1.0 if active else 0.0,
                "acceptance_window_reachable_probability": 1.0 if active else None,
                "success_probability": 1.0 if active else None,
                "accepted_terminal": {
                    "apparent_endpoint": endpoint,
                    "persistent_endpoint": endpoint,
                },
                "failed_terminal": {
                    "apparent_endpoint": endpoint,
                    "persistent_endpoint": endpoint,
                },
            }
        records.append(
            {
                "target": target,
                "tolerance": 0.1,
                "accepted_noncorrupt_residual": {
                    "fit_count": 10,
                    "support": [-0.1, 0.1],
                    "bin_edges": [-0.1, 0.0, 0.1],
                    "bin_probabilities": probabilities,
                },
                "outcome_model": {
                    "corrupt_identity_fraction": 0.0,
                    "noncorrupt_success_probability": 1.0,
                    "noncorrupt_reachability": {
                        "available": True,
                        "classes": classes,
                    },
                    "failed_noncorrupt_terminal": {
                        "apparent_endpoint": endpoint,
                        "persistent_endpoint": endpoint,
                    },
                    "corrupt_terminal": {
                        "apparent_endpoint": endpoint,
                        "persistent_endpoint": endpoint,
                    },
                },
            }
        )
    return {
        "schema": "ebl.ibm_reram.bounded_piecewise_uniform_endpoint_model",
        "schema_version": 2,
        "coordinate": RAW_ACTIVE_COORDINATE,
        "metadata": {
            "preset": "reram_array_om",
            "enable_published_corruption": False,
        },
        "conditions": {
            CONDITION_KEY: {
                "fit_status": "fit",
                "reachability_fit_status": "fit",
                "adequate": True,
                "validation": {"per_target": records},
            }
        },
    }


def _model():
    return build_ibm_reram_accepted_endpoint_model(
        _artifact(),
        condition_key=CONDITION_KEY,
        expected_coordinate=RAW_ACTIVE_COORDINATE,
    )


def _sampler(seed: int = 71) -> IbmReramPositiveGProgrammingErrorSampler:
    return IbmReramPositiveGProgrammingErrorSampler(
        _model(),
        IbmReramPositiveGPopulationHwaConfig(seed=seed),
        artifact_sha256="a" * 64,
    )


def test_positive_g_population_hwa_maps_g_to_x_and_ramps_without_clipping() -> None:
    sampler = _sampler()
    sampler.set_epoch(1)
    target_g = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0])

    draw = sampler.sample_full_conductance(target_g)

    assert draw.strength == pytest.approx(0.1)
    assert torch.equal(draw.target_x, target_g / 2.0)
    assert bool(torch.all(draw.apparent_x >= 0.0))
    assert bool(torch.all(draw.apparent_x <= 1.0))
    assert torch.allclose(
        draw.applied_g,
        2.0 * (draw.target_x + 0.1 * draw.residual_x),
    )
    assert bool(torch.all(draw.applied_g >= 0.0))
    assert bool(torch.all(draw.applied_g <= 2.0))
    assert sampler.report()["fixed_array_identity_during_training"] is False
    assert sampler.report()["endpoint_range_policy"] == (
        "rejection_resample_no_clipping"
    )


def test_positive_g_population_hwa_has_exact_clean_and_full_strength_limits() -> None:
    sampler = _sampler(seed=711)
    target = torch.tensor([0.0, 0.25, 1.0, 1.75, 2.0])

    sampler.set_epoch(0)
    clean = sampler.sample_full_conductance(target)
    sampler.set_epoch(10)
    full = sampler.sample_full_conductance(target)
    next_full = sampler.sample_full_conductance(target)

    assert clean.strength == 0.0
    assert torch.equal(clean.applied_g, target)
    assert full.strength == 1.0
    assert torch.equal(full.applied_g, 2.0 * full.apparent_x)
    assert not torch.equal(full.residual_x, next_full.residual_x)


def test_positive_g_population_hwa_state_resumes_exact_random_stream() -> None:
    first = _sampler(seed=72)
    first.set_epoch(10)
    target = torch.full((32,), 1.0)
    first.sample_full_conductance(target)
    saved = first.state_dict()
    expected = first.sample_full_conductance(target)

    resumed = _sampler(seed=72)
    resumed.load_state_dict(saved)
    actual = resumed.sample_full_conductance(target)

    assert actual.epoch == 10
    assert actual.strength == 1.0
    assert torch.equal(actual.residual_x, expected.residual_x)
    assert torch.equal(actual.applied_g, expected.applied_g)


def test_positive_g_modifier_uses_one_temporary_forward_state_and_restores_master() -> None:
    parameter = DenseWeight(
        (2,),
        (2,),
        gain=1.0,
        device="cpu",
        clamp=True,
        clamp_min=0.0,
        clamp_max=2.0,
        init_mode="bounded_range_uniform",
    )
    binding = ParameterBinding(
        key="base.dense_weight.0",
        parameter=parameter,
        role="dense_weight",
    )
    clean = torch.tensor([[0.25, 0.75], [1.25, 1.75]])
    parameter.state.copy_(clean)
    modifier = IbmReramPositiveGPopulationHwaModifier([binding], _sampler(seed=73))
    modifier.set_epoch(10)

    with modifier.training_context():
        temporary = parameter.state.detach().clone()
        assert not torch.equal(temporary, clean)
        assert bool(torch.all(temporary >= 0.0))
        assert bool(torch.all(temporary <= 2.0))
        # Re-reading inside the context observes exactly the same realization.
        assert torch.equal(parameter.state, temporary)

    assert torch.equal(parameter.state, clean)


def test_raw_active_loader_rejects_legacy_coordinate_and_corrupt_kernel(
    tmp_path: Path,
) -> None:
    path = tmp_path / "endpoint.json"
    path.write_text(json.dumps(_artifact()), encoding="utf-8")
    model, digest, resolved = load_raw_active_accepted_endpoint_model(
        path,
        condition_key=CONDITION_KEY,
        expected_sha256=sha256_file(path),
    )
    assert model.coordinate == RAW_ACTIVE_COORDINATE
    assert digest == sha256_file(path)
    assert resolved == path.resolve()

    wrong = _artifact()
    wrong["coordinate"] = "x=(w+1)/2"
    path.write_text(json.dumps(wrong), encoding="utf-8")
    with pytest.raises(ValueError, match="coordinate"):
        load_raw_active_accepted_endpoint_model(path, condition_key=CONDITION_KEY)

    corrupt = _artifact()
    corrupt["metadata"]["enable_published_corruption"] = True
    path.write_text(json.dumps(corrupt), encoding="utf-8")
    with pytest.raises(ValueError, match="healthy"):
        load_raw_active_accepted_endpoint_model(path, condition_key=CONDITION_KEY)
