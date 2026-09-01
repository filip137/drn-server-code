from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

from experiments.mnist_analog_relu.config import parse_crossbar_config
from experiments.mnist_analog_relu import runtime
from experiments.mnist_relu.model import BiasFreeReluTeacher
from training.ibm_om_standard_crossbar import build_crossbar_layout
from training.ibm_reram_hwa import IbmReramArrayPopulation


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "examples"
    / "mnist_analog_relu"
    / "ibm_om_onchip_importance"
    / "hwa_noise_diagnostic_repaired_stochastic_hwa.json"
)


def _population(size: int, layout) -> IbmReramArrayPopulation:
    return IbmReramArrayPopulation(
        assignment_seed=87004,
        corruption_policy="counterfactual_repaired",
        binding_keys=tuple(tile.key for tile in layout),
        binding_shapes=tuple(tile.shape for tile in layout),
        binding_sampling_seeds=(11, 12, 13),
        donor_sampling_seeds=(21, 22, 23),
        nominal_dw_min=0.0949,
        dw_min_std=0.009,
        write_noise_std=1.4113,
        max_bound=torch.full((size,), 0.25),
        min_bound=torch.full((size,), -0.25),
        dwmin_up=torch.full((size,), 0.0949),
        dwmin_down=torch.full((size,), 0.0949),
        reference=torch.zeros(size),
        corrupt=torch.zeros(size, dtype=torch.bool),
        published_corrupt=torch.zeros(size, dtype=torch.bool),
        fingerprint="hwa-noise-runtime-fixture",
        aihwkit_version="1.1.0",
    )


def test_aihwkit_om_apparent_write_noise_is_replayable_and_unclamped() -> None:
    persistent = torch.ones(1_000, dtype=torch.float32)
    first_generator = torch.Generator(device="cpu").manual_seed(123)
    second_generator = torch.Generator(device="cpu").manual_seed(123)

    first, first_noise = runtime._sample_aihwkit_om_apparent_write_noise(
        persistent_q=persistent,
        nominal_dw_min=0.0949,
        write_noise_std=1.4113,
        relative_scale=1.0,
        generator=first_generator,
    )
    second, second_noise = runtime._sample_aihwkit_om_apparent_write_noise(
        persistent_q=persistent,
        nominal_dw_min=0.0949,
        write_noise_std=1.4113,
        relative_scale=1.0,
        generator=second_generator,
    )

    assert torch.equal(first, second)
    assert torch.equal(first_noise, second_noise)
    assert torch.allclose(first - persistent, first_noise, atol=1e-7, rtol=0.0)
    assert bool(torch.any(first > 1.0))


def test_aihwkit_om_apparent_write_noise_has_preset_sigma() -> None:
    persistent = torch.zeros(250_000, dtype=torch.float32)
    apparent, noise = runtime._sample_aihwkit_om_apparent_write_noise(
        persistent_q=persistent,
        nominal_dw_min=0.0949,
        write_noise_std=1.4113,
        relative_scale=1.0,
        generator=torch.Generator(device="cpu").manual_seed(456),
    )

    expected_sigma = 1.4113 * 0.0949
    assert torch.equal(apparent, noise)
    assert float(noise.mean()) == pytest.approx(0.0, abs=1e-3)
    assert float(noise.std(unbiased=False)) == pytest.approx(
        expected_sigma, rel=0.01
    )


def test_stochastic_hwa_replays_noise_and_deploys_persistent_support_state() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    spec = parse_crossbar_config(payload)
    spec = replace(
        spec,
        offchip=replace(spec.offchip, maximum_batches=1),
        evaluation=replace(
            spec.evaluation,
            sample_limit=2,
            maximum_validation_batches=1,
        ),
    )
    layout = build_crossbar_layout((784, 50, 10), maximum_input_size=512)
    size = sum(tile.cells for tile in layout)
    population = _population(size, layout)
    source = torch.linspace(-0.5, 0.5, size, dtype=torch.float32)
    codebook_values = torch.stack((-torch.ones(size), torch.ones(size)))
    torch.manual_seed(9)
    teacher = BiasFreeReluTeacher(device=torch.device("cpu"))
    inputs = torch.rand(2, 784, generator=torch.Generator().manual_seed(10))
    labels = torch.tensor([1, 2], dtype=torch.int64)
    loader = [(inputs, labels)]

    def run(settings=spec.offchip):
        return runtime._offchip_adapt(
            source_requested=source,
            population=population,
            codebook_values=codebook_values,
            spec=replace(spec, offchip=settings),
            layout=layout,
            digital_scales=(1.0, 1.0),
            teacher=teacher,
            validation_loader=loader,
            train_loader=loader,
            device=torch.device("cpu"),
        )

    first_requested, first_report, first_state = run()
    second_requested, second_report, second_state = run()

    assert torch.equal(first_requested, second_requested)
    assert torch.equal(
        first_state["fixed_final_master_q"],
        second_state["fixed_final_master_q"],
    )
    assert torch.equal(
        first_state["forward_noise_generator_final_state"],
        second_state["forward_noise_generator_final_state"],
    )
    assert first_report["epochs"] == second_report["epochs"]
    assert first_report["stochastic_programming_during_training"] is False
    assert first_report["stochastic_apparent_forward_noise_during_training"] is True
    assert first_report["persistent_state_updates_during_training"] is False
    assert first_report["forward_noise"]["sigma_q"] == pytest.approx(
        1.4113 * 0.0949
    )
    assert first_report["forward_noise"]["native_rng_stream_parity"] is False
    assert (
        first_report["forward_noise"]["program_verify_conditioned_endpoint_sampling"]
        is False
    )
    for epoch in first_report["epochs"]:
        assert epoch["forward_noise"]["full_array_draws"] == 1
        assert epoch["forward_noise"]["scalar_values"] == size
        assert epoch["forward_noise"]["observed_std_q"] == pytest.approx(
            1.4113 * 0.0949, rel=0.02
        )
    expected_deployment = torch.maximum(
        torch.minimum(first_state["fixed_final_master_q"], population.logical_max),
        population.logical_min,
    )
    assert torch.equal(first_requested, expected_deployment)

    exact_requested, exact_report, exact_state = run(
        replace(
            spec.offchip,
            deployment_target="fixed_final_master_fault_blind_pv",
        )
    )
    assert torch.equal(exact_requested, exact_state["fixed_final_master_q"])
    assert torch.equal(
        exact_state["fixed_final_deployment_q"],
        exact_state["fixed_final_master_q"],
    )
    assert not torch.equal(
        exact_requested,
        exact_state["fixed_final_realized_q"],
    )
    assert exact_report["deployment_target"] == (
        "fixed_final_master_fault_blind_pv"
    )

    changed_noise = replace(spec.offchip.forward_noise, seed=88043)
    changed_requested, changed_report, _changed_state = run(
        replace(spec.offchip, forward_noise=changed_noise)
    )
    assert not torch.equal(first_requested, changed_requested)
    assert (
        first_report["epochs"][0]["forward_noise"][
            "noise_tensor_sequence_sha256"
        ]
        != changed_report["epochs"][0]["forward_noise"][
            "noise_tensor_sequence_sha256"
        ]
    )
