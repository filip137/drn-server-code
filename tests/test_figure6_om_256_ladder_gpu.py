from __future__ import annotations

from pathlib import Path

import pytest
import torch

from experiments.mnist_relu.model import BiasFreeReluTeacher
from experiments.mnist_relu_drn.figure6_om_256_ladder import (
    _DEFAULT_CONFIG,
    _field,
    _initial_masters,
    _one_forward_gradient,
    _runtime,
)
from experiments.mnist_relu_drn.figure6_om_256_ladder_config import load_ladder_config
from experiments.mnist_relu_drn.figure6_om_pulse import (
    OM_CYCLE_NOISE_STD,
    OM_NOMINAL_DW_MIN_RAW_A,
    OM_WRITE_NOISE_STD,
    PersistentFigure6OmPulseAdam,
    program_progress_with_verify,
)
from experiments.mnist_relu_drn.hfo2_figure6_drn import (
    lift_endpoint_gradients,
    map_masters_to_conductance,
)
from training.checkpoint import encode_named_weights, save_encoded_named_weights
from training.ibm_reram_hwa import IbmReramArrayPopulation


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires real CUDA")


def _repaired_population(runtime: dict, cells: int) -> IbmReramArrayPopulation:
    mask = torch.zeros(cells, dtype=torch.bool)
    mask[[7, cells - 9]] = True
    return IbmReramArrayPopulation(
        assignment_seed=103101,
        corruption_policy="counterfactual_repaired",
        binding_keys=tuple(
            binding.key for binding in runtime["stack"].bundle.catalog.trainable
        ),
        binding_shapes=((1568, 512), (512, 20)),
        binding_sampling_seeds=(11, 12),
        donor_sampling_seeds=(21, 22),
        nominal_dw_min=OM_NOMINAL_DW_MIN_RAW_A,
        dw_min_std=OM_CYCLE_NOISE_STD,
        write_noise_std=OM_WRITE_NOISE_STD,
        max_bound=torch.ones(cells, dtype=torch.float32),
        min_bound=-torch.ones(cells, dtype=torch.float32),
        dwmin_up=torch.full((cells,), OM_NOMINAL_DW_MIN_RAW_A),
        dwmin_down=torch.full((cells,), OM_NOMINAL_DW_MIN_RAW_A),
        reference=torch.zeros(cells, dtype=torch.float32),
        corrupt=torch.zeros(cells, dtype=torch.bool),
        published_corrupt=mask,
        fingerprint="width256-gpu-canary-repaired",
        aihwkit_version="1.1.0",
    )


def test_width256_teacher_hwa_pv_fault_and_apparent_adam_canary(
    tmp_path: Path,
) -> None:
    device = torch.device("cuda")
    config = load_ladder_config(_DEFAULT_CONFIG)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    source_teacher = BiasFreeReluTeacher(device=device, dims=(784, 256, 10))
    encoded = encode_named_weights(
        source_teacher.catalog,
        metadata={
            "experiment_id": "mnist_relu.v1",
            "architecture": source_teacher.architecture,
            "dims": list(source_teacher.dims),
        },
    )
    teacher_path = tmp_path / "teacher.pt"
    save_encoded_named_weights(teacher_path, encoded, catalog=source_teacher.catalog)

    spec, teacher, runtime = _runtime(
        config,
        teacher_path=teacher_path,
        gain=1.0,
        smoke=True,
    )
    stack = runtime["stack"]
    field = _field(103001, device=device)
    masters = _initial_masters(teacher, device)
    inputs = torch.randn(2, 784, device=device)
    with torch.no_grad():
        labels = teacher.logits(inputs).argmax(dim=1)

    physical, _metrics = _one_forward_gradient(
        stack=stack,
        teacher=teacher,
        inputs=inputs,
        labels=labels,
        full_g=map_masters_to_conductance(masters, field),
    )
    logical = lift_endpoint_gradients(masters, physical, field)
    assert [tuple(value.shape) for value in logical] == [(784, 256), (256, 10)]
    assert all(torch.isfinite(value).all() for value in logical)

    population = _repaired_population(runtime, field.devices)
    sparse_target = tuple(
        torch.zeros(shape, device=device, dtype=torch.float32)
        for shape in field.shapes
    )
    sparse_target[0].reshape(-1)[0] = 0.05
    programmed, result = program_progress_with_verify(
        field,
        sparse_target,
        population,
        pulse_noise_seed=103201,
        maximum_pulses=4,
    )
    assert int(result.total_pulses.sum().item()) > 0
    faulted = programmed.clone_for_new_pulse_phase(pulse_noise_seed=103401)
    faulted.inject_post_pv_reset_stuck_faults(
        population.published_corrupt,
        observation_seed=103301,
    )
    stuck_before = faulted.raw_a[faulted.corrupt].clone()

    apparent_gradients, _metrics = _one_forward_gradient(
        stack=stack,
        teacher=teacher,
        inputs=inputs,
        labels=labels,
        full_g=faulted.apparent_full_conductance,
    )
    optimizer = PersistentFigure6OmPulseAdam(
        faulted,
        learning_rate_progress=3e-5,
        pulse_cap=4,
        pulse_selection_seed=103501,
    )
    report = optimizer.step(apparent_gradients)
    assert report.step == 1
    assert torch.equal(faulted.raw_a[faulted.corrupt], stuck_before)
    assert spec.model.dims == (1568, 512, 20)
