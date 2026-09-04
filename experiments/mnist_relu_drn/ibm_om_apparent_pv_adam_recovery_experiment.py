"""Adam-only recovery from exact old-OM and Figure-6-OM P&V states.

The primary device-facing state is always the current held apparent state.
Adam is digital control logic: its commands are converted to Bernoulli-selected
open-loop IBM-OM pulses, and only those pulses mutate persistent conductance.
Persistent-state forwards are retained solely as explicitly labelled secondary
diagnostics.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import torch

from experiments.artifacts import RunStore, atomic_write_json, sha256_file
from experiments.mnist_relu_drn.components import build_student_stack
from experiments.mnist_relu_drn.figure6_om_pulse import (
    Figure6OmPulsePlant,
    PersistentFigure6OmPulseAdam,
    validate_paired_om_populations,
)
from experiments.mnist_relu_drn.hfo2_figure6_drn import (
    ENDPOINT_LAYOUTS,
    EndpointField,
    flatten_physical,
)
from experiments.mnist_relu_drn.hfo2_figure6_drn_experiment import (
    EXPECTED_TEACHER_SHA256,
    FIXED_LOGIT_GAIN,
    PROFILE_SETTINGS,
    _student_payload,
)
from experiments.mnist_relu_drn.hfo2_figure6_endpoint_regimes import (
    INDEPENDENT_ENDPOINTS,
)
from experiments.mnist_relu_drn.ibm_om_apparent_pv_adam_recovery import (
    CORRUPTIONS,
    CORRUPTION_CORRUPT,
    CORRUPTION_REPAIRED,
    EXPERIMENT_ID,
    EXTENDED_FIGURE6_EXPERIMENT_ID,
    EXTENDED_FIGURE6_LEARNING_RATE,
    EXTENDED_FIGURE6_PULSE_CAP_PER_CELL,
    EXTENDED_FIGURE6_TRAINING_EPOCHS,
    EXTENDED_FIGURE6_TRAINING_EXAMPLES,
    FORWARD_STATE,
    LEARNING_RATE_GRID,
    MAIN_TRAINING_EXAMPLES,
    MODELS,
    MODEL_FIGURE6,
    MODEL_WINSORIZED,
    NOMINAL_DELTA_PROGRESS,
    PERSISTENT_ROLE,
    PULSE_CAP_PER_CELL,
    SCHEMA_VERSION,
    SCREEN_BATCHES,
    TARGETS,
    TARGET_DIRECT,
    TARGET_HWA,
    UPDATE_RULE,
    ArmKey,
    counts_by_layer,
    select_common_learning_rate,
    winsorized_apparent_projection,
    winsorized_full_conductance,
)
from experiments.mnist_relu_drn.ibm_om_baseline_spacing_pv import (
    apply_full_conductance_targets,
)
from experiments.mnist_relu_drn.ibm_om_bounded_codebook_scheme_screen import (
    _evaluate_detailed,
    _tensor_sha256,
)
from experiments.mnist_relu_drn.ibm_om_corrupt_source_hwa_cross_array_pulse_adam import (
    PairedWinsorizedOmPopulation,
)
from experiments.mnist_relu_drn.ibm_om_winsorized_cross_array_open_loop_adam import (
    ColumnSerialOpenLoopAdam,
    open_loop_pulse_port,
)
from experiments.mnist_relu_drn.runtime import _load_teacher
from experiments.mnist_shared import build_mnist_loaders, limited
from experiments.mnist_relu_drn.config import parse_student_config, resolve_student_spec
from experiments.schema import RunMode
from training.checkpoint import atomic_torch_save
from training.ibm_reram_hwa import IbmReramArrayPopulation, load_om_array_population
from training.ibm_reram_program_verify import IbmReramRawActivePlant
from training.ibm_reram_raw_active_program_verify import (
    array_population_as_pulse_population,
)


_ROOT = Path(__file__).resolve().parents[2]
_EXPECTED_SHAPES = ((1568, 100), (100, 20))
_SCREEN_PULSE_SELECTION_SEED = 98101
_MAIN_PULSE_SELECTION_SEED = 98201
_OLD_P0_RANDOM_DRAWS = 257
_OLD_RECOVERY_RANDOM_DRAWS = 385

_DEFAULT_OLD_SOURCE = (
    _ROOT
    / "results"
    / "exploratory_noncanonical-ibm-om-population-hwa-cross-array"
    / "20260902T174313.144856Z-4602e186-5b00f3fc"
)
_DEFAULT_FIGURE6_SOURCE = (
    _ROOT
    / "simulation_results"
    / "figure6_om_pulse_corruption"
    / "20260904T100623.123164Z-94d89292-b8cfae43"
)
_DEFAULT_OUTPUT = _ROOT / "simulation_results" / "ibm_om_apparent_pv_adam_recovery"

_OLD_POPULATION_HASH = {
    CORRUPTION_REPAIRED: (
        "b02af797bb84eaa93794edbd3c6db1818e3f31647e0b6a4e9c9ea1a75c98236a"
    ),
    CORRUPTION_CORRUPT: (
        "b189051337dc0f827e3b76b67997da744efe605ccc327456ecc1a70f03873954"
    ),
}
_OLD_PAIR_RECEIPT_HASH = (
    "f149a99b3e87ba4be80f25742a639b45dcd67b8b00b55812b99fcb9e10ecd6f7"
)
_FIGURE6_POPULATION_HASH = {
    CORRUPTION_REPAIRED: (
        "15b03f7852c3cfb5f46385206d2eb45dafc14e5b4109c6b96881d3aa7caa000b"
    ),
    CORRUPTION_CORRUPT: (
        "4f8657abb63c9d394b688fe892563b7dcbd5cbaf09516c1a788a435cf72ef853"
    ),
}
_FIGURE6_RECEIPT_HASH = {
    CORRUPTION_REPAIRED: (
        "d16ac9fa3708fe6c683716ef25514d1afc4a95118097408e6088b0697efde722"
    ),
    CORRUPTION_CORRUPT: (
        "34f7872da3a8e90162ac6eaa7d3d0d2a5f49025a1031af97f4f9089d51363b5b"
    ),
}
_P0_HASH = {
    ArmKey(MODEL_WINSORIZED, TARGET_DIRECT, CORRUPTION_REPAIRED): (
        "855dc0b2d6a6c84af983f65ab3618c3af2669f09343610a531551cdd7df6cb66"
    ),
    ArmKey(MODEL_WINSORIZED, TARGET_DIRECT, CORRUPTION_CORRUPT): (
        "9567dbbeab40f278c9bcc1a844a95a2ffc8960b0b4462f49bb75f2e3fee74df7"
    ),
    ArmKey(MODEL_WINSORIZED, TARGET_HWA, CORRUPTION_REPAIRED): (
        "2183ea1e3c3faa4e23229cf703204b9f02b2a3b5ecf0604e073ca1d6057c6f1e"
    ),
    ArmKey(MODEL_WINSORIZED, TARGET_HWA, CORRUPTION_CORRUPT): (
        "7191a0cdab5b5a429f6898269940a026b627ee2bde74c4ff5429ceecec81391b"
    ),
    ArmKey(MODEL_FIGURE6, TARGET_DIRECT, CORRUPTION_REPAIRED): (
        "35a2d07a0929fe763c3555c76cc1d89e165e330bb17245543707b551c4da8b0b"
    ),
    ArmKey(MODEL_FIGURE6, TARGET_DIRECT, CORRUPTION_CORRUPT): (
        "274dc8d0aaba094909a3431f0834240d97476646fa6a66c4d933fb0f01b1764c"
    ),
    ArmKey(MODEL_FIGURE6, TARGET_HWA, CORRUPTION_REPAIRED): (
        "f693c8ddbed927d26debfcd5185e2f21eac2d450a65aa0480b03449cacf3b902"
    ),
    ArmKey(MODEL_FIGURE6, TARGET_HWA, CORRUPTION_CORRUPT): (
        "70505ebcfa832676e1c300b62aeaa3769ac462107c321ce30d753cfb6290ad27"
    ),
}


def _old_policy_name(corruption: str) -> str:
    return (
        "counterfactual_repaired"
        if corruption == CORRUPTION_REPAIRED
        else "published_corrupt"
    )


def _figure6_policy_name(corruption: str) -> str:
    return (
        "counterfactual_repaired"
        if corruption == CORRUPTION_REPAIRED
        else "published"
    )


def _old_population_path(source: Path, corruption: str) -> Path:
    name = "repaired_population.npz" if corruption == CORRUPTION_REPAIRED else "published_population.npz"
    return source / "artifacts/array_A_pre_hwa/identity/winsorized" / name


def _old_pair_receipt_path(source: Path) -> Path:
    return source / "artifacts/array_A_pre_hwa/identity/paired_winsorized_assignment.json"


def _figure6_population_path(source: Path, corruption: str) -> Path:
    return source / "artifacts" / f"om_population__{_figure6_policy_name(corruption)}.npz"


def _figure6_receipt_path(source: Path, corruption: str) -> Path:
    return source / "artifacts" / f"om_population__{_figure6_policy_name(corruption)}.receipt.json"


def _p0_path(old_source: Path, figure6_source: Path, arm: ArmKey) -> Path:
    if arm.model == MODEL_WINSORIZED:
        state = "raw_relu" if arm.target == TARGET_DIRECT else "population_hwa_continuous"
        policy = _old_policy_name(arm.corruption)
        return (
            old_source
            / "artifacts/deployments/array_A"
            / state
            / f"{policy}_seed_89402.pt"
        )
    target = "direct" if arm.target == TARGET_DIRECT else "endpoint_noise_HWA"
    policy = _figure6_policy_name(arm.corruption)
    return (
        figure6_source
        / "checkpoints"
        / f"{policy}__{INDEPENDENT_ENDPOINTS}__{target}.pt"
    )


def _check_hash(path: Path, expected: str, role: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Expected {role} at {path}.")
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(
            f"{role} SHA-256 mismatch: expected={expected}, observed={observed}."
        )


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Expected readable JSON at {path}.") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected a JSON object at {path}.")
    return value


def _load_old_populations(source: Path) -> dict[str, IbmReramArrayPopulation]:
    receipt_path = _old_pair_receipt_path(source)
    _check_hash(receipt_path, _OLD_PAIR_RECEIPT_HASH, "old paired assignment receipt")
    receipt = _load_json(receipt_path)
    if (
        receipt.get("schema")
        != "ebl.mnist_relu_drn.ibm_om_paired_winsorized_assignment_receipt"
        or receipt.get("schema_version") != 1
        or receipt.get("assignment_seed") != 87004
    ):
        raise ValueError("Old paired assignment receipt changed semantically.")
    populations = {}
    for corruption in CORRUPTIONS:
        path = _old_population_path(source, corruption)
        _check_hash(path, _OLD_POPULATION_HASH[corruption], f"old {corruption} population")
        population = load_om_array_population(path)
        expected_policy = (
            "counterfactual_repaired"
            if corruption == CORRUPTION_REPAIRED
            else "published"
        )
        receipt_role = "repaired" if corruption == CORRUPTION_REPAIRED else "published"
        declared = receipt.get("winsorized", {}).get(receipt_role, {})
        if (
            population.assignment_seed != 87004
            or population.corruption_policy != expected_policy
            or population.binding_shapes != _EXPECTED_SHAPES
            or declared.get("sha256") != _OLD_POPULATION_HASH[corruption]
            or declared.get("population_fingerprint") != population.fingerprint
        ):
            raise ValueError(f"Old {corruption} population contract changed.")
        populations[corruption] = population
    PairedWinsorizedOmPopulation(
        published=populations[CORRUPTION_CORRUPT],
        repaired=populations[CORRUPTION_REPAIRED],
        report=receipt["pair"],
    )
    return populations


def _load_figure6_populations(source: Path) -> dict[str, IbmReramArrayPopulation]:
    populations = {}
    for corruption in CORRUPTIONS:
        population_path = _figure6_population_path(source, corruption)
        receipt_path = _figure6_receipt_path(source, corruption)
        _check_hash(
            population_path,
            _FIGURE6_POPULATION_HASH[corruption],
            f"Figure-6 {corruption} population",
        )
        _check_hash(
            receipt_path,
            _FIGURE6_RECEIPT_HASH[corruption],
            f"Figure-6 {corruption} receipt",
        )
        receipt = _load_json(receipt_path)
        population = load_om_array_population(population_path)
        expected_policy = _figure6_policy_name(corruption)
        if (
            receipt.get("schema") != "ebl.ibm_reram.om_array_population_receipt"
            or receipt.get("schema_version") != 1
            or receipt.get("population_sha256")
            != _FIGURE6_POPULATION_HASH[corruption]
            or receipt.get("population_fingerprint") != population.fingerprint
            or population.assignment_seed != 97001
            or population.corruption_policy != expected_policy
            or population.binding_shapes != _EXPECTED_SHAPES
        ):
            raise ValueError(f"Figure-6 {corruption} population contract changed.")
        populations[corruption] = population
    validate_paired_om_populations(
        populations[CORRUPTION_CORRUPT],
        populations[CORRUPTION_REPAIRED],
    )
    return populations


@dataclass
class _LoadedPlant:
    arm: ArmKey
    source_path: Path
    source_sha256: str
    source_payload: Mapping[str, Any]
    population: IbmReramArrayPopulation
    plant: Any
    binding_shapes: tuple[tuple[int, int], tuple[int, int]]
    conductance_ceiling: float
    continuation_adjustments: Mapping[str, Any]

    def apparent_full_g(self) -> tuple[torch.Tensor, ...]:
        if self.arm.model == MODEL_WINSORIZED:
            return winsorized_full_conductance(
                self.plant.apparent,
                self.binding_shapes,
                apparent=True,
            )
        return self.plant.apparent_full_conductance

    def persistent_full_g(self) -> tuple[torch.Tensor, ...]:
        if self.arm.model == MODEL_WINSORIZED:
            return winsorized_full_conductance(
                self.plant.persistent,
                self.binding_shapes,
                apparent=False,
            )
        return self.plant.full_conductance

    def persistent_flat(self) -> torch.Tensor:
        return (
            self.plant.persistent
            if self.arm.model == MODEL_WINSORIZED
            else self.plant.raw_a
        )

    def apparent_flat(self) -> torch.Tensor:
        return (
            self.plant.apparent
            if self.arm.model == MODEL_WINSORIZED
            else self.plant.apparent_raw_a
        )

    def state_dict(self) -> Mapping[str, Any]:
        return self.plant.state_dict()


def _load_old_p0(
    *,
    source: Path,
    arm: ArmKey,
    population: IbmReramArrayPopulation,
    device: torch.device,
) -> _LoadedPlant:
    path = _p0_path(source, Path(), arm)
    _check_hash(path, _P0_HASH[arm], f"P0 {arm.slug}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError(f"Could not load old P0 {path}.") from error
    expected_state = "raw_relu" if arm.target == TARGET_DIRECT else "population_hwa_continuous"
    expected_role = _old_policy_name(arm.corruption)
    state = payload.get("continuation_state") if isinstance(payload, Mapping) else None
    if (
        payload.get("schema") != "ebl.mnist_relu_drn.ibm_om_population_hwa_deployment"
        or payload.get("schema_version") != 1
        or payload.get("array_label") != "A"
        or payload.get("assignment_seed") != 87004
        or payload.get("endpoint_seed") != 89402
        or payload.get("state_label") != expected_state
        or payload.get("population_role") != expected_role
        or payload.get("population_fingerprint") != population.fingerprint
        or payload.get("target_clipping") is not False
        or not isinstance(state, Mapping)
        or state.get("state_coordinate") != "native_raw_active_a"
        or state.get("maximum_random_draws") != _OLD_P0_RANDOM_DRAWS
    ):
        raise ValueError(f"Old P0 semantic contract changed for {arm.slug}.")
    pulse_population = array_population_as_pulse_population(population)
    # The saved P&V buffer covers one RESET observation plus 128 two-draw
    # pulses (257 draws).  Recovery can add 64 more two-draw pulses.  Extending
    # the deterministic per-cell buffers preserves every existing draw and
    # consumes no draw; it only makes indices 257..384 available.
    extended_state = dict(state)
    extended_state["maximum_random_draws"] = _OLD_RECOVERY_RANDOM_DRAWS
    plant = IbmReramRawActivePlant(
        pulse_population,
        seeds=tuple(int(value) for value in state["seeds"]),
        device=device,
        maximum_random_draws=_OLD_RECOVERY_RANDOM_DRAWS,
    )
    plant.load_state_dict(extended_state)
    if (
        not torch.equal(plant.persistent.detach().cpu(), state["persistent"])
        or not torch.equal(plant.apparent.detach().cpu(), state["apparent"])
    ):
        raise RuntimeError("Restoring old P0 changed its persistent/apparent state.")
    return _LoadedPlant(
        arm=arm,
        source_path=path,
        source_sha256=_P0_HASH[arm],
        source_payload=payload,
        population=population,
        plant=plant,
        binding_shapes=_EXPECTED_SHAPES,
        conductance_ceiling=2.0,
        continuation_adjustments={
            "rng_buffer_capacity_extension": {
                "from_draws_per_cell": _OLD_P0_RANDOM_DRAWS,
                "to_draws_per_cell": _OLD_RECOVERY_RANDOM_DRAWS,
                "reason": "reserve_two_draws_for_each_of_64_recovery_pulses",
                "persistent_or_apparent_state_changed": False,
                "draw_indices_changed": False,
                "draws_consumed": 0,
                "per_cell_stream_prefix_preserved": True,
            }
        },
    )


def _endpoint_field_from_checkpoint(
    payload: Mapping[str, Any], device: torch.device
) -> EndpointField:
    endpoint = payload.get("target_endpoint")
    if not isinstance(endpoint, Mapping):
        raise ValueError("Figure-6 P0 lacks target endpoint tensors.")
    report = endpoint.get("report")
    if not isinstance(report, Mapping) or not isinstance(report.get("population"), Mapping):
        raise ValueError("Figure-6 P0 lacks its endpoint population report.")
    return EndpointField(
        reset=tuple(
            value.to(device=device, dtype=torch.float32)
            for value in endpoint["reset"]
        ),
        set=tuple(
            value.to(device=device, dtype=torch.float32)
            for value in endpoint["set"]
        ),
        shapes=_EXPECTED_SHAPES,
        layouts=ENDPOINT_LAYOUTS,
        population_report=report["population"],
    )


def _load_figure6_p0(
    *,
    source: Path,
    arm: ArmKey,
    population: IbmReramArrayPopulation,
    device: torch.device,
) -> _LoadedPlant:
    path = _p0_path(Path(), source, arm)
    _check_hash(path, _P0_HASH[arm], f"P0 {arm.slug}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError(f"Could not load Figure-6 P0 {path}.") from error
    expected_target = "direct" if arm.target == TARGET_DIRECT else "endpoint_noise_HWA"
    state = (
        payload.get("recovery", {}).get("initial_plant_state")
        if isinstance(payload, Mapping)
        else None
    )
    if (
        payload.get("schema") != "ebl.figure6_om_pulse_corruption_arm"
        or payload.get("schema_version") != 1
        or payload.get("corruption_policy") != _figure6_policy_name(arm.corruption)
        or payload.get("population_fingerprint") != population.fingerprint
        or payload.get("regime") != INDEPENDENT_ENDPOINTS
        or payload.get("target_kind") != expected_target
        or not isinstance(state, Mapping)
        or state.get("schema") != "ebl.figure6_om_pulse_plant"
        or state.get("population_fingerprint") != population.fingerprint
    ):
        raise ValueError(f"Figure-6 P0 semantic contract changed for {arm.slug}.")
    field = _endpoint_field_from_checkpoint(payload, device)
    plant = Figure6OmPulsePlant(
        field,
        tuple(torch.zeros(shape, device=device) for shape in _EXPECTED_SHAPES),
        population,
        pulse_noise_seed=int(state["pulse_noise_seed"]),
    )
    plant.load_state_dict(state)
    if (
        bool(torch.any(plant.pulse_count != 0))
        or not torch.equal(plant.raw_a.detach().cpu(), state["persistent_raw_a"])
        or not torch.equal(plant.apparent_raw_a.detach().cpu(), state["apparent_raw_a"])
    ):
        raise RuntimeError("Restoring Figure-6 P0 changed its exact recovery start.")
    maximum = max(float(value.max().item()) for value in field.set)
    if maximum > 6.0:
        raise RuntimeError("Figure-6 endpoint exceeds the frozen DRN conductance ceiling.")
    return _LoadedPlant(
        arm=arm,
        source_path=path,
        source_sha256=_P0_HASH[arm],
        source_payload=payload,
        population=population,
        plant=plant,
        binding_shapes=_EXPECTED_SHAPES,
        conductance_ceiling=6.0,
        continuation_adjustments={},
    )


def _load_p0(
    *,
    old_source: Path,
    figure6_source: Path,
    arm: ArmKey,
    population: IbmReramArrayPopulation,
    device: torch.device,
) -> _LoadedPlant:
    if arm.model == MODEL_WINSORIZED:
        return _load_old_p0(
            source=old_source,
            arm=arm,
            population=population,
            device=device,
        )
    return _load_figure6_p0(
        source=figure6_source,
        arm=arm,
        population=population,
        device=device,
    )


def _student_spec(conductance_ceiling: float) -> Any:
    profile = dict(PROFILE_SETTINGS["full"])
    payload = _student_payload(profile)
    payload["model"]["conductance_max"] = float(conductance_ceiling)
    payload["modes"]["train"]["num_epochs"] = 1
    payload["modes"]["train"]["learning_rates"] = [0.0, 0.0]
    return resolve_student_spec(parse_student_config(payload), RunMode.TRAIN)


def _fresh_loaders(spec: Any) -> Any:
    return build_mnist_loaders(
        spec.data,
        data_seed=spec.runtime.data_seed,
        calibration_examples=spec.mapping.calibration_examples,
        calibration_batch_size=spec.mapping.calibration_batch_size,
    )


def _apply_full_g(
    stack: Any,
    full_g: Sequence[torch.Tensor],
    *,
    conductance_ceiling: float,
) -> None:
    apply_full_conductance_targets(
        stack.bundle.catalog,
        tuple(full_g),
        conductance_min=0.0,
        conductance_max=float(conductance_ceiling),
    )


def _evaluate_full_g(
    *,
    stack: Any,
    teacher: Any,
    loader: Iterable,
    full_g: Sequence[torch.Tensor],
    conductance_ceiling: float,
    sample_limit: int | None,
) -> Mapping[str, Any]:
    _apply_full_g(stack, full_g, conductance_ceiling=conductance_ceiling)
    metrics, prediction = _evaluate_detailed(
        stack,
        teacher,
        loader,
        sample_limit=sample_limit,
    )
    result = dict(metrics)
    result["prediction_sha256"] = _tensor_sha256(prediction.to(torch.int64))
    return result


def _one_apparent_forward_gradient(
    *,
    stack: Any,
    teacher: Any,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    full_g: Sequence[torch.Tensor],
    conductance_ceiling: float,
) -> tuple[tuple[torch.Tensor, ...], Mapping[str, float | int]]:
    """Compute one physical gradient from the supplied held apparent state."""

    _apply_full_g(
        stack,
        full_g,
        conductance_ceiling=conductance_ceiling,
    )
    with torch.no_grad():
        teacher_logits = teacher.logits(inputs)
    stack.network.set_input(inputs, reset=True)
    stack.minimizer.compute_equilibrium()
    stack.cost.set_teacher(teacher_logits, labels)
    with torch.no_grad():
        per_example = stack.cost.eval()
        prediction = stack.cost.student_logits().argmax(dim=1)
        teacher_prediction = teacher_logits.argmax(dim=1)
        metrics: Mapping[str, float | int] = {
            "examples": int(labels.numel()),
            "kl_sum": float(per_example.to(torch.float64).sum().item()),
            "correct": int(prediction.eq(labels).sum().item()),
            "teacher_agreement": int(
                prediction.eq(teacher_prediction).sum().item()
            ),
        }
    gradients = tuple(stack.differentiator.compute_gradient())
    if len(gradients) != 2 or any(
        not bool(torch.isfinite(value).all()) for value in gradients
    ):
        raise FloatingPointError("Apparent-state physical DRN gradient is invalid.")
    return gradients, metrics


def _projection_report(loaded: _LoadedPlant) -> Mapping[str, Any]:
    if loaded.arm.model == MODEL_WINSORIZED:
        return winsorized_apparent_projection(loaded.plant.apparent)
    literal = flatten_physical(loaded.plant.apparent_progress_unprojected)
    below = literal < 0.0
    above = literal > 1.0
    return {
        "coordinate": "literal_held_apparent_progress=(apparent_raw_a+1)/2",
        "forward_projection": "clamp_to_[0,1]_before_endpoint_embedding",
        "below_RESET": int(below.sum().item()),
        "above_SET": int(above.sum().item()),
        "projected_cells": int((below | above).sum().item()),
        "literal_minimum": float(literal.min().item()),
        "literal_mean": float(literal.mean().item()),
        "literal_maximum": float(literal.max().item()),
    }


def _evaluate_plant(
    *,
    loaded: _LoadedPlant,
    stack: Any,
    teacher: Any,
    loader: Iterable,
    sample_limit: int | None,
) -> Mapping[str, Any]:
    apparent = _evaluate_full_g(
        stack=stack,
        teacher=teacher,
        loader=loader,
        full_g=loaded.apparent_full_g(),
        conductance_ceiling=loaded.conductance_ceiling,
        sample_limit=sample_limit,
    )
    persistent = _evaluate_full_g(
        stack=stack,
        teacher=teacher,
        loader=loader,
        full_g=loaded.persistent_full_g(),
        conductance_ceiling=loaded.conductance_ceiling,
        sample_limit=sample_limit,
    )
    return {
        "primary_state": FORWARD_STATE,
        "apparent": apparent,
        "persistent_secondary": persistent,
        "persistent_role": PERSISTENT_ROLE,
        "apparent_projection": dict(_projection_report(loaded)),
    }


def _build_optimizer(
    loaded: _LoadedPlant,
    *,
    learning_rate: float,
    pulse_cap_per_cell: int,
    pulse_selection_seed: int,
) -> Any:
    if loaded.arm.model == MODEL_WINSORIZED:
        return ColumnSerialOpenLoopAdam(
            open_loop_pulse_port(loaded.plant),
            device=loaded.plant.device,
            binding_shapes=loaded.binding_shapes,
            learning_rate_raw_x=float(learning_rate),
            nominal_delta_x=NOMINAL_DELTA_PROGRESS,
            pulse_cap=int(pulse_cap_per_cell),
            pulse_selection_seed=int(pulse_selection_seed),
        )
    return PersistentFigure6OmPulseAdam(
        loaded.plant,
        learning_rate_progress=float(learning_rate),
        pulse_cap=int(pulse_cap_per_cell),
        pulse_selection_seed=int(pulse_selection_seed),
    )


def _empty_pulse_totals() -> dict[str, Any]:
    return {
        "requested": 0,
        "applied": 0,
        "capped": 0,
        "SET": 0,
        "RESET": 0,
        "effective_state_changes": 0,
        "probability_clipped_cells_across_steps": 0,
        "mean_probability_sum": 0.0,
        "maximum_probability": 0.0,
        "steps": 0,
    }


def _add_pulse_report(
    totals: dict[str, Any],
    *,
    model: str,
    pulse: Any,
    probability_clipped: int,
) -> None:
    if model == MODEL_WINSORIZED:
        totals["requested"] += int(pulse.requested_cell_coincidences)
        totals["applied"] += int(pulse.applied_cell_pulses)
        totals["capped"] += int(pulse.capped_cell_requests)
        totals["SET"] += int(pulse.upward_pulses)
        totals["RESET"] += int(pulse.downward_pulses)
    else:
        totals["requested"] += int(pulse.requested_cells)
        totals["applied"] += int(pulse.pulsed_cells)
        totals["capped"] += int(pulse.capped_cells)
        totals["SET"] += int(pulse.upward_pulses)
        totals["RESET"] += int(pulse.downward_pulses)
        totals["effective_state_changes"] += int(pulse.effective_state_changes)
    totals["probability_clipped_cells_across_steps"] += int(probability_clipped)
    totals["mean_probability_sum"] += float(pulse.mean_probability)
    totals["maximum_probability"] = max(
        float(totals["maximum_probability"]),
        float(pulse.maximum_probability),
    )
    totals["steps"] += 1


def _optimizer_pulse_count(loaded: _LoadedPlant, optimizer: Any) -> torch.Tensor:
    return (
        optimizer.pulse_count
        if loaded.arm.model == MODEL_WINSORIZED
        else loaded.plant.pulse_count
    )


def _float_tensor_report(value: torch.Tensor) -> Mapping[str, Any]:
    flat = value.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    return {
        "minimum": float(flat.min().item()),
        "mean": float(flat.mean().item()),
        "rms": float(flat.square().mean().sqrt().item()),
        "maximum": float(flat.max().item()),
        "sha256": _tensor_sha256(value.detach().cpu()),
    }


def _state_report(
    loaded: _LoadedPlant,
    optimizer: Any,
    *,
    initial_persistent: torch.Tensor,
    pulse_cap_per_cell: int,
) -> Mapping[str, Any]:
    count = _optimizer_pulse_count(loaded, optimizer)
    corrupt = loaded.population.corrupt.to(loaded.plant.device)
    persistent = loaded.persistent_flat()
    if bool(torch.any(corrupt)) and not torch.equal(
        persistent[corrupt], initial_persistent[corrupt]
    ):
        raise RuntimeError("A published corrupt singleton moved persistently.")
    if loaded.arm.model == MODEL_WINSORIZED:
        lower = loaded.population.min_bound.to(loaded.plant.device)
        upper = loaded.population.max_bound.to(loaded.plant.device)
        saturation = {
            "at_sampled_lower": int((persistent == lower).sum().item()),
            "at_sampled_upper": int((persistent == upper).sum().item()),
        }
    else:
        saturation = {
            "at_normalized_RESET": int((persistent == -1.0).sum().item()),
            "at_normalized_SET": int((persistent == 1.0).sum().item()),
        }
    return {
        "persistent_raw_a": _float_tensor_report(persistent),
        "held_apparent_raw_a": _float_tensor_report(loaded.apparent_flat()),
        "apparent_projection": dict(_projection_report(loaded)),
        "saturation": saturation,
        "pulses_by_layer": counts_by_layer(count, loaded.binding_shapes),
        "cells_at_pulse_cap": int((count >= pulse_cap_per_cell).sum().item()),
        "maximum_pulses_per_cell": int(count.max().item()),
        "corrupt_cells": int(corrupt.sum().item()),
        "corrupt_pulse_attempts": int(count[corrupt].sum().item()),
        "corrupt_persistent_immobility_verified": True,
    }


def _optimizer_report(optimizer: Any) -> Mapping[str, Any]:
    state = optimizer.state_dict()
    return {
        "schema": state["schema"],
        "step": int(state["step"]),
        "first_moment": _float_tensor_report(state["first_moment"]),
        "second_moment": _float_tensor_report(state["second_moment"]),
        "pulse_selection_rng_state_sha256": _tensor_sha256(
            state["pulse_selection_rng_state"]
        ),
        "authoritative_weight_shadow": state["authoritative_weight_shadow"],
    }


def _run_recovery(
    *,
    arm: ArmKey,
    old_source: Path,
    figure6_source: Path,
    populations: Mapping[str, IbmReramArrayPopulation],
    stack: Any,
    teacher: Any,
    loaders: Any,
    learning_rate: float,
    training_epochs: int,
    pulse_cap_per_cell: int,
    pulse_selection_seed: int,
    maximum_batches: int | None,
    evaluation_sample_limit: int | None,
    include_test: bool,
    progress_interval: int,
    store: RunStore,
    checkpoint_prefix: str,
) -> Mapping[str, Any]:
    loaded = _load_p0(
        old_source=old_source,
        figure6_source=figure6_source,
        arm=arm,
        population=populations[arm.corruption],
        device=stack.device,
    )
    initial_state = loaded.state_dict()
    initial_persistent = loaded.persistent_flat().detach().clone()
    initial_apparent = loaded.apparent_flat().detach().clone()
    optimizer = _build_optimizer(
        loaded,
        learning_rate=learning_rate,
        pulse_cap_per_cell=pulse_cap_per_cell,
        pulse_selection_seed=pulse_selection_seed,
    )
    train_generator_start = loaders.train_generator.get_state().cpu().clone()
    before_validation = _evaluate_plant(
        loaded=loaded,
        stack=stack,
        teacher=teacher,
        loader=loaders.validation,
        sample_limit=evaluation_sample_limit,
    )
    before_test = (
        _evaluate_plant(
            loaded=loaded,
            stack=stack,
            teacher=teacher,
            loader=loaders.test,
            sample_limit=evaluation_sample_limit,
        )
        if include_test
        else None
    )

    totals = {
        "examples": 0,
        "batches": 0,
        "kl": 0.0,
        "correct": 0,
        "agreement": 0,
    }
    pulses = _empty_pulse_totals()
    if isinstance(training_epochs, bool) or training_epochs < 1:
        raise ValueError("training_epochs must be a positive integer.")
    if isinstance(pulse_cap_per_cell, bool) or pulse_cap_per_cell < 1:
        raise ValueError("pulse_cap_per_cell must be a positive integer.")
    epoch_reports = []
    for epoch_index in range(1, int(training_epochs) + 1):
        epoch_start = dict(totals)
        epoch_start_pulses = int(pulses["applied"])
        epoch_batches = 0
        for batch_index, (inputs, labels) in enumerate(
            limited(loaders.train, maximum_batches),
            start=1,
        ):
            inputs = inputs.to(stack.device, dtype=torch.float32)
            labels = labels.to(stack.device, dtype=torch.long)
            # Mandatory state contract: the gradient-producing forward consumes
            # the held apparent state immediately preceding this update.
            physical, metrics = _one_apparent_forward_gradient(
                stack=stack,
                teacher=teacher,
                inputs=inputs,
                labels=labels,
                full_g=loaded.apparent_full_g(),
                conductance_ceiling=loaded.conductance_ceiling,
            )
            clipped_before = int(
                getattr(optimizer, "total_probability_clipped_cells", 0)
            )
            pulse = optimizer.step(physical)
            clipped_after = int(
                getattr(optimizer, "total_probability_clipped_cells", 0)
            )
            _add_pulse_report(
                pulses,
                model=arm.model,
                pulse=pulse,
                probability_clipped=clipped_after - clipped_before,
            )
            examples = int(metrics["examples"])
            totals["examples"] += examples
            totals["batches"] += 1
            totals["kl"] += float(metrics["kl_sum"])
            totals["correct"] += int(metrics["correct"])
            totals["agreement"] += int(metrics["teacher_agreement"])
            epoch_batches += 1
            if batch_index % progress_interval == 0:
                print(
                    f"{checkpoint_prefix} epoch={epoch_index}/{training_epochs} "
                    f"batch={batch_index} total_batches={totals['batches']} "
                    f"examples={totals['examples']} pulses={pulses['applied']}",
                    flush=True,
                )
        if epoch_batches < 1:
            raise RuntimeError(f"Recovery epoch {epoch_index} processed no batches.")
        epoch_examples = int(totals["examples"] - epoch_start["examples"])
        epoch_validation = _evaluate_plant(
            loaded=loaded,
            stack=stack,
            teacher=teacher,
            loader=loaders.validation,
            sample_limit=evaluation_sample_limit,
        )
        epoch_reports.append(
            {
                "epoch": epoch_index,
                "examples": epoch_examples,
                "batches": epoch_batches,
                "kl_teacher_student": (
                    float(totals["kl"] - epoch_start["kl"]) / epoch_examples
                ),
                "student_accuracy": (
                    int(totals["correct"] - epoch_start["correct"])
                    / epoch_examples
                ),
                "teacher_agreement": (
                    int(totals["agreement"] - epoch_start["agreement"])
                    / epoch_examples
                ),
                "applied_pulses": int(pulses["applied"]) - epoch_start_pulses,
                "cumulative_applied_pulses": int(pulses["applied"]),
                "validation": epoch_validation,
            }
        )
        print(
            f"{checkpoint_prefix} epoch={epoch_index}/{training_epochs} complete "
            f"apparent_val="
            f"{100.0*float(epoch_validation['apparent']['student_accuracy']):.2f}% "
            f"persistent_val="
            f"{100.0*float(epoch_validation['persistent_secondary']['student_accuracy']):.2f}% "
            f"cumulative_pulses={pulses['applied']}",
            flush=True,
        )
    if totals["examples"] < 1 or totals["batches"] < 1:
        raise RuntimeError("Recovery processed no training examples.")
    if torch.equal(loaded.apparent_flat(), initial_apparent) and pulses["applied"]:
        raise RuntimeError("Applied pulses failed to refresh any held apparent state.")

    after_validation = epoch_reports[-1]["validation"]
    after_test = (
        _evaluate_plant(
            loaded=loaded,
            stack=stack,
            teacher=teacher,
            loader=loaders.test,
            sample_limit=evaluation_sample_limit,
        )
        if include_test
        else None
    )
    train_generator_end = loaders.train_generator.get_state().cpu().clone()
    pulse_steps = int(pulses.pop("steps"))
    probability_sum = float(pulses.pop("mean_probability_sum"))
    pulses["mean_probability_across_steps"] = probability_sum / pulse_steps
    pulses["steps"] = pulse_steps
    pulses["verify_reads"] = 0
    pulse_count = _optimizer_pulse_count(loaded, optimizer)
    if int(pulse_count.sum().item()) != int(pulses["applied"]):
        raise RuntimeError("Pulse accumulator and per-cell counters diverged.")
    state_report = _state_report(
        loaded,
        optimizer,
        initial_persistent=initial_persistent,
        pulse_cap_per_cell=pulse_cap_per_cell,
    )
    optimizer_report = _optimizer_report(optimizer)
    training = {
        "forward_state": FORWARD_STATE,
        "persistent_forward_used_for_gradients": False,
        "epochs": int(training_epochs),
        "epoch_reports": epoch_reports,
        "examples": int(totals["examples"]),
        "batches": int(totals["batches"]),
        "kl_teacher_student": float(totals["kl"]) / int(totals["examples"]),
        "student_accuracy": int(totals["correct"]) / int(totals["examples"]),
        "teacher_agreement": int(totals["agreement"]) / int(totals["examples"]),
    }
    checkpoint_payload = {
        "schema": "ebl.mnist_relu_drn.ibm_om_apparent_pv_adam_recovery_state",
        "schema_version": 1,
        "evidence_tier": "exploratory_noncanonical",
        "arm": {
            "model": arm.model,
            "target": arm.target,
            "corruption": arm.corruption,
            "slug": arm.slug,
        },
        "source_p0": {
            "path": str(loaded.source_path),
            "sha256": loaded.source_sha256,
            "state": initial_state,
            "continuation_adjustments": dict(loaded.continuation_adjustments),
        },
        "forward_contract": {
            "primary_state": FORWARD_STATE,
            "persistent_state": PERSISTENT_ROLE,
            "updates": UPDATE_RULE,
            "verify_reads_during_recovery": 0,
            "authoritative_weight_shadow": None,
        },
        "learning_rate_progress": float(learning_rate),
        "nominal_delta_progress": NOMINAL_DELTA_PROGRESS,
        "pulse_cap_per_cell": int(pulse_cap_per_cell),
        "pulse_selection_seed": int(pulse_selection_seed),
        "initial_persistent_sha256": _tensor_sha256(initial_persistent.cpu()),
        "initial_apparent_sha256": _tensor_sha256(initial_apparent.cpu()),
        "final_plant_state": loaded.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "train_generator_start_state": train_generator_start,
        "train_generator_end_state": train_generator_end,
        "training": training,
        "pulses": dict(pulses),
        "state_diagnostics": state_report,
        "before_validation": before_validation,
        "after_validation": after_validation,
        "before_test": before_test,
        "after_test": after_test,
    }
    rate_token = format(float(learning_rate), ".0e").replace("-", "m")
    checkpoint_path = atomic_torch_save(
        checkpoint_payload,
        store.run_dir / "checkpoints" / f"{checkpoint_prefix}__lr_{rate_token}.pt",
    )
    round_trip = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    final_state = round_trip["final_plant_state"]
    persistent_key = (
        "persistent"
        if arm.model == MODEL_WINSORIZED
        else "persistent_raw_a"
    )
    apparent_key = "apparent" if arm.model == MODEL_WINSORIZED else "apparent_raw_a"
    if (
        round_trip.get("schema")
        != "ebl.mnist_relu_drn.ibm_om_apparent_pv_adam_recovery_state"
        or round_trip.get("arm", {}).get("slug") != arm.slug
        or round_trip.get("source_p0", {}).get("sha256") != loaded.source_sha256
        or not torch.equal(
            final_state[persistent_key], loaded.persistent_flat().detach().cpu()
        )
        or not torch.equal(
            final_state[apparent_key], loaded.apparent_flat().detach().cpu()
        )
        or int(round_trip["optimizer_state"]["step"]) != int(totals["batches"])
    ):
        raise RuntimeError("Recovery checkpoint failed its strict state round-trip.")

    report = {
        "arm": checkpoint_payload["arm"],
        "source_p0": {
            "path": str(loaded.source_path),
            "sha256": loaded.source_sha256,
            "exact_initial_persistent_sha256": checkpoint_payload[
                "initial_persistent_sha256"
            ],
            "exact_initial_apparent_sha256": checkpoint_payload[
                "initial_apparent_sha256"
            ],
            "loaded_recovery_initial_state_not_selected_old_recovery": True,
            "continuation_adjustments": dict(loaded.continuation_adjustments),
        },
        "learning_rate_progress": float(learning_rate),
        "training_epochs": int(training_epochs),
        "pulse_cap_per_cell": int(pulse_cap_per_cell),
        "pulse_selection_seed": int(pulse_selection_seed),
        "training": training,
        "pulses": dict(pulses),
        "state_diagnostics": state_report,
        "optimizer": optimizer_report,
        "before_validation": before_validation,
        "after_validation": after_validation,
        "before_test": before_test,
        "after_test": after_test,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }
    store.append_metric({"mode": "apparent_state_adam_recovery", **report})
    print(
        f"{checkpoint_prefix} complete apparent_val="
        f"{100.0*float(after_validation['apparent']['student_accuracy']):.2f}% "
        f"persistent_val="
        f"{100.0*float(after_validation['persistent_secondary']['student_accuracy']):.2f}% "
        f"pulses={pulses['applied']}",
        flush=True,
    )
    del optimizer, loaded
    gc.collect()
    torch.cuda.empty_cache()
    return report


def _source_contract() -> Mapping[str, Any]:
    return {
        "teacher_sha256": EXPECTED_TEACHER_SHA256,
        "old_population_sha256": dict(_OLD_POPULATION_HASH),
        "old_pair_receipt_sha256": _OLD_PAIR_RECEIPT_HASH,
        "figure6_population_sha256": dict(_FIGURE6_POPULATION_HASH),
        "figure6_receipt_sha256": dict(_FIGURE6_RECEIPT_HASH),
        "P0_sha256": {arm.slug: digest for arm, digest in _P0_HASH.items()},
    }


def _input(role: str, path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Expected input {role} at {path}.")
    return {"role": role, "path": str(path.resolve()), "sha256": sha256_file(path)}


def _screen_inputs(
    teacher: Path, old_source: Path, figure6_source: Path
) -> tuple[Mapping[str, Any], ...]:
    inputs = [_input("frozen_relu_teacher", teacher)]
    for model in MODELS:
        arm = ArmKey(model, TARGET_DIRECT, CORRUPTION_REPAIRED)
        inputs.append(_input(f"P0_{arm.slug}", _p0_path(old_source, figure6_source, arm)))
    inputs.extend(
        (
            _input("old_repaired_population", _old_population_path(old_source, CORRUPTION_REPAIRED)),
            _input("old_corrupt_population", _old_population_path(old_source, CORRUPTION_CORRUPT)),
            _input("old_pair_receipt", _old_pair_receipt_path(old_source)),
            _input("figure6_repaired_population", _figure6_population_path(figure6_source, CORRUPTION_REPAIRED)),
            _input("figure6_repaired_receipt", _figure6_receipt_path(figure6_source, CORRUPTION_REPAIRED)),
            _input("figure6_corrupt_population", _figure6_population_path(figure6_source, CORRUPTION_CORRUPT)),
            _input("figure6_corrupt_receipt", _figure6_receipt_path(figure6_source, CORRUPTION_CORRUPT)),
        )
    )
    return tuple(inputs)


def _arm_inputs(
    *,
    teacher: Path,
    old_source: Path,
    figure6_source: Path,
    arm: ArmKey,
    receipt: Path | None,
) -> tuple[Mapping[str, Any], ...]:
    inputs = [
        _input("frozen_relu_teacher", teacher),
        _input(f"P0_{arm.slug}", _p0_path(old_source, figure6_source, arm)),
    ]
    if arm.model == MODEL_WINSORIZED:
        inputs.extend(
            (
                _input("old_repaired_population", _old_population_path(old_source, CORRUPTION_REPAIRED)),
                _input("old_corrupt_population", _old_population_path(old_source, CORRUPTION_CORRUPT)),
                _input("old_pair_receipt", _old_pair_receipt_path(old_source)),
            )
        )
    else:
        for corruption in CORRUPTIONS:
            inputs.append(
                _input(
                    f"figure6_{corruption}_population",
                    _figure6_population_path(figure6_source, corruption),
                )
            )
            inputs.append(
                _input(
                    f"figure6_{corruption}_receipt",
                    _figure6_receipt_path(figure6_source, corruption),
                )
            )
    if receipt is not None:
        inputs.append(_input("frozen_common_learning_rate_receipt", receipt))
    return tuple(inputs)


def _require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("Apparent-state OM recovery requires CUDA; CPU fallback is forbidden.")
    device = torch.device("cuda")
    probe = torch.arange(16, device=device, dtype=torch.float32).square().sum()
    if float(probe.item()) != 1240.0:
        raise RuntimeError("CUDA arithmetic canary failed.")
    return device


def _prepare_runtime(
    *, teacher_path: Path, conductance_ceiling: float, device: torch.device
) -> tuple[Any, Any, Any]:
    spec = _student_spec(conductance_ceiling)
    teacher, teacher_metadata = _load_teacher(teacher_path, device=device, spec=spec)
    stack = build_student_stack(spec, enable_measured=False)
    stack.cost.gain = FIXED_LOGIT_GAIN
    shapes = tuple(tuple(binding.state.shape) for binding in stack.bundle.catalog.trainable)
    if shapes != _EXPECTED_SHAPES:
        raise RuntimeError("Frozen DRN topology changed.")
    return spec, teacher, {"stack": stack, "teacher_metadata": teacher_metadata}


def _validate_common_inputs(
    teacher_path: Path, old_source: Path, figure6_source: Path
) -> None:
    _check_hash(teacher_path, EXPECTED_TEACHER_SHA256, "frozen ReLU teacher")
    _load_old_populations(old_source)
    _load_figure6_populations(figure6_source)
    for arm, digest in _P0_HASH.items():
        _check_hash(_p0_path(old_source, figure6_source, arm), digest, f"P0 {arm.slug}")


def run_screen(args: argparse.Namespace) -> Path:
    device = _require_cuda()
    teacher_path = args.teacher_weights.expanduser().resolve()
    old_source = args.old_source_run.expanduser().resolve()
    figure6_source = args.figure6_source_run.expanduser().resolve()
    _validate_common_inputs(teacher_path, old_source, figure6_source)
    contract = {
        "schema": EXPERIMENT_ID,
        "schema_version": SCHEMA_VERSION,
        "mode": "common_learning_rate_screen",
        "evidence_tier": "exploratory_noncanonical",
        "anchors": [
            ArmKey(model, TARGET_DIRECT, CORRUPTION_REPAIRED).slug
            for model in MODELS
        ],
        "learning_rate_grid": list(LEARNING_RATE_GRID),
        "batches_per_anchor": SCREEN_BATCHES,
        "validation_examples": 5000,
        "test_opened": False,
        "forward_state": FORWARD_STATE,
        "persistent_role": PERSISTENT_ROLE,
        "update_rule": UPDATE_RULE,
        "pulse_cap_per_cell": PULSE_CAP_PER_CELL,
        "nominal_delta_progress": NOMINAL_DELTA_PROGRESS,
        "pulse_selection_seed": _SCREEN_PULSE_SELECTION_SEED,
        "source_contract": _source_contract(),
    }
    store = RunStore.create(
        output_root=args.output_dir / "screen",
        experiment_id=EXPERIMENT_ID,
        resolved_config=contract,
        command=sys.argv,
        repo_root=_ROOT,
        input_artifacts=_screen_inputs(teacher_path, old_source, figure6_source),
        resume_capability="unsupported",
        run_id=args.run_id,
    )
    try:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        old_populations = _load_old_populations(old_source)
        figure6_populations = _load_figure6_populations(figure6_source)
        runtimes = {}
        for model, ceiling in ((MODEL_WINSORIZED, 2.0), (MODEL_FIGURE6, 6.0)):
            spec, teacher, runtime = _prepare_runtime(
                teacher_path=teacher_path,
                conductance_ceiling=ceiling,
                device=device,
            )
            runtimes[model] = (spec, teacher, runtime)
        candidates = []
        for learning_rate in LEARNING_RATE_GRID:
            anchors = {}
            for model in MODELS:
                arm = ArmKey(model, TARGET_DIRECT, CORRUPTION_REPAIRED)
                spec, teacher, runtime = runtimes[model]
                loaders = _fresh_loaders(spec)
                populations = (
                    old_populations if model == MODEL_WINSORIZED else figure6_populations
                )
                prefix = f"screen__{arm.slug}"
                report = _run_recovery(
                    arm=arm,
                    old_source=old_source,
                    figure6_source=figure6_source,
                    populations=populations,
                    stack=runtime["stack"],
                    teacher=teacher,
                    loaders=loaders,
                    learning_rate=learning_rate,
                    training_epochs=1,
                    pulse_cap_per_cell=PULSE_CAP_PER_CELL,
                    pulse_selection_seed=_SCREEN_PULSE_SELECTION_SEED,
                    maximum_batches=SCREEN_BATCHES,
                    evaluation_sample_limit=None,
                    include_test=False,
                    progress_interval=128,
                    store=store,
                    checkpoint_prefix=prefix,
                )
                anchors[model] = report
            candidates.append(
                {"learning_rate": float(learning_rate), "anchors": anchors}
            )
        selection = select_common_learning_rate(candidates)
        receipt = {
            "schema": "ebl.mnist_relu_drn.ibm_om_apparent_pv_adam_learning_rate",
            "schema_version": 1,
            "evidence_tier": "exploratory_noncanonical",
            "screen_run_id": store.manifest["run_id"],
            "screen_manifest_sha256": sha256_file(store.run_dir / "manifest.json"),
            "forward_state": FORWARD_STATE,
            "persistent_role": PERSISTENT_ROLE,
            "test_opened": False,
            "learning_rate_grid": list(LEARNING_RATE_GRID),
            "screen_batches_per_anchor": SCREEN_BATCHES,
            "anchors": [
                ArmKey(model, TARGET_DIRECT, CORRUPTION_REPAIRED).slug
                for model in MODELS
            ],
            "source_contract": _source_contract(),
            "selection": selection,
            "candidates": candidates,
        }
        receipt_path = store.run_dir / "artifacts" / "common_learning_rate_receipt.json"
        atomic_write_json(receipt_path, receipt)
        summary = {
            "schema": EXPERIMENT_ID,
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "mode": "common_learning_rate_screen",
            "selected_learning_rate_progress": selection["learning_rate"],
            "selection": selection,
            "test_opened": False,
            "primary_state": FORWARD_STATE,
            "persistent_state": PERSISTENT_ROLE,
            "receipt": str(receipt_path),
            "receipt_sha256": sha256_file(receipt_path),
        }
        summary_path = store.run_dir / "scientific_summary.json"
        atomic_write_json(summary_path, summary)
        result = store.complete(
            metrics=summary,
            artifacts=(
                store.artifact_record(receipt_path, kind="frozen_learning_rate_receipt"),
                store.artifact_record(summary_path, kind="scientific_summary"),
                *(
                    store.artifact_record(
                        Path(anchor["checkpoint"]),
                        kind="screen_anchor_checkpoint",
                    )
                    for candidate in candidates
                    for anchor in candidate["anchors"].values()
                ),
            ),
        )
        print(
            f"screen complete learning_rate={selection['learning_rate']:.1e} "
            f"receipt={receipt_path} result={result}",
            flush=True,
        )
        return result
    except BaseException as error:
        store.fail(error)
        raise


def _load_learning_rate_receipt(path: Path) -> tuple[float, Mapping[str, Any]]:
    receipt = _load_json(path)
    if (
        receipt.get("schema")
        != "ebl.mnist_relu_drn.ibm_om_apparent_pv_adam_learning_rate"
        or receipt.get("schema_version") != 1
        or receipt.get("forward_state") != FORWARD_STATE
        or receipt.get("persistent_role") != PERSISTENT_ROLE
        or receipt.get("test_opened") is not False
        or receipt.get("learning_rate_grid") != list(LEARNING_RATE_GRID)
        or receipt.get("screen_batches_per_anchor") != SCREEN_BATCHES
        or receipt.get("source_contract") != _source_contract()
    ):
        raise ValueError("Learning-rate receipt violates the frozen screen contract.")
    candidates = receipt.get("candidates")
    selection = receipt.get("selection")
    if not isinstance(candidates, list) or not isinstance(selection, Mapping):
        raise ValueError("Learning-rate receipt lacks candidates or selection.")
    reproduced = select_common_learning_rate(candidates)
    if reproduced != selection:
        raise ValueError("Learning-rate receipt selection cannot be reproduced.")
    return float(selection["learning_rate"]), receipt


def run_arm(args: argparse.Namespace) -> Path:
    device = _require_cuda()
    arm = ArmKey(args.model, args.target, args.corruption)
    teacher_path = args.teacher_weights.expanduser().resolve()
    old_source = args.old_source_run.expanduser().resolve()
    figure6_source = args.figure6_source_run.expanduser().resolve()
    _check_hash(teacher_path, EXPECTED_TEACHER_SHA256, "frozen ReLU teacher")
    _check_hash(
        _p0_path(old_source, figure6_source, arm),
        _P0_HASH[arm],
        f"P0 {arm.slug}",
    )
    canary_batches = args.canary_batches
    if canary_batches is None:
        if args.learning_rate_receipt is None or args.development_learning_rate is not None:
            raise ValueError("A production arm requires only --learning-rate-receipt.")
        receipt_path = args.learning_rate_receipt.expanduser().resolve()
        learning_rate, receipt = _load_learning_rate_receipt(receipt_path)
        execution_kind = "main_full_epoch"
        maximum_batches = None
        evaluation_sample_limit = None
        include_test = True
    else:
        if canary_batches < 1:
            raise ValueError("--canary-batches must be positive.")
        if args.development_learning_rate is None or args.learning_rate_receipt is not None:
            raise ValueError(
                "A canary requires only --development-learning-rate and no frozen receipt."
            )
        learning_rate = float(args.development_learning_rate)
        if learning_rate not in LEARNING_RATE_GRID:
            raise ValueError("Canary development rate must belong to the frozen grid.")
        receipt_path = None
        receipt = None
        execution_kind = "reduced_canary"
        maximum_batches = int(canary_batches)
        evaluation_sample_limit = int(args.canary_evaluation_examples)
        include_test = False
    populations = (
        _load_old_populations(old_source)
        if arm.model == MODEL_WINSORIZED
        else _load_figure6_populations(figure6_source)
    )
    contract = {
        "schema": EXPERIMENT_ID,
        "schema_version": SCHEMA_VERSION,
        "mode": "main_arm" if canary_batches is None else "canary_arm",
        "execution_kind": execution_kind,
        "evidence_tier": "exploratory_noncanonical",
        "arm": {
            "model": arm.model,
            "target": arm.target,
            "corruption": arm.corruption,
            "slug": arm.slug,
        },
        "learning_rate_progress": learning_rate,
        "learning_rate_receipt_sha256": (
            None if receipt_path is None else sha256_file(receipt_path)
        ),
        "training_examples": (
            MAIN_TRAINING_EXAMPLES if maximum_batches is None else None
        ),
        "maximum_batches": maximum_batches,
        "evaluation_sample_limit": evaluation_sample_limit,
        "test_opened": include_test,
        "forward_state": FORWARD_STATE,
        "persistent_role": PERSISTENT_ROLE,
        "update_rule": UPDATE_RULE,
        "pulse_cap_per_cell": PULSE_CAP_PER_CELL,
        "nominal_delta_progress": NOMINAL_DELTA_PROGRESS,
        "pulse_selection_seed": _MAIN_PULSE_SELECTION_SEED,
        "source_contract": _source_contract(),
    }
    output_root = args.output_dir / ("canaries" if canary_batches is not None else "arms") / arm.slug
    store = RunStore.create(
        output_root=output_root,
        experiment_id=EXPERIMENT_ID,
        resolved_config=contract,
        command=sys.argv,
        repo_root=_ROOT,
        input_artifacts=_arm_inputs(
            teacher=teacher_path,
            old_source=old_source,
            figure6_source=figure6_source,
            arm=arm,
            receipt=receipt_path,
        ),
        resume_capability="unsupported",
        run_id=args.run_id,
    )
    try:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        ceiling = 2.0 if arm.model == MODEL_WINSORIZED else 6.0
        spec, teacher, runtime = _prepare_runtime(
            teacher_path=teacher_path,
            conductance_ceiling=ceiling,
            device=device,
        )
        loaders = _fresh_loaders(spec)
        report = _run_recovery(
            arm=arm,
            old_source=old_source,
            figure6_source=figure6_source,
            populations=populations,
            stack=runtime["stack"],
            teacher=teacher,
            loaders=loaders,
            learning_rate=learning_rate,
            training_epochs=1,
            pulse_cap_per_cell=PULSE_CAP_PER_CELL,
            pulse_selection_seed=_MAIN_PULSE_SELECTION_SEED,
            maximum_batches=maximum_batches,
            evaluation_sample_limit=evaluation_sample_limit,
            include_test=include_test,
            progress_interval=1 if maximum_batches is not None else 500,
            store=store,
            checkpoint_prefix=arm.slug,
        )
        if maximum_batches is None and (
            report["training"]["examples"] != MAIN_TRAINING_EXAMPLES
            or report["training"]["batches"] != math.ceil(MAIN_TRAINING_EXAMPLES / 16)
        ):
            raise RuntimeError("Main arm did not process exactly one 55,000-example epoch.")
        summary = {
            "schema": EXPERIMENT_ID,
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "mode": contract["mode"],
            "execution_kind": execution_kind,
            "evidence_tier": "exploratory_noncanonical",
            "claim_scope": "model-based_hybrid_not_measured_hardware",
            "arm": report["arm"],
            "learning_rate_progress": learning_rate,
            "learning_rate_receipt": (
                None
                if receipt_path is None
                else {
                    "path": str(receipt_path),
                    "sha256": sha256_file(receipt_path),
                    "screen_run_id": receipt["screen_run_id"],
                }
            ),
            "primary_state": FORWARD_STATE,
            "persistent_state": PERSISTENT_ROLE,
            "updates": UPDATE_RULE,
            "result": report,
        }
        summary_path = store.run_dir / "scientific_summary.json"
        atomic_write_json(summary_path, summary)
        result = store.complete(
            metrics=summary,
            artifacts=(
                store.artifact_record(summary_path, kind="scientific_summary"),
                store.artifact_record(
                    Path(report["checkpoint"]), kind="exact_recovery_checkpoint"
                ),
            ),
        )
        print(f"arm complete {arm.slug} result={result}", flush=True)
        return result
    except BaseException as error:
        store.fail(error)
        raise


def run_extended_figure6_corrupt_arm(args: argparse.Namespace) -> Path:
    """Run the frozen three-epoch/high-cap/higher-LR Figure-6 follow-up."""

    device = _require_cuda()
    arm = ArmKey(MODEL_FIGURE6, args.target, CORRUPTION_CORRUPT)
    teacher_path = args.teacher_weights.expanduser().resolve()
    old_source = args.old_source_run.expanduser().resolve()
    figure6_source = args.figure6_source_run.expanduser().resolve()
    _check_hash(teacher_path, EXPECTED_TEACHER_SHA256, "frozen ReLU teacher")
    _check_hash(
        _p0_path(old_source, figure6_source, arm),
        _P0_HASH[arm],
        f"P0 {arm.slug}",
    )
    populations = _load_figure6_populations(figure6_source)

    canary_batches = args.canary_batches
    if canary_batches is None:
        training_epochs = EXTENDED_FIGURE6_TRAINING_EPOCHS
        maximum_batches = None
        evaluation_sample_limit = None
        include_test = True
        execution_kind = "three_full_epochs"
    else:
        if canary_batches < 1:
            raise ValueError("--canary-batches must be positive.")
        training_epochs = 1
        maximum_batches = int(canary_batches)
        evaluation_sample_limit = int(args.canary_evaluation_examples)
        include_test = False
        execution_kind = "reduced_extension_canary"

    contract = {
        "schema": EXTENDED_FIGURE6_EXPERIMENT_ID,
        "schema_version": 1,
        "mode": (
            "extended_figure6_corrupt_arm"
            if canary_batches is None
            else "extended_figure6_corrupt_canary"
        ),
        "execution_kind": execution_kind,
        "evidence_tier": "exploratory_noncanonical",
        "arm": {
            "model": arm.model,
            "target": arm.target,
            "corruption": arm.corruption,
            "slug": arm.slug,
        },
        "starting_state": "same_exact_saved_program_and_verify_P0_as_one_epoch_arm",
        "interventions_against_one_epoch_main": {
            "training_epochs": {"before": 1, "after": training_epochs},
            "learning_rate_progress": {
                "before": 1.0e-5,
                "after": EXTENDED_FIGURE6_LEARNING_RATE,
            },
            "pulse_cap_per_cell": {
                "before": PULSE_CAP_PER_CELL,
                "after": EXTENDED_FIGURE6_PULSE_CAP_PER_CELL,
            },
        },
        "learning_rate_progress": EXTENDED_FIGURE6_LEARNING_RATE,
        "learning_rate_rationale": (
            "next_higher_member_of_the_previously_executed_frozen_grid_and_"
            "the_rate_used_by_the_recent_successful_old_OM_Adam_control"
        ),
        "training_epochs": training_epochs,
        "training_examples": (
            EXTENDED_FIGURE6_TRAINING_EXAMPLES
            if maximum_batches is None
            else None
        ),
        "maximum_batches_per_epoch": maximum_batches,
        "evaluation_sample_limit": evaluation_sample_limit,
        "test_opened": include_test,
        "forward_state": FORWARD_STATE,
        "persistent_role": PERSISTENT_ROLE,
        "update_rule": UPDATE_RULE,
        "pulse_cap_per_cell": EXTENDED_FIGURE6_PULSE_CAP_PER_CELL,
        "pulse_cap_rationale": "three_times_the_one_epoch_cap_for_three_epochs",
        "nominal_delta_progress": NOMINAL_DELTA_PROGRESS,
        "pulse_selection_seed": _MAIN_PULSE_SELECTION_SEED,
        "source_contract": _source_contract(),
    }
    branch = "extension_canaries" if canary_batches is not None else "extensions"
    output_root = (
        args.output_dir
        / branch
        / "figure6_corrupt__3ep__lr_3em05__cap_192"
        / arm.slug
    )
    store = RunStore.create(
        output_root=output_root,
        experiment_id=EXTENDED_FIGURE6_EXPERIMENT_ID,
        resolved_config=contract,
        command=sys.argv,
        repo_root=_ROOT,
        input_artifacts=_arm_inputs(
            teacher=teacher_path,
            old_source=old_source,
            figure6_source=figure6_source,
            arm=arm,
            receipt=None,
        ),
        resume_capability="unsupported",
        run_id=args.run_id,
    )
    try:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        spec, teacher, runtime = _prepare_runtime(
            teacher_path=teacher_path,
            conductance_ceiling=6.0,
            device=device,
        )
        loaders = _fresh_loaders(spec)
        report = _run_recovery(
            arm=arm,
            old_source=old_source,
            figure6_source=figure6_source,
            populations=populations,
            stack=runtime["stack"],
            teacher=teacher,
            loaders=loaders,
            learning_rate=EXTENDED_FIGURE6_LEARNING_RATE,
            training_epochs=training_epochs,
            pulse_cap_per_cell=EXTENDED_FIGURE6_PULSE_CAP_PER_CELL,
            pulse_selection_seed=_MAIN_PULSE_SELECTION_SEED,
            maximum_batches=maximum_batches,
            evaluation_sample_limit=evaluation_sample_limit,
            include_test=include_test,
            progress_interval=1 if maximum_batches is not None else 500,
            store=store,
            checkpoint_prefix=(
                f"{arm.slug}__ep_{training_epochs}__cap_"
                f"{EXTENDED_FIGURE6_PULSE_CAP_PER_CELL}"
            ),
        )
        if maximum_batches is None and (
            report["training"]["epochs"] != EXTENDED_FIGURE6_TRAINING_EPOCHS
            or report["training"]["examples"]
            != EXTENDED_FIGURE6_TRAINING_EXAMPLES
            or report["training"]["batches"]
            != (
                EXTENDED_FIGURE6_TRAINING_EPOCHS
                * math.ceil(MAIN_TRAINING_EXAMPLES / 16)
            )
            or len(report["training"]["epoch_reports"])
            != EXTENDED_FIGURE6_TRAINING_EPOCHS
        ):
            raise RuntimeError("Extended arm did not process exactly three full epochs.")
        summary = {
            "schema": EXTENDED_FIGURE6_EXPERIMENT_ID,
            "schema_version": 1,
            "status": "complete",
            "mode": contract["mode"],
            "execution_kind": execution_kind,
            "evidence_tier": "exploratory_noncanonical",
            "claim_scope": "model-based_hybrid_not_measured_hardware",
            "arm": report["arm"],
            "starting_state": contract["starting_state"],
            "interventions_against_one_epoch_main": contract[
                "interventions_against_one_epoch_main"
            ],
            "learning_rate_progress": EXTENDED_FIGURE6_LEARNING_RATE,
            "training_epochs": training_epochs,
            "pulse_cap_per_cell": EXTENDED_FIGURE6_PULSE_CAP_PER_CELL,
            "primary_state": FORWARD_STATE,
            "persistent_state": PERSISTENT_ROLE,
            "updates": UPDATE_RULE,
            "result": report,
        }
        summary_path = store.run_dir / "scientific_summary.json"
        atomic_write_json(summary_path, summary)
        result = store.complete(
            metrics=summary,
            artifacts=(
                store.artifact_record(summary_path, kind="scientific_summary"),
                store.artifact_record(
                    Path(report["checkpoint"]), kind="exact_recovery_checkpoint"
                ),
            ),
        )
        print(f"extended arm complete {arm.slug} result={result}", flush=True)
        return result
    except BaseException as error:
        store.fail(error)
        raise


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--teacher-weights",
        type=Path,
        default=_ROOT / "data/mnist_relu_teacher_fixed_init_20260816.pt",
    )
    parser.add_argument("--old-source-run", type=Path, default=_DEFAULT_OLD_SOURCE)
    parser.add_argument(
        "--figure6-source-run", type=Path, default=_DEFAULT_FIGURE6_SOURCE
    )
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--run-id", type=str)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    screen = subparsers.add_parser("screen", help="Run the frozen common Adam LR screen.")
    _add_common_arguments(screen)
    arm = subparsers.add_parser("arm", help="Run one exact P0 recovery arm.")
    _add_common_arguments(arm)
    arm.add_argument("--model", choices=MODELS, required=True)
    arm.add_argument("--target", choices=TARGETS, required=True)
    arm.add_argument("--corruption", choices=CORRUPTIONS, required=True)
    arm.add_argument("--learning-rate-receipt", type=Path)
    arm.add_argument("--development-learning-rate", type=float)
    arm.add_argument("--canary-batches", type=int)
    arm.add_argument("--canary-evaluation-examples", type=int, default=64)
    extended = subparsers.add_parser(
        "extended-figure6-corrupt-arm",
        help="Run one frozen three-epoch/high-cap/higher-LR Figure-6 corrupt arm.",
    )
    _add_common_arguments(extended)
    extended.add_argument("--target", choices=TARGETS, required=True)
    extended.add_argument("--canary-batches", type=int)
    extended.add_argument("--canary-evaluation-examples", type=int, default=64)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "screen":
        run_screen(args)
    elif args.command == "extended-figure6-corrupt-arm":
        run_extended_figure6_corrupt_arm(args)
    else:
        run_arm(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
