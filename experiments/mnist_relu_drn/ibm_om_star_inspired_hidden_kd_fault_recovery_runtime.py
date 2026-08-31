"""Exploratory STAR-inspired hidden-state BPTT recovery on corrupt IBM OM cells.

The experiment deliberately separates permanent corrupt-device faults from
program-and-verify error.  It begins from one hash-pinned, ideal-loaded,
ReLU-derived four-device DRN state, applies AIHWKit-compatible permanent
faults, and compares output-only BPTT with three hidden-repair objectives.

The hidden error is defined locally at the hidden state, but gradients are
computed by full BPTT and Adam moments remain digital.  This module therefore
must not be described as exact STAR or local on-chip learning.
"""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import gc
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from experiments.artifacts import RunStore, atomic_write_json, sha256_file
from experiments.mnist_relu_drn.components import (
    StudentStack,
    apply_targets,
    build_student_stack,
)
from experiments.mnist_relu_drn.ibm_om_baseline_selection_runtime import _input
from experiments.mnist_relu_drn.ibm_om_bounded_codebook_scheme_screen import (
    _tensor_sha256,
)
from experiments.mnist_relu_drn.ibm_om_permanent_faults import (
    PermanentFaultOverlay,
    sample_aihwkit_compatible_permanent_fault_overlay,
)
from experiments.mnist_relu_drn.ibm_om_star_inspired_hidden_kd_fault_recovery_config import (
    StarInspiredHiddenKdFaultRecoveryTrainSpec,
)
from experiments.mnist_relu_drn.ibm_om_winsorized_cross_array_open_loop_adam import (
    ColumnSerialOpenLoopAdam,
)
from experiments.mnist_relu_drn.ibm_om_winsorized_onchip_adam import (
    flatten_physical,
    split_flat_physical,
    sync_catalog_from_persistent,
)
from experiments.mnist_relu_drn.ibm_om_winsorized_qat_runtime import (
    _load_winsorized_population,
)
from experiments.mnist_relu_drn.runtime import _load_teacher
from experiments.mnist_relu_drn.star_hidden_objective import (
    ClassPrototypeTable,
    HEALTHY_CLASS_HIDDEN_KL,
    HEALTHY_CLASS_HIDDEN_MSE,
    RELU_SAMPLE_HIDDEN_KL,
    StarInspiredHiddenObjective,
    accumulate_class_prototypes,
    fit_positive_hidden_gain,
    halves_scores,
    paired_scores,
    teacher_to_student_kl,
)
from experiments.mnist_shared import build_mnist_loaders, limited
from experiments.schema import to_plain_data
from training.checkpoint import atomic_torch_save
from training.ibm_reram_program_verify import IbmReramRawActivePlant, derive_seed
from training.ibm_reram_raw_active_program_verify import matched_trajectory_seeds
from training.sgd import Backprop


if TYPE_CHECKING:
    from ebl.cli import TrainRequest


_ROOT = Path(__file__).resolve().parents[2]
SUMMARY_SCHEMA = (
    "ebl.mnist_relu_drn.ibm_om_star_inspired_hidden_kd_fault_recovery"
)
SUMMARY_SCHEMA_VERSION = 1
EVIDENCE_TIER = "exploratory_noncanonical"
CLAIM_LABEL = "STAR-inspired hidden-state BPTT fault recovery"
ARMS = (
    "output_kl_only",
    "relu_sample_hidden_kl",
    "healthy_class_hidden_kl",
    "healthy_class_hidden_mse",
)
MODE_BY_ARM = {
    "output_kl_only": RELU_SAMPLE_HIDDEN_KL,
    "relu_sample_hidden_kl": RELU_SAMPLE_HIDDEN_KL,
    "healthy_class_hidden_kl": HEALTHY_CLASS_HIDDEN_KL,
    "healthy_class_hidden_mse": HEALTHY_CLASS_HIDDEN_MSE,
}


def _source_paths(protocol: Any) -> dict[str, Path]:
    base = (
        _ROOT
        / "results"
        / protocol.artifacts.result_id
        / protocol.artifacts.run_relative_path
    )
    identity = base / "artifacts/fresh_target_87004/winsorized_identity"
    mapping = base / "artifacts/fresh_target_87004"
    return {
        "summary": base / "artifacts/scientific_summary.json",
        "population": identity / "winsorized_population.npz",
        "population_receipt": identity / "winsorized_population.receipt.json",
        "mapping": mapping / "baseline_anchor_mapping.npz",
        "mapping_receipt": mapping / "baseline_anchor_mapping.receipt.json",
    }


def _validate_source_paths(protocol: Any) -> dict[str, Path]:
    paths = _source_paths(protocol)
    expected = {
        "summary": protocol.artifacts.source_summary_sha256,
        "population": protocol.artifacts.population_sha256,
        "population_receipt": protocol.artifacts.population_receipt_sha256,
        "mapping": protocol.artifacts.mapping_sha256,
        "mapping_receipt": protocol.artifacts.mapping_receipt_sha256,
    }
    for name, path in paths.items():
        if not path.is_file() or sha256_file(path) != expected[name]:
            raise ValueError(f"Hash-pinned STAR source artifact {name!r} is missing or changed.")
    return paths


def _load_healthy_initializer(
    path: Path,
    *,
    expected_sha256: str,
    expected_tensor_sha256: Sequence[str],
    expected_assignment_seed: int,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], dict[str, Any]]:
    """Load ideal quantized full-G tensors and derive native raw-a explicitly."""

    if sha256_file(path) != expected_sha256:
        raise ValueError("Healthy initializer mapping SHA-256 mismatch.")
    try:
        with np.load(path, allow_pickle=False) as raw:
            if (
                str(raw["schema"].item())
                != "ebl.mnist_relu_drn.ibm_om_baseline_physical_mapping"
                or int(raw["schema_version"].item()) != 1
                or int(raw["assignment_seed"].item()) != expected_assignment_seed
                or float(raw["conductance_min"].item()) != 0.0
                or float(raw["conductance_max"].item()) != 2.0
            ):
                raise ValueError("Healthy initializer mapping semantic mismatch.")
            full_g = tuple(
                torch.from_numpy(raw[f"layer_{index}_quantized_conductance"].copy())
                for index in range(2)
            )
    except (OSError, KeyError, ValueError) as error:
        raise ValueError("Expected a readable pickle-free healthy initializer.") from error
    if tuple(tuple(value.shape) for value in full_g) != ((1568, 100), (100, 20)):
        raise ValueError("Healthy initializer physical shapes changed.")
    observed_hashes = tuple(_tensor_sha256(value) for value in full_g)
    if observed_hashes != tuple(expected_tensor_sha256):
        raise ValueError("Healthy initializer tensor hash mismatch.")
    if any(
        value.dtype != torch.float32
        or not bool(torch.isfinite(value).all())
        or bool(torch.any(value < 0.0))
        or bool(torch.any(value > 2.0))
        for value in full_g
    ):
        raise ValueError("Healthy initializer must contain finite non-negative G in [0,2].")
    raw_a = tuple(value - 1.0 for value in full_g)
    return full_g, raw_a, {
        "mapping_sha256": expected_sha256,
        "full_conductance_tensor_sha256": list(observed_hashes),
        "raw_a_tensor_sha256": [_tensor_sha256(value) for value in raw_a],
        "coordinate_transform": "G=raw_a+1=2*x",
        "negative_conductance_count": sum(
            int((value < 0.0).sum().item()) for value in full_g
        ),
    }


def _stack_with_objective(
    base: StudentStack,
    *,
    arm: str,
    hidden_weight: float,
    hidden_gain: float,
    hidden_temperature: float,
    prototypes: ClassPrototypeTable,
    output_gain: float,
) -> StudentStack:
    if arm not in MODE_BY_ARM:
        raise ValueError(f"Unknown STAR-inspired arm {arm!r}.")
    layers = tuple(base.bundle.energy.layers())
    if len(layers) != 3:
        raise RuntimeError("Expected input, hidden, and output DRN layers.")
    objective = StarInspiredHiddenObjective(
        layers[1],
        layers[2],
        hidden_objective=MODE_BY_ARM[arm],
        hidden_weight=hidden_weight,
        output_gain=output_gain,
        hidden_gain=(hidden_gain if arm == "relu_sample_hidden_kl" else 1.0),
        hidden_temperature=(
            hidden_temperature if arm == "relu_sample_hidden_kl" else 1.0
        ),
        class_prototypes=(
            prototypes if arm.startswith("healthy_class") else None
        ),
    )
    differentiator = Backprop(
        base.bundle.energy.params(),
        list(layers[1:]),
        objective,
        base.training_minimizer,
    )
    return replace(base, cost=objective, differentiator=differentiator)


def _bind_targets(
    stack: StudentStack,
    teacher: Any,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    *,
    include_relu_hidden: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        teacher_hidden = teacher.hidden(inputs)
        teacher_logits = teacher.logits(inputs)
    stack.cost.set_teacher(
        teacher_logits,
        labels,
        teacher_hidden if include_relu_hidden else None,
    )
    return teacher_hidden, teacher_logits


def _physical_gradients(
    stack: StudentStack,
    teacher: Any,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    *,
    hidden_weight: float,
) -> tuple[torch.Tensor, ...]:
    stack.cost.hidden_weight = hidden_weight
    stack.network.set_input(inputs, reset=True)
    stack.minimizer.compute_equilibrium()
    _bind_targets(
        stack,
        teacher,
        inputs,
        labels,
        include_relu_hidden=(stack.cost.hidden_objective == RELU_SAMPLE_HIDDEN_KL),
    )
    gradients = tuple(stack.differentiator.compute_gradient())
    if len(gradients) != 2:
        raise RuntimeError("Expected two physical BPTT gradients.")
    for gradient, binding in zip(gradients, stack.bundle.catalog.trainable):
        if tuple(gradient.shape) != tuple(binding.state.shape) or not bool(
            torch.isfinite(gradient).all()
        ):
            raise RuntimeError("BPTT returned an invalid physical gradient.")
    return gradients


def _component_gradient_report(
    stack: StudentStack,
    teacher: Any,
    loader: Iterable,
    *,
    maximum_batches: int | None,
) -> dict[str, Any]:
    """Measure output-only and hidden-only physical gradients at one state."""

    layer_sums = [
        {
            "out_sq": 0.0,
            "hidden_sq": 0.0,
            "dot": 0.0,
            "count": 0,
            "out_batch_rms_sum": 0.0,
            "hidden_batch_rms_sum": 0.0,
        }
        for _ in stack.bundle.catalog.trainable
    ]
    batches = 0
    examples = 0
    for inputs, labels in limited(loader, maximum_batches):
        inputs = inputs.to(stack.device, dtype=torch.float32)
        labels = labels.to(stack.device, dtype=torch.long)
        output = _physical_gradients(
            stack, teacher, inputs, labels, hidden_weight=0.0
        )
        total = _physical_gradients(
            stack, teacher, inputs, labels, hidden_weight=1.0
        )
        hidden = tuple(value - base for value, base in zip(total, output))
        for item, out, auxiliary in zip(layer_sums, output, hidden):
            out64 = out.detach().to(torch.float64)
            aux64 = auxiliary.detach().to(torch.float64)
            item["out_sq"] += float(out64.square().sum().item())
            item["hidden_sq"] += float(aux64.square().sum().item())
            item["dot"] += float((out64 * aux64).sum().item())
            item["count"] += int(out64.numel())
            item["out_batch_rms_sum"] += float(out64.square().mean().sqrt().item())
            item["hidden_batch_rms_sum"] += float(
                aux64.square().mean().sqrt().item()
            )
        batches += 1
        examples += int(labels.shape[0])
    if not batches:
        raise RuntimeError("Expected gradient diagnostics to process batches.")
    layers = []
    for index, item in enumerate(layer_sums):
        pooled_out_rms = math.sqrt(item["out_sq"] / item["count"])
        pooled_hidden_rms = math.sqrt(item["hidden_sq"] / item["count"])
        out_rms = item["out_batch_rms_sum"] / batches
        hidden_rms = item["hidden_batch_rms_sum"] / batches
        denominator = math.sqrt(item["out_sq"] * item["hidden_sq"])
        layers.append(
            {
                "layer": index,
                "rms_aggregation": "mean_of_batch_rms",
                "output_gradient_rms": out_rms,
                "hidden_gradient_rms": hidden_rms,
                "pooled_output_gradient_rms": pooled_out_rms,
                "pooled_hidden_gradient_rms": pooled_hidden_rms,
                "hidden_to_output_rms_ratio": (
                    hidden_rms / out_rms if out_rms > 0.0 else None
                ),
                "cosine": item["dot"] / denominator if denominator > 0.0 else None,
            }
        )
    return {"examples": examples, "batches": batches, "layers": layers}


def _lambda_anchor(gradient_report: Mapping[str, Any]) -> float:
    layer = gradient_report["layers"][0]
    output_rms = float(layer["output_gradient_rms"])
    hidden_rms = float(layer["hidden_gradient_rms"])
    if not math.isfinite(output_rms) or not math.isfinite(hidden_rms) or hidden_rms <= 0.0:
        raise RuntimeError("Cannot calibrate hidden loss from a zero/non-finite W1 gradient.")
    return output_rms / hidden_rms


def _apply_flat_full_g(stack: StudentStack, full_g: torch.Tensor, shapes: Sequence[Sequence[int]]) -> None:
    if bool(torch.any(full_g < 0.0)) or bool(torch.any(full_g > 2.0)):
        raise RuntimeError("Circuit-facing full conductance left non-negative [0,2].")
    apply_targets(stack.bundle.catalog, split_flat_physical(full_g, shapes))


def _healthy_prototypes(
    stack: StudentStack,
    loader: Any,
    *,
    expected_examples: int,
) -> tuple[ClassPrototypeTable, str]:
    deterministic = DataLoader(
        loader.train.dataset,
        batch_size=128,
        shuffle=False,
    )
    index_hasher = sha256()
    dataset = loader.train.dataset
    indices = getattr(dataset, "indices", None)
    if indices is None:
        raise RuntimeError("Expected the MNIST train loader to expose frozen split indices.")
    index_tensor = torch.as_tensor(indices, dtype=torch.int64).contiguous()
    index_hasher.update(index_tensor.numpy().tobytes(order="C"))

    def batches():
        with torch.no_grad():
            for inputs, labels in deterministic:
                inputs = inputs.to(stack.device, dtype=torch.float32)
                stack.network.set_input(inputs, reset=True)
                stack.minimizer.compute_equilibrium()
                yield (
                    stack.bundle.energy.layers()[1].state.detach().cpu(),
                    labels.detach().cpu(),
                )

    table = accumulate_class_prototypes(
        batches(),
        num_classes=10,
        split="train",
        expected_examples=expected_examples,
    )
    return table, index_hasher.hexdigest()


def _hidden_calibration(
    stack: StudentStack,
    teacher: Any,
    loader: Iterable,
    *,
    gain_min: float,
    gain_max: float,
    gain_steps: int,
) -> dict[str, Any]:
    teacher_hidden = []
    student_hidden = []
    with torch.no_grad():
        for inputs, _labels in loader:
            inputs = inputs.to(stack.device, dtype=torch.float32)
            teacher_hidden.append(teacher.hidden(inputs).detach().cpu())
            stack.network.set_input(inputs, reset=True)
            stack.minimizer.compute_equilibrium()
            student_hidden.append(
                halves_scores(stack.bundle.energy.layers()[1].state).detach().cpu()
            )
    teacher_value = torch.cat(teacher_hidden)
    student_value = torch.cat(student_hidden)
    temperature = float(teacher_value.to(torch.float64).square().mean().sqrt().item())
    fit = fit_positive_hidden_gain(
        student_value,
        teacher_value,
        temperature=temperature,
        gain_min=gain_min,
        gain_max=gain_max,
        steps=gain_steps,
    )
    return {
        **fit,
        "examples": int(teacher_value.shape[0]),
        "teacher_hidden_sha256": _tensor_sha256(teacher_value),
        "student_hidden_sha256": _tensor_sha256(student_value),
        "temperature_policy": "healthy_teacher_hidden_rms",
        "per_unit_affine_calibration": False,
    }


def _evaluate(
    stack: StudentStack,
    teacher: Any,
    loader: Iterable,
    *,
    prototypes: ClassPrototypeTable,
    relu_hidden_gain: float,
    relu_hidden_temperature: float,
) -> tuple[dict[str, Any], torch.Tensor]:
    totals = {
        "examples": 0,
        "correct": 0,
        "teacher_correct": 0,
        "output_kl": 0.0,
        "relu_hidden_kl": 0.0,
        "class_hidden_kl": 0.0,
        "class_hidden_mse": 0.0,
    }
    predictions = []
    raw_prototypes = prototypes.raw_means.to(stack.device, dtype=torch.float32)
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(stack.device, dtype=torch.float32)
            labels = labels.to(stack.device, dtype=torch.long)
            teacher_hidden = teacher.hidden(inputs)
            teacher_logits = teacher.logits(inputs)
            stack.network.set_input(inputs, reset=True)
            stack.minimizer.compute_equilibrium()
            raw_hidden = stack.bundle.energy.layers()[1].state
            hidden = halves_scores(raw_hidden)
            student_logits = (
                paired_scores(stack.bundle.energy.layers()[2].state) * stack.cost.gain
            )
            teacher_log_prob = F.log_softmax(teacher_logits, dim=1)
            teacher_prob = teacher_log_prob.exp()
            output_kl = (
                teacher_prob
                * (teacher_log_prob - F.log_softmax(student_logits, dim=1))
            ).sum(dim=1)
            selected_raw = raw_prototypes.index_select(0, labels)
            relu_kl = teacher_to_student_kl(
                teacher_hidden,
                hidden * relu_hidden_gain,
                temperature=relu_hidden_temperature,
            )
            class_kl = teacher_to_student_kl(
                halves_scores(selected_raw), hidden, temperature=1.0
            )
            class_mse = 0.5 * (raw_hidden - selected_raw).square().mean(dim=1)
            prediction = student_logits.argmax(dim=1)
            predictions.append(prediction.cpu())
            count = int(labels.shape[0])
            totals["examples"] += count
            totals["correct"] += int(prediction.eq(labels).sum().item())
            totals["teacher_correct"] += int(
                teacher_logits.argmax(dim=1).eq(labels).sum().item()
            )
            totals["output_kl"] += float(output_kl.sum().item())
            totals["relu_hidden_kl"] += float(relu_kl.sum().item())
            totals["class_hidden_kl"] += float(class_kl.sum().item())
            totals["class_hidden_mse"] += float(class_mse.sum().item())
    examples = int(totals["examples"])
    if examples < 1:
        raise RuntimeError("Expected evaluation examples.")
    prediction_tensor = torch.cat(predictions).to(torch.int64)
    return {
        "examples": examples,
        "student_correct": int(totals["correct"]),
        "student_accuracy": totals["correct"] / examples,
        "teacher_correct": int(totals["teacher_correct"]),
        "teacher_accuracy": totals["teacher_correct"] / examples,
        "output_kl": totals["output_kl"] / examples,
        "relu_sample_hidden_kl": totals["relu_hidden_kl"] / examples,
        "healthy_class_hidden_kl": totals["class_hidden_kl"] / examples,
        "healthy_class_hidden_mse": totals["class_hidden_mse"] / examples,
        "prediction_sha256": _tensor_sha256(prediction_tensor),
    }, prediction_tensor


def _assert_evaluation_identity(
    report: Mapping[str, Any],
    *,
    split: str,
    expected_examples: int,
    expected_correct: int,
    expected_prediction_sha256: str,
) -> None:
    observed = (
        int(report["examples"]),
        int(report["student_correct"]),
        str(report["prediction_sha256"]),
    )
    expected = (
        expected_examples,
        expected_correct,
        expected_prediction_sha256,
    )
    if observed != expected:
        raise RuntimeError(
            f"Healthy {split} prediction identity changed: "
            f"observed={observed!r}, expected={expected!r}."
        )


def _fault_p0_raw_a(
    healthy_raw_a: torch.Tensor,
    overlay: PermanentFaultOverlay,
    *,
    device: torch.device,
) -> torch.Tensor:
    state = healthy_raw_a.detach().to(device=device, dtype=torch.float32).clone()
    overlay.clamp_raw_a_(state)
    if bool(torch.any(state + 1.0 < 0.0)):
        raise RuntimeError("Faulted P0 produced a negative circuit conductance.")
    return state


def _new_fault_plant(
    *,
    population: Any,
    overlay: PermanentFaultOverlay,
    p0_raw_a: torch.Tensor,
    fault_seed: int,
    device: torch.device,
    pulse_cap: int,
) -> IbmReramRawActivePlant:
    seeds = matched_trajectory_seeds(
        population,
        endpoint_seed=derive_seed(fault_seed, "star_fault_only_pulse_plant"),
    )
    plant = IbmReramRawActivePlant(
        overlay.pulse_population,
        seeds=seeds,
        device=device,
        maximum_random_draws=pulse_cap,
    )
    plant.persistent.copy_(p0_raw_a.to(device))
    plant.apparent.copy_(plant.persistent)
    return plant


def _train_continuous(
    *,
    stack: StudentStack,
    teacher: Any,
    overlay: PermanentFaultOverlay,
    p0_raw_a: torch.Tensor,
    loader: Iterable,
    learning_rate_raw_x: float,
    hidden_weight: float,
    maximum_batches: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    x = torch.nn.Parameter((p0_raw_a + 1.0) / 2.0)
    optimizer = torch.optim.Adam(
        [x],
        lr=learning_rate_raw_x,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    before_g = (p0_raw_a + 1.0).detach().clone()
    examples = 0
    batches = 0
    for inputs, labels in limited(loader, maximum_batches):
        _apply_flat_full_g(stack, 2.0 * x.detach(), overlay.binding_shapes)
        inputs = inputs.to(stack.device, dtype=torch.float32)
        labels = labels.to(stack.device, dtype=torch.long)
        gradient_g = _physical_gradients(
            stack,
            teacher,
            inputs,
            labels,
            hidden_weight=hidden_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        x.grad = 2.0 * flatten_physical(gradient_g).detach()
        optimizer.step()
        with torch.no_grad():
            raw_a = 2.0 * x - 1.0
            overlay.clamp_raw_a_(raw_a)
            x.copy_((raw_a + 1.0) / 2.0)
        examples += int(labels.shape[0])
        batches += 1
    final_g = (2.0 * x.detach()).clone()
    transition = overlay.transition_report(
        before_g,
        final_g,
        coordinate="full_conductance_g",
    )
    if transition["fault_moved_count"] != 0:
        raise RuntimeError("Continuous actuator moved a permanent fault.")
    return final_g, {
        "examples": examples,
        "batches": batches,
        "learning_rate_raw_x": learning_rate_raw_x,
        "hidden_weight": hidden_weight,
        "transition": transition,
        "full_conductance_sha256": _tensor_sha256(final_g.cpu()),
    }


def _train_pulse(
    *,
    name: str,
    stack: StudentStack,
    teacher: Any,
    population: Any,
    overlay: PermanentFaultOverlay,
    p0_raw_a: torch.Tensor,
    fault_seed: int,
    loader: Any,
    train_generator_state: torch.Tensor,
    learning_rate_raw_x: float,
    hidden_weight: float,
    pulse_selection_seed: int,
    protocol: Any,
) -> tuple[IbmReramRawActivePlant, ColumnSerialOpenLoopAdam, dict[str, Any]]:
    plant = _new_fault_plant(
        population=population,
        overlay=overlay,
        p0_raw_a=p0_raw_a,
        fault_seed=fault_seed,
        device=stack.device,
        pulse_cap=protocol.training.pulse_cap,
    )
    port, ledger = overlay.make_write_port(plant.controller_port())
    optimizer = ColumnSerialOpenLoopAdam(
        port,
        device=stack.device,
        binding_shapes=population.binding_shapes,
        learning_rate_raw_x=learning_rate_raw_x,
        nominal_delta_x=protocol.training.nominal_delta_x,
        pulse_cap=protocol.training.pulse_cap,
        pulse_selection_seed=pulse_selection_seed,
        beta1=protocol.training.beta1,
        beta2=protocol.training.beta2,
        epsilon=protocol.training.epsilon,
    )
    loader.train_generator.set_state(train_generator_state.clone())
    before = plant.persistent.detach().clone()
    examples = 0
    batches = 0
    dispatched = 0
    capped = 0
    probability_clipped = 0
    for inputs, labels in loader.train:
        sync_catalog_from_persistent(
            stack.bundle.catalog,
            plant,
            binding_shapes=population.binding_shapes,
        )
        inputs = inputs.to(stack.device, dtype=torch.float32)
        labels = labels.to(stack.device, dtype=torch.long)
        gradients = _physical_gradients(
            stack,
            teacher,
            inputs,
            labels,
            hidden_weight=hidden_weight,
        )
        step = optimizer.step(gradients)
        dispatched += step.requested_cell_coincidences
        capped += step.capped_cell_requests
        probability_clipped += step.probability_clipped_cells
        examples += int(labels.shape[0])
        batches += 1
    transition = overlay.transition_report(
        before,
        plant.persistent,
        coordinate="raw_a",
    )
    if transition["fault_moved_count"] != 0:
        raise RuntimeError("Open-loop pulse recovery moved a permanent fault.")
    ledger_report = ledger.report()
    if ledger_report["verify_reads"] != 0:
        raise RuntimeError("Fault recovery unexpectedly used verify reads.")
    return plant, optimizer, {
        "arm": name,
        "examples": examples,
        "batches": batches,
        "hidden_weight": hidden_weight,
        "learning_rate_raw_x": learning_rate_raw_x,
        "dispatched_pulse_attempts": dispatched,
        "capped_pulse_attempts": capped,
        "probability_clipped_cell_events": probability_clipped,
        "write_ledger": ledger_report,
        "transition": transition,
        "program_verify": False,
        "inference_read_noise": False,
        "cycle_to_cycle_noise": False,
        "apparent_write_noise": False,
    }


def _save_fault_artifact(
    store: RunStore,
    *,
    seed: int,
    overlay: PermanentFaultOverlay,
    p0_raw_a: torch.Tensor,
) -> tuple[Path, Path]:
    root = store.run_dir / "artifacts/faults"
    root.mkdir(parents=True, exist_ok=True)
    npz = root / f"fault_seed_{seed}.npz"
    np.savez_compressed(
        npz,
        schema=np.asarray("ebl.mnist_relu_drn.ibm_om_permanent_fault_state"),
        schema_version=np.asarray(1, dtype=np.int64),
        fault_seed=np.asarray(seed, dtype=np.int64),
        fault_mask=overlay.fault_mask.cpu().numpy(),
        raw_a_stuck=overlay.raw_a_stuck.cpu().numpy(),
        full_conductance_stuck=overlay.full_conductance_stuck.cpu().numpy(),
        post_fault_p0_raw_a=p0_raw_a.detach().cpu().numpy(),
        post_fault_p0_full_conductance=(p0_raw_a.detach().cpu() + 1.0).numpy(),
    )
    receipt = root / f"fault_seed_{seed}.receipt.json"
    atomic_write_json(
        receipt,
        {
            **overlay.receipt.as_dict(),
            "artifact": npz.name,
            "artifact_sha256": sha256_file(npz),
            "post_fault_p0_raw_a_sha256": _tensor_sha256(p0_raw_a.cpu()),
            "post_fault_p0_full_conductance_sha256": _tensor_sha256(
                (p0_raw_a + 1.0).cpu()
            ),
            "initialization": "oracle_ideal_quantized_ReLU_derived_G_then_fault_overlay",
            "program_verify": False,
        },
    )
    return npz, receipt


def _save_prototypes(
    store: RunStore,
    prototypes: ClassPrototypeTable,
    *,
    train_split_sha256: str,
) -> tuple[Path, Path]:
    npz = store.run_dir / "artifacts/star_targets.npz"
    np.savez_compressed(
        npz,
        schema=np.asarray("ebl.mnist_relu_drn.star_healthy_class_prototypes"),
        schema_version=np.asarray(1, dtype=np.int64),
        raw_means=prototypes.raw_means.numpy(),
        decoded_means=prototypes.decoded_means.numpy(),
        class_counts=prototypes.class_counts.numpy(),
        examples=np.asarray(prototypes.examples, dtype=np.int64),
        split=np.asarray(prototypes.split),
    )
    receipt = store.run_dir / "artifacts/star_targets.receipt.json"
    atomic_write_json(
        receipt,
        {
            "schema": "ebl.mnist_relu_drn.star_healthy_class_prototypes_receipt",
            "schema_version": 1,
            "artifact": npz.name,
            "artifact_sha256": sha256_file(npz),
            "split": "train",
            "examples": prototypes.examples,
            "class_counts": prototypes.class_counts.tolist(),
            "train_split_indices_sha256": train_split_sha256,
            "accumulation_dtype": "float64",
            "raw_means_sha256": _tensor_sha256(prototypes.raw_means),
            "decoded_means_sha256": _tensor_sha256(prototypes.decoded_means),
            "validation_or_test_used": False,
        },
    )
    return npz, receipt


def _selection_key(report: Mapping[str, Any]) -> tuple[Any, ...]:
    multiplier = float(report["lambda_multiplier"])
    validation = report["validation"]
    return (
        -int(validation["student_correct"]),
        float(validation["output_kl"]),
        abs(multiplier - 1.0),
        multiplier,
    )


def _relative_pulse_mismatch(attempts: int, baseline_attempts: int) -> float:
    return abs(int(attempts) - int(baseline_attempts)) / max(
        int(baseline_attempts), 1
    )


def _summaries_by_arm(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = {}
    for arm in ARMS:
        selected = [row for row in rows if row["arm"] == arm]
        if not selected:
            continue
        accuracies = [float(row["test"]["student_accuracy"]) for row in selected]
        result[arm] = {
            "masks": len(selected),
            "mean_test_accuracy": sum(accuracies) / len(accuracies),
            "minimum_test_accuracy": min(accuracies),
            "maximum_test_accuracy": max(accuracies),
            "test_accuracy_by_fault_seed": {
                str(row["fault_seed"]): row["test"]["student_accuracy"]
                for row in selected
            },
            "mean_attempted_pulses": sum(
                int(row["training"]["write_ledger"]["attempted_pulse_requests"])
                for row in selected
            )
            / len(selected),
        }
    baseline = result.get("output_kl_only")
    if baseline is not None:
        for arm, value in result.items():
            value["mean_test_gain_vs_output_only"] = (
                value["mean_test_accuracy"] - baseline["mean_test_accuracy"]
            )
    return result


def run_train(request: "TrainRequest") -> int:
    spec = request.spec
    if not isinstance(spec, StarInspiredHiddenKdFaultRecoveryTrainSpec):
        raise TypeError("Expected the dedicated STAR-inspired fault-recovery spec.")
    if request.teacher_weights is None:
        raise ValueError("Expected explicit --teacher-weights.")
    if any(
        value is not None
        for value in (
            request.weights,
            request.base_weights,
            request.resume,
            request.device_data,
            request.device_model,
        )
    ):
        raise ValueError("STAR-inspired recovery rejects weights/base/resume/device inputs.")
    protocol = spec.protocol
    teacher_path = request.teacher_weights.expanduser().resolve()
    if sha256_file(teacher_path) != protocol.artifacts.teacher_sha256:
        raise ValueError("Frozen ReLU teacher SHA-256 mismatch.")
    paths = _validate_source_paths(protocol)
    store = RunStore.create(
        output_root=request.output_dir,
        experiment_id=spec.experiment_id,
        resolved_config=to_plain_data(spec),
        command=request.command,
        repo_root=_ROOT,
        input_artifacts=(
            _input("teacher_weights", teacher_path),
            _input("source_87004_summary", paths["summary"]),
            _input("source_87004_winsorized_population", paths["population"]),
            _input("source_87004_population_receipt", paths["population_receipt"]),
            _input("source_87004_ideal_quantized_mapping", paths["mapping"]),
            _input("source_87004_mapping_receipt", paths["mapping_receipt"]),
        ),
        resume_capability="unsupported",
    )
    artifacts = []
    try:
        if protocol.execution.device != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("STAR-inspired fault recovery requires local CUDA.")
        torch.manual_seed(spec.student.runtime.seed)
        torch.cuda.manual_seed_all(spec.student.runtime.seed)
        device = torch.device("cuda")
        loaders = build_mnist_loaders(
            spec.student.data,
            data_seed=spec.student.runtime.data_seed,
            calibration_examples=spec.student.mapping.calibration_examples,
            calibration_batch_size=spec.student.mapping.calibration_batch_size,
        )
        teacher, teacher_metadata = _load_teacher(
            teacher_path,
            device=device,
            spec=spec.student,
        )
        if not hasattr(teacher, "hidden"):
            raise RuntimeError("STAR-inspired hidden KD requires a ReLU teacher representation.")
        base_stack = build_student_stack(spec.student, enable_measured=False)
        population = _load_winsorized_population(
            paths["population"],
            paths["population_receipt"],
            expected_sha256=protocol.artifacts.population_sha256,
            expected_fingerprint=protocol.artifacts.population_fingerprint,
            expected_assignment_seed=protocol.target_assignment_seed,
            source_joint_hardware_instance_id=(
                protocol.artifacts.source_hardware_instance_id
            ),
        )
        healthy_g_layers, healthy_a_layers, initializer_report = _load_healthy_initializer(
            paths["mapping"],
            expected_sha256=protocol.artifacts.mapping_sha256,
            expected_tensor_sha256=protocol.artifacts.mapping_tensor_sha256,
            expected_assignment_seed=protocol.target_assignment_seed,
        )
        if tuple(population.binding_shapes) != tuple(
            tuple(value.shape) for value in healthy_g_layers
        ):
            raise RuntimeError("Population and healthy initializer bindings differ.")
        healthy_g = flatten_physical(healthy_g_layers).to(device)
        healthy_a = flatten_physical(healthy_a_layers).to(device)
        _apply_flat_full_g(base_stack, healthy_g, population.binding_shapes)
        base_stack.cost.gain = protocol.execution.fixed_logit_gain

        prototypes, train_split_sha256 = _healthy_prototypes(
            base_stack,
            loaders,
            expected_examples=protocol.objectives.class_prototype_examples,
        )
        prototype_npz, prototype_receipt = _save_prototypes(
            store,
            prototypes,
            train_split_sha256=train_split_sha256,
        )
        artifacts.extend(
            (
                store.artifact_record(prototype_npz, kind="healthy_class_prototypes"),
                store.artifact_record(
                    prototype_receipt, kind="healthy_class_prototypes_receipt"
                ),
            )
        )
        hidden_calibration = _hidden_calibration(
            base_stack,
            teacher,
            loaders.calibration,
            gain_min=spec.student.mapping.logit_gain_min,
            gain_max=spec.student.mapping.logit_gain_max,
            gain_steps=spec.student.mapping.logit_gain_steps,
        )
        relu_gain = float(hidden_calibration["gain"])
        relu_temperature = float(hidden_calibration["temperature"])
        screen_loader = DataLoader(
            loaders.calibration.dataset,
            batch_size=protocol.training.batch_size,
            shuffle=False,
        )
        if len(screen_loader) != math.ceil(
            protocol.training.continuous_screen_examples
            / protocol.training.batch_size
        ):
            raise RuntimeError(
                "The deterministic screening loader does not cover the declared "
                "example budget at the recovery batch size."
            )
        healthy_stack = _stack_with_objective(
            base_stack,
            arm="output_kl_only",
            hidden_weight=0.0,
            hidden_gain=relu_gain,
            hidden_temperature=relu_temperature,
            prototypes=prototypes,
            output_gain=protocol.execution.fixed_logit_gain,
        )
        healthy_validation, _ = _evaluate(
            healthy_stack,
            teacher,
            loaders.validation,
            prototypes=prototypes,
            relu_hidden_gain=relu_gain,
            relu_hidden_temperature=relu_temperature,
        )
        _assert_evaluation_identity(
            healthy_validation,
            split="validation",
            expected_examples=5_000,
            expected_correct=protocol.artifacts.healthy_validation_correct,
            expected_prediction_sha256=(
                protocol.artifacts.healthy_validation_prediction_sha256
            ),
        )

        fault_seeds = (
            protocol.faults.development_seed,
            *protocol.faults.evaluation_seeds,
        )
        overlays: dict[int, PermanentFaultOverlay] = {}
        p0_by_seed: dict[int, torch.Tensor] = {}
        fault_artifact_paths = {}
        for seed in fault_seeds:
            overlay = sample_aihwkit_compatible_permanent_fault_overlay(
                population,
                mask_seed=seed,
                corrupt_probability=protocol.faults.probability,
                corrupt_raw_a_range=protocol.faults.corrupt_devices_range_raw_a,
            )
            if (
                overlay.pulse_population.dw_min_std != 0.0
                or overlay.pulse_population.write_noise_std != 0.0
            ):
                raise RuntimeError("Fault-only overlay retained programming/write noise.")
            p0 = _fault_p0_raw_a(healthy_a, overlay, device=device)
            fault_npz, fault_receipt = _save_fault_artifact(
                store,
                seed=seed,
                overlay=overlay,
                p0_raw_a=p0,
            )
            artifacts.extend(
                (
                    store.artifact_record(fault_npz, kind="permanent_fault_state"),
                    store.artifact_record(
                        fault_receipt, kind="permanent_fault_receipt"
                    ),
                )
            )
            overlays[seed] = overlay
            p0_by_seed[seed] = p0
            fault_artifact_paths[seed] = {
                "state": str(fault_npz),
                "state_sha256": sha256_file(fault_npz),
                "receipt": str(fault_receipt),
                "receipt_sha256": sha256_file(fault_receipt),
            }
            print(
                f"prepared AIHWKit-compatible fault mask seed={seed} "
                f"faults={overlay.receipt.realized_fault_count} "
                f"G_stuck=[{overlay.receipt.full_conductance_stuck_minimum:.6f},"
                f"{overlay.receipt.full_conductance_stuck_maximum:.6f}]",
                flush=True,
            )

        dev_seed = protocol.faults.development_seed
        dev_overlay = overlays[dev_seed]
        dev_p0 = p0_by_seed[dev_seed]
        _apply_flat_full_g(
            base_stack,
            dev_p0 + 1.0,
            population.binding_shapes,
        )
        faulted_dev_stack = _stack_with_objective(
            base_stack,
            arm="output_kl_only",
            hidden_weight=0.0,
            hidden_gain=relu_gain,
            hidden_temperature=relu_temperature,
            prototypes=prototypes,
            output_gain=protocol.execution.fixed_logit_gain,
        )
        faulted_dev_validation, _ = _evaluate(
            faulted_dev_stack,
            teacher,
            loaders.validation,
            prototypes=prototypes,
            relu_hidden_gain=relu_gain,
            relu_hidden_temperature=relu_temperature,
        )

        lambda_calibration = {}
        lambda_anchor = {"output_kl_only": 0.0}
        for arm in ARMS[1:]:
            _apply_flat_full_g(base_stack, dev_p0 + 1.0, population.binding_shapes)
            diagnostic_stack = _stack_with_objective(
                base_stack,
                arm=arm,
                hidden_weight=1.0,
                hidden_gain=relu_gain,
                hidden_temperature=relu_temperature,
                prototypes=prototypes,
                output_gain=protocol.execution.fixed_logit_gain,
            )
            report = _component_gradient_report(
                diagnostic_stack,
                teacher,
                screen_loader,
                maximum_batches=None,
            )
            anchor = _lambda_anchor(report)
            lambda_calibration[arm] = report
            lambda_anchor[arm] = anchor
            store.append_metric(
                {
                    "mode": "lambda_calibration",
                    "arm": arm,
                    "fault_seed": dev_seed,
                    "lambda_anchor": anchor,
                    "gradient": report,
                    "test_opened": False,
                }
            )

        continuous_screen = []
        screen_batches = math.ceil(
            protocol.training.continuous_screen_examples / protocol.training.batch_size
        )
        for arm in ARMS[1:]:
            for multiplier in protocol.objectives.lambda_multipliers:
                stack = _stack_with_objective(
                    base_stack,
                    arm=arm,
                    hidden_weight=lambda_anchor[arm] * multiplier,
                    hidden_gain=relu_gain,
                    hidden_temperature=relu_temperature,
                    prototypes=prototypes,
                    output_gain=protocol.execution.fixed_logit_gain,
                )
                final_g, train_report = _train_continuous(
                    stack=stack,
                    teacher=teacher,
                    overlay=dev_overlay,
                    p0_raw_a=dev_p0,
                    loader=screen_loader,
                    learning_rate_raw_x=protocol.training.learning_rate_raw_x,
                    hidden_weight=lambda_anchor[arm] * multiplier,
                    maximum_batches=screen_batches,
                )
                _apply_flat_full_g(stack, final_g, population.binding_shapes)
                validation, _ = _evaluate(
                    stack,
                    teacher,
                    loaders.validation,
                    prototypes=prototypes,
                    relu_hidden_gain=relu_gain,
                    relu_hidden_temperature=relu_temperature,
                )
                row = {
                    "arm": arm,
                    "fault_seed": dev_seed,
                    "lambda_anchor": lambda_anchor[arm],
                    "lambda_multiplier": multiplier,
                    "hidden_weight": lambda_anchor[arm] * multiplier,
                    "training": train_report,
                    "validation": validation,
                    "test_opened": False,
                }
                continuous_screen.append(row)
                store.append_metric({"mode": "continuous_screen", **row})
                print(
                    f"continuous screen arm={arm} multiplier={multiplier:g} "
                    f"validation={100.0*validation['student_accuracy']:.2f}% "
                    "test=sealed",
                    flush=True,
                )

        selected = {"output_kl_only": {"lambda_multiplier": 0.0, "hidden_weight": 0.0}}
        for arm in ARMS[1:]:
            candidates = [row for row in continuous_screen if row["arm"] == arm]
            winner = min(candidates, key=_selection_key)
            selected[arm] = {
                "lambda_anchor": lambda_anchor[arm],
                "lambda_multiplier": winner["lambda_multiplier"],
                "hidden_weight": winner["hidden_weight"],
                "development_validation": winner["validation"],
            }
        settings_frozen_before_test = True

        continuous_heldout = []
        heldout_seed = protocol.faults.evaluation_seeds[0]
        for arm in ARMS:
            overlay = overlays[heldout_seed]
            p0 = p0_by_seed[heldout_seed]
            stack = _stack_with_objective(
                base_stack,
                arm=arm,
                hidden_weight=selected[arm]["hidden_weight"],
                hidden_gain=relu_gain,
                hidden_temperature=relu_temperature,
                prototypes=prototypes,
                output_gain=protocol.execution.fixed_logit_gain,
            )
            final_g, train_report = _train_continuous(
                stack=stack,
                teacher=teacher,
                overlay=overlay,
                p0_raw_a=p0,
                loader=screen_loader,
                learning_rate_raw_x=protocol.training.learning_rate_raw_x,
                hidden_weight=selected[arm]["hidden_weight"],
                maximum_batches=screen_batches,
            )
            _apply_flat_full_g(stack, final_g, population.binding_shapes)
            validation, _ = _evaluate(
                stack,
                teacher,
                loaders.validation,
                prototypes=prototypes,
                relu_hidden_gain=relu_gain,
                relu_hidden_temperature=relu_temperature,
            )
            row = {
                "arm": arm,
                "fault_seed": heldout_seed,
                "training": train_report,
                "validation": validation,
                "test_opened": False,
            }
            continuous_heldout.append(row)
            store.append_metric({"mode": "continuous_heldout_diagnostic", **row})

        initial_train_generator_state = loaders.train_generator.get_state().clone()
        common_pulse_selection_seed_by_fault = {
            seed: derive_seed(seed, "star_hidden_kd_common_bernoulli_pulses")
            for seed in fault_seeds
        }

        # Development pulse arms provide pulse-count comparability only.  Test
        # remains sealed and all hidden-objective settings are already frozen.
        development_pulse = []
        for arm in ARMS:
            stack = _stack_with_objective(
                base_stack,
                arm=arm,
                hidden_weight=selected[arm]["hidden_weight"],
                hidden_gain=relu_gain,
                hidden_temperature=relu_temperature,
                prototypes=prototypes,
                output_gain=protocol.execution.fixed_logit_gain,
            )
            plant, optimizer, train_report = _train_pulse(
                name=arm,
                stack=stack,
                teacher=teacher,
                population=population,
                overlay=dev_overlay,
                p0_raw_a=dev_p0,
                fault_seed=dev_seed,
                loader=loaders,
                train_generator_state=initial_train_generator_state,
                learning_rate_raw_x=protocol.training.learning_rate_raw_x,
                hidden_weight=selected[arm]["hidden_weight"],
                pulse_selection_seed=common_pulse_selection_seed_by_fault[dev_seed],
                protocol=protocol,
            )
            sync_catalog_from_persistent(
                stack.bundle.catalog, plant, binding_shapes=population.binding_shapes
            )
            validation, _ = _evaluate(
                stack,
                teacher,
                loaders.validation,
                prototypes=prototypes,
                relu_hidden_gain=relu_gain,
                relu_hidden_temperature=relu_temperature,
            )
            row = {
                "arm": arm,
                "fault_seed": dev_seed,
                "training": train_report,
                "validation": validation,
                "test_opened": False,
            }
            development_pulse.append(row)
            store.append_metric({"mode": "pulse_development", **row})
            del plant, optimizer
            gc.collect()
            torch.cuda.empty_cache()

        output_attempts = next(
            row["training"]["write_ledger"]["attempted_pulse_requests"]
            for row in development_pulse
            if row["arm"] == "output_kl_only"
        )
        pulse_mismatch = {}
        for row in development_pulse:
            if row["arm"] == "output_kl_only":
                continue
            attempts = row["training"]["write_ledger"]["attempted_pulse_requests"]
            pulse_mismatch[row["arm"]] = _relative_pulse_mismatch(
                attempts, output_attempts
            )
        needs_pulse_matched_control = any(
            value > protocol.training.pulse_count_mismatch_fraction
            for value in pulse_mismatch.values()
        )
        pulse_matched_lr_by_arm: dict[str, float] = {}
        pulse_match_development = []
        if needs_pulse_matched_control:
            lr_attempts = {}
            for learning_rate in protocol.training.output_only_diagnostic_learning_rates_raw_x:
                stack = _stack_with_objective(
                    base_stack,
                    arm="output_kl_only",
                    hidden_weight=0.0,
                    hidden_gain=relu_gain,
                    hidden_temperature=relu_temperature,
                    prototypes=prototypes,
                    output_gain=protocol.execution.fixed_logit_gain,
                )
                plant, optimizer, train_report = _train_pulse(
                    name=f"output_kl_only_lr_{learning_rate:g}",
                    stack=stack,
                    teacher=teacher,
                    population=population,
                    overlay=dev_overlay,
                    p0_raw_a=dev_p0,
                    fault_seed=dev_seed,
                    loader=loaders,
                    train_generator_state=initial_train_generator_state,
                    learning_rate_raw_x=learning_rate,
                    hidden_weight=0.0,
                    pulse_selection_seed=common_pulse_selection_seed_by_fault[dev_seed],
                    protocol=protocol,
                )
                attempts = train_report["write_ledger"]["attempted_pulse_requests"]
                lr_attempts[learning_rate] = attempts
                pulse_match_development.append(
                    {
                        "learning_rate_raw_x": learning_rate,
                        "attempted_pulses": attempts,
                        "accuracy_not_used_for_selection": True,
                        "test_opened": False,
                    }
                )
                del plant, optimizer
                gc.collect()
                torch.cuda.empty_cache()
            for row in development_pulse:
                arm = row["arm"]
                if arm == "output_kl_only" or pulse_mismatch.get(arm, 0.0) <= (
                    protocol.training.pulse_count_mismatch_fraction
                ):
                    continue
                target = row["training"]["write_ledger"]["attempted_pulse_requests"]
                pulse_matched_lr_by_arm[arm] = min(
                    lr_attempts,
                    key=lambda lr: (abs(lr_attempts[lr] - target), abs(lr - 3e-5), lr),
                )

        evaluation_rows = []
        matched_control_rows = []
        for seed in protocol.faults.evaluation_seeds:
            overlay = overlays[seed]
            p0 = p0_by_seed[seed]
            _apply_flat_full_g(base_stack, p0 + 1.0, population.binding_shapes)
            faulted_stack = _stack_with_objective(
                base_stack,
                arm="output_kl_only",
                hidden_weight=0.0,
                hidden_gain=relu_gain,
                hidden_temperature=relu_temperature,
                prototypes=prototypes,
                output_gain=protocol.execution.fixed_logit_gain,
            )
            faulted_test, _ = _evaluate(
                faulted_stack,
                teacher,
                loaders.test,
                prototypes=prototypes,
                relu_hidden_gain=relu_gain,
                relu_hidden_temperature=relu_temperature,
            )
            for arm in ARMS:
                # Every matched arm and its P0 gradient diagnostic starts
                # from the exact same post-fault conductance state.  The
                # shared catalog may still contain the previous arm's final
                # state, so restore it explicitly before any computation.
                _apply_flat_full_g(base_stack, p0 + 1.0, population.binding_shapes)
                stack = _stack_with_objective(
                    base_stack,
                    arm=arm,
                    hidden_weight=selected[arm]["hidden_weight"],
                    hidden_gain=relu_gain,
                    hidden_temperature=relu_temperature,
                    prototypes=prototypes,
                    output_gain=protocol.execution.fixed_logit_gain,
                )
                p0_gradient = _component_gradient_report(
                    stack,
                    teacher,
                    screen_loader,
                    maximum_batches=1,
                )
                plant, optimizer, train_report = _train_pulse(
                    name=f"fault_{seed}_{arm}",
                    stack=stack,
                    teacher=teacher,
                    population=population,
                    overlay=overlay,
                    p0_raw_a=p0,
                    fault_seed=seed,
                    loader=loaders,
                    train_generator_state=initial_train_generator_state,
                    learning_rate_raw_x=protocol.training.learning_rate_raw_x,
                    hidden_weight=selected[arm]["hidden_weight"],
                    pulse_selection_seed=common_pulse_selection_seed_by_fault[seed],
                    protocol=protocol,
                )
                sync_catalog_from_persistent(
                    stack.bundle.catalog,
                    plant,
                    binding_shapes=population.binding_shapes,
                )
                validation, _ = _evaluate(
                    stack,
                    teacher,
                    loaders.validation,
                    prototypes=prototypes,
                    relu_hidden_gain=relu_gain,
                    relu_hidden_temperature=relu_temperature,
                )
                test, _ = _evaluate(
                    stack,
                    teacher,
                    loaders.test,
                    prototypes=prototypes,
                    relu_hidden_gain=relu_gain,
                    relu_hidden_temperature=relu_temperature,
                )
                final_gradient = _component_gradient_report(
                    stack,
                    teacher,
                    screen_loader,
                    maximum_batches=1,
                )
                row = {
                    "fault_seed": seed,
                    "arm": arm,
                    "faulted_p0_test": faulted_test,
                    "training": train_report,
                    "validation": validation,
                    "test": test,
                    "gradient_locality": {
                        "p0": p0_gradient,
                        "final": final_gradient,
                    },
                    "test_opened_after_settings_frozen": settings_frozen_before_test,
                }
                checkpoint = atomic_torch_save(
                    {
                        "schema": (
                            "ebl.mnist_relu_drn.star_inspired_hidden_kd_"
                            "fault_recovery_state"
                        ),
                        "schema_version": 1,
                        "evidence_tier": EVIDENCE_TIER,
                        "claim_label": CLAIM_LABEL,
                        "fault_seed": seed,
                        "arm": arm,
                        "fault_receipt": overlay.receipt.as_dict(),
                        "plant_continuation_state": plant.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "selected_hidden_weight": selected[arm]["hidden_weight"],
                        "metrics": row,
                        "full_conductance_nonnegative": True,
                        "program_verify": False,
                        "verify_reads": 0,
                    },
                    store.run_dir / f"checkpoints/fault_{seed}_{arm}.pt",
                )
                row["checkpoint"] = str(checkpoint)
                row["checkpoint_sha256"] = sha256_file(checkpoint)
                artifacts.append(
                    store.artifact_record(checkpoint, kind="fault_recovery_state")
                )
                evaluation_rows.append(row)
                store.append_metric({"mode": "pulse_evaluation", **row})
                print(
                    f"pulse recovery seed={seed} arm={arm} "
                    f"test={100.0*test['student_accuracy']:.2f}% "
                    f"attempts={train_report['write_ledger']['attempted_pulse_requests']} "
                    "verify=0",
                    flush=True,
                )
                del plant, optimizer
                gc.collect()
                torch.cuda.empty_cache()

            for compared_arm, learning_rate in pulse_matched_lr_by_arm.items():
                stack = _stack_with_objective(
                    base_stack,
                    arm="output_kl_only",
                    hidden_weight=0.0,
                    hidden_gain=relu_gain,
                    hidden_temperature=relu_temperature,
                    prototypes=prototypes,
                    output_gain=protocol.execution.fixed_logit_gain,
                )
                plant, optimizer, train_report = _train_pulse(
                    name=f"fault_{seed}_pulse_match_for_{compared_arm}",
                    stack=stack,
                    teacher=teacher,
                    population=population,
                    overlay=overlay,
                    p0_raw_a=p0,
                    fault_seed=seed,
                    loader=loaders,
                    train_generator_state=initial_train_generator_state,
                    learning_rate_raw_x=learning_rate,
                    hidden_weight=0.0,
                    pulse_selection_seed=common_pulse_selection_seed_by_fault[seed],
                    protocol=protocol,
                )
                sync_catalog_from_persistent(
                    stack.bundle.catalog,
                    plant,
                    binding_shapes=population.binding_shapes,
                )
                test, _ = _evaluate(
                    stack,
                    teacher,
                    loaders.test,
                    prototypes=prototypes,
                    relu_hidden_gain=relu_gain,
                    relu_hidden_temperature=relu_temperature,
                )
                control = {
                    "fault_seed": seed,
                    "control_for_arm": compared_arm,
                    "learning_rate_raw_x": learning_rate,
                    "training": train_report,
                    "test": test,
                    "selection_used_accuracy": False,
                }
                matched_control_rows.append(control)
                store.append_metric({"mode": "pulse_count_matched_control", **control})
                del plant, optimizer
                gc.collect()
                torch.cuda.empty_cache()

        _apply_flat_full_g(healthy_stack, healthy_g, population.binding_shapes)
        healthy_test, _ = _evaluate(
            healthy_stack,
            teacher,
            loaders.test,
            prototypes=prototypes,
            relu_hidden_gain=relu_gain,
            relu_hidden_temperature=relu_temperature,
        )
        _assert_evaluation_identity(
            healthy_test,
            split="test",
            expected_examples=10_000,
            expected_correct=protocol.artifacts.healthy_test_correct,
            expected_prediction_sha256=(
                protocol.artifacts.healthy_test_prediction_sha256
            ),
        )
        aggregates = _summaries_by_arm(evaluation_rows)
        summary = {
            "schema": SUMMARY_SCHEMA,
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "evidence_tier": EVIDENCE_TIER,
            "claim_label": CLAIM_LABEL,
            "claim_boundary": {
                "exact_star": False,
                "local_learning": False,
                "hidden_error_location": "DRN hidden state",
                "credit_assignment": "global BPTT through reciprocal equilibrium",
                "optimizer_moments": "digital Adam",
                "device_model": "analyst-Winsorized AIHWKit 1.1 IBM OM",
                "array_identities": 1,
                "independent_fault_masks": len(protocol.faults.evaluation_seeds),
            },
            "source": {
                "teacher_metadata": teacher_metadata,
                "initializer": initializer_report,
                "population_fingerprint": population.fingerprint,
                "target_assignment_seed": protocol.target_assignment_seed,
                "initialization": (
                    "frozen_ReLU_derived_ideal_quantized_full_G_without_PV_QAT_or_Adam"
                ),
                "healthy_validation": healthy_validation,
                "healthy_test": healthy_test,
            },
            "coordinate_contract": {
                "AIHWKit_device_coordinate": "raw_a in [-1,1]",
                "DRN_circuit_conductance": "G=raw_a+1=2*x in [0,2]",
                "negative_conductance_allowed": False,
            },
            "fault_contract": {
                "policy": protocol.faults.policy,
                "probability": protocol.faults.probability,
                "corrupt_devices_range_raw_a": (
                    protocol.faults.corrupt_devices_range_raw_a
                ),
                "stuck_conductance_nominal_interval": [0.99, 1.01],
                "development_seed": dev_seed,
                "evaluation_seeds": list(protocol.faults.evaluation_seeds),
                "receipts": {
                    str(seed): overlays[seed].receipt.as_dict() for seed in fault_seeds
                },
                "artifacts": {
                    str(seed): fault_artifact_paths[seed] for seed in fault_seeds
                },
                "mask_exposed_to_learner": False,
            },
            "noise_separation": {
                "program_verify": False,
                "verify_reads": 0,
                "cycle_to_cycle_noise": False,
                "apparent_write_noise": False,
                "inference_read_noise": False,
                "stochasticity": "Bernoulli open-loop pulse selection only",
            },
            "hidden_targets": {
                "prototype_artifact": str(prototype_npz),
                "prototype_artifact_sha256": sha256_file(prototype_npz),
                "train_split_indices_sha256": train_split_sha256,
                "hidden_calibration": hidden_calibration,
            },
            "development": {
                "faulted_validation": faulted_dev_validation,
                "lambda_calibration": lambda_calibration,
                "continuous_screen": continuous_screen,
                "selected": selected,
                "continuous_heldout_91502": continuous_heldout,
                "pulse": development_pulse,
                "pulse_mismatch_fraction": pulse_mismatch,
                "pulse_count_matched_control_required": needs_pulse_matched_control,
                "pulse_match_development": pulse_match_development,
                "test_opened": False,
            },
            "evaluation": {
                "settings_frozen_before_test": settings_frozen_before_test,
                "rows": evaluation_rows,
                "pulse_count_matched_controls": matched_control_rows,
                "aggregates": aggregates,
            },
            "success_criteria": {
                "mean_test_gain_vs_output_only_minimum": 0.02,
                "paired_mask_wins_minimum": 3,
                "corresponding_hidden_metric_reduction_minimum": 0.20,
                "zero_stuck_movement_required": True,
                "zero_verify_reads_required": True,
            },
            "limitations": [
                "one model-based Winsorized array identity",
                "four independent synthetic permanent-fault masks",
                "full BPTT and digital teacher/Adam state",
                "AIHWKit-compatible fault shape but programming/read noise disabled",
                "exploratory noncanonical dirty-worktree execution",
            ],
        }
        summary_path = store.run_dir / "artifacts/scientific_summary.json"
        atomic_write_json(summary_path, summary)
        artifacts.append(store.artifact_record(summary_path, kind="scientific_summary"))
        result_path = store.complete(
            metrics={
                "healthy_test_accuracy": healthy_test["student_accuracy"],
                "faulted_test_accuracy_mean": sum(
                    row["faulted_p0_test"]["student_accuracy"]
                    for row in evaluation_rows
                    if row["arm"] == "output_kl_only"
                )
                / len(protocol.faults.evaluation_seeds),
                "output_only_test_accuracy_mean": aggregates["output_kl_only"][
                    "mean_test_accuracy"
                ],
                "relu_sample_hidden_kl_test_accuracy_mean": aggregates[
                    "relu_sample_hidden_kl"
                ]["mean_test_accuracy"],
                "healthy_class_hidden_kl_test_accuracy_mean": aggregates[
                    "healthy_class_hidden_kl"
                ]["mean_test_accuracy"],
                "healthy_class_hidden_mse_test_accuracy_mean": aggregates[
                    "healthy_class_hidden_mse"
                ]["mean_test_accuracy"],
                "verify_reads": 0,
                "negative_conductance_count": 0,
            },
            artifacts=artifacts,
        )
        print(f"run_dir={store.run_dir}", flush=True)
        print(f"result={result_path}", flush=True)
        return 0
    except BaseException as error:
        store.fail(error)
        raise


__all__ = [
    "ARMS",
    "CLAIM_LABEL",
    "EVIDENCE_TIER",
    "SUMMARY_SCHEMA",
    "SUMMARY_SCHEMA_VERSION",
    "run_train",
]
