"""CUDA runtime for a short IBM-OM 784-256-128-10 pulse-Adam check.

The network forward always consumes the held apparent effective device state
``q=a-r``.  Adam moments are digital, but every selected update is issued as
one open-loop OM pulse against the hidden persistent state; the plant then
refreshes the apparent state of exactly the cells for which a write was
attempted.  The matched FP32 control uses the same initialization, minibatch
order, architecture, input range, objective, and five-epoch budget.
"""

from __future__ import annotations

from hashlib import sha256
import math
import os
from pathlib import Path
from typing import Any, Iterable, Sequence, TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from experiments.artifacts import RunStore, atomic_write_json, sha256_file
from experiments.mnist_analog_relu.runtime import (
    _float_summary,
    _population_defect_report,
    _program_endpoint,
    _sample_population,
)
from experiments.mnist_om_3fc_adam.config import ThreeFcAdamSpec
from experiments.mnist_relu.model import BiasFreeReluTeacher
from experiments.schema import to_plain_data
from training.checkpoint import atomic_torch_save, load_named_weights
from training.ibm_om_standard_crossbar import (
    CrossbarTileSpec,
    IbmOmEffectiveCrossbarPlant,
    tensor_sha256,
)

if TYPE_CHECKING:
    from ebl.cli import TrainRequest


_ROOT = Path(__file__).resolve().parents[2]
_TEACHER_MEAN = 0.1307
_TEACHER_STD = 0.3


class _FlattenedDataset(Dataset):
    def __init__(self, source: Dataset) -> None:
        self.source = source

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image, label = self.source[index]
        return image.reshape(-1), int(label)


def _input(role: str, path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Expected {role} to identify an existing file. Provided value: {resolved!s}."
        )
    return {"role": role, "path": str(resolved), "sha256": sha256_file(resolved)}


def _prediction_digest(values: list[torch.Tensor]) -> str:
    joined = torch.cat(values).detach().cpu().to(torch.int64).contiguous()
    return sha256(joined.numpy().tobytes()).hexdigest()


def _mnist_dataset() -> tuple[Dataset, Dataset]:
    root = Path(
        os.environ.get("EBL_MNIST_ROOT", str(Path.home() / "datasets" / "mnist"))
    ).expanduser()
    transform = transforms.ToTensor()
    try:
        train = datasets.MNIST(
            root=str(root), train=True, download=False, transform=transform
        )
        test = datasets.MNIST(
            root=str(root), train=False, download=False, transform=transform
        )
    except RuntimeError as error:
        raise RuntimeError(
            "Expected an existing torchvision MNIST dataset with raw train/test "
            f"files below {root!s}."
        ) from error
    return _FlattenedDataset(train), _FlattenedDataset(test)


def _build_loaders(
    *,
    train_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int,
    data_seed: int,
) -> tuple[Iterable, Iterable]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(data_seed)
    train = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        drop_last=False,
    )
    test = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    return train, test


def _build_layout(
    dims: Sequence[int],
    *,
    bias: bool,
    maximum_input_size: int,
) -> tuple[CrossbarTileSpec, ...]:
    dimensions = tuple(int(value) for value in dims)
    if len(dimensions) != 4 or any(value < 1 for value in dimensions):
        raise ValueError("Expected positive 784-256-128-10 dimensions.")
    result: list[CrossbarTileSpec] = []
    for layer_index, (logical_inputs, outputs) in enumerate(
        zip(dimensions[:-1], dimensions[1:], strict=True)
    ):
        physical_inputs = logical_inputs + int(bias)
        tile_count = math.ceil(physical_inputs / maximum_input_size)
        base, remainder = divmod(physical_inputs, tile_count)
        start = 0
        for tile_index in range(tile_count):
            width = base + int(tile_index < remainder)
            stop = start + width
            result.append(
                CrossbarTileSpec(
                    key=f"crossbar.layer{layer_index}.tile{tile_index}",
                    layer_index=layer_index,
                    tile_index=tile_index,
                    input_start=start,
                    input_stop=stop,
                    out_features=outputs,
                )
            )
            start = stop
    return tuple(result)


def _aihwkit_default_matrices(
    dims: Sequence[int],
    *,
    bias: bool,
    seed: int,
) -> tuple[torch.Tensor, ...]:
    """Reproduce AnalogLinear/PyTorch Linear initialization and draw order."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    matrices = []
    for inputs, outputs in zip(dims[:-1], dims[1:], strict=True):
        native_weight = torch.empty((outputs, inputs), dtype=torch.float32)
        torch.nn.init.kaiming_uniform_(
            native_weight,
            a=math.sqrt(5.0),
            generator=generator,
        )
        logical = native_weight.transpose(0, 1).contiguous()
        if bias:
            bound = 1.0 / math.sqrt(inputs)
            native_bias = torch.empty((outputs,), dtype=torch.float32)
            torch.nn.init.uniform_(
                native_bias,
                -bound,
                bound,
                generator=generator,
            )
            logical = torch.cat((logical, native_bias.unsqueeze(0)), dim=0)
        matrices.append(logical)
    return tuple(matrices)


def _flatten_matrices(
    matrices: Sequence[torch.Tensor],
    layout: Sequence[CrossbarTileSpec],
) -> torch.Tensor:
    values = tuple(matrices)
    pieces = []
    for tile in layout:
        matrix = values[tile.layer_index]
        pieces.append(
            matrix[tile.input_start : tile.input_stop].contiguous().reshape(-1)
        )
    return torch.cat(pieces)


def _matrices_from_state(
    state: torch.Tensor,
    layout: Sequence[CrossbarTileSpec],
) -> tuple[torch.Tensor, ...]:
    flat = state.reshape(-1)
    expected = sum(tile.cells for tile in layout)
    if flat.numel() != expected:
        raise ValueError("Expected one effective q state for every crosspoint.")
    layer_count = max(tile.layer_index for tile in layout) + 1
    pieces: list[list[torch.Tensor]] = [[] for _ in range(layer_count)]
    offset = 0
    for tile in layout:
        stop = offset + tile.cells
        pieces[tile.layer_index].append(flat[offset:stop].reshape(tile.shape))
        offset = stop
    return tuple(torch.cat(layer, dim=0) for layer in pieces)


def _forward_matrices(
    inputs: torch.Tensor,
    matrices: Sequence[torch.Tensor],
    *,
    bias: bool,
) -> torch.Tensor:
    value = inputs
    values = tuple(matrices)
    for layer_index, matrix in enumerate(values):
        if bias:
            value = torch.cat(
                (
                    value,
                    torch.ones(
                        (value.shape[0], 1),
                        device=value.device,
                        dtype=value.dtype,
                    ),
                ),
                dim=1,
            )
        value = value @ matrix
        if layer_index + 1 < len(values):
            value = torch.sigmoid(value)
    return value


def _forward_state(
    inputs: torch.Tensor,
    state: torch.Tensor,
    layout: Sequence[CrossbarTileSpec],
    *,
    bias: bool,
) -> torch.Tensor:
    return _forward_matrices(
        inputs,
        _matrices_from_state(state, layout),
        bias=bias,
    )


def _load_teacher(
    path: Path,
    *,
    expected_sha256: str,
    device: torch.device,
) -> BiasFreeReluTeacher:
    if sha256_file(path) != expected_sha256:
        raise ValueError("Expected the frozen teacher checkpoint SHA-256 to match.")
    teacher = BiasFreeReluTeacher(device=device)
    loaded = load_named_weights(path, teacher.catalog)
    if loaded.metadata.get("architecture") != "bias_free_relu_784_50_10":
        raise ValueError("Expected the frozen bias-free 784-50-10 ReLU teacher.")
    teacher.eval()
    return teacher


@torch.no_grad()
def _evaluate_matrices(
    *,
    matrices: Sequence[torch.Tensor],
    teacher: BiasFreeReluTeacher,
    loader: Iterable,
    device: torch.device,
    bias: bool,
) -> dict[str, Any]:
    values = tuple(matrix.detach().to(device=device) for matrix in matrices)
    totals = {
        "student_correct": 0,
        "teacher_correct": 0,
        "agreement": 0,
        "kl": 0.0,
        "cross_entropy": 0.0,
    }
    examples = 0
    student_predictions: list[torch.Tensor] = []
    teacher_predictions: list[torch.Tensor] = []
    for inputs, labels in loader:
        raw_inputs = inputs.to(device=device, dtype=torch.float32)
        labels = labels.to(device=device, dtype=torch.int64)
        student_logits = _forward_matrices(raw_inputs, values, bias=bias)
        teacher_inputs = (raw_inputs - _TEACHER_MEAN) / _TEACHER_STD
        teacher_logits = teacher.logits(teacher_inputs)
        teacher_log_probability = F.log_softmax(teacher_logits, dim=1)
        teacher_probability = teacher_log_probability.exp()
        student_log_probability = F.log_softmax(student_logits, dim=1)
        student_prediction = student_logits.argmax(dim=1)
        teacher_prediction = teacher_logits.argmax(dim=1)
        totals["student_correct"] += int(student_prediction.eq(labels).sum().item())
        totals["teacher_correct"] += int(teacher_prediction.eq(labels).sum().item())
        totals["agreement"] += int(student_prediction.eq(teacher_prediction).sum().item())
        totals["kl"] += float(
            (
                teacher_probability
                * (teacher_log_probability - student_log_probability)
            )
            .sum()
            .item()
        )
        totals["cross_entropy"] += float(
            F.cross_entropy(student_logits, labels, reduction="sum").item()
        )
        examples += int(labels.numel())
        student_predictions.append(student_prediction.cpu())
        teacher_predictions.append(teacher_prediction.cpu())
    if examples == 0:
        raise RuntimeError("Expected evaluation to process MNIST examples.")
    return {
        "examples": examples,
        "student_accuracy": totals["student_correct"] / examples,
        "student_correct": totals["student_correct"],
        "cross_entropy": totals["cross_entropy"] / examples,
        "teacher_accuracy": totals["teacher_correct"] / examples,
        "teacher_agreement": totals["agreement"] / examples,
        "prediction_flips_from_teacher": examples - totals["agreement"],
        "kl_teacher_student": totals["kl"] / examples,
        "student_prediction_sha256": _prediction_digest(student_predictions),
        "teacher_prediction_sha256": _prediction_digest(teacher_predictions),
    }


def _evaluate_state(
    *,
    state: torch.Tensor,
    layout: Sequence[CrossbarTileSpec],
    teacher: BiasFreeReluTeacher,
    loader: Iterable,
    device: torch.device,
    bias: bool,
) -> dict[str, Any]:
    matrices = _matrices_from_state(state.detach().to(device=device), layout)
    return _evaluate_matrices(
        matrices=matrices,
        teacher=teacher,
        loader=loader,
        device=device,
        bias=bias,
    )


def _layer_slices(
    layout: Sequence[CrossbarTileSpec],
) -> tuple[slice, ...]:
    layer_count = max(tile.layer_index for tile in layout) + 1
    counts = [0 for _ in range(layer_count)]
    for tile in layout:
        counts[tile.layer_index] += tile.cells
    result = []
    offset = 0
    for count in counts:
        result.append(slice(offset, offset + count))
        offset += count
    return tuple(result)


class _PulseAdam:
    """Digital Adam moments converted to Bernoulli one-pulse OM commands."""

    def __init__(
        self,
        *,
        size: int,
        layout: Sequence[CrossbarTileSpec],
        learning_rates: Sequence[float],
        betas: tuple[float, float],
        epsilon: float,
        nominal_dw_min: float,
        maximum_updates_per_cell: int,
        generator: torch.Generator,
        device: torch.device,
    ) -> None:
        self.device = device
        self.generator = generator
        self.beta_1, self.beta_2 = betas
        self.epsilon = epsilon
        self.nominal_dw_min = nominal_dw_min
        self.maximum_updates_per_cell = maximum_updates_per_cell
        self.step_index = 0
        self.first_moment = torch.zeros(size, dtype=torch.float32, device=device)
        self.second_moment = torch.zeros_like(self.first_moment)
        self.pulse_count = torch.zeros(size, dtype=torch.int64, device=device)
        self.layer_slices = _layer_slices(layout)
        if len(learning_rates) != len(self.layer_slices):
            raise ValueError("Expected one Adam learning rate per physical layer.")
        self.learning_rate = torch.empty_like(self.first_moment)
        for layer_slice, rate in zip(
            self.layer_slices, learning_rates, strict=True
        ):
            self.learning_rate[layer_slice] = float(rate)
        self.requested = 0
        self.applied = 0
        self.probability_clipped = 0
        self.blocked_at_cap = 0

    @torch.no_grad()
    def step(
        self,
        gradient: torch.Tensor,
        plant: IbmOmEffectiveCrossbarPlant,
    ) -> dict[str, Any]:
        value = gradient.detach().to(device=self.device, dtype=torch.float32).reshape(-1)
        if value.shape != self.first_moment.shape or not bool(
            torch.all(torch.isfinite(value))
        ):
            raise ValueError("Expected one finite apparent-state gradient per cell.")
        self.step_index += 1
        self.first_moment.mul_(self.beta_1).add_(value, alpha=1.0 - self.beta_1)
        self.second_moment.mul_(self.beta_2).addcmul_(
            value, value, value=1.0 - self.beta_2
        )
        first_hat = self.first_moment / (1.0 - self.beta_1**self.step_index)
        second_hat = self.second_moment / (1.0 - self.beta_2**self.step_index)
        command = -self.learning_rate * first_hat / (
            second_hat.sqrt() + self.epsilon
        )
        raw_probability = command.abs() / self.nominal_dw_min
        probability = raw_probability.clamp(max=1.0)
        candidate = torch.rand(
            probability.shape,
            dtype=torch.float32,
            device=self.device,
            generator=self.generator,
        ) < probability
        at_cap = self.pulse_count >= self.maximum_updates_per_cell
        selected = candidate & ~at_cap
        direction = torch.sign(command).to(torch.int8) * selected.to(torch.int8)
        plant.pulse(direction)
        applied = int(selected.sum().item())
        self.pulse_count += selected.to(torch.int64)
        self.requested += int((raw_probability > 0.0).sum().item())
        self.applied += applied
        self.probability_clipped += int((raw_probability > 1.0).sum().item())
        self.blocked_at_cap += int((candidate & at_cap).sum().item())
        return {
            "applied_pulses": applied,
            "maximum_probability_before_clip": float(raw_probability.max().item()),
        }

    def report(self) -> dict[str, Any]:
        return {
            "optimizer_steps": self.step_index,
            "requested_nonzero_commands": self.requested,
            "commanded_pulses": self.applied,
            "commanded_cells": int((self.pulse_count > 0).sum().item()),
            "commanded_pulses_by_layer": [
                int(self.pulse_count[value].sum().item())
                for value in self.layer_slices
            ],
            "maximum_pulses_per_cell": int(self.pulse_count.max().item()),
            "cells_at_structural_cap": int(
                (self.pulse_count >= self.maximum_updates_per_cell).sum().item()
            ),
            "structural_cap_one_pulse_per_optimizer_step": (
                self.maximum_updates_per_cell
            ),
            "probability_clipped": self.probability_clipped,
            "blocked_at_cap": self.blocked_at_cap,
            "digital_first_moment_values": self.first_moment.numel(),
            "digital_second_moment_values": self.second_moment.numel(),
        }


def _train_digital_control(
    *,
    initial_matrices: Sequence[torch.Tensor],
    train_loader: Iterable,
    test_loader: Iterable,
    teacher: BiasFreeReluTeacher,
    spec: ThreeFcAdamSpec,
    device: torch.device,
    store: RunStore,
) -> tuple[list[dict[str, Any]], tuple[torch.Tensor, ...]]:
    parameters = [
        torch.nn.Parameter(value.detach().to(device=device).clone())
        for value in initial_matrices
    ]
    optimizer = torch.optim.Adam(
        [
            {"params": [parameter], "lr": learning_rate}
            for parameter, learning_rate in zip(
                parameters,
                spec.optimizer.learning_rates_q,
                strict=True,
            )
        ],
        betas=(spec.optimizer.beta_1, spec.optimizer.beta_2),
        eps=spec.optimizer.epsilon,
    )
    epochs = []
    for epoch in range(1, spec.optimizer.epochs + 1):
        examples = 0
        correct = 0
        loss_sum = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device, dtype=torch.int64)
            optimizer.zero_grad(set_to_none=True)
            logits = _forward_matrices(inputs, parameters, bias=spec.model.bias)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            count = int(labels.numel())
            examples += count
            loss_sum += float(loss.item()) * count
            correct += int(logits.argmax(dim=1).eq(labels).sum().item())
        test = _evaluate_matrices(
            matrices=parameters,
            teacher=teacher,
            loader=test_loader,
            device=device,
            bias=spec.model.bias,
        )
        report = {
            "epoch": epoch,
            "examples": examples,
            "train_cross_entropy": loss_sum / examples,
            "train_pre_update_accuracy": correct / examples,
            "test": test,
        }
        epochs.append(report)
        store.append_metric({"mode": "digital_adam_epoch", **report})
    return epochs, tuple(value.detach().cpu() for value in parameters)


def _train_om(
    *,
    plant: IbmOmEffectiveCrossbarPlant,
    layout: Sequence[CrossbarTileSpec],
    train_loader: Iterable,
    test_loader: Iterable,
    teacher: BiasFreeReluTeacher,
    spec: ThreeFcAdamSpec,
    device: torch.device,
    store: RunStore,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pulse_generator = torch.Generator(device="cuda")
    pulse_generator.manual_seed(
        spec.runtime.seed + spec.device.assignment_seed + spec.device.endpoint_seed
    )
    optimizer = _PulseAdam(
        size=plant.size,
        layout=layout,
        learning_rates=spec.optimizer.learning_rates_q,
        betas=(spec.optimizer.beta_1, spec.optimizer.beta_2),
        epsilon=spec.optimizer.epsilon,
        nominal_dw_min=plant.population.nominal_dw_min,
        maximum_updates_per_cell=spec.optimizer.maximum_one_pulse_updates_per_cell,
        generator=pulse_generator,
        device=device,
    )
    epochs = []
    for epoch in range(1, spec.optimizer.epochs + 1):
        examples = 0
        correct = 0
        loss_sum = 0.0
        epoch_pulses = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device, dtype=torch.int64)
            apparent_q = plant.apparent.detach().requires_grad_(True)
            logits = _forward_state(
                inputs,
                apparent_q,
                layout,
                bias=spec.model.bias,
            )
            loss = F.cross_entropy(logits, labels)
            gradient = torch.autograd.grad(loss, apparent_q, only_inputs=True)[0]
            update = optimizer.step(gradient, plant)
            count = int(labels.numel())
            examples += count
            loss_sum += float(loss.item()) * count
            correct += int(logits.argmax(dim=1).eq(labels).sum().item())
            epoch_pulses += int(update["applied_pulses"])
        apparent_test = _evaluate_state(
            state=plant.apparent,
            layout=layout,
            teacher=teacher,
            loader=test_loader,
            device=device,
            bias=spec.model.bias,
        )
        persistent_test = _evaluate_state(
            state=plant.persistent,
            layout=layout,
            teacher=teacher,
            loader=test_loader,
            device=device,
            bias=spec.model.bias,
        )
        report = {
            "epoch": epoch,
            "examples": examples,
            "train_cross_entropy": loss_sum / examples,
            "train_pre_update_accuracy": correct / examples,
            "commanded_pulses": epoch_pulses,
            "apparent_test": apparent_test,
            "persistent_test_diagnostic": persistent_test,
            "apparent_q_sha256": tensor_sha256(plant.apparent),
            "persistent_q_sha256": tensor_sha256(plant.persistent),
        }
        epochs.append(report)
        store.append_metric({"mode": "om_pulse_adam_epoch", **report})
    return epochs, optimizer.report()


def _validate_request(request: Any, spec: ThreeFcAdamSpec) -> tuple[Path, Path]:
    for field in ("weights", "base_weights", "resume", "device_data", "device_model"):
        if getattr(request, field, None) is not None:
            raise ValueError(f"Expected this from-scratch experiment not to use --{field}.")
    if request.teacher_weights is None:
        raise ValueError("Expected an explicit --teacher-weights artifact for KL evaluation.")
    teacher = request.teacher_weights.expanduser().resolve()
    configured = Path(spec.evaluation.teacher_weights_path)
    if not configured.is_absolute():
        configured = (_ROOT / configured).resolve()
    if teacher != configured:
        raise ValueError("Expected --teacher-weights to match the frozen config path.")
    sampler_value = os.environ.get("EBL_AIHWKIT_PYTHON")
    if not sampler_value:
        raise RuntimeError(
            "Expected EBL_AIHWKIT_PYTHON to name the pinned AIHWKit 1.1.0 sampler."
        )
    sampler = Path(sampler_value).expanduser().resolve()
    if not sampler.is_file() or not os.access(sampler, os.X_OK):
        raise RuntimeError("Expected an executable AIHWKit sampling interpreter.")
    return teacher, sampler


def run_train(request: "TrainRequest") -> int:
    spec = request.spec
    if not isinstance(spec, ThreeFcAdamSpec):
        raise TypeError("Expected a ThreeFcAdamSpec train request.")
    teacher_path, sampler = _validate_request(request, spec)
    store = RunStore.create(
        output_root=request.output_dir,
        experiment_id=spec.experiment_id,
        resolved_config=to_plain_data(spec),
        command=request.command,
        repo_root=_ROOT,
        input_artifacts=(
            _input("teacher_weights", teacher_path),
            _input("aihwkit_python", sampler),
        ),
        resume_capability="unsupported",
    )
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("Expected CUDA before starting the training run.")
        torch.manual_seed(spec.runtime.seed)
        torch.cuda.manual_seed_all(spec.runtime.seed)
        device = torch.device("cuda", torch.cuda.current_device())
        probe = torch.ones((1,), dtype=torch.float32, device=device)
        if probe.device.type != "cuda":
            raise RuntimeError("Expected the resolved runtime tensor to be CUDA-resident.")

        teacher = _load_teacher(
            teacher_path,
            expected_sha256=spec.evaluation.teacher_weights_sha256,
            device=device,
        )
        train_dataset, test_dataset = _mnist_dataset()
        if len(train_dataset) != spec.data.train_examples or len(test_dataset) != spec.data.test_examples:
            raise RuntimeError("Expected the canonical 60,000/10,000 MNIST split.")
        digital_train, digital_test = _build_loaders(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            batch_size=spec.data.batch_size,
            data_seed=spec.runtime.data_seed,
        )
        om_train, om_test = _build_loaders(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            batch_size=spec.data.batch_size,
            data_seed=spec.runtime.data_seed,
        )

        layout = _build_layout(
            spec.model.dims,
            bias=spec.model.bias,
            maximum_input_size=spec.model.maximum_input_size,
        )
        initial_matrices = _aihwkit_default_matrices(
            spec.model.dims,
            bias=spec.model.bias,
            seed=spec.source.initialization_seed,
        )
        source_requested = _flatten_matrices(initial_matrices, layout)
        if bool(torch.any(source_requested < -1.0)) or bool(
            torch.any(source_requested > 1.0)
        ):
            raise RuntimeError("Expected every initial requested q value in [-1, 1].")

        digital_epochs, digital_final = _train_digital_control(
            initial_matrices=initial_matrices,
            train_loader=digital_train,
            test_loader=digital_test,
            teacher=teacher,
            spec=spec,
            device=device,
            store=store,
        )

        artifact_root = store.run_dir / "artifacts"
        artifact_root.mkdir(parents=True, exist_ok=True)
        population, population_receipt, population_artifacts = _sample_population(
            layout=layout,
            assignment_seed=spec.device.assignment_seed,
            corruption_policy=spec.device.corruption_policy,
            sampler=sampler,
            artifact_root=artifact_root,
            role="om_3fc",
            bound_policy=spec.device.bound_policy,
            required_aihwkit_version=spec.runtime.required_aihwkit_version,
        )
        plant, programming_report = _program_endpoint(
            population=population,
            requested=source_requested,
            assignment_seed=spec.device.assignment_seed,
            endpoint_seed=spec.device.endpoint_seed,
            maximum_pulses=spec.device.maximum_initial_programming_pulses,
            tolerance_x=spec.device.verify_tolerance_q / 2.0,
            device=device,
            stream_role="three_fc_from_scratch_initialization",
            random_stream_fingerprint=population.fingerprint,
        )
        initial_state_path = artifact_root / "initial_programmed_state.pt"
        atomic_torch_save(plant.state_dict(), initial_state_path)
        initial_apparent_test = _evaluate_state(
            state=plant.apparent,
            layout=layout,
            teacher=teacher,
            loader=om_test,
            device=device,
            bias=spec.model.bias,
        )
        initial_persistent_test = _evaluate_state(
            state=plant.persistent,
            layout=layout,
            teacher=teacher,
            loader=om_test,
            device=device,
            bias=spec.model.bias,
        )
        store.append_metric(
            {
                "mode": "om_programmed_initialization",
                "network_forward_state": "apparent_q",
                "hidden_update_state": "persistent_q",
                "apparent_test": initial_apparent_test,
                "persistent_test_diagnostic": initial_persistent_test,
            }
        )

        om_epochs, optimizer_report = _train_om(
            plant=plant,
            layout=layout,
            train_loader=om_train,
            test_loader=om_test,
            teacher=teacher,
            spec=spec,
            device=device,
            store=store,
        )
        torch.cuda.synchronize(device)
        if not torch.cuda.is_available() or plant.apparent.device.type != "cuda":
            raise RuntimeError("Expected CUDA to remain active through final evaluation.")

        digital_path = artifact_root / "digital_adam_final.pt"
        final_state_path = artifact_root / "om_pulse_adam_final_state.pt"
        atomic_torch_save(
            {
                "schema": "ebl.mnist_om_3fc_digital_adam",
                "schema_version": 1,
                "matrices": digital_final,
            },
            digital_path,
        )
        atomic_torch_save(plant.state_dict(), final_state_path)
        final_apparent = om_epochs[-1]["apparent_test"]
        final_persistent = om_epochs[-1]["persistent_test_diagnostic"]
        final_digital = digital_epochs[-1]["test"]
        summary = {
            "schema": "ebl.mnist_ibm_om_3fc_adam_summary",
            "schema_version": 1,
            "evidence_tier": "exploratory_noncanonical",
            "runtime": {
                "device": str(device),
                "cuda_device_name": torch.cuda.get_device_name(device),
                "torch_version": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "aihwkit_sampling_version": spec.runtime.required_aihwkit_version,
            },
            "architecture": {
                "dims": list(spec.model.dims),
                "bias": spec.model.bias,
                "hidden_activation": spec.model.hidden_activation,
                "forward": "OM_q_MVM-sigmoid-OM_q_MVM-sigmoid-OM_q_MVM",
                "logical_parameters": int(source_requested.numel()),
                "tile_layout": [
                    {"key": tile.key, "shape": list(tile.shape)} for tile in layout
                ],
            },
            "scientific_contract": {
                "epochs": spec.optimizer.epochs,
                "batch_size": spec.data.batch_size,
                "training_examples_per_epoch": spec.data.train_examples,
                "student_input_preprocessing": spec.data.preprocessing,
                "initial_requested_q_range": list(
                    spec.source.initial_requested_range
                ),
                "mapping": spec.model.weight_mapping,
                "network_forward_state": "held_apparent_q",
                "optimizer_target_state": "hidden_persistent_q_via_OM_pulses",
                "apparent_refresh": "touched_cells_after_every_write_attempt",
                "optimizer": (
                    "digital_Adam_moments_then_Bernoulli_one_pulse_OM_command"
                ),
                "learning_rates_q": list(spec.optimizer.learning_rates_q),
                "checkpoint_policy": spec.optimizer.checkpoint_policy,
                "corruption_policy": spec.device.corruption_policy,
                "published_corrupt_probability": (
                    spec.device.corrupt_devices_probability
                ),
                "reference_std": spec.device.reference_std,
                "aihwkit_AnalogAdam_note": (
                    "AIHWKit AnalogAdam applies Adam only to ordinary digital "
                    "parameters; analog contexts still call the tile update directly. "
                    "This experiment therefore implements explicit Adam moments "
                    "before persistent OM pulse commands."
                ),
            },
            "source": {
                "initialization": spec.source.initialization,
                "initialization_seed": spec.source.initialization_seed,
                "requested_q_sha256": tensor_sha256(source_requested),
                "requested_q": _float_summary(source_requested),
            },
            "device_population": {
                "fingerprint": population.fingerprint,
                "receipt": population_receipt,
                "defects": _population_defect_report(population),
            },
            "initial_programming": programming_report,
            "initial": {
                "apparent_test": initial_apparent_test,
                "persistent_test_diagnostic": initial_persistent_test,
            },
            "digital_adam_epochs": digital_epochs,
            "om_pulse_adam_epochs": om_epochs,
            "optimizer_report": optimizer_report,
            "headline": {
                "fixed_epoch": spec.optimizer.epochs,
                "apparent_accuracy": final_apparent["student_accuracy"],
                "apparent_kl_teacher_student": final_apparent[
                    "kl_teacher_student"
                ],
                "apparent_teacher_agreement": final_apparent[
                    "teacher_agreement"
                ],
                "persistent_accuracy_diagnostic": final_persistent[
                    "student_accuracy"
                ],
                "persistent_kl_teacher_student_diagnostic": final_persistent[
                    "kl_teacher_student"
                ],
                "matched_digital_accuracy": final_digital["student_accuracy"],
                "matched_digital_kl_teacher_student": final_digital[
                    "kl_teacher_student"
                ],
                "gap_to_matched_digital_percentage_points": 100.0
                * (
                    final_apparent["student_accuracy"]
                    - final_digital["student_accuracy"]
                ),
                "gap_to_ibm_ttv2_percentage_points": 100.0
                * (
                    final_apparent["student_accuracy"]
                    - spec.evaluation.ibm_ttv2_reference_accuracy
                ),
                "ibm_ttv2_reference_accuracy": (
                    spec.evaluation.ibm_ttv2_reference_accuracy
                ),
                "ibm_fp32_reference_accuracy": (
                    spec.evaluation.ibm_fp32_reference_accuracy
                ),
            },
            "limitations": [
                "One assignment and one initialization seed; IBM reports the median of five trials.",
                "Five Adam epochs are compared descriptively with IBM's 100-epoch TTv2 curve.",
                "This is hybrid digital-moment pulse-Adam, not TTv2 and not a claim of fully in-memory Adam.",
                "The device evidence is the model-based AIHWKit preset, not raw fabricated-array traces.",
                "The retained AIHWKit preset default has 5% reference variation; Fig. 13 of the IBM paper describes a 1% symmetry-point reference-write assumption.",
                "The frozen 784-50-10 ReLU teacher is evaluation-only and differs from the 3FC student architecture.",
            ],
        }
        summary_path = artifact_root / "summary.json"
        atomic_write_json(summary_path, summary)
        artifacts = [
            store.artifact_record(path, kind=kind)
            for path, kind in population_artifacts
        ]
        artifacts.extend(
            (
                store.artifact_record(
                    initial_state_path, kind="initial_programmed_om_state"
                ),
                store.artifact_record(
                    final_state_path, kind="final_persistent_and_apparent_om_state"
                ),
                store.artifact_record(
                    digital_path, kind="matched_digital_adam_checkpoint"
                ),
                store.artifact_record(summary_path, kind="scientific_summary"),
            )
        )
        store.complete(
            metrics={
                "evidence_tier": "exploratory_noncanonical",
                "network_forward_state": "apparent_q",
                "hidden_update_state": "persistent_q",
                "epochs": spec.optimizer.epochs,
                "assignment_seed": spec.device.assignment_seed,
                "endpoint_seed": spec.device.endpoint_seed,
                "headline": summary["headline"],
            },
            artifacts=artifacts,
        )
        return 0
    except BaseException as error:
        store.fail(error)
        raise


__all__ = [
    "_PulseAdam",
    "_aihwkit_default_matrices",
    "_build_layout",
    "_flatten_matrices",
    "_forward_matrices",
    "_matrices_from_state",
    "run_train",
]
