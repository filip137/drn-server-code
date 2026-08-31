"""STAR-inspired hidden-state objectives for global BPTT recovery.

This module deliberately implements a *state-augmented BPTT* control, not the
local centered-equilibrium-propagation STAR rule.  The auxiliary error is
defined at the DRN hidden layer, while :class:`training.sgd.Backprop` remains
responsible for differentiating the summed scalar through the complete
reciprocal network.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import torch
import torch.nn.functional as F

from experiments.mnist_relu_drn.components import TeacherKLDivergence
from model.function.interaction import Function


RELU_SAMPLE_HIDDEN_KL = "relu_sample_hidden_kl"
HEALTHY_CLASS_HIDDEN_KL = "healthy_class_hidden_kl"
HEALTHY_CLASS_HIDDEN_MSE = "healthy_class_hidden_mse"
HIDDEN_OBJECTIVES = frozenset(
    {
        RELU_SAMPLE_HIDDEN_KL,
        HEALTHY_CLASS_HIDDEN_KL,
        HEALTHY_CLASS_HIDDEN_MSE,
    }
)


def halves_scores(hidden: torch.Tensor) -> torch.Tensor:
    """Decode ``[positive-half, negative-half]`` physical hidden states."""

    if hidden.ndim != 2 or hidden.shape[1] % 2:
        raise ValueError(
            "Expected hidden state with shape (batch, 2 * hidden_width). "
            f"Provided value: {tuple(hidden.shape)!r}."
        )
    width = hidden.shape[1] // 2
    return hidden[:, :width] - hidden[:, width:]


def paired_scores(output: torch.Tensor) -> torch.Tensor:
    """Decode adjacent positive/negative physical output pairs."""

    if output.ndim != 2 or output.shape[1] % 2:
        raise ValueError(
            "Expected output state with shape (batch, 2 * classes). "
            f"Provided value: {tuple(output.shape)!r}."
        )
    return output[:, 0::2] - output[:, 1::2]


def teacher_to_student_kl(
    teacher_features: torch.Tensor,
    student_features: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Return per-example ``T^2 KL(teacher || student)``.

    The explicit argument order avoids the easy-to-miss reversal imposed by
    :func:`torch.nn.functional.kl_div`.
    """

    numeric_temperature = float(temperature)
    if not math.isfinite(numeric_temperature) or numeric_temperature <= 0.0:
        raise ValueError(
            "Expected temperature to be positive and finite. "
            f"Provided value: {temperature!r}."
        )
    if (
        teacher_features.ndim != 2
        or student_features.ndim != 2
        or teacher_features.shape != student_features.shape
    ):
        raise ValueError(
            "Expected matching rank-2 teacher and student features. "
            f"Provided value: teacher={tuple(teacher_features.shape)!r}, "
            f"student={tuple(student_features.shape)!r}."
        )

    teacher = teacher_features.detach().to(
        device=student_features.device,
        dtype=student_features.dtype,
    )
    teacher_log_prob = F.log_softmax(teacher / numeric_temperature, dim=1)
    teacher_prob = teacher_log_prob.exp()
    student_log_prob = F.log_softmax(
        student_features / numeric_temperature,
        dim=1,
    )
    return (
        numeric_temperature**2
        * (teacher_prob * (teacher_log_prob - student_log_prob)).sum(dim=1)
    )


def fit_positive_hidden_gain(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    *,
    temperature: float,
    gain_min: float,
    gain_max: float,
    steps: int,
) -> dict[str, float | bool]:
    """Fit one positive scalar by a deterministic logarithmic KL grid."""

    lower = float(gain_min)
    upper = float(gain_max)
    count = int(steps)
    numeric_temperature = float(temperature)
    if (
        not math.isfinite(lower)
        or not math.isfinite(upper)
        or lower <= 0.0
        or upper <= lower
        or count < 2
    ):
        raise ValueError(
            "Expected 0 < gain_min < gain_max and steps >= 2. Provided "
            f"value: gain_min={gain_min!r}, gain_max={gain_max!r}, "
            f"steps={steps!r}."
        )
    if not math.isfinite(numeric_temperature) or numeric_temperature <= 0.0:
        raise ValueError(
            "Expected temperature to be positive and finite. "
            f"Provided value: {temperature!r}."
        )
    if (
        student_hidden.ndim != 2
        or teacher_hidden.ndim != 2
        or student_hidden.shape != teacher_hidden.shape
        or student_hidden.numel() == 0
    ):
        raise ValueError(
            "Expected non-empty matching rank-2 hidden feature tensors. "
            f"Provided value: student={tuple(student_hidden.shape)!r}, "
            f"teacher={tuple(teacher_hidden.shape)!r}."
        )

    student = student_hidden.detach().to(torch.float64)
    teacher = teacher_hidden.detach().to(
        device=student.device,
        dtype=torch.float64,
    )
    if not bool(torch.isfinite(student).all()) or not bool(
        torch.isfinite(teacher).all()
    ):
        raise ValueError("Expected finite hidden features for gain fitting.")

    def mean_kl(gain: torch.Tensor) -> torch.Tensor:
        return teacher_to_student_kl(
            teacher,
            student * gain,
            temperature=numeric_temperature,
        ).mean()

    gains = torch.logspace(
        math.log10(lower),
        math.log10(upper),
        count,
        dtype=torch.float64,
        device=student.device,
    )
    losses = torch.stack([mean_kl(gain) for gain in gains])
    index = int(losses.argmin().item())
    selected = float(gains[index].item())
    unit_gain = torch.ones((), dtype=torch.float64, device=student.device)
    return {
        "gain": selected,
        "raw_kl": float(mean_kl(unit_gain).item()),
        "calibrated_kl": float(losses[index].item()),
        "student_hidden_rms": float(torch.sqrt(student.square().mean()).item()),
        "teacher_hidden_rms": float(torch.sqrt(teacher.square().mean()).item()),
        "temperature": numeric_temperature,
        "gain_at_grid_boundary": index in {0, count - 1},
    }


@dataclass(frozen=True)
class ClassPrototypeTable:
    """Healthy-DRN class means accumulated on CPU in float64."""

    raw_means: torch.Tensor
    decoded_means: torch.Tensor
    class_counts: torch.Tensor
    split: str
    examples: int


def accumulate_class_prototypes(
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    *,
    num_classes: int,
    split: str,
    expected_examples: int | None = None,
) -> ClassPrototypeTable:
    """Accumulate raw hidden-state class means with float64 CPU sums.

    ``split`` is an explicit guard against accidentally calibrating class
    targets on validation or test examples.  Each batch contains an already
    settled healthy DRN hidden state and its labels.
    """

    if split != "train":
        raise ValueError(
            "Expected healthy class prototypes to use split='train'. "
            f"Provided value: {split!r}."
        )
    classes = int(num_classes)
    if classes <= 1:
        raise ValueError(
            "Expected num_classes > 1. "
            f"Provided value: {num_classes!r}."
        )
    if expected_examples is not None and int(expected_examples) <= 0:
        raise ValueError(
            "Expected expected_examples to be positive when provided. "
            f"Provided value: {expected_examples!r}."
        )

    sums: torch.Tensor | None = None
    counts = torch.zeros(classes, dtype=torch.int64, device="cpu")
    total = 0
    for hidden, labels in batches:
        if hidden.ndim != 2 or hidden.shape[1] % 2:
            raise ValueError(
                "Expected every hidden-state batch to have shape "
                "(batch, 2 * hidden_width). Provided value: "
                f"{tuple(hidden.shape)!r}."
            )
        if labels.ndim != 1 or labels.shape[0] != hidden.shape[0]:
            raise ValueError(
                "Expected one rank-1 label per hidden state. Provided value: "
                f"hidden={tuple(hidden.shape)!r}, labels={tuple(labels.shape)!r}."
            )
        if labels.dtype.is_floating_point or labels.dtype == torch.bool:
            raise ValueError("Expected integer class labels for prototypes.")
        hidden64 = hidden.detach().to(
            device="cpu",
            dtype=torch.float64,
        )
        labels64 = labels.detach().to(device="cpu", dtype=torch.int64)
        if not bool(torch.isfinite(hidden64).all()):
            raise ValueError("Expected finite healthy hidden states.")
        if bool(((labels64 < 0) | (labels64 >= classes)).any()):
            raise ValueError(
                f"Expected labels in [0, {classes})."
            )
        if sums is None:
            sums = torch.zeros(
                (classes, hidden64.shape[1]),
                dtype=torch.float64,
                device="cpu",
            )
        elif hidden64.shape[1] != sums.shape[1]:
            raise ValueError(
                "Expected a fixed hidden width across prototype batches. "
                f"Provided value: first={sums.shape[1]}, "
                f"current={hidden64.shape[1]}."
            )
        sums.index_add_(0, labels64, hidden64)
        counts.add_(torch.bincount(labels64, minlength=classes))
        total += int(hidden64.shape[0])

    if sums is None or total == 0:
        raise ValueError("Expected at least one hidden-state batch.")
    if expected_examples is not None and total != int(expected_examples):
        raise ValueError(
            "Expected the declared number of prototype examples. "
            f"Provided value: observed={total}, expected={expected_examples}."
        )
    if bool((counts == 0).any()):
        missing = torch.nonzero(counts == 0, as_tuple=False).flatten().tolist()
        raise ValueError(
            "Expected at least one training example for every class. "
            f"Missing classes: {missing!r}."
        )

    raw = sums / counts.to(torch.float64).unsqueeze(1)
    return ClassPrototypeTable(
        raw_means=raw,
        decoded_means=halves_scores(raw),
        class_counts=counts,
        split=split,
        examples=total,
    )


class StarInspiredHiddenObjective(Function):
    """Output KD plus one hidden-state repair objective for global BPTT."""

    def __init__(
        self,
        hidden_layer,
        output_layer,
        *,
        hidden_objective: str,
        hidden_weight: float = 0.0,
        output_gain: float = 1.0,
        hidden_gain: float = 1.0,
        hidden_temperature: float = 1.0,
        class_prototypes: ClassPrototypeTable | torch.Tensor | None = None,
    ) -> None:
        if hidden_objective not in HIDDEN_OBJECTIVES:
            raise ValueError(
                "Expected a registered STAR-inspired hidden objective. "
                f"Provided value: {hidden_objective!r}."
            )
        if int(hidden_layer.shape[0]) % 2:
            raise ValueError(
                "Expected an even physical hidden width. "
                f"Provided value: {tuple(hidden_layer.shape)!r}."
            )
        if int(output_layer.shape[0]) % 2:
            raise ValueError(
                "Expected an even physical output width. "
                f"Provided value: {tuple(output_layer.shape)!r}."
            )

        self._hidden_layer = hidden_layer
        self._output_layer = output_layer
        self._hidden_objective = hidden_objective
        self._output = TeacherKLDivergence(output_layer, gain=output_gain)
        self._hidden_weight = 0.0
        self._hidden_gain = 1.0
        self._hidden_temperature = 1.0
        self._teacher_hidden: torch.Tensor | None = None
        self._labels: torch.Tensor | None = None
        self._class_prototypes: torch.Tensor | None = None
        self.hidden_weight = hidden_weight
        self.hidden_gain = hidden_gain
        self.hidden_temperature = hidden_temperature
        if class_prototypes is not None:
            self.set_class_prototypes(class_prototypes)
        Function.__init__(self, [hidden_layer, output_layer], [])

    @property
    def hidden_objective(self) -> str:
        return self._hidden_objective

    @property
    def gain(self) -> float:
        """Fixed output-logit gain, matching ``TeacherKLDivergence``."""

        return self._output.gain

    @gain.setter
    def gain(self, value: float) -> None:
        self._output.gain = value

    @property
    def hidden_weight(self) -> float:
        return self._hidden_weight

    @hidden_weight.setter
    def hidden_weight(self, value: float) -> None:
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < 0.0:
            raise ValueError(
                "Expected hidden weight to be finite and non-negative. "
                f"Provided value: {value!r}."
            )
        self._hidden_weight = numeric

    @property
    def hidden_gain(self) -> float:
        return self._hidden_gain

    @hidden_gain.setter
    def hidden_gain(self, value: float) -> None:
        numeric = float(value)
        if not math.isfinite(numeric) or numeric <= 0.0:
            raise ValueError(
                "Expected hidden gain to be positive and finite. "
                f"Provided value: {value!r}."
            )
        if (
            self._hidden_objective != RELU_SAMPLE_HIDDEN_KL
            and numeric != 1.0
        ):
            raise ValueError(
                "Expected class-prototype objectives to use fixed hidden "
                f"gain 1.0. Provided value: {value!r}."
            )
        self._hidden_gain = numeric

    @property
    def hidden_temperature(self) -> float:
        return self._hidden_temperature

    @hidden_temperature.setter
    def hidden_temperature(self, value: float) -> None:
        numeric = float(value)
        if not math.isfinite(numeric) or numeric <= 0.0:
            raise ValueError(
                "Expected hidden temperature to be positive and finite. "
                f"Provided value: {value!r}."
            )
        if (
            self._hidden_objective != RELU_SAMPLE_HIDDEN_KL
            and numeric != 1.0
        ):
            raise ValueError(
                "Expected class-prototype objectives to use fixed hidden "
                f"temperature 1.0. Provided value: {value!r}."
            )
        self._hidden_temperature = numeric

    def decoded_hidden(self) -> torch.Tensor:
        return halves_scores(self._hidden_layer.state)

    def student_logits(self) -> torch.Tensor:
        return self._output.student_logits()

    def set_class_prototypes(
        self,
        prototypes: ClassPrototypeTable | torch.Tensor,
    ) -> None:
        raw = (
            prototypes.raw_means
            if isinstance(prototypes, ClassPrototypeTable)
            else prototypes
        )
        expected = (
            int(self._output_layer.shape[0]) // 2,
            int(self._hidden_layer.shape[0]),
        )
        if raw.ndim != 2 or tuple(raw.shape) != expected:
            raise ValueError(
                "Expected raw class prototypes with shape "
                f"{expected!r}. Provided value: {tuple(raw.shape)!r}."
            )
        if not bool(torch.isfinite(raw).all()):
            raise ValueError("Expected finite raw class prototypes.")
        self._class_prototypes = raw.detach().to(
            device="cpu",
            dtype=torch.float64,
        ).clone()

    def set_teacher_hidden(self, hidden: torch.Tensor) -> None:
        expected_width = int(self._hidden_layer.shape[0]) // 2
        if hidden.ndim != 2 or hidden.shape[1] != expected_width:
            raise ValueError(
                "Expected teacher hidden features with shape "
                f"(batch, {expected_width}). Provided value: "
                f"{tuple(hidden.shape)!r}."
            )
        self._teacher_hidden = hidden.detach()

    def set_teacher(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> None:
        self._output.set_teacher(logits, labels)
        self._set_labels(labels, batch_size=logits.shape[0], device=logits.device)
        if hidden is not None:
            if hidden.shape[0] != logits.shape[0]:
                raise ValueError(
                    "Expected teacher logits and hidden features to have "
                    "the same batch size. Provided value: "
                    f"logits={tuple(logits.shape)!r}, "
                    f"hidden={tuple(hidden.shape)!r}."
                )
            self.set_teacher_hidden(hidden)
        elif self._hidden_objective == RELU_SAMPLE_HIDDEN_KL:
            self._teacher_hidden = None

    def set_target(self, labels: torch.Tensor) -> None:
        self._output.set_target(labels)
        batch_size = int(self._output_layer.state.shape[0])
        self._set_labels(labels, batch_size=batch_size, device=labels.device)

    def _set_labels(
        self,
        labels: torch.Tensor,
        *,
        batch_size: int,
        device: torch.device,
    ) -> None:
        classes = int(self._output_layer.shape[0]) // 2
        if labels.ndim != 1 or labels.shape[0] != batch_size:
            raise ValueError(
                "Expected one rank-1 label per example. Provided value: "
                f"labels={tuple(labels.shape)!r}, batch_size={batch_size}."
            )
        resolved = labels.detach().to(device=device, dtype=torch.long)
        if bool(((resolved < 0) | (resolved >= classes)).any()):
            raise ValueError(f"Expected labels in [0, {classes}).")
        self._labels = resolved

    def output_loss(self) -> torch.Tensor:
        return self._output.eval()

    def hidden_loss(self) -> torch.Tensor:
        student = self.decoded_hidden()
        if self._hidden_objective == RELU_SAMPLE_HIDDEN_KL:
            if self._teacher_hidden is None:
                raise RuntimeError(
                    "Expected per-sample ReLU hidden targets before "
                    "evaluating hidden KL."
                )
            if self._teacher_hidden.shape[0] != student.shape[0]:
                raise RuntimeError(
                    "Expected teacher and student hidden batches to match. "
                    f"Provided value: teacher={self._teacher_hidden.shape[0]}, "
                    f"student={student.shape[0]}."
                )
            return teacher_to_student_kl(
                self._teacher_hidden,
                student * self._hidden_gain,
                temperature=self._hidden_temperature,
            )

        target_raw = self._prototype_targets(student)
        if self._hidden_objective == HEALTHY_CLASS_HIDDEN_KL:
            return teacher_to_student_kl(
                halves_scores(target_raw),
                student,
                temperature=1.0,
            )
        return 0.5 * (
            self._hidden_layer.state - target_raw
        ).square().mean(dim=1)

    def _prototype_targets(self, student: torch.Tensor) -> torch.Tensor:
        if self._class_prototypes is None:
            raise RuntimeError(
                "Expected healthy DRN class prototypes before evaluating "
                "a class-prototype objective."
            )
        if self._labels is None:
            raise RuntimeError(
                "Expected labels before selecting healthy class prototypes."
            )
        if self._labels.shape[0] != student.shape[0]:
            raise RuntimeError(
                "Expected labels and student hidden batches to match. "
                f"Provided value: labels={self._labels.shape[0]}, "
                f"student={student.shape[0]}."
            )
        labels = self._labels.to(device=student.device)
        prototypes = self._class_prototypes.to(
            device=student.device,
            dtype=student.dtype,
        )
        return prototypes.index_select(0, labels)

    def component_losses(self) -> dict[str, torch.Tensor]:
        output = self.output_loss()
        hidden = self.hidden_loss()
        weighted = hidden * self._hidden_weight
        return {
            "output": output,
            "hidden": hidden,
            "weighted_hidden": weighted,
            "total": output + weighted,
        }

    def eval(self) -> torch.Tensor:
        output = self.output_loss()
        # Preserve exact TeacherKLDivergence value and gradient behavior for
        # the lambda=0 control, including when no hidden target is bound.
        if self._hidden_weight == 0.0:
            return output
        return output + self._hidden_weight * self.hidden_loss()

    def error_fn(self) -> torch.Tensor:
        return self._output.error_fn()

    def _get_output(self) -> torch.Tensor:
        return self.student_logits()


__all__ = [
    "ClassPrototypeTable",
    "HEALTHY_CLASS_HIDDEN_KL",
    "HEALTHY_CLASS_HIDDEN_MSE",
    "HIDDEN_OBJECTIVES",
    "RELU_SAMPLE_HIDDEN_KL",
    "StarInspiredHiddenObjective",
    "accumulate_class_prototypes",
    "fit_positive_hidden_gain",
    "halves_scores",
    "paired_scores",
    "teacher_to_student_kl",
]
