"""Teacher-free, layer-local STAR nudging for dual-rail DRNs.

The primary STAR representation in this module is the raw physical voltage
state stored by the DRN's hidden and output layers.  The cost contains only
independent, label-addressed quadratic state errors.  It therefore introduces
no teacher query, transpose error path, backpropagation graph, or optimizer
state.

Logical rail differences are intentionally exposed only through the explicitly
named ablation conversion at the bottom of this module.  They are not used by
the primary nudge.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from model.function.interaction import QFunction
from model.variable.layer import Layer


DRN_LOGICAL_RAIL_DIFFERENCE_ABLATION = (
    "drn.logical_rail_difference_ablation.v1"
)


def _finite_nonnegative_gain(value: float, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Expected {name} to be a finite non-negative number.")
    gain = float(value)
    if not math.isfinite(gain) or gain < 0.0:
        raise ValueError(f"Expected {name} to be a finite non-negative number.")
    return gain


def _target_table(
    value: torch.Tensor,
    *,
    name: str,
    width: int,
) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != 2
        or value.shape[0] < 1
        or value.shape[1] != width
        or not value.is_floating_point()
        or not bool(torch.all(torch.isfinite(value)))
    ):
        raise ValueError(
            f"Expected {name} to be a finite floating class-by-{width} target table."
        )
    return value.detach().clone()


class DrnLocalStarNudge(QFunction):
    """A label-addressed quadratic nudge on raw physical DRN rail voltages.

    For hidden state ``h``, output state ``o``, and class label ``y``, the
    per-example cost is

    ``0.5 * hidden_gain * ||h - hidden_means[y]||^2``
    ``+ 0.5 * output_gain * ||o - output_means[y]||^2``.

    Both terms are local to their layer.  The object implements the analytic
    gradient and quadratic-coefficient interfaces consumed by
    :class:`training.sgd.AugmentedFunction` and centered equilibrium
    propagation.  Its parameter list is empty because STAR targets are fixed
    local memory, not trainable model parameters.
    """

    def __init__(
        self,
        hidden_layer: Layer,
        output_layer: Layer,
        hidden_target_means: torch.Tensor,
        output_target_means: torch.Tensor,
        *,
        hidden_gain: float,
        output_gain: float,
    ) -> None:
        if hidden_layer is output_layer:
            raise ValueError("Expected distinct hidden and output DRN layers.")
        hidden_state = self._raw_state(hidden_layer, role="hidden")
        output_state = self._raw_state(output_layer, role="output")
        if hidden_state.shape[0] != output_state.shape[0]:
            raise ValueError("Expected hidden and output states to share a batch size.")

        self._hidden_layer = hidden_layer
        self._output_layer = output_layer
        self._hidden_gain = _finite_nonnegative_gain(
            hidden_gain,
            name="hidden_gain",
        )
        self._output_gain = _finite_nonnegative_gain(
            output_gain,
            name="output_gain",
        )
        self._hidden_target_means = _target_table(
            hidden_target_means,
            name="hidden_target_means",
            width=hidden_state.shape[1],
        )
        self._output_target_means = _target_table(
            output_target_means,
            name="output_target_means",
            width=output_state.shape[1],
        )
        if self._hidden_target_means.shape[0] != self._output_target_means.shape[0]:
            raise ValueError(
                "Expected hidden and output STAR tables to have the same class count."
            )
        self._num_classes = int(self._hidden_target_means.shape[0])
        self._labels: torch.Tensor | None = None
        super().__init__([hidden_layer, output_layer], [])

    @staticmethod
    def _raw_state(layer: Layer, *, role: str) -> torch.Tensor:
        state = getattr(layer, "state", None)
        if (
            not isinstance(state, torch.Tensor)
            or state.ndim != 2
            or state.shape[1] < 2
            or state.shape[1] % 2 != 0
            or not state.is_floating_point()
            or not bool(torch.all(torch.isfinite(state)))
        ):
            raise ValueError(
                f"Expected {role} layer.state to be a finite floating batch-by-even-rail matrix."
            )
        return state

    @property
    def num_classes(self) -> int:
        return self._num_classes

    def set_target(self, labels: torch.Tensor) -> None:
        """Select one stored target row per example using integer labels."""

        hidden_state, output_state = self._current_states()
        if (
            not isinstance(labels, torch.Tensor)
            or labels.dtype != torch.int64
            or labels.ndim != 1
            or labels.shape[0] != hidden_state.shape[0]
        ):
            raise ValueError(
                "Expected one int64 STAR class label per current DRN example."
            )
        if bool(torch.any(labels < 0)) or bool(torch.any(labels >= self._num_classes)):
            raise ValueError(
                f"Expected STAR class labels in [0, {self._num_classes - 1}]."
            )
        if output_state.shape[0] != labels.shape[0]:
            raise ValueError("Expected hidden, output, and label batch sizes to match.")
        self._labels = labels.detach().clone()

    def _current_states(self) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_state = self._raw_state(self._hidden_layer, role="hidden")
        output_state = self._raw_state(self._output_layer, role="output")
        if (
            hidden_state.shape[0] != output_state.shape[0]
            or hidden_state.shape[1] != self._hidden_target_means.shape[1]
            or output_state.shape[1] != self._output_target_means.shape[1]
        ):
            raise ValueError(
                "Expected current DRN states to retain the target-table widths and a shared batch size."
            )
        return hidden_state, output_state

    def _current_labels(self, *, device: torch.device, batch_size: int) -> torch.Tensor:
        if self._labels is None:
            raise RuntimeError("STAR targets are unset; call set_target(labels) first.")
        if self._labels.shape[0] != batch_size:
            raise RuntimeError(
                "Current DRN batch size changed after STAR labels were selected; "
                "call set_target(labels) for the new batch."
            )
        return self._labels.to(device=device)

    def _layer_parts(
        self,
        layer: Layer,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        hidden_state, output_state = self._current_states()
        if layer is self._hidden_layer:
            state = hidden_state
            table = self._hidden_target_means
            gain = self._hidden_gain
        elif layer is self._output_layer:
            state = output_state
            table = self._output_target_means
            gain = self._output_gain
        else:
            raise ValueError("Expected the declared hidden or output STAR layer.")
        labels = self._current_labels(device=state.device, batch_size=state.shape[0])
        selected = table.to(device=state.device, dtype=state.dtype)[labels]
        return state, selected, gain

    def eval(self) -> torch.Tensor:
        """Return one summed hidden/output STAR loss per example."""

        hidden_state, hidden_target, hidden_gain = self._layer_parts(
            self._hidden_layer
        )
        output_state, output_target, output_gain = self._layer_parts(
            self._output_layer
        )
        hidden_loss = 0.5 * hidden_gain * (hidden_state - hidden_target).square().sum(
            dim=1
        )
        output_loss = 0.5 * output_gain * (output_state - output_target).square().sum(
            dim=1
        )
        return hidden_loss + output_loss

    def grad_layer_fn(self, layer: Layer):
        """Return the analytic layer-local STAR error function."""

        self._declared_gain(layer)

        def gradient() -> torch.Tensor:
            state, selected, gain = self._layer_parts(layer)
            return gain * (state - selected)

        return gradient

    def a_coef_fn(self, layer: Layer):
        """Return the diagonal quadratic coefficient ``gain / 2``."""

        gain = self._declared_gain(layer)
        return lambda: 0.5 * gain

    def b_coef_fn(self, layer: Layer):
        """Return the label-addressed linear coefficient ``-gain * mean``."""

        self._declared_gain(layer)

        def coefficient() -> torch.Tensor:
            _state, selected, gain = self._layer_parts(layer)
            return -gain * selected

        return coefficient

    def _declared_gain(self, layer: Layer) -> float:
        if layer is self._hidden_layer:
            return self._hidden_gain
        if layer is self._output_layer:
            return self._output_gain
        raise ValueError("Expected the declared hidden or output STAR layer.")


@dataclass(frozen=True)
class DrnLogicalRailDifferenceAblation:
    """Explicit non-primary logical view derived from raw physical rails."""

    hidden_logical: torch.Tensor
    output_logical: torch.Tensor
    representation_id: str = DRN_LOGICAL_RAIL_DIFFERENCE_ABLATION


def convert_raw_rails_to_logical_difference_ablation(
    hidden_raw_rails: torch.Tensor,
    output_raw_rails: torch.Tensor,
) -> DrnLogicalRailDifferenceAblation:
    """Convert raw rails using the MNIST DRN's explicit ablation layouts.

    Hidden rails use the ``halves`` layout, ``positive - negative``.  Output
    rails use adjacent pairs, ``even - odd``.  The function name and return
    type deliberately retain ``ablation`` so these reduced states cannot be
    confused with the primary physical-voltage representation.
    """

    def raw(value: torch.Tensor, *, role: str) -> torch.Tensor:
        if (
            not isinstance(value, torch.Tensor)
            or value.ndim != 2
            or value.shape[1] < 2
            or value.shape[1] % 2 != 0
            or not value.is_floating_point()
            or not bool(torch.all(torch.isfinite(value)))
        ):
            raise ValueError(
                f"Expected {role} raw rails to be a finite floating batch-by-even-rail matrix."
            )
        return value

    hidden = raw(hidden_raw_rails, role="hidden")
    output = raw(output_raw_rails, role="output")
    if hidden.shape[0] != output.shape[0]:
        raise ValueError("Expected hidden and output raw rails to share a batch size.")
    hidden_half = hidden.shape[1] // 2
    return DrnLogicalRailDifferenceAblation(
        hidden_logical=hidden[:, :hidden_half] - hidden[:, hidden_half:],
        output_logical=output[:, 0::2] - output[:, 1::2],
    )


__all__ = [
    "DRN_LOGICAL_RAIL_DIFFERENCE_ABLATION",
    "DrnLocalStarNudge",
    "DrnLogicalRailDifferenceAblation",
    "convert_raw_rails_to_logical_difference_ablation",
]
