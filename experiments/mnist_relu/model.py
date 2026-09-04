"""Bias-free ReLU teacher with stable experiment checkpoint keys."""

from __future__ import annotations

import torch

from model.resistive.builders import ParameterBinding, ParameterCatalog


class TeacherWeight:
    def __init__(self, shape: tuple[int, int], *, device: torch.device) -> None:
        self.state = torch.nn.Parameter(
            torch.empty(shape, dtype=torch.float32, device=device)
        )
        self.min_cond = None
        self.max_cond = None

    def clamp_(self) -> None:
        return None


class BiasFreeReluTeacher:
    def __init__(
        self,
        *,
        device: torch.device,
        dims: tuple[int, int, int] = (784, 50, 10),
    ) -> None:
        if dims not in {(784, 50, 10), (784, 256, 10)}:
            raise ValueError(
                "Expected bias-free ReLU teacher dims to be either "
                "(784, 50, 10) or (784, 256, 10). "
                f"Provided value: {dims!r}."
            )
        self.dims = dims
        self.input_weight = TeacherWeight((dims[0], dims[1]), device=device)
        self.output_weight = TeacherWeight((dims[1], dims[2]), device=device)
        torch.nn.init.kaiming_uniform_(
            self.input_weight.state.T,
            a=0.0,
            nonlinearity="relu",
        )
        torch.nn.init.kaiming_uniform_(
            self.output_weight.state.T,
            a=0.0,
            nonlinearity="linear",
        )
        self.catalog = ParameterCatalog(
            (
                ParameterBinding(
                    key="teacher.dense_weight.0",
                    parameter=self.input_weight,
                    group="teacher",
                    role="dense_weight",
                ),
                ParameterBinding(
                    key="teacher.dense_weight.1",
                    parameter=self.output_weight,
                    group="teacher",
                    role="dense_weight",
                ),
            )
        )

    def hidden(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the frozen teacher's post-ReLU hidden representation."""

        return torch.relu(inputs @ self.input_weight.state)

    def representations(self, inputs: torch.Tensor) -> torch.Tensor:
        """Alias for :meth:`hidden` used by representation-distillation arms."""

        return self.hidden(inputs)

    def logits(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.hidden(inputs) @ self.output_weight.state

    def parameters(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.input_weight.state, self.output_weight.state

    def eval(self) -> "BiasFreeReluTeacher":
        return self

    def train(self) -> "BiasFreeReluTeacher":
        return self

    @property
    def architecture(self) -> str:
        return "bias_free_relu_" + "_".join(str(value) for value in self.dims)


__all__ = ["BiasFreeReluTeacher"]
