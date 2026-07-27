"""Parameter-modifier seam used by the generic training runtime."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from typing import Any, ContextManager, Optional, Protocol, runtime_checkable


@runtime_checkable
class ParameterModifier(Protocol):
    """Temporarily alter parameters for a complete training or eval phase.

    The method names intentionally match the existing hardware-aware modifier,
    but this protocol contains no hardware-specific policy.
    """

    def training_context(self) -> ContextManager[Any]:
        """Return one context for a complete training minibatch."""

    def evaluation_context(self) -> ContextManager[Any]:
        """Return one context that may span a complete evaluation loader."""

    def state_dict(self) -> dict:
        """Return modifier state required for continuation."""

    def load_state_dict(self, state_dict: Mapping) -> None:
        """Restore modifier continuation state."""


class NoOpParameterModifier:
    """A stateless modifier that leaves parameters untouched."""

    def training_context(self) -> ContextManager["NoOpParameterModifier"]:
        return nullcontext(self)

    def evaluation_context(self) -> ContextManager["NoOpParameterModifier"]:
        return nullcontext(self)

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state_dict: Mapping) -> None:
        if not isinstance(state_dict, Mapping) or state_dict:
            raise ValueError(
                "Expected the no-op parameter modifier state to be an empty "
                f"mapping. Provided value: {state_dict!r}."
            )


NO_OP_PARAMETER_MODIFIER = NoOpParameterModifier()


def modifier_or_default(
    modifier: Optional[ParameterModifier],
) -> ParameterModifier:
    """Resolve ``None`` to the shared stateless no-op modifier."""

    if modifier is None:
        return NO_OP_PARAMETER_MODIFIER
    return modifier
