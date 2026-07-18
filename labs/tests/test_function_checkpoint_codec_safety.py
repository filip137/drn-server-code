import copy

import numpy as np
import pytest
import torch

from labs.common import export_pt_to_npz
from labs.tools.zero_last_param import zero_last_param
from model.function.interaction import (
    FUNCTION_CHECKPOINT_FORMAT,
    FUNCTION_CHECKPOINT_VERSION,
    Function,
)


class _CheckpointParameter:
    def __init__(self, name, state):
        self.name = name
        self.state = state


class _CheckpointFunction(Function):
    def __init__(self, params):
        super().__init__([], params)

    def eval(self):
        return torch.zeros(1)


def _function():
    return _CheckpointFunction(
        [
            _CheckpointParameter("weight", torch.tensor([1.0, 2.0])),
            _CheckpointParameter("bias", torch.tensor([3.0])),
        ]
    )


def test_versioned_checkpoint_round_trip_has_explicit_schema(tmp_path):
    function = _function()
    checkpoint = tmp_path / "model.pt"
    function.save(checkpoint)

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert payload["format"] == FUNCTION_CHECKPOINT_FORMAT
    assert payload["version"] == FUNCTION_CHECKPOINT_VERSION
    assert [entry["name"] for entry in payload["schema"]] == ["weight", "bias"]

    function.params()[0].state.fill_(99.0)
    function.params()[1].state.fill_(99.0)
    function.load(checkpoint)
    assert torch.equal(function.params()[0].state, torch.tensor([1.0, 2.0]))
    assert torch.equal(function.params()[1].state, torch.tensor([3.0]))


def test_historical_checkpoint_requires_exact_count_and_shape(tmp_path):
    function = _function()
    checkpoint = tmp_path / "legacy.pt"
    torch.save([torch.tensor([4.0, 5.0]), torch.tensor([6.0])], checkpoint)
    function.load(checkpoint)
    assert torch.equal(function.params()[0].state, torch.tensor([4.0, 5.0]))

    torch.save([torch.tensor([7.0, 8.0])], checkpoint)
    with pytest.raises(ValueError, match="exactly 2 parameter states"):
        function.load(checkpoint)

    torch.save([torch.tensor([[7.0, 8.0]]), torch.tensor([9.0])], checkpoint)
    with pytest.raises(ValueError, match="shape"):
        function.load(checkpoint)


@pytest.mark.parametrize(
    "bad_state",
    [torch.tensor([1.0, 2.0], dtype=torch.float64), torch.tensor([1.0, float("nan")])],
)
def test_malformed_legacy_checkpoint_is_atomic(tmp_path, bad_state):
    function = _function()
    before = [param.state.clone() for param in function.params()]
    checkpoint = tmp_path / "bad.pt"
    torch.save([bad_state, torch.tensor([9.0])], checkpoint)

    with pytest.raises(ValueError):
        function.load(checkpoint)

    assert all(
        torch.equal(param.state, old_state)
        for param, old_state in zip(function.params(), before)
    )


def test_versioned_checkpoint_rejects_reordered_schema_atomically(tmp_path):
    function = _function()
    checkpoint = tmp_path / "model.pt"
    function.save(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    malformed = copy.deepcopy(payload)
    malformed["schema"] = list(reversed(malformed["schema"]))
    malformed["states"] = list(reversed(malformed["states"]))
    torch.save(malformed, checkpoint)
    before = [param.state.clone() for param in function.params()]

    with pytest.raises(ValueError, match="schema entry"):
        function.load(checkpoint)

    assert all(
        torch.equal(param.state, old_state)
        for param, old_state in zip(function.params(), before)
    )


def test_checkpoint_export_and_zero_last_param_support_versioned_codec(tmp_path):
    function = _function()
    checkpoint = tmp_path / "model.pt"
    function.save(checkpoint)

    npz_path = export_pt_to_npz(checkpoint)
    with np.load(npz_path) as arrays:
        assert set(arrays.files) == {"weight", "bias"}
        assert np.array_equal(arrays["weight"], np.asarray([1.0, 2.0]))

    zeroed_path = tmp_path / "zeroed.pt"
    zero_last_param(checkpoint, zeroed_path)
    payload = torch.load(zeroed_path, map_location="cpu", weights_only=True)
    assert payload["format"] == FUNCTION_CHECKPOINT_FORMAT
    assert payload["version"] == FUNCTION_CHECKPOINT_VERSION

    restored = _function()
    restored.load(zeroed_path)
    assert torch.equal(restored.params()[0].state, torch.tensor([1.0, 2.0]))
    assert torch.equal(restored.params()[1].state, torch.zeros(1))


def test_zero_last_param_keeps_historical_checkpoint_usable(tmp_path):
    source = tmp_path / "legacy.pt"
    target = tmp_path / "legacy_zeroed.pt"
    torch.save([torch.tensor([2.0, 3.0]), torch.tensor([4.0])], source)

    zero_last_param(source, target)

    restored = _function()
    restored.load(target)
    assert torch.equal(restored.params()[0].state, torch.tensor([2.0, 3.0]))
    assert torch.equal(restored.params()[1].state, torch.zeros(1))
