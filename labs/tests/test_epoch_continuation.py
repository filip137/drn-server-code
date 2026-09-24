import json
from types import SimpleNamespace

import pytest
import torch

from experiments.epoch_continuation import EpochContinuation
from experiments.exact_run import build_train_command


def candidate(tmp_path, contract=None):
    p = SimpleNamespace(name="weight", state=torch.nn.Parameter(torch.tensor([0.4], dtype=torch.float64)))
    opt = torch.optim.Adam([p.state], lr=.01)
    p.state.grad = torch.ones_like(p.state)
    opt.step()
    loader = SimpleNamespace(_train_sampler=SimpleNamespace(generator=torch.Generator().manual_seed(4)),
                             _worker_generator=torch.Generator().manual_seed(8))
    continuation = EpochContinuation(
        run_dir=tmp_path, config={"batch_state_policy": "reset_each_batch", "evaluation": {"official_test": {"policy": "disabled"}}},
        epochs=3, chunk_epochs=1, parameters=[p], optimizer=opt, estimator=object(),
        loaders=loader, recorder=None, contract=contract or {"source": "frozen"})
    return continuation, p, opt


def test_restores_adam_state_and_refuses_changed_source(tmp_path):
    first, p, opt = candidate(tmp_path)
    assert first.pause_after(1, history={"loss": [1.]}, best_accuracy=.5, best_epoch=1)
    second, restored, other_opt = candidate(tmp_path)
    restored.state.data.fill_(99)
    saved = second.restore()
    assert saved["epoch"] == 1
    assert torch.equal(restored.state, p.state)
    for key, value in opt.state[p.state].items():
        assert torch.equal(value, other_opt.state[restored.state][key])
    changed, _, _ = candidate(tmp_path, {"source": "changed"})
    with pytest.raises(ValueError, match="contract mismatch"):
        changed.restore()


def test_refuses_changed_artifact_tail(tmp_path):
    (tmp_path / "metrics.jsonl").write_text('{"epoch":1}\n')
    first, _, _ = candidate(tmp_path)
    first.pause_after(1, history={}, best_accuracy=.5, best_epoch=1)
    (tmp_path / "metrics.jsonl").write_text('{"epoch":1}\n{"epoch":2}\n')
    second, _, _ = candidate(tmp_path)
    with pytest.raises(ValueError, match="artifact changed"):
        second.restore()


def test_refuses_damaged_checkpoint_before_loading(tmp_path):
    first, _, _ = candidate(tmp_path)
    first.pause_after(1, history={}, best_accuracy=.5, best_epoch=1)
    (tmp_path / "continuation.pt").write_bytes(b"damaged")
    second, _, _ = candidate(tmp_path)
    with pytest.raises(ValueError, match="hash mismatch"):
        second.restore()


def test_chunking_does_not_override_epoch_budget(tmp_path):
    cmd = build_train_command(tmp_path / "config.json", tmp_path / "run", epoch_chunk_size=10)
    assert "--epochs" not in cmd
    assert cmd[cmd.index("--epoch-chunk-size") + 1] == "10"


def test_continuation_smoke_exercises_two_restarts(tmp_path):
    cmd = build_train_command(tmp_path / "config.json", tmp_path / "run", smoke=True, epoch_chunk_size=1)
    assert cmd[cmd.index("--epochs") + 1] == "3"
    assert cmd[cmd.index("--max-batches") + 1] == "2"
    assert "--skip-terminal-official-test" in cmd
