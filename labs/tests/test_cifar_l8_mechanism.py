import copy

import torch
from torch.nn import functional as F

from labs.cifar_l8_analog import CifarL8Analog
from labs.tests.test_cifar_l8_amplification import config
from experiments.analyze_cifar_l8_mechanism import (
    adam_proposal_stats, capture_bn, check_unchanged, describe_states, equivalence_step,
)
from experiments.train_cifar_l8_analog import make_optimizer


def test_equivalence_includes_adam_history_and_bn_buffers():
    torch.manual_seed(67)
    baseline_config = config(1., 1.)
    legacy_config = config(4., .25)
    legacy_config['block_output_normalization'] = 'voltage'
    baseline, legacy = CifarL8Analog(baseline_config), CifarL8Analog(legacy_config)
    legacy.restore(baseline.snapshot())
    optimizers = [make_optimizer(m, baseline_config)[0] for m in (baseline, legacy)]
    for _ in range(3):
        row = equivalence_step([baseline, legacy], optimizers, torch.randn(4, 3, 8, 8),
                               torch.tensor([0, 1, 2, 3]), [6, 6, 4])
        assert row['passed']
        assert row['maximum_relative_error'] == 0.
    assert int(next(iter(optimizers[0].state.values()))['step']) == 3


def test_adam_preview_matches_real_next_projected_update_and_replay_restores_bn():
    torch.manual_seed(71)
    c = config(1., 1.)
    model = CifarL8Analog(c)
    optimizer, _ = make_optimizer(model, c)
    x, y = torch.randn(4, 3, 8, 8), torch.tensor([0, 1, 2, 3])
    _, logits = model.bptt(x, [6, 6, 4]); F.cross_entropy(logits, y).backward()
    optimizer.step(); model.project_(); model.detach_state_(); optimizer.zero_grad(set_to_none=True)
    saved = model.snapshot()
    checkpoint = {'model': saved, 'optimizer': copy.deepcopy(optimizer.state_dict())}
    captured = []; handles = capture_bn(model, captured)
    _, logits = model.bptt(x, [6, 6, 4]); F.cross_entropy(logits, y).backward()
    for handle in handles:
        handle.remove()
    rows = adam_proposal_stats(model, checkpoint, c)
    check_unchanged(model, saved, buffers=False)
    states = describe_states(model, 'training')
    assert len(states) == 8
    assert all(r['projected_kkt_relative_l2'] >= 0 for r in states)
    assert len(captured) == 6
    assert all(0 < r['epsilon_share_median'] <= 1 for r in captured)
    before = {n: p.detach().clone() for n, p in model.trainable_tensors().items()}
    optimizer.step(); model.project_()
    for row in rows:
        name = row['parameter']
        actual = float((model.trainable_tensors()[name].detach() - before[name]).norm())
        torch.testing.assert_close(torch.tensor(row['projected_adam_proposal_l2']),
                                   torch.tensor(actual), rtol=3e-5, atol=2e-8)
    model.restore(saved); check_unchanged(model, saved)
