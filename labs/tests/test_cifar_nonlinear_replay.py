"""Bounded replay orchestration and evidence contracts on CPU."""
import copy
import pytest
import torch

from experiments.replay_cifar_nonlinear_head import (
    Deadline, PRECISIONS, TARGETS, calibration_search, capture_hashes, snapshot_hashes,
    tensor_hash, validate_evidence, weighted_mean,
)
from experiments.calibrate_cifar_block_beta import capture_batches
from experiments.compare_cifar_block_beta_gradients import (
    accumulate_displacement, agreement, eqprop_gradients, local_bptt_gradients, state_measurement_fields,
)
from experiments.replay_cifar_precision_response import check_snapshot, convert_model
from labs.cifar_l8_analog import CifarL8Analog
from labs.tests.test_cifar_nonlinear_head import config


def test_search_miss_preserves_history_but_fatal_failures_propagate():
    miss = calibration_search(lambda beta: {'relative_rms': 1.}, 2., 1.)
    assert miss['status'] == 'missing' and len(miss['history']) == 12
    assert miss['beta_injected_B'] is None
    good = calibration_search(lambda beta: {'relative_rms': beta * 2}, 3., 1.)
    assert good['status'] == 'matched' and good['beta_injected_B'] == 1.5
    for error in (RuntimeError('CUDA out of memory'), FloatingPointError('Nonfinite'), TimeoutError('budget')):
        def measure(beta):
            raise error
        with pytest.raises(type(error), match=str(error)):
            calibration_search(measure, 1., 1.)


def test_deadline_and_weighted_gradient_pooling(monkeypatch):
    import experiments.replay_cifar_nonlinear_head as runner
    clock = [10.]
    monkeypatch.setattr(runner.time, 'monotonic', lambda: clock[0])
    deadline = Deadline(2.)
    clock[0] = 11.9
    deadline.check()
    clock[0] = 12.
    with pytest.raises(TimeoutError):
        deadline.check()
    first, second = torch.tensor([1., 3.]), torch.tensor([5., -1.])
    assert torch.equal(weighted_mean([first, second], [1, 3]), torch.tensor([4., 0.]))
    assert torch.equal(first, torch.tensor([1., 3.]))
    with pytest.raises(ValueError):
        weighted_mean([first], [0])


def test_hashes_include_shape_dtype_and_exclude_captured_force():
    value = torch.tensor([1., 2.])
    assert len({tensor_hash(value), tensor_hash(value.reshape(1, 2)), tensor_hash(value.double())}) == 3
    saved = dict(input=value, states=[value * 2], gradient=value * 3)
    original = capture_hashes(saved)
    saved['gradient'] += 100
    assert original == capture_hashes(saved)


@pytest.mark.parametrize('hidden', [0, 4])
def test_real_tiny_capture_replay_and_precise_initializer_restore(hidden):
    torch.manual_seed(41)
    c = config(hidden)
    initializer = CifarL8Analog(c)
    convert_model(initializer, torch.float64)
    with torch.no_grad():
        # This low-order bit would disappear if the restore preceded conversion.
        initializer.head.weight.state.flatten()[0] += 1e-10
    snapshot = initializer.snapshot()
    x, y = torch.randn(2, 3, 8, 8), torch.tensor([0, 1])
    hashes = []
    for dtype in (torch.float64, torch.float32):
        model = CifarL8Analog(c)
        convert_model(model, dtype)
        model.restore(snapshot)
        before = model.snapshot()
        check_snapshot(snapshot, before)
        saved = capture_batches(model, [(x.to(dtype), y)], c['iterations'], 'cpu')[0]
        block = model.analog_blocks[0]
        local = local_bptt_gradients(block, saved[0], c['iterations'][0], 'cpu')
        estimate, displacement = eqprop_gradients(block, saved[0], .01, c['iterations'][0], 'cpu',
                                                  measure_layer_displacement=True)
        assert all(torch.isfinite(v).all() for v in local + estimate)
        assert agreement(estimate[0], local[0])['eqprop_norm'] >= 0
        total = {}
        accumulate_displacement(total, displacement)
        accumulate_displacement(total, displacement)
        measured = state_measurement_fields(total, block.free_layers()[0].name)
        single = state_measurement_fields(displacement, block.free_layers()[0].name)
        assert measured['state_displacement']['elements'] == 2 * single['state_displacement']['elements']
        assert measured['state_displacement']['relative_rms'] == single['state_displacement']['relative_rms']
        check_snapshot(before, model.snapshot())
        hashes.append(snapshot_hashes(before))
        assert hashes[-1] == snapshot_hashes(model.snapshot())
    assert hashes[0]['all_sha256'] != hashes[1]['all_sha256']


def evidence():
    c = dict(relative_rms_targets=TARGETS, cohort_indices=list(range(128)))
    calibration = [dict(block=b+1, target_index=i, target_relative_rms=t, status='missing',
        beta_injected_B=None, reason='No target match within 12 measurements',
        history=[dict(relative_target_error=1.)]) for b, targets in enumerate(TARGETS) for i, t in enumerate(targets)]
    identity = dict(parameters_and_bn_unchanged=True, checkpoint_bytes_unchanged=True,
        config_bytes_unchanged=True, source_bytes_unchanged=True, precision_checks_passed=True,
        captures=[dict(precision=p, batch=b, block=k) for p in PRECISIONS for b in range(4) for k in range(1, 4)])
    return c, calibration, dict(layers=[], batches=[], cohort_indices=list(range(128))), identity


def test_validator_distinguishes_missing_calibration_from_missing_execution():
    c, calibration, comparison, identity = evidence()
    checked = validate_evidence(c, calibration, comparison, identity)
    assert checked['targets_attempted'] == 19 and len(checked['coverage_missing']) == 19
    assert checked['scientific_coverage_complete'] is False
    with pytest.raises(ValueError, match='every declared target'):
        validate_evidence(c, calibration[:-1], comparison, identity)
    identity['parameters_and_bn_unchanged'] = False
    with pytest.raises(ValueError, match='Immutable identity'):
        validate_evidence(c, calibration, comparison, identity)


def test_validator_requires_every_layer_batch_precision_but_not_cosine_pass():
    c, calibration, comparison, identity = evidence()
    calibration[0].update(status='matched', beta_injected_B=.01, history=[dict(relative_target_error=.01)])
    for precision in PRECISIONS:
        for layer in range(1, 4):
            row = dict(precision=precision, block=1, target_index=0, layer=layer,
                beta_injected_B=.01, target_relative_rms=TARGETS[0][0], examples=128,
                common_full64_local_bptt={'cosine': -1.}, own_precision_local_bptt={'cosine': -1.},
                state_displacement={}, output_displacement={}, output_force_rms=1., injected_force_rms=.01)
            comparison['layers'].append(row)
            comparison['batches'].extend(dict(row, batch=b, examples=32) for b in range(4))
    assert validate_evidence(c, calibration, comparison, identity)['targets_matched'] == 1
    comparison['batches'].pop()
    with pytest.raises(ValueError, match='coverage'):
        validate_evidence(c, calibration, comparison, identity)


def trained_config(epoch=1, hidden=0):
    training = dict(gradient_method='bptt', precision='float32', epochs=5,
                    head_hidden_units=hidden, arm_id='ours', study_id='training')
    return dict(checkpoint_epoch=epoch, model_config=training,
        training_provenance=dict(initial_trunk_sha256='a' * 64,
            training_run=f'train/{hidden}', training_checkpoint_sha256='b' * 64,
            training_source_sha256={'trainer.py': 'c' * 64}, training_config=copy.deepcopy(training)))


@pytest.mark.parametrize('epoch', [1, 5])
def test_trained_checkpoint_requires_exact_epoch_config_and_float32_bptt(epoch):
    from experiments.replay_cifar_nonlinear_head import validate_checkpoint
    cfg = trained_config(epoch)
    checkpoint = dict(epoch=epoch, config=copy.deepcopy(cfg['model_config']),
                      model={'module': {'weight': torch.ones(2)}})
    validate_checkpoint(checkpoint, cfg, 'b' * 64)
    changed = copy.deepcopy(checkpoint)
    changed['epoch'] = 5 if epoch == 1 else 1
    with pytest.raises(ValueError, match='epoch'):
        validate_checkpoint(changed, cfg, 'b' * 64)
    with pytest.raises(ValueError, match='hash mismatch'):
        validate_checkpoint(checkpoint, cfg, 'd' * 64)
    changed = copy.deepcopy(checkpoint)
    changed['config']['gradient_method'] = 'conv_eqprop_autograd_boundaries'
    with pytest.raises(ValueError, match='training config'):
        validate_checkpoint(changed, cfg, 'b' * 64)
    changed = copy.deepcopy(checkpoint)
    changed['model']['module']['weight'] = torch.ones(2, dtype=torch.float64)
    with pytest.raises(ValueError, match='non-float32'):
        validate_checkpoint(changed, cfg, 'b' * 64)
    changed_cfg = copy.deepcopy(cfg)
    changed_cfg['model_config']['epochs'] = 10
    changed_cfg['training_provenance']['training_config']['epochs'] = 10
    with pytest.raises(ValueError, match='five-epoch'):
        validate_checkpoint(checkpoint, changed_cfg, 'b' * 64)
    validate_checkpoint(dict(epoch=0), {}, 'any-initializer-hash')
    with pytest.raises(ValueError, match='epoch'):
        validate_checkpoint(dict(epoch=1), {}, 'any-initializer-hash')


def pair_fixture(epoch=1):
    runs = {}
    for arm, hidden in [('single', 0), ('hidden512', 512)]:
        cfg = trained_config(epoch, hidden)
        cfg.update(study_id='paired-study', arm_id=arm, cohort_indices=list(range(128)),
                   relative_rms_targets=copy.deepcopy(TARGETS))
        captures = [dict(precision=p, batch=b, block=k, input=f'{arm}-{p}-{b}-{k}', states=[arm])
                    for p in PRECISIONS for b in range(4) for k in range(1, 4)]
        runs[arm] = dict(manifest=dict(source_sha256={'replay.py': 'same'},
            training_data_sha256={'data_batch_1': 'same'}, resolved_config=cfg, checkpoint_sha256='b' * 64),
            identity=dict(captures=captures, models={p: dict(trunk_sha256=arm,
                tensors={'conductances': {'blocks.0.weights.0': arm}}) for p in PRECISIONS}))
    return runs


def test_pair_analysis_allows_expected_learned_differences_only_with_matched_provenance():
    from experiments.analyze_cifar_nonlinear_head import validate_pair
    runs = pair_fixture()
    matched = validate_pair(runs)
    assert matched['trained_trunk_divergence_expected'] is True
    assert matched['changed_capture_count'] == 24
    assert matched['cross_head_free_states_equal'] is False
    for field, value in [('initial_trunk_sha256', 'e' * 64),
                         ('training_source_sha256', {'trainer.py': 'different'})]:
        changed = copy.deepcopy(runs)
        changed['hidden512']['manifest']['resolved_config']['training_provenance'][field] = value
        with pytest.raises(ValueError, match=field):
            validate_pair(changed)
    changed = copy.deepcopy(runs)
    cfg = changed['hidden512']['manifest']['resolved_config']
    cfg['model_config']['learning_rate'] = .123
    cfg['training_provenance']['training_config']['learning_rate'] = .123
    with pytest.raises(ValueError, match='configuration differs'):
        validate_pair(changed)
    changed = copy.deepcopy(runs)
    changed['hidden512']['manifest']['resolved_config']['checkpoint_epoch'] = 5
    with pytest.raises(ValueError, match='epoch differs'):
        validate_pair(changed)


def test_initial_pair_still_requires_identical_trunks_and_all_free_states():
    from experiments.analyze_cifar_nonlinear_head import validate_pair
    runs = pair_fixture(0)
    with pytest.raises(ValueError, match='trunk hash differs'):
        validate_pair(runs)
    runs['hidden512']['identity']['models'] = copy.deepcopy(runs['single']['identity']['models'])
    with pytest.raises(ValueError, match='free-state captures differ'):
        validate_pair(runs)
    runs['hidden512']['identity']['captures'] = copy.deepcopy(runs['single']['identity']['captures'])
    result = validate_pair(runs)
    assert result['exact_trunk_tensors_both_precisions'] is True
    assert result['exact_block_inputs_and_free_states_all_24_captures'] is True


@pytest.mark.parametrize('epoch', [0, 1, 5])
def test_target_grid_is_frozen_by_checkpoint_epoch(epoch):
    from experiments.replay_cifar_nonlinear_head import TRAINED_TARGETS, validate_target_grid
    expected = TRAINED_TARGETS if epoch else TARGETS
    cfg = dict(checkpoint_epoch=epoch, relative_rms_targets=copy.deepcopy(expected))
    validate_target_grid(cfg)
    assert [len(targets) for targets in expected] == [6, 6, 7]
    cfg['relative_rms_targets'] = copy.deepcopy(TARGETS if epoch else TRAINED_TARGETS)
    with pytest.raises(ValueError, match='target grids'):
        validate_target_grid(cfg)
    cfg['relative_rms_targets'] = copy.deepcopy(expected)
    cfg['relative_rms_targets'][0].append(.5)
    with pytest.raises(ValueError, match='target grids'):
        validate_target_grid(cfg)
