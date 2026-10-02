"""Convolution-only centered EqProp learning and epoch-wise beta calibration.

Digital parameters and inter-block forces use autograd through the free T pass.
Only convolution conductance gradients are replaced. This is a hybrid learner,
not end-to-end EqProp through the digital boundaries.
"""
from contextlib import ExitStack, contextmanager
import copy
import math
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from experiments import reporting
from experiments.calibrate_cifar_block_beta import (
    FrozenOutputCost, capture_batches, mask_output_force, measure_block, nudged_minimizer, pooled_rms,
    search_beta, validate_force_mask,
)
from experiments.compare_cifar_block_beta_gradients import (
    agreement, eqprop_gradients, local_bptt_gradients,
)
from labs.cifar_l8_analog import batchnorm_minibatch_stats_only
from training.sgd import AugmentedFunction


METHOD = 'conv_eqprop_autograd_boundaries'


def validate_spec(config):
    spec = config['conv_eqprop']
    if config.get('gradient_method') != METHOD or spec['policy'] not in ('fixed', 'epoch', 'periodic'):
        raise ValueError('Explicit hybrid method and fixed/epoch/periodic beta policy required')
    for key in ('initial_betas', 'rms_targets'):
        if len(spec[key]) != 3 or not all(math.isfinite(v) and v > 0 for v in spec[key]):
            raise ValueError(f'Invalid {key}')
    indices = spec['cohort_indices']
    if len(indices) != 256 or len(set(indices)) != 256:
        raise ValueError('Calibration requires 256 distinct training indices')
    if not 0 < spec['relative_tolerance'] < 1 or spec['max_evaluations'] < 1:
        raise ValueError('Invalid calibration search limits')
    if config['label_smoothing'] != 0 or spec.get('read_noise', 0) != 0:
        raise ValueError('This hybrid comparison uses unsmoothed CE and no read noise')
    if spec['policy'] == 'periodic':
        if type(spec.get('calibrations_per_epoch')) is not int or spec['calibrations_per_epoch'] < 1:
            raise ValueError('Periodic beta requires positive integer calibrations_per_epoch')
    if type(spec.get('calibration_batch_size', config['batch_size'])) is not int or spec.get('calibration_batch_size', config['batch_size']) < 1:
        raise ValueError('Invalid calibration_batch_size')
    if not isinstance(spec.get('measure_training_rms', False), bool):
        raise ValueError('measure_training_rms must be boolean')
    validate_force_mask(spec.get('force_mask', 'none'))
    return spec


def initial_beta_state(config):
    spec = validate_spec(config)
    state = dict(policy=spec['policy'], values=list(spec['initial_betas']),
                 calibrated_for_epoch=1, initialization_reused=True, calibration_count=0)
    if spec['policy'] == 'periodic':
        state.update(calibrated_for_epoch=0, initialization_reused=False,
                     calibration_count=0, last_calibration_completed_updates=None)
    return state


def restore_beta_state(checkpoint, model, config):
    spec = validate_spec(config)
    state = copy.deepcopy(checkpoint.get('conv_beta_state'))
    if not state or state['policy'] != spec['policy']:
        raise ValueError('Missing or incompatible checkpoint beta state')
    if len(state['values']) != 3 or not all(math.isfinite(v) and v > 0 for v in state['values']):
        raise ValueError('Invalid checkpoint beta vector')
    expected = (1 if spec['policy'] == 'fixed' else checkpoint['epoch']
                if spec['policy'] == 'periodic' else max(1, checkpoint['epoch']))
    if state['calibrated_for_epoch'] != expected:
        raise ValueError('Beta calibration does not match the completed epoch boundary')
    if spec['policy'] == 'fixed' and state['values'] != spec['initial_betas']:
        raise ValueError('Fixed beta changed')
    if spec['policy'] == 'fixed' and state.get('calibration_count', 0) != 0:
        raise ValueError('Fixed beta checkpoint contains recalibration')
    if spec['policy'] == 'periodic':
        count = spec['calibrations_per_epoch'] * checkpoint['epoch']
        if state.get('calibration_count') != count:
            raise ValueError('Checkpoint beta calibration count disagrees with completed epochs')
        if count == 0:
            if state['values'] != spec['initial_betas'] or state.get('last_calibration_completed_updates') is not None:
                raise ValueError('Invalid initial periodic beta state')
        else:
            batches = state.get('batches_in_calibrated_epoch')
            if type(batches) is not int or batches < spec['calibrations_per_epoch']:
                raise ValueError('Invalid beta calibration epoch length')
            expected_update = calibration_updates(batches, spec['calibrations_per_epoch'])[-1]
            if state.get('last_calibration_completed_updates') != expected_update:
                raise ValueError('Missing last periodic beta calibration')
    model.conv_beta_state.clear()
    model.conv_beta_state.update(state)


def validate_training_beta_case(case):
    """Validate optional minibatch feedback before any training or model allocation."""
    if not isinstance(case.get('measure_training_rms', False), bool):
        raise ValueError('measure_training_rms must be boolean')
    if 'beta_feedback' in case and 'beta_search' in case:
        raise ValueError('beta_feedback and beta_search are mutually exclusive')
    enabled = 'beta_feedback' in case or 'beta_search' in case or case.get('measure_training_rms', False)
    if enabled and case.get('method') != 'hybrid_eqprop':
        raise ValueError('Training beta measurement requires hybrid_eqprop')
    if not enabled:
        return False
    positive = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v > 0
    values = case.get('betas')
    if not isinstance(values, list) or len(values) != 3 or not all(positive(v) for v in values):
        raise ValueError('Invalid initial training betas')
    controller = 'beta_search' if 'beta_search' in case else 'beta_feedback'
    if controller in case:
        spec = case[controller]
        fields = {'targets', 'factor_bounds', 'beta_bounds', 'rms_floor'}
        fields |= {'relative_tolerance', 'max_evaluations'} if controller == 'beta_search' else {'damping'}
        if not isinstance(spec, dict) or set(spec) != fields:
            raise ValueError('Invalid ' + controller + ' keys')
        targets = spec['targets']
        if not isinstance(targets, list) or len(targets) != 3 or not all(positive(v) for v in targets):
            raise ValueError('Invalid ' + controller + ' targets')
        for key in ('factor_bounds', 'beta_bounds'):
            limits = spec[key]
            if not isinstance(limits, list) or len(limits) != 2 or not all(positive(v) for v in limits) or limits[0] > limits[1]:
                raise ValueError('Invalid ' + controller + ' ' + key)
        if not positive(spec['rms_floor']):
            raise ValueError('Invalid ' + controller + ' rms_floor')
        if controller == 'beta_search':
            if (not positive(spec['relative_tolerance']) or spec['relative_tolerance'] >= 1 or
                    type(spec['max_evaluations']) is not int or not 1 <= spec['max_evaluations'] <= 5):
                raise ValueError('Invalid beta_search tolerance or evaluation cap')
        elif not positive(spec['damping']) or spec['damping'] > 1:
            raise ValueError('Invalid beta_feedback damping')
        if not spec['factor_bounds'][0] <= 1 <= spec['factor_bounds'][1]:
            raise ValueError(controller + ' factor bounds must include one')
        if not all(spec['beta_bounds'][0] <= v <= spec['beta_bounds'][1] for v in values):
            raise ValueError('Initial betas outside ' + controller + ' bounds')
    return True


def initial_training_beta_state(case):
    if not validate_training_beta_case(case):
        raise ValueError('Training beta instrumentation is not enabled')
    if 'beta_search' in case:
        return dict(policy='current_batch', values=list(case['betas']),
                    search_spec=copy.deepcopy(case['beta_search']), successful_updates=0,
                    search_counts=dict(pairs=[0, 0, 0], tolerance_misses=[0, 0, 0]))
    return dict(policy='minibatch' if 'beta_feedback' in case else 'fixed', values=list(case['betas']),
                feedback_spec=copy.deepcopy(case.get('beta_feedback')), successful_updates=0,
                clip_counts=dict(factor=[0, 0, 0], beta=[0, 0, 0]))


def validate_training_beta_state(state, case, updates):
    measured = validate_training_beta_case(case)
    policy = 'current_batch' if 'beta_search' in case else 'minibatch' if 'beta_feedback' in case else 'fixed'
    if not isinstance(state, dict) or state.get('policy') != policy:
        raise ValueError('Resume beta policy mismatch')
    values = state.get('values')
    if (not isinstance(values, list) or len(values) != 3 or
            not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v > 0 for v in values)):
        raise ValueError('Invalid resume beta vector')
    if policy == 'fixed' and values != case['betas']:
        raise ValueError('Resume fixed beta mismatch')
    if not measured:  # Original fixed checkpoints have no minibatch counters.
        return
    spec_name = 'search_spec' if policy == 'current_batch' else 'feedback_spec'
    case_spec = case.get('beta_search') if policy == 'current_batch' else case.get('beta_feedback')
    if state.get(spec_name) != case_spec:
        raise ValueError('Resume beta feedback specification mismatch')
    if type(state.get('successful_updates')) is not int or state['successful_updates'] != updates:
        raise ValueError('Resume beta successful-update counter mismatch')
    counts = state.get('search_counts') if policy == 'current_batch' else state.get('clip_counts')
    keys = {'pairs', 'tolerance_misses'} if policy == 'current_batch' else {'factor', 'beta'}
    if not isinstance(counts, dict) or set(counts) != keys:
        raise ValueError('Invalid beta clip counters')
    for key, values_count in counts.items():
        minimum = updates if policy == 'current_batch' and key == 'pairs' else 0
        maximum = updates * case_spec['max_evaluations'] if policy == 'current_batch' and key == 'pairs' else updates
        if (not isinstance(values_count, list) or len(values_count) != 3 or
                not all(type(v) is int and minimum <= v <= maximum for v in values_count)):
            raise ValueError('Invalid beta clip counters')
        if policy == 'fixed' and any(values_count):
            raise ValueError('Fixed beta has nonzero clip counters')
    if policy in ('minibatch', 'current_batch'):
        lower, upper = case_spec['beta_bounds']
        if not all(lower <= v <= upper for v in values):
            raise ValueError('Resume beta outside bounds')
    if updates == 0 and values != case['betas']:
        raise ValueError('Initial beta state differs from declared betas')


def next_training_beta_state(state, case, measurements):
    """Propose only: the caller commits after a successful finite optimizer update."""
    validate_training_beta_state(state, case, state.get('successful_updates'))
    if len(measurements) != 3 or not all(math.isfinite(m['relative_rms']) and m['relative_rms'] >= 0 for m in measurements):
        raise ValueError('Invalid training relative RMS')
    proposal = copy.deepcopy(state)
    if 'beta_search' in case:
        records = [measured['search'] for measured in measurements]
        for i, record in enumerate(records):
            validate_beta_search_record(record, state['values'][i], case['beta_search']['targets'][i], case['beta_search'])
            chosen = record['trials'][record['chosen_index']]
            if any(value != chosen[key] for key, value in measurements[i].items() if key != 'search'):
                raise ValueError('Selected beta measurement mismatch')
            proposal['values'][i] = record['selected_beta']
            proposal['search_counts']['pairs'][i] += record['pair_count']
            proposal['search_counts']['tolerance_misses'][i] += int(record['tolerance_miss'])
        proposal['successful_updates'] += 1
        return proposal, dict(training_rms=[{k:v for k,v in measured.items() if k != 'search'} for measured in measurements],
            starting_betas=list(state['values']), betas_used=list(proposal['values']),
            betas_next=list(proposal['values']), beta_search=records,
            beta_successful_updates=proposal['successful_updates'])
    spec = case.get('beta_feedback')
    factor_flags, beta_flags = [], []
    for i, (beta, measured) in enumerate(zip(state['values'], measurements)):
        factor_clipped = beta_clipped = False
        if spec is not None:
            raw_factor = (spec['targets'][i] / max(measured['relative_rms'], spec['rms_floor'])) ** spec['damping']
            factor = min(spec['factor_bounds'][1], max(spec['factor_bounds'][0], raw_factor))
            raw_beta = beta * factor
            proposal['values'][i] = min(spec['beta_bounds'][1], max(spec['beta_bounds'][0], raw_beta))
            factor_clipped, beta_clipped = factor != raw_factor, proposal['values'][i] != raw_beta
        factor_flags.append(factor_clipped)
        beta_flags.append(beta_clipped)
        proposal['clip_counts']['factor'][i] += int(factor_clipped)
        proposal['clip_counts']['beta'][i] += int(beta_clipped)
    proposal['successful_updates'] += 1
    return proposal, dict(training_rms=measurements, betas_used=list(state['values']),
        betas_next=list(proposal['values']), factor_clipped=factor_flags, beta_clipped=beta_flags,
        beta_successful_updates=proposal['successful_updates'])


def displacement_rms(displacement):
    centered, free = displacement['centered_squares'], displacement['free_squares']
    if not math.isfinite(centered) or not math.isfinite(free):
        raise FloatingPointError('Nonfinite training displacement')
    if centered < 0 or free <= 0:
        raise ValueError('Training displacement requires nonnegative squares and positive free RMS')
    measured = pooled_rms(centered, free, displacement['elements'])
    if not all(math.isfinite(measured[k]) for k in ('absolute_rms', 'free_rms', 'relative_rms')):
        raise FloatingPointError('Nonfinite training displacement')
    return measured


def search_next_beta(trials, target, spec):
    brackets = [(low['beta'], high['beta']) for low in trials for high in trials
                if low['beta'] < high['beta'] and low['relative_rms'] < target < high['relative_rms']]
    if brackets:
        lower, upper = min(brackets, key=lambda pair: (math.log(pair[1]) - math.log(pair[0]), pair))
        return math.sqrt(lower) * math.sqrt(upper)
    last = trials[-1]
    factor = target / max(last['relative_rms'], spec['rms_floor'])
    factor = min(spec['factor_bounds'][1], max(spec['factor_bounds'][0], factor))
    return min(spec['beta_bounds'][1], max(spec['beta_bounds'][0], last['beta'] * factor))


def select_current_beta(measure, beta, target, spec):
    """Cache only best/current gradients; history contains scalar measurements."""
    starting_beta, trials, best = beta, [], None
    reason = 'evaluation_cap'
    for _ in range(spec['max_evaluations']):
        gradients, measured = measure(beta)
        if not all(torch.isfinite(gradient).all() for gradient in gradients):
            raise FloatingPointError('Nonfinite parameter gradient')
        relative = measured['relative_rms']
        error = abs(relative / target - 1)
        score = abs(math.log(max(relative, spec['rms_floor'])) - math.log(target))
        trials.append(dict(beta=beta, **measured, target_error=error, log_error=score))
        key = (score, beta)
        if best is None or key < best[0]:
            best = (key, gradients, measured, len(trials) - 1)
        if error <= spec['relative_tolerance']:
            best = (key, gradients, measured, len(trials) - 1)
            reason = 'tolerance'
            break
        if len(trials) == spec['max_evaluations']:
            break
        beta = search_next_beta(trials, target, spec)
        if any(beta == trial['beta'] for trial in trials):
            reason = 'repeated_beta'
            break
    chosen = trials[best[3]]
    record = dict(starting_beta=starting_beta, selected_beta=chosen['beta'], trials=trials,
                  chosen_index=best[3], target_error=chosen['target_error'], log_error=chosen['log_error'],
                  tolerance_miss=chosen['target_error'] > spec['relative_tolerance'],
                  pair_count=len(trials), stop_reason=reason)
    return best[1], dict(best[2], search=record)


def validate_beta_search_record(record, starting_beta, target, spec):
    """Check bounded trial coverage and selection, without replaying a solver."""
    trials = record['trials']
    if (not 1 <= len(trials) <= spec['max_evaluations'] or type(record['pair_count']) is not int
            or record['pair_count'] != len(trials) or type(record['chosen_index']) is not int
            or type(record['tolerance_miss']) is not bool):
        raise ValueError('Invalid beta search pair count')
    if record['starting_beta'] != starting_beta or trials[0]['beta'] != starting_beta:
        raise ValueError('Beta search starting beta mismatch')
    tested = set()
    for i, trial in enumerate(trials):
        beta, relative = trial['beta'], trial['relative_rms']
        if (not math.isfinite(beta) or not spec['beta_bounds'][0] <= beta <= spec['beta_bounds'][1]
                or beta in tested or not math.isfinite(relative) or relative < 0
                or not math.isfinite(trial['free_rms']) or trial['free_rms'] <= 0
                or not math.isfinite(trial['absolute_rms']) or trial['absolute_rms'] < 0):
            raise ValueError('Invalid beta search trial')
        tested.add(beta)
        if i and not math.isclose(beta, search_next_beta(trials[:i], target, spec), rel_tol=1e-12, abs_tol=0):
            raise ValueError('Beta search proposal mismatch')
        error = abs(relative / target - 1)
        score = abs(math.log(max(relative, spec['rms_floor'])) - math.log(target))
        if not math.isclose(trial['target_error'], error, rel_tol=1e-12, abs_tol=1e-15) or not math.isclose(trial['log_error'], score, rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError('Beta search target error mismatch')
        if i < len(trials) - 1 and error <= spec['relative_tolerance']:
            raise ValueError('Beta search continued after tolerance was reached')
    met = trials[-1]['target_error'] <= spec['relative_tolerance']
    chosen_index = len(trials) - 1 if met else min(range(len(trials)), key=lambda i: (trials[i]['log_error'], trials[i]['beta']))
    chosen = trials[chosen_index]
    expected_reason = 'tolerance' if met else 'evaluation_cap' if len(trials) == spec['max_evaluations'] else 'repeated_beta'
    if (record['chosen_index'] != chosen_index or record['selected_beta'] != chosen['beta']
            or record['target_error'] != chosen['target_error'] or record['log_error'] != chosen['log_error']
            or record['tolerance_miss'] != (not met) or record['stop_reason'] != expected_reason):
        raise ValueError('Beta search selection mismatch')
    if expected_reason == 'repeated_beta' and search_next_beta(trials, target, spec) not in tested:
        raise ValueError('Beta search stopped before its declared cap')


def hybrid_backward(model, x, y, iterations, betas, *, measure_displacement=False, beta_search=None,
                    skip_conv_autograd=None, cache_conv_coefficients=None, force_mask='none',
                    mask_records=None):
    """Backpropagate boundary forces/digital gradients, then estimate conv gradients.

    ``force_mask='inactive_output_channels'`` removes the injected force from
    block-output channels that are off for the whole minibatch; their diode
    clamps give BPTT zero derivative there. One record per block is appended
    to ``mask_records`` when a list is supplied.

    Float64 convolution conductances remain constants in the free graph by
    default: their autograd gradients would be discarded for the EqProp pair.
    Set ``skip_conv_autograd=False`` to retain that original computation for A/B
    profiling. This does not truncate derivatives through the free solver states.
    An untracked input with a frozen first input gain uses the original path so
    its physical output remains connected to autograd for boundary-force capture.
    Constant float64 coefficients are scoped to this update when conv autograd
    is skipped; set ``cache_conv_coefficients=False`` for an uncached A/B run.
    """
    if not model.training:
        raise ValueError('Hybrid training requires training mode')
    if beta_search is not None:
        validate_training_beta_case(dict(method='hybrid_eqprop', betas=list(betas), beta_search=beta_search))
        if not measure_displacement:
            raise ValueError('Current-batch beta search requires displacement measurement')
    saved, outputs, handles = [], [], []
    if skip_conv_autograd is None:
        skip_conv_autograd = x.dtype == torch.float64
    skip_conv_autograd = skip_conv_autograd and (
        x.requires_grad or model.analog_blocks[0].input_gain.requires_grad)
    if cache_conv_coefficients is None:
        cache_conv_coefficients = True
    cache_conv_coefficients = cache_conv_coefficients and skip_conv_autograd and x.dtype == torch.float64
    conv_flags = [(weight.state, weight.state.requires_grad)
                  for block in model.analog_blocks for weight in block.weights]
    def capture(block, args, output):
        physical = block.free_layers()[-1].state
        physical.retain_grad()
        outputs.append(physical)
        saved.append(dict(input=block.boundary_layer.state.detach().clone(),
                          states=[layer.state.detach().clone() for layer in block.free_layers()]))
    coefficient_caches = ExitStack()
    try:
        if cache_conv_coefficients:
            for block in model.analog_blocks:
                for interaction in block.interactions:
                    coefficient_caches.enter_context(interaction.cache_a_col2im())
        handles = [block.register_forward_hook(capture) for block in model.analog_blocks]
        try:
            if skip_conv_autograd:
                for parameter, _ in conv_flags:
                    parameter.requires_grad_(False)
            logits = model(x, iterations, reset=True)
            loss = F.cross_entropy(logits, y, reduction='sum')
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite hybrid loss')
            loss.backward()
        finally:
            # trainable_tensors() filters requires_grad; restore before gradient
            # normalization/validation and before handing tensors to EqProp.
            for parameter, requires_grad in conv_flags:
                parameter.requires_grad_(requires_grad)
        for record, output in zip(saved, outputs):
            if output.grad is None or not torch.isfinite(output.grad).all():
                raise FloatingPointError('Missing/nonfinite boundary force')
            # The solver injects the derivative of summed CE; the estimator
            # already divides by the actual batch size, including a short tail.
            record['gradient'], mask = mask_output_force(output.grad.detach(), record['states'][-1], force_mask)
            if mask_records is not None:
                mask_records.append(mask)
        for parameter in model.trainable_tensors().values():
            if parameter.grad is not None:
                parameter.grad.div_(len(y))
        prediction, scalar_loss = logits.detach(), loss.detach() / len(y)
        del logits, loss, outputs
        measurements = []
        for i, (block, record, count, beta) in enumerate(zip(model.analog_blocks, saved, iterations, betas)):
            if beta_search is not None:
                def measure(candidate):
                    gradients, displacement = eqprop_gradients(block, record, candidate, count, x.device,
                                                               cpu_output=False, measure_displacement=True)
                    return gradients, displacement_rms(displacement)
                gradients, measured = select_current_beta(measure, beta, beta_search['targets'][i], beta_search)
            else:
                gradients, displacement = eqprop_gradients(block, record, beta, count, x.device,
                                               cpu_output=False, measure_displacement=measure_displacement)
                if measure_displacement:
                    measured = displacement_rms(displacement)
            if measure_displacement:
                measurements.append(measured)
            for weight, grad in zip(block.weights, gradients):
                weight.state.grad = grad.to(dtype=weight.state.dtype)
        # Check the actual update tensors on every batch, including unchanged
        # autograd gradients for the dense head, BN and gains.
        finite = [torch.isfinite(p.grad).all() for p in model.trainable_tensors().values()
                  if p.grad is not None]
        if len(finite) != len(model.trainable_tensors()) or not torch.stack(finite).all():
            raise FloatingPointError('Missing/nonfinite hybrid parameter gradient')
        return (prediction, scalar_loss, measurements) if measure_displacement else (prediction, scalar_loss)
    finally:
        coefficient_caches.close()
        for handle in handles:
            handle.remove()
        model.detach_state_()


@contextmanager
def replay_without_training_effects(model):
    modes = [(module, module.training) for module in model.modules()]
    buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
    variables = [v for block in model.analog_blocks for v in (block.boundary_layer, *block.free_layers())]
    variables.extend((model.head.boundary, model.head.output))
    # Solvers replace their state tensors. Retain detached references so replay
    # does not leave its cohort states in the training model or duplicate VRAM.
    states = [(variable, variable.state.detach()) for variable in variables]
    rng = (torch.get_rng_state(), np.random.get_state(), random.getstate(),
           torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)
    try:
        yield
        for name, value in model.named_buffers():
            if not torch.equal(value, buffers[name]):
                raise RuntimeError(f'Replay changed buffer {name}')
    finally:
        with torch.no_grad():
            for name, value in model.named_buffers():
                value.copy_(buffers[name])
        for module, training in modes:
            module.training = training
        model.detach_state_()
        for variable, state in states:
            variable.state = state
        torch.set_rng_state(rng[0]); np.random.set_state(rng[1]); random.setstate(rng[2])
        if rng[3] is not None:
            torch.cuda.set_rng_state_all(rng[3])


def calibration_updates(batches, count):
    """Completed optimizer updates at which to calibrate, before the next update."""
    if type(batches) is not int or type(count) is not int or not 1 <= count <= batches:
        raise ValueError('Calibration count must fit the epoch optimizer updates')
    return [(index * batches + count - 1) // count for index in range(count)]


class EpochBeta:
    def __init__(self, model, config, batches, device, run_dir, deadline):
        self.model, self.config = model, config
        self.spec = validate_spec(config)
        self.batches, self.device, self.run_dir, self.deadline = batches, device, run_dir, deadline
        model.conv_beta_state = initial_beta_state(config)
        self.last_calibration = None

    @property
    def state(self):
        return self.model.conv_beta_state

    def force_mask_kwargs(self):
        # Leave the original capture call unchanged unless masking is requested.
        mask = self.spec.get('force_mask', 'none')
        return {} if mask == 'none' else {'force_mask': mask}

    def check_time(self):
        if time.monotonic() > self.deadline:
            raise TimeoutError('Beta measurement/calibration exceeded run deadline')

    def typed_batches(self):
        dtype = self.model.head.weight.state.dtype
        for x, y in self.batches:
            yield x.to(dtype=dtype), y

    def before_epoch(self, epoch):
        if self.spec['policy'] == 'periodic':
            raise ValueError('Periodic calibration requires before_update and the epoch batch count')
        if self.spec['policy'] == 'fixed' or epoch <= self.state['calibrated_for_epoch']:
            return 0.
        return self.calibrate(epoch)

    def before_update(self, epoch, completed_updates, batches):
        if self.spec['policy'] != 'periodic':
            return self.before_epoch(epoch) if completed_updates == 0 else 0.
        updates = calibration_updates(batches, self.spec['calibrations_per_epoch'])
        if completed_updates not in updates:
            return 0.
        if (self.state['calibrated_for_epoch'] == epoch and
                self.state['last_calibration_completed_updates'] == completed_updates):
            return 0.
        expected_count = (epoch - 1) * len(updates) + updates.index(completed_updates)
        if self.state['calibration_count'] != expected_count:
            raise ValueError('Missed or out-of-order periodic beta calibration')
        return self.calibrate(epoch, completed_updates, batches)

    def calibrate(self, epoch, completed_updates=0, batches=None):
        start = time.monotonic()
        history, values = [], []
        periodic = self.spec['policy'] == 'periodic'
        suffix = f'_u{completed_updates:05}' if periodic else ''
        path = self.run_dir / 'artifacts' / f'beta_calibration_e{epoch:03}{suffix}.json'
        progress = dict(epoch=epoch, completed_updates=completed_updates,
                        batches=batches, optimizer_progress=completed_updates / batches if batches else 0.,
                        starting_betas=list(self.state['values']))
        with replay_without_training_effects(self.model):
            self.check_time()
            records = capture_batches(self.model, self.typed_batches(), self.config['iterations'], self.device,
                                      **self.force_mask_kwargs())
            for i, block in enumerate(self.model.analog_blocks):
                cost = FrozenOutputCost(block.free_layers()[-1])
                aug = AugmentedFunction(block.energy, cost, nudging_mode='current', current_scale=1.)
                minimizer = nudged_minimizer(block, aug, self.config['iterations'][i])
                trials = []
                def measure(beta):
                    self.check_time()
                    result = measure_block(block, minimizer, aug, cost, records, i, beta, self.device)
                    trials.append(dict(beta=beta, **result))
                    reporting.atomic_write_json(path, dict(**progress, complete=False,
                        completed_blocks=history, current_block=i+1, trials=trials))
                    reporting.update_status_progress(self.run_dir, dict(stage='beta_calibration',
                        **progress, block=i+1, evaluations=len(trials), beta=beta))
                    return result
                beta, attempts = search_beta(measure, self.spec['rms_targets'][i], metric='relative_rms',
                    tolerance=self.spec['relative_tolerance'], max_evaluations=self.spec['max_evaluations'],
                    initial_beta=self.state['values'][i])
                values.append(beta)
                history.append(dict(block=i+1, beta=beta, target=self.spec['rms_targets'][i], trials=attempts))
        # Commit the vector only when all three searches succeed.
        self.state.update(values=values, calibrated_for_epoch=epoch, initialization_reused=False)
        self.state['calibration_count'] = self.state.get('calibration_count', 0) + 1
        if periodic:
            self.state.update(last_calibration_completed_updates=completed_updates,
                              batches_in_calibrated_epoch=batches)
        seconds = time.monotonic()-start
        self.last_calibration = dict(**progress, betas=list(values), elapsed_seconds=seconds,
                                     calibration_count=self.state.get('calibration_count'), artifact=str(path))
        reporting.atomic_write_json(path, dict(**progress, complete=True, blocks=history,
                                              beta_state=self.state, elapsed_seconds=seconds))
        return seconds

    def observe(self, epoch):
        """Actual beta, RMS and requested cosine checkpoints; no retuning."""
        if 'observation_epochs' in self.spec and epoch not in self.spec['observation_epochs']:
            return None
        start = time.monotonic()
        cosine = epoch in self.spec.get('cosine_epochs', [])
        sums = [dict(centered_squares=0., free_squares=0., elements=0, unchanged_elements=0) for _ in range(3)]
        pooled, batch_rows, count = {}, [], 0
        names = [name for name, _ in self.model.named_conductances() if name.startswith('blocks.')]
        with replay_without_training_effects(self.model):
            for batch, (x, y) in enumerate(self.typed_batches()):
                self.check_time()
                saved = capture_batches(self.model, [(x,y)], self.config['iterations'], self.device,
                                        **self.force_mask_kwargs())[0]
                if cosine:
                    with batchnorm_minibatch_stats_only(self.model):
                        _, tracked = self.model.bptt(x.to(self.device), self.config['iterations'])
                        native = torch.autograd.grad(F.cross_entropy(tracked, y.to(self.device)),
                            [p for name,p in self.model.named_conductances() if name.startswith('blocks.')])
                    native = [g.detach().cpu().double() for g in native]
                    del tracked
                    self.model.detach_state_()
                offset = 0
                for i, block in enumerate(self.model.analog_blocks):
                    if cosine:
                        local = local_bptt_gradients(block, saved[i], self.config['iterations'][i], self.device)
                        estimate, displacement = eqprop_gradients(block, saved[i], self.state['values'][i],
                                                                  self.config['iterations'][i], self.device)
                    else:
                        cost = FrozenOutputCost(block.free_layers()[-1])
                        aug = AugmentedFunction(block.energy,cost,nudging_mode='current',current_scale=1.)
                        minimizer = nudged_minimizer(block,aug,self.config['iterations'][i])
                        rms = measure_block(block,minimizer,aug,cost,[saved],i,self.state['values'][i],self.device)
                        displacement = dict(centered_squares=rms['absolute_rms']**2*rms['elements'],
                            free_squares=rms['free_rms']**2*rms['elements'],elements=rms['elements'])
                    for key,value in displacement.items():sums[i][key] += value
                    if cosine:
                        for j, ep in enumerate(estimate):
                            name = names[offset+j]
                            row = dict(batch=batch, name=name, block=i+1,
                                native_bptt=agreement(ep, native[offset+j]), local_bptt=agreement(ep, local[j]))
                            batch_rows.append(row)
                            if name not in pooled:
                                pooled[name] = {key:g*len(y) for key,g in [('ep',ep),('native',native[offset+j]),('local',local[j])]}
                            else:
                                for key,g in [('ep',ep),('native',native[offset+j]),('local',local[j])]:
                                    pooled[name][key].add_(g,alpha=len(y))
                    offset += len(block.weights)
                count += len(y)
                reporting.update_status_progress(self.run_dir,dict(stage='beta_observation',epoch=epoch,
                                                                   examples=count,cosine=cosine))
        blocks = [dict(block=i+1, beta=self.state['values'][i],
                       **pooled_rms(s['centered_squares'],s['free_squares'],s['elements']),
                       unchanged_fraction=s['unchanged_elements']/s['elements'] if cosine else None) for i,s in enumerate(sums)]
        layers = [dict(name=name,native_bptt=agreement(p['ep']/count,p['native']/count),
                       local_bptt=agreement(p['ep']/count,p['local']/count)) for name,p in pooled.items()]
        result = dict(epoch=epoch, examples=count, beta_state=copy.deepcopy(self.state), blocks=blocks,
                      layers=layers, batches=batch_rows, elapsed_seconds=time.monotonic()-start)
        reporting.atomic_write_json(self.run_dir/'artifacts'/f'beta_boundary_e{epoch:03}.json',result)
        return result
