"""Read-only CIFAR centered EqProp versus native and matched-local BPTT."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from experiments import reporting
from experiments.calibrate_cifar_block_beta import (FrozenOutputCost, capture_batches,
    measure_block, nudged_minimizer, pooled_rms, search_beta, validate_force_mask)
from experiments.train_cifar_l8_analog import Cohort, deterministic_setup, loader
from experiments.train_cifar_l8_analog import make_split, read_training_data
from labs.cifar_l8_analog import CifarL8Analog, batchnorm_minibatch_stats_only
from model.resistive.interaction import ConvResistive
from training.sgd import AugmentedFunction, EquilibriumProp


def bounded_conv_energy_gradient(interaction):
    """Same mean squared voltage-difference derivative without an N*L*O*K tensor."""
    weight, out_channels, in_channels, kh, kw, *_ = interaction._conv_geometry()
    x = interaction._layer_pre.state
    if interaction._layer_pre.name != 'Layer_0':
        x = x * interaction._voltage_amp_value(like=x)
    y = interaction._layer_post.state
    cols = F.unfold(x, (kh, kw), padding=interaction._P, stride=interaction._S,
                    dilation=interaction._D)
    targets = y.flatten(2)
    pre_square = cols.square().sum(dim=(0, 2)).unsqueeze(0)
    post_square = targets.square().sum(dim=(0, 2)).unsqueeze(1)
    cross = torch.einsum('nkl,nol->ok', cols, targets)
    gradient = (pre_square + post_square - 2 * cross) * (.5 / y.shape[0])
    return gradient.reshape_as(weight) * interaction._amp_factor()


def stable_centered_conv_gradient(interaction, minus, plus, beta):
    """Factor the difference of squares before reduction; exact centered EP formula.

    Subtracting separately reduced energies loses the weak first-layer response
    against a large fixed input-voltage squared term. Here that term cancels
    algebraically, before any floating-point reduction.
    """
    fn = interaction
    def phase(layer, states):
        return states.get(layer.name, layer.state)
    xm, xp = phase(fn._layer_pre, minus), phase(fn._layer_pre, plus)
    ym, yp = phase(fn._layer_post, minus), phase(fn._layer_post, plus)
    pre_scale = fn._pre_scale()
    dx = (xp - xm) * pre_scale
    mx = ((xp + xm) * .5) * pre_scale
    dy, my = (yp - ym).flatten(2), ((yp + ym) * .5).flatten(2)
    weight = fn._weight.state
    geometry = dict(kernel_size=weight.shape[-2:], padding=fn._P, stride=fn._S, dilation=fn._D)
    dx, mx = F.unfold(dx, **geometry), F.unfold(mx, **geometry)
    value = (dx * mx).sum(dim=(0, 2)).unsqueeze(0) + (dy * my).sum(dim=(0, 2)).unsqueeze(1)
    value = value - torch.einsum('nkl,nol->ok', dx, my) - torch.einsum('nkl,nol->ok', mx, dy)
    return value.reshape_as(weight) * (fn._amp_factor() / (2 * beta * yp.shape[0]))


def restore_block(block, saved, device):
    block.boundary_layer.state = saved['input'].to(device).detach().clone()
    for layer, state in zip(block.free_layers(), saved['states']):
        layer.state = state.to(device).detach().clone()


def local_bptt_gradients(block, saved, iterations, device):
    """One matched reference, reusable across beta values for this batch/block."""
    restore_block(block, saved, device)
    block.minimizer.num_iterations = iterations
    block.minimizer.compute_equilibrium()
    force = saved['gradient'].to(device)
    objective = (block.free_layers()[-1].state * force).sum() / force.shape[0]
    reference = torch.autograd.grad(objective, [w.state for w in block.weights])
    reference = [g.detach().cpu().double() for g in reference]
    del objective
    block.detach_state_()
    return reference


def eqprop_gradients(block, saved, beta, iterations, device, *, cpu_output=True,
                     measure_displacement=True, measure_layer_displacement=False,
                     phase_mode='sequential'):
    """Centered EP from native-K endpoints; optionally solve both signs as one batch."""
    if phase_mode not in ('sequential', 'packed'):
        raise ValueError(f'Unknown EqProp phase_mode: {phase_mode!r}')
    if measure_layer_displacement and not measure_displacement:
        raise ValueError('Layer displacement requires displacement measurement')
    restore_block(block, saved, device)
    cost = FrozenOutputCost(block.free_layers()[-1])
    cost.gradient = saved['gradient'].to(device)
    augmented = AugmentedFunction(block.energy, cost, nudging_mode='current', current_scale=1.)
    minimizer = nudged_minimizer(block, augmented, iterations)
    estimator = EquilibriumProp(block.weights, block.free_layers(), augmented, cost,
                                minimizer, variant='centered', nudging=beta)
    # Only ConvResistive depends on each ConvWeight; keep native phase/sign/scaling.
    interactions = [fn for fn in block.energy._interactions if isinstance(fn, ConvResistive)]
    assert len(interactions) == len(block.weights)
    ordered = []
    for weight in block.weights:
        interaction, = [fn for fn in interactions if fn._weight is weight]
        ordered.append(interaction)
    displacement = {}
    def parameter_gradients(minus, plus):
        dtype = block.weights[0].state.dtype
        assert all(t.dtype == dtype for t in [*minus.values(), *plus.values()]), 'Mixed endpoint precision'
        if measure_displacement:
            output_name = block.free_layers()[-1].name
            centered = (plus[output_name].double() - minus[output_name].double()) * .5
            free = saved['states'][-1].double()
            displacement.update(centered_squares=float(centered.square().sum()),
                                free_squares=float(free.square().sum()), elements=free.numel(),
                                unchanged_elements=int((centered == 0).sum()))
            if measure_layer_displacement:
                layers = []
                for index, (layer, initial) in enumerate(zip(block.free_layers(), saved['states'])):
                    delta = (plus[layer.name].double() - minus[layer.name].double()) * .5
                    free_state = initial.double()
                    centered_squares = float(delta.square().sum())
                    free_squares = float(free_state.square().sum())
                    elements = free_state.numel()
                    unchanged = int((delta == 0).sum())
                    layers.append(dict(layer=index + 1, state_name=layer.name,
                        centered_squares=centered_squares, free_squares=free_squares,
                        elements=elements, unchanged_elements=unchanged,
                        absolute_rms=(centered_squares / elements) ** .5,
                        free_rms=(free_squares / elements) ** .5,
                        relative_rms=(centered_squares / free_squares) ** .5 if free_squares else None,
                        unchanged_fraction=unchanged / elements))
                force_rms = float(saved['gradient'].double().square().mean().sqrt())
                displacement.update(layer_displacements=layers, output_force_rms=force_rms,
                                    injected_force_rms=abs(beta) * force_rms)
        gradients = [stable_centered_conv_gradient(fn, minus, plus, beta) for fn in ordered]
        assert all(g.dtype == dtype for g in gradients), 'Mixed gradient precision'
        return gradients
    estimator._standard_param_grads = parameter_gradients
    with torch.no_grad():
        if phase_mode == 'sequential':
            estimate = [g.detach() for g in estimator.compute_gradient()]
        else:
            boundary = block.boundary_layer.state
            layers = block.free_layers()
            free_states = [layer.state for layer in layers]
            batch_size = boundary.shape[0]
            restored_states = free_states
            try:
                block.boundary_layer.state = torch.cat((boundary, boundary), dim=0)
                for layer, state in zip(layers, free_states):
                    layer.state = torch.cat((state, state), dim=0)
                # Current nudging adds beta * dC/dy to b. Order the packed
                # halves as minus/plus while keeping scalar nudging positive.
                cost.gradient = torch.cat((-cost.gradient, cost.gradient), dim=0)
                augmented.prepare_nudging()
                augmented.nudging = beta
                endpoints = minimizer.compute_equilibrium()
                minus = {name: state[:batch_size] for name, state in endpoints.items()}
                plus = {name: state[batch_size:] for name, state in endpoints.items()}
                # The first convolution reads the boundary directly. Restore B
                # before reduction, so normalization remains the original batch.
                block.boundary_layer.state = boundary
                restored_states = [plus[layer.name].clone() for layer in layers]
                for layer, state in zip(layers, restored_states):
                    layer.state = state
                estimate = [g.detach() for g in parameter_gradients(minus, plus)]
            finally:
                block.boundary_layer.state = boundary
                for layer, state in zip(layers, restored_states):
                    layer.state = state
        if cpu_output:
            estimate = [g.cpu().double() for g in estimate]
    block.detach_state_()
    return estimate, displacement


def reference_record(saved):
    """BPTT references always use the captured force, never a masked nudging force."""
    return dict(saved, gradient=saved['unmasked_gradient']) if 'unmasked_gradient' in saved else saved


def local_gradients(block, saved, beta, iterations, device):
    """Compatibility helper for one-beta numerical comparisons."""
    reference = local_bptt_gradients(block, saved, iterations, device)
    estimate, _ = eqprop_gradients(block, saved, beta, iterations, device)
    return estimate, reference


def agreement(estimate, reference):
    a, b = estimate.reshape(-1), reference.reshape(-1)
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise FloatingPointError('Nonfinite parameter gradient')
    na, nb = float(a.norm()), float(b.norm())
    return dict(cosine=float(torch.dot(a, b) / (na * nb)) if na and nb else None,
                eqprop_norm=na, bptt_norm=nb, norm_ratio=na / nb if nb else None,
                eqprop_rms=na / math.sqrt(a.numel()), bptt_rms=nb / math.sqrt(b.numel()),
                zero_epsilon=1e-12,
                eqprop_below_epsilon_fraction=float((a.abs() <= 1e-12).double().mean()),
                bptt_below_epsilon_fraction=float((b.abs() <= 1e-12).double().mean()),
                eqprop_zero_fraction=float((a == 0).double().mean()),
                bptt_zero_fraction=float((b == 0).double().mean()))


DISPLACEMENT_SUMS = ('centered_squares', 'free_squares', 'elements', 'unchanged_elements')


def accumulate_displacement(total, measured):
    """Pool raw endpoint sums, never averages of batch RMS values."""
    for key in DISPLACEMENT_SUMS:
        total[key] = total.get(key, 0) + measured[key]
    if 'layer_displacements' not in measured:
        return
    if 'layer_displacements' not in total:
        total['layer_displacements'] = [dict(layer=row['layer'], state_name=row['state_name'])
                                        for row in measured['layer_displacements']]
    assert [(r['layer'], r['state_name']) for r in total['layer_displacements']] == [
        (r['layer'], r['state_name']) for r in measured['layer_displacements']]
    for pooled, row in zip(total['layer_displacements'], measured['layer_displacements']):
        for key in DISPLACEMENT_SUMS:
            pooled[key] = pooled.get(key, 0) + row[key]
    # The force acts on the block output, whose size can differ from this weight's state.
    for field in ('output_force', 'injected_force'):
        key = field + '_squares'
        total[key] = total.get(key, 0.) + measured[field + '_rms'] ** 2 * measured['elements']


def state_measurement_fields(displacement, state_name):
    """Attach a weight's postsynaptic state and the block's output-force RMS."""
    raw, = [row for row in displacement['layer_displacements'] if row['state_name'] == state_name]
    row = {key: raw[key] for key in ('layer', 'state_name', *DISPLACEMENT_SUMS)}
    if row['elements'] <= 0:
        raise ValueError('State displacement requires nonempty states')
    if not all(math.isfinite(row[key]) for key in DISPLACEMENT_SUMS):
        raise FloatingPointError('Nonfinite state displacement')
    row.update(absolute_rms=math.sqrt(row['centered_squares'] / row['elements']),
        free_rms=math.sqrt(row['free_squares'] / row['elements']),
        relative_rms=(math.sqrt(row['centered_squares'] / row['free_squares'])
                      if row['free_squares'] else None),
        unchanged_fraction=row['unchanged_elements'] / row['elements'])
    result = dict(state_displacement=row, force_elements=displacement['elements'])
    for field in ('output_force', 'injected_force'):
        result[field + '_rms'] = (math.sqrt(displacement[field + '_squares'] / displacement['elements'])
            if field + '_squares' in displacement else displacement[field + '_rms'])
        if not math.isfinite(result[field + '_rms']):
            raise FloatingPointError('Nonfinite output force')
    return result


def check_environment(expected):
    """Fail before replay when a declared scientific runtime does not match."""
    actual = {'torch': str(torch.__version__), 'cuda': torch.version.cuda}
    for key, value in (expected or {}).items():
        if key not in actual:
            raise ValueError(f'Unsupported expected_environment key: {key}')
        if actual[key] != value:
            raise RuntimeError(f'Environment mismatch for {key}: expected {value!r}, got {actual[key]!r}')
    return actual


def calibrate_displacement_vectors(model, records, config, cfg, device, out, start, base_betas,
                                   blocks=(1, 2, 3)):
    """Find scheme-specific injected B values on the same fixed full cohort.

    Blocks outside ``blocks`` are not compared, so they keep an unused unit
    multiplier instead of spending calibration solves.
    """
    targets = cfg['relative_rms_targets']
    calibration = cfg['displacement_calibration']
    vectors = [[None] * 3 for _ in targets]
    histories = []
    for i, block in enumerate(model.analog_blocks):
        if i + 1 not in blocks:
            for vector in vectors:
                vector[i] = 1.
            continue
        cost = FrozenOutputCost(block.free_layers()[-1])
        augmented = AugmentedFunction(block.energy, cost, nudging_mode='current', current_scale=1.)
        minimizer = nudged_minimizer(block, augmented, config['iterations'][i])
        previous_beta = calibration['initial_beta_guesses'][i]
        previous_rms = calibration['initial_guess_relative_rms']
        for case, target in enumerate(targets):
            def measure(beta):
                if time.monotonic() - start > cfg['budget_seconds']:
                    raise TimeoutError('Scheme beta calibration budget exhausted')
                value = measure_block(block, minimizer, augmented, cost, records, i, beta, device)
                reporting.append_metric(out / 'metrics.jsonl', dict(stage='displacement_calibration',
                    block=i + 1, case_index=case, target_relative_rms=target, beta_injected_B=beta, **value))
                reporting.update_status_progress(out, dict(stage='displacement_calibration', block=i + 1,
                    case_index=case, target_relative_rms=target, beta_injected_B=beta))
                return value
            beta, history = search_beta(measure, target, metric='relative_rms',
                tolerance=calibration['relative_tolerance'], max_evaluations=calibration['max_evaluations'],
                initial_beta=previous_beta * target / previous_rms)
            previous_beta, previous_rms = beta, history[-1]['relative_rms']
            vectors[case][i] = beta / base_betas[i]
            histories.append(dict(block=i + 1, case_index=case, target_relative_rms=target,
                beta_injected_B=beta, achieved=history[-1], history=history))
            reporting.atomic_write_json(out / 'artifacts/displacement_calibration.json', histories)
            print('CALIBRATED', i + 1, target, beta, previous_rms, flush=True)
    reporting.atomic_write_json(out / 'artifacts/resolved_beta_vectors.json',
        dict(base_betas=base_betas, beta_multiplier_vectors=vectors, relative_rms_targets=targets,
             **({'calibrated_blocks': list(blocks)} if tuple(blocks) != (1, 2, 3) else {})))
    return vectors


def run(args):
    # Reuse the full-model conversion verified in the matched precision study.
    from experiments.replay_cifar_precision_response import convert_model, verify_dtype, check_snapshot
    root = args.input_root
    cfg = json.loads(args.config.read_text())
    precision = cfg.get('precision', 'float32')
    assert precision in ('float32', 'float64'), 'Unsupported replay precision'
    dtype = getattr(torch, precision)
    environment = check_environment(cfg.get('expected_environment'))
    measure_layers = cfg.get('measure_layer_displacement', False)
    force_mask = validate_force_mask(cfg.get('force_mask', 'none'))
    config = json.loads((root / 'inputs/model_config.json').read_text())
    identity = json.loads(args.source_identity.read_text())
    checkpoint_file = cfg.get('checkpoint_file', 'inputs/initial_model.pt')
    checkpoint_epoch = cfg.get('checkpoint_epoch', cfg.get('initializer_epoch', 0))
    assert checkpoint_file in cfg['input_sha256'], 'Checkpoint hash must be declared'
    checkpoint_path = root / checkpoint_file
    source = Path(__file__).resolve().parents[1]
    for name, digest in identity['source_sha256'].items():
        assert reporting.sha256_file(source / name) == digest, name
    for name, digest in cfg['input_sha256'].items():
        assert reporting.sha256_file(root / name) == digest, name
    deterministic_setup(config['seed'])
    device = torch.device(args.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('Assigned comparison requires a CUDA GPU')
    out = args.output_dir
    reporting.start_run(out, dict(study_id=cfg['study_id'], run_id=out.name,
        arm_id=config['scheme'] + '_block_beta_gradients', seed=config['seed'], smoke=False,
        scheme=config['scheme'], voltage_amp=config['solver']['voltage_amp'],
        current_amp=config['solver']['current_amp'],
        block_output_normalization=config.get('block_output_normalization', 'none'),
        checkpoint_epoch=checkpoint_epoch, checkpoint_file=checkpoint_file,
        checkpoint_sha256=cfg['input_sha256'][checkpoint_file],
        evidence_class='cifar_finite_iteration_gradient_comparison', resolved_config=cfg,
        source_identity=identity, gpu=torch.cuda.get_device_name(device),
        environment=environment, precision=precision,
        **({'force_mask': force_mask} if force_mask != 'none' else {})))
    start = time.monotonic()
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        assert checkpoint['epoch'] == checkpoint_epoch, 'Wrong checkpoint epoch'
        model = CifarL8Analog(config, device)
        model.restore(checkpoint['model'])
        convert_model(model, dtype)
        before = model.snapshot()
        check_snapshot(checkpoint['model'], before)
        del checkpoint
        model.train()
        cohort = json.loads((root / 'inputs/cohort.json').read_text())
        indices = cohort['indices']
        assert len(indices) == len(set(indices)) == 256
        if 'base_betas' in cfg:
            betas = cfg['base_betas']
        else:
            matches = json.loads((root / 'inputs/matches.json').read_text())
            betas = [row['beta_injected_B'] for row in matches]
        assert len(betas) == 3 and all(b > 0 for b in betas)
        multipliers = cfg.get('beta_multipliers', [1.])
        multiplier_vectors = cfg.get('beta_multiplier_vectors', [[v] * 3 for v in multipliers])
        targets = cfg.get('relative_rms_targets', [None] * len(multiplier_vectors))
        images, labels = read_training_data(args.dataset_root)
        split = make_split(labels, config['split']['seed'])
        assert set(indices) <= set(split['train'])
        data = Cohort(images, labels, indices, config, augment=False)
        def typed_batches():
            # Preprocess identically in float32, then exactly promote image values.
            for x, y in loader(data, cfg['batch_size']):
                yield x.to(dtype=dtype), y
        selected_blocks = cfg.get('diagnostic_blocks', [1, 2, 3])
        assert selected_blocks and len(set(selected_blocks)) == len(selected_blocks)
        assert set(selected_blocks) <= {1, 2, 3}
        cached_records = None
        if 'displacement_calibration' in cfg:
            assert 'nudged_iteration_vectors' not in cfg, 'Use fixed-beta cases for a nudged-K sweep'
            cached_records = capture_batches(model, typed_batches(), config['iterations'], device,
                                             force_mask=force_mask, keep_unmasked=True)
            verify_dtype(model, dtype)
            assert all(t.dtype == dtype for batch in cached_records for saved in batch
                       for t in (saved['input'], saved['gradient'], *saved['states'])), 'Mixed saved-state precision'
            multiplier_vectors = calibrate_displacement_vectors(model, cached_records, config, cfg,
                device, out, start, betas, blocks=selected_blocks)
        assert all(len(v) == 3 and all(x > 0 for x in v) for v in multiplier_vectors)
        nudged_vectors = cfg.get('nudged_iteration_vectors', [config['iterations']] * len(multiplier_vectors))
        assert len(nudged_vectors) == len(multiplier_vectors)
        assert all(len(v) == 3 and all(isinstance(k, int) and k > 0 for k in v) for v in nudged_vectors)
        assert len(multiplier_vectors) == len({(tuple(v), tuple(k)) for v, k in zip(multiplier_vectors, nudged_vectors)})
        expected_weights = sum(len(model.analog_blocks[b - 1].weights) for b in selected_blocks)
        assert len(targets) == len(multiplier_vectors)
        names_params = [(name, param) for name, param in model.named_conductances() if name.startswith('blocks.')]
        assert len(names_params) == 8
        references = {name: {} for name, _ in names_params}
        pooled = {case: {} for case in range(len(multiplier_vectors))}
        displacements = {(case, block): dict(centered_squares=0., free_squares=0., elements=0, unchanged_elements=0)
                         for case in pooled for block in selected_blocks}
        batch_rows = []
        postsynaptic_names = {}
        count = 0
        for batch_index, (x, y) in enumerate(typed_batches()):
            if time.monotonic() - start > cfg['budget_seconds']:
                raise TimeoutError('Gradient comparison budget exhausted')
            saved = (cached_records[batch_index] if cached_records is not None else
                     capture_batches(model, [(x, y)], config['iterations'], device,
                                     force_mask=force_mask, keep_unmasked=True)[0])
            with batchnorm_minibatch_stats_only(model):
                _, tracked = model.bptt(x.to(device), config['iterations'])
                loss = F.cross_entropy(tracked, y.to(device))
                native = torch.autograd.grad(loss, [p for _, p in names_params])
            assert all(g.dtype == dtype for g in native), 'Mixed BPTT precision'
            verify_dtype(model, dtype)
            native = [g.detach().cpu().double() for g in native]
            model.detach_state_()
            del loss, tracked
            offset = 0
            for i, block in enumerate(model.analog_blocks):
                if i + 1 not in selected_blocks:
                    offset += len(block.weights)
                    continue
                local = local_bptt_gradients(block, reference_record(saved[i]), config['iterations'][i], device)
                mask_fields = {}
                if force_mask != 'none':
                    masked_local = local_bptt_gradients(block, saved[i], config['iterations'][i], device)
                    mask_fields = dict(force_mask=saved[i]['force_mask'],
                                       masked_force_reference=[agreement(m, r) for m, r in zip(masked_local, local)])
                    del masked_local
                if measure_layers:
                    for j, weight in enumerate(block.weights):
                        interaction, = [fn for fn in block.energy._interactions
                            if isinstance(fn, ConvResistive) and fn._weight is weight]
                        postsynaptic_names[names_params[offset + j][0]] = interaction._layer_post.name
                for j, reference in enumerate(local):
                    name = names_params[offset + j][0]
                    for key, gradient in [('native', native[offset + j]), ('local', reference)]:
                        if key not in references[name]:
                            references[name][key] = gradient * len(y)
                        else:
                            references[name][key].add_(gradient, alpha=len(y))
                for case, vector in enumerate(multiplier_vectors):
                    if time.monotonic() - start > cfg['budget_seconds']:
                        raise TimeoutError('Gradient sweep budget exhausted')
                    multiplier = vector[i]
                    beta = betas[i] * multiplier
                    nudged_k = nudged_vectors[case][i]
                    ep, displacement = eqprop_gradients(block, saved[i], beta, nudged_k, device,
                        measure_layer_displacement=measure_layers)
                    accumulate_displacement(displacements[case, i + 1], displacement)
                    rms = pooled_rms(displacement['centered_squares'], displacement['free_squares'], displacement['elements'])
                    for j, (estimate, reference) in enumerate(zip(ep, local)):
                        name = names_params[offset + j][0]
                        row = dict(batch=batch_index, examples=len(y), cohort='calibration' if batch_index < 4 else 'confirmation',
                            block=i + 1, layer=j + 1, name=name, beta_injected_B=beta, beta_multiplier=multiplier,
                            case_index=case, target_relative_rms=targets[case],
                            nudged_iterations=nudged_k, reference_iterations=config['iterations'][i],
                            output_displacement=rms, native_bptt=agreement(estimate, native[offset + j]),
                            local_bptt=agreement(estimate, reference),
                            local_vs_native_bptt=agreement(reference, native[offset + j]))
                        if mask_fields:
                            row.update(force_mask=mask_fields['force_mask'],
                                       masked_force_reference=mask_fields['masked_force_reference'][j])
                        if measure_layers:
                            row.update(state_measurement_fields(displacement, postsynaptic_names[name]))
                        batch_rows.append(row)
                        reporting.append_metric(out / 'metrics.jsonl', row)
                        if name not in pooled[case]:
                            pooled[case][name] = estimate * len(y)
                        else:
                            pooled[case][name].add_(estimate, alpha=len(y))
                reporting.update_status_progress(out, dict(stage='gradient_comparison', examples=count,
                    batch=batch_index, block=i + 1, multipliers_completed=len(multiplier_vectors)))
                offset += len(block.weights)
            count += len(y)
            reporting.update_status_progress(out, dict(stage='gradient_comparison', examples=count, batch=batch_index))
            print('BATCH', batch_index, 'examples', count, flush=True)
            del saved, native, ep, local
        rows = []
        for case, estimates in pooled.items():
            for name, estimate in estimates.items():
                estimate = estimate / count
                gradients = {k: g / count for k, g in references[name].items()}
                _, block_index, _, layer_index = name.split('.')
                block_number = int(block_index) + 1
                multiplier = multiplier_vectors[case][int(block_index)]
                d = displacements[case, block_number]
                rms = pooled_rms(d['centered_squares'], d['free_squares'], d['elements'])
                rms['unchanged_fraction'] = d['unchanged_elements'] / d['elements']
                row = dict(name=name, block=block_number, layer=int(layer_index) + 1,
                    beta_injected_B=betas[int(block_index)] * multiplier, beta_multiplier=multiplier,
                    case_index=case, target_relative_rms=targets[case],
                    nudged_iterations=nudged_vectors[case][int(block_index)],
                    reference_iterations=config['iterations'][int(block_index)],
                    examples=count, output_displacement=rms,
                    native_bptt=agreement(estimate, gradients['native']),
                    local_bptt=agreement(estimate, gradients['local']),
                    local_vs_native_bptt=agreement(gradients['local'], gradients['native']))
                if measure_layers:
                    row.update(state_measurement_fields(d, postsynaptic_names[name]))
                rows.append(row)
        verify_dtype(model, dtype)
        after = model.snapshot()
        for group in before:
            assert before[group].keys() == after[group].keys()
            assert all(torch.equal(before[group][k], after[group][k]) for k in before[group]), 'Model/BN values changed'
        assert count == 256 and len(rows) == expected_weights * len(multiplier_vectors)
        assert len(batch_rows) == expected_weights * len(multiplier_vectors) * (count // cfg['batch_size'])
        assert reporting.sha256_file(checkpoint_path) == cfg['input_sha256'][checkpoint_file], 'Checkpoint bytes changed'
        reporting.atomic_write_json(out / 'artifacts/comparison.json', dict(layers=rows, batches=batch_rows))
        reporting.complete_run(out, terminal_metrics=dict(layers=rows, examples=count,
            beta_multiplier_vectors=multiplier_vectors, relative_rms_targets=targets,
            nudged_iteration_vectors=nudged_vectors, reference_iterations=config['iterations'],
            diagnostic_blocks=selected_blocks,
            measure_layer_displacement=measure_layers,
            **({'force_mask': force_mask} if force_mask != 'none' else {}),
            **({'beta_multipliers': multipliers} if 'beta_multiplier_vectors' not in cfg and
                'displacement_calibration' not in cfg else {}),
            scheme=config['scheme'], precision=precision, precision_checks_passed=True,
            checkpoint_epoch=checkpoint_epoch, checkpoint_file=checkpoint_file,
            checkpoint_sha256=cfg['input_sha256'][checkpoint_file], checkpoint_bytes_unchanged=True,
            parameters_and_bn_unchanged=True, elapsed_seconds=time.monotonic() - start,
            peak_memory_reserved_bytes=torch.cuda.max_memory_reserved(device)),
            completion=dict(criteria_met=True, official_test_read=False,
                compared_all_conv_weights=set(selected_blocks) == {1, 2, 3},
                compared_all_selected_conv_weights=True))
        errors = reporting.validate_run(out)
        if errors:
            raise RuntimeError(errors)
        print(json.dumps(rows, indent=2), flush=True)
    except BaseException as exc:
        reporting.fail_run(out, error=exc)
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('config', 'input-root', 'output-dir', 'dataset-root', 'source-identity'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--device', default='cuda')
    run(p.parse_args())
