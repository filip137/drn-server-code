"""Read-only CIFAR replay: float32 free/autograd with float64 nudged solves."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import torch

from experiments import reporting
from experiments.calibrate_cifar_block_beta import capture_batches
from experiments.compare_cifar_block_beta_gradients import (
    agreement, accumulate_displacement, eqprop_gradients, local_bptt_gradients,
    state_measurement_fields,
)
from experiments.replay_cifar_precision_response import convert_model, check_snapshot, verify_dtype
from experiments.train_cifar_l8_analog import Cohort, deterministic_setup, loader, make_split, read_training_data
from labs.cifar_l8_analog import CifarL8Analog


MODES = ('full64', 'full32', 'split32_free64_nudged')


def promote_saved(saved32):
    """Exact promotion before solving; the captured float32 tensors stay immutable."""
    tensors = [saved32['input'], saved32['gradient'], *saved32['states']]
    if not all(t.dtype == torch.float32 for t in tensors):
        raise ValueError('Split replay requires a float32 free/forcing capture')
    return dict(input=saved32['input'].double(), gradient=saved32['gradient'].double(),
                states=[t.double() for t in saved32['states']])


def eqprop_from_float32(block64, saved32, beta, iterations, device, **kwargs):
    """Solve both nudged phases in a separate float64 donor, then evaluate in64.

    Donor weights must be exact promotions of the float32 source weights. The
    caller constructs that donor once; this helper never casts live parameters.
    Its transient boundary/free states are restored, including on an exception.
    """
    variables = [block64.boundary_layer, *block64.free_layers(), *block64.weights, *block64.biases]
    if not all(v.state.dtype == torch.float64 for v in variables):
        raise ValueError('Nudged donor must already be entirely float64')
    live = [(v, v.state) for v in [block64.boundary_layer, *block64.free_layers()]]
    try:
        return eqprop_gradients(block64, promote_saved(saved32), beta, iterations, device, **kwargs)
    finally:
        for variable, state in live:
            variable.state = state


def difference_sums(actual, reference):
    a, b = actual.detach().cpu().double(), reference.detach().cpu().double()
    if a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError('Mismatched or nonfinite precision comparison')
    return dict(elements=a.numel(), actual_squares=float(a.square().sum()),
                reference_squares=float(b.square().sum()), difference_squares=float((a-b).square().sum()),
                dot=float((a*b).sum()))


def finish_difference(raw):
    result = dict(raw)
    n, a, b, d = (raw[k] for k in ['elements', 'actual_squares', 'reference_squares', 'difference_squares'])
    result.update(actual_rms=math.sqrt(a/n), reference_rms=math.sqrt(b/n), difference_rms=math.sqrt(d/n),
                  relative_l2=math.sqrt(d/b) if b else None,
                  cosine=raw['dot']/math.sqrt(a*b) if a and b else None)
    return result


def add_tensor(pool, key, value, examples):
    if key not in pool:
        pool[key] = value.detach().cpu().double() * examples
    else:
        pool[key].add_(value.detach().cpu().double(), alpha=examples)


def run(args):
    config = json.loads(args.config.read_text())
    assert config['scheme'] == 'ours' and config['iterations'] == [6, 6, 4]
    assert config['batch_size'] == 32
    indices = config['conv_eqprop']['cohort_indices']
    assert len(indices) == len(set(indices)) == 256
    assert len(args.betas) == 3 and all(math.isfinite(b) and b > 0 for b in args.betas)
    deterministic_setup(config['seed'])
    device, out = torch.device(args.device), args.run_dir
    source = Path(__file__).resolve().parents[1]
    files = [p for folder in ['experiments', 'labs', 'model', 'training'] for p in (source/folder).rglob('*.py')]
    input_paths = [args.config, args.checkpoint, *[args.dataset_root/f'data_batch_{i}' for i in range(1, 6)]]
    hashes = {str(p.resolve()): reporting.sha256_file(p) for p in input_paths}
    source_hashes = {str(p.relative_to(source)): reporting.sha256_file(p) for p in files}
    reporting.start_run(out, dict(study_id='cifar-split-precision-20261001-v1', run_id=out.name,
        arm_id='split_precision_replay', seed=config['seed'], smoke=False, mode='diagnostic',
        evidence_class='cifar_finite_iteration_split_precision_replay', resolved_config=config,
        command=sys.argv, input_sha256=hashes, source_sha256=source_hashes,
        source_identity_kind='per-file SHA256 of executed snapshot',
        runtime=reporting.runtime_context(target=args.target),
        environment=dict(torch=str(torch.__version__), cuda=torch.version.cuda,
                         gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None),
        betas=args.betas, rms_targets=[.01, .005, 1e-5], budget_seconds=args.budget_seconds,
        modes=list(MODES), dtype_contract={'full64':'all float64', 'full32':'all float32',
            'split32_free64_nudged':'float32 free/autograd; exact-promoted parameters, states, forcing; float64 phases/formula'},
        dataset=dict(name='CIFAR10', source_split='training', cohort_indices=indices,
                     augmentation=False, official_test_read=False),
        updates=0, read_noise=0., batchnorm='minibatch statistics with frozen running buffers',
        reference='block-local finite-K BPTT from the captured free state; not equilibrium or full-network BPTT'))
    start = time.monotonic()
    try:
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        models, snapshots = {}, {}
        for mode, dtype in [('full64', torch.float64), ('full32', torch.float32),
                            ('split32_free64_nudged', torch.float64)]:
            model = CifarL8Analog(config, device)
            convert_model(model, dtype)  # Before restore: do not round the full64 checkpoint.
            initial = snapshots['full32'] if mode == MODES[2] else checkpoint['model']
            model.restore(initial)
            snapshots[mode] = model.snapshot()
            check_snapshot(initial, snapshots[mode])
            models[mode] = model
        epoch = checkpoint['epoch']
        del checkpoint
        images, labels = read_training_data(args.dataset_root)
        assert set(indices) <= set(make_split(labels, config['split']['seed'])['train'])
        data = Cohort(images, labels, indices, config, augment=False)
        names = [n for n, _ in models['full64'].named_conductances() if n.startswith('blocks.')]
        assert len(names) == 8
        pooled, state_pools, difference_pools = {}, {}, {}
        batches, cross_batches, reference_batches = [], [], []
        count = 0
        for batch, (x, y) in enumerate(loader(data, 32)):
            if time.monotonic()-start > args.budget_seconds:
                raise TimeoutError('Split precision replay budget exhausted')
            captures = {mode: capture_batches(models[mode], [(x.to(dtype=dtype), y)],
                         config['iterations'], device)[0]
                        for mode, dtype in [('full64', torch.float64), ('full32', torch.float32)]}
            offset = 0
            for i, iterations in enumerate(config['iterations']):
                saved64, saved32 = captures['full64'][i], captures['full32'][i]
                for field, a, b in [('input', saved32['input'], saved64['input']),
                                     ('forcing', saved32['gradient'], saved64['gradient']),
                                     *[(f'free_layer_{j+1}', a, b) for j, (a, b) in enumerate(zip(saved32['states'], saved64['states']))]]:
                    raw = difference_sums(a, b)
                    row = dict(batch=batch, block=i+1, field=field, **finish_difference(raw))
                    cross_batches.append(row)
                    key = (i+1, field)
                    if key not in difference_pools: difference_pools[key] = dict(raw)
                    else:
                        for k, v in raw.items(): difference_pools[key][k] += v
                local, estimates, displacements = {}, {}, {}
                for mode in MODES:
                    if time.monotonic()-start > args.budget_seconds:
                        raise TimeoutError('Split precision replay budget exhausted')
                    block = models[mode].analog_blocks[i]
                    saved = saved64 if mode == 'full64' else saved32
                    local[mode] = local_bptt_gradients(block,
                        promote_saved(saved) if mode == MODES[2] else saved, iterations, device)
                    fn = eqprop_from_float32 if mode == MODES[2] else eqprop_gradients
                    estimates[mode], displacements[mode] = fn(block, saved, args.betas[i], iterations,
                        device, measure_layer_displacement=True)
                    accumulate_displacement(state_pools.setdefault((mode, i+1), {}), displacements[mode])
                for j, name in enumerate(names[offset:offset+len(local['full64'])]):
                    for mode in MODES:
                        estimate, reference = estimates[mode][j], local[mode][j]
                        add_tensor(pooled, (mode, name, 'eqprop'), estimate, len(y))
                        add_tensor(pooled, (mode, name, 'local_bptt'), reference, len(y))
                        row = dict(batch=batch, examples=len(y), mode=mode, block=i+1, layer=j+1, name=name,
                            beta_injected_B=args.betas[i], common_full64_local_bptt=agreement(estimate, local['full64'][j]),
                            common_full64_eqprop=agreement(estimate, estimates['full64'][j]),
                            own_local_bptt=agreement(estimate, reference),
                            **state_measurement_fields(displacements[mode], models[mode].analog_blocks[i].free_layers()[j].name))
                        batches.append(row)
                        reporting.append_metric(out/'metrics.jsonl', dict(stage='gradient_comparison', **row))
                        reference_batches.append(dict(batch=batch, mode=mode, name=name,
                            **agreement(reference, local['full64'][j])))
                offset += len(local['full64'])
                progress = dict(stage='split_precision_replay', batch=batch+1, batches=8, block=i+1,
                                elapsed_seconds=time.monotonic()-start)
                reporting.update_status_progress(out, progress)
                print(json.dumps(progress), flush=True)
            count += len(y)
        rows, reference_rows = [], []
        for mode in MODES:
            for name in names:
                _, block_index, _, layer_index = name.split('.')
                i, j = int(block_index), int(layer_index)
                estimate = pooled[mode, name, 'eqprop']/count
                reference = pooled[mode, name, 'local_bptt']/count
                rows.append(dict(mode=mode, name=name, block=i+1, layer=j+1, examples=count,
                    beta_injected_B=args.betas[i],
                    common_full64_local_bptt=agreement(estimate, pooled['full64', name, 'local_bptt']/count),
                    common_full64_eqprop=agreement(estimate, pooled['full64', name, 'eqprop']/count),
                    own_local_bptt=agreement(estimate, reference),
                    **state_measurement_fields(state_pools[mode, i+1], models[mode].analog_blocks[i].free_layers()[j].name)))
                reference_rows.append(dict(mode=mode, name=name,
                    **agreement(reference, pooled['full64', name, 'local_bptt']/count)))
        assert count == 256 and len(batches) == 8*8*3 and len(rows) == 8*3
        for mode, model in models.items():
            check_snapshot(snapshots[mode], model.snapshot())
            verify_dtype(model, torch.float32 if mode == 'full32' else torch.float64)
        for path, digest in hashes.items(): assert reporting.sha256_file(Path(path)) == digest, path
        for path, digest in source_hashes.items(): assert reporting.sha256_file(source/path) == digest, path
        torch.save(dict(example_weighted_gradient_sums=pooled, examples=count), out/'artifacts/pooled_gradients.pt')
        reporting.atomic_write_json(out/'artifacts/comparison.json', dict(epoch=epoch, layers=rows,
            batches=batches, reference_comparison=reference_rows, reference_batches=reference_batches,
            cross_precision=[dict(block=k[0], field=k[1], **finish_difference(v)) for k,v in difference_pools.items()],
            cross_precision_batches=cross_batches, cohort_indices=indices))
        reporting.complete_run(out, terminal_metrics=dict(epoch=epoch, examples=count, rows=len(rows),
            batch_rows=len(batches), parameters_and_bn_unchanged=True, checkpoint_bytes_unchanged=True,
            actual_parameter_dtypes={mode: sorted({str(p.dtype) for p in model.trainable_tensors().values()})
                                     for mode, model in models.items()},
            precision_checks_passed=True, elapsed_seconds=time.monotonic()-start,
            peak_memory_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else 0),
            completion=dict(criteria_met=True, official_test_read=False, all_three_modes=True, updates=0))
    except BaseException as exc:
        reporting.fail_run(out, error=exc)
        raise
    errors = reporting.validate_run(out)
    if errors: raise RuntimeError(errors)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['config', 'checkpoint', 'dataset-root', 'run-dir']:
        p.add_argument('--'+name, type=Path, required=name == 'run-dir')
    p.add_argument('--betas', type=float, nargs=3)
    p.add_argument('--target', default='trex')
    p.add_argument('--device', default='cuda')
    p.add_argument('--budget-seconds', type=float, default=600.)
    p.add_argument('--validate', action='store_true')
    args = p.parse_args()
    if args.validate:
        errors = reporting.validate_run(args.run_dir)
        if errors: raise RuntimeError(errors)
        result = json.loads((args.run_dir/'result.json').read_text())
        assert result['completion']['criteria_met']
        print('VALIDATED', args.run_dir)
        return
    if any(getattr(args, name) is None for name in ['config', 'checkpoint', 'dataset_root', 'betas']):
        p.error('--config --checkpoint --dataset-root --betas are required for replay')
    run(args)


if __name__ == '__main__':
    main()
