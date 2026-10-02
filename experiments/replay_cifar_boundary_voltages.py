"""Read-only, matched-cohort CIFAR voltage measurements across BN boundaries."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from experiments import reporting
from experiments.train_cifar_l8_analog import (
    Cohort, deterministic_setup, loader, make_split, read_training_data,
)
from labs.cifar_l8_analog import CifarL8Analog, batchnorm_minibatch_stats_only


def statistics(values, channels=False):
    x = values.detach().double()
    if not torch.isfinite(x).all():
        raise FloatingPointError('Nonfinite measured voltage')
    result = dict(elements=x.numel(), sum=float(x.sum()), squares=float(x.square().sum()),
                  minimum=float(x.min()), maximum=float(x.max()), zeros=int((x == 0).sum()))
    if channels:
        dims = (0, *range(2, x.ndim))
        result['channels'] = dict(elements=x.numel() // x.shape[1],
            sum=x.sum(dims).cpu().tolist(), squares=x.square().sum(dims).cpu().tolist())
    return summarize(result)


def summarize(row):
    row = dict(row)
    n = row['elements']
    mean, square = row['sum'] / n, row['squares'] / n
    row.update(mean=mean, rms=math.sqrt(square), std=math.sqrt(max(0., square - mean**2)),
               zero_fraction=row['zeros'] / n)
    if 'channels' in row:
        channel = dict(row['channels'])
        channel['mean'] = [s / channel['elements'] for s in channel['sum']]
        channel['rms'] = [math.sqrt(s / channel['elements']) for s in channel['squares']]
        channel['std'] = [math.sqrt(max(0., r*r - m*m))
                          for r, m in zip(channel['rms'], channel['mean'])]
        row['channels'] = channel
    return row


def pool(rows):
    out = {key: sum(r[key] for r in rows) for key in ('elements', 'sum', 'squares', 'zeros')}
    out.update(minimum=min(r['minimum'] for r in rows), maximum=max(r['maximum'] for r in rows))
    if 'channels' in rows[0]:
        out['channels'] = dict(elements=sum(r['channels']['elements'] for r in rows))
        for key in ('sum', 'squares'):
            out['channels'][key] = [sum(v) for v in zip(*(r['channels'][key] for r in rows))]
    return summarize(out)


def capture(model, x, iterations):
    """Observe actual forward tensors; reconstruction is used only as a guard."""
    measured, bn_details, handles, bn_outputs = {}, [], [], {}

    def boundary(name, values, state, gain):
        expected = torch.cat((gain * values, -gain * values), dim=1)
        if not torch.equal(state, expected):
            raise AssertionError(f'{name}: physical boundary differs from signed gained input')
        measured[name + '/logical_input'] = statistics(values)
        measured[name + '/injected_physical'] = statistics(state)

    def block_hook(index, block, args, output):
        name = f'block{index + 1}'
        if index and not torch.equal(args[0], bn_outputs[index - 1]):
            raise AssertionError('Next block did not receive the captured BN output')
        boundary(name, args[0], block.boundary_layer.state, block.input_gain)
        for layer_index, layer in enumerate(block.free_layers(), 1):
            measured[f'{name}/conv{layer_index}_state'] = statistics(layer.state)
        measured[name + '/decoded_output'] = statistics(output)

    def bn_hook(index, bn, args, output):
        name = f'block{index + 1}'
        values = args[0].detach().double()
        mean = values.mean((0, 2, 3), keepdim=True)
        variance = values.var((0, 2, 3), unbiased=False, keepdim=True)
        normalized = (values - mean) / torch.sqrt(variance + bn.eps)
        expected = normalized * bn.weight.double()[None, :, None, None]
        expected = expected + bn.bias.double()[None, :, None, None]
        torch.testing.assert_close(output.double(), expected, atol=2e-5, rtol=2e-5)
        measured[name + '/pre_bn'] = statistics(values, channels=True)
        measured[name + '/normalized_before_affine'] = statistics(normalized, channels=True)
        measured[name + '/post_bn'] = statistics(output, channels=True)
        bn_outputs[index] = output
        bn_details.append(dict(block=index + 1, eps=bn.eps,
            batch_variance=variance.flatten().cpu().tolist(),
            variance_fraction=(variance / (variance + bn.eps)).flatten().cpu().tolist(),
            reconstruction_max_abs_error=float((expected - output.double()).abs().max())))

    def head_hook(head, args, output):
        assert torch.equal(args[0], bn_outputs[2].flatten(1))
        boundary('head', args[0], head.boundary.state, F.softplus(head._input_gain_raw))

    for i, block in enumerate(model.analog_blocks):
        handles.append(block.register_forward_hook(lambda b, a, o, i=i: block_hook(i, b, a, o)))
        handles.append(model.bridges[i][1].register_forward_hook(
            lambda b, a, o, i=i: bn_hook(i, b, a, o)))
    handles.append(model.head.register_forward_hook(head_hook))
    try:
        with torch.no_grad(), batchnorm_minibatch_stats_only(model):
            logits = model(x, iterations, reset=True)
            if not torch.isfinite(logits).all():
                raise FloatingPointError('Nonfinite logits')
    finally:
        for handle in handles:
            handle.remove()
        model.detach_state_()
    assert len(measured) == 28
    return measured, bn_details


def run(args):
    cfg = json.loads(args.config.read_text())
    identity = json.loads(args.source_identity.read_text())
    source = Path(__file__).resolve().parents[1]
    for name, digest in identity['source_sha256'].items():
        assert reporting.sha256_file(source / name) == digest, name
    for name, digest in cfg['input_sha256'].items():
        assert reporting.sha256_file(args.input_root / name) == digest, name
    environment = dict(torch=str(torch.__version__), cuda=torch.version.cuda)
    if not args.smoke:
        assert environment == cfg['expected_environment'], environment
        assert args.device == 'cuda' and torch.cuda.is_available()
    config = json.loads((args.input_root / 'inputs/model_config.json').read_text())
    cohort = json.loads((args.input_root / 'inputs/cohort.json').read_text())
    indices = cohort['indices'][:2] if args.smoke else cohort['indices']
    epochs = [0] if args.smoke else cfg['epochs']
    deterministic_setup(config['seed'])
    out = args.output_dir
    reporting.start_run(out, dict(study_id=cfg['study_id'], run_id=out.name,
        arm_id='ours_boundary_voltages', evidence_class='cifar_finite_iteration_voltage_replay',
        seed=config['seed'], smoke=args.smoke, resolved_config=cfg, model_config=config,
        source_identity=identity, environment=environment, runtime=reporting.runtime_context(),
        dataset='CIFAR10 training cohort', input_root=str(args.input_root),
        gpu=torch.cuda.get_device_name() if args.device == 'cuda' else None))
    start = time.monotonic()
    try:
        images, labels = read_training_data(args.dataset_root)
        assert set(indices) <= set(make_split(labels, config['split']['seed'])['train'])
        assert len(indices) == len(set(indices)) == (2 if args.smoke else 256)
        data = Cohort(images, labels, indices, config, augment=False)
        rows, batch_rows, all_bn_details, parameters, comparisons = [], [], [], [], []
        for epoch in epochs:
            checkpoint_path = args.input_root / f'inputs/e{epoch}.pt'
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            assert checkpoint['epoch'] == epoch
            model = CifarL8Analog(config, args.device)
            model.restore(checkpoint['model'])
            before = model.snapshot()
            for group in before:
                assert before[group].keys() == checkpoint['model'][group].keys()
                assert all(torch.equal(v, checkpoint['model'][group][k]) for k, v in before[group].items())
            del checkpoint
            model.train()
            gain = [float(b.input_gain.detach()) for b in model.analog_blocks]
            gain.append(float(F.softplus(model.head._input_gain_raw).detach()))
            parameters.append(dict(epoch=epoch, input_gains=gain,
                batch_norm=[dict(block=i+1, eps=b[1].eps, gamma=b[1].weight.detach().cpu().tolist(),
                                beta=b[1].bias.detach().cpu().tolist()) for i, b in enumerate(model.bridges)]))
            epoch_batches = []
            for batch, (x, _) in enumerate(loader(data, 2 if args.smoke else cfg['batch_size'])):
                if time.monotonic() - start > (300 if args.smoke else cfg['budget_seconds']):
                    raise TimeoutError('Voltage replay budget exhausted')
                measured, bn_details = capture(model, x.to(args.device), config['iterations'])
                epoch_batches.append(measured)
                batch_rows.extend(dict(epoch=epoch, batch=batch, stage=k, **v) for k, v in measured.items())
                all_bn_details.extend(dict(epoch=epoch, batch=batch, **v) for v in bn_details)
                progress = dict(epoch=epoch, batch=batch + 1, examples=(batch + 1) * len(x),
                                elapsed_seconds=time.monotonic() - start)
                reporting.append_metric(out / 'metrics.jsonl', dict(record_type='voltage_batch', **progress))
                reporting.update_status_progress(out, progress)
                print(json.dumps(progress), flush=True)
            pooled = {key: pool([b[key] for b in epoch_batches]) for key in epoch_batches[0]}
            rows.extend(dict(epoch=epoch, stage=k, **v) for k, v in pooled.items())
            if not args.smoke:
                reference = cfg['reference_free_rms'][str(epoch)]
                for stage, ref in reference.items():
                    actual = pooled[stage]['rms']
                    assert math.isclose(actual, ref, rel_tol=2e-5, abs_tol=1e-9), (epoch, stage, actual, ref)
                    comparisons.append(dict(epoch=epoch, stage=stage, reference=ref, actual=actual,
                                            relative_error=abs(actual/ref - 1)))
            after = model.snapshot()
            for group in before:
                assert before[group].keys() == after[group].keys()
                assert all(torch.equal(before[group][k], after[group][k]) for k in before[group]), 'Model/BN changed'
            del model, before, after
        for name, digest in cfg['input_sha256'].items():
            assert reporting.sha256_file(args.input_root / name) == digest, name
        reporting.atomic_write_json(out / 'artifacts/voltages.json', dict(
            rows=rows, batches=batch_rows, bn_batches=all_bn_details, parameters=parameters,
            reference_comparisons=comparisons, cohort_indices=indices))
        reporting.complete_run(out, terminal_metrics=dict(epochs=epochs, examples_per_epoch=len(indices),
            rows=len(rows), batches=len(batch_rows), checkpoint_bytes_unchanged=True,
            parameters_and_bn_unchanged=True, boundary_identities_pass=True,
            reference_rms_matched=not args.smoke, elapsed_seconds=time.monotonic() - start,
            peak_memory_reserved_bytes=torch.cuda.max_memory_reserved() if args.device == 'cuda' else 0),
            completion=dict(criteria_met=True, official_test_read=False))
    except BaseException as exc:
        reporting.fail_run(out, error=exc)
        raise
    errors = reporting.validate_run(out)
    assert not errors, errors


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'input-root', 'output-dir', 'dataset-root', 'source-identity'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--smoke', action='store_true')
    run(parser.parse_args())
