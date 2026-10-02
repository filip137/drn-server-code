"""Read-only centered, frozen-current beta calibration for the three CIFAR blocks."""
from __future__ import annotations
import argparse
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from experiments import reporting
from experiments.train_cifar_l8_analog import Cohort, deterministic_setup, loader, make_split, read_training_data
from labs.cifar_l8_analog import CifarL8Analog, batchnorm_minibatch_stats_only
from labs.custom_minimizer import CustomQuadraticMinimizer
from model.function.interaction import Function
from training.sgd import AugmentedFunction


FORCE_MASKS = ('none', 'inactive_output_channels')


def validate_force_mask(mode):
    if mode not in FORCE_MASKS:
        raise ValueError(f'Expected force_mask in {FORCE_MASKS}; got {mode!r}.')
    return mode


def inactive_output_channels(state):
    """Logical channels whose excitatory and inhibitory states are zero for the whole minibatch.

    Their differential output is identically zero, so the following BatchNorm
    has zero batch variance and scales the backpropagated force by
    gamma/sqrt(eps). The clamped diodes give these units zero derivative: BPTT
    ignores any force on them, while a large finite nudge can switch them on.
    """
    if state.ndim != 4 or state.size(1) % 2:
        raise ValueError('Expected a four-dimensional differential state with an even channel count; '
                         f'got shape={tuple(state.shape)}.')
    active = (state != 0).any(dim=(0, 2, 3))
    excitatory, inhibitory = active.chunk(2)
    return ~(excitatory | inhibitory)


def mask_output_force(force, state, mode):
    """Return the force to inject and a compact record of what the mask removed."""
    validate_force_mask(mode)
    if mode == 'none':
        return force, dict(force_mask=mode)
    inactive = inactive_output_channels(state)
    physical = torch.cat((inactive, inactive))[None, :, None, None]
    total = float(force.double().square().sum())
    removed = float(force.masked_fill(~physical, 0.).double().square().sum())
    return force.masked_fill(physical, 0.), dict(force_mask=mode, inactive_channels=int(inactive.sum()),
        masked_force_square_share=removed / total if total else 0.)


class FrozenOutputCost(Function):
    """A fixed external loss derivative, with no batch-mean rescaling."""
    def __init__(self, layer):
        super().__init__([layer], [])
        self.layer = layer
        self.gradient = None

    def _grad(self, layer, mean=False):
        assert layer is self.layer and not mean
        if self.gradient is None:
            raise RuntimeError('Output derivative has not been captured')
        return self.gradient

    def eval(self):
        return (self.layer.state * self.gradient).flatten(1).sum(1)


def nudged_minimizer(block, augmented, iterations):
    s = block.solver
    return CustomQuadraticMinimizer(augmented, free_layers=block.free_layers(), num_iterations=iterations,
        mode=s.mode, non_linearity=s.non_linearity, quadratic_diode_param=dict(s.quadratic_diode_param),
        exponential_diode_param=dict(s.exponential_diode_param), voltage_amp=s.voltage_amp,
        current_amp=s.current_amp, hard_sigmoid_param=dict(s.hard_sigmoid_param), iv_data=s.iv_data,
        iv_data_path=s.iv_data_path, double_diode_updater=s.double_diode_updater,
        adaptive_equilibrium=s.adaptive_equilibrium, overrelaxation_factor=s.overrelaxation_factor,
        single_diode_updater=s.single_diode_updater, minimizer_settings=s.minimizer_settings,
        damping=s.damping, experimental_newton_max_steps=s.experimental_newton_max_steps)


def search_beta(measure, target, *, metric, tolerance=.05, max_evaluations=24, initial_beta=1.):
    """Bracket then bisect in log beta; do not assume linear response at large beta."""
    low = high = None
    beta = float(initial_beta)
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError('Initial beta must be finite and positive')
    history = []
    for _ in range(max_evaluations):
        row = dict(measure(beta))
        value = row[metric]
        if not math.isfinite(value) or value < 0:
            raise FloatingPointError(f'Invalid displacement {value}')
        error = abs(value / target - 1)
        history.append(dict(beta_injected_B=beta, relative_target_error=error, **row))
        if error <= tolerance:
            return beta, history
        if value < target:
            low = beta
        else:
            high = beta
        if low is not None and high is not None:
            beta = math.sqrt(low * high)
        elif value == 0:
            beta *= 100.
        else:
            beta *= min(100., max(.01, target / value))
        if not 1e-20 <= beta <= 1e20:
            raise RuntimeError('Beta search exceeded declared bounds')
    raise RuntimeError(f'No target match within {max_evaluations} measurements; last={history[-1]}')


def pooled_rms(centered_squares, free_squares, count):
    if count <= 0 or free_squares <= 0:
        raise ValueError('Output free RMS must be positive')
    return dict(absolute_rms=math.sqrt(centered_squares / count), free_rms=math.sqrt(free_squares / count),
                relative_rms=math.sqrt(centered_squares / free_squares), elements=count)


def capture_batches(model, batches, iterations, device, force_mask='none', keep_unmasked=False):
    """Differentiate the free forward CE; preserve BN running buffers and weights.

    ``force_mask`` changes only the force later injected into each block; with
    ``keep_unmasked`` the original force is also retained for BPTT references.
    """
    validate_force_mask(force_mask)
    captured = []
    model.train()
    for x, y in batches:
        physical_outputs = []
        hooks = [b.register_forward_hook(lambda b, args, out: physical_outputs.append(b.free_layers()[-1].state))
                 for b in model.analog_blocks]
        try:
            with batchnorm_minibatch_stats_only(model):
                logits = model(x.to(device), iterations, reset=True)
                gradients = torch.autograd.grad(F.cross_entropy(logits, y.to(device), reduction='sum'), physical_outputs)
            record = []
            for block, grad in zip(model.analog_blocks, gradients):
                if not torch.isfinite(grad).all() or not torch.any(grad != 0):
                    raise FloatingPointError('Output force is nonfinite or dead')
                force, mask = mask_output_force(grad.detach(), block.free_layers()[-1].state.detach(), force_mask)
                entry = dict(input=block.boundary_layer.state.detach().cpu().clone(),
                    states=[layer.state.detach().cpu().clone() for layer in block.free_layers()],
                    gradient=force.cpu().clone())
                if force_mask != 'none':
                    entry['force_mask'] = mask
                    if keep_unmasked:
                        entry['unmasked_gradient'] = grad.detach().cpu().clone()
                record.append(entry)
            captured.append(record)
        finally:
            for hook in hooks: hook.remove()
        model.detach_state_()
        del logits, gradients, physical_outputs
    return captured


def measure_block(block, minimizer, augmented, cost, records, block_index, beta, device):
    centered_squares = free_squares = 0.
    count = 0
    with torch.no_grad():
        for batch in records:
            saved = batch[block_index]
            block.boundary_layer.state = saved['input'].to(device)
            cost.gradient = saved['gradient'].to(device)
            augmented.prepare_nudging()
            force = augmented._nudging._force.clone()
            outputs = []
            for sign in (-1., 1.):
                for layer, state in zip(block.free_layers(), saved['states']):
                    layer.state = state.to(device).clone()
                augmented.nudging = sign * beta
                minimizer.compute_equilibrium()
                output = block.free_layers()[-1].state
                if not torch.isfinite(output).all():
                    raise FloatingPointError('Nonfinite nudged endpoint')
                outputs.append(output.detach().cpu().double())
                assert torch.equal(augmented._nudging._force, force), 'Current changed between phases'
            centered = (outputs[1] - outputs[0]) / 2
            free = saved['states'][-1].double()
            centered_squares += float(centered.square().sum())
            free_squares += float(free.square().sum())
            count += free.numel()
    return pooled_rms(centered_squares, free_squares, count)


def run(args):
    cfg = json.loads(args.config.read_text())
    root = args.input_root
    config = json.loads((root / cfg['model_config']).read_text())
    checkpoint_path = root / cfg['checkpoint']
    assert reporting.sha256_file(checkpoint_path) == cfg['checkpoint_sha256']
    identity = json.loads(args.source_identity.read_text())
    source = Path(__file__).resolve().parents[1]
    for name, digest in identity['source_sha256'].items():
        assert reporting.sha256_file(source / name) == digest, name
    deterministic_setup(config['seed'])
    device = torch.device(args.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('This assigned calibration requires a CUDA GPU')
    out = args.output_dir
    reporting.start_run(out,dict(study_id=cfg['study_id'],run_id=out.name,arm_id='ours_block_beta',seed=config['seed'],
        smoke=False,evidence_class='cifar_frozen_current_beta_calibration',resolved_config=cfg,source_identity=identity,
        checkpoint_sha256=cfg['checkpoint_sha256'],device=str(device),gpu=torch.cuda.get_device_name(device),
        environment={'torch':str(torch.__version__),'cuda':torch.version.cuda}))
    start = time.monotonic()
    try:
        checkpoint = torch.load(checkpoint_path,map_location='cpu',weights_only=False)
        assert checkpoint['epoch'] == cfg['checkpoint_epoch']
        model = CifarL8Analog(config,device)
        model.restore(checkpoint['model'])
        before = model.snapshot()
        del checkpoint
        images, labels = read_training_data(args.dataset_root)
        split = make_split(labels,config['split']['seed'])
        gen = np.random.default_rng(cfg['cohort_seed'])
        indices = gen.choice(split['train'],size=cfg['calibration_examples']+cfg['confirmation_examples'],replace=False).tolist()
        reporting.atomic_write_json(out/'artifacts/cohort.json',{'indices':indices,'source_split':'training',
            'calibration_examples':cfg['calibration_examples'],'confirmation_examples':cfg['confirmation_examples'],'official_test_read':False})
        data = Cohort(images,labels,indices,config,augment=False)
        records = capture_batches(model,loader(data,cfg['batch_size']),config['iterations'],device)
        n = cfg['calibration_examples']//cfg['batch_size']
        calibration, confirmation = records[:n], records[n:]
        matches = []
        for i, (block, target) in enumerate(zip(model.analog_blocks,cfg['targets'])):
            cost = FrozenOutputCost(block.free_layers()[-1])
            # B is the coefficient actually added to the local energy row; no implicit amplification factor.
            augmented = AugmentedFunction(block.energy,cost,nudging_mode='current',current_scale=1.)
            minimizer = nudged_minimizer(block,augmented,config['iterations'][i])
            def measure(beta):
                if time.monotonic()-start > cfg['budget_seconds']:
                    raise TimeoutError('Calibration budget exhausted')
                value = measure_block(block,minimizer,augmented,cost,calibration,i,beta,device)
                reporting.append_metric(out/'metrics.jsonl',dict(block=i+1,beta_injected_B=beta,target=target,**value))
                reporting.update_status_progress(out,dict(stage='beta_calibration',block=i+1,beta_injected_B=beta))
                print('MEASURE',i+1,beta,value,flush=True)
                return value
            beta, history = search_beta(measure,target,metric=cfg['metric'],tolerance=cfg['relative_tolerance'],max_evaluations=cfg['max_evaluations'])
            heldout = measure_block(block,minimizer,augmented,cost,confirmation,i,beta,device)
            matches.append(dict(block=i+1,conv_layers=len(block.free_layers()),target=target,beta_injected_B=beta,
                calibration=history[-1],confirmation=heldout,confirmation_relative_target_error=abs(heldout[cfg['metric']]/target-1),history=history))
            reporting.atomic_write_json(out/'artifacts/matches.json',matches)
            print('MATCH',matches[-1],flush=True)
        after = model.snapshot()
        for group in before:
            assert before[group].keys()==after[group].keys()
            assert all(torch.equal(before[group][k],after[group][k]) for k in before[group]), 'Model/BN values changed'
        reporting.complete_run(out,terminal_metrics={'matches':matches,'elapsed_seconds':time.monotonic()-start,
            'peak_memory_reserved_bytes':torch.cuda.max_memory_reserved(device),'parameters_and_bn_unchanged':True},
            completion={'criteria_met':True,'official_test_read':False,'all_calibration_targets_met':True})
        errors=reporting.validate_run(out)
        if errors: raise RuntimeError(errors)
    except BaseException as exc:
        reporting.fail_run(out,error=exc)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--input-root',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--dataset-root',type=Path,required=True)
    p.add_argument('--source-identity',type=Path,required=True);p.add_argument('--device',default='cuda')
    run(p.parse_args())
