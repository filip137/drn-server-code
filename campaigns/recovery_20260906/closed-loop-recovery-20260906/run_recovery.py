"""Paired exploratory recovery from the reviewed, immutable HWA/P&V states."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def cpu_tree(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_tree(v) for v in value)
    return value


class Drn:
    def __init__(self, plan, condition, lr, evaluation_limit):
        import torch
        from experiments.mnist_relu_drn import figure6_om_256_ladder as engine
        self.engine = engine
        source = plan['drn']
        config = json.loads(Path(source['config']).read_text())
        deployment = engine._load_torch(Path(source['deployment']), schema='ebl.figure6_om_256_deployment_replica')
        prepared = engine._load_torch(Path(source['prepared']), schema='ebl.figure6_om_256_ladder_prepared')
        spec, teacher, runtime = engine._runtime(config, teacher_path=Path(prepared['teacher']['path']), gain=float(prepared['fixed_logit_gain']), smoke=False)
        self.stack, self.teacher = runtime['stack'], teacher
        self.device = self.stack.device
        field = engine._field_from_payload(deployment['endpoint_field'], device=self.device)
        population = engine.load_om_array_population(Path(source['population']))
        state = deployment['targets']['hwa']['clean_adam_start' if condition == 'healthy' else 'corrupt_adam_start']
        self.plant = engine._plant_from_state(field=field, population=population, state=state)
        self.data = engine._loaders(spec, data_seed=int(deployment['seeds']['data_order']))
        self.open_optimizer = engine.PersistentFigure6OmPulseAdam(self.plant, learning_rate_progress=lr, pulse_cap=plan['total_pulse_cap'], pulse_selection_seed=int(deployment['seeds']['adam_selection']))
        self.evaluation_limit = evaluation_limit
        self.tolerance = plan['drn']['verify_tolerance_progress']
        self.seeds = deployment['seeds']
        self.source_paths = {'deployment': source['deployment'], 'prepared': source['prepared'], 'population': source['population'], 'teacher': prepared['teacher']['path']}
        self.metadata = {'fault_model': 'synthetic_post_PV_RESET_stuck', 'gain': prepared['fixed_logit_gain'], 'coordinate': 'progress', 'nominal_pulse': 0.04745, 'source_revision': source['source_revision']}

    def apparent(self):
        # Match the original P&V verify port. Only the network forward projects
        # progress to [0,1]; verification retains the actual noisy observation.
        return (self.plant.apparent_raw_a + 1) / 2

    def persistent(self):
        return (self.plant.raw_a + 1) / 2

    def gradient(self, inputs, labels):
        gradients, metrics = self.engine._one_forward_gradient(stack=self.stack, teacher=self.teacher, inputs=inputs, labels=labels, full_g=self.plant.apparent_full_conductance)
        self.physical_gradient = gradients
        return self.engine.flatten_physical(tuple(g.detach() * window for g, window in zip(gradients, self.plant.field.window, strict=True))), float(metrics['kl_sum']) / int(metrics['examples'])

    def open_step(self, gradient):
        value = self.open_optimizer.step(self.physical_gradient)
        return {'pulses': value.pulsed_cells, 'verify_rounds': 0, 'blocked_at_cap': value.capped_cells}

    def pulse(self, direction):
        self.plant.pulse(direction, pulse_cap=self.open_optimizer.pulse_cap)

    def evaluate(self, loader):
        return self.engine._evaluate_plant(stack=self.stack, teacher=self.teacher, loader=loader, plant=self.plant, sample_limit=self.evaluation_limit)

    def state(self):
        return self.plant.state_dict()

    def restore(self, state):
        self.plant.load_state_dict(state)

    def invariant(self):
        import torch
        p = self.plant
        return {'bounds_respected': bool(((p.raw_a >= -1) & (p.raw_a <= 1)).all()), 'faults_immutable': bool(torch.equal(p.raw_a[p.corrupt], p.stuck_raw_a[p.corrupt])), 'fault_cells': int(p.corrupt.sum()), 'fault_pulses': int(p.pulse_count[p.corrupt].sum())}


class Crossbar:
    def __init__(self, plan, condition, lr, evaluation_limit):
        import torch
        from experiments.mnist_analog_relu import staged_runtime as engine
        from experiments.mnist_analog_relu.staged_config import parse_staged_crossbar_config, resolve_staged_crossbar_spec
        from experiments.schema import RunMode
        self.engine = engine
        source = plan['crossbar']
        spec = resolve_staged_crossbar_spec(parse_staged_crossbar_config(json.loads(Path(source['config']).read_text())), RunMode.TRAIN)
        self.spec = spec
        self.device = torch.device('cuda:0')
        self.layout = engine._layout(spec)
        self.origin = engine.load_device_state(Path(source[f'{condition}_p0']))
        self.teacher, _, teacher_hash = engine._load_teacher(Path(source['teacher']), spec=spec, device=self.device)
        if teacher_hash != self.origin.teacher_sha256:
            raise ValueError('Expected the exact original crossbar teacher.')
        self.plant = engine._restore_current(self.origin, layout=self.layout, device=self.device)
        self.port = self.plant.restricted_recovery_update_port()
        self.data = engine.build_mnist_loaders(spec.data, data_seed=spec.runtime.data_seed)
        seed = engine._recovery_seed(runtime_seed=spec.runtime.seed, assignment_seed=self.origin.assignment_seed, endpoint_seed=self.origin.endpoint_seed)
        self.open_optimizer = engine.PulseAdam(size=self.plant.size, layout=self.layout, learning_rates=(lr, lr), betas=(0.9, 0.999), epsilon=1e-8, layer_scope='all', nominal_dw_min=engine.population_nominal_step(self.origin.healthy_population), pulse_cap_per_cell=plan['total_pulse_cap'], generator=torch.Generator(device=self.device).manual_seed(seed), device=self.device)
        self.evaluation_limit = evaluation_limit
        self.tolerance = source['verify_tolerance_q']
        self.seeds = {'data_order': spec.runtime.data_seed, 'assignment': self.origin.assignment_seed, 'endpoint': self.origin.endpoint_seed, 'pulse_selection': seed}
        self.source_paths = {'deployment': source[f'{condition}_p0'], 'teacher': source['teacher']}
        self.metadata = {'fault_model': 'native_OM_effective_weight_post_PV_fault', 'digital_scales': self.origin.current.digital_scales, 'coordinate': 'q=a-r', 'nominal_pulse': self.open_optimizer.nominal_dw_min, 'source_revision': source['source_revision']}
        self.start_pulse_count = (self.plant.upward_pulses + self.plant.downward_pulses).clone()

    def apparent(self):
        return self.port.apparent

    def persistent(self):
        return self.plant.persistent

    def gradient(self, inputs, labels):
        import torch
        q = self.apparent().detach().clone().requires_grad_(True)
        logits = self.engine.standard_crossbar_logits(inputs, q, self.layout, digital_scales=self.origin.current.digital_scales)
        loss = self.engine._adam_objective_loss(objective='teacher_kl', logits=logits, labels=labels, inputs=inputs, teacher=self.teacher)
        return torch.autograd.grad(loss, q, only_inputs=True)[0], float(loss.item())

    def open_step(self, gradient):
        value = self.open_optimizer.step(gradient, self.port)
        return {'pulses': value['applied_pulses'], 'verify_rounds': 0, 'blocked_at_cap': value['blocked_at_cap']}

    def pulse(self, direction):
        self.port.pulse(direction)

    def evaluate(self, loader):
        result = self.engine._evaluate_plant_states(plant=self.plant, digital_scales=self.origin.current.digital_scales, layout=self.layout, teacher=self.teacher, loader=loader, device=self.device, maximum_batches=None, sample_limit=self.evaluation_limit)
        return {'apparent': result['apparent_forward'], 'persistent_secondary_diagnostic': result['persistent_diagnostic']}

    def state(self):
        return self.plant.state_dict()

    def restore(self, state):
        faulted = self.origin.current.state_kind == 'faulted'
        self.plant.load_state_dict(state,
            expected_fault_source_population=self.origin.published_population if faulted else None,
            expected_pre_fault_state=self.origin.healthy_p0.plant_state if faulted else None)

    def invariant(self):
        import torch
        p = self.plant
        healthy = ~p.post_deployment_fault_mask
        bounds = ((p.persistent >= p.population.logical_min - 1e-6) & (p.persistent <= p.population.logical_max + 1e-6))[healthy]
        fault_count = (p.upward_pulses + p.downward_pulses - self.start_pulse_count)[~healthy]
        return {'bounds_respected': bool(bounds.all()), 'faults_immutable': bool(torch.equal(p.persistent[~healthy], p.post_deployment_stuck_persistent_q[~healthy])), 'fault_cells': int((~healthy).sum()), 'fault_pulses': int(fault_count.sum())}


def run(args):
    sys.path.insert(0, str(HERE / f'{args.architecture}_code'))
    import torch
    from closed_loop import ClosedLoopAdam
    from training.checkpoint import atomic_torch_save
    from itertools import islice
    torch.set_num_threads(1)
    assert torch.cuda.is_available(), 'Expected real CUDA; CPU fallback is prohibited.'
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    plan = json.loads(args.plan.read_text())
    lr = plan[args.architecture]['learning_rates'][args.condition]
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    start = time.time()
    status = {'status': 'initializing', 'pid': os.getpid(), 'started': start, 'architecture': args.architecture, 'condition': args.condition, 'writer': args.writer, 'heartbeat': start}
    write_json(out / 'status.json', status)
    try:
        adapter = (Drn if args.architecture == 'drn' else Crossbar)(plan, args.condition, lr, args.evaluation_limit)
        closed = ClosedLoopAdam(adapter.apparent, adapter.pulse, learning_rate=lr, tolerance=adapter.tolerance, maximum_pulses=plan['maximum_verify_pulses_per_update'], total_pulse_cap=plan['total_pulse_cap'])
        optimizer = closed if args.writer == 'closed_loop_pv' else adapter.open_optimizer
        before = {'apparent_sha256': tensor_hash(adapter.apparent()), 'persistent_sha256': tensor_hash(adapter.persistent())}
        manifest = {'evidence_class': 'exploratory_noncanonical', 'plan_sha256': file_hash(args.plan), 'architecture': args.architecture, 'condition': args.condition, 'writer': args.writer, 'learning_rate': lr, 'objective': 'teacher_kl', 'primary_state': 'held_apparent', 'epochs': args.epochs, 'maximum_batches': args.maximum_batches, 'evaluation_limit': args.evaluation_limit, 'seeds': adapter.seeds, 'source': adapter.metadata, 'inputs': {k: {'path': str(v), 'sha256': file_hash(v)} for k, v in adapter.source_paths.items()}, 'initial_state': before, 'cuda': torch.cuda.get_device_name(), 'code_sha256': {p.name: file_hash(p) for p in (HERE / 'run_recovery.py', HERE / 'closed_loop.py')}}
        write_json(out / 'manifest.json', manifest)
        initial = adapter.evaluate(adapter.data.validation)
        write_json(out / 'initial.json', initial)
        best = {'epoch': 0, 'validation': initial, 'pulses': 0}
        def snapshot():
            return {'plant': cpu_tree(adapter.state()), 'optimizer': cpu_tree(optimizer.state_dict()), 'train_generator': adapter.data.train_generator.get_state().clone()}
        best_state = snapshot()
        p0_state = best_state['plant']
        def rank(candidate):
            m = candidate['validation']['apparent']
            return m['kl_teacher_student'], -m['student_accuracy'], candidate['epoch']
        records = []
        total_pulses = 0
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.time()
            loss_sum, examples, writer_time = 0., 0, 0.
            stream = hashlib.sha256()
            for batch, (inputs, labels) in enumerate(islice(adapter.data.train, args.maximum_batches), 1):
                stream.update(inputs.numpy().tobytes())
                stream.update(labels.numpy().tobytes())
                inputs, labels = inputs.to(adapter.device), labels.to(adapter.device)
                gradient, loss = adapter.gradient(inputs, labels)
                torch.cuda.synchronize()
                write_start = time.time()
                report = closed.step(gradient) if args.writer == 'closed_loop_pv' else adapter.open_step(gradient)
                torch.cuda.synchronize()
                writer_time += time.time() - write_start
                total_pulses += report['pulses']
                examples += labels.numel()
                loss_sum += loss * labels.numel()
                if batch == 1 or batch % 100 == 0:
                    status.update(status='running', epoch=epoch, batch=batch, pulses=total_pulses, heartbeat=time.time(), writer_seconds=writer_time)
                    write_json(out / 'status.json', status)
            if args.maximum_batches == 3438 and examples != 55000:
                raise RuntimeError(f'Expected 55000 training examples; observed {examples}.')
            validation = adapter.evaluate(adapter.data.validation)
            candidate = {'epoch': epoch, 'validation': validation, 'pulses': total_pulses}
            if rank(candidate) < rank(best):
                best, best_state = candidate, snapshot()
            invariants = adapter.invariant()
            if not invariants['bounds_respected'] or not invariants['faults_immutable']:
                raise RuntimeError(f'Physical invariant failure: {invariants}')
            record = dict(candidate, train_kl=loss_sum/examples, examples=examples, batches=batch, data_stream_sha256=stream.hexdigest(), seconds=time.time()-epoch_start, writer_seconds=writer_time, verify_reads=closed.verify_reads if args.writer == 'closed_loop_pv' else 0, burst_exhaustions=closed.burst_exhaustions if args.writer == 'closed_loop_pv' else 0, invariants=invariants)
            records.append(record)
            with (out / 'metrics.jsonl').open('a') as f:
                f.write(json.dumps(record, allow_nan=False) + '\n')
            atomic_torch_save({'manifest': manifest, 'epoch': epoch, 'state': snapshot(), 'best': best, 'best_state': best_state}, out / 'continuation.pt')
            print(json.dumps({'epoch': epoch, 'validation': validation['apparent'], 'pulses': total_pulses, 'seconds': record['seconds'], 'writer_seconds': writer_time}), flush=True)
        final_state = snapshot()
        adapter.restore(best_state['plant'])
        replay = adapter.evaluate(adapter.data.validation)
        if abs(replay['apparent']['kl_teacher_student'] - best['validation']['apparent']['kl_teacher_student']) > 1e-7:
            raise RuntimeError('Selected checkpoint failed validation replay.')
        selected_test = adapter.evaluate(adapter.data.test) if args.test else None
        final_test, initial_test = None, None
        if args.test:
            adapter.restore(final_state['plant'])
            final_test = adapter.evaluate(adapter.data.test)
            adapter.restore(p0_state)
            initial_test = adapter.evaluate(adapter.data.test)
        atomic_torch_save({'manifest': manifest, 'best': best, 'state': best_state}, out / 'selected.pt')
        result = {'status': 'complete', 'manifest': manifest, 'initial': initial, 'initial_test': initial_test, 'selected': best, 'final': candidate, 'selected_test': selected_test, 'final_test': final_test, 'epochs': records, 'total_seconds': time.time()-start, 'total_pulses': total_pulses, 'verify_reads': closed.verify_reads if args.writer == 'closed_loop_pv' else 0, 'selected_replay_passed': True, 'final_state_hashes': {k: tensor_hash(v) for k, v in final_state['plant'].items() if isinstance(v, torch.Tensor) and k in {'persistent', 'apparent', 'persistent_raw_a', 'apparent_raw_a'}}}
        write_json(out / 'result.json', result)
        status.update(status='complete', heartbeat=time.time(), seconds=time.time()-start)
        write_json(out / 'status.json', status)
    except BaseException as error:
        status.update(status='failed', error=repr(error), heartbeat=time.time())
        write_json(out / 'status.json', status)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, default=HERE / 'plan.json')
    parser.add_argument('--architecture', choices=['drn', 'crossbar'], required=True)
    parser.add_argument('--condition', choices=['healthy', 'faulted'], required=True)
    parser.add_argument('--writer', choices=['open_loop', 'closed_loop_pv'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--maximum-batches', type=int, default=3438)
    parser.add_argument('--evaluation-limit', type=int)
    parser.add_argument('--test', action='store_true')
    run(parser.parse_args())
