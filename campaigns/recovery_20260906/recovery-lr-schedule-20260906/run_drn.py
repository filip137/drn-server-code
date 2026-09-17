"""Exploratory recovery from immutable Figure-6/OM DRN deployments."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
CODE = HERE.parent / 'ibm-om-cell-aware-quantized'
sys.path.insert(0, str(CODE))
import torch
from experiments.mnist_relu_drn import figure6_om_256_ladder as engine
from training.checkpoint import atomic_torch_save

def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temp.replace(path)

def rate_for_epoch(arm, epoch):
    schedule = arm['schedule']
    progress = min((epoch-1)/(schedule['decay_epochs']-1), 1.0)
    factor = schedule['final_factor']**progress if schedule['kind']=='exponential' else 1.0
    return arm['learning_rate']*factor

def rank(evaluation, fraction):
    primary = evaluation['apparent']
    return (primary['kl_teacher_student'], -primary['student_accuracy'], fraction)

def run(args):
    plan = json.loads(args.plan.read_text())
    arm = next(a for a in plan['drn']['arms'] if a['arm_id']==args.arm)
    replica = next(r for r in plan['drn']['replicas'] if r['id']==args.replica)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    start = time.time()
    status = dict(status='initializing', arm=args.arm, replica=args.replica, pid=os.getpid(), started_at=start)
    write_json(out/'status.json', status)
    try:
        torch.set_num_threads(1)
        config = json.loads(Path(plan['drn']['original_config']).read_text())
        deployment_path = Path(replica['deployment'])
        deployment = engine._load_torch(deployment_path, schema='ebl.figure6_om_256_deployment_replica')
        prepared = engine._load_torch(Path(plan['drn']['prepared']), schema='ebl.figure6_om_256_ladder_prepared')
        spec, teacher, runtime = engine._runtime(config, teacher_path=Path(prepared['teacher']['path']), gain=float(prepared['fixed_logit_gain']), smoke=False)
        stack = runtime['stack']
        population = engine.load_om_array_population(Path(replica['repaired_population']))
        field = engine._field_from_payload(deployment['endpoint_field'], device=stack.device)
        target = deployment['targets']['hwa']
        original_state = target['clean_adam_start' if arm['condition']=='healthy' else 'corrupt_adam_start']
        plant = engine._plant_from_state(field=field, population=population, state=original_state)
        loaders = engine._loaders(spec, data_seed=int(deployment['seeds']['data_order']))
        optimizer = engine.PersistentFigure6OmPulseAdam(plant, learning_rate_progress=arm['learning_rate'], pulse_cap=640, pulse_selection_seed=int(deployment['seeds']['adam_selection']), beta1=0.9, beta2=0.999, epsilon=config['adam']['epsilon'])
        def evaluate(loader):
            return engine._evaluate_plant(stack=stack, teacher=teacher, loader=loader, plant=plant, sample_limit=64 if args.smoke else None)
        def snapshot():
            return dict(plant=engine._cpu_tree(plant.state_dict()), optimizer=engine._cpu_tree(optimizer.state_dict()), train_generator=loaders.train_generator.get_state().cpu().clone())
        initial = evaluate(loaders.validation)
        best = dict(epoch=0.0, validation=initial, pulses=0)
        best_state = snapshot()
        candidates = [best]
        manifest = dict(evidence_class='exploratory_noncanonical', objective='teacher_KL', primary_state='held_apparent', selection='minimum_validation_KL_then_accuracy_then_earlier_fraction_including_P0', arm=arm, replica=replica, config=config, deployment_sha256=engine.sha256_file(deployment_path), source_code=str(CODE), source_revision=plan['drn']['source_revision'], seeds=deployment['seeds'], fixed_gain=float(prepared['fixed_logit_gain']), smoke=args.smoke, cuda_device=torch.cuda.get_device_name())
        write_json(out/'manifest.json', manifest)
        write_json(out/'initial.json', initial)
        epochs = 2 if args.smoke else arm['epochs']
        total_pulses = 0
        with (out/'metrics.jsonl').open('w', buffering=1) as metrics_file:
            for epoch in range(1, epochs+1):
                lr = rate_for_epoch(arm, epoch)
                optimizer.learning_rate_progress = lr
                train_kl = 0.0
                examples = 0
                max_batches = 2 if args.smoke else None
                for batch, (inputs, labels) in enumerate(engine.limited(loaders.train, max_batches),1):
                    inputs = inputs.to(stack.device, dtype=torch.float32)
                    labels = labels.to(stack.device, dtype=torch.long)
                    physical, m = engine._one_forward_gradient(stack=stack, teacher=teacher, inputs=inputs, labels=labels, full_g=plant.apparent_full_conductance)
                    pulse = optimizer.step(physical)
                    total_pulses += pulse.pulsed_cells
                    examples += int(m['examples'])
                    train_kl += float(m['kl_sum'])
                    if batch % 500 == 0:
                        status.update(status='running', epoch=epoch, batch=batch, pulses=total_pulses, heartbeat=time.time())
                        write_json(out/'status.json', status)
                    if not args.smoke and epoch==1 and batch in {860,1719,2579}:
                        validation = evaluate(loaders.validation)
                        fraction = examples/55000
                        candidate = dict(epoch=fraction, validation=validation, pulses=total_pulses)
                        candidates.append(candidate)
                        metrics_file.write(json.dumps(dict(kind='within_first_epoch', **candidate, learning_rate=lr))+'\n')
                        if rank(validation,fraction)<rank(best['validation'],best['epoch']):
                            best, best_state = candidate, snapshot()
                if not args.smoke and (examples!=55000 or batch!=3438):
                    raise RuntimeError(f'Expected 55000 examples and 3438 batches; observed {examples}, {batch}.')
                validation = evaluate(loaders.validation)
                candidate = dict(epoch=float(epoch), validation=validation, pulses=total_pulses)
                candidates.append(candidate)
                if rank(validation,epoch)<rank(best['validation'],best['epoch']):
                    best, best_state = candidate, snapshot()
                record = dict(kind='epoch', **candidate, learning_rate=lr, training_kl=train_kl/examples, examples=examples, batches=batch, elapsed_seconds=time.time()-start)
                metrics_file.write(json.dumps(record)+'\n')
                status.update(status='running', epoch=epoch, batch=batch, pulses=total_pulses, validation_kl=validation['apparent']['kl_teacher_student'], selected_epoch=best['epoch'], selected_kl=best['validation']['apparent']['kl_teacher_student'], heartbeat=time.time())
                write_json(out/'status.json', status)
                print(f"{args.arm}/{args.replica} epoch={epoch}/{epochs} lr={lr:.3g} KL={validation['apparent']['kl_teacher_student']:.6f} best={best['validation']['apparent']['kl_teacher_student']:.6f}@{best['epoch']:.3f}", flush=True)
        final_state = snapshot()
        atomic_torch_save(dict(schema='exploratory_drn_schedule_state', manifest=manifest, epoch=float(epochs), state=final_state, best=best, best_state=best_state), out/'final_state.pt')
        atomic_torch_save(dict(schema='exploratory_drn_schedule_state', manifest=manifest, epoch=best['epoch'], state=best_state, best=best), out/'selected_state.pt')
        plant.load_state_dict(best_state['plant'])
        selected_replay = evaluate(loaders.validation)
        if abs(selected_replay['apparent']['kl_teacher_student']-best['validation']['apparent']['kl_teacher_student'])>1e-7:
            raise RuntimeError('Selected state did not replay its validation KL.')
        selected_test = evaluate(loaders.test) if args.test and not args.smoke else None
        result = dict(status='complete', evidence_class='exploratory_noncanonical', arm=arm, replica=replica['id'], epochs_completed=epochs, initial=initial, selected=best, final=candidate, selected_test=selected_test, candidates=candidates, total_pulses=total_pulses, elapsed_seconds=time.time()-start, selected_replay_passed=True)
        write_json(out/'result.json', result)
        status.update(status='complete', heartbeat=time.time(), elapsed_seconds=time.time()-start)
        write_json(out/'status.json', status)
    except BaseException as error:
        status.update(status='failed', error=repr(error), heartbeat=time.time())
        write_json(out/'status.json',status)
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--arm',required=True)
    p.add_argument('--replica',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--test',action='store_true')
    run(p.parse_args())
