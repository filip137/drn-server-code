"""Read-only DRN replay on the common official test cohort after selection."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'ibm-om-cell-aware-quantized'))
import torch
from experiments.mnist_relu_drn import figure6_om_256_ladder as engine
from experiments.artifacts import atomic_write_json,sha256_file

def main(args):
    if args.output.exists():raise FileExistsError(args.output)
    torch.set_num_threads(1)
    plan=json.loads(args.plan.read_text())
    replica=next(r for r in plan['drn']['replicas'] if r['id']==args.replica)
    deployment=engine._load_torch(Path(replica['deployment']),schema='ebl.figure6_om_256_deployment_replica')
    prepared=engine._load_torch(Path(plan['drn']['prepared']),schema='ebl.figure6_om_256_ladder_prepared')
    config=json.loads(Path(plan['drn']['original_config']).read_text())
    spec,teacher,runtime=engine._runtime(config,teacher_path=Path(prepared['teacher']['path']),gain=float(prepared['fixed_logit_gain']),smoke=False)
    stack=runtime['stack']
    field=engine._field_from_payload(deployment['endpoint_field'],device=stack.device)
    population=engine.load_om_array_population(Path(replica['repaired_population']))
    if args.state:
        payload=torch.load(args.state,map_location='cpu',weights_only=False)
        if payload.get('schema')=='exploratory_drn_schedule_state':
            state=payload['state']['plant']
            selected_epoch=payload['epoch']
            assert payload['manifest']['replica']['id']==args.replica
            assert payload['manifest']['arm']['condition']==args.condition
        elif payload.get('schema')=='ebl.figure6_om_256_adam_arm':
            report=payload['report']
            state=report['selected_plant_state']
            selected_epoch=report['selected_epoch']
        else:raise ValueError('Expected a declared selected DRN recovery checkpoint.')
        state_path=args.state
    else:
        state=deployment['targets']['hwa']['clean_adam_start' if args.condition=='healthy' else 'corrupt_adam_start']
        selected_epoch=0;state_path=Path(replica['deployment'])
    plant=engine._plant_from_state(field=field,population=population,state=state)
    before=dict(apparent=engine._tensor_sha256(plant.apparent_raw_a),persistent=engine._tensor_sha256(plant.raw_a))
    data=engine._loaders(spec,data_seed=int(deployment['seeds']['data_order']))
    evaluation=engine._evaluate_plant(stack=stack,teacher=teacher,loader=getattr(data,args.split),plant=plant,sample_limit=None)
    after=dict(apparent=engine._tensor_sha256(plant.apparent_raw_a),persistent=engine._tensor_sha256(plant.raw_a))
    assert before==after
    assert evaluation['apparent']['examples']==(10000 if args.split=='test' else 5000)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    result=dict(schema='ebl.recovery_readout.v1',status='complete',architecture='drn',split=args.split,condition=args.condition,replica=args.replica,evidence_class='exploratory_noncanonical',state_path=str(state_path.resolve()),state_sha256=sha256_file(state_path),selected_epoch=selected_epoch,evaluation=evaluation,unchanged_state_check=dict(before=before,after=after,passed=True),writes=0,selection_performed=False,cuda_device=torch.cuda.get_device_name())
    atomic_write_json(args.output,result)
    print(json.dumps(dict(output=str(args.output),split=args.split,kl=evaluation['apparent']['kl_teacher_student'],accuracy=evaluation['apparent']['student_accuracy'])),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--replica',required=True);p.add_argument('--condition',choices=['healthy','faulted'],required=True);p.add_argument('--state',type=Path);p.add_argument('--split',choices=['validation','test'],default='validation');p.add_argument('--output',type=Path,required=True);main(p.parse_args())
