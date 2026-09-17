"""Read-only replay after validation-based recovery selection."""
import json
from pathlib import Path
import torch
from experiments.artifacts import atomic_write_json, sha256_file
from experiments.mnist_shared import build_mnist_loaders
from experiments.schema import RunMode
from experiments.mnist_analog_relu.staged_config import parse_staged_crossbar_config, resolve_staged_crossbar_spec
from experiments.mnist_analog_relu.staged_artifacts import load_device_state
from experiments.mnist_analog_relu import staged_runtime as engine
from training.ibm_om_standard_crossbar import tensor_sha256

def run(args):
    if args.output.exists():raise FileExistsError(args.output)
    torch.set_num_threads(1)
    spec=resolve_staged_crossbar_spec(parse_staged_crossbar_config(json.loads(args.config.read_text())),RunMode.TRAIN)
    device=engine._cuda_device(spec)
    teacher,_,teacher_sha=engine._load_teacher(args.teacher_weights,spec=spec,device=device)
    state=load_device_state(args.device_state)
    layout=engine._layout(spec)
    engine._validate_origin(state,spec=spec,teacher_sha256=teacher_sha,layout=layout,allowed_roles={'healthy_p0','faulted_p0','adam_final'})
    plant=engine._restore_current(state,layout=layout,device=device)
    before=dict(apparent=tensor_sha256(plant.apparent),persistent=tensor_sha256(plant.persistent))
    data=build_mnist_loaders(spec.data,data_seed=spec.runtime.data_seed)
    evaluation=engine._evaluate_plant_states(plant=plant,digital_scales=state.current.digital_scales,layout=tuple(layout),teacher=teacher,loader=getattr(data,args.split),device=device,maximum_batches=None,sample_limit=None)
    after=dict(apparent=tensor_sha256(plant.apparent),persistent=tensor_sha256(plant.persistent))
    assert before==after,'Readout mutated held device state.'
    assert evaluation['apparent_forward']['examples']==(10000 if args.split=='test' else 5000)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    result=dict(schema='ebl.recovery_readout.v1',status='complete',architecture='crossbar',split=args.split,evidence_class='exploratory_noncanonical',state_path=str(args.device_state.resolve()),state_sha256=sha256_file(args.device_state),teacher_sha256=teacher_sha,assignment_seed=state.assignment_seed,endpoint_seed=state.endpoint_seed,state_role=state.role,evaluation=evaluation,unchanged_state_check=dict(before=before,after=after,passed=True),writes=0,selection_performed=False,cuda_device=torch.cuda.get_device_name())
    atomic_write_json(args.output,result)
    print(json.dumps(dict(output=str(args.output),split=args.split,kl=evaluation['apparent_forward']['kl_teacher_student'],accuracy=evaluation['apparent_forward']['student_accuracy'])),flush=True)
    return 0
