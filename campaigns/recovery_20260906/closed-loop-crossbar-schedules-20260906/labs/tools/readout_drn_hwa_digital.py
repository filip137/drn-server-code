"""Read-only digital DRN HWA at the original endpoint law's zero-noise values."""
import hashlib
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis/digital_hwa_open_loop_20260908"
path = OUT / "drn_digital_readout.json"
if path.exists():
    raise FileExistsError(path)
sys.path.insert(0, str(ROOT.parent / "ibm-om-cell-aware-quantized"))
from experiments.mnist_relu_drn import figure6_om_256_ladder as engine
from experiments.mnist_relu_drn.hfo2_figure6_endpoint_regimes import RESET_STATE_CENTER, SET_STATE_CENTER

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

start = time.time()
torch.set_num_threads(1)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
assert torch.cuda.is_available()
# The runtime performs an actual CUDA allocation, matmul and synchronization.
source_path = ROOT / "analysis/stochastic_hwa_predeployment_20260908/drn_readout.json"
source = json.loads(source_path.read_text())
inputs = source["inputs"]
for record in inputs.values():
    assert sha(record["path"]) == record["sha256"]
hwa = engine._load_torch(Path(inputs["hwa_master"]["path"]), schema="ebl.figure6_om_256_hwa_master")
config_path = Path(source["original_config"]["path"])
assert sha(config_path) == source["original_config"]["sha256"]
spec, teacher, runtime = engine._runtime(json.loads(config_path.read_text()),
    teacher_path=Path(inputs["teacher"]["path"]), gain=source["fixed_logit_gain"], smoke=False)
stack = runtime["stack"]
masters = engine._masters_to_device(hwa["selected_master"], stack.device)
before = [engine._tensor_sha256(m) for m in masters]
assert RESET_STATE_CENTER == 0.1 and SET_STATE_CENTER == 2.0
field = engine.EndpointField(
    reset=tuple(torch.full(shape, RESET_STATE_CENTER, device=stack.device) for shape in engine._SHAPES),
    set=tuple(torch.full(shape, SET_STATE_CENTER, device=stack.device) for shape in engine._SHAPES),
    shapes=engine._SHAPES, layouts=engine.ENDPOINT_LAYOUTS,
    population_report={"kind": "digital_nominal_zero_endpoint_noise"})
full_g = engine.map_masters_to_conductance(masters, field)
engine._apply_full_g(stack, full_g)
loaders = engine._loaders(spec, data_seed=42)
evaluation, _ = engine._evaluate_detailed(stack, teacher, loaders.test, sample_limit=None)
assert evaluation["examples"] == 10000
assert evaluation["fixed_logit_gain"] == source["fixed_logit_gain"]
assert before == [engine._tensor_sha256(m) for m in masters]
for record in inputs.values():
    assert sha(record["path"]) == record["sha256"]
result = dict(status="complete", architecture="drn", evaluation=evaluation,
    readout="frozen_digital_HWA_master_on_nominal_Drn_without_hardware_noise",
    nominal_reset=RESET_STATE_CENTER, nominal_set=SET_STATE_CENTER,
    fixed_logit_gain=source["fixed_logit_gain"], inputs=inputs,
    config=source["original_config"], source_readout=str(source_path),
    source_readout_sha256=sha(source_path), protocol_sha256=sha(OUT / "readout_protocol.md"),
    code_sha256={str(p): sha(p) for p in (Path(__file__), Path(engine.__file__))},
    conductance_sha256=[engine._tensor_sha256(g) for g in full_g],
    writes=0, optimizer_updates=0, calibration_refit=False, stochastic_noise=False,
    input_files_unchanged=True, cuda_canary_passed=True, cuda_device=torch.cuda.get_device_name(),
    seconds=time.time()-start)
path.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
print(json.dumps({"output": str(path), "test_KL": evaluation["kl_teacher_student"],
    "accuracy": evaluation["student_accuracy"], "seconds": result["seconds"]}), flush=True)
