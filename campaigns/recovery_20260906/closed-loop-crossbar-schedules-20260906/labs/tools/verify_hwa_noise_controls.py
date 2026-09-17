"""Check unchanged deployment physics against the original saved P&V states."""
import argparse
from pathlib import Path
import torch
from run_hwa_noise_sweep import Crossbar, Drn, ROOT, WORK, read, write

parser = argparse.ArgumentParser()
parser.add_argument("--architecture", choices=("crossbar", "drn"), required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
torch.set_num_threads(1)
torch.manual_seed(42)
engine = (Crossbar if args.architecture == "crossbar" else Drn)(True)
path = Path(engine.metadata["original_hwa"]["path"])
if args.architecture == "crossbar":
    old = engine.e.load_hwa_master(path)
    masters = tuple(old.fixed_final_master_q[s].to(engine.device) for s in engine.f.layer_cell_slices(engine.layout))
    plant, _ = engine.program(masters, "evaluation")
    expected = engine.e._restore_current(engine.origin, layout=engine.layout, device=engine.device)
    assert torch.equal(plant.apparent, expected.apparent)
    assert torch.equal(plant.persistent, expected.persistent)
    expected_clean = read(ROOT / "analysis/hwa_pre_pv_20260908/crossbar_readout.json")["unmodified_digital_master_diagnostic"]
else:
    old = engine.e._load_torch(path, schema="ebl.figure6_om_256_hwa_master")
    masters = engine.e._masters_to_device(old["selected_master"], engine.device)
    plant, _ = engine.program(masters, "evaluation")
    source = read(WORK / "closed-loop-recovery-20260906/plan.json")["drn"]
    deployment = engine.e._load_torch(Path(source["deployment"]), schema="ebl.figure6_om_256_deployment_replica")
    expected = deployment["targets"]["hwa"]["clean_adam_start"]
    assert torch.equal(plant.apparent_raw_a.cpu(), expected["apparent_raw_a"])
    assert torch.equal(plant.raw_a.cpu(), expected["persistent_raw_a"])
    expected_clean = read(ROOT / "analysis/digital_hwa_open_loop_20260908/drn_digital_readout.json")["evaluation"]
clean = engine.clean(masters, list(engine.data.test), None)
assert clean["examples"] == 10000
assert clean["student_accuracy"] == expected_clean["student_accuracy"]
assert abs(clean["kl_teacher_student"] - expected_clean["kl_teacher_student"]) < 1e-7
result = dict(status="passed", architecture=args.architecture, original_master=str(path),
    exact_original_PV_apparent_state=True, exact_original_PV_persistent_state=True,
    clean_accuracy_and_KL_replay=True, clean_test=clean)
write(args.output / f"{args.architecture}_unchanged_controls.json", result)
print(result)
