"""Reconstruct the exact saved crossbar HWA identity draw with pinned AIHWKit."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis/stochastic_hwa_predeployment_20260908/inputs"
OUT.mkdir(parents=True, exist_ok=True)
baseline = json.loads((ROOT / "analysis/hwa_pre_pv_20260908/crossbar_readout.json").read_text())
master_path = Path(baseline["inputs"]["hwa_master"]["path"])
master = torch.load(master_path, map_location="cpu", weights_only=False)
sampling = master["report"]["sampling"]["sampling"]
population = OUT / "hwa_healthy_assignment_sampled_population.npz"
receipt = OUT / "reconstructed_population.receipt.json"
if population.exists():
    raise FileExistsError(population)
subprocess.run([
    "/home/filip/miniconda3/envs/aihwkit/bin/python3.12", "-m",
    "experiments.reram_program_verify.hwa_population_sampler",
    "--request-json", json.dumps(sampling["request"]),
    "--output", str(population), "--receipt", str(receipt),
], cwd=ROOT / "crossbar_code", check=True)
actual = json.loads(receipt.read_text())
observed_hash = hashlib.sha256(population.read_bytes()).hexdigest()
assert actual["population_fingerprint"] == sampling["population_fingerprint"]
assert observed_hash == sampling["population_sha256"]
proof = dict(status="complete", original_population_sha256=sampling["population_sha256"],
    reconstructed_population_sha256=observed_hash,
    population_fingerprint=actual["population_fingerprint"], exact_file_reconstruction=True,
    reason="Original Akib artifact could not be read: SSH reported no route to host.",
    request=sampling["request"], original_hwa_master=str(master_path),
    original_hwa_master_sha256=hashlib.sha256(master_path.read_bytes()).hexdigest())
(OUT / "reconstruction_verification.json").write_text(json.dumps(proof, indent=2) + "\n")
print(json.dumps(proof))
