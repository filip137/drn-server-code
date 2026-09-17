"""Matched local HWA noise ablation; select by healthy post-P&V validation KL."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT.parent
SCALES = (1.0, 0.5, 0.25, 0.0)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot(values):
    return tuple(v.detach().cpu().clone() for v in values)


def token(scale):
    return format(scale, "g").replace(".", "p")


def key(metric, epoch):
    return (metric["kl_teacher_student"], -metric["student_accuracy"], epoch)


class Crossbar:
    def __init__(self, smoke):
        sys.path.insert(0, str(ROOT / "crossbar_code"))
        from experiments.mnist_analog_relu import staged_runtime as e
        from experiments.mnist_analog_relu import runtime as f
        from experiments.mnist_analog_relu.staged_config import parse_staged_crossbar_config, resolve_staged_crossbar_spec
        from experiments.schema import RunMode
        from training.ibm_reram_hwa import load_om_array_population
        self.e, self.f = e, f
        old = read(ROOT / "analysis/hwa_pre_pv_20260908/crossbar_readout.json")
        self.config_path = ROOT / "crossbar_code/examples/mnist_analog_relu/ibm_om_crossbar_staged_v2/offchip_hwa.json"
        self.config = read(self.config_path)
        self.spec = resolve_staged_crossbar_spec(parse_staged_crossbar_config(self.config), RunMode.TRAIN)
        self.device = torch.device("cuda:0")
        self.teacher, _, teacher_hash = e._load_teacher(Path(old["inputs"]["teacher"]["path"]), spec=self.spec, device=self.device)
        self.layout = e._layout(self.spec)
        requested, self.scales = e.map_logical_weights(tuple(v.detach().cpu() for v in self.teacher.parameters()), self.layout,
            weight_scaling_omega=self.spec.mapping.trained_source_weight_scaling_omega)
        self.initial = tuple(requested[s].to(self.device) for s in f.layer_cell_slices(self.layout))
        hwa = e.load_hwa_master(Path(old["inputs"]["hwa_master"]["path"]))
        assert tuple(self.scales) == tuple(hwa.digital_scales)
        assert teacher_hash == hwa.teacher_sha256
        pop_path = ROOT / "analysis/stochastic_hwa_predeployment_20260908/inputs/hwa_healthy_assignment_sampled_population.npz"
        self.train_pop = load_om_array_population(pop_path)
        assert self.train_pop.fingerprint == hwa.hwa_population_fingerprint
        self.origin = e.load_device_state(Path(old["inputs"]["deployment"]["path"]))
        self.minimum = self.train_pop.logical_min.to(self.device)
        self.maximum = self.train_pop.logical_max.to(self.device)
        self.rates = tuple(hwa.report["hwa"]["effective_q_learning_rates"])
        self.sigma = hwa.report["hwa"]["forward_noise"]["sigma_q"]
        self.generator = torch.Generator(device=self.device).manual_seed(hwa.report["hwa"]["forward_noise"]["resolved_seed"])
        self.data = e.build_mnist_loaders(self.spec.data, data_seed=42)
        self.metadata = dict(teacher=old["inputs"]["teacher"], config=str(self.config_path), config_sha256=sha(self.config_path),
            rates=list(self.rates), base_sigma_q=self.sigma, digital_scales=list(self.scales),
            original_hwa=old["inputs"]["hwa_master"], train_population=str(pop_path),
            train_population_fingerprint=self.train_pop.fingerprint,
            development_assignment=self.train_pop.assignment_seed, development_program_seed=2091401,
            evaluation_assignment=self.origin.assignment_seed, evaluation_program_seed=self.origin.endpoint_seed,
            deployment_input=old["inputs"]["deployment"], master_bounds=[-1.0, 1.0],
            noise_intervention="Multiply additive Gaussian training sigma only; keep sampled support projection fixed.",
            clean_readout="Unprojected digital master, no cell-bound variation, write noise, programming or faults.")

    def prepare_batch(self, inputs, labels, batch):
        with torch.no_grad():
            self.teacher_logp = F.log_softmax(self.teacher.logits(inputs), dim=1)
            self.teacher_p = self.teacher_logp.exp()
        # Common standard-normal draws across the four arms, including alpha=0.
        self.noise = torch.randn(self.minimum.shape, generator=self.generator, device=self.device) * self.sigma

    def gradient(self, masters, inputs, labels, scale, batch):
        master = torch.cat(tuple(masters))
        base = torch.maximum(torch.minimum(master, self.maximum), self.minimum)
        apparent = base + scale * self.noise
        straight_through = master + (apparent - master).detach()
        logits = self.e.standard_crossbar_logits(inputs, straight_through, self.layout, digital_scales=self.scales)
        loss = (self.teacher_p * (self.teacher_logp - F.log_softmax(logits, dim=1))).sum(dim=1).mean()
        loss.backward()
        return float(loss.detach()) * len(labels)

    def evaluate(self, state, loader, limit):
        return self.f._evaluate(effective_state=state, digital_scales=self.scales, layout=self.layout,
            teacher=self.teacher, loader=loader, device=self.device, maximum_batches=None, sample_limit=limit)

    def clean(self, masters, loader, limit):
        return self.evaluate(torch.cat(tuple(masters)), loader, limit)

    def program(self, masters, role):
        population = self.train_pop if role == "development" else self.origin.healthy_population
        seed = 2091401 if role == "development" else self.origin.endpoint_seed
        return self.f._program_endpoint(population=population, requested=torch.cat(tuple(masters)).detach(),
            assignment_seed=population.assignment_seed, endpoint_seed=seed,
            maximum_pulses=128, tolerance_x=0.023725, device=self.device,
            stream_role=self.e.STAGED_PROGRAM_STREAM_ROLE, random_stream_fingerprint=population.fingerprint)

    def evaluate_plant(self, plant, loader, limit, diagnostic=False):
        result = {"apparent": self.evaluate(plant.apparent, loader, limit)}
        if diagnostic:
            result["persistent_secondary_diagnostic"] = self.evaluate(plant.persistent, loader, limit)
        return result

    def fault(self, plant):
        mask, stuck, _ = self.f._matched_published_fault_overlay(healthy=self.origin.healthy_population,
            published=self.origin.published_population, preset_default_corrupt_devices_prob=0.0,
            enabled_corrupt_devices_prob=0.1348, corrupt_devices_range=0.01)
        return plant.apply_stuck_at_fault_transition(mask=mask, stuck_persistent_q=stuck,
            transition_id=f"staged-assignment-{self.origin.assignment_seed}-endpoint-{self.origin.endpoint_seed}",
            source_population_fingerprint=self.origin.published_population.fingerprint)


class Drn:
    def __init__(self, smoke):
        sys.path.insert(0, str(WORK / "ibm-om-cell-aware-quantized"))
        from experiments.mnist_relu_drn import figure6_om_256_ladder as e
        self.e = e
        old = read(ROOT / "analysis/hwa_pre_pv_20260908/drn_readout.json")
        self.config_path = Path(old["config"]["path"])
        self.config = read(self.config_path)
        prepared_path = Path(old["inputs"]["prepared"]["path"])
        prepared = e._load_torch(prepared_path, schema="ebl.figure6_om_256_ladder_prepared")
        hwa = e._load_torch(Path(old["inputs"]["hwa_master"]["path"]), schema="ebl.figure6_om_256_hwa_master")
        self.spec, self.teacher, runtime = e._runtime(self.config, teacher_path=Path(old["inputs"]["teacher"]["path"]),
            gain=float(prepared["fixed_logit_gain"]), smoke=False)
        self.stack = runtime["stack"]
        self.device = self.stack.device
        self.initial = e._masters_to_device(prepared["direct_master"], self.device)
        self.rates = tuple(hwa["report"]["rates"])
        self.nominal = e.EndpointField(reset=tuple(torch.full(s, 0.1, device=self.device) for s in e._SHAPES),
            set=tuple(torch.full(s, 2.0, device=self.device) for s in e._SHAPES), shapes=e._SHAPES,
            layouts=e.ENDPOINT_LAYOUTS, population_report={"kind": "uniform_nominal_no_variation"})
        base = e._augmented_hwa_bank(self.config, device=self.device, smoke=smoke)
        self.banks = {}
        for scale in SCALES:
            if scale == 1:
                self.banks[scale] = base
            elif scale == 0:
                self.banks[scale] = tuple(self.nominal for _ in base)
            else:
                self.banks[scale] = tuple(e.EndpointField(
                    reset=tuple(0.1 + scale * (v - 0.1) for v in field.reset),
                    set=tuple(2.0 + scale * (v - 2.0) for v in field.set), shapes=field.shapes,
                    layouts=field.layouts, population_report={"training_noise_multiplier": scale}) for field in base)
        self.dev_field = e._field(103001, device=self.device)
        self.eval_field = e._field(104001, device=self.device)
        self.dev_pop = e.load_om_array_population(prepared_path.parent / "om_development_counterfactual_repaired.npz")
        source = read(WORK / "closed-loop-recovery-20260906/plan.json")["drn"]
        self.eval_pop = e.load_om_array_population(Path(source["population"]))
        deployment = e._load_torch(Path(source["deployment"]), schema="ebl.figure6_om_256_deployment_replica")
        self.seeds = deployment["seeds"]
        self.data = e._loaders(self.spec, data_seed=42)
        self.metadata = dict(teacher=old["inputs"]["teacher"], prepared=old["inputs"]["prepared"],
            original_hwa=old["inputs"]["hwa_master"], config=str(self.config_path), config_sha256=sha(self.config_path),
            rates=list(self.rates), fixed_logit_gain=prepared["fixed_logit_gain"],
            training_endpoint_seeds=self.config["endpoint_model"]["train_seeds"], bank_size=len(base),
            development_assignment=self.dev_pop.assignment_seed, development_endpoint_seed=103001,
            development_program_seed=103201, evaluation_assignment=self.eval_pop.assignment_seed,
            evaluation_seeds=self.seeds, master_bounds=[-1.0, 1.0],
            noise_intervention="Interpolate sampled training RESET/SET endpoints toward nominal 0.1/2.0; deployment endpoints and OM write noise unchanged.",
            clean_readout="Original DRN solver, same signed master and four-cell mapping, homogeneous RESET=0.1 and SET=2.0, no write noise or faults.")

    def prepare_batch(self, inputs, labels, batch):
        pass

    def gradient(self, masters, inputs, labels, scale, batch):
        bank = self.banks[scale]
        field = bank[batch % len(bank)]
        physical, metrics = self.e._one_forward_gradient(stack=self.stack, teacher=self.teacher,
            inputs=inputs, labels=labels, full_g=self.e.map_masters_to_conductance(masters, field))
        gradients = self.e.lift_endpoint_gradients(masters, physical, field)
        for master, gradient in zip(masters, gradients, strict=True):
            master.grad = gradient
        return metrics["kl_sum"]

    def clean(self, masters, loader, limit):
        self.e._apply_full_g(self.stack, self.e.map_masters_to_conductance(masters, self.nominal))
        return self.e._evaluate_detailed(self.stack, self.teacher, loader, sample_limit=limit)[0]

    def program(self, masters, role):
        field = self.dev_field if role == "development" else self.eval_field
        population = self.dev_pop if role == "development" else self.eval_pop
        seed = 103201 if role == "development" else self.seeds["program_verify"]
        plant, result = self.e.program_masters_with_verify(masters, field, population, pulse_noise_seed=seed,
            tolerance_progress=0.023725, maximum_pulses=128, noisy_initial_reset_verify=True,
            preaccept_exact_healthy_reset_targets=True)
        return plant, self.e._program_verify_report(target_progress=self.e.master_to_progress(masters, field), plant=plant, result=result)

    def evaluate_plant(self, plant, loader, limit, diagnostic=False):
        if diagnostic:
            return self.e._evaluate_plant(stack=self.stack, teacher=self.teacher, loader=loader, plant=plant, sample_limit=limit)
        self.e._apply_full_g(self.stack, plant.apparent_full_conductance)
        return {"apparent": self.e._evaluate_detailed(self.stack, self.teacher, loader, sample_limit=limit)[0]}

    def fault(self, plant):
        return plant.inject_post_pv_reset_stuck_faults(self.eval_pop.published_corrupt,
            observation_seed=self.seeds["fault_observation"])


def run(args):
    out = args.output / args.architecture
    if (out / "result.json").exists() or (out / "history.jsonl").exists():
        raise FileExistsError(f"Expected a new run directory; found previous results in {out}")
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    def status(**values):
        record = dict(pid=os.getpid(), architecture=args.architecture, heartbeat=time.time(), elapsed_seconds=time.time()-start, **values)
        write(out / "status.json", record)
        print(json.dumps(record), flush=True)
    status(status="initializing")
    try:
        torch.set_num_threads(1)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        assert torch.cuda.is_available()
        x = torch.arange(4096, device="cuda", dtype=torch.float32).reshape(64, 64)
        assert torch.isfinite(x @ x.T).all()
        torch.cuda.synchronize()
        engine = (Crossbar if args.architecture == "crossbar" else Drn)(args.smoke)
        limit = 64 if args.smoke else None
        validation = list(engine.data.validation)
        if args.smoke:
            validation = validation[:4]
        epochs = 1 if args.smoke else 10
        states = {}
        for scale in SCALES:
            masters = tuple(torch.nn.Parameter(v.detach().clone()) for v in engine.initial)
            optimizer = torch.optim.Adam([{"params": [m], "lr": r} for m, r in zip(masters, engine.rates, strict=True)],
                betas=(0.9, 0.999), eps=1e-8)
            states[scale] = dict(masters=masters, optimizer=optimizer, best=None)
        assert all(all(torch.equal(a, b) for a, b in zip(engine.initial, s["masters"])) for s in states.values())
        write(out / "manifest.json", dict(evidence_class="exploratory_noncanonical_model_based", multipliers=SCALES,
            epochs=epochs, training_examples_per_epoch=16 if args.smoke else 55000, validation_examples=64 if args.smoke else 5000,
            initialization="Original pre-HWA teacher-derived master, identical across arms", data_seed=42,
            selection="Minimum healthy post-P&V validation KL over epochs 1..10; ties higher accuracy then earlier epoch. Noise multiplier selected by the same metric, then lower multiplier.",
            deployment_noise_scaled=False, nominal_conductance_limits_scaled=False, calibration_refit=False,
            original_recovery_rerun=False, smoke=args.smoke, source_sha256=sha(__file__), source=str(Path(__file__).resolve()),
            torch_version=torch.__version__, cuda_device=torch.cuda.get_device_name(), actual_cuda_canary_passed=True,
            architecture_inputs=engine.metadata))
        initial_plant, initial_programming = engine.program(engine.initial, "development")
        initial_validation = engine.evaluate_plant(initial_plant, validation, limit)
        write(out / "initial_validation.json", dict(clean=engine.clean(engine.initial, validation, limit),
            post_pv=initial_validation, programming=initial_programming, eligible_for_selection=False))
        del initial_plant
        status(status="training", epoch=0, required_epochs=epochs)
        global_batch = 0
        for epoch in range(1, epochs + 1):
            totals = {s: 0.0 for s in SCALES}
            examples = 0
            order_digest = hashlib.sha256()
            for batch, (inputs, labels) in enumerate(engine.data.train, start=1):
                order_digest.update(inputs.numpy().tobytes())
                order_digest.update(labels.numpy().tobytes())
                inputs = inputs.to(engine.device, dtype=torch.float32)
                labels = labels.to(engine.device, dtype=torch.long)
                engine.prepare_batch(inputs, labels, global_batch)
                for scale, state in states.items():
                    state["optimizer"].zero_grad(set_to_none=True)
                    totals[scale] += engine.gradient(state["masters"], inputs, labels, scale, global_batch)
                    state["optimizer"].step()
                    with torch.no_grad():
                        for master in state["masters"]:
                            master.clamp_(-1.0, 1.0)
                            if not bool(torch.isfinite(master).all()):
                                raise FloatingPointError("Nonfinite HWA master")
                examples += len(labels)
                global_batch += 1
                if batch % 250 == 0:
                    status(status="training", epoch=epoch, batch=batch, examples=examples, required_epochs=epochs)
                if args.smoke:
                    break
            assert examples == (16 if args.smoke else 55000)
            for scale, state in states.items():
                status(status="validation_PV", epoch=epoch, multiplier=scale)
                plant, programming = engine.program(state["masters"], "development")
                evaluation = engine.evaluate_plant(plant, validation, limit)
                metric = evaluation["apparent"]
                record = dict(epoch=epoch, multiplier=scale, training_examples=examples, train_KL=totals[scale]/examples,
                    batch_order_sha256=order_digest.hexdigest(), clean_validation=engine.clean(state["masters"], validation, limit),
                    post_pv_validation=evaluation, programming=programming)
                with (out / "history.jsonl").open("a") as stream:
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
                case = out / f"noise_{token(scale)}"
                case.mkdir(exist_ok=True)
                torch.save({"master": snapshot(state["masters"]), "epoch": epoch, "multiplier": scale}, case / f"epoch_{epoch:02d}.pt")
                if state["best"] is None or key(metric, epoch) < key(state["best"]["post_pv_validation"]["apparent"], state["best"]["epoch"]):
                    state["best"] = record
                    torch.save({"master": snapshot(state["masters"]), "epoch": epoch, "multiplier": scale,
                        "development_programmed_state": plant.state_dict()}, case / "selected.pt")
                    write(case / "selection.json", record)
                del plant
                status(status="epoch_complete", epoch=epoch, multiplier=scale, validation_KL=metric["kl_teacher_student"],
                    validation_accuracy=metric["student_accuracy"], clean_accuracy=record["clean_validation"]["student_accuracy"])
        selected_scale = min(SCALES, key=lambda s: (*key(states[s]["best"]["post_pv_validation"]["apparent"], states[s]["best"]["epoch"])[:2], s))
        # Freeze selection before opening test data. All test arms are reported.
        write(out / "selection.json", dict(selected_multiplier=selected_scale,
            per_multiplier={str(s): states[s]["best"] for s in SCALES}, test_used=False))
        test = list(engine.data.test)
        if args.smoke:
            test = test[:4]
        results = []
        for scale in SCALES:
            status(status="final_test", multiplier=scale)
            case = out / f"noise_{token(scale)}"
            saved = torch.load(case / "selected.pt", map_location="cpu", weights_only=False)
            masters = tuple(v.to(engine.device) for v in saved["master"])
            clean = engine.clean(masters, test, limit)
            plant, programming = engine.program(masters, "evaluation")
            healthy = engine.evaluate_plant(plant, test, limit, diagnostic=True)
            healthy_state = plant.state_dict()
            fault_receipt = engine.fault(plant)
            faulted = engine.evaluate_plant(plant, test, limit, diagnostic=True)
            torch.save({"master": snapshot(masters), "healthy_programmed_state": healthy_state,
                "faulted_programmed_state": plant.state_dict(), "architecture_inputs": engine.metadata}, case / "deployment.pt")
            result = dict(multiplier=scale, selected_epoch=saved["epoch"], selected_by_validation=scale == selected_scale,
                clean_test=clean, healthy_pv_test=healthy, faulted_pv_test=faulted,
                programming=programming, fault_receipt=fault_receipt,
                selected_validation=states[scale]["best"]["post_pv_validation"]["apparent"])
            assert clean["examples"] == healthy["apparent"]["examples"] == faulted["apparent"]["examples"] == (64 if args.smoke else 10000)
            write(case / "result.json", result)
            results.append(result)
        write(out / "result.json", dict(status="complete", architecture=args.architecture, selected_multiplier=selected_scale,
            smoke=args.smoke, results=results, seconds=time.time()-start, test_used_for_selection=False))
        status(status="complete", completed_cases=len(results), selected_multiplier=selected_scale)
    except BaseException as error:
        status(status="failed", error=repr(error))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=("crossbar", "drn"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
