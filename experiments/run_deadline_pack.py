"""Run a small concurrent exact-config pack with an absolute UTC deadline."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time


def stamp():
    return datetime.now(timezone.utc).isoformat()


def dump(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path, action="append")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target", required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("/home/filip/datasets/mnist"))
    parser.add_argument("--deadline", required=True, type=float)
    parser.add_argument("--max-seconds", type=float, default=28800)
    parser.add_argument("--expected-seconds", type=float)
    parser.add_argument("--mode", choices=("smoke", "timing", "production"), required=True)
    args = parser.parse_args()
    os.chdir(args.source)
    sys.path.insert(0, str(args.source))
    from experiments.exact_run import load_exact_config
    from experiments.reporting import validate_run

    if not 1 <= len(args.config) <= 3:
        raise ValueError("A pack must contain one to three workers.")
    cutoff = min(args.deadline, time.time() + args.max_seconds)
    if cutoff - time.time() < 120:
        raise ValueError("Insufficient time before the absolute deadline.")
    if args.mode == "production" and (args.expected_seconds is None or
            args.expected_seconds + 300 > cutoff - time.time()):
        raise ValueError("Measured completion estimate does not fit with deadline margin.")
    # The pack owns only a verified empty lane. Never stop another GPU process.
    occupied = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid",
                                       "--format=csv,noheader,nounits"], text=True).strip()
    if occupied:
        raise RuntimeError(f"GPU occupied before pack admission: {occupied}")
    subprocess.run(["sha256sum", "--check", "--quiet", "INPUT_SHA256SUMS"], check=True)
    configs = []
    for path in args.config:
        path = path.resolve()
        data = load_exact_config(path)
        if data["evaluation"]["official_test"]["policy"] != "disabled":
            raise ValueError("Official test must be disabled.")
        if args.mode == "production" and (data.get("max_batches") is not None
                                         or data.get("max_test_batches") is not None):
            raise ValueError("Production must have the full training/evaluation budget.")
        init = Path(data["init_checkpoint_path"])
        if hashlib.sha256(init.read_bytes()).hexdigest() != data["initialization"]["checkpoint_sha256"]:
            raise ValueError("Initializer bytes changed.")
        configs.append((path, data))

    args.output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    env.update(KMP_DISABLE_SHM="1", KMP_SHM_DISABLE="1", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
               PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1",
               MPLCONFIGDIR="/tmp/mpl-read-noise", TF_CPP_MIN_LOG_LEVEL="2",
               EXPERIMENT_SOURCE_COMMIT=(args.source / "SOURCE_COMMIT").read_text().strip(),
               EXPERIMENT_SOURCE_ARCHIVE_SHA256=(args.source / "SOURCE_ARCHIVE_SHA256").read_text().strip(),
               GIT_CEILING_DIRECTORIES=str(args.source.parent))
    record = dict(state="running", mode=args.mode, target=args.target, pid=os.getpid(),
                  started_at=stamp(), deadline_utc=datetime.fromtimestamp(cutoff, timezone.utc).isoformat(),
                  expected_seconds=args.expected_seconds, cases=[], official_test_read=False)
    children = []
    readers = []
    (args.output / "pack_pid").write_text(str(os.getpid()) + "\n")

    def read_log(process, log, item, started):
        with (args.output / f"{item['case']}.events.jsonl").open("x") as events:
            for line in process.stdout:
                log.write(line)
                log.flush()
                match = re.match(r"^(\[Validation\] )?Epoch (\d+)/(\d+) \| Batch (\d+)/(\d+)", line)
                if match:
                    event = dict(seconds=time.monotonic() - started,
                                 validation=bool(match[1]), epoch=int(match[2]),
                                 batch=int(match[4]), loader_batches=int(match[5]))
                    events.write(json.dumps(event) + "\n")
                    events.flush()

    def stop(signum, frame):
        raise InterruptedError(f"Pack received signal {signum}")

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        for path, data in configs:
            out = args.output / path.stem
            command = [sys.executable, "-m", "experiments.exact_run", str(path),
                       "--index", "0", "--output-root", str(out), "--device", "cuda",
                       "--dataset-root", str(args.dataset_root), "--target", args.target,
                       "--study-id", data["study_id"], "--summary-json", str(args.output / f"{path.stem}.summary.json")]
            if args.mode == "smoke":
                command.append("--smoke")
            log = (args.output / f"{path.stem}.log").open("x")
            started = time.monotonic()
            process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, start_new_session=True)
            item = dict(case=path.stem, config=str(path), pid=process.pid, command=command,
                        output=str(out), started_at=stamp(), returncode=None)
            record["cases"].append(item)
            children.append((process, log, item))
            reader = threading.Thread(target=read_log, args=(process, log, item, started), daemon=True)
            reader.start()
            readers.append(reader)
        dump(args.output / "pack_status.json", record)
        while any(process.poll() is None for process, _, _ in children):
            if time.time() >= cutoff - 25:
                raise TimeoutError("Absolute pack deadline reached; unfinished cases retained.")
            for process, log, item in children:
                if process.poll() is not None and item["returncode"] is None:
                    item.update(returncode=process.returncode, completed_at=stamp())
            record["heartbeat_at"] = stamp()
            dump(args.output / "pack_status.json", record)
            time.sleep(2)
        for reader in readers:
            reader.join(timeout=10)
        for process, log, item in children:
            log.close()
            if item["returncode"] is None:
                item.update(returncode=process.returncode, completed_at=stamp())
            results = list(Path(item["output"]).rglob("result.json"))
            item["results"] = [str(p) for p in results]
            item["validation_errors"] = (["Expected exactly one successful result"] if
                len(results) != 1 else list(validate_run(results[0].parent)))
        okay = all(item["returncode"] == 0 and not item["validation_errors"]
                   for item in record["cases"])
        record["state"] = "complete" if okay else "failed"
        record["returncode"] = 0 if okay else 1
    except BaseException as exc:
        record.update(state="failed", returncode=1, error=repr(exc))
        for process, _, _ in children:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        end = time.monotonic() + 15
        while time.monotonic() < end and any(p.poll() is None for p, _, _ in children):
            time.sleep(.2)
        for process, _, item in children:
            if item["returncode"] is not None:
                continue
            # Always check the process group, including grandchildren of an exited parent.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    finally:
        for reader in readers:
            reader.join(timeout=5)
        for process, log, item in children:
            log.close()
            if item["returncode"] is None:
                item.update(returncode=process.poll(), completed_at=stamp())
        record["completed_at"] = stamp()
        dump(args.output / "pack_status.json", record)
        (args.output / "pack_exit_code").write_text(str(record["returncode"]) + "\n")
    return record["returncode"]


if __name__ == "__main__":
    raise SystemExit(main())
