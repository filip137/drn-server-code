#!/usr/bin/env python3
import argparse
from pathlib import Path
import shlex
import subprocess


def _build_remote_cmd(args) -> str:
    quoted = {
        "workdir": shlex.quote(args.remote_workdir),
        "python": shlex.quote(args.remote_python),
        "script": shlex.quote(args.remote_script),
        "mode": shlex.quote(args.mode),
        "weights": shlex.quote(args.weights),
        "inputs": shlex.quote(args.inputs),
        "config": shlex.quote(args.config),
        "model": shlex.quote(args.model),
    }
    cmd = (
        f"cd {quoted['workdir']} && source .cshrc && "
        f"{quoted['python']} {quoted['script']} --mode {quoted['mode']} "
        f"--weights {quoted['weights']} --inputs {quoted['inputs']} "
        f"--config {quoted['config']} --model {quoted['model']}"
    )
    if args.include_bias:
        cmd += " --include-bias"
    return cmd


def _expand_remote_home(path_str, remote_user):
    if not path_str or not path_str.startswith("~/"):
        return path_str
    home = f"/home/{remote_user}" if remote_user else "/home/filip"
    return str(Path(home) / path_str[2:])


def _map_nomcool_path(path_str, nom_prefix, max_prefix):
    path = Path(path_str).expanduser()
    nom_prefix = Path(nom_prefix).expanduser()
    try:
        relative = path.relative_to(nom_prefix)
    except ValueError:
        return str(path)
    return str(Path(max_prefix) / relative)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="run_spice_validate.py",
        description="Run remote SPICE validation via SSH.",
    )
    p.add_argument("--weights", required=True, help="Path to flattened weights .npz (remote path).")
    p.add_argument("--inputs", required=True, help="Path to flattened inputs .npz (remote path).")
    p.add_argument("--config", required=True, help="Path to config.json (remote path).")
    p.add_argument("--model", required=True, help="Model key (e.g., drn-xs).")
    p.add_argument("--mode", default="pca", help="Mode for validate_cd_results.py.")
    p.add_argument("--host", default="filip@maxicao", help="SSH host.")
    p.add_argument(
        "--nom-prefix",
        default="/home/filip/paper_simulation_results_",
        help="Nom-cool path prefix to map to maxicao paths.",
    )
    p.add_argument(
        "--max-prefix",
        default="/home/filip/paper_maxicao_simulations",
        help="Maxicao path prefix to map to.",
    )
    p.add_argument("--remote-workdir", default="/home/filip/CMOS130", help="Remote working directory.")
    p.add_argument(
        "--remote-python",
        default="/home/filip/miniconda3/envs/mycondaenv/bin/python",
        help="Remote python executable.",
    )
    p.add_argument(
        "--remote-script",
        default="/home/filip/simulations/improved_simulation_functions/validate_cd_results.py",
        help="Remote validation script.",
    )
    p.add_argument("--include-bias", action="store_true", help="Pass --include-bias to the remote script.")
    args = p.parse_args(argv)

    remote_user = args.host.split("@", 1)[0] if "@" in args.host else None
    args.remote_workdir = _expand_remote_home(args.remote_workdir, remote_user)
    args.remote_python = _expand_remote_home(args.remote_python, remote_user)
    args.remote_script = _expand_remote_home(args.remote_script, remote_user)
    args.weights = _map_nomcool_path(args.weights, args.nom_prefix, args.max_prefix)
    args.inputs = _map_nomcool_path(args.inputs, args.nom_prefix, args.max_prefix)
    args.config = _map_nomcool_path(args.config, args.nom_prefix, args.max_prefix)

    remote_cmd = _build_remote_cmd(args)
    subprocess.run(["ssh", args.host, remote_cmd], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
