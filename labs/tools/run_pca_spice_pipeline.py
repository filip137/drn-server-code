#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

FUNCTIONS_DIR = Path(__file__).resolve().parent
for path in (FUNCTIONS_DIR,):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from run_pca_flatten import run_pca_flatten  # noqa: E402
from run_spice_validate import main as run_spice_main  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="run_pca_spice_pipeline.py",
        description="Run PCA sweep + flatten, then launch remote SPICE validation.",
    )
    p.add_argument("--config", required=True, help="Path to JSON config.")
    p.add_argument("--model-key", required=True, help="Model key in config['models'].")
    p.add_argument("--weights-pt", required=True, help="Path to model weights (.pt).")
    p.add_argument("--num-iterations", type=int, required=True, help="Minimizer iterations for PCA sweep.")
    p.add_argument("--pca-steps", type=int, default=30, help="Grid steps per PCA axis.")
    p.add_argument("--pca-n-sigma", type=float, default=3.0, help="Extent of PCA sweep in stddevs.")
    p.add_argument("--pca-batch-size", type=int, default=None, help="Override PCA grid batch size.")
    p.add_argument("--input-layer", default=None, help="Input layer to flatten (default: first key).")
    p.add_argument(
        "--output-root",
        default="/home/filip/paper_simulation_results_/cases",
        help="Root folder for outputs (creates a timestamped subfolder).",
    )
    p.add_argument("--mode", default="pca", help="Mode for validate_cd_results.py.")
    p.add_argument("--host", default="filip@maxica", help="SSH host.")
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

    outputs = run_pca_flatten(
        config_path=args.config,
        model_key=args.model_key,
        weights_pt=args.weights_pt,
        num_iterations=args.num_iterations,
        pca_steps=args.pca_steps,
        pca_n_sigma=args.pca_n_sigma,
        pca_batch_size=args.pca_batch_size,
        input_layer=args.input_layer,
        output_root=args.output_root,
    )

    spice_args = [
        "--weights",
        str(outputs["weights_flat"]),
        "--inputs",
        str(outputs["inputs_flat"]),
        "--config",
        str(outputs["config_copy"]),
        "--model",
        args.model_key,
        "--mode",
        args.mode,
        "--host",
        args.host,
        "--nom-prefix",
        args.nom_prefix,
        "--max-prefix",
        args.max_prefix,
        "--remote-workdir",
        args.remote_workdir,
        "--remote-python",
        args.remote_python,
        "--remote-script",
        args.remote_script,
    ]
    if args.include_bias:
        spice_args.append("--include-bias")

    return run_spice_main(spice_args)


if __name__ == "__main__":
    raise SystemExit(main())
