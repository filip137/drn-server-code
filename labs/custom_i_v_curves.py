from pathlib import Path

import numpy as np
import torch


def double_exponential_diode(v_off, n_points, min, max, *, I_s, V_t):
    """
    Generate samples for K * sinh(v / V_t), where K = 2 * I_s * exp(-v_off / V_t).

    Returns:
        Tensor with shape (2, n_points): [i_values, v_values].
    """
    v = torch.linspace(min, max, int(n_points))
    v_off = torch.as_tensor(v_off, dtype=v.dtype, device=v.device)
    V_t = torch.as_tensor(V_t, dtype=v.dtype, device=v.device)
    K = 2.0 * I_s * torch.exp(-v_off / V_t)
    i = K * torch.sinh(v / V_t)
    return torch.stack((i, v), dim=0)


def write_double_exponential_diode_npz(
    output_path,
    v_off,
    n_points,
    min,
    max,
    *,
    I_s,
    V_t,
):
    """Write an i-v curve npz with keys: iv, i, v."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    iv = double_exponential_diode(v_off, n_points, min, max, I_s=I_s, V_t=V_t)
    i = iv[0].detach().cpu().numpy()
    v = iv[1].detach().cpu().numpy()
    np.savez(output_path, iv=iv.detach().cpu().numpy(), i=i, v=v)
    return output_path


def write_spice_table_from_iv_npz(
    npz_path,
    out_path,
    *,
    subckt_name="exp_model",
    node_a="a",
    node_b="b",
    v_fmt="{: .6g}",
    i_fmt="{: .6g}",
    g_name="GEXP",
):
    """Write a SPICE subcircuit file with a TABLE-based VCCS from an iv npz."""
    npz_path = Path(npz_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with np.load(npz_path) as data:
        if "iv" in data:
            iv = data["iv"]
            if iv.ndim != 2 or iv.shape[0] != 2:
                raise ValueError(f"Unexpected iv shape in {npz_path}: {iv.shape}")
            i_vals = iv[0]
            v_vals = iv[1]
        elif "i" in data and "v" in data:
            i_vals = data["i"]
            v_vals = data["v"]
        else:
            raise KeyError("Expected keys 'iv' or both 'i' and 'v' in npz.")

    if i_vals.shape != v_vals.shape:
        raise ValueError(f"Shape mismatch: i {i_vals.shape}, v {v_vals.shape}")

    lines = []
    lines.append("* ------------------------------------------------------------")
    lines.append("* Experimental tabulated diode: I = f(V)")
    lines.append("* Implemented as VCCS with TABLE mapping V(a,b) -> I(a->b)")
    lines.append("* ------------------------------------------------------------")
    lines.append(f".SUBCKT {subckt_name} {node_a} {node_b}")
    lines.append("")
    lines.append("* G element: current from node a to node b")
    lines.append(f"{g_name} {node_a} {node_b} TABLE V({node_a},{node_b})=")

    for v, i in zip(v_vals, i_vals):
        lines.append(f"+ ( {v_fmt.format(v)}  {i_fmt.format(i)} )")

    lines.append("")
    lines.append(f".ENDS {subckt_name}")
    lines.append("")

    out_path.write_text("\n".join(lines))
    return out_path


def _build_arg_parser():
    import argparse

    p = argparse.ArgumentParser(
        prog="custom_i_v_curves.py",
        description="Generate and save experimental i-v curves.",
    )
    p.add_argument("--output", help="Output .npz path.")
    p.add_argument("--v-off", type=float, help="Diode offset voltage.")
    p.add_argument("--n-points", type=int, help="Number of samples.")
    p.add_argument("--v-min", type=float, help="Minimum voltage.")
    p.add_argument("--v-max", type=float, help="Maximum voltage.")
    p.add_argument("--I-s", type=float, dest="I_s", help="Diode saturation current.")
    p.add_argument("--V-t", type=float, dest="V_t", help="Thermal voltage.")
    p.add_argument("--spice-from-npz", help="Input .npz path to convert to SPICE table.")
    p.add_argument("--spice-output", help="Output SPICE .subckt path.")
    p.add_argument("--spice-subckt", default="exp_model", help="Subcircuit name.")
    p.add_argument("--spice-node-a", default="a", help="Node a name.")
    p.add_argument("--spice-node-b", default="b", help="Node b name.")
    p.add_argument("--spice-v-fmt", default="{: .6g}", help="Voltage format string.")
    p.add_argument("--spice-i-fmt", default="{: .6g}", help="Current format string.")
    p.add_argument("--spice-g-name", default="GEXP", help="VCCS element name.")
    return p


def main(argv=None) -> int:
    args = _build_arg_parser().parse_args(argv)
    if args.spice_from_npz:
        if not args.spice_output:
            raise SystemExit("--spice-output is required with --spice-from-npz.")
        write_spice_table_from_iv_npz(
            args.spice_from_npz,
            args.spice_output,
            subckt_name=args.spice_subckt,
            node_a=args.spice_node_a,
            node_b=args.spice_node_b,
            v_fmt=args.spice_v_fmt,
            i_fmt=args.spice_i_fmt,
            g_name=args.spice_g_name,
        )
        print(f"Wrote SPICE table to {args.spice_output}")
        return 0

    missing = [
        name
        for name, val in [
            ("--output", args.output),
            ("--v-off", args.v_off),
            ("--n-points", args.n_points),
            ("--v-min", args.v_min),
            ("--v-max", args.v_max),
            ("--I-s", args.I_s),
            ("--V-t", args.V_t),
        ]
        if val is None
    ]
    if missing:
        raise SystemExit(f"Missing required args for generation: {', '.join(missing)}")

    write_double_exponential_diode_npz(
        args.output,
        v_off=args.v_off,
        n_points=args.n_points,
        min=args.v_min,
        max=args.v_max,
        I_s=args.I_s,
        V_t=args.V_t,
    )
    print(f"Wrote i-v curve to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
