from pathlib import Path
import argparse
import numpy as np


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
    """
    Read an iv .npz (keys: iv or i/v) and write a SPICE TABLE VCCS subckt.
    """
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

    lines = [
        "* ------------------------------------------------------------",
        "* Experimental tabulated diode: I = f(V)",
        "* Implemented as VCCS with TABLE mapping V(a,b) -> I(a->b)",
        "* ------------------------------------------------------------",
        f".SUBCKT {subckt_name} {node_a} {node_b}",
        "",
        "* G element: current from node a to node b",
        f"{g_name} {node_a} {node_b} TABLE V({node_a},{node_b})=",
    ]
    for v, i in zip(v_vals, i_vals):
        lines.append(f"+ ( {v_fmt.format(v)}  {i_fmt.format(i)} )")
    lines += ["", f".ENDS {subckt_name}", ""]

    out_path.write_text("\n".join(lines))
    return out_path


def _build_arg_parser():
    p = argparse.ArgumentParser(
        prog="iv_spice_table.py",
        description="Convert an iv .npz file to a SPICE TABLE VCCS subckt.",
    )
    p.add_argument("--npz", required=True, help="Input iv .npz path.")
    p.add_argument("--out", required=True, help="Output .subckt path.")
    p.add_argument("--subckt", default="exp_model", help="Subcircuit name.")
    p.add_argument("--node-a", default="a", help="Node a name.")
    p.add_argument("--node-b", default="b", help="Node b name.")
    p.add_argument("--v-fmt", default="{: .6g}", help="Voltage format string.")
    p.add_argument("--i-fmt", default="{: .6g}", help="Current format string.")
    p.add_argument("--g-name", default="GEXP", help="VCCS element name.")
    return p


def main(argv=None) -> int:
    args = _build_arg_parser().parse_args(argv)
    write_spice_table_from_iv_npz(
        args.npz,
        args.out,
        subckt_name=args.subckt,
        node_a=args.node_a,
        node_b=args.node_b,
        v_fmt=args.v_fmt,
        i_fmt=args.i_fmt,
        g_name=args.g_name,
    )
    print(f"Wrote SPICE table to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
