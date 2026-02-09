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


def _build_arg_parser():
    import argparse

    p = argparse.ArgumentParser(
        prog="custom_i_v_curves.py",
        description="Generate and save experimental i-v curves.",
    )
    p.add_argument("--output", required=True, help="Output .npz path.")
    p.add_argument("--v-off", type=float, required=True, help="Diode offset voltage.")
    p.add_argument("--n-points", type=int, required=True, help="Number of samples.")
    p.add_argument("--v-min", type=float, required=True, help="Minimum voltage.")
    p.add_argument("--v-max", type=float, required=True, help="Maximum voltage.")
    p.add_argument("--I-s", type=float, required=True, dest="I_s", help="Diode saturation current.")
    p.add_argument("--V-t", type=float, required=True, dest="V_t", help="Thermal voltage.")
    return p


def main(argv=None) -> int:
    args = _build_arg_parser().parse_args(argv)
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
