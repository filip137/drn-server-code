import os
import time
from pathlib import Path

import numpy as np
import torch
from model.minimizer.minimizer import LayerUpdater, Minimizer
from model.resistive.minimizer import lambertw
from model.resistive.minimizer import (
    QuadraticUpdater,
    AdaptiveQuadraticUpdater,
    QuadraticDoubleDiodeUpdaterOffset,
    ExponentialDoubleDiodeUpdater,
    ExponentialSingleDiodeUpdater,
    HardSigmoidUpdater,
)
from model.resistive.layer import NonlinearResistiveLayer

DEFAULT_IV_CURVE_PATH = Path("/home/filip/server_code/labs/i_v_npz/experimental_iv_curve.npz")


def _load_iv_data(path: Path) -> torch.Tensor:
    data = np.load(path)
    if "iv" in data:
        iv = data["iv"]
    elif "i" in data and "v" in data:
        iv = np.stack([data["i"], data["v"]], axis=0)
    else:
        raise ValueError(f"{path} must contain 'iv' or both 'i' and 'v' arrays.")
    return torch.as_tensor(iv)


class CustomQuadraticUpdater(QuadraticUpdater):
    """Override pre_activate if you want custom quadratic updates."""


class CustomAdaptiveQuadraticUpdater(AdaptiveQuadraticUpdater):
    """Override pre_activate if you want custom LPW updates."""


class CustomQuadraticDoubleDiodeUpdaterOffset(QuadraticDoubleDiodeUpdaterOffset):
    """Override pre_activate if you want custom quadratic double-diode updates."""


class CustomExponentialDoubleDiodeUpdater(ExponentialDoubleDiodeUpdater):
    """Override pre_activate if you want custom exponential double-diode updates."""


class CustomExponentialSingleDiodeUpdater(ExponentialSingleDiodeUpdater):
    """Override pre_activate if you want custom exponential single-diode updates."""

    def lambertw_large(self, z, terms=4):
        L1 = torch.log(z)
        L2 = torch.log(L1)
        w = L1 - L2
        if terms >= 2:
            w = w + L2 / L1
        if terms >= 3:
            w = w + (L2 * (-2.0 + L2)) / (2.0 * L1**2)
        if terms >= 4:
            w = w + (L2 * (6.0 - 9.0*L2 + 2.0*L2**2)) / (6.0 * L1**3)
        return w

    def _lambertw0(self, z):
        if hasattr(torch, "special") and hasattr(torch.special, "lambertw"):
            return torch.special.lambertw(z).real
        return lambertw(z).real  # assumes you have lambertw imported elsewhere

    # ---------- Forward diode ----------
    # 2 a v + b + I_s (exp((v - v_on)/vT) - 1) = 0
    def lambert_single_forward(
        self, a, b, I_s, v_on, vT,
        z_thresh=1e10, polish=False,
        abs_tol=1e-6, rel_tol=1e-6,
        exp_clip=80.0, a_min=1e-30, max_inner_iter=8
    ):
        out_dtype = a.dtype
        device = a.device

        a64 = torch.clamp(a.to(torch.float64), min=a_min)
        b64 = b.to(torch.float64)

        Is64  = torch.as_tensor(I_s, dtype=torch.float64, device=device)
        vT64  = torch.as_tensor(vT,  dtype=torch.float64, device=device)
        von64 = torch.as_tensor(v_on, dtype=torch.float64, device=device)
        self._debug_tensor("a64", a64)
        self._debug_tensor("b64", b64)
        self._debug_tensor("Is64", Is64)
        self._debug_tensor("vT64", vT64)
        self._debug_tensor("von64", von64)

        # Ashift = (b - I_s)/(2a)
        Ashift = (b64 - Is64) / (2.0 * a64)
        self._debug_tensor("Ashift_fwd", Ashift)

        # z = (I_s/(2 a vT)) * exp(-(v_on + Ashift)/vT)
        z = (Is64 / (2.0 * a64 * vT64)) * torch.exp(-(Ashift + von64) / vT64)
        self._debug_tensor("z_fwd", z)

        use_asym = z > z_thresh
        W0 = torch.empty_like(z, dtype=torch.float64, device=device)

        if (~use_asym).any():
            z_small = torch.clamp(z[~use_asym], max=float(z_thresh))
            W0[~use_asym] = self._lambertw0(z_small).to(torch.float64)

        if use_asym.any():
            z_large = torch.clamp(z[use_asym], min=1.0)
            W0[use_asym] = self.lambertw_large(z_large).to(torch.float64)
        self._debug_tensor("W0_fwd", W0)

        # v* = -Ashift - vT * W0
        x = -Ashift - vT64 * W0
        self._debug_tensor("x_fwd", x)

        if polish:
            for _ in range(max_inner_iter):
                arg = (x - von64) / vT64
                arg = torch.clamp(arg, min=-exp_clip, max=exp_clip)
                e = torch.exp(arg)

                f  = 2.0*a64*x + b64 + Is64*(e - 1.0)
                df = 2.0*a64   + (Is64/vT64)*e
                if not torch.isfinite(f).all() or not torch.isfinite(df).all():
                    self._debug_tensor("f_fwd", f)
                    self._debug_tensor("df_fwd", df)
                    break

                step = f / df
                x = x - step

                max_abs_f = torch.max(torch.abs(f))
                scale = torch.max(2.0*torch.abs(a64)*torch.abs(x) + torch.abs(b64) + torch.abs(Is64) + 1.0)
                if (max_abs_f < abs_tol) and (max_abs_f/(scale + 1e-30) < rel_tol):
                    break

        return x.to(out_dtype)

    # ---------- Reversed diode (your corrected equation) ----------
    # 2 a v + b - I_s (exp((-(v + v_on))/vT) - 1) = 0
    # <=> 2 a v + (b + I_s) - I_s exp(-(v + v_on)/vT) = 0
    def lambert_single_reverse(
        self, a, b, I_s, v_on, vT,
        z_thresh=1e10, polish=False,
        abs_tol=1e-10, rel_tol=1e-10,
        exp_clip=80.0, a_min=1e-30, max_inner_iter=8
    ):
        out_dtype = a.dtype
        device = a.device

        a64 = torch.clamp(a.to(torch.float64), min=a_min)
        b64 = b.to(torch.float64)

        Is64  = torch.as_tensor(I_s, dtype=torch.float64, device=device)
        vT64  = torch.as_tensor(vT,  dtype=torch.float64, device=device)
        von64 = torch.as_tensor(v_on, dtype=torch.float64, device=device)
        self._debug_tensor("a64", a64)
        self._debug_tensor("b64", b64)
        self._debug_tensor("Is64", Is64)
        self._debug_tensor("vT64", vT64)
        self._debug_tensor("von64", von64)

        # Ashift = (b + I_s)/(2a)
        Ashift = (b64 + Is64) / (2.0 * a64)
        self._debug_tensor("Ashift_rev", Ashift)

        # z = (I_s/(2 a vT)) * exp( -v_on/vT + Ashift/vT )
        z = (Is64 / (2.0 * a64 * vT64)) * torch.exp((-von64 + Ashift) / vT64)
        self._debug_tensor("z_rev", z)

        use_asym = z > z_thresh
        W0 = torch.empty_like(z, dtype=torch.float64, device=device)

        if (~use_asym).any():
            z_small = torch.clamp(z[~use_asym], max=float(z_thresh))
            W0[~use_asym] = self._lambertw0(z_small).to(torch.float64)

        if use_asym.any():
            z_large = torch.clamp(z[use_asym], min=1.0)
            W0[use_asym] = self.lambertw_large(z_large).to(torch.float64)
        self._debug_tensor("W0_rev", W0)

        # v* = -Ashift + vT * W0
        x = -Ashift + vT64 * W0
        self._debug_tensor("x_rev", x)

        if polish:
            for _ in range(max_inner_iter):
                arg = -(x + von64) / vT64
                arg = torch.clamp(arg, min=-exp_clip, max=exp_clip)
                e = torch.exp(arg)

                # f = 2 a v + b + I_s - I_s e^{-(v+v_on)/vT}
                f  = 2.0*a64*x + b64 + Is64 - Is64*e
                # df = 2 a + (I_s/vT) e^{-(v+v_on)/vT}
                df = 2.0*a64   + (Is64/vT64)*e
                if not torch.isfinite(f).all() or not torch.isfinite(df).all():
                    self._debug_tensor("f_rev", f)
                    self._debug_tensor("df_rev", df)
                    break

                step = f / df
                x = x - step

                max_abs_f = torch.max(torch.abs(f))
                scale = torch.max(2.0*torch.abs(a64)*torch.abs(x) + torch.abs(b64) + torch.abs(Is64) + 1.0)
                if (max_abs_f < abs_tol) and (max_abs_f/(scale + 1e-30) < rel_tol):
                    break

        return x.to(out_dtype)


class CustomHardSigmoidUpdater(HardSigmoidUpdater):
    """Override pre_activate if you want custom hard-sigmoid updates."""


class ExperimentalIVcurveUpdater(LayerUpdater):
    """LayerUpdater for experimental I-V curve updates."""

    def __init__(self, layer, fn, iv_data, *, damping: float = 0.5, clamp: bool = True):
        super().__init__(layer, fn)
        self._iv_data = iv_data
        self._damping = float(damping)
        self._clamp = bool(clamp)
        self._print_newton_iters = os.environ.get("LABS_PRINT_NEWTON_ITERS") == "1"
        self._print_every = int(os.environ.get("LABS_PRINT_NEWTON_ITERS_EVERY", "1"))
        self._print_counter = 0
        self._print_newton_timing = os.environ.get("LABS_PRINT_NEWTON_TIMING") == "1"
        self._timing_every = int(os.environ.get("LABS_PRINT_NEWTON_TIMING_EVERY", "1"))
        self._timing_counter = 0
        self._timing_total = 0.0
        self._timing_calls = 0
        # Standard quadratic coefficients (matching QuadraticUpdater interface)
        self._a = fn.a_coef_fn(layer)
        self._b = fn.b_coef_fn(layer)

    def pre_activate(self):
        b = self._b()
        a = self._a()
        if not isinstance(self._layer, NonlinearResistiveLayer):
            return -b / (2.0 * a)
        i_data = self._iv_data[0].to(device=a.device, dtype=a.dtype)
        v_data = self._iv_data[1].to(device=a.device, dtype=a.dtype)
        slope = (i_data[1:] - i_data[:-1]) / (v_data[1:] - v_data[:-1])
        v_min = v_data[0]
        v_max = v_data[-1]
        eps = torch.finfo(a.dtype).eps

        def newton_step(v):
            idx = (torch.bucketize(v, v_data) - 1).clamp(0, len(v_data) - 2)
            deriv = slope[idx]
            i = i_data[idx] + (v - v_data[idx]) * deriv
            fun = 2 * a * v + b + i
            fun_prime = 2 * a + deriv + eps
            v_new = v - self._damping * (fun / fun_prime)
            if self._clamp:
                v_new = torch.clamp(v_new, min=v_min, max=v_max)
            return v_new

        max_number_steps = 100
        timing_start = time.perf_counter() if self._print_newton_timing else None
        v_old = -b / (2 * a)
        tol = 1e-5
        active = torch.ones_like(v_old, dtype=torch.bool)
        iters = torch.zeros_like(v_old, dtype=torch.int32)
        for step in range(max_number_steps):
            v = newton_step(v_old)
            delta = torch.abs(v - v_old)
            converged = delta < tol
            newly_converged = active & converged
            if newly_converged.any().item():
                iters[newly_converged] = step + 1
                active = active & ~converged
            v_old = v
            if not active.any().item():
                break
        if active.any().item():
            iters[active] = max_number_steps
        if self._print_newton_iters:
            self._print_counter += 1
            if self._print_counter % max(self._print_every, 1) == 0:
                mean_iters = float(iters.float().mean().item())
                max_iters = int(iters.max().item())
                print(
                    f"ExperimentalIVcurveUpdater: avg newton iters={mean_iters:.2f} "
                    f"(max={max_iters})"
                )
        if self._print_newton_timing:
            elapsed = time.perf_counter() - timing_start
            self._timing_total += elapsed
            self._timing_calls += 1
            self._timing_counter += 1
            if self._timing_counter % max(self._timing_every, 1) == 0:
                avg = self._timing_total / max(self._timing_calls, 1)
                print(
                    f"ExperimentalIVcurveUpdater: avg newton time={avg:.6f}s "
                    f"(last={elapsed:.6f}s)"
                )
        return v_old

class CustomQuadraticMinimizer(Minimizer):
    """Quadratic minimizer that uses the custom updater classes above."""

    def __init__(
        self,
        fn,
        free_layers,
        num_iterations,
        mode,
        non_linearity,
        quadratic_diode_param,
        exponential_diode_param,
        voltage_amp,
        current_amp,
        hard_sigmoid_param=None,
        iv_data=None,
        iv_data_path=None,
    ):
        quadratic_params = dict(quadratic_diode_param)
        exponential_params = dict(exponential_diode_param)
        hard_sigmoid_params = dict(hard_sigmoid_param or {})
        if not hard_sigmoid_params:
            if "diode_conductance" in quadratic_params:
                hard_sigmoid_params["g_on"] = quadratic_params["diode_conductance"]
                hard_sigmoid_params["g_off"] = quadratic_params["diode_conductance"]
        if "g_on" not in hard_sigmoid_params and "diode_conductance" in quadratic_params:
            hard_sigmoid_params["g_on"] = quadratic_params["diode_conductance"]
        if "g_off" not in hard_sigmoid_params and "g_on" in hard_sigmoid_params:
            hard_sigmoid_params["g_off"] = hard_sigmoid_params["g_on"]
        if "v_min" not in hard_sigmoid_params and "v_min" in quadratic_params:
            hard_sigmoid_params["v_min"] = quadratic_params["v_min"]
        if "v_max" not in hard_sigmoid_params and "v_max" in quadratic_params:
            hard_sigmoid_params["v_max"] = quadratic_params["v_max"]

        if non_linearity == "perfect_diode":
            updaters = [CustomQuadraticUpdater(layer, fn) for layer in free_layers]
        elif non_linearity == "lpw_diode":
            updaters = [CustomAdaptiveQuadraticUpdater(layer, fn, quadratic_params) for layer in free_layers]
        elif non_linearity == "double_diode_quadratic":
            updaters = [CustomQuadraticDoubleDiodeUpdaterOffset(layer, fn, quadratic_params) for layer in free_layers]
        elif non_linearity == "double_diode_exponential":
            updaters = [CustomExponentialDoubleDiodeUpdater(layer, fn, exponential_params) for layer in free_layers]
        elif non_linearity == "single_diode_exponential":
            updaters = [CustomExponentialSingleDiodeUpdater(layer, fn, exponential_params) for layer in free_layers]
        elif non_linearity == "hard_sigmoid":
            updaters = [CustomHardSigmoidUpdater(layer, fn, hard_sigmoid_params) for layer in free_layers]
        elif non_linearity == "experimental":
            if iv_data is None:
                env_path = os.environ.get("LABS_IV_CURVE_PATH")
                path = Path(env_path) if env_path else (Path(iv_data_path) if iv_data_path else DEFAULT_IV_CURVE_PATH)
                if not path.exists():
                    raise FileNotFoundError(
                        f"Experimental IV curve not found at {path}. "
                        "Generate it under /home/filip/server_code/labs/i_v_npz "
                        "or pass iv_data/iv_data_path."
                    )
                iv_data = _load_iv_data(path)
            updaters = [ExperimentalIVcurveUpdater(layer, fn, iv_data) for layer in free_layers]
        elif non_linearity == "linear":
            updaters = [CustomQuadraticUpdater(layer, fn) for layer in free_layers]
        else:
            raise ValueError(
                "non_linearity must be 'perfect_diode', 'double_diode_quadratic', "
                "'lpw_diode', 'double_diode_exponential', 'hard_sigmoid', or 'linear'; got {}".format(
                    non_linearity
                )
            )

        super().__init__(fn, updaters, num_iterations, mode, voltage_amp, current_amp)

        for updater in self._updaters:
            updater.voltage_amp = self.voltage_amp
            updater.current_amp = self.current_amp

        self._non_linearity = non_linearity
        self._quadratic_params = quadratic_params
        self._exponential_params = exponential_params
        self._hard_sigmoid_params = hard_sigmoid_params
