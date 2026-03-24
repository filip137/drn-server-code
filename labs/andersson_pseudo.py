# Minimal Anderson Acceleration (AA) for a fixed-point iteration x <- G(x)
# Target use: DRN full sweep map G = (red exact solve) then (black exact solve)
# No safeguarding (no energy/residual acceptance), just AA + optional damping.

from collections import deque

def anderson_solve(x0, G, max_iters=200, m_max=8, omega=1.0, tol=1e-4):
    """
    x0: initial state vector (flattened voltages of all free nodes), shape (n,)
    G : function implementing ONE PLAIN iteration: x_plain_next = G(x)
        (e.g., full red+black sweep with exact block solves)
    m_max: Anderson history length
    omega: damping / relaxation on the Anderson candidate (omega=1 => no damping)
    tol: stop when ||x_{k+1} - x_k||_inf / max(||x_{k+1}||_inf, V_floor) < tol
    """

    # Store last m columns of Δx and Δf (each is length n)
    dX = deque(maxlen=m_max)   # Δx_k = x_{k+1} - x_k
    dF = deque(maxlen=m_max)   # Δf_k = f_{k+1} - f_k, where f_k = G(x_k) - x_k

    x = x0
    f_prev = None
    x_prev = None

    V_floor = 5e-3  # just for stopping; tweak/remove if you want

    for k in range(max_iters):
        # Plain fixed-point step
        x_plain = G(x)         # shape (n,)
        f = x_plain - x        # fixed-point residual/update direction, shape (n,)

        # Update history using previous (x_prev, f_prev)
        if x_prev is not None:
            dx = x - x_prev          # Δx_{k-1}
            df = f - f_prev          # Δf_{k-1}
            dX.append(dx)
            dF.append(df)

        # Anderson candidate (difference form)
        x_candidate = x_plain  # fallback if not enough history
        m = len(dF)
        if m >= 1:
            # Build small m×m system: (ΔF^T ΔF) γ = ΔF^T f
            # Implemented via dot products; no explicit n×m matrix needed.
            A = [[dot(dF[i], dF[j]) for j in range(m)] for i in range(m)]
            b = [dot(dF[i], f) for i in range(m)]

            # Solve A γ = b  (use any small linear solver; add ridge if needed)
            gamma = solve_small_linear_system(A, b)  # returns list/array length m

            # x_AA = x + f - Σ_i gamma_i * Δx_i
            corr = sum(gamma[i] * dX[i] for i in range(m))
            x_aa = x + f - corr
            x_candidate = x_aa

        # Optional damping (same pattern as your SOR)
        x_new = x + omega * (x_candidate - x)

        # Stopping criterion (update-based, cheap)
        dx_new = x_new - x
        if max_abs(dx_new) / max(max_abs(x_new), V_floor) < tol:
            return x_new

        # Shift state
        x_prev = x
        f_prev = f
        x = x_new

    return x


# --- helpers (pseudocode) ---
def dot(a, b): ...
def max_abs(v): ...
def solve_small_linear_system(A, b): ...