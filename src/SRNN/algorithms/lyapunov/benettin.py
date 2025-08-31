from __future__ import annotations

import numpy as np
from typing import Tuple, Callable

from ...params import Params
from ...simulate import solve
from ...utils.get_minmax_range import get_minmax_range


def benettin_algorithm(
    X_traj: np.ndarray,
    t_traj: np.ndarray,
    dt: float,
    fs: float,
    d0: float,
    T: Tuple[float, float],
    lya_dt: float,
    params: Params,
    dyn_factory: Callable[[float, np.ndarray], np.ndarray] | None,
    t_ex: np.ndarray,
    u_ex: np.ndarray,
    method: str = "BDF",
    rtol: float = 1e-7,
    atol: float = 1e-8,
    max_step: float | None = None,
):
    """
    Compute largest Lyapunov exponent using Benettin's algorithm.

    X_traj: (nt, N_states) fiducial trajectory sampled on t_traj
    Returns: (LLE, local_lya, finite_lya, t_lya)
    """
    if not (isinstance(lya_dt, (float, int)) and lya_dt > 0):
        raise ValueError("lya_dt must be positive scalar")

    deci_lya = int(round(lya_dt * fs))
    if deci_lya < 1:
        raise ValueError("lya_dt * fs must be >= 1 sample")

    tau_lya = dt * deci_lya
    t_lya = t_traj[::deci_lya].copy()
    # Keep intervals fully inside T
    if t_lya.size and (t_lya[-1] + tau_lya > T[1]):
        t_lya = t_lya[:-1]
    nt_lya = t_lya.size

    local_lya = np.zeros(nt_lya)
    finite_lya = np.full(nt_lya, np.nan)

    n_state = X_traj.shape[1]
    rng = np.random.default_rng()
    pert = rng.standard_normal(n_state)
    pert = pert / np.linalg.norm(pert) * d0

    bounds = get_minmax_range(params)
    min_b = bounds[:, 0]
    max_b = bounds[:, 1]

    sum_log = 0.0

    for k in range(nt_lya):
        idx_start = k * deci_lya
        idx_end = idx_start + deci_lya

        X_start = X_traj[idx_start].copy()
        X_end_true = X_traj[idx_end].copy()

        X_pert = X_start + pert
        # apply bounds
        mask = ~np.isnan(min_b)
        X_pert[mask] = np.maximum(X_pert[mask], min_b[mask])
        mask = ~np.isnan(max_b)
        X_pert[mask] = np.minimum(X_pert[mask], max_b[mask])

        # integrate perturbed trajectory over [t_k, t_k + tau_lya] using detailed grid
        t_seg = np.arange(t_lya[k], t_lya[k] + tau_lya + 0.5 * dt, dt)
        _, X_seg = solve(
            (t_seg[0], t_seg[-1]),
            t_seg,
            X_pert,
            t_ex,
            u_ex,
            params,
            method=method,
            rtol=rtol,
            atol=atol,
            max_step=max_step if max_step is not None else dt,
        )
        X_pert_end = X_seg[-1]

        delta = X_pert_end - X_end_true
        d_k = np.linalg.norm(delta)
        local_lya[k] = np.log(d_k / d0) / tau_lya

        if not np.isfinite(local_lya[k]):
            # truncate results
            local_lya = local_lya[:k]
            finite_lya = finite_lya[:k]
            t_lya = t_lya[:k]
            break

        pert = delta / (d_k + 1e-15) * d0
        if t_lya[k] >= 0:
            sum_log += np.log(d_k / d0)
            finite_lya[k] = sum_log / max(t_lya[k] + tau_lya, np.finfo(float).eps)

    finite = finite_lya[~np.isnan(finite_lya)]
    LLE = finite[-1] if finite.size else 0.0
    return LLE, local_lya, finite_lya, t_lya

