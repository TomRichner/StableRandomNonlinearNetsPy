from __future__ import annotations

from typing import Literal, Callable, Tuple

import numpy as np
from scipy.integrate import solve_ivp

from .params import Params
from .dynamics import make_rhs


def solve(
    T: Tuple[float, float],
    t_eval: np.ndarray,
    X0: np.ndarray,
    t_ex: np.ndarray,
    u_ex: np.ndarray,
    params: Params,
    method: Literal["RK45", "BDF"] = "BDF",
    rtol: float = 1e-7,
    atol: float | np.ndarray = 1e-8,
    max_step: float | None = None,
):
    """
    Thin wrapper over scipy.integrate.solve_ivp.

    Returns (t_out, X_out) where X_out is (nt, N_states) to match MATLAB row-major convention.
    """
    rhs = make_rhs(t_ex=t_ex, u_ex=u_ex, params=params)

    sol = solve_ivp(
        rhs,
        t_span=T,
        y0=np.asarray(X0).reshape(-1),
        t_eval=np.asarray(t_eval).reshape(-1),
        method=method,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
        vectorized=False,
    )
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")

    # sol.y shape: (N_states, nt); transpose to (nt, N_states)
    X_out = sol.y.T.copy()
    return sol.t.copy(), X_out

