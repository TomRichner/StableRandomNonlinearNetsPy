from __future__ import annotations

import numpy as np

from .activation import get_activation
from .params import Params


def compute_dependent(
    a_E_ts: np.ndarray | None,
    a_I_ts: np.ndarray | None,
    b_E_ts: np.ndarray | None,
    b_I_ts: np.ndarray | None,
    u_d_ts: np.ndarray,
    params: Params,
):
    """
    Vectorized computation of firing rates r and outputs p over time.

    Inputs use shapes like MATLAB helper:
    - a_E_ts: (n_E, n_a_E, nt) or None
    - a_I_ts: (n_I, n_a_I, nt) or None
    - b_E_ts: (n_E, n_b_E, nt) or None
    - b_I_ts: (n_I, n_b_I, nt) or None
    - u_d_ts: (n, nt)
    Returns (r, p): both (n, nt)
    """
    if u_d_ts is None or u_d_ts.size == 0:
        return np.empty((0, 0)), np.empty((0, 0))

    n = params.n
    nt = u_d_ts.shape[1]

    E_idx = params.E_indices
    I_idx = params.I_indices
    c_SFA_full = params.c_SFA

    u_eff = u_d_ts.copy()

    if params.n_E > 0 and params.n_a_E > 0 and a_E_ts is not None:
        # sum over timescales axis
        sum_a_E = np.sum(a_E_ts, axis=1)  # (n_E, nt)
        u_eff[E_idx, :] -= (c_SFA_full[E_idx][:, None] * sum_a_E)

    if params.n_I > 0 and params.n_a_I > 0 and a_I_ts is not None:
        sum_a_I = np.sum(a_I_ts, axis=1)  # (n_I, nt)
        u_eff[I_idx, :] -= (c_SFA_full[I_idx][:, None] * sum_a_I)

    act = get_activation(params.activation_function)
    r = act(u_eff)
    p = r.copy()

    if params.n_E > 0 and params.n_b_E > 0 and b_E_ts is not None:
        prod_b_E = np.prod(b_E_ts, axis=1)  # (n_E, nt)
        p[E_idx, :] *= prod_b_E

    if params.n_I > 0 and params.n_b_I > 0 and b_I_ts is not None:
        prod_b_I = np.prod(b_I_ts, axis=1)  # (n_I, nt)
        p[I_idx, :] *= prod_b_I

    return r, p

