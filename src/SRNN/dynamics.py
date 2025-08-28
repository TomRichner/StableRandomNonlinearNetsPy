from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d

from .activation import get_activation
from .params import Params


def make_rhs(
    t_ex: np.ndarray,
    u_ex: np.ndarray,
    params: Params,
):
    """
    Create RHS function f(t, X) for solve_ivp from MATLAB SRNN_NL.m.

    t_ex: (nt,) time grid for external input
    u_ex: (n, nt) external input per neuron over time
    params: model parameters
    """
    t_ex = np.asarray(t_ex).reshape(-1)
    u_ex = np.asarray(u_ex)
    if u_ex.shape[0] != params.n:
        raise ValueError("u_ex must be shape (n, nt)")

    # Interpolant returns shape (n,) for scalar t
    u_interp = interp1d(
        t_ex,
        u_ex.T,  # (nt, n)
        kind="linear",
        axis=0,
        bounds_error=False,
        fill_value=np.nan,
        assume_sorted=True,
    )

    act = get_activation(params.activation_function)

    n = params.n
    n_E = params.n_E
    n_I = params.n_I
    n_a_E = params.n_a_E
    n_a_I = params.n_a_I
    n_b_E = params.n_b_E
    n_b_I = params.n_b_I

    E_idx = params.E_indices
    I_idx = params.I_indices

    M = params.M
    tau_d = params.tau_d
    c_SFA = params.c_SFA
    F_STD = params.F_STD
    tau_STD = params.tau_STD

    tau_a_E = params.tau_a_E
    tau_a_I = params.tau_a_I
    tau_b_E = params.tau_b_E
    tau_b_I = params.tau_b_I

    def rhs(t: float, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X).reshape(-1)

        u = u_interp(t).astype(float, copy=False)  # (n,)

        # Inline unpacking
        idx = 0
        len_a_E = n_E * n_a_E
        if len_a_E > 0:
            a_E = X[idx : idx + len_a_E].reshape(n_E, n_a_E)
        else:
            a_E = None
        idx += len_a_E

        len_a_I = n_I * n_a_I
        if len_a_I > 0:
            a_I = X[idx : idx + len_a_I].reshape(n_I, n_a_I)
        else:
            a_I = None
        idx += len_a_I

        len_b_E = n_E * n_b_E
        if len_b_E > 0:
            b_E = X[idx : idx + len_b_E].reshape(n_E, n_b_E)
        else:
            b_E = None
        idx += len_b_E

        len_b_I = n_I * n_b_I
        if len_b_I > 0:
            b_I = X[idx : idx + len_b_I].reshape(n_I, n_b_I)
        else:
            b_I = None
        idx += len_b_I

        u_d = X[idx : idx + n]

        # Dependent variables
        u_eff = u_d.copy()
        if n_E > 0 and n_a_E > 0 and a_E is not None:
            u_eff[E_idx] -= c_SFA[E_idx] * np.sum(a_E, axis=1)
        if n_I > 0 and n_a_I > 0 and a_I is not None:
            u_eff[I_idx] -= c_SFA[I_idx] * np.sum(a_I, axis=1)

        r = act(u_eff)
        p = r.copy()
        if n_E > 0 and n_b_E > 0 and b_E is not None:
            p[E_idx] *= np.prod(b_E, axis=1)
        if n_I > 0 and n_b_I > 0 and b_I is not None:
            p[I_idx] *= np.prod(b_I, axis=1)

        # Derivatives
        d_a_E = []
        if n_E > 0 and n_a_E > 0 and a_E is not None:
            d_a_E_mat = (r[E_idx][:, None] - a_E) / tau_a_E
            # Zero where c_SFA is zero
            mask = c_SFA[E_idx] == 0
            if np.any(mask):
                d_a_E_mat[mask, :] = 0.0
            d_a_E = d_a_E_mat.reshape(-1)

        d_a_I = []
        if n_I > 0 and n_a_I > 0 and a_I is not None:
            d_a_I_mat = (r[I_idx][:, None] - a_I) / tau_a_I
            mask = c_SFA[I_idx] == 0
            if np.any(mask):
                d_a_I_mat[mask, :] = 0.0
            d_a_I = d_a_I_mat.reshape(-1)

        d_b_E = []
        if n_E > 0 and n_b_E > 0 and b_E is not None:
            d_b_E_mat = (1.0 - b_E) / tau_b_E - (F_STD[E_idx][:, None] * p[E_idx][:, None]) / tau_STD
            mask = F_STD[E_idx] == 0
            if np.any(mask):
                d_b_E_mat[mask, :] = 0.0
            d_b_E = d_b_E_mat.reshape(-1)

        d_b_I = []
        if n_I > 0 and n_b_I > 0 and b_I is not None:
            d_b_I_mat = (1.0 - b_I) / tau_b_I - (F_STD[I_idx][:, None] * p[I_idx][:, None]) / tau_STD
            mask = F_STD[I_idx] == 0
            if np.any(mask):
                d_b_I_mat[mask, :] = 0.0
            d_b_I = d_b_I_mat.reshape(-1)

        d_u_d = (-u_d + u + M @ p) / tau_d

        return np.concatenate([
            np.asarray(d_a_E, dtype=float).reshape(-1),
            np.asarray(d_a_I, dtype=float).reshape(-1),
            np.asarray(d_b_E, dtype=float).reshape(-1),
            np.asarray(d_b_I, dtype=float).reshape(-1),
            d_u_d.reshape(-1),
        ])

    return rhs

