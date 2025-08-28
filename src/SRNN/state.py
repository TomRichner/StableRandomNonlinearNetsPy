from __future__ import annotations

import numpy as np
from .params import Params


def unpack_state(X: np.ndarray, params: Params):
    X = np.asarray(X).reshape(-1)

    n_E = params.n_E
    n_I = params.n_I
    n_a_E = params.n_a_E
    n_a_I = params.n_a_I
    n_b_E = params.n_b_E
    n_b_I = params.n_b_I
    n = params.n

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

    return a_E, a_I, b_E, b_I, u_d


def unpack_trajectory(X_traj: np.ndarray, params: Params):
    # X_traj: (nt, N_states)
    X_traj = np.asarray(X_traj)
    assert X_traj.ndim == 2
    nt, _ = X_traj.shape

    n_E = params.n_E
    n_I = params.n_I
    n_a_E = params.n_a_E
    n_a_I = params.n_a_I
    n_b_E = params.n_b_E
    n_b_I = params.n_b_I
    n = params.n

    idx = 0

    len_a_E = n_E * n_a_E
    if len_a_E > 0:
        a_E = (
            X_traj[:, idx : idx + len_a_E].T.reshape(n_E, n_a_E, nt)
        )
    else:
        a_E = None
    idx += len_a_E

    len_a_I = n_I * n_a_I
    if len_a_I > 0:
        a_I = (
            X_traj[:, idx : idx + len_a_I].T.reshape(n_I, n_a_I, nt)
        )
    else:
        a_I = None
    idx += len_a_I

    len_b_E = n_E * n_b_E
    if len_b_E > 0:
        b_E = (
            X_traj[:, idx : idx + len_b_E].T.reshape(n_E, n_b_E, nt)
        )
    else:
        b_E = None
    idx += len_b_E

    len_b_I = n_I * n_b_I
    if len_b_I > 0:
        b_I = (
            X_traj[:, idx : idx + len_b_I].T.reshape(n_I, n_b_I, nt)
        )
    else:
        b_I = None
    idx += len_b_I

    u_d = X_traj[:, idx : idx + n].T  # (n, nt)

    return a_E, a_I, b_E, b_I, u_d

