from __future__ import annotations

import numpy as np
from ..params import Params


def get_minmax_range(params: Params) -> np.ndarray:
    """
    Return bounds for each state in the packed state vector as (N_states, 2).
    NaN means unbounded. b_* states are clipped to [0, 1].
    """
    n_E = params.n_E
    n_I = params.n_I
    n_a_E = params.n_a_E
    n_a_I = params.n_a_I
    n_b_E = params.n_b_E
    n_b_I = params.n_b_I
    n = params.n

    num_a_E = n_E * n_a_E
    num_a_I = n_I * n_a_I
    num_b_E = n_E * n_b_E
    num_b_I = n_I * n_b_I
    num_u_d = n

    N_states = num_a_E + num_a_I + num_b_E + num_b_I + num_u_d
    if N_states == 0:
        return np.empty((0, 2))

    bounds = np.full((N_states, 2), np.nan, dtype=float)
    idx = 0
    # a_E: unbounded
    idx += num_a_E
    # a_I: unbounded
    idx += num_a_I
    # b_E: [0,1]
    if num_b_E > 0:
        bounds[idx : idx + num_b_E, :] = np.array([0.0, 1.0])
    idx += num_b_E
    # b_I: [0,1]
    if num_b_I > 0:
        bounds[idx : idx + num_b_I, :] = np.array([0.0, 1.0])
    # remaining u_d: unbounded
    return bounds

