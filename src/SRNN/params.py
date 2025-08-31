from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


Activation = Callable[[np.ndarray], np.ndarray]


@dataclass
class Params:
    """
    Dataclass holding all parameters for an SRNN model instance.

    Attributes:
    ----------
    n_E, n_I: int
        Number of excitatory and inhibitory neurons.
    E_indices, I_indices: np.ndarray
        Arrays of integer indices for E and I neurons.
    n_a_E, n_a_I: int
        Number of spike-frequency adaptation (SFA) timescales for E/I pops.
    n_b_E, n_b_I: int
        Number of short-term depression (STD) timescales for E/I pops.
    tau_a_E, tau_a_I: np.ndarray | None
        Time constants for SFA variables.
    tau_b_E, tau_b_I: np.ndarray | None
        Time constants for STD variables.
    tau_d: float
        Time constant for synaptic drive dynamics `u_d`.
    n: int
        Total number of neurons (n_E + n_I).
    M: np.ndarray
        (n, n) connectivity matrix.
    EI_vec: np.ndarray
        (n,) vector with +1 for excitatory, -1 for inhibitory neurons.
    c_SFA: np.ndarray
        (n,) vector of coupling strengths for SFA. A neuron's SFA is the
        sum of its `a` variables scaled by its `c_SFA` value.
    F_STD: np.ndarray
        (n,) vector of factors for STD.
    tau_STD: float
        Time constant for STD recovery.
    activation_function: Optional[Activation]
        A callable activation function (e.g., ReLU). If None, defaults to identity.
    """
    # Sizes
    n_E: int
    n_I: int
    E_indices: np.ndarray
    I_indices: np.ndarray
    n_a_E: int
    n_a_I: int
    n_b_E: int
    n_b_I: int

    # Time constants
    tau_a_E: np.ndarray | None
    tau_a_I: np.ndarray | None
    tau_b_E: np.ndarray | None
    tau_b_I: np.ndarray | None
    tau_d: float

    # Network
    n: int
    M: np.ndarray
    EI_vec: np.ndarray

    # Adaptation/depression coefficients
    c_SFA: np.ndarray
    F_STD: np.ndarray
    tau_STD: float

    # Optional activation
    activation_function: Optional[Activation] = None


def package_params(
    n_E: int,
    n_I: int,
    E_indices: np.ndarray,
    I_indices: np.ndarray,
    n_a_E: int,
    n_a_I: int,
    n_b_E: int,
    n_b_I: int,
    tau_a_E: np.ndarray | list[float] | None,
    tau_a_I: np.ndarray | list[float] | None,
    tau_b_E: np.ndarray | list[float] | None,
    tau_b_I: np.ndarray | list[float] | None,
    tau_d: float,
    n: int,
    M: np.ndarray,
    c_SFA: np.ndarray | list[float],
    F_STD: np.ndarray | list[float],
    tau_STD: float,
    EI_vec: np.ndarray | list[int],
    activation_function: Optional[Activation] = None,
) -> Params:
    def _arr(x, dtype=float):
        if x is None:
            return None
        a = np.asarray(x, dtype=dtype)
        return a.reshape(-1) if a.ndim > 1 else a

    return Params(
        n_E=int(n_E),
        n_I=int(n_I),
        E_indices=np.asarray(E_indices, dtype=int).reshape(-1),
        I_indices=np.asarray(I_indices, dtype=int).reshape(-1),
        n_a_E=int(n_a_E),
        n_a_I=int(n_a_I),
        n_b_E=int(n_b_E),
        n_b_I=int(n_b_I),
        tau_a_E=_arr(tau_a_E) if (n_a_E > 0) else None,
        tau_a_I=_arr(tau_a_I) if (n_a_I > 0) else None,
        tau_b_E=_arr(tau_b_E) if (n_b_E > 0) else None,
        tau_b_I=_arr(tau_b_I) if (n_b_I > 0) else None,
        tau_d=float(tau_d),
        n=int(n),
        M=np.asarray(M, dtype=float),
        EI_vec=_arr(EI_vec, dtype=int),
        c_SFA=_arr(c_SFA),
        F_STD=_arr(F_STD),
        tau_STD=float(tau_STD),
        activation_function=activation_function,
    )

