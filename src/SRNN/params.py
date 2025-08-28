from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


Activation = Callable[[np.ndarray], np.ndarray]


@dataclass
class Params:
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

