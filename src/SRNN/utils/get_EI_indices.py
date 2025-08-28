import numpy as np


def get_EI_indices(EI_vec: np.ndarray):
    EI_vec = np.asarray(EI_vec).reshape(-1)
    E_indices = np.where(EI_vec == 1)[0]
    I_indices = np.where(EI_vec == -1)[0]
    n_E = E_indices.size
    n_I = I_indices.size
    return E_indices, I_indices, n_E, n_I

