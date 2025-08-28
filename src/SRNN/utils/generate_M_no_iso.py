from __future__ import annotations

import numpy as np


def generate_M_no_iso(n: int, w: dict, sparsity: float, EI: float):
    """
    Generate a sparse, strongly connected weight matrix with Dale's law and class-specific scaling.

    Parameters
    - n: number of neurons
    - w: dict with keys 'EE','EI','IE','II','selfE','selfI'
    - sparsity: fraction of off-diagonal entries set to zero (0..1)
    - EI: fraction of excitatory neurons (0..1)

    Returns
    - A: (n,n) weight matrix
    - EI_vec: (n,) vector with +1 for excitatory, -1 for inhibitory
    """

    rng = np.random.default_rng()

    EI_vec = -np.ones(n, dtype=int)
    EI_vec[: int(round(EI * n))] = 1

    # off-diagonal capacity
    E0 = n * (n - 1)
    E_keep = int(round((1 - sparsity) * E0))
    if E_keep < n:
        raise ValueError(
            "Requested sparsity too high; cannot ensure strong connectivity"
        )

    mask = np.zeros((n, n), dtype=bool)

    # Hamiltonian cycle for strong connectivity
    perm = rng.permutation(n)
    nxt = np.roll(perm, -1)
    mask[perm, nxt] = True

    # Add random edges to reach target density
    E_add = E_keep - mask.sum()
    if E_add > 0:
        avail = np.logical_not(mask) & (~np.eye(n, dtype=bool))
        idxs = np.vstack(np.nonzero(avail)).T
        pick_idx = rng.choice(idxs.shape[0], size=E_add, replace=False)
        sel = idxs[pick_idx]
        mask[sel[:, 0], sel[:, 1]] = True

    # Draw weights and impose structure
    A = rng.standard_normal((n, n))
    A[~mask] = 0.0
    np.fill_diagonal(A, 0.0)

    # Dale's law (columns are presynaptic sources)
    A[:, EI_vec == 1] = np.abs(A[:, EI_vec == 1])
    A[:, EI_vec == -1] = -np.abs(A[:, EI_vec == -1])

    # Class-specific scaling
    A[np.ix_(EI_vec == -1, EI_vec == 1)] *= w.get("EI", 1.0)
    A[np.ix_(EI_vec == 1, EI_vec == -1)] *= w.get("IE", 1.0)
    A[np.ix_(EI_vec == 1, EI_vec == 1)] *= w.get("EE", 1.0)
    A[np.ix_(EI_vec == -1, EI_vec == -1)] *= w.get("II", 1.0)

    # Self connections
    A[np.diag_indices(n)] = w.get("selfI", 0.0)
    diag_idx_E = np.where(EI_vec == 1)[0]
    A[diag_idx_E, diag_idx_E] = w.get("selfE", 0.0)

    return A, EI_vec

