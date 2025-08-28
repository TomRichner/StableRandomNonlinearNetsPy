from __future__ import annotations

import numpy as np
from scipy.signal import square

from SRNN.utils.generate_M_no_iso import generate_M_no_iso
from SRNN.utils.get_EI_indices import get_EI_indices
from SRNN.params import package_params
from SRNN.simulate import solve
from SRNN.state import unpack_trajectory
from SRNN.dependent import compute_dependent
from SRNN.algorithms.lyapunov.benettin import benettin_algorithm


def main():
    rng = np.random.default_rng(42)

    # Network
    n = 10
    mean_in_out_degree = 5
    density = mean_in_out_degree / (n - 1)
    sparsity = 1 - density
    EI = 0.7
    scale = 0.5 / 0.79782
    w = dict(EE=scale * 1, EI=scale * 1, IE=scale * 1, II=scale * 0.5, selfE=0.0, selfI=0.0)

    M, EI_vec = generate_M_no_iso(n, w, sparsity, EI)
    E_idx, I_idx, n_E, n_I = get_EI_indices(EI_vec)

    # Time
    fs = 1000.0
    dt = 1.0 / fs
    T = (-10.0, 10.0)
    nt = int(round((T[1] - T[0]) * fs)) + 1
    t = np.linspace(T[0], T[1], nt)

    # External input u_ex (n, nt)
    u_ex = np.zeros((n, nt))
    stim_b0 = 0.5
    amp = 0.5
    dur = 3
    n_dur = int(fs * dur)
    # indices offset
    off = int(-T[0] * fs)
    # square wave
    t_sin = t[:n_dur]
    f_sin = np.ones_like(t_sin) * 1.0
    # mimic MATLAB sign(sin) and cos sequence on neuron 1
    idx1 = off + int(fs * 6)
    u_ex[0, idx1 : idx1 + n_dur] = stim_b0 + amp * np.sign(np.sin(2 * np.pi * f_sin * t_sin))
    idx2 = off + int(fs * 1)
    u_ex[0, idx2 : idx2 + n_dur] = stim_b0 + amp * (-np.cos(2 * np.pi * f_sin * t_sin))

    # DC ramp then constant
    DC = 0.1
    ramp_duration = 5.0
    ramp_mask = t <= (T[0] + ramp_duration)
    ramp_profile = np.linspace(0.0, DC, ramp_mask.sum())
    u_dc = np.full(nt, DC)
    u_dc[ramp_mask] = ramp_profile
    u_ex = u_ex + u_dc

    # Params
    tau_STD = 0.5
    n_a_E = 3
    n_a_I = 0
    n_b_E = 1
    n_b_I = 0

    tau_a_E = np.logspace(np.log10(0.3), np.log10(15.0), n_a_E) if n_a_E > 0 else None
    tau_a_I = None
    tau_b_E = (4 * tau_STD) if n_b_E == 1 else np.logspace(np.log10(0.6), np.log10(9.0), n_b_E)
    tau_b_I = None
    tau_d = 0.025

    c_SFA = (np.equal(EI_vec, 1).astype(float) * (1.0 / n_a_E)) if n_a_E > 0 else np.zeros(n)
    F_STD = (np.equal(EI_vec, 1).astype(float))

    params = package_params(
        n_E,
        n_I,
        E_idx,
        I_idx,
        n_a_E,
        n_a_I,
        n_b_E,
        n_b_I,
        tau_a_E,
        tau_a_I,
        np.atleast_1d(tau_b_E) if n_b_E > 0 else None,
        tau_b_I,
        tau_d,
        n,
        M,
        c_SFA,
        F_STD,
        tau_STD,
        EI_vec,
    )

    # Initial conditions
    a0_E = np.zeros(n_E * n_a_E) if (n_E > 0 and n_a_E > 0) else np.array([])
    a0_I = np.zeros(n_I * n_a_I) if (n_I > 0 and n_a_I > 0) else np.array([])
    b0_E = np.ones(n_E * n_b_E) if (n_E > 0 and n_b_E > 0) else np.array([])
    b0_I = np.ones(n_I * n_b_I) if (n_I > 0 and n_b_I > 0) else np.array([])
    u_d0 = np.zeros(n)
    X0 = np.concatenate([a0_E, a0_I, b0_E, b0_I, u_d0])

    # Integrate
    t_out, X = solve(T, t, X0, t, u_ex, params, method="BDF", rtol=1e-7, atol=1e-8, max_step=dt)

    # Unpack and compute dependent variables
    a_E_ts, a_I_ts, b_E_ts, b_I_ts, u_d_ts = unpack_trajectory(X, params)
    r, p = compute_dependent(a_E_ts, a_I_ts, b_E_ts, b_I_ts, u_d_ts, params)

    # Example: compute LLE via Benettin
    LLE, local_lya, finite_lya, t_lya = benettin_algorithm(
        X, t_out, dt, fs, d0=1e-3, T=T, lya_dt=0.5 * params.tau_d, params=params,
        dyn_factory=None, t_ex=t, u_ex=u_ex, method="BDF"
    )
    print({"LLE": float(LLE), "last_finite": float(finite_lya[~np.isnan(finite_lya)][-1]) if np.any(~np.isnan(finite_lya)) else None})

    # Minimal plots if desired
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
        axes[0].plot(t_out, r.T)
        axes[0].set_ylabel("r (Hz)")
        axes[1].plot(t_out, u_d_ts.T)
        axes[1].set_ylabel("u_d")
        axes[2].plot(t_lya, local_lya, label="local")
        axes[2].plot(t_lya, finite_lya, label="finite")
        axes[2].legend()
        axes[2].set_xlabel("t (s)")
        fig.tight_layout()
        plt.show()
    except Exception as e:
        # plotting optional
        pass


if __name__ == "__main__":
    main()

