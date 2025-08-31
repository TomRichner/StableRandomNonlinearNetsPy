from __future__ import annotations

from typing import Literal, Tuple

import numpy as np

from .params import Params
from .state import unpack_trajectory
from .dependent import compute_dependent
from .simulate import solve as solve_ivp_func
from .algorithms.lyapunov.benettin import benettin_algorithm

class Trajectory:
    """Holds the results of an SRNN simulation."""

    def __init__(self, t: np.ndarray, X: np.ndarray, params: Params, t_ex: np.ndarray | None = None, u_ex: np.ndarray | None = None, solver_options: dict = None):
        self.t = t
        self.X = X
        self.params = params
        self.t_ex = t_ex
        self.u_ex = u_ex
        self.solver_options = solver_options if solver_options is not None else {}
        self._unpacked_state = None
        self._dependent_vars = None
        self._sfa_contrib = None
        self._std_prod = None

    def _unpack_state(self):
        if self._unpacked_state is None:
            self._unpacked_state = unpack_trajectory(self.X, self.params)
        return self._unpacked_state

    def _compute_dependent(self):
        if self._dependent_vars is None:
            a_E_ts, a_I_ts, b_E_ts, b_I_ts, u_d_ts = self._unpack_state()
            self._dependent_vars = compute_dependent(
                a_E_ts, a_I_ts, b_E_ts, b_I_ts, u_d_ts, self.params
            )
        return self._dependent_vars

    @property
    def a_E_ts(self) -> np.ndarray | None:
        return self._unpack_state()[0]

    @property
    def a_I_ts(self) -> np.ndarray | None:
        return self._unpack_state()[1]

    @property
    def b_E_ts(self) -> np.ndarray | None:
        return self._unpack_state()[2]

    @property
    def b_I_ts(self) -> np.ndarray | None:
        return self._unpack_state()[3]

    @property
    def u_d_ts(self) -> np.ndarray:
        return self._unpack_state()[4]

    @property
    def r(self) -> np.ndarray:
        """Firing rates (Hz)."""
        return self._compute_dependent()[0]

    @property
    def p(self) -> np.ndarray:
        """Synaptic outputs."""
        return self._compute_dependent()[1]

    @property
    def sfa_contrib(self) -> np.ndarray:
        """SFA contribution c_SFA * sum(a_*)."""
        if self._sfa_contrib is None:
            sfa = np.zeros((self.params.n, self.t.size))
            if (self.params.n_E > 0) and (self.params.n_a_E > 0) and (self.a_E_ts is not None):
                sum_a_E = np.sum(self.a_E_ts, axis=1)
                sfa[self.params.E_indices, :] += self.params.c_SFA[self.params.E_indices][:, None] * sum_a_E
            if (self.params.n_I > 0) and (self.params.n_a_I > 0) and (self.a_I_ts is not None):
                sum_a_I = np.sum(self.a_I_ts, axis=1)
                sfa[self.params.I_indices, :] += self.params.c_SFA[self.params.I_indices][:, None] * sum_a_I
            self._sfa_contrib = sfa
        return self._sfa_contrib

    @property
    def std_prod(self) -> np.ndarray:
        """STD product prod(b_*)."""
        if self._std_prod is None:
            std = np.ones((self.params.n, self.t.size))
            if (self.params.n_E > 0) and (self.params.n_b_E > 0) and (self.b_E_ts is not None):
                prod_b_E = np.prod(self.b_E_ts, axis=1)
                std[self.params.E_indices, :] = prod_b_E
            if (self.params.n_I > 0) and (self.params.n_b_I > 0) and (self.b_I_ts is not None):
                prod_b_I = np.prod(self.b_I_ts, axis=1)
                std[self.params.I_indices, :] = prod_b_I
            self._std_prod = std
        return self._std_prod

    def calculate_lle(self, dt: float, fs: float, d0: float = 1e-3, lya_dt: float | None = None) -> Tuple:
        """
        Compute the Largest Lyapunov Exponent (LLE) for the trajectory.

        Wraps the benettin_algorithm.
        """
        if lya_dt is None:
            lya_dt = 0.5 * self.params.tau_d
            
        return benettin_algorithm(
            X_traj=self.X,
            t_traj=self.t,
            dt=dt,
            fs=fs,
            d0=d0,
            T=(self.t[0], self.t[-1]),
            lya_dt=lya_dt,
            params=self.params,
            dyn_factory=None,
            t_ex=self.t_ex,
            u_ex=self.u_ex,
            **self.solver_options
        )
        
    def plot(self, lle_results: Tuple | None = None):
        """Generates a default plot of the trajectory."""
        import matplotlib.pyplot as plt

        n_plots = 6 if lle_results else 5
        fig, axes = plt.subplots(n_plots, 1, figsize=(11, 10), sharex=True)
        
        ax_idx = 0
        if self.u_ex is not None and self.t_ex is not None:
             axes[ax_idx].plot(self.t_ex, self.u_ex.T, alpha=0.7)
             axes[ax_idx].set_ylabel("u_ex")
             ax_idx += 1

        axes[ax_idx].plot(self.t, self.r.T)
        axes[ax_idx].set_ylabel("r (Hz)")
        ax_idx += 1
        
        axes[ax_idx].plot(self.t, self.u_d_ts.T)
        axes[ax_idx].set_ylabel("u_d")
        ax_idx += 1

        axes[ax_idx].plot(self.t, self.sfa_contrib.T)
        axes[ax_idx].set_ylabel("SFA c*sum(a)")
        ax_idx += 1

        axes[ax_idx].plot(self.t, self.std_prod.T)
        axes[ax_idx].set_ylabel("STD prod(b)")
        axes[ax_idx].set_ylim(0, 1.05)
        ax_idx += 1
        
        if lle_results:
            _, local_lya, finite_lya, t_lya = lle_results
            axes[ax_idx].plot(t_lya, local_lya, label="local")
            axes[ax_idx].plot(t_lya, finite_lya, label="finite")
            axes[ax_idx].legend()
            ax_idx += 1

        axes[-1].set_xlabel("t (s)")
        fig.tight_layout()
        plt.show()
        return fig, axes


class SRNN:
    """
    A class to encapsulate a Stabilized Random Neural Network model.
    """

    def __init__(self, params: Params):
        self.params = params

    def get_initial_state(self) -> np.ndarray:
        """
        Generates a default initial state vector for the simulation.
        This represents a system at rest (no adaptation, full synaptic resources).
        """
        p = self.params
        a0_E = np.zeros(p.n_E * p.n_a_E) if (p.n_E > 0 and p.n_a_E > 0) else np.array([])
        a0_I = np.zeros(p.n_I * p.n_a_I) if (p.n_I > 0 and p.n_a_I > 0) else np.array([])
        b0_E = np.ones(p.n_E * p.n_b_E) if (p.n_E > 0 and p.n_b_E > 0) else np.array([])
        b0_I = np.ones(p.n_I * p.n_b_I) if (p.n_I > 0 and p.n_b_I > 0) else np.array([])
        u_d0 = np.zeros(p.n)
        return np.concatenate([a0_E, a0_I, b0_E, b0_I, u_d0])

    def solve(
        self,
        T: Tuple[float, float],
        t_eval: np.ndarray,
        X0: np.ndarray,
        t_ex: np.ndarray,
        u_ex: np.ndarray,
        method: Literal["RK45", "BDF"] = "BDF",
        rtol: float = 1e-7,
        atol: float | np.ndarray = 1e-8,
        max_step: float | None = None,
    ) -> Trajectory:
        """
        Run the simulation.

        Wraps the scipy.integrate.solve_ivp solver.
        """
        solver_options = {
            "method": method,
            "rtol": rtol,
            "atol": atol,
            "max_step": max_step,
        }
        t_out, X_out = solve_ivp_func(
            T=T,
            t_eval=t_eval,
            X0=X0,
            t_ex=t_ex,
            u_ex=u_ex,
            params=self.params,
            **solver_options
        )
        return Trajectory(t_out, X_out, self.params, t_ex=t_ex, u_ex=u_ex, solver_options=solver_options)
