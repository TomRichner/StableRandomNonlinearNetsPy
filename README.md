# StableRandomNonlinearNetsPy

Continuous‑time RNNs with spike‑frequency adaptation (SFA) and short‑term synaptic depression (STD).

### Goal
Port the MATLAB repo `dual_adaptation_random_matrix_theory` to Python here, preserving functionality and figures:
- Copy over `docs/`
- Reimplement the SRNN ODE (`SRNN_NL.m`) and example (`SRNN_basic_example.m`)
- Add plotting (time series, raster/stacked, network graph)
- Implement Lyapunov exponent tooling (Benettin, QR; SVD optional)

### Python packages
- Core runtime
  - numpy: array math
  - scipy: ODE integration (`integrate.solve_ivp` with `RK45` and `BDF`), interpolation (`interpolate.interp1d`), linear algebra
  - matplotlib: plotting
  - numba (optional): JIT for RHS and dependent-variable computations
  - networkx (optional): network visualization similar to MATLAB `digraph`
  - seaborn (optional): nicer palettes for figures
  - tqdm (optional): progress bars for parameter sweeps
  - h5py or numpy.savez: saving results
  - joblib (optional): parallel sweeps
- Lyapunov utilities
  - numpy.linalg.qr, numpy.linalg.svd for QR/SVD methods
  - Optional alt: jax + diffrax (future) for autodiff Jacobians and high‑performance ODEs
- Dev/test
  - pytest (tests), black/ruff (format/lint), mypy (types), pre-commit (hooks)

### Proposed structure
```
SRNN_py/
  docs/                      # copied from MATLAB repo
  src/SRNN/
    __init__.py
    activation.py            # ReLU default, plug-in activations
    params.py                # Dataclass for all parameters (E/I, taus, M, etc.)
    state.py                 # pack/unpack state vectors <-> structured views
    dynamics.py              # ODE RHS (port of SRNN_NL.m)
    simulate.py              # solve_ivp wrapper (RK45/ode45-like, BDF/ode15s-like)
    dependent.py             # port of compute_dependent_variables.m
    utils/
      generate_M_no_iso.py   # port of generate_M_no_iso.m
      get_EI_indices.py      # port of get_EI_indices.m
      package_params.py      # creates Params object
    plotting/
      tseries.py             # port of SRNN_tseries_figure.m
      network_graph.py       # network plot (widths/colors by E/I)
    algorithms/lyapunov/
      benettin.py            # largest LE
      qr.py                  # full spectrum via QR
      svd.py                 # (optional) SVD spectrum
  examples/
    basic_example.py         # port of SRNN_basic_example.m
  tests/
    test_dynamics.py         # shape/consistency tests
    test_lorenz_lya.py       # parity with provided tests where applicable
```

### Porting plan (no coding yet)
1) Mirror documentation
   - Copy `dual_adaptation_random_matrix_theory/docs/` into `SRNN_py/docs/`.

2) Core ODE and params
   - Implement `Params` dataclass; replicate E/I splits, timescales, coefficients, and `M`.
   - Port `SRNN_NL.m` as `dynamics.py` RHS callable. Build `interp1d` once per simulation for `u_ex`.
   - Port `package_params.m`, `get_EI_indices.m`, `unpack_SRNN_state.m` (as `state.py` helpers), `compute_dependent_variables.m` (as `dependent.py`).

3) Solver wrapper and example
   - `simulate.py`: thin wrapper over `solve_ivp` with methods `RK45` (ode45‑like) and `BDF` (ode15s‑like), fixed `t_eval` grid.
   - Reproduce `SRNN_basic_example.m` in `examples/basic_example.py` (same seed, params, and stimuli).

4) Plotting parity
   - Port `SRNN_tseries_figure.m` to `plotting/tseries.py` (input, mean E/I rates, stacked or raster, SFA sum, STD product, Lyapunov panel).
   - `plotting/network_graph.py` using networkx to mimic MATLAB digraph figure.

5) Lyapunov exponents
   - Benettin: integrate reference + renormalized perturbation over intervals; record local/finite/global LLE.
   - QR method: evolve tangent dynamics via Jacobian‑vector products and periodic QR; start with analytic Jacobian or finite differences; consider optional JAX later.

6) Validation
   - Shape/unit tests; reproducibility (fixed RNG seed), basic parity vs MATLAB outputs for small cases.
   - Numerical note: minor differences expected due to solver tolerances and interpolation.

### Minimal dependencies to start
```
pip install numpy scipy matplotlib
```
Optional for speed/UX later: `numba networkx seaborn tqdm h5py joblib`.

### Notes on numerical behavior
- Use `solve_ivp` with `BDF` for stiff cases (ode15s‑like) and `RK45` for non‑stiff (ode45‑like).
- Build/expose solver tolerances (`rtol`, `atol`, `max_step`) to match MATLAB runs.
- Cache `u_ex` interpolant per simulation to avoid per‑call allocation in the RHS.
