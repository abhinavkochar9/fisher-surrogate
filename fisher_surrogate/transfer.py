"""Non-normal linear transfer system (revised paper Sec. 6.10, Eq. 24; Table 20, Fig. 9;
and the coarse-solver baseline axis (B) of Sec. 6.9, Table 19).

    x' = A x + b,   A = [[-1, gamma], [0, -2]],   b = [1, 1],
    observable y = x1,   x1(0) = c in [0.3, 1.3],  x2(0) = 6,  gamma in [0.02, 0.3].

Equilibrium x* = -A^{-1} b  (x2* = 1/2, x1* = 1 + gamma/2).  The transient of x1
shows a genuine interior overshoot for most (c, gamma), so E_over is a true
overshoot metric here.  The residual surrogate is

    y_hat = y_base(t; c) + gamma r_theta(c, gamma, t),

with y_base the gamma = 0 response, obtained by numerical integration on a fine
table and linearly interpolated (the matrix-exponential solution is never used
by any surrogate).  ``euler_baseline`` replaces the table by a forward-Euler
integration with step h (Sec. 6.9 (B)).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from .metrics import crossing_time

X2_0 = 6.0
B_VEC = np.array([1.0, 1.0])
C_RANGE = (0.3, 1.3)
GAMMA_RANGE = (0.02, 0.3)
T_MAX = 5.0
FEATURES = ["c", "gamma", "t"]


def a_matrix(gamma):
    return np.array([[-1.0, gamma], [0.0, -2.0]])


def equilibrium(gamma):
    return -np.linalg.solve(a_matrix(gamma), B_VEC)


def rhs(t, x, gamma):
    return a_matrix(gamma) @ x + B_VEC


def integrate(c, gamma, t_grid, rtol=1e-10, atol=1e-12):
    sol = solve_ivp(rhs, (0.0, float(t_grid[-1])), [c, X2_0], t_eval=t_grid, args=(gamma,),
                    method="RK45", rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y  # (2, T)


def baseline_table(c, t_fine=None):
    """Numerically tabulated gamma = 0 response of x1 (fine grid, then interpolated)."""
    if t_fine is None:
        t_fine = np.linspace(0.0, T_MAX, 2001)
    return t_fine, integrate(c, 0.0, t_fine)[0]


def y_base(c, t):
    t_fine, yb = baseline_table(c)
    return np.interp(t, t_fine, yb)


def euler_baseline(c, t, h):
    """Forward-Euler integration of the gamma = 0 system with step h, interpolated to t."""
    n = int(np.ceil(T_MAX / h))
    tt = np.arange(n + 1) * h
    x = np.array([c, X2_0])
    ys = [x[0]]
    A0 = a_matrix(0.0)
    for _ in range(n):
        x = x + h * (A0 @ x + B_VEC)
        ys.append(x[0])
    return np.interp(t, tt, np.array(ys))


def make_dataset(n_pairs, n_time=80, seed=321, c_range=C_RANGE, gamma_range=GAMMA_RANGE,
                 baseline="table", euler_h=None):
    """Long-form frame with columns pair_id, c, gamma, t, y_clean, y_base, level, y_eq, T_cross_true."""
    rng = np.random.default_rng(seed)
    cs = rng.uniform(*c_range, size=n_pairs)
    gs = rng.uniform(*gamma_range, size=n_pairs)
    t = np.linspace(0.0, T_MAX, n_time)
    t_fine = np.linspace(0.0, T_MAX, 5001)
    frames = []
    for pid, (c, g) in enumerate(zip(cs, gs)):
        y = integrate(c, g, t)[0]
        y_fine = integrate(c, g, t_fine)[0]
        xeq = equilibrium(g)
        tc = crossing_time(t_fine, y_fine, xeq[0])
        yb = y_base(c, t) if baseline == "table" else euler_baseline(c, t, euler_h)
        frames.append(pd.DataFrame({
            "pair_id": pid, "c": c, "gamma": g, "t": t, "y_clean": y, "y_base": yb,
            "y_base_exact": y_base(c, t) if baseline != "table" else yb,
            "level": xeq[0], "y_eq": xeq[0], "T_cross_true": tc,
            "y_max_true": float(y_fine.max()), "t_peak_true": float(t_fine[np.argmax(y_fine)]),
        }))
    return pd.concat(frames, ignore_index=True)


def overshoot_stats(df):
    """Fraction of trajectories with a genuine interior overshoot (peak strictly inside (0, T_max))."""
    g = df.groupby("pair_id").first()
    interior = (g["t_peak_true"] > 0) & (g["t_peak_true"] < T_MAX - 1e-9) & (g["y_max_true"] > g["y_eq"] + 1e-6)
    return {"n_pairs": int(len(g)), "n_overshoot": int(interior.sum()),
            "max_peak_time": float(g.loc[interior, "t_peak_true"].max()) if interior.any() else np.nan,
            "max_overshoot": float((g.loc[interior, "y_max_true"] - g.loc[interior, "y_eq"]).max()) if interior.any() else np.nan}
