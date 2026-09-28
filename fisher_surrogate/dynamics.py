"""Fisher-regularized Gaussian-manifold variance dynamics.

Reduced ODE (paper Eq. 10):

    du/dt = 2(1 - u) + eps / u,     u(0) = u0,

with the unregularized Ornstein--Uhlenbeck (OU) baseline (eps = 0)

    u_OU(t; u0) = 1 + (u0 - 1) exp(-2 t)

and the Fisher-displaced equilibrium (paper Eq. 11)

    u*_eps = (1 + sqrt(1 + 2 eps)) / 2.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import quad, solve_ivp

U_FLOOR = 1e-12  # guards the eps/u term; u is strictly increasing on (0, 1] so this is never active in practice


def u_star(eps):
    """Fisher-displaced equilibrium variance u*_eps (Eq. 11)."""
    eps = np.asarray(eps, dtype=float)
    return (1.0 + np.sqrt(1.0 + 2.0 * eps)) / 2.0


def u_ou(t, u0):
    """Unregularized OU variance trajectory (closed form)."""
    t = np.asarray(t, dtype=float)
    u0 = np.asarray(u0, dtype=float)
    return 1.0 + (u0 - 1.0) * np.exp(-2.0 * t)


def rhs(u, eps):
    """Right-hand side of the reduced variance ODE."""
    u_safe = np.maximum(u, U_FLOOR)
    return 2.0 * (1.0 - u_safe) + eps / u_safe


def cross_dissipation_gaussian(sigma):
    """Gaussian-manifold cross-dissipation coefficient D_x(sigma) = (sigma^2 - 1) / (2 sigma^4) (Eq. 12)."""
    sigma = np.asarray(sigma, dtype=float)
    return (sigma**2 - 1.0) / (2.0 * sigma**4)


def crossing_time_exact(u0, eps):
    """Closed-form acceleration-window crossing time T_acc(u0) (Eq. 13).

    Returns nan if u0 >= 1 (no crossing from below) or eps <= 0 (the unregularized
    trajectory approaches u = 1 asymptotically and never crosses it).
    """
    if u0 >= 1.0 or eps <= 0.0:
        return np.nan
    val, _ = quad(lambda u: u / (2.0 * u * (1.0 - u) + eps), u0, 1.0)
    return float(val)


def integrate_rk4(u0, eps, t_grid):
    """Fixed-step RK4 on the prescribed grid.

    This is the integrator used by the PyTorch-based experiment notebooks
    (physics-informed, DeepONet, noise, OOD).
    """
    t_grid = np.asarray(t_grid, dtype=float)
    u = np.zeros_like(t_grid)
    u[0] = float(u0)
    for k in range(1, len(t_grid)):
        dt = t_grid[k] - t_grid[k - 1]
        uk = u[k - 1]
        k1 = rhs(uk, eps)
        k2 = rhs(uk + 0.5 * dt * k1, eps)
        k3 = rhs(uk + 0.5 * dt * k2, eps)
        k4 = rhs(uk + dt * k3, eps)
        u[k] = max(uk + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4), U_FLOOR)
    return u


def integrate_adaptive(u0, eps, t_grid, rtol=1e-9, atol=1e-11, method="RK45"):
    """Adaptive solve_ivp integration sampled on ``t_grid``.

    This is the integrator used by the scikit-learn-based notebooks
    (baseline table, generalization table, ablation table).
    """
    t_grid = np.asarray(t_grid, dtype=float)
    sol = solve_ivp(
        lambda t, y: [rhs(y[0], eps)],
        (float(t_grid[0]), float(t_grid[-1])),
        [float(u0)],
        t_eval=t_grid,
        method=method,
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise RuntimeError(f"ODE solve failed for u0={u0}, eps={eps}: {sol.message}")
    return sol.y[0]


def solve_variance(u0, eps, t_grid, integrator="adaptive", **kwargs):
    """Dispatch to ``integrate_adaptive`` (default) or ``integrate_rk4``."""
    if integrator == "adaptive":
        return integrate_adaptive(u0, eps, t_grid, **kwargs)
    if integrator == "rk4":
        return integrate_rk4(u0, eps, t_grid)
    raise ValueError(f"unknown integrator {integrator!r}")


def u_ou_kappa(t, u0, kappa):
    """Structurally misspecified OU baseline 1 + (u0 - 1) exp(-2 kappa t) (revised Sec. 6.9 (C)).

    kappa = 1 recovers the exact unregularized trajectory.
    """
    t = np.asarray(t, dtype=float)
    u0 = np.asarray(u0, dtype=float)
    return 1.0 + (u0 - 1.0) * np.exp(-2.0 * kappa * t)


def acceleration_window_bound(eps):
    """C_eps = int_0^1 u / (2u(1-u) + eps) du, the u0-independent bound on T_acc (Appendix B)."""
    val, _ = quad(lambda u: u / (2.0 * u * (1.0 - u) + eps), 0.0, 1.0)
    return float(val)
