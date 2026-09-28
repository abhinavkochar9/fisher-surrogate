"""Surrogate dataset whose target is the variance trajectory of the full
Fisher-regularized PDE started from symmetric-bimodal initial densities
(revised paper Sec. 6.9 (A), Table 19).

    rho_0 = (1/2) N(-a, s^2) + (1/2) N(a, s^2),  s^2 = u0 - a^2,  a = alpha sqrt(u0),

so that a = 0 (alpha = 0) recovers the Gaussian case for which the reduced ODE
baseline is exact, and the baseline error ||u_PDE - u_ODE|| grows with the
separation.  Inputs are (u0, eps, a, t); the baseline column ``u_base`` holds
the reduced-ODE solution with the same u0 and eps.

Simulations are cached as a CSV (``results/pde_bimodal/dataset_*.csv``) because
each PDE trajectory takes a few seconds.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import pde
from .dynamics import crossing_time_exact, integrate_adaptive, u_star
from .metrics import crossing_time

FEATURES_A = ["u0", "eps", "a", "t"]
U0_RANGE_PDE = (0.2, 1.5)
ALPHA_RANGE = (0.0, 0.85)


def sample_triples(n_pairs, seed, u0_range=U0_RANGE_PDE, eps_range=(0.01, 0.5), alpha_range=ALPHA_RANGE,
                   n_exact=None, ratio_range=None):
    """i.i.d. uniform (u0, eps, alpha); the first ``n_exact`` triples get alpha = 0 (Gaussian).

    With ``a = alpha*sqrt(u0)`` and component variance ``u0 - a**2`` the bimodality ratio is
    ``a/sigma_c = alpha/sqrt(1 - alpha**2)`` independently of u0 (bimodal iff alpha > 1/sqrt 2).
    If ``ratio_range`` is given, the separation is sampled uniformly in that ratio instead and
    mapped back through ``alpha = r/sqrt(1 + r**2)`` -- the revision uses (0, 1.5), whose
    endpoint is the Table 2 symmetric-bimodal family (a = 0.5, sigma2 = 0.11 at u0 = 0.36).
    """
    rng = np.random.default_rng(seed)
    u0 = rng.uniform(*u0_range, size=n_pairs)
    eps = rng.uniform(*eps_range, size=n_pairs)
    if ratio_range is not None:
        r = rng.uniform(*ratio_range, size=n_pairs)
        alpha = r / np.sqrt(1.0 + r**2)
    else:
        alpha = rng.uniform(*alpha_range, size=n_pairs)
    if n_exact:
        alpha[:n_exact] = 0.0
    return u0, eps, alpha


def simulate_triple(u0, eps, alpha, t_grid, nx=384, L=6.0, rtol=1e-7, atol=1e-10):
    grid = pde.Grid(L, nx)
    a = alpha * np.sqrt(u0)
    rho0 = pde.density_symmetric_bimodal(grid.x, a=a, u0=u0)
    res = pde.simulate(rho0, eps, grid, t_max=float(t_grid[-1]), dt=np.inf, n_out=len(t_grid),
                       rtol=rtol, atol=atol)
    return res


def build_dataset(n_pairs, n_time=80, t_max=5.0, seed=555, cache: Path | None = None, n_exact=None,
                  nx=384, L=6.0, verbose=True, **sample_kw):
    if cache is not None and Path(cache).exists():
        return pd.read_csv(cache)
    t = np.linspace(0.0, t_max, n_time)
    u0s, epss, alphas = sample_triples(n_pairs, seed, n_exact=n_exact, **sample_kw)
    frames = []
    for pid, (u0, eps, al) in enumerate(zip(u0s, epss, alphas)):
        if verbose:
            print(f"  PDE trajectory {pid + 1}/{n_pairs}: u0={u0:.3f} eps={eps:.3f} alpha={al:.3f}", flush=True)
        res = simulate_triple(u0, eps, al, t, nx=nx, L=L)
        u_ode = integrate_adaptive(u0, eps, t)
        a = al * np.sqrt(u0)
        frames.append(pd.DataFrame({
            "pair_id": pid, "u0": u0, "eps": eps, "a": a, "alpha": al, "t": t,
            "u_clean": res.u, "u_base": u_ode,
            "baseline_rmse": float(np.sqrt(np.mean((res.u - u_ode) ** 2))),
            "T_cross_true": crossing_time(res.t, res.u, 1.0),
            "T_cross_ode": crossing_time_exact(u0, eps),
            "T_x_true": res.T_x, "u_eq": float(u_star(eps)),
        }))
    df = pd.concat(frames, ignore_index=True)
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache, index=False)
    return df


def baseline_error_bins(df, n_bins=3, labels=("small", "medium", "large")):
    """Stratify pairs by baseline RMSE: 'exact' for a = 0, then quantile bins of the rest."""
    per = df.groupby("pair_id")[["a", "baseline_rmse"]].first()
    lab = pd.Series("exact", index=per.index)
    mask = per["a"] > 0
    if mask.sum() >= n_bins:
        lab[mask] = pd.qcut(per.loc[mask, "baseline_rmse"], n_bins, labels=list(labels)).astype(str)
    return df["pair_id"].map(lab)
