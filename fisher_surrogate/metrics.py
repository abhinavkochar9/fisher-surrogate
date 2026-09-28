"""Trajectory-level and mechanism-level metrics (revised paper Sec. 4.5).

Unified definitions used for EVERY table of the revision
---------------------------------------------------------

    E_traj  : mean squared trajectory error on the evaluation grid          (Eq. 19)
    E_cross : |T_hat - T_true| for trajectories whose TRUE trajectory        (Eq. 20)
              crosses ``level`` from below.  Detection: first grid index k
              with u[k-1] < level <= u[k], linear interpolation, then --
              for continuous surrogates -- root-finding refinement of the
              surrogate crossing inside the bracketing grid interval.
              A surrogate that produces NO upward crossing is assigned
              T_hat = T_max and counted (``n_cross_fail``).  A surrogate that
              produces a crossing where the truth has none is counted as
              ``n_cross_spurious`` (not part of E_cross, but reported).
    E_over  : |max_t u_hat - max_t u| over the evaluation grid               (Eq. 21)
              ("finite-horizon excursion").
    E_eq    : |u_hat(T_max) - u_eq|, with ``u_eq`` the closed-form           (Eq. 22)
              Fisher-displaced equilibrium u*_eps by default, or any
              per-trajectory attractor supplied by the caller (numerically
              obtained stationary state, transfer-system equilibrium).

The pre-revision variants (missing crossing *excluded*; E_over as excess
above u*) are kept behind ``cross_missing="exclude"`` and
``over_mode="eq_clipped"`` so that the submitted tables remain reproducible,
but every script in ``experiments/`` now defaults to the unified definitions.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import t as student_t

from .dynamics import crossing_time_exact, u_star

METRICS = ["E_traj", "E_cross", "E_over", "E_eq"]
COUNTS = ["n_cross", "n_cross_fail", "n_cross_spurious"]

UNIFIED_METRIC_KW = dict(cross_missing="penalize", over_mode="max", refine=True)
LEGACY_TORCH_METRIC_KW = dict(cross_missing="penalize", over_mode="eq_clipped", refine=False)
LEGACY_SKLEARN_METRIC_KW = dict(cross_missing="exclude", over_mode="max", refine=False)


# ----------------------------------------------------------------------------
# crossing detection
# ----------------------------------------------------------------------------

def crossing_bracket(t, u, level=1.0):
    """Index k of the first upward crossing (u[k-1] < level <= u[k]); None if none."""
    u = np.asarray(u, float)
    if u[0] >= level:
        return None
    above = np.where(u >= level)[0]
    if len(above) == 0:
        return None
    return int(above[0])


def crossing_time(t, u, level=1.0, pred_fn=None, xtol=1e-10):
    """First upward crossing of ``level``.

    Grid detection + linear interpolation; if ``pred_fn(t_array) -> u_array`` is
    given (a surrogate that is continuous in t) the crossing is refined by
    Brent root-finding inside the bracketing interval.  Returns nan if the
    trajectory does not cross from below.
    """
    t = np.asarray(t, float)
    u = np.asarray(u, float)
    k = crossing_bracket(t, u, level)
    if k is None:
        return np.nan
    t0, t1 = t[k - 1], t[k]
    a, b = u[k - 1], u[k]
    t_lin = float(t1) if abs(b - a) < 1e-14 else float(t0 + (level - a) * (t1 - t0) / (b - a))
    if pred_fn is None:
        return t_lin
    try:
        f = lambda tt: float(np.asarray(pred_fn(np.array([tt])), float).reshape(-1)[0]) - level
        fa, fb = f(t0), f(t1)
        if fa < 0.0 <= fb:
            return float(brentq(f, t0, t1, xtol=xtol))
    except Exception:  # pragma: no cover - fall back to the interpolated value
        pass
    return t_lin


def grid_resolution(t):
    t = np.asarray(t, float)
    return float((t[-1] - t[0]) / (len(t) - 1))


# ----------------------------------------------------------------------------
# per-trajectory metrics
# ----------------------------------------------------------------------------

def trajectory_metrics(t, u_true, u_pred, eps=None, level=1.0, cross_missing="penalize",
                       over_mode="max", refine=True, pred_fn=None, t_cross_true=None,
                       u_eq=None):
    """Per-trajectory metrics.

    Parameters
    ----------
    eps          : Fisher strength; used for the closed-form u*_eps and T_acc unless
                   ``u_eq`` / ``t_cross_true`` are given explicitly.
    level        : regime-transition level (1.0 on the Fisher benchmark; x1* on the
                   transfer system).
    pred_fn      : optional continuous surrogate ``t_array -> u_array`` for crossing
                   refinement (ignored if ``refine=False``).
    t_cross_true : true crossing time.  If None it is taken from the closed-form
                   quadrature when ``eps`` is given and level == 1, otherwise from the
                   (fine) true trajectory by interpolation.
    u_eq         : attractor reference; defaults to u*_eps.
    """
    t = np.asarray(t, float)
    u_true = np.asarray(u_true, float)
    u_pred = np.asarray(u_pred, float)
    if u_eq is None:
        if eps is None:
            raise ValueError("need eps or u_eq")
        u_eq = float(u_star(eps))

    e_traj = float(np.mean((u_pred - u_true) ** 2))

    # --- regime-transition timing ---
    if t_cross_true is None:
        if eps is not None and level == 1.0 and u_true[0] < 1.0:
            t_cross_true = crossing_time_exact(float(u_true[0]), float(eps))
        else:
            t_cross_true = crossing_time(t, u_true, level)
    e_cross, fail, spurious = np.nan, 0, 0
    tc_pred = crossing_time(t, u_pred, level, pred_fn if refine else None)
    if not np.isnan(t_cross_true):
        if np.isnan(tc_pred):
            fail = 1
            if cross_missing == "penalize":
                e_cross = abs(float(t[-1]) - float(t_cross_true))
        else:
            e_cross = abs(float(tc_pred) - float(t_cross_true))
    elif not np.isnan(tc_pred):
        spurious = 1

    # --- finite-horizon excursion ---
    if over_mode == "max":
        e_over = abs(float(np.max(u_pred)) - float(np.max(u_true)))
    elif over_mode == "eq_clipped":
        e_over = abs(max(0.0, float(np.max(u_pred)) - u_eq) - max(0.0, float(np.max(u_true)) - u_eq))
    else:
        raise ValueError(over_mode)

    # --- asymptotic displacement ---
    e_eq = abs(float(u_pred[-1]) - float(u_eq))
    return {"E_traj": e_traj, "E_cross": e_cross, "E_over": e_over, "E_eq": e_eq,
            "T_cross_true": float(t_cross_true), "T_cross_pred": float(tc_pred),
            "cross_fail": fail, "cross_spurious": spurious}


def compute_metrics(pred_df, true_col="u_clean", pred_col="u_pred", pred_fn_factory=None,
                    level=1.0, level_col=None, u_eq_col=None, t_cross_col=None,
                    return_per_trajectory=False, **kw):
    """Aggregate metrics over a long-form DataFrame with a ``pair_id`` column.

    ``pred_fn_factory(group_df) -> (t_array -> u_array)`` builds the continuous
    surrogate for one trajectory (used for crossing refinement).  ``level_col``,
    ``u_eq_col`` and ``t_cross_col`` name optional per-trajectory columns that
    override the closed-form Fisher references (transfer system, PDE targets).
    """
    rows = []
    for pid, g in pred_df.groupby("pair_id"):
        g = g.sort_values("t")
        eps = float(g["eps"].iloc[0]) if "eps" in g else None
        lvl = float(g[level_col].iloc[0]) if level_col else level
        ueq = float(g[u_eq_col].iloc[0]) if u_eq_col else None
        tct = float(g[t_cross_col].iloc[0]) if t_cross_col else None
        fn = pred_fn_factory(g) if pred_fn_factory is not None else None
        m = trajectory_metrics(g["t"].to_numpy(), g[true_col].to_numpy(), g[pred_col].to_numpy(),
                               eps=eps, level=lvl, pred_fn=fn, t_cross_true=tct, u_eq=ueq, **kw)
        m["pair_id"] = pid
        rows.append(m)
    per = pd.DataFrame(rows)
    out = {m: float(np.nanmean(per[m])) if per[m].notna().any() else np.nan for m in METRICS}
    out["n_cross"] = int(per["T_cross_true"].notna().sum())
    out["n_cross_fail"] = int(per["cross_fail"].sum())
    out["n_cross_spurious"] = int(per["cross_spurious"].sum())
    out["n_traj"] = int(len(per))
    if return_per_trajectory:
        return out, per
    return out


# ----------------------------------------------------------------------------
# multi-seed summaries
# ----------------------------------------------------------------------------

def summarize_seeds(seed_df, group_cols=("model",), metrics=METRICS, count_cols=COUNTS):
    """Mean +/- 95% t-CI half-width across seeds (paper Sec. 6.1); counts are summed
    over seeds and also reported as a per-seed mean."""
    rows = []
    for key, g in seed_df.groupby(list(group_cols), sort=False):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_cols, key))
        for m in metrics:
            if m not in g:
                continue
            vals = g[m].to_numpy(float)
            n = int(np.sum(~np.isnan(vals)))
            row[f"{m}_mean"] = float(np.nanmean(vals)) if n else np.nan
            if n > 1:
                tcrit = float(student_t.ppf(0.975, n - 1))
                row[f"{m}_ci"] = tcrit * float(np.nanstd(vals, ddof=1)) / math.sqrt(n)
            else:
                row[f"{m}_ci"] = np.nan
            row[f"{m}_n"] = n
        for c in count_cols:
            if c in g:
                row[f"{c}_total"] = int(g[c].sum())
                row[f"{c}_per_seed"] = float(g[c].mean())
        if "train_time_s" in g:
            row["train_time_s_mean"] = float(g["train_time_s"].mean())
        rows.append(row)
    return pd.DataFrame(rows)
