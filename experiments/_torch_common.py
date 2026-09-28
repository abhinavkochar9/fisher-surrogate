"""Shared helpers for the PyTorch multi-seed experiments.

Every script defaults to the UNIFIED metric definitions of the revision
(``metrics.UNIFIED_METRIC_KW``: missing crossing -> T_max with failure count,
E_over = |max u_hat - max u|, root-finding refinement of the crossing time on the
continuous surrogate).  ``--legacy-metrics`` reproduces the pre-revision torch
notebook definitions for auditing the submitted numbers.
"""
from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fisher_surrogate.common import (BASE_SEED, RESIDUAL_SEED_OFFSET,  # noqa: E402
                                     experiment_parser)
from fisher_surrogate.data import (FEATURES, DataConfig, build_splits, residual_to_u,  # noqa: E402
                                   xy, xy_direct, xy_residual)
from fisher_surrogate.metrics import (LEGACY_TORCH_METRIC_KW, UNIFIED_METRIC_KW,  # noqa: E402
                                      compute_metrics)
from fisher_surrogate.models.torch_models import (TrainConfig, n_params, predict_mlp,  # noqa: E402
                                                  train_mlp)


def torch_parser(description):
    p = experiment_parser(description)
    p.add_argument("--n-pairs", type=int, default=300)
    p.add_argument("--n-time", type=int, default=80)
    p.add_argument("--legacy-metrics", action="store_true",
                   help="pre-revision torch-notebook metric definitions (audit only)")
    return p


def resolve(args, default_seeds=10):
    """Apply --quick / --n-seeds / --device and build data + train configs."""
    n_seeds = args.n_seeds or default_seeds
    dcfg = DataConfig(n_pairs=args.n_pairs, n_time=args.n_time, integrator="rk4")
    tcfg = TrainConfig()
    if args.device:
        tcfg.device = args.device
    if args.quick:
        dcfg.n_pairs, dcfg.n_time, n_seeds = 60, 40, 2
        tcfg.epochs, tcfg.patience = 60, 20
    metric_kw = LEGACY_TORCH_METRIC_KW if getattr(args, "legacy_metrics", False) else UNIFIED_METRIC_KW
    return n_seeds, dcfg, tcfg, metric_kw


def config_dict(args, dcfg, tcfg, n_seeds, metric_kw, **extra):
    return {"args": vars(args), "data": asdict(dcfg), "train": asdict(tcfg),
            "n_seeds": n_seeds, "metrics": metric_kw, **extra}


# ----------------------------------------------------------------------------
# MLP fitting / evaluation with generic feature lists
# ----------------------------------------------------------------------------

class MLPFit:
    """Trained MLP + scalers + feature list.  Knows how to predict on a frame and how to
    build the continuous-in-t function used for crossing refinement.

    ``base``: how the baseline entering a residual model / a 'u_base' input feature is
    evaluated at arbitrary t --
        ("ou",)             exact OU trajectory 1 + (u0 - 1) e^{-2t}   (analytic)
        ("kappa", kappa)    misspecified rate 1 + (u0 - 1) e^{-2 kappa t} (analytic)
        ("column",)         interpolate the trajectory's own ``u_base`` column (PDE / Euler)
    """

    def __init__(self, model, xs, ys, features, device, residual=False, scaled=True, base=("ou",), scale_col="eps"):
        self.model, self.xs, self.ys, self.features = model, xs, ys, list(features)
        self.device, self.residual, self.scaled, self.base = device, residual, scaled, tuple(base)
        self.scale_col = scale_col
        self.train_time_s = getattr(model, "train_time_s", np.nan)
        self.epochs_run = getattr(model, "epochs_run", np.nan)
        self.n_params = n_params(model)

    def baseline(self, df, group=None):
        t = df["t"].to_numpy(float)
        if self.base[0] == "ou":
            return 1.0 + (df["u0"].to_numpy(float) - 1.0) * np.exp(-2.0 * t)
        if self.base[0] == "kappa":
            return 1.0 + (df["u0"].to_numpy(float) - 1.0) * np.exp(-2.0 * self.base[1] * t)
        if group is None:
            return df["u_base"].to_numpy(float)
        return np.interp(t, group["t"].to_numpy(float), group["u_base"].to_numpy(float))

    def predict_frame(self, df, group=None):
        df = df.copy()
        if "u_base" in self.features and ("u_base" not in df or group is not None):
            df["u_base"] = self.baseline(df, group)
        r = predict_mlp(self.model, df[self.features].to_numpy(float), self.xs, self.ys, self.device)
        if not self.residual:
            return r
        base = self.baseline(df, group)
        return base + (df[self.scale_col].to_numpy(float) * r if self.scaled else r)

    def pred_fn_factory(self, group):
        row = group.iloc[0]
        const = [c for c in self.features if c not in ("t", "u_base")]

        def fn(t):
            t = np.asarray(t, float).reshape(-1)
            g = pd.DataFrame({c: np.full(len(t), float(row[c])) for c in const})
            g["t"] = t
            for c in ("u0", "eps"):
                if c not in g and c in group:
                    g[c] = float(row[c])
            return self.predict_frame(g, group=group)
        return fn


def fit_direct(train_df, val_df, seed, tcfg, target="u_clean", features=FEATURES, base=("ou",)):
    x, y = xy(train_df, features, target)
    xv, yv = xy(val_df, features, target)
    model, xs, ys = train_mlp(x, y, xv, yv, seed, tcfg)
    return MLPFit(model, xs, ys, features, tcfg.device, base=base)


def fit_residual(train_df, val_df, seed, tcfg, target="u_clean", scaled=True, features=FEATURES,
                 base=("ou",), base_col=None, scale_col="eps"):
    """Residual MLP u_hat = base + eps * r_theta.  With ``base_col`` the residual target is taken
    against that column of the frame (imperfect baselines); ``base`` tells the evaluator how to
    evaluate the same baseline at arbitrary t."""
    if base_col is None:
        x, r = xy_residual(train_df, target, scaled)
        xv, rv = xy_residual(val_df, target, scaled)
    else:
        from fisher_surrogate.data import xy_residual_col
        x, r = xy_residual_col(train_df, features, target, base_col, scaled, scale_col)
        xv, rv = xy_residual_col(val_df, features, target, base_col, scaled, scale_col)
    model, xs, rs = train_mlp(x, r, xv, rv, seed + RESIDUAL_SEED_OFFSET, tcfg)
    return MLPFit(model, xs, rs, features, tcfg.device, residual=True, scaled=scaled, base=base, scale_col=scale_col)


def evaluate(fit, test_df, metric_kw, **compute_kw):
    """Predict on ``test_df`` and compute unified metrics; returns (metrics dict, pred frame)."""
    pred = test_df.copy()
    pred["u_pred"] = fit.predict_frame(test_df)
    factory = fit.pred_fn_factory if metric_kw.get("refine", False) else None
    m = compute_metrics(pred, pred_fn_factory=factory, **compute_kw, **metric_kw)
    m["train_time_s"] = getattr(fit, "train_time_s", np.nan)
    m["epochs_run"] = getattr(fit, "epochs_run", np.nan)
    m["n_params"] = getattr(fit, "n_params", np.nan)
    return m, pred


# backwards-compatible names used by the original scripts
def eval_direct(fit, test_df, tcfg=None, **metric_kw):
    return evaluate(fit, test_df, metric_kw)


def eval_residual(fit, test_df, tcfg=None, scaled=True, **metric_kw):
    return evaluate(fit, test_df, metric_kw)


def seed_of(i):
    return BASE_SEED + i


__all__ = ["torch_parser", "resolve", "config_dict", "build_splits", "fit_direct", "fit_residual",
           "evaluate", "eval_direct", "eval_residual", "seed_of", "MLPFit", "FEATURES"]
