"""Gaussian-process baselines (revised paper Sec. 6.3 and 6.5; Table 10, Table 12).

    GP, zero mean          : u ~ GP(0, k),      k = RBF-ARD + white noise
    GP, u_OU prior mean    : u - u_OU ~ GP(0, k)  -- the Kennedy--O'Hagan discrepancy
                             model with unit scaling (the standard way of injecting
                             the same baseline information the residual MLP gets).

Both regress on (u0, eps, t) after z-scoring inputs and targets on the training
split, fit hyper-parameters by marginal-likelihood maximisation (L-BFGS-B, sklearn
default, ``n_restarts_optimizer`` restarts) on a random subsample of ``n_sub``
training points, and expose ``predict_frame`` (continuous posterior mean) so that
the crossing time can be located by root-finding exactly as for the neural models.
"""
from __future__ import annotations

import time

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

from ..dynamics import u_ou

FEATURES = ["u0", "eps", "t"]


class GPSurrogate:
    """RBF-ARD Gaussian process on (u0, eps, t) with optional u_OU prior mean."""

    def __init__(self, prior_mean="zero", n_sub=2000, seed=0, n_restarts=2, features=FEATURES,
                 baseline_col=None):
        assert prior_mean in ("zero", "ou", "column")
        self.prior_mean = prior_mean
        self.n_sub = n_sub
        self.seed = seed
        self.n_restarts = n_restarts
        self.features = list(features)
        self.baseline_col = baseline_col  # for prior_mean="column": baseline stored in a df column
        self.gp = None
        self.fit_time_s = None

    # ----- baseline handling -----
    def _baseline(self, df):
        if self.prior_mean == "zero":
            return np.zeros(len(df))
        if self.prior_mean == "ou":
            return u_ou(df["t"].to_numpy(float), df["u0"].to_numpy(float))
        return df[self.baseline_col].to_numpy(float)

    # ----- fitting -----
    def fit(self, train_df, target="u_clean"):
        rng = np.random.default_rng(self.seed)
        idx = np.arange(len(train_df))
        if self.n_sub is not None and self.n_sub < len(idx):
            idx = rng.choice(idx, self.n_sub, replace=False)
        sub = train_df.iloc[idx]
        X = sub[self.features].to_numpy(float)
        y = sub[target].to_numpy(float) - self._baseline(sub)
        self.x_mean, self.x_std = X.mean(0), X.std(0) + 1e-12
        self.y_mean, self.y_std = y.mean(), y.std() + 1e-12
        Xs = (X - self.x_mean) / self.x_std
        ys = (y - self.y_mean) / self.y_std
        kernel = (ConstantKernel(1.0, (1e-3, 1e3))
                  * RBF(length_scale=np.ones(X.shape[1]), length_scale_bounds=(1e-2, 1e2))
                  + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-10, 1e-1)))
        self.gp = GaussianProcessRegressor(kernel=kernel, normalize_y=False, alpha=1e-10,
                                           n_restarts_optimizer=self.n_restarts, random_state=self.seed)
        t0 = time.perf_counter()
        self.gp.fit(Xs, ys)
        self.fit_time_s = time.perf_counter() - t0
        self.n_train_points = int(len(idx))
        return self

    @property
    def n_hyperparameters(self):
        return int(self.gp.kernel_.theta.size)

    def kernel_summary(self):
        return str(self.gp.kernel_)

    # ----- prediction -----
    def predict_frame(self, df):
        X = df[self.features].to_numpy(float)
        ys = self.gp.predict((X - self.x_mean) / self.x_std)
        return ys * self.y_std + self.y_mean + self._baseline(df)

    def pred_fn_factory(self, group):
        """Continuous posterior mean along t for a single trajectory (for root-finding)."""
        row = group.iloc[0]

        def fn(t):
            t = np.asarray(t, float).reshape(-1)
            d = {c: np.full(len(t), float(row[c])) for c in self.features if c != "t"}
            d["t"] = t
            import pandas as pd
            g = pd.DataFrame(d)
            if self.baseline_col is not None and self.prior_mean == "column":
                g[self.baseline_col] = np.interp(t, group["t"].to_numpy(), group[self.baseline_col].to_numpy())
            return self.predict_frame(g)
        return fn
