"""Fit/evaluate adapters for the Neural ODE and Gaussian-process baselines so that all
scripts share one code path (revision items R2.4, R4.2-4.4, R5.2)."""
from __future__ import annotations

import numpy as np

from _torch_common import evaluate
from fisher_surrogate.models.gp_models import GPSurrogate
from fisher_surrogate.models.neural_ode import (HAVE_TORCHDIFFEQ, NeuralODESurrogate, ODEConfig,
                                                fisher_known_field)
from fisher_surrogate.common import BASE_SEED

NODE_SEED_OFFSET = 15000
GP_SEED_OFFSET = 17000

# Two Dormand--Prince tolerance settings reported in the revision (Sec. 6.5, Table 3).
NODE_TOLERANCES = {"standard": (1e-6, 1e-8), "tight": (1e-8, 1e-10)}


def fit_neural_ode(train_df, val_df, seed, tcfg, residual=False, rtol=1e-6, atol=1e-8, target="u_clean",
                   method="dopri5"):
    ode = ODEConfig(rtol=rtol, atol=atol, method=method if HAVE_TORCHDIFFEQ else "rk4")
    node = NeuralODESurrogate(["u0"], ["eps"], obs_index=0,
                              known_field=fisher_known_field if residual else None,
                              param_index=0 if residual else None, cfg=tcfg, ode=ode)
    node.fit(train_df, val_df, seed + NODE_SEED_OFFSET + (1 if residual else 0), target=target)
    return node


def fit_gp(train_df, seed, prior_mean="zero", n_sub=2000, target="u_clean", features=("u0", "eps", "t"),
           baseline_col=None):
    gp = GPSurrogate(prior_mean=prior_mean, n_sub=n_sub, seed=seed + GP_SEED_OFFSET, features=features,
                     baseline_col=baseline_col)
    gp.fit(train_df, target=target)
    gp.train_time_s = gp.fit_time_s
    gp.epochs_run = np.nan
    gp.n_params = gp.n_hyperparameters
    return gp


__all__ = ["fit_neural_ode", "fit_gp", "evaluate", "NODE_TOLERANCES", "HAVE_TORCHDIFFEQ", "BASE_SEED"]
