"""scikit-learn surrogates used for the baseline, generalization and
architecture-ablation tables (paper Tables 3, 4, 5).

Two hyper-parameter presets exist because the two original notebooks
differed:

    "generalization" -- tanh, lr 1e-3, max_iter 2000, val_frac 0.15,
                        alpha 1e-6  (fisher_residual_mlp_surrogate_generalization.ipynb)
    "ablation"       -- relu, lr 5e-4, max_iter 1500, val_frac 0.10
                        (fisher_surrogate_ablation_colab.ipynb)
"""
from __future__ import annotations

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from ..dynamics import u_ou

PRESETS = {
    "generalization": dict(activation="tanh", learning_rate_init=1e-3, max_iter=2000,
                           validation_fraction=0.15, n_iter_no_change=50, alpha=1e-6),
    "ablation": dict(activation="relu", learning_rate_init=5e-4, max_iter=1500,
                     validation_fraction=0.10, n_iter_no_change=60, alpha=1e-4),
}


def build_linear(input_scaling=True):
    steps = [("x_scaler", StandardScaler())] if input_scaling else []
    return Pipeline(steps + [("linear", LinearRegression())])


def build_polynomial(degree=3, input_scaling=True):
    steps = [("x_scaler", StandardScaler())] if input_scaling else []
    return Pipeline(steps + [("poly", PolynomialFeatures(degree=degree, include_bias=False)),
                             ("linear", LinearRegression())])


def build_mlp(hidden_layers=(64, 64), input_scaling=True, target_scaling=True,
              random_state=0, preset="generalization", **overrides):
    hp = {**PRESETS[preset], **overrides}
    mlp = MLPRegressor(hidden_layer_sizes=hidden_layers, solver="adam", batch_size=512,
                       early_stopping=True, tol=1e-8, random_state=random_state, **hp)
    steps = [("x_scaler", StandardScaler())] if input_scaling else []
    base = Pipeline(steps + [("mlp", mlp)])
    if target_scaling:
        return TransformedTargetRegressor(regressor=base, transformer=StandardScaler())
    return base


class ResidualOUSurrogate:
    """Structure-aware residual surrogate (paper Eq. 16):

        u_hat(t; u0, eps) = u_OU(t; u0) + eps * r_theta(u0, eps, t)

    Set ``scaled=False`` for the unscaled variant u_OU + r_theta (Table 6).
    Feature columns are (u0, eps, t).
    """

    def __init__(self, hidden_layers=(64, 64), input_scaling=True, target_scaling=True,
                 random_state=0, preset="generalization", scaled=True, **overrides):
        self.scaled = scaled
        self.model = build_mlp(hidden_layers, input_scaling, target_scaling, random_state,
                               preset, **overrides)

    @staticmethod
    def baseline(X):
        X = np.asarray(X, float)
        return u_ou(X[:, 2], X[:, 0])

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, float).ravel()
        r = y - self.baseline(X)
        if self.scaled:
            r = r / np.maximum(X[:, 1], 1e-12)
        self.model.fit(X, r)
        return self

    def predict(self, X):
        X = np.asarray(X, float)
        r = np.asarray(self.model.predict(X), float).ravel()
        if self.scaled:
            r = X[:, 1] * r
        return self.baseline(X) + r
