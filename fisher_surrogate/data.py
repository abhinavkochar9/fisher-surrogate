"""Dataset generation with parameter-pair (trajectory-level) splits.

All datasets are long-form pandas DataFrames with columns
``pair_id, u0, eps, t, u_clean`` (plus ``regime`` for OOD sets).
Splitting is always over ``pair_id`` so that no time sample of a test
trajectory is seen during training (paper Sec. 4.2 / 6.1).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .dynamics import solve_variance, u_ou

FEATURES = ["u0", "eps", "t"]

# Standard training domain (paper Sec. 6.1)
TRAIN_U0_RANGE = (0.2, 2.5)
TRAIN_EPS_RANGE = (0.01, 0.5)
T_MAX = 5.0

# OOD regimes (paper Table 15). Note the sklearn generalization notebook
# used eps in [0.5, 0.8] for "High-eps extrapolation" (paper Table 5),
# while the torch OOD notebook used [0.5, 1.0] (paper Table 16).
OOD_REGIMES = {
    "Low variance": dict(u0_range=(0.05, 0.2), eps_range=TRAIN_EPS_RANGE, t_range=(0.0, 5.0)),
    "High variance": dict(u0_range=(2.5, 4.0), eps_range=TRAIN_EPS_RANGE, t_range=(0.0, 5.0)),
    "High Fisher strength": dict(u0_range=TRAIN_U0_RANGE, eps_range=(0.5, 1.0), t_range=(0.0, 5.0)),
    "Longer time": dict(u0_range=TRAIN_U0_RANGE, eps_range=TRAIN_EPS_RANGE, t_range=(5.0, 10.0)),
}


@dataclass
class DataConfig:
    n_pairs: int = 300
    n_time: int = 80
    t_max: float = T_MAX
    u0_range: tuple = TRAIN_U0_RANGE
    eps_range: tuple = TRAIN_EPS_RANGE
    data_seed: int = 123
    split_seed: int = 999
    train_frac: float = 0.70
    val_frac: float = 0.15
    integrator: str = "rk4"  # "rk4" (torch notebooks) or "adaptive" (sklearn notebooks)
    extra: dict = field(default_factory=dict)


def sample_pairs(n_pairs, u0_range, eps_range, seed):
    """Uniform i.i.d. sampling of (u0, eps) pairs (answers R2's 'sampling type' question)."""
    rng = np.random.default_rng(seed)
    u0s = rng.uniform(u0_range[0], u0_range[1], size=n_pairs)
    epss = rng.uniform(eps_range[0], eps_range[1], size=n_pairs)
    return u0s, epss


def make_dataset(n_pairs, n_time, u0_range=TRAIN_U0_RANGE, eps_range=TRAIN_EPS_RANGE,
                 t_range=(0.0, T_MAX), seed=123, integrator="rk4", regime=None):
    """Generate clean trajectories on a uniform time grid over ``t_range``."""
    u0s, epss = sample_pairs(n_pairs, u0_range, eps_range, seed)
    t_grid = np.linspace(t_range[0], t_range[1], n_time)
    # Always integrate from t=0 so a t_range starting later still has the right state.
    if t_range[0] > 0.0:
        n_pre = int(round(n_time * t_range[0] / (t_range[1] - t_range[0])))
        t_full = np.concatenate([np.linspace(0.0, t_range[0], n_pre + 1)[:-1], t_grid])
    else:
        t_full = t_grid
    frames = []
    for pair_id, (u0, eps) in enumerate(zip(u0s, epss)):
        u_full = solve_variance(u0, eps, t_full, integrator=integrator)
        u = u_full[-n_time:]
        frames.append(pd.DataFrame({
            "pair_id": pair_id, "u0": u0, "eps": eps, "t": t_grid, "u_clean": u,
        }))
    df = pd.concat(frames, ignore_index=True)
    if regime is not None:
        df["regime"] = regime
    return df


def make_ood_dataset(regime, n_pairs=120, n_time=80, seed=777, integrator="rk4"):
    spec = OOD_REGIMES[regime]
    return make_dataset(n_pairs, n_time, seed=seed, integrator=integrator, regime=regime, **spec)


def split_pair_ids(n_pairs, train_frac=0.70, val_frac=0.15, seed=999):
    rng = np.random.default_rng(seed)
    ids = np.arange(n_pairs)
    rng.shuffle(ids)
    n_train = int(train_frac * n_pairs)
    n_val = int(val_frac * n_pairs)
    return set(ids[:n_train]), set(ids[n_train:n_train + n_val]), set(ids[n_train + n_val:])


def split_dataset(df, cfg: DataConfig):
    tr, va, te = split_pair_ids(cfg.n_pairs, cfg.train_frac, cfg.val_frac, cfg.split_seed)
    return (df[df.pair_id.isin(tr)].copy(),
            df[df.pair_id.isin(va)].copy(),
            df[df.pair_id.isin(te)].copy())


def build_splits(cfg: DataConfig):
    df = make_dataset(cfg.n_pairs, cfg.n_time, cfg.u0_range, cfg.eps_range,
                      (0.0, cfg.t_max), cfg.data_seed, cfg.integrator)
    return split_dataset(df, cfg)


def add_target_noise(df, rel_noise, seed, ref_std=None, col="u_clean", out="u_target"):
    """Add N(0, (r*std)^2) noise to targets (paper Sec. 6.5). Returns a copy."""
    out_df = df.copy()
    if ref_std is None:
        ref_std = df[col].std()
    rng = np.random.default_rng(seed)
    out_df[out] = out_df[col] + rng.normal(0.0, rel_noise * ref_std, size=len(out_df))
    return out_df


def xy_direct(df, target="u_clean"):
    """Features (u0, eps, t) and direct target u."""
    return df[FEATURES].to_numpy(float), df[[target]].to_numpy(float)


def xy_residual(df, target="u_clean", scaled=True):
    """Features and residual target r* = (u - u_OU) / eps (or u - u_OU if ``scaled=False``)."""
    x = df[FEATURES].to_numpy(float)
    base = u_ou(df["t"].to_numpy(), df["u0"].to_numpy())
    r = df[target].to_numpy() - base
    if scaled:
        r = r / df["eps"].to_numpy()
    return x, r.reshape(-1, 1)


def residual_to_u(df, r_pred, scaled=True):
    """Map a predicted residual back to u_hat = u_OU + eps * r (or + r)."""
    base = u_ou(df["t"].to_numpy(), df["u0"].to_numpy())
    r_pred = np.asarray(r_pred).reshape(-1)
    return base + (df["eps"].to_numpy() * r_pred if scaled else r_pred)


# ----------------------------------------------------------------------------
# Revision additions
# ----------------------------------------------------------------------------

EPS_BINS = [("[0.01, 0.1]", 0.01, 0.1), ("[0.1, 0.3]", 0.1, 0.3), ("[0.3, 0.5]", 0.3, 0.5)]
EPS_OOD_RANGE = (0.5, 1.0)


def add_baseline_column(df, kind="ou", kappa=1.0, col="u_base"):
    """Attach the embedded baseline as a column: exact OU (kind='ou') or the misspecified
    rate variant 1 + (u0 - 1) exp(-2 kappa t) (kind='kappa', Sec. 6.9 (C))."""
    from .dynamics import u_ou_kappa
    out = df.copy()
    if kind == "ou":
        out[col] = u_ou(out["t"].to_numpy(), out["u0"].to_numpy())
    elif kind == "kappa":
        out[col] = u_ou_kappa(out["t"].to_numpy(), out["u0"].to_numpy(), kappa)
    else:
        raise ValueError(kind)
    return out


def xy(df, features, target="u_clean"):
    """Generic feature/target extraction (feature-matched models use FEATURES + ['u_base'])."""
    return df[list(features)].to_numpy(float), df[[target]].to_numpy(float)


def xy_residual_col(df, features=FEATURES, target="u_clean", base_col="u_base", scaled=True, scale_col="eps"):
    """Residual target against an arbitrary baseline column (imperfect-baseline experiments and the
    transfer system, where the perturbation parameter is gamma)."""
    x = df[list(features)].to_numpy(float)
    r = df[target].to_numpy() - df[base_col].to_numpy()
    if scaled:
        r = r / df[scale_col].to_numpy()
    return x, r.reshape(-1, 1)


def residual_col_to_u(df, r_pred, base_col="u_base", scaled=True):
    r_pred = np.asarray(r_pred).reshape(-1)
    return df[base_col].to_numpy() + (df["eps"].to_numpy() * r_pred if scaled else r_pred)


def make_eps_zero_dataset(n_pairs=60, n_time=80, seed=4242, u0_range=TRAIN_U0_RANGE, integrator="rk4"):
    """eps = 0 test set (Sec. 6.2, Table 8): the residual model is exact by construction."""
    df = make_dataset(n_pairs, n_time, u0_range, (0.0, 0.0), seed=seed, integrator=integrator)
    df["eps"] = 0.0
    df["u_clean"] = u_ou(df["t"].to_numpy(), df["u0"].to_numpy())
    return df


def eps_bin_label(eps):
    for name, lo, hi in EPS_BINS:
        if lo <= eps <= hi:
            return name
    return "other"
