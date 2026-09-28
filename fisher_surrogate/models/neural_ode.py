"""Neural ODE baselines (revised paper Sec. 6.3 and 6.5; Tables 10, 12, 18).

    Neural ODE (full field)   :  du/dt = g_phi(u, eps, t),                u(0) = u0
    Residual Neural ODE (UDE) :  du/dt = 2(1 - u) + eps g_phi(u, eps, t), u(0) = u0

Both share the (64, 64) tanh vector-field network, the adaptive Dormand--Prince
(dopri5) solver of ``torchdiffeq`` with explicitly reported ``rtol``/``atol``,
the parameter-pair split, the trajectory MSE loss (on z-scored u), Adam with the
common learning rate, and the common early-stopping rule.  Gradients are taken
by direct back-propagation through the solver (no adjoint), which is exact up to
solver tolerance for a problem of this size.

The same class handles the non-normal transfer system (state dimension 2,
observable x1, perturbation parameter gamma) through ``known_field`` and
``param_index``; see ``experiments/run_transfer_system.py``.

If ``torchdiffeq`` is not installed a fixed-step RK4 fallback (``method="rk4"``)
is used and reported in the config.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .torch_models import TrainConfig, _mlp, set_seed

try:
    from torchdiffeq import odeint as _odeint
    HAVE_TORCHDIFFEQ = True
except ModuleNotFoundError:  # pragma: no cover
    _odeint = None
    HAVE_TORCHDIFFEQ = False


@dataclass
class ODEConfig:
    rtol: float = 1e-6
    atol: float = 1e-8
    method: str = "dopri5"       # torchdiffeq method; "rk4" fixed-step fallback
    rk4_substeps: int = 4        # only for the fallback
    batch_pairs: int = 512       # pairs per batch (full batch for the standard split)


def fisher_known_field(u, params, t):
    """2(1 - u) (the unregularized OU vector field), shape (B, 1)."""
    return 2.0 * (1.0 - u)


class NeuralODE(nn.Module):
    """du/dt = f_known(u, p, t) + scale(p) * g_phi(u, p, t)  (or g_phi alone)."""

    def __init__(self, state_dim, param_dim, hidden=(64, 64), activation="tanh",
                 known_field=None, param_index=None, x_scale=None):
        super().__init__()
        self.state_dim, self.param_dim = state_dim, param_dim
        self.net = _mlp(state_dim + param_dim + 1, hidden, state_dim, activation)
        self.known_field = known_field          # callable or None (full-field model)
        self.param_index = param_index          # index of the perturbation parameter for eps-scaling
        # input standardisation (mean/std) for (state, params, t), set from training data
        self.register_buffer("in_mean", torch.zeros(state_dim + param_dim + 1))
        self.register_buffer("in_std", torch.ones(state_dim + param_dim + 1))
        self._params = None

    def set_params(self, p):
        self._params = p  # (B, param_dim)

    def forward(self, t, x):
        p = self._params
        tt = t.expand(x.shape[0], 1) if torch.is_tensor(t) else torch.full((x.shape[0], 1), float(t), device=x.device)
        z = torch.cat([x, p, tt], dim=1)
        g = self.net((z - self.in_mean) / self.in_std)
        if self.known_field is None:
            return g
        scale = p[:, self.param_index:self.param_index + 1] if self.param_index is not None else 1.0
        return self.known_field(x, p, tt) + scale * g

    def integrate(self, x0, p, t, ode: ODEConfig):
        """Integrate from x0 (B, d) over the 1-D time tensor t (must start at 0). Returns (T, B, d)."""
        self.set_params(p)
        if HAVE_TORCHDIFFEQ and ode.method != "rk4":
            return _odeint(self, x0, t, rtol=ode.rtol, atol=ode.atol, method=ode.method)
        return _rk4_torch(self, x0, t, ode.rk4_substeps)


def _rk4_torch(f, x0, t, substeps):
    out = [x0]
    x = x0
    for k in range(1, len(t)):
        h = (t[k] - t[k - 1]) / substeps
        tt = t[k - 1]
        for _ in range(substeps):
            k1 = f(tt, x)
            k2 = f(tt + 0.5 * h, x + 0.5 * h * k1)
            k3 = f(tt + 0.5 * h, x + 0.5 * h * k2)
            k4 = f(tt + h, x + h * k3)
            x = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            tt = tt + h
        out.append(x)
    return torch.stack(out, 0)


# ----------------------------------------------------------------------------
# data plumbing: long-form DataFrame -> (x0, params, T x B target)
# ----------------------------------------------------------------------------

def frame_to_arrays(df, state_cols, param_cols, target_col, t_col="t"):
    """Pivot a long-form frame into per-pair arrays on the common grid."""
    pairs = sorted(df["pair_id"].unique())
    t = np.sort(df[t_col].unique())
    x0, P, Y = [], [], []
    for pid in pairs:
        g = df[df.pair_id == pid].sort_values(t_col)
        x0.append([float(g[c].iloc[0]) for c in state_cols])
        P.append([float(g[c].iloc[0]) for c in param_cols])
        Y.append(g[target_col].to_numpy(float))
    return np.array(pairs), t, np.array(x0), np.array(P), np.array(Y).T  # Y: (T, B)


class NeuralODESurrogate:
    """Trainer / predictor wrapper with the common protocol."""

    def __init__(self, state_cols, param_cols, obs_index=0, known_field=None, param_index=None,
                 x0_from_obs=None, cfg: TrainConfig = None, ode: ODEConfig = None):
        self.state_cols, self.param_cols = list(state_cols), list(param_cols)
        self.obs_index = obs_index
        self.known_field, self.param_index = known_field, param_index
        self.cfg = cfg or TrainConfig()
        self.ode = ode or ODEConfig()
        self.model = None
        self.train_time_s = None

    def fit(self, train_df, val_df, seed, target="u_clean"):
        set_seed(seed)
        dev = self.cfg.device
        _, t_tr, x0_tr, p_tr, y_tr = frame_to_arrays(train_df, self.state_cols, self.param_cols, target)
        _, t_va, x0_va, p_va, y_va = frame_to_arrays(val_df, self.state_cols, self.param_cols, target)
        self.model = NeuralODE(len(self.state_cols), len(self.param_cols), self.cfg.hidden, self.cfg.activation,
                               self.known_field, self.param_index).to(dev)
        # input standardisation from training data (state ~ target values, params, t)
        in_mean = np.concatenate([[y_tr.mean()] * len(self.state_cols), p_tr.mean(0), [t_tr.mean()]])
        in_std = np.concatenate([[y_tr.std() + 1e-12] * len(self.state_cols), p_tr.std(0) + 1e-12, [t_tr.std() + 1e-12]])
        self.model.in_mean.copy_(torch.tensor(in_mean, dtype=torch.float32))
        self.model.in_std.copy_(torch.tensor(in_std, dtype=torch.float32))
        self.y_mean, self.y_std = float(y_tr.mean()), float(y_tr.std() + 1e-12)
        T = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)
        t_tr_t, t_va_t = T(t_tr), T(t_va)
        x0_tr_t, p_tr_t, y_tr_t = T(x0_tr), T(p_tr), T(y_tr)
        x0_va_t, p_va_t, y_va_t = T(x0_va), T(p_va), T(y_va)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        n = x0_tr.shape[0]
        rng = np.random.default_rng(seed)
        best_val, best_state, wait = float("inf"), None, 0
        t0 = time.perf_counter()
        self.epochs_run = 0
        for epoch in range(self.cfg.epochs):
            self.model.train()
            perm = rng.permutation(n)
            for start in range(0, n, self.ode.batch_pairs):
                idx = torch.tensor(perm[start:start + self.ode.batch_pairs], device=dev)
                opt.zero_grad()
                pred = self.model.integrate(x0_tr_t[idx], p_tr_t[idx], t_tr_t, self.ode)[:, :, self.obs_index]
                loss = torch.mean(((pred - y_tr_t[:, idx]) / self.y_std) ** 2)
                loss.backward()
                opt.step()
            self.model.eval()
            with torch.no_grad():
                pv = self.model.integrate(x0_va_t, p_va_t, t_va_t, self.ode)[:, :, self.obs_index]
                val = torch.mean(((pv - y_va_t) / self.y_std) ** 2).item()
            self.epochs_run = epoch + 1
            if val < best_val:
                best_val, wait = val, 0
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                wait += 1
                if wait >= self.cfg.patience:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.train_time_s = time.perf_counter() - t0
        self.best_val = best_val
        return self

    @property
    def n_params(self):
        return sum(p.numel() for p in self.model.net.parameters())

    @property
    def epochs_run(self):
        return getattr(self, "_epochs_run", np.nan)

    @epochs_run.setter
    def epochs_run(self, v):
        self._epochs_run = v

    def predict_traj(self, x0, params, t):
        """x0: (B, d) array, params: (B, p) array, t: 1-D array starting at 0 -> (T, B) observable."""
        dev = self.cfg.device
        self.model.eval()
        with torch.no_grad():
            out = self.model.integrate(torch.tensor(np.asarray(x0), dtype=torch.float32, device=dev),
                                       torch.tensor(np.asarray(params), dtype=torch.float32, device=dev),
                                       torch.tensor(np.asarray(t), dtype=torch.float32, device=dev), self.ode)
        return out[:, :, self.obs_index].cpu().numpy()

    def predict_frame(self, df):
        """Long-form prediction on the frame's own grid (must be a common grid per pair)."""
        pairs, t, x0, p, _ = frame_to_arrays(df, self.state_cols, self.param_cols, df.columns[-1])
        # x0 is the state at t = 0; if the frame's grid does not start at 0 (e.g. the half-step
        # shifted grid of run_offgrid_time.py) integrate from 0 and drop that point, exactly as
        # pred_fn_factory does. Without this, odeint would treat x0 as the state at t[0].
        t0_missing = float(t[0]) > 0.0
        tt = np.concatenate([[0.0], t]) if t0_missing else t
        Y = self.predict_traj(x0, p, tt)
        if t0_missing:
            Y = Y[1:]
        pred = np.empty(len(df))
        for j, pid in enumerate(pairs):
            mask = (df.pair_id == pid).to_numpy()
            order = np.argsort(df.loc[mask, "t"].to_numpy())
            vals = np.empty(mask.sum()); vals[order] = Y[:, j]
            pred[mask] = vals
        return pred

    def pred_fn_factory(self, group):
        row = group.iloc[0]
        x0 = np.array([[float(row[c]) for c in self.state_cols]])
        p = np.array([[float(row[c]) for c in self.param_cols]])

        def fn(t):
            t = np.asarray(t, float).reshape(-1)
            tt = np.concatenate([[0.0], t]) if t[0] > 0 else t
            y = self.predict_traj(x0, p, tt)[:, 0]
            return y[1:] if t[0] > 0 else y
        return fn
