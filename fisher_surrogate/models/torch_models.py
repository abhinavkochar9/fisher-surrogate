"""PyTorch surrogates used for the multi-seed comparison tables
(paper Tables 6-8, 10, 12, 13).

Common training protocol across all torch notebooks:
    (64, 64) tanh MLP, Adam, lr 2e-3, batch 512, up to 1500 epochs,
    early stopping on validation MSE with patience 150, best-state restore,
    z-score scaling of inputs and targets fitted on the training split.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TrainConfig:
    hidden: tuple = (64, 64)
    activation: str = "tanh"
    epochs: int = 1500
    batch_size: int = 512
    lr: float = 2e-3
    patience: int = 150
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class Scaler:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x):
        mean = x.mean(axis=0, keepdims=True)
        std = x.std(axis=0, keepdims=True)
        return cls(mean, np.where(std < 1e-12, 1.0, std))

    def transform(self, x):
        return (x - self.mean) / self.std

    def inverse(self, x):
        return x * self.std + self.mean

    def to_torch(self, device):
        return (torch.tensor(self.mean, dtype=torch.float32, device=device),
                torch.tensor(self.std, dtype=torch.float32, device=device))


def _act(name):
    return {"relu": nn.ReLU, "gelu": nn.GELU}.get(name, nn.Tanh)


def _mlp(input_dim, hidden, output_dim, activation):
    layers, prev = [], input_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), _act(activation)()]
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


class MLP(nn.Module):
    def __init__(self, input_dim=3, hidden=(64, 64), activation="tanh"):
        super().__init__()
        self.net = _mlp(input_dim, hidden, 1, activation)

    def forward(self, x):
        return self.net(x)


class DeepONet(nn.Module):
    """Lightweight branch-trunk model: branch(u0, eps) . trunk(t) + bias (paper Sec. 6.3)."""

    def __init__(self, branch_dim=2, trunk_dim=1, p=64, hidden=(64, 64), activation="tanh"):
        super().__init__()
        self.branch = _mlp(branch_dim, hidden, p, activation)
        self.trunk = _mlp(trunk_dim, hidden, p, activation)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, b, t):
        return torch.sum(self.branch(b) * self.trunk(t), dim=1, keepdim=True) + self.bias


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def _early_stop_loop(model, opt, loader, step_fn, val_fn, cfg: TrainConfig):
    """Adam + early stopping on validation MSE (patience ``cfg.patience``), best-state restore.
    Records ``model.train_time_s``, ``model.epochs_run`` and ``model.best_val`` for the
    specification table (revision item R2.6)."""
    best_state, best_val, wait = None, float("inf"), 0
    t0 = time.perf_counter()
    epochs_run = 0
    for _ in range(cfg.epochs):
        epochs_run += 1
        model.train()
        for batch in loader:
            opt.zero_grad()
            loss = step_fn(batch)
            loss.backward()
            opt.step()
        model.eval()
        val_loss = val_fn()
        if val_loss < best_val:
            best_val, wait = val_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= cfg.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.train_time_s = time.perf_counter() - t0
    model.epochs_run = epochs_run
    model.best_val = best_val
    return model


# ----------------------------------------------------------------------------
# Direct / residual MLP (target already transformed by the caller)
# ----------------------------------------------------------------------------

def train_mlp(x_train, y_train, x_val, y_val, seed, cfg: TrainConfig = TrainConfig()):
    """Train a scaled MLP regressor. Returns (model, x_scaler, y_scaler).

    For the residual surrogates the caller passes the residual target
    r* = (u - u_OU)/eps (see ``data.xy_residual``) and maps back with
    ``data.residual_to_u``.
    """
    set_seed(seed)
    dev = cfg.device
    xs, ys = Scaler.fit(x_train), Scaler.fit(y_train)
    xt = torch.tensor(xs.transform(x_train), dtype=torch.float32)
    yt = torch.tensor(ys.transform(y_train), dtype=torch.float32)
    xv = torch.tensor(xs.transform(x_val), dtype=torch.float32, device=dev)
    yv = torch.tensor(ys.transform(y_val), dtype=torch.float32, device=dev)
    loader = DataLoader(TensorDataset(xt, yt), batch_size=cfg.batch_size, shuffle=True)
    model = MLP(x_train.shape[1], cfg.hidden, cfg.activation).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    mse = nn.MSELoss()

    def step(batch):
        xb, yb = (b.to(dev) for b in batch)
        return mse(model(xb), yb)

    def val():
        with torch.no_grad():
            return mse(model(xv), yv).item()

    _early_stop_loop(model, opt, loader, step, val, cfg)
    return model, xs, ys


def predict_mlp(model, x, xs, ys, device="cpu"):
    model.eval()
    with torch.no_grad():
        p = model(torch.tensor(xs.transform(x), dtype=torch.float32, device=device)).cpu().numpy()
    return ys.inverse(p).reshape(-1)


# ----------------------------------------------------------------------------
# Physics-informed residual MLP (paper Sec. 4.4, Eq. 18)
# ----------------------------------------------------------------------------

def _u_ou_t(t, u0):
    return 1.0 + (u0 - 1.0) * torch.exp(-2.0 * t)


def _residual_forward(model, x_raw, x_mean, x_std, r_mean, r_std):
    u0, eps, t = x_raw[:, 0:1], x_raw[:, 1:2], x_raw[:, 2:3]
    r = model((x_raw - x_mean) / x_std) * r_std + r_mean
    return _u_ou_t(t, u0) + eps * r


def ode_residual(u_hat, x_raw, eps):
    """R = du_hat/dt - [2(1 - u_hat) + eps/u_hat] via autograd w.r.t. the raw time input."""
    du_dt = torch.autograd.grad(u_hat.sum(), x_raw, create_graph=True, retain_graph=True)[0][:, 2:3]
    u_safe = torch.clamp(u_hat, min=1e-6)
    return du_dt - (2.0 * (1.0 - u_safe) + eps / u_safe)


def train_physics_informed_residual(x_train, u_train, x_val, u_val, lam, seed,
                                    cfg: TrainConfig = TrainConfig()):
    """Residual surrogate trained on L = L_traj + lam * L_phys. Returns (model, x_scaler, r_scaler)."""
    set_seed(seed)
    dev = cfg.device
    r_train = (u_train.reshape(-1) - (1.0 + (x_train[:, 0] - 1.0) * np.exp(-2.0 * x_train[:, 2]))) / x_train[:, 1]
    xs, rs = Scaler.fit(x_train), Scaler.fit(r_train.reshape(-1, 1))
    x_mean, x_std = xs.to_torch(dev)
    r_mean, r_std = rs.to_torch(dev)
    loader = DataLoader(TensorDataset(torch.tensor(x_train, dtype=torch.float32),
                                      torch.tensor(u_train, dtype=torch.float32)),
                        batch_size=cfg.batch_size, shuffle=True)
    xv = torch.tensor(x_val, dtype=torch.float32, device=dev)
    uv = torch.tensor(u_val, dtype=torch.float32, device=dev)
    model = MLP(3, cfg.hidden, cfg.activation).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    def step(batch):
        xb, ub = (b.to(dev) for b in batch)
        xb = xb.clone().detach().requires_grad_(True)
        u_hat = _residual_forward(model, xb, x_mean, x_std, r_mean, r_std)
        loss_traj = torch.mean((u_hat - ub) ** 2)
        if lam > 0:
            loss_phys = torch.mean(ode_residual(u_hat, xb, xb[:, 1:2]) ** 2)
            return loss_traj + lam * loss_phys
        return loss_traj

    def val():
        with torch.no_grad():
            return torch.mean((_residual_forward(model, xv, x_mean, x_std, r_mean, r_std) - uv) ** 2).item()

    _early_stop_loop(model, opt, loader, step, val, cfg)
    return model, xs, rs


def predict_physics_informed_residual(model, x, xs, rs, device="cpu", with_ode_residual=False):
    model.eval()
    x_mean, x_std = xs.to_torch(device)
    r_mean, r_std = rs.to_torch(device)
    xr = torch.tensor(x, dtype=torch.float32, device=device).requires_grad_(with_ode_residual)
    if with_ode_residual:
        u_hat = _residual_forward(model, xr, x_mean, x_std, r_mean, r_std)
        res = ode_residual(u_hat, xr, xr[:, 1:2])
        return u_hat.detach().cpu().numpy().reshape(-1), res.detach().cpu().numpy().reshape(-1)
    with torch.no_grad():
        u_hat = _residual_forward(model, xr, x_mean, x_std, r_mean, r_std)
    return u_hat.cpu().numpy().reshape(-1)


# ----------------------------------------------------------------------------
# DeepONet-style baseline
# ----------------------------------------------------------------------------

def train_deeponet(b_train, t_train, y_train, b_val, t_val, y_val, seed, p=64,
                   cfg: TrainConfig = TrainConfig()):
    set_seed(seed)
    dev = cfg.device
    bs, ts, ys = Scaler.fit(b_train), Scaler.fit(t_train), Scaler.fit(y_train)
    tens = lambda s, a: torch.tensor(s.transform(a), dtype=torch.float32)
    loader = DataLoader(TensorDataset(tens(bs, b_train), tens(ts, t_train), tens(ys, y_train)),
                        batch_size=cfg.batch_size, shuffle=True)
    bv, tv, yv = (tens(s, a).to(dev) for s, a in ((bs, b_val), (ts, t_val), (ys, y_val)))
    model = DeepONet(p=p, hidden=cfg.hidden, activation=cfg.activation).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    mse = nn.MSELoss()

    def step(batch):
        bb, tt, yy = (b.to(dev) for b in batch)
        return mse(model(bb, tt), yy)

    def val():
        with torch.no_grad():
            return mse(model(bv, tv), yv).item()

    _early_stop_loop(model, opt, loader, step, val, cfg)
    return model, bs, ts, ys


def predict_deeponet(model, b, t, bs, ts, ys, device="cpu"):
    model.eval()
    with torch.no_grad():
        p = model(torch.tensor(bs.transform(b), dtype=torch.float32, device=device),
                  torch.tensor(ts.transform(t), dtype=torch.float32, device=device)).cpu().numpy()
    return ys.inverse(p).reshape(-1)


# ----------------------------------------------------------------------------
# Residual MLP with mechanism-level terms in the training loss (revised Sec. 6.8,
# reviewer R2.7): L = L_traj + beta * (E_eq + E_cross)
# ----------------------------------------------------------------------------

def _soft_crossing_time(u_hat, t, level=1.0):
    """Differentiable first-upward-crossing time on a grid (B, T): the bracketing index is
    found on the detached prediction, the crossing time by linear interpolation in the
    bracketing values (differentiable).  Returns (T_hat, mask) with mask=1 where a crossing
    exists; T_hat = t[-1] (constant) where it does not."""
    with torch.no_grad():
        above = (u_hat >= level)
        above[:, 0] = False
        has = above.any(dim=1)
        k = torch.argmax(above.int(), dim=1)
        k = torch.where(has, k, torch.ones_like(k))
    idx = torch.arange(u_hat.shape[0], device=u_hat.device)
    a, b = u_hat[idx, k - 1], u_hat[idx, k]
    t0, t1 = t[k - 1], t[k]
    denom = torch.where((b - a).abs() < 1e-12, torch.ones_like(b), b - a)
    t_hat = t0 + (level - a) * (t1 - t0) / denom
    t_hat = torch.where(has, t_hat, torch.full_like(t_hat, float(t[-1])))
    return t_hat, has.float()


def train_residual_mechanism_loss(X, U, T_true, u_eq, t_grid, x_val, u_val, beta, seed,
                                  cfg: TrainConfig = TrainConfig(), batch_pairs=64):
    """X: (N, T, 3) features per trajectory, U: (N, T) targets, T_true: (N,) true crossing
    times (nan if none), u_eq: (N,) equilibria.  Validation on the plain trajectory MSE.
    Returns (model, x_scaler, r_scaler)."""
    set_seed(seed)
    dev = cfg.device
    N, T, _ = X.shape
    x_flat = X.reshape(-1, 3)
    r_flat = (U.reshape(-1) - (1.0 + (x_flat[:, 0] - 1.0) * np.exp(-2.0 * x_flat[:, 2]))) / x_flat[:, 1]
    xs, rs = Scaler.fit(x_flat), Scaler.fit(r_flat.reshape(-1, 1))
    x_mean, x_std = xs.to_torch(dev)
    r_mean, r_std = rs.to_torch(dev)
    Xt = torch.tensor(X, dtype=torch.float32, device=dev)
    Ut = torch.tensor(U, dtype=torch.float32, device=dev)
    Tt = torch.tensor(np.nan_to_num(T_true, nan=0.0), dtype=torch.float32, device=dev)
    has_true = torch.tensor(~np.isnan(T_true), dtype=torch.float32, device=dev)
    ueq = torch.tensor(u_eq, dtype=torch.float32, device=dev)
    tg = torch.tensor(t_grid, dtype=torch.float32, device=dev)
    xv = torch.tensor(x_val, dtype=torch.float32, device=dev)
    uv = torch.tensor(u_val, dtype=torch.float32, device=dev)
    model = MLP(3, cfg.hidden, cfg.activation).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loader = DataLoader(TensorDataset(torch.arange(N)), batch_size=batch_pairs, shuffle=True)

    def step(batch):
        idx = batch[0].to(dev)
        xb = Xt[idx].reshape(-1, 3)
        u_hat = _residual_forward(model, xb, x_mean, x_std, r_mean, r_std).reshape(len(idx), T)
        loss = torch.mean((u_hat - Ut[idx]) ** 2)
        if beta > 0:
            e_eq = torch.mean(torch.abs(u_hat[:, -1] - ueq[idx]))
            t_hat, _ = _soft_crossing_time(u_hat, tg)
            e_cross = torch.sum(torch.abs(t_hat - Tt[idx]) * has_true[idx]) / torch.clamp(has_true[idx].sum(), min=1.0)
            loss = loss + beta * (e_eq + e_cross)
        return loss

    def val():
        with torch.no_grad():
            return torch.mean((_residual_forward(model, xv, x_mean, x_std, r_mean, r_std) - uv) ** 2).item()

    _early_stop_loop(model, opt, loader, step, val, cfg)
    return model, xs, rs
