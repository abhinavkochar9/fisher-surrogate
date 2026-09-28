"""Revised paper Table 20 / Figure 9 -- transfer to the non-normal linear system of Eq. (24)
with a numerically tabulated baseline; extended (R2.4/R4.3) with the direct MLP + y_base
input and the residual Neural ODE.

    Direct MLP                (c, gamma, t) -> x1
    Direct MLP + y_base input (c, gamma, t, y_base) -> x1
    Residual MLP              y_base(t; c) + gamma r_theta(c, gamma, t)
    Residual Neural ODE       x' = A_0 x + b + gamma g_phi(x, gamma, t), observable x1

Metrics use the numerically located equilibrium x1* = 1 + gamma/2 as both the crossing level
and the attractor, and the true interior peak for E_over (a genuine overshoot here).

Run:  python experiments/run_transfer_system.py [--quick] [--n-pairs 400]
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from _baselines_common import NODE_SEED_OFFSET
from _torch_common import config_dict, evaluate, fit_direct, fit_residual, resolve, seed_of, torch_parser
from fisher_surrogate import latex, transfer
from fisher_surrogate.common import out_dir, save_results
from fisher_surrogate.data import split_pair_ids
from fisher_surrogate.metrics import summarize_seeds
from fisher_surrogate.models.neural_ode import HAVE_TORCHDIFFEQ, NeuralODESurrogate, ODEConfig

TF = transfer.FEATURES  # (c, gamma, t)
TRANSFER_KW = dict(true_col="y_clean", level_col="level", u_eq_col="y_eq", t_cross_col="T_cross_true")


def transfer_known_field(x, p, t):
    """A_0 x + b with A_0 = diag(-1, -2), b = (1, 1)."""
    return torch.stack([-x[:, 0] + 1.0, -2.0 * x[:, 1] + 1.0], dim=1)


def prepare(df):
    df = df.copy()
    df["u_base"] = df["y_base"]   # generic baseline column name used by MLPFit
    df["x2_0"] = transfer.X2_0
    return df


def fit_transfer_node(train_df, val_df, seed, tcfg, rtol=1e-6, atol=1e-8):
    ode = ODEConfig(rtol=rtol, atol=atol, method="dopri5" if HAVE_TORCHDIFFEQ else "rk4")
    node = NeuralODESurrogate(["c", "x2_0"], ["gamma"], obs_index=0, known_field=transfer_known_field,
                              param_index=0, cfg=tcfg, ode=ode)
    return node.fit(train_df, val_df, seed + NODE_SEED_OFFSET + 7, target="y_clean")


def split(df, n_pairs, seed=999):
    tr, va, te = split_pair_ids(n_pairs, 0.70, 0.15, seed)
    return (df[df.pair_id.isin(tr)].copy(), df[df.pair_id.isin(va)].copy(), df[df.pair_id.isin(te)].copy())


def main():
    p = torch_parser(__doc__)
    p.set_defaults(n_pairs=400)
    p.add_argument("--node-seeds", type=int, default=5)
    p.add_argument("--skip-node", action="store_true")
    args = p.parse_args()
    n_seeds, dcfg, tcfg, mkw = resolve(args)
    node_seeds = 1 if args.quick else args.node_seeds
    if args.quick:
        tcfg.epochs, tcfg.patience = 30, 10
    out = out_dir(args, "transfer_system")
    df = prepare(transfer.make_dataset(dcfg.n_pairs, dcfg.n_time))
    train_df, val_df, test_df = split(df, dcfg.n_pairs)
    stats = transfer.overshoot_stats(test_df)
    print("held-out overshoot statistics:", stats)

    rows, preds = [], {}
    for i in range(n_seeds):
        s = seed_of(i)
        print(f"seed {i:02d}", flush=True)
        fits = {
            "Direct MLP": fit_direct(train_df, val_df, s, tcfg, target="y_clean", features=TF, base=("column",)),
            "Direct MLP + $y_{\\mathrm{base}}$ input": fit_direct(train_df, val_df, s + 100, tcfg, target="y_clean",
                                                                  features=TF + ["u_base"], base=("column",)),
            "Residual MLP": fit_residual(train_df, val_df, s, tcfg, target="y_clean", features=TF, base=("column",),
                                         base_col="u_base", scale_col="gamma"),
        }
        for name, fit in fits.items():
            m, pred = evaluate(fit, test_df, mkw, **TRANSFER_KW)
            rows.append({"model": name, "seed": i, **m})
            if i == 0:
                preds[name] = pred["u_pred"].to_numpy()
    if not args.skip_node:
        for i in range(node_seeds):
            print(f"seed {i:02d}: residual Neural ODE", flush=True)
            node = fit_transfer_node(train_df, val_df, seed_of(i), tcfg)
            m, pred = evaluate(node, test_df, mkw, **TRANSFER_KW)
            rows.append({"model": "Residual Neural ODE", "seed": i, **m, "rtol": node.ode.rtol, "atol": node.ode.atol})
            if i == 0:
                preds["Residual Neural ODE"] = pred["u_pred"].to_numpy()
    seed_df = pd.DataFrame(rows)
    summary = summarize_seeds(seed_df, ["model"])
    tex = latex.table_ci(summary, ["model"], "Transfer to the non-normal system of Eq.~(24) with a numerically "
                         f"tabulated baseline (mean $\\pm$ 95\\% CI over $S={n_seeds}$ seeds; $S={node_seeds}$ for the "
                         "residual Neural ODE).", "tab:transfer_system")
    fig = test_df[["pair_id", "c", "gamma", "t", "y_clean", "y_base", "level", "T_cross_true"]].copy()
    for name, v in preds.items():
        fig[name] = v
    fig.to_csv(out / "transfer_predictions_seed0.csv", index=False)
    save_results(out, "transfer_system", seed_df, summary, tex,
                 config_dict(args, dcfg, tcfg, n_seeds, mkw, overshoot_stats=stats, node_seeds=node_seeds))


if __name__ == "__main__":
    main()
