"""Revised paper Table 12 -- DeepONet-style branch-trunk baseline (auxiliary architecture
comparison) vs Direct / Residual / PI-Residual (lambda = 0.1); unified metrics.

Run:  python experiments/run_deeponet.py [--quick] [--p 64]
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from _torch_common import (build_splits, config_dict, evaluate, fit_direct, fit_residual, resolve,
                           seed_of, torch_parser)
from fisher_surrogate import latex
from fisher_surrogate.common import DEEPONET_SEED_OFFSET, out_dir, save_results
from fisher_surrogate.data import xy_direct
from fisher_surrogate.metrics import summarize_seeds
from fisher_surrogate.models.torch_models import n_params, predict_deeponet, train_deeponet
from run_physics_informed import fit_pi


class DeepONetFit:
    def __init__(self, model, bs, ts, ys, device):
        self.model, self.bs, self.ts, self.ys, self.device = model, bs, ts, ys, device
        self.train_time_s, self.epochs_run, self.n_params = model.train_time_s, model.epochs_run, n_params(model)

    def predict_frame(self, df, group=None):
        return predict_deeponet(self.model, df[["u0", "eps"]].to_numpy(float), df[["t"]].to_numpy(float),
                                self.bs, self.ts, self.ys, self.device)

    def pred_fn_factory(self, group):
        row = group.iloc[0]
        return lambda t: self.predict_frame(pd.DataFrame({"u0": float(row["u0"]), "eps": float(row["eps"]),
                                                          "t": np.asarray(t, float).reshape(-1)}))


def fit_deeponet(train_df, val_df, seed, p, tcfg):
    bt = lambda df: (df[["u0", "eps"]].to_numpy(float), df[["t"]].to_numpy(float))
    b, t = bt(train_df); bv, tv = bt(val_df)
    _, u_tr = xy_direct(train_df); _, u_va = xy_direct(val_df)
    model, bs, ts, ys = train_deeponet(b, t, u_tr, bv, tv, u_va, seed + DEEPONET_SEED_OFFSET, p, tcfg)
    return DeepONetFit(model, bs, ts, ys, tcfg.device)


def main():
    p = torch_parser(__doc__)
    p.add_argument("--p", type=int, default=64, help="DeepONet latent dimension")
    p.add_argument("--physics-lambda", type=float, default=0.1)
    args = p.parse_args()
    n_seeds, dcfg, tcfg, mkw = resolve(args)
    out = out_dir(args, "deeponet")
    train_df, val_df, test_df = build_splits(dcfg)
    rows = []
    for i in range(n_seeds):
        s = seed_of(i)
        print(f"seed {i:02d}", flush=True)
        rows.append({"model": "Direct MLP", "seed": i, **evaluate(fit_direct(train_df, val_df, s, tcfg), test_df, mkw)[0]})
        rows.append({"model": "Residual MLP", "seed": i, **evaluate(fit_residual(train_df, val_df, s, tcfg), test_df, mkw)[0]})
        rows.append({"model": f"PI-Residual, $\\lambda={args.physics_lambda}$", "seed": i,
                     **evaluate(fit_pi(train_df, val_df, args.physics_lambda, s, tcfg), test_df, mkw)[0]})
        rows.append({"model": "DeepONet-style baseline", "seed": i,
                     **evaluate(fit_deeponet(train_df, val_df, s, args.p, tcfg), test_df, mkw)[0]})
    seed_df = pd.DataFrame(rows)
    summary = summarize_seeds(seed_df, ["model"])
    tex = latex.table_ci(summary, ["model"], "DeepONet-style branch--trunk baseline "
                         f"(mean $\\pm$ 95\\% CI over $S={n_seeds}$ seeds).", "tab:deeponet_baseline")
    save_results(out, "deeponet", seed_df, summary, tex, config_dict(args, dcfg, tcfg, n_seeds, mkw))


if __name__ == "__main__":
    main()
