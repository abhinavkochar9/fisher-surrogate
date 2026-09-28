"""Revised paper Table 7 (residual-parameterization ablation) and Table 9
(physics-informed residual variants), mean +/- 95% CI over S = 10 seeds, UNIFIED metrics.

Models: Direct MLP, unscaled residual u_OU + r, scaled residual u_OU + eps r (proposed),
PI-Residual with lambda in {0.01, 0.1, 1.0}.

Run:  python experiments/run_physics_informed.py [--quick] [--lambdas 0.01 0.1 1.0]
"""
from __future__ import annotations

import pandas as pd

from _torch_common import (build_splits, config_dict, evaluate, fit_direct, fit_residual, resolve,
                           seed_of, torch_parser)
from fisher_surrogate import latex
from fisher_surrogate.common import PI_SEED_OFFSET, out_dir, save_results
from fisher_surrogate.data import xy_direct
from fisher_surrogate.metrics import summarize_seeds
from fisher_surrogate.models.torch_models import (n_params, predict_physics_informed_residual,
                                                  train_physics_informed_residual)


class PIFit:
    """Adapter so a physics-informed residual model can be evaluated like an MLPFit."""

    def __init__(self, model, xs, rs, device):
        self.model, self.xs, self.rs, self.device = model, xs, rs, device
        self.train_time_s = model.train_time_s
        self.epochs_run = model.epochs_run
        self.n_params = n_params(model)

    def predict_frame(self, df, group=None):
        return predict_physics_informed_residual(self.model, df[["u0", "eps", "t"]].to_numpy(float),
                                                 self.xs, self.rs, self.device)

    def pred_fn_factory(self, group):
        row = group.iloc[0]

        def fn(t):
            import numpy as np
            t = np.asarray(t, float).reshape(-1)
            g = pd.DataFrame({"u0": float(row["u0"]), "eps": float(row["eps"]), "t": t})
            return self.predict_frame(g)
        return fn


def fit_pi(train_df, val_df, lam, seed, tcfg):
    x_tr, u_tr = xy_direct(train_df)
    x_va, u_va = xy_direct(val_df)
    model, xs, rs = train_physics_informed_residual(x_tr, u_tr, x_va, u_va, lam,
                                                    seed + int(lam * 10000) + PI_SEED_OFFSET, tcfg)
    return PIFit(model, xs, rs, tcfg.device)


def main():
    p = torch_parser(__doc__)
    p.add_argument("--lambdas", type=float, nargs="*", default=[0.01, 0.1, 1.0])
    args = p.parse_args()
    n_seeds, dcfg, tcfg, mkw = resolve(args)
    out = out_dir(args, "physics_informed")
    train_df, val_df, test_df = build_splits(dcfg)

    rows = []
    for i in range(n_seeds):
        s = seed_of(i)
        print(f"seed {i:02d}", flush=True)
        rows.append({"model": "Direct MLP", "seed": i, **evaluate(fit_direct(train_df, val_df, s, tcfg), test_df, mkw)[0]})
        rows.append({"model": "Unscaled residual MLP", "seed": i,
                     **evaluate(fit_residual(train_df, val_df, s, tcfg, scaled=False), test_df, mkw)[0]})
        rows.append({"model": "Scaled residual MLP (proposed)", "seed": i,
                     **evaluate(fit_residual(train_df, val_df, s, tcfg), test_df, mkw)[0]})
        for lam in args.lambdas:
            rows.append({"model": f"PI-Residual, $\\lambda={lam}$", "seed": i,
                         **evaluate(fit_pi(train_df, val_df, lam, s, tcfg), test_df, mkw)[0]})
    seed_df = pd.DataFrame(rows)
    summary = summarize_seeds(seed_df, ["model"])
    order = ["Direct MLP", "Unscaled residual MLP", "Scaled residual MLP (proposed)"] + \
            [f"PI-Residual, $\\lambda={lam}$" for lam in args.lambdas]
    tex = latex.table_ci(summary, ["model"], "Residual-parameterization ablation and physics-informed variants "
                         f"(mean $\\pm$ 95\\% CI over $S={n_seeds}$ seeds; unified metric definitions).",
                         "tab:residual_ablation", order=order)
    save_results(out, "physics_informed", seed_df, summary, tex, config_dict(args, dcfg, tcfg, n_seeds, mkw))


if __name__ == "__main__":
    main()
