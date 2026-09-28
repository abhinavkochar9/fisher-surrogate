"""Revised paper Table 10 -- feature-matched baselines: every competitor receives the
unregularized OU baseline u_OU (reviewers R2.4, R4.3, R5.2).

    Direct MLP (no baseline)                 (u0, eps, t) -> u
    Direct MLP + u_OU input                  (u0, eps, t, u_OU) -> u
    Unscaled residual (difference model)     u_OU + r_theta
    Scaled residual MLP (proposed)           u_OU + eps r_theta
    GP, zero mean                            RBF-ARD + white noise on (u0, eps, t)
    GP, u_OU prior mean                      Kennedy--O'Hagan discrepancy GP
    Neural ODE (full field)                  du/dt = g_phi(u, eps, t)
    Residual Neural ODE (UDE)                du/dt = 2(1 - u) + eps g_phi(u, eps, t)

Mean +/- 95% CI over S = 10 seeds (S = 5 for the Neural ODE variants by default); the GP is
deterministic given the training subsample (its seed changes the subsample only).

Run:  python experiments/run_feature_matched.py [--quick] [--node-seeds 5] [--gp-sub 2000]
"""
from __future__ import annotations

import pandas as pd

from _baselines_common import NODE_TOLERANCES, fit_gp, fit_neural_ode
from _torch_common import (FEATURES, build_splits, config_dict, evaluate, fit_direct, fit_residual,
                           resolve, seed_of, torch_parser)
from fisher_surrogate import latex
from fisher_surrogate.common import out_dir, save_results
from fisher_surrogate.data import add_baseline_column
from fisher_surrogate.metrics import summarize_seeds

ORDER = ["Direct MLP (no baseline)", "Direct MLP + $u_{\\mathrm{OU}}$ input", "Unscaled residual (difference model)",
         "Scaled residual MLP (proposed)", "GP, zero mean", "GP, $u_{\\mathrm{OU}}$ prior mean",
         "Neural ODE (full field)", "Residual Neural ODE"]


def main():
    p = torch_parser(__doc__)
    p.add_argument("--node-seeds", type=int, default=5)
    p.add_argument("--node-tol", default="standard", choices=list(NODE_TOLERANCES))
    p.add_argument("--gp-sub", type=int, default=2000)
    p.add_argument("--gp-seeds", type=int, default=10)
    p.add_argument("--skip-node", action="store_true")
    args = p.parse_args()
    n_seeds, dcfg, tcfg, mkw = resolve(args)
    node_seeds = 1 if args.quick else args.node_seeds
    gp_sub = 300 if args.quick else args.gp_sub
    gp_seeds = 1 if args.quick else args.gp_seeds
    if args.quick:
        tcfg.epochs, tcfg.patience = 30, 10
    out = out_dir(args, "feature_matched")
    train_df, val_df, test_df = (add_baseline_column(d) for d in build_splits(dcfg))
    feats_b = FEATURES + ["u_base"]
    rtol, atol = NODE_TOLERANCES[args.node_tol]

    rows = []
    for i in range(n_seeds):
        s = seed_of(i)
        print(f"seed {i:02d}: MLP family", flush=True)
        rows.append({"model": ORDER[0], "seed": i, **evaluate(fit_direct(train_df, val_df, s, tcfg), test_df, mkw)[0]})
        rows.append({"model": ORDER[1], "seed": i,
                     **evaluate(fit_direct(train_df, val_df, s + 100, tcfg, features=feats_b), test_df, mkw)[0]})
        rows.append({"model": ORDER[2], "seed": i, **evaluate(fit_residual(train_df, val_df, s, tcfg, scaled=False), test_df, mkw)[0]})
        rows.append({"model": ORDER[3], "seed": i, **evaluate(fit_residual(train_df, val_df, s, tcfg), test_df, mkw)[0]})
    for i in range(gp_seeds):
        print(f"GP subsample seed {i:02d}", flush=True)
        for name, pm in ((ORDER[4], "zero"), (ORDER[5], "ou")):
            gp = fit_gp(train_df, seed_of(i), prior_mean=pm, n_sub=gp_sub)
            m, _ = evaluate(gp, test_df, mkw)
            rows.append({"model": name, "seed": i, **m, "gp_kernel": gp.kernel_summary(), "gp_n_sub": gp.n_train_points})
    if not args.skip_node:
        for i in range(node_seeds):
            s = seed_of(i)
            for name, res in ((ORDER[6], False), (ORDER[7], True)):
                print(f"seed {i:02d}: {name}", flush=True)
                node = fit_neural_ode(train_df, val_df, s, tcfg, residual=res, rtol=rtol, atol=atol)
                m, _ = evaluate(node, test_df, mkw)
                rows.append({"model": name, "seed": i, **m, "rtol": rtol, "atol": atol, "solver": node.ode.method})
    seed_df = pd.DataFrame(rows)
    summary = summarize_seeds(seed_df, ["model"])
    order = [o for o in ORDER if o in set(summary["model"])]
    tex = latex.table_ci(summary, ["model"], "Feature-matched baselines: every model receives the unregularized OU "
                         f"baseline. Mean $\\pm$ 95\\% CI over $S={n_seeds}$ seeds ($S={node_seeds}$ for Neural ODE "
                         f"variants; $S={gp_seeds}$ subsample seeds for the GP). Missing-crossing counts are reported in "
                         "the summary CSV.", "tab:feature_matched", order=order)
    save_results(out, "feature_matched", seed_df, summary, tex,
                 config_dict(args, dcfg, tcfg, n_seeds, mkw, node_seeds=node_seeds, node_tol=(rtol, atol),
                             gp_sub=gp_sub, gp_seeds=gp_seeds))


if __name__ == "__main__":
    main()
