"""Revised paper Table 13 -- Neural ODE (two Dormand--Prince tolerance settings) and
Gaussian-process baselines with the GP crossing time located by root-finding on the
posterior mean and the number of non-crossing test trajectories reported (R4.2, R4.4).

Run:  python experiments/run_neural_ode_gp.py [--quick] [--node-seeds 5]
"""
from __future__ import annotations

import pandas as pd

from _baselines_common import NODE_TOLERANCES, fit_gp, fit_neural_ode
from _torch_common import (build_splits, config_dict, evaluate, fit_direct, fit_residual, resolve,
                           seed_of, torch_parser)
from fisher_surrogate import latex
from fisher_surrogate.common import out_dir, save_results
from fisher_surrogate.metrics import summarize_seeds


def main():
    p = torch_parser(__doc__)
    p.add_argument("--node-seeds", type=int, default=5)
    p.add_argument("--gp-sub", type=int, default=2000)
    p.add_argument("--gp-seeds", type=int, default=10)
    args = p.parse_args()
    n_seeds, dcfg, tcfg, mkw = resolve(args)
    node_seeds = 1 if args.quick else args.node_seeds
    gp_sub = 300 if args.quick else args.gp_sub
    gp_seeds = 1 if args.quick else args.gp_seeds
    if args.quick:
        tcfg.epochs, tcfg.patience = 30, 10
    out = out_dir(args, "neural_ode_gp")
    train_df, val_df, test_df = build_splits(dcfg)
    rows = []
    for tol_name, (rtol, atol) in NODE_TOLERANCES.items():
        for i in range(node_seeds):
            s = seed_of(i)
            for name, res in ((f"Neural ODE ({tol_name}: rtol={rtol:g}, atol={atol:g})", False),
                              (f"Residual Neural ODE ({tol_name}: rtol={rtol:g}, atol={atol:g})", True)):
                print(f"seed {i:02d}: {name}", flush=True)
                node = fit_neural_ode(train_df, val_df, s, tcfg, residual=res, rtol=rtol, atol=atol)
                m, _ = evaluate(node, test_df, mkw)
                rows.append({"model": name, "seed": i, **m, "rtol": rtol, "atol": atol, "solver": node.ode.method,
                             "epochs_run": node.epochs_run})
    for i in range(gp_seeds):
        gp = fit_gp(train_df, seed_of(i), prior_mean="zero", n_sub=gp_sub)
        m, _ = evaluate(gp, test_df, mkw)
        rows.append({"model": "Gaussian process", "seed": i, **m, "gp_kernel": gp.kernel_summary(), "gp_n_sub": gp.n_train_points})
    for i in range(n_seeds):
        s = seed_of(i)
        print(f"seed {i:02d}: MLPs", flush=True)
        rows.append({"model": "Direct MLP", "seed": i, **evaluate(fit_direct(train_df, val_df, s, tcfg), test_df, mkw)[0]})
        rows.append({"model": "Residual MLP (proposed)", "seed": i, **evaluate(fit_residual(train_df, val_df, s, tcfg), test_df, mkw)[0]})
    seed_df = pd.DataFrame(rows)
    summary = summarize_seeds(seed_df, ["model"])
    tex = latex.table_ci(summary, ["model"], "Neural ODE (two solver tolerance settings) and Gaussian-process "
                         f"baselines. Neural ODE: mean $\\pm$ 95\\% CI over $S={node_seeds}$ seeds; GP: mean $\\pm$ 95\\% CI "
                         f"over $S={gp_seeds}$ subsample seeds; GP crossing time by root-finding on the posterior mean "
                         "(failures counted in the summary CSV).", "tab:neural_ode_gp")
    save_results(out, "neural_ode_gp", seed_df, summary, tex,
                 config_dict(args, dcfg, tcfg, n_seeds, mkw, node_seeds=node_seeds, tolerances=NODE_TOLERANCES,
                             gp_sub=gp_sub, gp_seeds=gp_seeds))


if __name__ == "__main__":
    main()
