"""Revised paper Table 8 -- error of the direct and scaled-residual MLPs stratified by
Fisher strength (reviewer R2.8): in-distribution bins [0.01,0.1], [0.1,0.3], [0.3,0.5],
the limit eps = 0 (residual exact by construction, direct must learn u_OU), and the OOD range
eps in [0.5, 1.0].  Models are trained once per seed on the standard split; only the test
data are stratified.

Run:  python experiments/run_eps_ablation.py [--quick]
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from _torch_common import (build_splits, config_dict, evaluate, fit_direct, fit_residual, resolve,
                           seed_of, torch_parser)
from fisher_surrogate import latex
from fisher_surrogate.common import OOD_DATA_SEED_BASE, out_dir, save_results
from fisher_surrogate.data import (EPS_BINS, EPS_OOD_RANGE, TRAIN_U0_RANGE, eps_bin_label, make_dataset,
                                   make_eps_zero_dataset)
from fisher_surrogate.metrics import summarize_seeds


def main():
    p = torch_parser(__doc__)
    p.add_argument("--n-extra-pairs", type=int, default=120, help="pairs for the eps=0 and OOD sets")
    args = p.parse_args()
    n_seeds, dcfg, tcfg, mkw = resolve(args)
    n_extra = 12 if args.quick else args.n_extra_pairs
    out = out_dir(args, "eps_ablation")
    train_df, val_df, test_df = build_splits(dcfg)
    test_df = test_df.copy()
    test_df["eps_bin"] = test_df["eps"].map(eps_bin_label)
    strata = {"$\\varepsilon = 0$ (exact)": make_eps_zero_dataset(n_extra, dcfg.n_time, seed=OOD_DATA_SEED_BASE + 41)}
    for name, lo, hi in EPS_BINS:
        sub = test_df[test_df.eps_bin == name]
        if len(sub):
            strata[name] = sub
    strata[f"[{EPS_OOD_RANGE[0]}, {EPS_OOD_RANGE[1]}] (OOD)"] = make_dataset(
        n_extra, dcfg.n_time, TRAIN_U0_RANGE, EPS_OOD_RANGE, seed=OOD_DATA_SEED_BASE + 42)
    rows = []
    for i in range(n_seeds):
        s = seed_of(i)
        print(f"seed {i:02d}", flush=True)
        direct = fit_direct(train_df, val_df, s, tcfg)
        resid = fit_residual(train_df, val_df, s, tcfg)
        for st, data in strata.items():
            rows.append({"eps_range": st, "model": "Direct MLP", "seed": i, "n_pairs": data.pair_id.nunique(),
                         **evaluate(direct, data, mkw)[0]})
            rows.append({"eps_range": st, "model": "Residual MLP", "seed": i, "n_pairs": data.pair_id.nunique(),
                         **evaluate(resid, data, mkw)[0]})
    seed_df = pd.DataFrame(rows)
    summary = summarize_seeds(seed_df, ["eps_range", "model"])
    tex = latex.table_ci(summary, ["eps_range", "model"], "Error of the direct and scaled-residual MLPs stratified by "
                         f"$\\varepsilon$ (in-distribution bins, $\\varepsilon=0$, and OOD); mean $\\pm$ 95\\% CI over $S={n_seeds}$ seeds.",
                         "tab:eps_ablation")
    save_results(out, "eps_ablation", seed_df, summary, tex, config_dict(args, dcfg, tcfg, n_seeds, mkw, n_extra=n_extra))


if __name__ == "__main__":
    main()
