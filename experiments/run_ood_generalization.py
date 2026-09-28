"""Revised paper Table 16 / Figure 6 -- out-of-distribution generalization (Table 15 regimes);
unified metrics, S = 10 seeds.  Regimes with no true crossing report E_cross as '--' and the
number of spurious surrogate crossings.

Run:  python experiments/run_ood_generalization.py [--quick]
"""
from __future__ import annotations

import pandas as pd

from _torch_common import (build_splits, config_dict, evaluate, fit_direct, fit_residual, resolve,
                           seed_of, torch_parser)
from fisher_surrogate import latex
from fisher_surrogate.common import OOD_DATA_SEED_BASE, out_dir, save_results
from fisher_surrogate.data import OOD_REGIMES, make_ood_dataset
from fisher_surrogate.metrics import summarize_seeds


def main():
    p = torch_parser(__doc__)
    p.add_argument("--n-ood-pairs", type=int, default=120)
    args = p.parse_args()
    n_seeds, dcfg, tcfg, mkw = resolve(args)
    n_ood = 12 if args.quick else args.n_ood_pairs
    out = out_dir(args, "ood_generalization")
    train_df, val_df, _ = build_splits(dcfg)
    rows = []
    for i in range(n_seeds):
        s = seed_of(i)
        print(f"seed {i:02d}", flush=True)
        direct = fit_direct(train_df, val_df, s, tcfg)
        resid = fit_residual(train_df, val_df, s, tcfg)
        for regime in OOD_REGIMES:
            ood = make_ood_dataset(regime, n_ood, dcfg.n_time, seed=OOD_DATA_SEED_BASE + i)
            rows.append({"regime": regime, "model": "Direct MLP", "seed": i, **evaluate(direct, ood, mkw)[0]})
            rows.append({"regime": regime, "model": "Residual MLP", "seed": i, **evaluate(resid, ood, mkw)[0]})
    seed_df = pd.DataFrame(rows)
    summary = summarize_seeds(seed_df, ["regime", "model"])
    tex = latex.table_ci(summary, ["regime", "model"], "Out-of-distribution generalization "
                         f"(mean $\\pm$ 95\\% CI over $S={n_seeds}$ seeds).", "tab:ood_generalization")
    save_results(out, "ood_generalization", seed_df, summary, tex, config_dict(args, dcfg, tcfg, n_seeds, mkw))


if __name__ == "__main__":
    main()
