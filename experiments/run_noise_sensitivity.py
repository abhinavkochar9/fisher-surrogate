"""Revised paper Table 14 / Figure 5 -- noise sensitivity (noise on training targets only);
unified metrics, S = 10 seeds.

Run:  python experiments/run_noise_sensitivity.py [--quick] [--levels 0 0.001 0.01 0.05]
"""
from __future__ import annotations

import pandas as pd

from _torch_common import (build_splits, config_dict, evaluate, fit_direct, fit_residual, resolve,
                           seed_of, torch_parser)
from fisher_surrogate import latex
from fisher_surrogate.common import NOISE_SEED_OFFSET, out_dir, save_results
from fisher_surrogate.data import add_target_noise
from fisher_surrogate.metrics import summarize_seeds


def main():
    p = torch_parser(__doc__)
    p.add_argument("--levels", type=float, nargs="*", default=[0.0, 0.001, 0.01, 0.05])
    args = p.parse_args()
    n_seeds, dcfg, tcfg, mkw = resolve(args)
    out = out_dir(args, "noise_sensitivity")
    train_df, val_df, test_df = build_splits(dcfg)
    ref_std = train_df["u_clean"].std()
    rows = []
    for r in args.levels:
        for i in range(n_seeds):
            s = seed_of(i)
            print(f"noise {r} seed {i:02d}", flush=True)
            tr = add_target_noise(train_df, r, s + NOISE_SEED_OFFSET, ref_std)
            va = add_target_noise(val_df, r, s + NOISE_SEED_OFFSET + 1, ref_std)
            rows.append({"noise": f"{100 * r:g}\\%", "model": "Direct MLP", "seed": i,
                         **evaluate(fit_direct(tr, va, s, tcfg, target="u_target"), test_df, mkw)[0]})
            rows.append({"noise": f"{100 * r:g}\\%", "model": "Residual MLP", "seed": i,
                         **evaluate(fit_residual(tr, va, s, tcfg, target="u_target"), test_df, mkw)[0]})
    seed_df = pd.DataFrame(rows)
    summary = summarize_seeds(seed_df, ["noise", "model"])
    tex = latex.table_ci(summary, ["noise", "model"], "Noise sensitivity (noise added only to training targets; "
                         f"mean $\\pm$ 95\\% CI over $S={n_seeds}$ seeds).", "tab:noise_sensitivity")
    save_results(out, "noise_sensitivity", seed_df, summary, tex, config_dict(args, dcfg, tcfg, n_seeds, mkw))


if __name__ == "__main__":
    main()
