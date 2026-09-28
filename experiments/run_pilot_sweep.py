"""Hyper-parameter pilot sweep (reviewer R2.6): widths {32, 64, 128} x learning rates
{1e-3, 2e-3, 5e-3} on the DIRECT MLP only, S = 3 seeds, validation MSE and test metrics.
This is the sweep from which the (64, 64) / lr 2e-3 setting was fixed once and applied
unchanged to every other model (no per-model tuning).  Reported in the supplementary material.

Run:  python experiments/run_pilot_sweep.py [--quick]
"""
from __future__ import annotations

from dataclasses import replace

import pandas as pd

from _torch_common import build_splits, config_dict, evaluate, fit_direct, resolve, seed_of, torch_parser
from fisher_surrogate import latex
from fisher_surrogate.common import out_dir, save_results
from fisher_surrogate.metrics import METRICS, summarize_seeds

WIDTHS = [32, 64, 128]
LRS = [1e-3, 2e-3, 5e-3]


def main():
    p = torch_parser(__doc__)
    args = p.parse_args()
    n_seeds, dcfg, tcfg, mkw = resolve(args, default_seeds=3)
    out = out_dir(args, "pilot_sweep")
    train_df, val_df, test_df = build_splits(dcfg)
    rows = []
    for w in WIDTHS:
        for lr in LRS:
            cfg = replace(tcfg, hidden=(w, w), lr=lr)
            for i in range(n_seeds):
                print(f"width {w} lr {lr} seed {i}", flush=True)
                fit = fit_direct(train_df, val_df, seed_of(i) + 30000, cfg)
                m, _ = evaluate(fit, test_df, mkw)
                rows.append({"width": f"({w},{w})", "lr": f"{lr:g}", "seed": i, "val_mse_scaled": fit.model.best_val, **m})
    seed_df = pd.DataFrame(rows)
    summary = summarize_seeds(seed_df, ["width", "lr"], metrics=["val_mse_scaled"] + METRICS)
    tex = latex.table_ci(summary, ["width", "lr"], f"Pilot sweep on the direct MLP (mean $\\pm$ 95\\% CI over $S={n_seeds}$ "
                         "seeds); the (64,64) / $2\\times10^{-3}$ setting was fixed from this sweep for all models.",
                         "tab:pilot_sweep")
    save_results(out, "pilot_sweep", seed_df, summary, tex, config_dict(args, dcfg, tcfg, n_seeds, mkw))


if __name__ == "__main__":
    main()
