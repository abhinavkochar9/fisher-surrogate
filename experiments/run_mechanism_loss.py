"""Table 18 (reviewer R2.7) -- mechanism-level quantities as training signal:
residual MLP trained on L_traj + beta (E_eq + E_cross), beta in {0, 0.1, 1}, with the
interpolated (differentiable) crossing time of Sec. 4.5.  Framed in the paper as a pointer to
future work; the evaluation uses the same held-out set and unified metrics.

Run:  python experiments/run_mechanism_loss.py [--quick] [--betas 0 0.1 1]
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from _torch_common import build_splits, config_dict, evaluate, resolve, seed_of, torch_parser
from fisher_surrogate import latex
from fisher_surrogate.common import out_dir, save_results
from fisher_surrogate.dynamics import crossing_time_exact, u_star
from fisher_surrogate.metrics import summarize_seeds
from fisher_surrogate.models.torch_models import train_residual_mechanism_loss
from run_physics_informed import PIFit


def frame_to_traj(df):
    t = np.sort(df["t"].unique())
    X, U, T, UE = [], [], [], []
    for _, g in df.groupby("pair_id"):
        g = g.sort_values("t")
        u0, eps = float(g.u0.iloc[0]), float(g.eps.iloc[0])
        X.append(np.column_stack([np.full(len(t), u0), np.full(len(t), eps), t]))
        U.append(g["u_clean"].to_numpy(float))
        T.append(crossing_time_exact(u0, eps))
        UE.append(float(u_star(eps)))
    return np.array(X), np.array(U), np.array(T), np.array(UE), t


def main():
    p = torch_parser(__doc__)
    p.add_argument("--betas", type=float, nargs="*", default=[0.0, 0.1, 1.0])
    args = p.parse_args()
    n_seeds, dcfg, tcfg, mkw = resolve(args)
    out = out_dir(args, "mechanism_loss")
    train_df, val_df, test_df = build_splits(dcfg)
    X, U, T, UE, tg = frame_to_traj(train_df)
    xv, uv = val_df[["u0", "eps", "t"]].to_numpy(float), val_df[["u_clean"]].to_numpy(float)
    rows = []
    for i in range(n_seeds):
        s = seed_of(i)
        for beta in args.betas:
            print(f"seed {i:02d} beta={beta}", flush=True)
            model, xs, rs = train_residual_mechanism_loss(X, U, T, UE, tg, xv, uv, beta, s + 21000, tcfg)
            fit = PIFit(model, xs, rs, tcfg.device)
            rows.append({"model": f"Residual MLP, $\\beta={beta}$", "beta": beta, "seed": i, **evaluate(fit, test_df, mkw)[0]})
    seed_df = pd.DataFrame(rows)
    summary = summarize_seeds(seed_df, ["model"])
    tex = latex.table_ci(summary, ["model"], "Residual MLP trained with $\\beta(E_{\\mathrm{eq}}+E_{\\mathrm{cross}})$ added to "
                         f"the trajectory loss (mean $\\pm$ 95\\% CI over $S={n_seeds}$ seeds).", "tab:mech_loss")
    save_results(out, "mechanism_loss", seed_df, summary, tex, config_dict(args, dcfg, tcfg, n_seeds, mkw))


if __name__ == "__main__":
    main()
