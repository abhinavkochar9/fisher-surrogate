"""Revised paper Sec. 6.8, Table 17, Appendix D (Tables D.22, D.23, Figures 7 and D.10) --
physics-informed gradient signal, off-grid ODE consistency and data efficiency.

For lambda in {0, 0.01, 0.1, 1.0}:
    L_phys (test)   mean squared ODE residual on the held-out grid at convergence      (Eq. 23)
    C_ODE           the same residual at N_c = 200 random collocation points (off-grid) (Eq. D.1)
    E_traj, E_cross, E_eq   on the SAME held-out set and with the SAME unified definitions
                    as every other table (removes the scale discrepancy R2/R4 noted)
and the data-efficiency study retrains all four models on 100 / 50 / 25 % of the training
trajectories over S = 3 sub-samples.

The original analysis used a smaller dataset (80 pairs -> 56 training trajectories); this
is reproduced with --n-pairs 80 (the default here) and S = 5 seeds.

Run:  python experiments/run_pi_gradient_analysis.py [--quick] [--n-pairs 80]
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from _torch_common import build_splits, config_dict, evaluate, resolve, seed_of, torch_parser
from fisher_surrogate import latex
from fisher_surrogate.common import out_dir, save_results
from fisher_surrogate.data import TRAIN_EPS_RANGE, TRAIN_U0_RANGE, xy_direct
from fisher_surrogate.metrics import METRICS, summarize_seeds
from fisher_surrogate.models.torch_models import predict_physics_informed_residual
from run_physics_informed import fit_pi

LAMBDAS = [0.0, 0.01, 0.1, 1.0]
FRACTIONS = [1.0, 0.5, 0.25]


def ode_residual_stats(fit, df):
    _, res = predict_physics_informed_residual(fit.model, df[["u0", "eps", "t"]].to_numpy(float), fit.xs, fit.rs,
                                               fit.device, with_ode_residual=True)
    return float(np.mean(res**2))


def collocation_frame(test_df, n_c, seed):
    rng = np.random.default_rng(seed)
    pairs = test_df.groupby("pair_id").first()[["u0", "eps"]]
    pick = pairs.iloc[rng.integers(0, len(pairs), size=n_c)]
    return pd.DataFrame({"u0": pick["u0"].to_numpy(), "eps": pick["eps"].to_numpy(),
                         "t": rng.uniform(0.0, test_df["t"].max(), size=n_c)})


def main():
    p = torch_parser(__doc__)
    p.set_defaults(n_pairs=80)
    p.add_argument("--n-collocation", type=int, default=200)
    p.add_argument("--de-seeds", type=int, default=3)
    args = p.parse_args()
    n_seeds, dcfg, tcfg, mkw = resolve(args, default_seeds=5)
    de_seeds = 1 if args.quick else args.de_seeds
    out = out_dir(args, "pi_gradient_analysis")
    train_df, val_df, test_df = build_splits(dcfg)
    coll = collocation_frame(test_df, args.n_collocation, 2027)

    # --- Table 17 / D.22 / Figure 7 ---
    rows = []
    for i in range(n_seeds):
        s = seed_of(i)
        for lam in LAMBDAS:
            print(f"seed {i:02d} lambda={lam}", flush=True)
            fit = fit_pi(train_df, val_df, lam, s, tcfg)
            m, _ = evaluate(fit, test_df, mkw)
            rows.append({"model": f"Residual MLP, $\\lambda={lam}$", "lambda": lam, "seed": i, **m,
                         "L_phys_test": ode_residual_stats(fit, test_df), "C_ODE_offgrid": ode_residual_stats(fit, coll)})
    seed_df = pd.DataFrame(rows)
    summary = summarize_seeds(seed_df, ["model"], metrics=METRICS + ["L_phys_test", "C_ODE_offgrid"])
    tex = pi_table(summary, n_seeds)
    save_results(out, "pi_gradient", seed_df, summary, tex, config_dict(args, dcfg, tcfg, n_seeds, mkw, lambdas=LAMBDAS))

    # --- Table D.23 / Figure D.10: data efficiency ---
    rows = []
    train_pairs = np.array(sorted(train_df.pair_id.unique()))
    for frac in FRACTIONS:
        n_keep = max(4, int(round(frac * len(train_pairs))))
        for j in range(de_seeds):
            rng = np.random.default_rng(3000 + j)
            keep = rng.choice(train_pairs, n_keep, replace=False)
            sub = train_df[train_df.pair_id.isin(keep)]
            for lam in LAMBDAS:
                print(f"data-efficiency frac={frac} subsample {j} lambda={lam}", flush=True)
                fit = fit_pi(sub, val_df, lam, seed_of(j) + 777, tcfg)
                m, _ = evaluate(fit, test_df, mkw)
                rows.append({"fraction": f"{int(100 * frac)}\\% ({n_keep})", "n_train": n_keep, "lambda": lam, "seed": j, **m})
    de = pd.DataFrame(rows)
    de_summary = summarize_seeds(de, ["fraction", "lambda"], metrics=METRICS)
    save_results(out, "data_efficiency", de, de_summary, de_table(de_summary, de_seeds))


def pi_table(summary, n_seeds):
    from fisher_surrogate.latex import sci_ci
    lines = [r"\begin{table}[t]", r"\centering", r"\scriptsize",
             rf"\caption{{Physics-informed residual MLP: ODE residual at convergence (test grid), off-grid consistency "
             rf"$C_{{\mathrm{{ODE}}}}$, and mechanism-level metrics on the standard held-out set under the unified "
             rf"definitions (mean $\pm$ 95\% CI over $S={n_seeds}$ seeds).}}", r"\label{tab:pi_gradient}",
             r"\begin{tabular}{lccccc}", r"\hline",
             r"Model & $\mathcal L_{\mathrm{phys}}$ (test) & $C_{\mathrm{ODE}}$ (off-grid) & $E_{\mathrm{traj}}$ & $E_{\mathrm{cross}}$ & $E_{\mathrm{eq}}$ \\", r"\hline"]
    for _, r in summary.iterrows():
        lines.append(f"{r['model']} & " + " & ".join(sci_ci(r[f'{m}_mean'], r[f'{m}_ci'])
                                                    for m in ("L_phys_test", "C_ODE_offgrid", "E_traj", "E_cross", "E_eq")) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def de_table(summary, de_seeds):
    from fisher_surrogate.latex import sci_ci
    fracs = list(dict.fromkeys(summary["fraction"]))
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             rf"\caption{{Data-efficiency analysis: $E_{{\mathrm{{traj}}}}$ versus training-set fraction (mean $\pm$ 95\% CI over $S={de_seeds}$ sub-samples).}}",
             r"\label{tab:data_efficiency}", r"\begin{tabular}{l" + "c" * len(LAMBDAS) + "}", r"\hline",
             "Fraction (traj.) & " + " & ".join(rf"$\lambda={l}$" for l in LAMBDAS) + r" \\", r"\hline"]
    for f in fracs:
        g = summary[summary.fraction == f].set_index("lambda")
        lines.append(f"{f} & " + " & ".join(sci_ci(g.loc[l, "E_traj_mean"], g.loc[l, "E_traj_ci"]) for l in LAMBDAS) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
