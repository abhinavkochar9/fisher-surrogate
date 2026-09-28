"""Revised paper Sec. 6.9 / Table 19 -- residual advantage as a function of baseline error
(reviewers R5.1, R2.1).  Three axes:

  (A) Non-Gaussian PDE targets with the Gaussian-reduced ODE as an approximate baseline.
      Target: variance of the full Fisher-regularized PDE from symmetric-bimodal initial
      densities with separation a = alpha sqrt(u0) (a = 0 is Gaussian, baseline exact).
      Inputs (u0, eps, a, t).  Models: direct MLP, direct MLP + u_ODE input, residual MLP
      with the reduced-ODE baseline.  Stratified by baseline RMSE ||u_PDE - u_ODE||.
  (B) Transfer system with a forward-Euler baseline of step h in {0.01, 0.05, 0.1, 0.25, 0.5}.
  (C) Fisher benchmark with a structurally wrong baseline rate kappa in {0.5, 0.8, 1.2, 2.0}
      in 1 + (u0 - 1) exp(-2 kappa t).

The PDE dataset of (A) is cached under results/pde_bimodal/ (its generation is the slow part;
use --n-pde-pairs to control it). A full run also writes the whole of Table 19 in the manuscript
layout to tables/final/table19_baseline_error.tex.

Run:  python experiments/run_imperfect_baseline.py [--quick] [--axes A B C]
      python experiments/run_imperfect_baseline.py --tables-only     # rebuild Table 19 from saved results
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from _torch_common import (FEATURES, build_splits, config_dict, evaluate, fit_direct, fit_residual, resolve,
                           seed_of, torch_parser)
from fisher_surrogate import latex, pde_dataset, transfer
from fisher_surrogate.common import QUICK_DIR, RESULTS_DIR, final_tables_dir, out_dir, save_results
from fisher_surrogate.data import add_baseline_column, split_pair_ids
from fisher_surrogate.metrics import METRICS, summarize_seeds
from run_transfer_system import TRANSFER_KW, prepare, split

KAPPAS = [0.5, 0.8, 1.0, 1.2, 2.0]
EULER_STEPS = [0.01, 0.05, 0.1, 0.25, 0.5]
PDE_KW = dict(true_col="u_clean", t_cross_col="T_cross_true", u_eq_col="u_eq")


def stratified_rows(seed_df, strata_col):
    """Mean +/- CI per (stratum, model) and residual/direct ratios per stratum."""
    summary = summarize_seeds(seed_df, [strata_col, "model"])
    ratios = []
    for st, g in summary.groupby(strata_col, sort=False):
        g = g.set_index("model")
        if "Direct MLP" in g.index and "Residual MLP" in g.index:
            ratios.append({strata_col: st, **{f"ratio_{m}": g.loc["Residual MLP", f"{m}_mean"] / g.loc["Direct MLP", f"{m}_mean"]
                                             for m in METRICS}})
    return summary, pd.DataFrame(ratios)


def axis_a(args, n_seeds, dcfg, tcfg, mkw, out):
    n_pde = 24 if args.quick else args.n_pde_pairs
    rr = "" if args.ratio_range is None else f"_r{args.ratio_range[0]:g}-{args.ratio_range[1]:g}"
    cache = (QUICK_DIR if args.quick else RESULTS_DIR) / "pde_bimodal" / f"dataset_n{n_pde}_t{dcfg.n_time}_nx{args.pde_nx}{rr}.csv"
    print(f"(A) building/loading PDE dataset ({n_pde} trajectories) -> {cache}", flush=True)
    df = pde_dataset.build_dataset(n_pde, n_time=dcfg.n_time, cache=cache, n_exact=max(2, n_pde // 8),
                                   nx=args.pde_nx, verbose=True,
                                   **({} if args.ratio_range is None else {"ratio_range": tuple(args.ratio_range)}))
    df["stratum"] = pde_dataset.baseline_error_bins(df)
    tr, va, te = split_pair_ids(n_pde, 0.70, 0.15, dcfg.split_seed)
    train_df, val_df, test_df = (df[df.pair_id.isin(s)].copy() for s in (tr, va, te))
    feats = pde_dataset.FEATURES_A
    rows = []
    for i in range(n_seeds):
        s = seed_of(i)
        print(f"(A) seed {i:02d}", flush=True)
        fits = {
            "Direct MLP": fit_direct(train_df, val_df, s, tcfg, features=feats, base=("column",)),
            "Direct MLP + $u_{\\mathrm{ODE}}$ input": fit_direct(train_df, val_df, s + 100, tcfg,
                                                                 features=feats + ["u_base"], base=("column",)),
            "Residual MLP": fit_residual(train_df, val_df, s, tcfg, features=feats, base=("column",), base_col="u_base"),
        }
        for name, fit in fits.items():
            _, per = evaluate_per(fit, test_df, mkw, PDE_KW)
            for st, g in per.groupby("stratum"):
                rows.append({"stratum": st, "model": name, "seed": i, "baseline_rmse": g["baseline_rmse"].mean(),
                             **{m: float(np.nanmean(g[m])) for m in METRICS},
                             "n_cross": int(g["T_cross_true"].notna().sum()), "n_cross_fail": int(g["cross_fail"].sum()),
                             "n_cross_spurious": int(g["cross_spurious"].sum()), "train_time_s": fit.train_time_s})
    seed_df = pd.DataFrame(rows)
    summary, ratios = stratified_rows(seed_df, "stratum")
    save_results(out, "imperfect_A_pde", seed_df, summary,
                 latex.table_ci(summary, ["stratum", "model"], "(A) Non-Gaussian PDE targets with the reduced-ODE "
                                f"baseline, stratified by baseline error (mean $\\pm$ 95\\% CI over $S={n_seeds}$ seeds).",
                                "tab:imperfect_A"))
    ratios.to_csv(out / "imperfect_A_pde_ratios.csv", index=False)
    print(ratios)


def evaluate_per(fit, test_df, mkw, ckw):
    """Like _torch_common.evaluate but returns the per-trajectory frame joined with test metadata."""
    from fisher_surrogate.metrics import compute_metrics
    pred = test_df.copy()
    pred["u_pred"] = fit.predict_frame(test_df)
    factory = fit.pred_fn_factory if mkw.get("refine", False) else None
    m, per = compute_metrics(pred, pred_fn_factory=factory, return_per_trajectory=True, **ckw, **mkw)
    meta = test_df.groupby("pair_id").first()
    per = per.join(meta[[c for c in ("stratum", "baseline_rmse", "a", "alpha", "kappa", "h") if c in meta]], on="pair_id")
    return m, per


def axis_b(args, n_seeds, dcfg, tcfg, mkw, out):
    rows = []
    for h in ([0.05, 0.5] if args.quick else EULER_STEPS):
        df = prepare(transfer.make_dataset(dcfg.n_pairs, dcfg.n_time, baseline="euler", euler_h=h))
        base_err = float(np.sqrt(np.mean((df["y_base"] - df["y_base_exact"]) ** 2)))
        train_df, val_df, test_df = split(df, dcfg.n_pairs)
        for i in range(n_seeds):
            s = seed_of(i)
            print(f"(B) h={h} seed {i:02d}", flush=True)
            fits = {
                "Direct MLP": fit_direct(train_df, val_df, s, tcfg, target="y_clean", features=transfer.FEATURES, base=("column",)),
                "Direct MLP + $y_{\\mathrm{base}}$ input": fit_direct(train_df, val_df, s + 100, tcfg, target="y_clean",
                                                                      features=transfer.FEATURES + ["u_base"], base=("column",)),
                "Residual MLP": fit_residual(train_df, val_df, s, tcfg, target="y_clean", features=transfer.FEATURES,
                                             base=("column",), base_col="u_base", scale_col="gamma"),
            }
            for name, fit in fits.items():
                m, _ = evaluate(fit, test_df, mkw, **TRANSFER_KW)
                rows.append({"h": h, "baseline_rmse": base_err, "model": name, "seed": i, **m})
    seed_df = pd.DataFrame(rows)
    summary, ratios = stratified_rows(seed_df, "h")
    save_results(out, "imperfect_B_euler", seed_df, summary,
                 latex.table_ci(summary, ["h", "model"], "(B) Transfer system with a forward-Euler baseline of step $h$ "
                                f"(mean $\\pm$ 95\\% CI over $S={n_seeds}$ seeds).", "tab:imperfect_B"))
    ratios.to_csv(out / "imperfect_B_euler_ratios.csv", index=False)
    print(ratios)


def axis_c(args, n_seeds, dcfg, tcfg, mkw, out):
    rows = []
    splits = build_splits(dcfg)
    for kappa in ([0.5, 2.0] if args.quick else KAPPAS):
        train_df, val_df, test_df = (add_baseline_column(d, "kappa", kappa) for d in splits)
        base_err = float(np.sqrt(np.mean((test_df["u_base"] - (1 + (test_df["u0"] - 1) * np.exp(-2 * test_df["t"]))) ** 2)))
        for i in range(n_seeds):
            s = seed_of(i)
            print(f"(C) kappa={kappa} seed {i:02d}", flush=True)
            fits = {
                "Direct MLP": fit_direct(train_df, val_df, s, tcfg),
                "Direct MLP + $u^{\\kappa}_{\\mathrm{OU}}$ input": fit_direct(train_df, val_df, s + 100, tcfg,
                                                                              features=FEATURES + ["u_base"], base=("kappa", kappa)),
                "Residual MLP": fit_residual(train_df, val_df, s, tcfg, base=("kappa", kappa), base_col="u_base"),
            }
            for name, fit in fits.items():
                m, _ = evaluate(fit, test_df, mkw)
                rows.append({"kappa": kappa, "baseline_rmse": base_err, "model": name, "seed": i, **m})
    seed_df = pd.DataFrame(rows)
    summary, ratios = stratified_rows(seed_df, "kappa")
    save_results(out, "imperfect_C_kappa", seed_df, summary,
                 latex.table_ci(summary, ["kappa", "model"], "(C) Fisher benchmark with a misspecified baseline rate "
                                f"$\\kappa$ (mean $\\pm$ 95\\% CI over $S={n_seeds}$ seeds).", "tab:imperfect_C"))
    ratios.to_csv(out / "imperfect_C_kappa_ratios.csv", index=False)
    print(ratios)


# Table 19 in the manuscript layout: (file stem, condition column, [(value, row label)], group heading)
FINAL_SPEC = [
    ("imperfect_A_pde", "stratum", [("exact", "$a=0$ (exact)"), ("small", "$a$ = small"), ("medium", "$a$ = medium"),
                                    ("large", "$a$ = large")], "(A) Non-Gaussian PDE; reduced-ODE baseline"),
    ("imperfect_B_euler", "h", [(h, f"$h={h:.2f}$") for h in EULER_STEPS], "(B) Transfer system; forward-Euler baseline"),
    ("imperfect_C_kappa", "kappa", [(1.0, "$\\kappa=1$ (exact)"), (0.8, "$\\kappa=0.8$"), (1.2, "$\\kappa=1.2$"),
                                    (0.5, "$\\kappa=0.5$"), (2.0, "$\\kappa=2.0$")], "(C) Fisher benchmark; misspecified rate"),
]


def _at(df, col, v):
    return df[np.isclose(df[col].astype(float), float(v))] if col != "stratum" else df[df[col] == v]


def final_table(out, quick=False):
    """Whole Table 19 from the saved per-axis results: baseline, direct and residual E_traj (seed means) and the
    residual/direct ratios of E_traj, E_cross and E_eq. Baseline E_traj = squared baseline RMSE of the condition."""
    lines = ["% Table 19 (tab:imperfect_baseline): seed means over S = 10 seeds; ratios = residual / direct.",
             "% Generated by: python experiments/run_imperfect_baseline.py [--tables-only]",
             r"\begin{tabular}{lcccccc}", r"\toprule",
             r"Condition & Baseline $E_{\mathrm{traj}}$ & Direct $E_{\mathrm{traj}}$ & Residual $E_{\mathrm{traj}}$ & "
             r"\multicolumn{3}{c}{Residual / direct} \\", r"\cmidrule(lr){5-7}",
             r" & & & & $E_{\mathrm{traj}}$ & $E_{\mathrm{cross}}$ & $E_{\mathrm{eq}}$ \\"]
    for stem, col, conds, heading in FINAL_SPEC:
        summ = pd.read_csv(out / f"{stem}_summary.csv")
        rat = pd.read_csv(out / f"{stem}_ratios.csv")
        base = pd.read_csv(out / f"{stem}_seed_results.csv").groupby(col).baseline_rmse.first()
        lines += [r"\midrule", rf"\multicolumn{{7}}{{l}}{{\textit{{{heading}}}}} \\"]
        for v, label in conds:
            keys = [k for k in base.index if (k == v if col == "stratum" else np.isclose(float(k), v))]
            if not keys or _at(rat, col, v).empty:   # condition not run (e.g. --quick)
                continue
            b = float(base[keys[0]])
            d = _at(summ, col, v).set_index("model")
            r = _at(rat, col, v).iloc[0]
            lines.append(f"{label} & {latex.cell(b * b)} & {latex.cell(d.loc['Direct MLP', 'E_traj_mean'])} & "
                         f"{latex.cell(d.loc['Residual MLP', 'E_traj_mean'])} & {latex.ratio(r.ratio_E_traj)} & "
                         f"{latex.ratio(r.ratio_E_cross)} & {latex.ratio(r.ratio_E_eq)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (final_tables_dir(quick) / "table19_baseline_error.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    p = torch_parser(__doc__)
    p.add_argument("--axes", nargs="*", default=["A", "B", "C"])
    p.add_argument("--n-pde-pairs", type=int, default=160)
    p.add_argument("--pde-nx", type=int, default=768)
    p.add_argument("--ratio-range", type=float, nargs=2, default=[0.0, 1.5],
                   help="sample the bimodality ratio a/sigma_c uniformly in this range (revision: 0 1.5)")
    p.add_argument("--tables-only", action="store_true", help="rebuild Table 19 from the results saved in --out")
    args = p.parse_args()
    if args.tables_only:
        final_table(out_dir(args, "imperfect_baseline"), args.quick)
        return
    n_seeds, dcfg, tcfg, mkw = resolve(args)
    if args.quick:
        tcfg.epochs, tcfg.patience = 30, 10
    out = out_dir(args, "imperfect_baseline")
    if "C" in args.axes:
        axis_c(args, n_seeds, dcfg, tcfg, mkw, out)
    if "B" in args.axes:
        axis_b(args, n_seeds, dcfg, tcfg, mkw, out)
    if "A" in args.axes:
        axis_a(args, n_seeds, dcfg, tcfg, mkw, out)
    save_results(out, "imperfect_baseline", config=config_dict(args, dcfg, tcfg, n_seeds, mkw, kappas=KAPPAS,
                                                                euler_steps=EULER_STEPS))
    if set(args.axes) == {"A", "B", "C"}:
        final_table(out, args.quick)


if __name__ == "__main__":
    main()
