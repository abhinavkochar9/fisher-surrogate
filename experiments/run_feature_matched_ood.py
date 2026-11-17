"""Round 2 (Reviewer 4; Reviewer 3, M2/M3): the feature-matched baselines of Table 10 on the
out-of-distribution regimes of Table 16, their terminal prediction error, and a horizon study.

Every model is trained exactly as in Table 10 / Table 13 (same split, seeds and code path); training is
deterministic on a fixed machine, so the in-distribution metrics reproduce the published per-seed values
(checked against results/feature_matched and results/neural_ode_gp and written to *_reproduction.csv).
Each trained model is then evaluated
  (i)   on the standard test set: unified metrics, the terminal error E_term = |u_hat(T_max) - u(T_max)|
        and the true trajectory's finite-time distance to equilibrium D_term = |u(T_max) - u*_eps|;
  (ii)  on the four regimes of Table 15 with the per-seed regime datasets of Table 16
        (seed OOD_DATA_SEED_BASE + seed index, 120 pairs each), as the existing Table 16 rows;
  (iii) at the horizons T in {5, 10, 20} on the standard test pairs: A_T = |u_hat(T) - u*_eps|,
        E_T = |u_hat(T) - u(T)| and D_T = |u(T) - u*_eps| (u(T) from the adaptive solver).
Families run as separate processes: --family mlp | gp | node | node_tight (the tight-tolerance Neural ODEs
of Table 13: test set and horizons only).  Test-set predictions are saved for every model and seed.

Run:  python experiments/run_feature_matched_ood.py --family mlp|gp|node|node_tight [--quick]
      python experiments/run_feature_matched_ood.py --tables-only     # summaries, LaTeX rows, figure data
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from _baselines_common import NODE_TOLERANCES, fit_gp, fit_neural_ode
from _torch_common import FEATURES, build_splits, config_dict, evaluate, fit_direct, fit_residual, resolve, seed_of, torch_parser
from fisher_surrogate import latex
from fisher_surrogate.common import OOD_DATA_SEED_BASE, RESULTS_DIR, final_tables_dir, out_dir, save_results
from fisher_surrogate.data import OOD_REGIMES, add_baseline_column, make_ood_dataset
from fisher_surrogate.dynamics import solve_variance, u_star
from fisher_surrogate.metrics import METRICS, summarize_seeds

UOU = "$u_{\\mathrm{OU}}$"
HORIZONS = (5.0, 10.0, 20.0)
TIGHT = "tight: rtol={:g}, atol={:g}".format(*NODE_TOLERANCES["tight"])
STD = NODE_TOLERANCES["standard"]
# family -> [(model name as in Table 10 / Table 13, fit(train, val, seed index, tcfg, args) -> model)]
FAMILIES = {
    "mlp": [("Direct MLP (no baseline)", lambda tr, va, i, c, a: fit_direct(tr, va, seed_of(i), c)),
            (f"Direct MLP + {UOU} input", lambda tr, va, i, c, a: fit_direct(tr, va, seed_of(i) + 100, c, features=FEATURES + ["u_base"])),
            ("Unscaled residual (difference model)", lambda tr, va, i, c, a: fit_residual(tr, va, seed_of(i), c, scaled=False)),
            ("Scaled residual MLP (proposed)", lambda tr, va, i, c, a: fit_residual(tr, va, seed_of(i), c))],
    "gp": [("GP, zero mean", lambda tr, va, i, c, a: fit_gp(tr, seed_of(i), prior_mean="zero", n_sub=a.gp_sub)),
           (f"GP, {UOU} prior mean", lambda tr, va, i, c, a: fit_gp(tr, seed_of(i), prior_mean="ou", n_sub=a.gp_sub))],
    "node": [("Neural ODE (full field)", lambda tr, va, i, c, a: fit_neural_ode(tr, va, seed_of(i), c, residual=False, rtol=STD[0], atol=STD[1])),
             ("Residual Neural ODE", lambda tr, va, i, c, a: fit_neural_ode(tr, va, seed_of(i), c, residual=True, rtol=STD[0], atol=STD[1]))],
    "node_tight": [(f"Neural ODE ({TIGHT})", lambda tr, va, i, c, a: fit_neural_ode(tr, va, seed_of(i), c, residual=False, rtol=1e-8, atol=1e-10)),
                   (f"Residual Neural ODE ({TIGHT})", lambda tr, va, i, c, a: fit_neural_ode(tr, va, seed_of(i), c, residual=True, rtol=1e-8, atol=1e-10))],
}
# published per-seed source of each model, for the reproduction check
PUBLISHED = {**{n: ("feature_matched", n) for f in ("mlp", "gp", "node") for n, _ in FAMILIES[f]},
             **{n: ("neural_ode_gp", n) for n, _ in FAMILIES["node_tight"]}}
TABLE16_ORDER = ["Direct MLP (no baseline)", f"Direct MLP + {UOU} input", "Unscaled residual (difference model)",
                 "Scaled residual MLP (proposed)", "GP, zero mean", f"GP, {UOU} prior mean", "Neural ODE (full field)",
                 "Residual Neural ODE"]
TABLE16_NAMES = {"Direct MLP (no baseline)": "Direct MLP", "Scaled residual MLP (proposed)": "Residual MLP",
                 "GP, zero mean": "Gaussian process", f"GP, {UOU} prior mean": f"GP + {UOU} prior mean",
                 "Unscaled residual (difference model)": "Unscaled residual MLP"}


def horizon_frame(test_df):
    pairs = test_df.groupby("pair_id")[["u0", "eps"]].first().reset_index()
    rows = []
    for _, p in pairs.iterrows():
        u = solve_variance(float(p.u0), float(p.eps), np.r_[0.0, HORIZONS], "adaptive")[1:]
        for T, uT in zip(HORIZONS, u):
            rows.append({"pair_id": p.pair_id, "u0": p.u0, "eps": p.eps, "t": T, "u_true": uT,
                         "u_star": float(u_star(p.eps)), "u_clean": uT})
    return pd.DataFrame(rows)


def horizon_rows(f, H):
    h = H.copy()
    h["u_pred"] = f.predict_frame(H.drop(columns=["u_true", "u_star"]))
    out = []
    for T, g in h.groupby("t"):
        out.append({"split": "horizon", "regime": f"T={T:g}", "A_T": float(np.mean(np.abs(g.u_pred - g.u_star))),
                    "E_T": float(np.mean(np.abs(g.u_pred - g.u_true))), "D_T": float(np.mean(np.abs(g.u_true - g.u_star)))})
    return out


def run_family(args, family, out):
    n_seeds, dcfg, tcfg, mkw = resolve(args)
    node_like = family.startswith("node")
    seeds = (1 if args.quick else args.node_seeds) if node_like else n_seeds
    if args.quick:
        tcfg.epochs, tcfg.patience = 30, 10
        args.gp_sub = 300
    n_ood = 12 if args.quick else args.n_ood_pairs
    train_df, val_df, test_df = (add_baseline_column(d) for d in build_splits(dcfg))   # exactly as Table 10
    H = horizon_frame(test_df)
    rows, preds = [], []
    seed_list = args.seeds if args.seeds else list(range(seeds))
    tag = family + (f"_s{min(seed_list)}-{max(seed_list)}" if args.seeds else "")
    for i in seed_list:
        oods = {} if family == "node_tight" else {r: make_ood_dataset(r, n_ood, dcfg.n_time, seed=OOD_DATA_SEED_BASE + i)
                                                  for r in OOD_REGIMES}
        for name, fitter in FAMILIES[family]:
            print(f"seed {i:02d}  {name}", flush=True)
            f = fitter(train_df, val_df, i, tcfg, args)
            m, pred = evaluate(f, test_df, mkw)
            rows.append({"split": "test", "regime": "in-distribution", "model": name, "seed": i, **m})
            preds.append(pred[["pair_id", "t", "u0", "eps", "u_clean", "u_pred"]].assign(model=name, seed=i))
            for regime, ood in oods.items():
                m, _ = evaluate(f, ood, mkw)
                rows.append({"split": "ood", "regime": regime, "model": name, "seed": i, **m})
            for hr in horizon_rows(f, H):
                rows.append({**hr, "model": name, "seed": i})
    pd.DataFrame(rows).to_csv(out / f"shard_{tag}_seed_results.csv", index=False)
    pd.concat(preds).to_csv(out / f"predictions_test_{tag}.csv.gz", index=False, float_format="%.12g")
    cfg = config_dict(args, dcfg, tcfg, n_seeds, mkw, family=family, seeds=seeds, n_ood_pairs=n_ood,
                      ood_data_seed="OOD_DATA_SEED_BASE + seed index", horizons=HORIZONS)
    Path(out / f"shard_{tag}_config.json").write_text(json.dumps(cfg, indent=2, default=str))


def reproduction(seed_df, out):
    """Per-seed comparison of the regenerated in-distribution metrics with the published Table 10 / 13 files."""
    rows = []
    for name, (sub, pub_name) in PUBLISHED.items():
        f = RESULTS_DIR / sub / f"{sub}_seed_results.csv"
        new = seed_df[(seed_df.split == "test") & (seed_df.model == name)].set_index("seed")
        if new.empty or not f.exists():
            continue
        old = pd.read_csv(f)
        old = old[old.model == pub_name].set_index("seed")
        j = new[METRICS].join(old[METRICS], rsuffix="_pub", how="inner")
        rel = max(float(np.nanmax(np.abs(j[m] - j[m + "_pub"]) / np.abs(j[m + "_pub"]))) for m in METRICS)
        rows.append({"model": name, "published_file": f"results/{sub}/{sub}_seed_results.csv", "seeds": len(j),
                     "max_rel_diff": rel})
    rep = pd.DataFrame(rows)
    rep.to_csv(out / "feature_matched_ood_reproduction.csv", index=False)
    print(rep.to_string(index=False))


def summaries(out, quick=False):
    shards = sorted(glob.glob(str(out / "shard_*_seed_results.csv")))
    seed_df = pd.concat([pd.read_csv(f) for f in shards], ignore_index=True)
    main = seed_df[seed_df.split != "horizon"]
    summary = summarize_seeds(main, ["split", "regime", "model"], metrics=METRICS + ["E_term", "D_term", "E_over_rise", "E_over_fall"])
    save_results(out, "feature_matched_ood", main, summary)
    hz = seed_df[seed_df.split == "horizon"].dropna(axis=1, how="all")
    hsum = summarize_seeds(hz, ["regime", "model"], metrics=["A_T", "E_T", "D_T"])
    save_results(out, "horizon", hz, hsum)
    reproduction(seed_df, out)
    tsum = transfer_terminal(out.parent / "transfer_terminal")
    (final_tables_dir(quick) / "round2_table16_feature_matched.tex").write_text(table16_rows(summary), encoding="utf-8")
    (final_tables_dir(quick) / "round2_terminal_error.tex").write_text(terminal_rows(summary, hsum, tsum), encoding="utf-8")


def transfer_terminal(d):
    """Terminal error of the Table 20 models: the per-seed files of run_transfer_system.py and
    run_gp_offdistribution.py --axes transfer, rerun into results/transfer_terminal (same seeds; their
    published metrics, predictions and LaTeX table are identical to transfer_system/ and gp_offdistribution/)."""
    frames = [pd.read_csv(d / f) for f in ("transfer_system_seed_results.csv", "gp_transfer_seed_results.csv")
              if (d / f).exists()]
    if not frames:
        return None
    s = summarize_seeds(pd.concat(frames, ignore_index=True), ["model"], metrics=["E_eq", "E_term", "D_term"])
    s.to_csv(d / "transfer_terminal_summary.csv", index=False)
    return s


def table16_rows(summary):
    lines = ["% Table 16 (tab:ood_generalization) rows: all models evaluated on the per-seed regime datasets of Table 16.",
             "% Mean +/- 95% CI over S = 10 seeds (S = 5 for the Neural ODE variants); last column n_fail (missed true",
             "% crossings, summed over seeds; '--' where the regime has no true crossing).",
             "% Generated by: python experiments/run_feature_matched_ood.py --tables-only"]
    s = summary[summary.split == "ood"]
    for regime in OOD_REGIMES:
        lines.append(f"% --- {regime}")
        for name in TABLE16_ORDER:
            r = s[(s.regime == regime) & (s.model == name)]
            if r.empty:
                continue
            r = r.iloc[0]
            cells = [latex.cell(r[f"{m}_mean"], r[f"{m}_ci"]) for m in METRICS]
            nf = "--" if int(r["n_cross_total"]) == 0 else str(int(r["n_cross_fail_total"]))
            lines.append(f" & {TABLE16_NAMES.get(name, name)} & " + " & ".join(cells) + f" & {nf} \\\\")
    return "\n".join(lines) + "\n"


def terminal_rows(summary, hsum, tsum=None):
    lines = ["% Terminal prediction error E_term = |u_hat(T_max) - u(T_max)| next to E_eq = |u_hat(T_max) - u*| and the true",
             "% trajectory's distance D_term = |u(T_max) - u*| (T_max = 5), standard test set; then the horizon study:",
             "% A_T = |u_hat(T) - u*| at T = 10, 20 (u(T) is within D_T of u* there).",
             "% Columns: model & E_eq & E_term & D_term & A_10 & A_20",
             "% Generated by: python experiments/run_feature_matched_ood.py --tables-only",
             "% --- Fisher benchmark: models of Tables 10 and 13"]
    s = summary[summary.split == "test"]
    for name in TABLE16_ORDER + [n for n, _ in FAMILIES["node_tight"]]:
        r = s[s.model == name]
        if r.empty:
            continue
        r = r.iloc[0]
        h = {T: hsum[(hsum.regime == f"T={T:g}") & (hsum.model == name)] for T in HORIZONS}
        hz = [latex.cell(h[T].iloc[0]["A_T_mean"], h[T].iloc[0]["A_T_ci"]) if len(h[T]) else "--" for T in (10.0, 20.0)]
        lines.append(" & ".join([name, latex.cell(r["E_eq_mean"], r["E_eq_ci"]), latex.cell(r["E_term_mean"], r["E_term_ci"]),
                                 latex.cell(r["D_term_mean"])] + hz) + r" \\")
    if tsum is not None:
        lines.append("% --- Transfer system: models of Table 20 (reference: the equilibrium x1* of Eq. (24), closed form; no horizon study)")
        for _, r in tsum.iterrows():
            lines.append(" & ".join([r["model"], latex.cell(r["E_eq_mean"], r["E_eq_ci"]),
                                     latex.cell(r["E_term_mean"], r["E_term_ci"]), latex.cell(r["D_term_mean"]), "--", "--"]) + r" \\")
    return "\n".join(lines) + "\n"


def main():
    p = torch_parser(__doc__)
    p.add_argument("--family", choices=list(FAMILIES))
    p.add_argument("--node-seeds", type=int, default=5)
    p.add_argument("--gp-sub", type=int, default=2000)
    p.add_argument("--n-ood-pairs", type=int, default=120)
    p.add_argument("--seeds", type=int, nargs="*", default=None, help="seed indices (default: all)")
    p.add_argument("--tables-only", action="store_true")
    p.add_argument("--no-summary", action="store_true", help="write the shard only (parallel runs)")
    args = p.parse_args()
    out = out_dir(args, "feature_matched_ood")
    if not args.tables_only:
        if args.family is None:
            raise SystemExit("--family is required (or --tables-only)")
        run_family(args, args.family, out)
        if args.no_summary:
            return
    summaries(out, args.quick)


if __name__ == "__main__":
    main()
