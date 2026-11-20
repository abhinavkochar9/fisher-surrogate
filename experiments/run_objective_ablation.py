"""Round 2 (Reviewer 3, M1): output parameterization versus loss weighting.

The published scaled residual MLP is trained on the z-scored residual target r = (u - u_OU)/eps, so its
training and validation loss is the trajectory MSE weighted by 1/eps^2.  The direct MLP and the unscaled
residual MLP are trained on z-scored targets u and u - u_OU, i.e. on the unweighted trajectory MSE (up to
a global constant), and the physics-informed residual on the unweighted MSE of the reconstructed u plus
the penalty.  This script separates the two effects, on the Table 7 / Table 9 setup (300 pairs, split
seed 999, S = 10 seeds, unified metrics):

  --part factorial  {direct, unscaled residual, scaled residual} x {unweighted, 1/eps^2-weighted}
                    trajectory MSE.  The cells direct/unweighted, unscaled/unweighted and scaled/1/eps^2
                    are the published models (same code path and seeds) and reproduce Table 7 exactly.
  --part pi         physics-informed scaled residual, lambda in {0.01, 0.1, 1.0}, under each weighting.
                    The penalty is the ODE residual of Eq. (18) in the units of the residual target,
                    weighted like the data term, so lambda = 0 is exactly the corresponding factorial
                    cell (same seed: a paired design).  Under the unweighted loss the objective equals the
                    published physics-informed objective up to a constant factor.
  --part noise      Table 14 under the common unweighted loss: direct, unscaled and scaled residual MLPs
                    (unweighted) and the published 1/eps^2-weighted scaled residual, at the noise levels
                    of Table 14 (same noise draws).
  --part ood        (Reviewer 3, M3) the six factorial models on the four regimes of Table 16 (the per-seed
                    regime datasets, seed OOD_DATA_SEED_BASE + seed index, 120 pairs) and at the horizons
                    T in {5, 10, 20} of run_feature_matched_ood.py, so that the extrapolation advantage can
                    be attributed to the parameterization or to the loss weighting.  The three published
                    cells reproduce the Table 16 rows.
  --part eps        Table 8 (error stratified by eps: eps = 0, the three in-distribution bins, the OOD range)
                    with the scaled residual trained with the unweighted loss next to the published direct
                    and 1/eps^2-weighted residual MLPs (same seeds and strata; the published rows reproduce).

Weights are applied in the z-scored target space of each parameterization and normalized to mean 1 on
the training split, so that the loss induced on u is the stated one (unweighted or 1/eps^2); validation
and early stopping use the same weighted data term.  Besides the unified metrics the script reports the
terminal error E_term = |u_hat(T_max) - u(T_max)| and the 1/eps^2-weighted trajectory MSE E_traj_w.

Run:  python experiments/run_objective_ablation.py --part factorial|pi|noise|ood|eps [--quick] [--tag NAME]
      python experiments/run_objective_ablation.py --tables-only      # summaries and LaTeX from saved shards
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from _torch_common import FEATURES, MLPFit, build_splits, config_dict, resolve, seed_of, torch_parser
from fisher_surrogate import latex
from fisher_surrogate.common import (NOISE_SEED_OFFSET, OOD_DATA_SEED_BASE, RESIDUAL_SEED_OFFSET, final_tables_dir,
                                     out_dir, save_results)
from fisher_surrogate.data import OOD_REGIMES, add_target_noise, make_ood_dataset
from fisher_surrogate.dynamics import u_ou
from fisher_surrogate.metrics import METRICS, compute_metrics, summarize_seeds
from fisher_surrogate.models.torch_models import train_mlp

PARAMS = {"direct": "Direct MLP", "unscaled": "Unscaled residual MLP", "scaled": "Scaled residual MLP"}
WEIGHTINGS = {"none": "unweighted", "inv_eps2": "$1/\\varepsilon^2$-weighted"}
NATIVE = {"direct": "none", "unscaled": "none", "scaled": "inv_eps2"}   # weighting of the published models
EXTRA = ["E_term", "E_traj_w"]


def target(df, param, col="u_clean"):
    u = df[col].to_numpy(float)
    if param == "direct":
        return u.reshape(-1, 1)
    r = u - u_ou(df["t"].to_numpy(float), df["u0"].to_numpy(float))
    return (r / df["eps"].to_numpy(float) if param == "scaled" else r).reshape(-1, 1)


def weights(train_df, val_df, param, weighting):
    """Per-sample weights in the z-scored target space of ``param`` such that the induced loss on u has
    the requested weighting; None when that is the plain loss of the parameterization."""
    if weighting == NATIVE[param]:
        return None, None

    def raw(df):
        eps = df["eps"].to_numpy(float)
        return eps ** 2 if param == "scaled" else 1.0 / eps ** 2   # (scaled -> unweighted), (others -> 1/eps^2)
    wt, wv = raw(train_df), raw(val_df)
    return wt / wt.mean(), wv / wt.mean()


def fit(train_df, val_df, param, weighting, seed, tcfg, lam=0.0, col="u_clean"):
    wt, wv = weights(train_df, val_df, param, weighting)
    x, xv = train_df[FEATURES].to_numpy(float), val_df[FEATURES].to_numpy(float)
    s = seed if param == "direct" else seed + RESIDUAL_SEED_OFFSET
    if lam > 0:
        assert param == "scaled"
        if wt is None:   # scaled + 1/eps^2: plain loss of the parameterization, weights 1
            wt, wv = np.ones(len(x)), np.ones(len(xv))
    model, xs, ys = train_mlp(x, target(train_df, param, col), xv, target(val_df, param, col), s, tcfg,
                              w_train=wt, w_val=wv, phys_lam=lam)
    return MLPFit(model, xs, ys, FEATURES, tcfg.device, residual=param != "direct", scaled=param == "scaled")


def evaluate(f, test_df, mkw):
    pred = test_df.copy()
    pred["u_pred"] = f.predict_frame(test_df)
    m, per = compute_metrics(pred, pred_fn_factory=f.pred_fn_factory if mkw.get("refine") else None,
                             return_per_trajectory=True, **mkw)
    eps = test_df.groupby("pair_id")["eps"].first().reindex(per["pair_id"]).to_numpy()
    if np.all(eps > 0):
        w = 1.0 / eps ** 2
        m["E_traj_w"] = float(np.sum(w * per["E_traj"].to_numpy()) / np.sum(w))
    else:   # the eps = 0 stratum of Table 8: the 1/eps^2-weighted error is undefined
        m["E_traj_w"] = float("nan")
    m.update(train_time_s=f.train_time_s, epochs_run=f.epochs_run, n_params=f.n_params,
             best_val=float(getattr(f.model, "best_val", np.nan)))
    return m


def part_factorial(train_df, val_df, test_df, seeds, tcfg, mkw):
    rows = []
    for i in seeds:
        for param in PARAMS:
            for weighting in WEIGHTINGS:
                print(f"seed {i:02d}  {param:8s}  {weighting}", flush=True)
                m = evaluate(fit(train_df, val_df, param, weighting, seed_of(i), tcfg), test_df, mkw)
                rows.append({"part": "factorial", "model": PARAMS[param], "loss": WEIGHTINGS[weighting], "lambda": 0.0,
                             "published": weighting == NATIVE[param], "seed": i, **m})
    return rows


def part_pi(train_df, val_df, test_df, seeds, tcfg, mkw, lambdas):
    rows = []
    for i in seeds:
        for weighting in WEIGHTINGS:
            for lam in lambdas:
                print(f"seed {i:02d}  PI lambda={lam:g}  {weighting}", flush=True)
                m = evaluate(fit(train_df, val_df, "scaled", weighting, seed_of(i), tcfg, lam=lam), test_df, mkw)
                rows.append({"part": "pi", "model": f"PI-Residual, $\\lambda={lam:g}$", "loss": WEIGHTINGS[weighting],
                             "lambda": lam, "published": False, "seed": i, **m})
    return rows


def part_noise(train_df, val_df, test_df, seeds, tcfg, mkw, levels):
    cells = [("direct", "none"), ("unscaled", "none"), ("scaled", "none"), ("scaled", "inv_eps2")]
    ref_std = train_df["u_clean"].std()
    rows = []
    for r in levels:
        for i in seeds:
            s = seed_of(i)
            tr = add_target_noise(train_df, r, s + NOISE_SEED_OFFSET, ref_std)    # the draws of Table 14
            va = add_target_noise(val_df, r, s + NOISE_SEED_OFFSET + 1, ref_std)
            for param, weighting in cells:
                print(f"noise {r:g} seed {i:02d}  {param:8s}  {weighting}", flush=True)
                m = evaluate(fit(tr, va, param, weighting, s, tcfg, col="u_target"), test_df, mkw)
                rows.append({"part": "noise", "noise": f"{100 * r:g}\\%", "model": PARAMS[param],
                             "loss": WEIGHTINGS[weighting], "lambda": 0.0, "published": weighting == NATIVE[param],
                             "seed": i, **m})
    return rows


def part_ood(train_df, val_df, test_df, seeds, tcfg, mkw, n_ood, n_time):
    from run_feature_matched_ood import horizon_frame, horizon_rows   # same horizon study as the other models
    H = horizon_frame(test_df)
    rows = []
    for i in seeds:
        oods = {r: make_ood_dataset(r, n_ood, n_time, seed=OOD_DATA_SEED_BASE + i) for r in OOD_REGIMES}
        for param in PARAMS:
            for weighting in WEIGHTINGS:
                print(f"ood seed {i:02d}  {param:8s}  {weighting}", flush=True)
                f = fit(train_df, val_df, param, weighting, seed_of(i), tcfg)
                base = {"part": "ood", "model": PARAMS[param], "loss": WEIGHTINGS[weighting], "lambda": 0.0,
                        "published": weighting == NATIVE[param], "seed": i}
                rows.append({**base, "split": "test", "regime": "in-distribution", **evaluate(f, test_df, mkw)})
                for regime, ood in oods.items():
                    rows.append({**base, "split": "ood", "regime": regime, **evaluate(f, ood, mkw)})
                rows += [{**base, **hr} for hr in horizon_rows(f, H)]
    return rows


def part_eps(train_df, val_df, test_df, seeds, tcfg, mkw, n_extra, n_time):
    from run_eps_ablation import eps_strata   # exactly the strata of Table 8
    strata = eps_strata(test_df, n_time, n_extra)
    rows = []
    for i in seeds:
        for param, weighting in [("direct", "none"), ("scaled", "none"), ("scaled", "inv_eps2")]:
            print(f"eps seed {i:02d}  {param:8s}  {weighting}", flush=True)
            f = fit(train_df, val_df, param, weighting, seed_of(i), tcfg)
            for st, data in strata.items():
                rows.append({"part": "eps", "eps_range": st, "model": PARAMS[param], "loss": WEIGHTINGS[weighting],
                             "lambda": 0.0, "published": weighting == NATIVE[param], "seed": i,
                             "n_pairs": data.pair_id.nunique(), **evaluate(f, data, mkw)})
    return rows


def summaries(out, quick=False):
    shards = sorted(glob.glob(str(out / "shard_*_seed_results.csv")))
    if not shards:
        raise SystemExit(f"no shards in {out}")
    seed_df = pd.concat([pd.read_csv(f) for f in shards], ignore_index=True)
    for part, g in seed_df.groupby("part", sort=False):
        g = _ints(g.dropna(axis=1, how="all"))
        if part == "ood":
            main = _ints(g[g.split != "horizon"].dropna(axis=1, how="all"))
            summary = summarize_seeds(main, ["split", "regime", "model", "loss"], metrics=METRICS + EXTRA)
            save_results(out, part, main, summary)
            hz = g[g.split == "horizon"].dropna(axis=1, how="all")
            hsum = summarize_seeds(hz, ["regime", "model", "loss"], metrics=["A_T", "E_T", "D_T"])
            save_results(out, "ood_horizon", hz, hsum)
            text = ood_tex(summary, hsum)
        elif part == "eps":
            summary = summarize_seeds(g, ["eps_range", "model", "loss"], metrics=METRICS + EXTRA)
            save_results(out, part, g, summary)
            text = eps_tex(summary)
        else:
            keys = (["noise"] if part == "noise" else []) + ["model", "loss"]
            summary = summarize_seeds(g, keys, metrics=METRICS + EXTRA)
            save_results(out, part, g, summary)
            text = rows_tex(part, summary)
        (final_tables_dir(quick) / f"round2_objective_{part}.tex").write_text(text, encoding="utf-8")


def _ints(df):
    """Count columns back to integers: merging parts whose rows lack them (the horizon rows) makes them float."""
    df = df.copy()
    for c in ("seed", "n_fall", "n_cross", "n_cross_fail", "n_cross_spurious", "n_traj", "epochs_run", "n_params",
              "n_pairs"):
        if c in df and df[c].notna().all():
            df[c] = df[c].astype(int)
    return df


def eps_tex(summary):
    """Table 8 layout: eps range & model & loss & four metrics & n_fail, then the direct/residual E_traj ratios."""
    lines = ["% Table 8 (tab:eps_ablation) with the scaled residual MLP trained with the unweighted loss as well;",
             "% the Direct MLP and the 1/eps^2-weighted Residual MLP rows are the published ones (same seeds).",
             "% Mean +/- 95% CI over S = 10 seeds; last column n_fail ('--': no true crossing in the stratum).",
             "% Generated by: python experiments/run_objective_ablation.py --tables-only"]
    for st, g in summary.groupby("eps_range", sort=False):
        lines.append(f"% --- {st}")
        for _, r in g.iterrows():
            cells = [latex.cell(r[f"{m}_mean"], r[f"{m}_ci"]) for m in METRICS]
            nf = "--" if int(r["n_cross_total"]) == 0 else str(int(r["n_cross_fail_total"]))
            lines.append(" & ".join([st, r["model"], r["loss"]] + cells + [nf]) + r" \\")
        d = g[g.model == PARAMS["direct"]].iloc[0]
        for _, r in g[g.model == PARAMS["scaled"]].iterrows():
            if r["E_traj_mean"] > 0:
                sep = d["E_traj_mean"] - d["E_traj_ci"] > r["E_traj_mean"] + r["E_traj_ci"]
                lines.append(f"%     direct / residual ({r['loss']}) on E_traj: {d['E_traj_mean'] / r['E_traj_mean']:.3g}x; "
                             f"95% intervals {'separated' if sep else 'overlap'}")
    return "\n".join(lines) + "\n"


def ood_tex(summary, hsum):
    """Per OOD regime: model & loss & four metrics & n_fail; then the horizon study A_T at T = 5, 10, 20."""
    lines = ["% Objective ablation out of distribution (Reviewer 3, M3): the six factorial models on the per-seed",
             "% regime datasets of Table 16; mean +/- 95% CI over S = 10 seeds; last column n_fail ('--': no true crossing).",
             "% Generated by: python experiments/run_objective_ablation.py --tables-only"]
    s = summary[summary.split == "ood"]
    for regime in OOD_REGIMES:
        lines.append(f"% --- {regime}")
        for _, r in s[s.regime == regime].iterrows():
            cells = [latex.cell(r[f"{m}_mean"], r[f"{m}_ci"]) for m in METRICS]
            nf = "--" if int(r["n_cross_total"]) == 0 else str(int(r["n_cross_fail_total"]))
            lines.append(" & ".join([r["model"], r["loss"]] + cells + [nf]) + r" \\")
    lines.append("% --- Horizon study on the standard test pairs: A_T = |u_hat(T) - u*_eps| at T = 5, 10, 20")
    for (model, loss), g in hsum.groupby(["model", "loss"], sort=False):
        a = {row.regime: latex.cell(row.A_T_mean, row.A_T_ci) for row in g.itertuples()}
        lines.append(" & ".join([model, loss] + [a.get(f"T={T}", "--") for T in (5, 10, 20)]) + r" \\")
    return "\n".join(lines) + "\n"


def rows_tex(part, summary):
    """Manuscript-layout rows: mean +/- 95% CI for the four metrics, then E_term and the weighted E_traj."""
    head = {"factorial": "% Objective ablation (Reviewer 3, M1): output parameterization x loss weighting, S = 10 seeds.",
            "pi": "% Physics-informed residual under each data objective, S = 10 seeds; lambda = 0 rows are the "
                  "scaled-residual rows of the factorial table (same seeds).",
            "noise": "% Noise sensitivity (Table 14 setup) under the common unweighted loss, S = 10 seeds."}[part]
    lines = [head, f"% Generated by: python experiments/run_objective_ablation.py --tables-only",
             "% Columns: " + ("noise & " if part == "noise" else "") +
             "model & loss & E_traj & E_cross & E_over & E_eq & E_term & E_traj (1/eps^2-weighted) & n_fail"]
    for _, r in summary.iterrows():
        lead = ([r["noise"]] if part == "noise" else []) + [r["model"], r["loss"]]
        cells = [latex.cell(r[f"{m}_mean"], r.get(f"{m}_ci")) for m in METRICS + EXTRA]
        lines.append(" & ".join(lead + cells + [str(int(r.get("n_cross_fail_total", 0)))]) + r" \\")
    return "\n".join(lines) + "\n"


def main():
    p = torch_parser(__doc__)
    p.add_argument("--part", choices=["factorial", "pi", "noise", "ood", "eps"])
    p.add_argument("--n-ood-pairs", type=int, default=120)
    p.add_argument("--n-extra-pairs", type=int, default=120, help="pairs for the eps=0 and OOD sets of Table 8")
    p.add_argument("--lambdas", type=float, nargs="*", default=[0.01, 0.1, 1.0])
    p.add_argument("--levels", type=float, nargs="*", default=[0.0, 0.001, 0.01, 0.05])
    p.add_argument("--tag", default=None, help="shard name (default: the part)")
    p.add_argument("--seeds", type=int, nargs="*", default=None, help="seed indices (default: all)")
    p.add_argument("--tables-only", action="store_true")
    p.add_argument("--no-summary", action="store_true", help="write the shard only (parallel runs)")
    args = p.parse_args()
    out = out_dir(args, "objective_ablation")
    if args.tables_only:
        summaries(out, args.quick)
        return
    if args.part is None:
        raise SystemExit("--part is required (or --tables-only)")
    n_seeds, dcfg, tcfg, mkw = resolve(args)
    train_df, val_df, test_df = build_splits(dcfg)
    seeds = args.seeds if args.seeds else list(range(n_seeds))
    if args.part == "factorial":
        rows = part_factorial(train_df, val_df, test_df, seeds, tcfg, mkw)
    elif args.part == "pi":
        rows = part_pi(train_df, val_df, test_df, seeds, tcfg, mkw, args.lambdas)
    elif args.part == "ood":
        rows = part_ood(train_df, val_df, test_df, seeds, tcfg, mkw, 12 if args.quick else args.n_ood_pairs, dcfg.n_time)
    elif args.part == "eps":
        rows = part_eps(train_df, val_df, test_df, seeds, tcfg, mkw, 12 if args.quick else args.n_extra_pairs, dcfg.n_time)
    else:
        rows = part_noise(train_df, val_df, test_df, seeds, tcfg, mkw, args.levels)
    tag = args.tag or args.part
    pd.DataFrame(rows).to_csv(out / f"shard_{tag}_seed_results.csv", index=False)
    cfg = config_dict(args, dcfg, tcfg, n_seeds, mkw, native_weighting=NATIVE,
                      seeds={"direct": "2026+i", "residual and PI": "2026+i+5000", "noise draws": "2026+i+10000(+1 val)"})
    Path(out / f"shard_{tag}_config.json").write_text(json.dumps(cfg, indent=2, default=str))
    if not args.no_summary:
        summaries(out, args.quick)


if __name__ == "__main__":
    main()
