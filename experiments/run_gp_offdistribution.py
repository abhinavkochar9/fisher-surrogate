"""Revised paper -- Gaussian-process baselines off-distribution (addendum to Tables 16 and 20).

The GP of Tables 10 and 13 was only ever evaluated *in distribution* on the Fisher benchmark,
where it is competitive with the scaled residual MLP.  R4.3 and R5.2 ask whether the
structure-aware advantage is architectural rather than an artefact of privileged information,
so the comparison has to be made where the construction is actually claimed to help.  This
script evaluates the same two GP variants in the two off-distribution settings already used
elsewhere in the paper, with identical splits, seeds and metric conventions:

    OOD (Table 16 regimes)  GP, zero mean  /  GP, u_OU prior mean.  Trained once on the
                            standard split and evaluated on each regime without retraining,
                            exactly as the direct and residual MLPs are in Table 16.
    Transfer (Table 20)     GP, zero mean  /  GP, y_base prior mean (Kennedy--O'Hagan) on
                            (c, gamma, t), against the same held-out pairs as Table 20.

The GP is deterministic given its subsample, so the seed index varies only the subsample (and,
for OOD, selects the same per-seed regime datasets the MLPs saw).

Run:  python experiments/run_gp_offdistribution.py [--quick] [--gp-sub 2000] [--gp-seeds 3]
"""
from __future__ import annotations

import pandas as pd

from _baselines_common import fit_gp
from _torch_common import config_dict, evaluate, resolve, seed_of, torch_parser
from run_transfer_system import TRANSFER_KW, prepare, split
from fisher_surrogate import latex, transfer
from fisher_surrogate.common import OOD_DATA_SEED_BASE, out_dir, save_results
from fisher_surrogate.data import OOD_REGIMES, build_splits, make_ood_dataset
from fisher_surrogate.metrics import summarize_seeds

OOD_VARIANTS = [("zero", "GP, zero mean"), ("ou", "GP, $u_{\\mathrm{OU}}$ prior mean")]
TRANSFER_VARIANTS = [("zero", "GP, zero mean"), ("column", "GP, $y_{\\mathrm{base}}$ prior mean")]


def run_ood(args, dcfg, mkw, gp_seeds, gp_sub, n_ood, out):
    train_df, _, _ = build_splits(dcfg)
    rows = []
    for i in range(gp_seeds):
        for pm, name in OOD_VARIANTS:
            print(f"[OOD] subsample seed {i:02d}  {name}", flush=True)
            gp = fit_gp(train_df, seed_of(i), prior_mean=pm, n_sub=gp_sub)
            for regime in OOD_REGIMES:
                ood = make_ood_dataset(regime, n_ood, dcfg.n_time, seed=OOD_DATA_SEED_BASE + i)
                m, _ = evaluate(gp, ood, mkw)
                rows.append({"regime": regime, "model": name, "seed": i, **m,
                             "gp_kernel": gp.kernel_summary(), "gp_n_sub": gp.n_train_points})
    seed_df = pd.DataFrame(rows)
    summary = summarize_seeds(seed_df, ["regime", "model"])
    tex = latex.table_ci(summary, ["regime", "model"],
                         "Gaussian-process baselines on the out-of-distribution regimes of Table~15 "
                         f"(mean $\\pm$ 95\\% CI over $S={gp_seeds}$ subsample seeds).", "tab:gp_ood")
    save_results(out, "gp_ood", seed_df, summary, tex)
    print(summary.to_string(index=False))
    return seed_df


def run_transfer(args, dcfg, mkw, gp_seeds, gp_sub, n_pairs, out):
    df = prepare(transfer.make_dataset(n_pairs, dcfg.n_time))
    train_df, _, test_df = split(df, n_pairs)
    rows = []
    for i in range(gp_seeds):
        for pm, name in TRANSFER_VARIANTS:
            print(f"[transfer] subsample seed {i:02d}  {name}", flush=True)
            gp = fit_gp(train_df, seed_of(i), prior_mean=pm, n_sub=gp_sub, target="y_clean",
                        features=transfer.FEATURES, baseline_col="u_base" if pm == "column" else None)
            m, pred = evaluate(gp, test_df, mkw, **TRANSFER_KW)
            rows.append({"model": name, "seed": i, **m,
                         "gp_kernel": gp.kernel_summary(), "gp_n_sub": gp.n_train_points})
            if i == 0 and pm == "zero":   # predictions of the seed-0 GP for the transfer-overshoot figure
                keep = [c for c in ("pair_id", "c", "gamma", "t", "y_clean", "y_base", "level", "T_cross_true") if c in test_df]
                fr = test_df[keep].copy(); fr["Gaussian process"] = pred["u_pred"].to_numpy()
                fr.to_csv(out / "gp_transfer_predictions_seed0.csv", index=False)
                if getattr(args, "pred_only", False):
                    print("saved gp_transfer_predictions_seed0.csv (--pred-only)"); print(m); return None
    seed_df = pd.DataFrame(rows)
    summary = summarize_seeds(seed_df, ["model"])
    tex = latex.table_ci(summary, ["model"],
                         "Gaussian-process baselines on the non-normal transfer system of Table~18 "
                         f"(mean $\\pm$ 95\\% CI over $S={gp_seeds}$ subsample seeds).", "tab:gp_transfer")
    save_results(out, "gp_transfer", seed_df, summary, tex)
    print(summary.to_string(index=False))
    return seed_df


def main():
    p = torch_parser(__doc__)
    p.add_argument("--n-ood-pairs", type=int, default=120)
    p.add_argument("--transfer-pairs", type=int, default=400)
    p.add_argument("--gp-sub", type=int, default=2000)
    p.add_argument("--gp-seeds", type=int, default=10)
    p.add_argument("--axes", nargs="*", default=["ood", "transfer"])
    p.add_argument("--pred-only", action="store_true",
                   help="transfer axis: fit only the seed-0 zero-mean GP and save its predictions (no summaries)")
    args = p.parse_args()
    n_seeds, dcfg, tcfg, mkw = resolve(args)
    gp_sub = 300 if args.quick else args.gp_sub
    gp_seeds = 1 if args.quick else args.gp_seeds
    n_ood = 12 if args.quick else args.n_ood_pairs
    n_pairs = dcfg.n_pairs if args.quick else args.transfer_pairs
    out = out_dir(args, "gp_offdistribution")
    if "ood" in args.axes:
        run_ood(args, dcfg, mkw, gp_seeds, gp_sub, n_ood, out)
    if "transfer" in args.axes:
        run_transfer(args, dcfg, mkw, gp_seeds, gp_sub, n_pairs, out)
    save_results(out, "gp_offdistribution",
                 config=config_dict(args, dcfg, tcfg, n_seeds, mkw, gp_sub=gp_sub, gp_seeds=gp_seeds,
                                    n_ood_pairs=n_ood, transfer_pairs=n_pairs))


if __name__ == "__main__":
    main()
