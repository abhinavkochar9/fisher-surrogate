"""Revised paper Tables 4, 5, 6 -- scikit-learn baselines (linear, polynomial, direct MLP),
residual-MLP generalization across regimes, and the architecture / normalization ablation.

Revision changes: every neural row is run over S = 10 seeds (mean +/- 95% CI); linear and
polynomial regression are deterministic given the split; ALL rows use the unified metric
definitions (missing crossing -> T_max with failure count, |max - max| excursion,
root-finding refinement).  The linear-regression crossing time is therefore a number.

Run:  python experiments/run_sklearn_tables.py [--quick] [--preset generalization|ablation]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fisher_surrogate import latex  # noqa: E402
from fisher_surrogate.common import experiment_parser, out_dir, save_results  # noqa: E402
from fisher_surrogate.data import FEATURES, make_dataset, split_pair_ids  # noqa: E402
from fisher_surrogate.metrics import (LEGACY_SKLEARN_METRIC_KW, UNIFIED_METRIC_KW,  # noqa: E402
                                      compute_metrics, summarize_seeds)
from fisher_surrogate.models.sklearn_models import (  # noqa: E402
    ResidualOUSurrogate, build_linear, build_mlp, build_polynomial)

REGIMES = [
    ("Interpolation", None, None),
    (r"Low-$u_{0}$ extrapolation", (0.05, 0.2), (0.01, 0.5)),
    (r"High-$u_{0}$ extrapolation", (2.5, 4.0), (0.01, 0.5)),
    (r"High-$\varepsilon$ extrapolation", (0.2, 2.5), (0.5, 0.8)),
]

ABLATION = [
    ("Small MLP $(32)$", "direct", (32,), True, True),
    ("Medium MLP $(64,64)$", "direct", (64, 64), True, True),
    ("Residual MLP $(64,64)$", "residual", (64, 64), True, True),
    ("Large MLP $(128,128)$", "direct", (128, 128), True, True),
    ("Medium MLP w/o input scaling", "direct", (64, 64), False, True),
    ("Medium MLP w/o target scaling", "direct", (64, 64), True, False),
]


def evaluate(model, df, **kw):
    pred = df.copy()
    pred["u_pred"] = model.predict(df[FEATURES].to_numpy(float))

    def factory(g):
        row = g.iloc[0]
        return lambda t: model.predict(np.column_stack([np.full(len(t), row["u0"]), np.full(len(t), row["eps"]),
                                                       np.asarray(t, float)]))
    return compute_metrics(pred, pred_fn_factory=factory, **kw)


def main():
    p = experiment_parser(__doc__)
    p.add_argument("--preset", default="generalization", choices=["generalization", "ablation"])
    p.add_argument("--legacy-metrics", action="store_true",
                   help="use the pre-revision sklearn definitions (exclude missing crossings, "
                        "no root-finding refinement) -- for auditing only")
    p.add_argument("--n-pairs", type=int, default=600)
    p.add_argument("--n-time", type=int, default=201)
    args = p.parse_args()
    n_seeds = args.n_seeds or 10
    if args.quick:
        args.n_pairs, args.n_time, n_seeds = 60, 41, 2
    out = out_dir(args, "sklearn_tables")
    mkw = LEGACY_SKLEARN_METRIC_KW if args.legacy_metrics else UNIFIED_METRIC_KW

    df = make_dataset(args.n_pairs, args.n_time, seed=100, integrator="adaptive")
    tr, va, te = split_pair_ids(args.n_pairs, 0.70, 0.15, seed=123)
    train_df = pd.concat([df[df.pair_id.isin(tr)], df[df.pair_id.isin(va)]])  # sklearn does its own val split
    test_df = df[df.pair_id.isin(te)].copy()
    X_train, y_train = train_df[FEATURES].to_numpy(float), train_df["u_clean"].to_numpy(float)
    grid_dt = args.n_time and 5.0 / (args.n_time - 1)
    print(f"train pairs {len(tr) + len(va)}, test pairs {len(te)}, grid {args.n_time} (dt = {grid_dt:.4f})")
    overrides = {"max_iter": 200} if args.quick else {}

    # --- Table 4: baselines ---
    rows = []
    for name, model in [("Linear regression", build_linear()), ("Polynomial regression", build_polynomial(3))]:
        t0 = time.perf_counter(); model.fit(X_train, y_train)
        rows.append({"model": name, "seed": 0, **evaluate(model, test_df, **mkw), "train_time_s": time.perf_counter() - t0})
    for i in range(n_seeds):
        model = build_mlp((64, 64), random_state=12 + i, preset=args.preset, **overrides)
        t0 = time.perf_counter(); model.fit(X_train, y_train)
        rows.append({"model": "Direct MLP surrogate", "seed": i, **evaluate(model, test_df, **mkw),
                     "train_time_s": time.perf_counter() - t0})
    t4 = pd.DataFrame(rows)
    s4 = summarize_seeds(t4, ["model"])
    save_results(out, "table4_baselines", t4, s4, latex.table_ci(
        s4, ["model"], "Baseline surrogate evaluation on held-out Fisher-regularized variance trajectories "
        "(linear/polynomial deterministic given the split; direct MLP mean $\\pm$ 95\\% CI over "
        f"$S={n_seeds}$ seeds; unified metric definitions).", "tab:surrogate-results"))

    # --- Table 5: residual-MLP generalization, S seeds ---
    rows = []
    n_reg = 12 if args.quick else 120
    regime_data = {name: (test_df if u0r is None else make_dataset(n_reg, args.n_time, u0r, epsr, seed=201 + k,
                                                                    integrator="adaptive"))
                   for k, (name, u0r, epsr) in enumerate(REGIMES)}
    for i in range(n_seeds):
        t0 = time.perf_counter()
        residual = ResidualOUSurrogate((64, 64), random_state=16 + i, preset=args.preset, **overrides).fit(X_train, y_train)
        tt = time.perf_counter() - t0
        for name, data in regime_data.items():
            rows.append({"regime": name, "seed": i, **evaluate(residual, data, **mkw), "train_time_s": tt})
    t5 = pd.DataFrame(rows)
    s5 = summarize_seeds(t5, ["regime"])
    save_results(out, "table5_generalization", t5, s5, latex.table_ci(
        s5, ["regime"], "Generalization of the structure-aware residual MLP across interpolation and selected "
        f"extrapolation regimes (mean $\\pm$ 95\\% CI over $S={n_seeds}$ seeds).", "tab:surrogate-generalization"))

    # --- Table 6: ablation, S seeds ---
    rows = []
    for name, kind, layers, in_s, tg_s in ABLATION:
        for i in range(n_seeds):
            rs = 11 + i
            model = (build_mlp(layers, in_s, tg_s, rs, args.preset, **overrides) if kind == "direct"
                     else ResidualOUSurrogate(layers, in_s, tg_s, rs, args.preset, **overrides))
            t0 = time.perf_counter(); model.fit(X_train, y_train)
            rows.append({"variant": name, "seed": i, **evaluate(model, test_df, **mkw), "train_time_s": time.perf_counter() - t0})
    t6 = pd.DataFrame(rows)
    s6 = summarize_seeds(t6, ["variant"])
    save_results(out, "table6_ablation", t6, s6, latex.table_ci(
        s6, ["variant"], "Ablation over architecture, residual parameterization, and normalization "
        f"(mean $\\pm$ 95\\% CI over $S={n_seeds}$ seeds).", "tab:surrogate-ablation"))
    save_results(out, "sklearn_tables", config={**vars(args), "n_seeds": n_seeds, "metrics": mkw,
                                                "grid_resolution": grid_dt})


if __name__ == "__main__":
    main()
