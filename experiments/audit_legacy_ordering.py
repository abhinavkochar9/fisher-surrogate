"""Metric-definition audit behind the unified-definitions paragraph: for every model-comparison table,
compare the within-table model ordering under the unified metric definitions (results/) with the
ordering under the pre-revision definitions (results/legacy_metrics_audit/, produced by
run_legacy_audit.sh with the same code, seeds and data). Models are ordered by the mean of each metric.

Writes results/legacy_metrics_audit/ordering_cells.csv (every table x group x metric cell) and
ordering_by_table.csv.   Run:  python experiments/audit_legacy_ordering.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fisher_surrogate.common import RESULTS_DIR  # noqa: E402

METRICS = ["E_traj", "E_cross", "E_over", "E_eq"]
LEGACY = RESULTS_DIR / "legacy_metrics_audit"
# manuscript label -> [(results dir, file stem, key column, {model renames} or None, {keep only} or None)], group column
TABLES = {
    "tab:surrogate_results": ([("sklearn_tables", "table4_baselines", "model", None, None)], None),
    "tab:surrogate_ablation": ([("sklearn_tables", "table6_ablation", "variant", None, None)], None),
    "tab:physics_informed_residual": ([("physics_informed", "physics_informed", "model", None, None)], None),
    "tab:deeponet_baseline": ([("deeponet", "deeponet", "model", None, None)], None),
    "tab:noise_sensitivity": ([("noise_sensitivity", "noise_sensitivity", "model", None, None)], "noise"),
    "tab:eps_ablation": ([("eps_ablation", "eps_ablation", "model", None, None)], "eps_range"),
    "tab:feature_matched": ([("feature_matched", "feature_matched", "model", None, None)], None),
    "tab:node_gp_baselines": ([("neural_ode_gp", "neural_ode_gp", "model", None, None)], None),
    "tab:phys_convergence": ([("pi_gradient_analysis", "pi_gradient", "model", None, None)], None),
    "tab:ood_generalization": ([("ood_generalization", "ood_generalization", "model", None, None),
                                ("gp_offdistribution", "gp_ood", "model", {"GP, zero mean": "Gaussian process"}, {"GP, zero mean"})], "regime"),
    "tab:transfer": ([("transfer_system", "transfer_system", "model", None, None),
                      ("gp_offdistribution", "gp_transfer", "model", {"GP, zero mean": "Gaussian process"}, {"GP, zero mean"})], None),
}


def load(root, parts):
    frames = []
    for d, stem, key, ren, keep in parts:
        s = pd.read_csv(root / d / f"{stem}_summary.csv")
        if keep:
            s = s[s[key].isin(keep)]
        s = s.rename(columns={key: "model"})
        if ren:
            s["model"] = s["model"].replace(ren)
        frames.append(s)
    return pd.concat(frames, ignore_index=True)


def main():
    cells = []
    for label, (parts, grp) in TABLES.items():
        U, L = load(RESULTS_DIR, parts), load(LEGACY, parts)
        keys = [None] if grp is None else list(dict.fromkeys(U[grp]))
        for k in keys:
            su = U if k is None else U[U[grp] == k]
            sl = L if k is None else L[L[grp] == k]
            for m in METRICS:
                cu = su.set_index("model")[f"{m}_mean"].dropna(); cl = sl.set_index("model")[f"{m}_mean"].dropna()
                common = [x for x in cu.index if x in cl.index]
                if len(common) < 2:
                    continue
                ou = list(cu.loc[common].sort_values().index); ol = list(cl.loc[common].sort_values().index)
                cells.append({"table": label, "group": "-" if k is None else k, "metric": m, "changed": ou != ol,
                              "unified_order": " < ".join(ou), "legacy_order": " < ".join(ol)})
    c = pd.DataFrame(cells)
    c.to_csv(LEGACY / "ordering_cells.csv", index=False)
    by = c.groupby("table", sort=False).agg(cells=("changed", "size"), changed=("changed", "sum")).reset_index()
    by.to_csv(LEGACY / "ordering_by_table.csv", index=False)
    print(by.to_string(index=False))
    print(f"\n{int(c.changed.sum())} of {len(c)} table x group x metric cells change ordering")


if __name__ == "__main__":
    main()
