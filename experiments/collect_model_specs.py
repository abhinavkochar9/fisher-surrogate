"""Revised paper Table 3 -- model specifications: architecture, exact parameter count,
optimizer, LR, epochs / early-stopping rule, batch size, solver tolerances, GP subsample,
and mean training time per seed, collected from the result CSVs of the other scripts.
Also writes the Time/seed column of Table 3 in the manuscript's row order to tables/final/table3_timing.tex.

Run after the experiments:  python experiments/collect_model_specs.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fisher_surrogate.common import RESULTS_DIR, final_tables_dir  # noqa: E402

SOURCES = {  # display name: (results subdir, seed_results file, model label in that file)
    "Linear regression": ("sklearn_tables", "table4_baselines_seed_results.csv", "Linear regression"),
    "Polynomial (deg. 3)": ("sklearn_tables", "table4_baselines_seed_results.csv", "Polynomial regression"),
    "Direct MLP": ("physics_informed", "physics_informed_seed_results.csv", "Direct MLP"),
    "Residual MLP (unscaled)": ("physics_informed", "physics_informed_seed_results.csv", "Unscaled residual MLP"),
    "Residual MLP (scaled)": ("physics_informed", "physics_informed_seed_results.csv", "Scaled residual MLP (proposed)"),
    "PI-Residual ($\\lambda=0.1$)": ("physics_informed", "physics_informed_seed_results.csv", "PI-Residual, $\\lambda=0.1$"),
    "DeepONet-style": ("deeponet", "deeponet_seed_results.csv", "DeepONet-style baseline"),
    # Neural ODE rows from the Table 13 runs, where the two solver tolerances are compared
    "Neural ODE": ("neural_ode_gp", "neural_ode_gp_seed_results.csv", "Neural ODE (standard: rtol=1e-06, atol=1e-08)"),
    "Neural ODE, tight tolerance": ("neural_ode_gp", "neural_ode_gp_seed_results.csv", "Neural ODE (tight: rtol=1e-08, atol=1e-10)"),
    "Residual Neural ODE": ("neural_ode_gp", "neural_ode_gp_seed_results.csv", "Residual Neural ODE (standard: rtol=1e-06, atol=1e-08)"),
    "Direct MLP + $u_{\\mathrm{OU}}$ input": ("feature_matched", "feature_matched_seed_results.csv", "Direct MLP + $u_{\\mathrm{OU}}$ input"),
    "Gaussian process": ("feature_matched", "feature_matched_seed_results.csv", "GP, zero mean"),
    "GP with OU prior mean": ("feature_matched", "feature_matched_seed_results.csv", "GP, $u_{\\mathrm{OU}}$ prior mean"),
}
STATIC = {
    "Linear regression": ("OLS on $(u_0,\\varepsilon,t)$", "closed form", "--", "--", "--"),
    "Polynomial (deg. 3)": ("OLS on cubic features", "closed form", "--", "--", "--"),
    "Direct MLP": ("$(u_0,\\varepsilon,t)\\to(64,64)\\to u$, tanh", "Adam, $2\\times10^{-3}$ const.", "1500 / pat. 150", "512", "--"),
    "Residual MLP (unscaled)": ("$u_{\\mathrm{OU}}+r_\\theta$, same net", "same", "same", "512", "--"),
    "Residual MLP (scaled)": ("$u_{\\mathrm{OU}}+\\varepsilon r_\\theta$, same net", "same", "same", "512", "--"),
    "PI-Residual ($\\lambda=0.1$)": ("scaled residual $+\\lambda\\mathcal L_{\\mathrm{phys}}$", "same", "same", "512", "autograd $\\partial_t$"),
    "DeepONet-style": ("branch $(u_0,\\varepsilon)\\to(64,64)\\to64$; trunk $t\\to(64,64)\\to64$", "same", "same", "512", "--"),
    "Neural ODE": ("$g_\\phi(u,\\varepsilon,t)$: (64,64), tanh", "same", "same", "full batch (pairs)", "dopri5"),
    "Neural ODE, tight tolerance": ("same", "same", "same", "full batch (pairs)", "dopri5"),
    "Residual Neural ODE": ("$2(1-u)+\\varepsilon g_\\phi$, same net", "same", "same", "full batch (pairs)", "dopri5"),
    "Direct MLP + $u_{\\mathrm{OU}}$ input": ("$(u_0,\\varepsilon,t,u_{\\mathrm{OU}})\\to(64,64)\\to u$", "same", "same", "512", "--"),
    "Gaussian process": ("RBF-ARD + white noise, zero mean", "L-BFGS marginal lik.", "--", "$n_{\\mathrm{sub}}$", "--"),
    "GP with OU prior mean": ("RBF-ARD, mean $u_{\\mathrm{OU}}$", "same", "--", "same", "--"),
}


def main():
    rows = []
    for name, (sub, fname, label) in SOURCES.items():
        path = RESULTS_DIR / sub / fname
        row = {"model": name, "architecture": STATIC[name][0], "optimizer": STATIC[name][1],
               "epochs_stop": STATIC[name][2], "batch": STATIC[name][3], "solver": STATIC[name][4],
               "n_params": np.nan, "time_per_seed_s": np.nan, "epochs_run_mean": np.nan}
        if path.exists():
            df = pd.read_csv(path)
            g = df[df.model == label]
            if len(g):
                row["n_params"] = g["n_params"].iloc[0] if "n_params" in g else np.nan
                row["time_per_seed_s"] = g["train_time_s"].mean() if "train_time_s" in g else np.nan
                row["epochs_run_mean"] = g["epochs_run"].mean() if "epochs_run" in g else np.nan
                if "rtol" in g and g["rtol"].notna().any():
                    row["solver"] = f"dopri5, rtol {g['rtol'].iloc[0]:g} / atol {g['atol'].iloc[0]:g}"
                if "gp_n_sub" in g and g["gp_n_sub"].notna().any():
                    row["batch"] = f"$n_{{\\mathrm{{sub}}}}={int(g['gp_n_sub'].iloc[0])}$"
                    row["architecture"] += f" ({int(g['n_params'].iloc[0])} hyper.)"
        else:
            print(f"missing {path} -- run the corresponding experiment first")
        rows.append(row)
    specs = pd.DataFrame(rows)
    out = RESULTS_DIR / "model_specs"
    out.mkdir(parents=True, exist_ok=True)
    specs.to_csv(out / "model_specs.csv", index=False)
    lines = [r"\begin{table}[t]", r"\centering", r"\scriptsize",
             r"\caption{Model specifications and training protocol for all surrogates (parameter counts exact; training "
             r"time = mean wall-clock per seed on the machine reported in the repository README).}",
             r"\label{tab:model_specs}", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{llrllllr}", r"\hline",
             r"Model & Architecture / inputs & \#params & Optimizer, LR & Epochs / stop & Batch & Solver / tolerances & Time per seed \\", r"\hline"]
    for _, r in specs.iterrows():
        npar = "--" if np.isnan(r.n_params) else f"{int(r.n_params):,}"
        tm = "$<1$ s" if (not np.isnan(r.time_per_seed_s) and r.time_per_seed_s < 1) else (
            "--" if np.isnan(r.time_per_seed_s) else f"{r.time_per_seed_s:.0f} s")
        lines.append(f"{r.model} & {r.architecture} & {npar} & {r.optimizer} & {r.epochs_stop} & {r.batch} & {r.solver} & {tm} \\\\")
    lines += [r"\hline", r"\end{tabular}%", "}", r"\end{table}"]
    (out / "model_specs_table.tex").write_text("\n".join(lines), encoding="utf-8")
    timing_table(specs)
    print(specs.to_string())


# Table 3 row name in the manuscript -> row of model_specs.csv
TABLE3_ROWS = [("Direct MLP", "Direct MLP"), ("Residual MLP (unscaled)", "Residual MLP (unscaled)"),
               ("Residual MLP (scaled)", "Residual MLP (scaled)"), ("PI-Residual", "PI-Residual ($\\lambda=0.1$)"),
               ("DeepONet-style", "DeepONet-style"), ("Neural ODE", "Neural ODE"),
               ("\\quad tight tolerance", "Neural ODE, tight tolerance"), ("Residual Neural ODE", "Residual Neural ODE"),
               ("Direct MLP $+u_{\\mathrm{OU}}$", "Direct MLP + $u_{\\mathrm{OU}}$ input"),
               ("Gaussian process", "Gaussian process"), ("GP $+$ OU prior mean", "GP with OU prior mean")]


def timing_table(specs):
    t = specs.set_index("model").time_per_seed_s
    lines = ["% Table 3 (tab:model_specs), Time/seed column: mean wall-clock training time per seed from the per-seed",
             "% result files (GP: L-BFGS fit on one n_sub subsample, on an otherwise idle machine).",
             "% Generated by: python experiments/collect_model_specs.py",
             r"\begin{tabular}{lr}", r"\toprule", r"Model & Time / seed \\", r"\midrule"]
    for row, key in TABLE3_ROWS:
        v = t.get(key, np.nan)
        lines.append(f"{row} & " + ("--" if np.isnan(v) else f"${v:.0f}$\\,s") + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (final_tables_dir() / "table3_timing.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
