"""Table 2 (tab:nongaussian_results), Figure 3 (fig:nongaussian_variance) data, and Appendix Table C.21
(tab:pde_convergence) -- direct simulation of the Fisher-regularized Fokker--Planck PDE for Gaussian and
non-Gaussian initial densities (reviewers R8.2, R5.3, R5.4).

For each initial density (all with the same initial variance u0 so that shape is the only
difference) the script records at every output time the variance u(t), the cross-dissipation
D_x(t) = -int rho d_x mu_0 d_x mu_F dx, and then reports

    T_acc   first crossing of u = 1                   (regime transition via the variance)
    T_x     first zero of D_x(t)                      (actual cooperation -> interference switch)
    max excess   max_t u(t) - 1
    u_inf   variance after long-time integration      (numerically obtained attractor)
    ||rho(T) - N(0,u*)||_L1,  |u_inf - u*|            (convergence to the exact stationary
                                                       solution of Proposition 1)

and the refinement study over (N_x, dt) in {(256, 2e-4), (512, 1e-4), (1024, 5e-5)} and
L in {6, 8} with the observed convergence order, the Richardson continuum intercept d0 of
T_x - T_acc, and the number of modes of each initial density on every grid.

Outputs: results/pde_nongaussian/{table2_summary.csv, table2_table.tex, pde_convergence_seed_results.csv,
pde_convergence_orders.csv, pde_richardson_intercepts.csv, pde_initial_modes.csv, pde_convergence_table.tex,
trajectories_reference.csv (u(t), D_x(t) per family for Figure 3), rho_final_*.npy}.

Run:  python experiments/run_pde_nongaussian.py [--quick] [--eps 0.1] [--u0 0.36] [--t-long 20]
      python experiments/run_pde_nongaussian.py --tables-only    # rebuild tables from saved results (seconds)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fisher_surrogate import pde  # noqa: E402
from fisher_surrogate.common import experiment_parser, out_dir, save_results  # noqa: E402
from fisher_surrogate.dynamics import crossing_time_exact  # noqa: E402

MODE_CHECK_NX = (256, 512, 768, 1024)   # the refinement grids and the N_x = 768 grid of the Table 19(A) data


def run_one(name, eps, u0, L, nx, dt, t_max, t_long, n_out, rtol, atol):
    t0 = time.perf_counter()
    res = pde.run_family(name, eps=eps, u0=u0, L=L, nx=nx, dt=dt, t_max=t_max, t_long=t_long, n_out=n_out,
                         rtol=rtol, atol=atol)
    st = pde.stationary_check(res)
    rho0 = pde.INITIAL_DENSITIES[name](pde.Grid(L, nx).x, u0=u0)
    row = {"family": name, "nx": nx, "dt": dt, "L": L, "n_modes_rho0": pde.count_modes(rho0), "T_acc": res.T_acc, "T_x": res.T_x,
           "T_x_minus_T_acc": res.T_x - res.T_acc, "max_excess": res.max_excess, "u0": res.u[0],
           "u_Tmax": res.u[-1], "u_inf": st["u_inf"], "u_star": st["u_star"],
           "final_error_Tmax": abs(res.u[-1] - st["u_inf"]), "L1_to_stationary": st["L1_to_stationary"],
           "abs_u_inf_minus_u_star": st["abs_u_inf_minus_u_star"], "mass_drift": float(np.abs(res.mass - 1).max()),
           "mean_T": res.m[-1], "wall_s": time.perf_counter() - t0, **{k: v for k, v in res.info.items()}}
    return row, res


def main():
    p = experiment_parser(__doc__)
    p.add_argument("--eps", type=float, default=0.1)
    p.add_argument("--u0", type=float, default=0.36)
    p.add_argument("--t-max", type=float, default=5.0)
    p.add_argument("--t-long", type=float, default=20.0)
    p.add_argument("--n-out", type=int, default=501)
    p.add_argument("--rtol", type=float, default=1e-8)
    p.add_argument("--atol", type=float, default=1e-11)
    p.add_argument("--families", nargs="*", default=list(pde.INITIAL_DENSITIES))
    p.add_argument("--no-refinement", action="store_true")
    p.add_argument("--tables-only", action="store_true",
                   help="rebuild the summaries and LaTeX tables from the results saved in --out, without simulating")
    args = p.parse_args()
    out = out_dir(args, "pde_nongaussian")
    if args.tables_only:
        cfg = json.loads((out / "pde_nongaussian_config.json").read_text())
        args.eps, args.u0, args.t_long = cfg["eps"], cfg["u0"], cfg["t_long"]
        conv = out / "pde_convergence_seed_results.csv"
        write_tables(out, pd.read_csv(out / "table2_summary.csv"), pd.read_csv(conv) if conv.exists() else None, args)
        return
    levels = dict(pde.REFINEMENT_LEVELS)
    domains = list(pde.DOMAINS)
    if args.quick:
        levels = {"coarse": (128, 4e-4), "ref.": (192, 2.7e-4), "fine": (256, 2e-4)}
        domains = [6.0]
        args.t_long, args.n_out = 8.0, 101

    # --- reference resolution: Table 2 + Figure 3 data ---
    ref_nx, ref_dt = levels["ref."]
    rows, traj = [], []
    for fam in args.families:
        print(f"[reference] {fam}", flush=True)
        row, res = run_one(fam, args.eps, args.u0, 6.0, ref_nx, ref_dt, args.t_max, args.t_long, args.n_out,
                           args.rtol, args.atol)
        rows.append(row)
        traj.append(pd.DataFrame({"family": fam, "t": res.t, "u": res.u, "Dx": res.Dx, "mass": res.mass, "mean": res.m}))
        np.save(out / f"rho_final_{fam.replace(' ', '_')}.npy", np.column_stack([res.grid.x, res.rho_final]))
    t2 = pd.DataFrame(rows)
    t2["T_acc_gaussian_exact"] = crossing_time_exact(args.u0, args.eps)
    t2.to_csv(out / "table2_summary.csv", index=False)
    pd.concat(traj).to_csv(out / "trajectories_reference.csv", index=False)
    print(t2[["family", "T_acc", "T_x", "max_excess", "u_inf", "u_star", "L1_to_stationary", "abs_u_inf_minus_u_star"]])

    # --- refinement study (Table C.21) ---
    conv = None
    if not args.no_refinement:
        rows = []
        for fam in args.families:
            for L in domains:
                for lvl, (nx, dt) in levels.items():
                    print(f"[refine] {fam} L={L} nx={nx} dt={dt}", flush=True)
                    row, _ = run_one(fam, args.eps, args.u0, L, nx, dt, args.t_max, args.t_long, args.n_out,
                                     args.rtol, args.atol)
                    rows.append({"level": lvl, **row})
        conv = pd.DataFrame(rows)
        conv.to_csv(out / "pde_convergence_seed_results.csv", index=False)
    write_tables(out, t2, conv, args)
    save_results(out, "pde_nongaussian", config={**vars(args), "levels": levels, "domains": domains,
                                                 "density_parameters": density_parameters(args.u0)})


def write_tables(out, t2, conv, args):
    (out / "table2_table.tex").write_text(table2_tex(t2, args), encoding="utf-8")
    if conv is None:
        return
    orders = convergence_orders(conv)
    orders.to_csv(out / "pde_convergence_orders.csv", index=False)
    icpt = richardson_intercepts(conv)
    icpt.to_csv(out / "pde_richardson_intercepts.csv", index=False)
    modes = initial_mode_counts(list(t2.family), args.u0, sorted(conv.L.unique()))
    modes.to_csv(out / "pde_initial_modes.csv", index=False)
    (out / "pde_convergence_table.tex").write_text(convergence_tex(conv, orders, icpt, modes, args), encoding="utf-8")
    print(orders)
    print(icpt[["family", "d0", "u", "abs_d0_over_u"]])
    print(modes.groupby("family", sort=False).n_modes.agg(["min", "max"]))


def convergence_orders(conv):
    orders = []
    for (fam, L), g in conv.groupby(["family", "L"], sort=False):
        g = g.set_index("level")
        if {"coarse", "ref.", "fine"} <= set(g.index):
            orders.append({"family": fam, "L": L, **{
                f"order_{q}": pde.observed_order(g.loc["coarse", q], g.loc["ref.", q], g.loc["fine", q])
                for q in ("T_acc", "T_x", "max_excess", "u_inf")}})
    return pd.DataFrame(orders)


def richardson_intercepts(conv):
    """Continuum intercept d0 of d(h) = T_x - T_acc = d0 + C h^2 (h = 1/N_x), per family.

    Least-squares fit over the three resolutions at each domain size L. The estimate is the mean over L and
    its uncertainty u = max(largest standard error of the fitted intercept, spread of d0 between the L)."""
    rows = []
    for fam, g in conv.groupby("family", sort=False):
        per_L = {}
        for L, s in g.groupby("L"):
            s = s.sort_values("nx")
            h2 = (1.0 / s.nx.to_numpy(float)) ** 2
            d = (s.T_x - s.T_acc).to_numpy(float)
            A = np.vstack([np.ones_like(h2), h2]).T
            coef, *_ = np.linalg.lstsq(A, d, rcond=None)
            dof = len(d) - 2
            se = float(np.sqrt(((d - A @ coef) ** 2).sum() / dof * np.linalg.inv(A.T @ A)[0, 0])) if dof > 0 else np.nan
            per_L[float(L)] = (float(coef[0]), se)
        d0s = [v[0] for v in per_L.values()]
        spread = max(d0s) - min(d0s) if len(d0s) > 1 else np.nan
        u = float(np.nanmax([spread, *[v[1] for v in per_L.values()]]))
        d0 = float(np.mean(d0s))
        rows.append({"family": fam, **{f"d0_L{L:g}": v[0] for L, v in per_L.items()},
                     **{f"se_L{L:g}": v[1] for L, v in per_L.items()},
                     "L_spread": spread, "d0": d0, "u": u, "abs_d0_over_u": abs(d0) / u})
    return pd.DataFrame(rows)


def initial_mode_counts(families, u0, domains, nxs=MODE_CHECK_NX):
    return pd.DataFrame([{"family": fam, "L": float(L), "nx": nx,
                          "n_modes": pde.count_modes(pde.INITIAL_DENSITIES[fam](pde.Grid(L, nx).x, u0=u0))}
                         for fam in families for L in domains for nx in nxs])


def _asym_params(u0, w=0.3):
    s2 = u0 / (1.0 + w * (1.0 - w) * pde.SEPARATION_RATIO**2); a1 = (1.0 - w) * pde.SEPARATION_RATIO * np.sqrt(s2)
    return {"w": w, "a1": float(a1), "sigma1_2": float(s2), "a2": float(w * a1 / (1.0 - w)), "sigma2_2": float(s2),
            "d_over_sigma": float(pde.SEPARATION_RATIO)}


def density_parameters(u0):
    return {"Gaussian": {"u0": u0},
            "Symmetric bimodal": {"a": 0.5, "sigma2": u0 - 0.25},   # a/sigma_c = 1.51 (revision: genuinely bimodal)
            "Asymmetric bimodal": _asym_params(u0),   # same d/sigma as the symmetric family, w = 0.3
            "Laplace": {"b": float(np.sqrt(u0 / 2))}}


def _sf(x, sf):
    """Manuscript number format: ``x`` to ``sf`` significant figures as $m{\\times}10^{e}$."""
    if x == 0:
        return "$0$"
    e = int(np.floor(np.log10(abs(x))))
    m = round(x / 10**e, sf - 1)
    if abs(m) >= 10:
        m /= 10
        e += 1
    return rf"${m:.{sf - 1}f}{{\times}}10^{{{e}}}$"


def _range(v):
    return f"${np.nanmin(v):.2f}$--${np.nanmax(v):.2f}$"


def table2_tex(t2, args):
    nx, L = int(t2.nx.iloc[0]), float(t2.L.iloc[0])
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\caption{Transient behavior for Gaussian and representative non-Gaussian initial densities "
             rf"($u_{{0}}={args.u0:g}$) under Fisher-regularized dynamics with $\varepsilon={args.eps:g}$. "
             r"$T_{\mathrm{acc}}$: first crossing of $u=1$; $T_{\times}$: first zero of the cross-dissipation "
             r"$D_{\times}(t)$ computed from the PDE solution; $u_{\infty}$: variance of the numerically obtained "
             r"stationary density (long-time integration, \ref{app:pde_numerics}); final error: "
             rf"$|u(T_{{\max}})-u_{{\infty}}|$. Reference resolution $N_{{x}}={nx}$, $L={L:g}$.}}",
             r"\label{tab:nongaussian_results}", r"\begin{tabular}{lccccc}", r"\toprule",
             r"Initial density & $T_{\mathrm{acc}}$ & $T_{\times}$ & Max excess & $u_{\infty}$ & Final error \\",
             r"\midrule"]
    for _, r in t2.iterrows():
        lines.append(f"{r['family']} & ${r['T_acc']:.4f}$ & ${r['T_x']:.4f}$ & {_sf(r['max_excess'], 4)} & "
                     f"${r['u_inf']:.4f}$ & {_sf(r['final_error_Tmax'], 3)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def convergence_tex(conv, orders, icpt, modes, args):
    """Appendix table in the manuscript layout: one row per family with coarse / ref. / fine values at L = 6;
    both domain sizes enter the observed-order range and the continuum intercept d0."""
    L0 = min(conv.L.unique())
    lines = [r"\begin{table*}[htbp]", r"\centering", r"\small",
             r"\caption{Grid, time-step, and domain convergence of the non-Gaussian PDE diagnostics "
             rf"($\varepsilon={args.eps:g}$, $u_{{0}}={args.u0:g}$), with the observed convergence order and the "
             r"independent stationary-state check. Modes: number of local maxima of the initial density, the same on "
             r"every grid used ($N_{x}\in\{256,512,768,1024\}$, $L\in\{6,8\}$).}",
             r"\label{tab:pde_convergence}", r"\begin{tabular}{lclcccccc}", r"\toprule",
             r"Initial density & Modes & $(N_x,\Delta t,L)$ & $T_{\mathrm{acc}}$ & $T_{\times}$ & Max excess & $u_{\infty}$ & "
             r"$\|\rho(T)-\rho^{*}_{\varepsilon}\|_{L^{1}}$ & $|u_\infty-u^{*}_{\varepsilon}|$ \\", r"\midrule"]
    for fam, g in conv[conv.L == L0].groupby("family", sort=False):
        g = g.set_index("level").loc[["coarse", "ref.", "fine"]]
        m = modes[modes.family == fam].n_modes.unique()
        tri = lambda col, f: " / ".join(f(v) for v in g[col])   # noqa: E731
        lines.append(" & ".join([fam, "/".join(str(int(k)) for k in m), "coarse / ref. / fine",
                                 tri("T_acc", lambda v: f"${v:.4f}$"), tri("T_x", lambda v: f"${v:.4f}$"),
                                 tri("max_excess", lambda v: _sf(v, 4)), tri("u_inf", lambda v: f"${v:.4f}$"),
                                 tri("L1_to_stationary", lambda v: _sf(v, 3)),
                                 tri("abs_u_inf_minus_u_star", lambda v: _sf(v, 3))]) + r" \\")
    lines.append(r"\multicolumn{3}{l}{Observed order (all families, $L\in\{6,8\}$)} & "
                 f"{_range(orders.order_T_acc)} & {_range(orders.order_T_x)} & & & & \\\\")
    per_fam = []
    for _, r in icpt.iterrows():
        e = int(np.floor(np.log10(r.u)))
        per_fam.append(rf"$({r.d0 / 10**e:+.1f}\pm{r.u / 10**e:.1f})\times10^{{{e}}}$")
    lines.append(r"\multicolumn{3}{l}{Extrapolated $d_{0}=\lim_{h\to0}(T_{\times}-T_{\mathrm{acc}})$ (estimate $\pm$ "
                 r"uncertainty; per family)} & \multicolumn{2}{c}{" + " / ".join(per_fam) + r"} & & & & \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
