"""All figures of the revised manuscript, from the library (variance dynamics, sign transition) and
from the result CSVs of the experiment scripts. Missing result files are skipped with a message.

Output file names are set in FIGURE_FILES below; they carry the final manuscript numbers and are the
file names of the manuscript's Figures/ folder.  Fig. 4 is drawn from the saved test-set predictions of
the published residual MLP (the manuscript's original Fig. 4 came from an archived notebook).

Run:  python experiments/make_figures.py [--out results/figures] [--fmt pdf]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import ticker  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fisher_surrogate.common import RESULTS_DIR  # noqa: E402
from fisher_surrogate.dynamics import cross_dissipation_gaussian, integrate_adaptive, u_star  # noqa: E402

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3})

# Output file stem for each figure (final manuscript numbering).
FIGURE_FILES = {   # the file names used in the manuscript's Figures/ folder
    "variance_dynamics": "fig1-gaussian_manifold_variance_dynamics_final",
    "sign_transition": "fig2-cross_dissipation_sign_transition_final",
    "nongaussian": "fig3-non_gaussian_variance_trajectories",
    "surrogate_trajectories": "fig4-surrogate_trajectories",
    "noise": "fig5-noise_sensitivity_traj_error",
    "ood": "fig6-ood_generalization_traj_error",
    "physics_informed": "fig7-fig_pi_analysis",
    "residual_advantage": "fig8-residual_advantage_vs_baseline_error",
    "transfer": "fig9-transfer_overshoot",
    "data_efficiency": "figD10-data_efficiency",
}


def _save(fig, out, key, fmt):
    # no creation timestamp, so that regenerating a figure from the same results gives the same file
    fig.savefig(out / f"{FIGURE_FILES[key]}.{fmt}", metadata={"CreationDate": None} if fmt == "pdf" else None)
    plt.close(fig)


def _log_ci(mean, half):
    """Asymmetric error bars for a positive quantity on a log axis.

    The 95% interval is formed in log space by the delta method, log(m) +/- (half/m), so it can never
    cross zero; returns (lower, upper) bar lengths for ax.errorbar(yerr=...)."""
    mean, half = np.asarray(mean, float), np.nan_to_num(np.asarray(half, float))
    rel = np.where(mean > 0, half / mean, 0.0)
    return np.vstack([mean - mean * np.exp(-rel), mean * np.exp(rel) - mean])


def _minor_log_labels(ax):
    """Label minor ticks on a log y-axis that spans less than about a decade."""
    lo, hi = ax.get_ylim()
    if hi / lo < 100:
        ax.yaxis.set_minor_formatter(ticker.LogFormatterSciNotation(labelOnlyBase=False, minor_thresholds=(2, 0.5)))
        ax.tick_params(axis="y", which="minor", labelsize=6)


def fig1(out, fmt):
    t = np.linspace(0, 5, 501)
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    for u0, eps in [(0.25, 0.05), (0.60, 0.15), (0.25, 0.30)]:
        ax.plot(t, integrate_adaptive(u0, eps, t), label=rf"$u_0={u0}$, $\varepsilon={eps}$")
        ax.axhline(u_star(eps), ls="--", lw=0.7, color="gray")
    ax.axhline(1.0, ls=":", color="k", lw=0.8, label="critical level $u=1$")
    ax.set(xlabel="$t$", ylabel="$u(t)$", title="Gaussian-manifold variance dynamics")
    ax.legend(fontsize=7); fig.tight_layout(); _save(fig, out, "variance_dynamics", fmt)


def fig2(out, fmt):
    s = np.linspace(0.55, 3.0, 400)
    D = cross_dissipation_gaussian(s)
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    ax.plot(s, D, color="C0"); ax.axhline(0, color="k", lw=0.7); ax.axvline(1, ls=":", color="k")
    ax.fill_between(s, D, 0, where=D < 0, color="C0", alpha=0.15); ax.fill_between(s, D, 0, where=D > 0, color="C3", alpha=0.15)
    ax.set(xlabel=r"$\sigma$", ylabel=r"$D_\times(\sigma)$", title="Cross-dissipation sign transition")
    fig.tight_layout(); _save(fig, out, "sign_transition", fmt)


def fig3(out, fmt):
    """(a) variance trajectories, inset u - u_Gauss; (b) D_x(t) with zero crossings and T_acc ticks;
    (c) T_acc and T_x per family with Richardson error estimates of the reference-grid values."""
    base = RESULTS_DIR / "pde_nongaussian"
    f = base / "trajectories_reference.csv"
    if not f.exists():
        return print("skip fig3 (run run_pde_nongaussian.py)")
    df = pd.read_csv(f); t2 = pd.read_csv(base / "table2_summary.csv").set_index("family")
    fams = list(dict.fromkeys(df.family)); col = {fam: f"C{i}" for i, fam in enumerate(fams)}
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(10.2, 3.0), gridspec_kw={"width_ratios": [1.15, 1.15, 0.8]})
    ug = df[df.family == "Gaussian"].set_index("t").u
    ins = a.inset_axes([0.42, 0.10, 0.54, 0.42])
    for fam in fams:
        g = df[df.family == fam]
        a.plot(g.t, g.u, color=col[fam], label=fam); b.plot(g.t, g.Dx, color=col[fam])
        b.plot(t2.loc[fam, "T_x"], 0, "o", ms=4, color=col[fam])
        b.plot(t2.loc[fam, "T_acc"], -0.6, "|", ms=9, mew=1.5, color=col[fam], clip_on=False)
        if fam != "Gaussian":
            ins.plot(g.t, g.set_index("t").u.values - ug.reindex(g.t).values, color=col[fam], lw=1)
    a.axhline(1, ls=":", color="k"); a.axhline(t2["u_star"].iloc[0], ls="--", color="gray")
    a.set(xlabel="$t$", ylabel="$u(t)$", title="(a) variance trajectories")
    ins.axhline(0, color="k", lw=0.6); ins.set_title(r"$u(t)-u_{\mathrm{Gauss}}(t)$", fontsize=7)
    ins.tick_params(labelsize=6); ins.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2)); ins.yaxis.get_offset_text().set_fontsize(6)
    b.axhline(0, color="k", lw=0.7); b.set(xlabel="$t$", ylabel=r"$D_\times(t)$", title=r"(b) cross-dissipation; $\bullet\,T_\times$, $|\,T_{\mathrm{acc}}$")
    b.set_ylim(-0.6, 0.2)
    b.legend(*a.get_legend_handles_labels(), fontsize=6.5, loc="lower right")   # (b) has free space; (a) does not
    conv = base / "pde_convergence_seed_results.csv"
    x = np.arange(len(fams))
    for q, mk, off, lab in [("T_acc", "o", -0.12, r"$T_{\mathrm{acc}}$"), ("T_x", "s", 0.12, r"$T_\times$")]:
        y = np.array([t2.loc[fam, q] for fam in fams])
        err = None
        if conv.exists():   # Richardson estimate of the reference-grid error: (4/3)|v(ref) - v(fine)|, L = 6
            cv = pd.read_csv(conv); cv = cv[cv.L == 6]
            err = np.array([4.0 / 3.0 * abs(cv[(cv.family == fam) & (cv.level == "ref.")][q].iloc[0]
                                            - cv[(cv.family == fam) & (cv.level == "fine")][q].iloc[0]) for fam in fams])
        c.errorbar(x + off, y, yerr=err, fmt=mk, ms=4, capsize=3, color="k" if q == "T_acc" else "C3", label=lab)
    c.set_xticks(x); c.set_xticklabels([fam.replace(" bimodal", "\nbimodal") for fam in fams], fontsize=6.5)
    c.set(ylabel="time", title=r"(c) $T_{\mathrm{acc}}$ vs $T_\times$ ($N_x=512$)"); c.legend(fontsize=7)
    fig.tight_layout(); _save(fig, out, "nongaussian", fmt)


def _errbar(ax, x, m, ci, log=False, **kw):
    ax.errorbar(x, m, yerr=_log_ci(m, ci) if log else ci, capsize=3, marker="o", **kw)


# Figure 4: the (u0, eps) cases of the original figure; the nearest unseen (test) pairs are drawn
FIG4_CASES = [(0.93, 0.15), (0.60, 0.03), (0.26, 0.05)]


def fig4(out, fmt):
    """True (solid) and residual-MLP predicted (dashed) variance trajectories for three unseen parameter
    pairs: the test pairs nearest to FIG4_CASES, from the saved test-set predictions of the published
    residual MLP (results/feature_matched_ood, seed index 0)."""
    files = sorted((RESULTS_DIR / "feature_matched_ood").glob("predictions_test_mlp_*.csv.gz"))
    if not files:
        return print("skip fig4 (run run_feature_matched_ood.py --family mlp)")
    d = pd.concat(pd.read_csv(f) for f in files)
    d = d[(d.model == "Scaled residual MLP (proposed)") & (d.seed == 0)]
    pairs = d.groupby("pair_id")[["u0", "eps"]].first()
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    used = []
    for k, (u0, eps) in enumerate(FIG4_CASES):
        dist = ((pairs.u0 - u0) / 2.3) ** 2 + ((pairs.eps - eps) / 0.49) ** 2    # ranges of the training box
        pid = dist.drop(index=used).idxmin(); used.append(pid)
        g = d[d.pair_id == pid].sort_values("t")
        ax.plot(g.t, g.u_clean, color=f"C{k}", lw=1.8,
                label=rf"true, $u_0={g.u0.iloc[0]:.2f}$, $\varepsilon={g.eps.iloc[0]:.2f}$")
        ax.plot(g.t, g.u_pred, color="k", lw=0.9, ls="--")
    ax.plot([], [], color="k", lw=0.9, ls="--", label="residual MLP (predicted)")
    ax.axhline(1.0, color="gray", ls=":", lw=1.0, label=r"$u=1$")
    ax.set(xlabel=r"$t$", ylabel=r"$u(t)$")
    ax.legend(fontsize=7, loc="lower right"); fig.tight_layout(); _save(fig, out, "surrogate_trajectories", fmt)


# Figure 5 series (round 2): (label, model, loss, color, line style, marker) in objective_ablation/noise_summary.csv
FIG5_SERIES = [("Direct MLP", "Direct MLP", "unweighted", "C1", "-", "o"),
               ("Residual MLP, $1/\\varepsilon^2$-weighted loss", "Scaled residual MLP", "$1/\\varepsilon^2$-weighted", "C0", "-", "o"),
               ("Residual MLP, unweighted loss", "Scaled residual MLP", "unweighted", "C0", "--", "s"),
               ("Unscaled residual MLP", "Unscaled residual MLP", "unweighted", "C2", "-.", "^")]


def fig5(out, fmt):
    """E_traj vs training-target noise.  With results/objective_ablation (round 2) the published direct and
    (1/eps^2-weighted) residual MLPs plus the residual MLPs trained with the common unweighted loss."""
    fr = RESULTS_DIR / "objective_ablation" / "noise_summary.csv"
    if fr.exists():
        s = pd.read_csv(fr)
        levels = sorted(s.noise.unique(), key=lambda n: float(n.replace("\\%", "")))
        fig, ax = plt.subplots(figsize=(4.6, 3.0))
        x = np.arange(len(levels)); k = len(FIG5_SERIES)
        for j, (lab, model, loss, col, ls, mk) in enumerate(FIG5_SERIES):
            g = s[(s.model == model) & (s.loss == loss)].set_index("noise").loc[levels]
            ax.errorbar(x + 0.2 * (j - (k - 1) / 2) / (k - 1), g.E_traj_mean, yerr=_log_ci(g.E_traj_mean, g.E_traj_ci),
                        capsize=2, marker=mk, ms=4, color=col, ls=ls, lw=1.1, label=lab)
        ax.set_xticks(x); ax.set_xticklabels([n.replace("\\", "") for n in levels])
        ax.set(yscale="log", xlabel="training-target noise level", ylabel=r"$E_{\mathrm{traj}}$", title="Noise sensitivity")
        ax.legend(fontsize=6.5); fig.tight_layout(); _save(fig, out, "noise", fmt)
        return
    f = RESULTS_DIR / "noise_sensitivity" / "noise_sensitivity_summary.csv"
    if not f.exists():
        return print("skip fig5")
    s = pd.read_csv(f); fig, ax = plt.subplots(figsize=(4.2, 2.8))
    for model, g in s.groupby("model"):
        _errbar(ax, np.arange(len(g)), g.E_traj_mean, g.E_traj_ci, label=model)
    ax.set_xticks(np.arange(s.noise.nunique())); ax.set_xticklabels([n.replace("\\", "") for n in s.noise.unique()])
    ax.set(yscale="log", xlabel="training-target noise level", ylabel=r"$E_{\mathrm{traj}}$", title="Noise sensitivity")
    ax.legend(); fig.tight_layout(); _save(fig, out, "noise", fmt)


# Figure 6 series: (label in the figure, model name in the results, color, line style, marker)
FIG6_SERIES = [("Direct MLP", "Direct MLP (no baseline)", "C1", "-", "o"),
               ("Direct MLP + $u_{\\mathrm{OU}}$ input", "Direct MLP + $u_{\\mathrm{OU}}$ input", "C1", "--", "s"),
               ("Residual MLP", "Scaled residual MLP (proposed)", "C0", "-", "o"),
               ("Unscaled residual MLP", "Unscaled residual (difference model)", "C0", "--", "s"),
               ("Gaussian process", "GP, zero mean", "C2", "-", "o"),
               ("GP + $u_{\\mathrm{OU}}$ prior mean", "GP, $u_{\\mathrm{OU}}$ prior mean", "C2", "--", "s"),
               ("Residual Neural ODE", "Residual Neural ODE", "C3", "-.", "^")]


def fig6(out, fmt):
    """E_traj on the four OOD regimes.  With results/feature_matched_ood (round 2) all models of Table 16,
    including the feature-matched baselines; otherwise the direct/residual MLPs and the zero-mean GP."""
    ffm = RESULTS_DIR / "feature_matched_ood" / "feature_matched_ood_summary.csv"
    if ffm.exists():
        s = pd.read_csv(ffm); s = s[s.split == "ood"]
        regs = [r for r in ("Low variance", "High variance", "High Fisher strength", "Longer time") if r in set(s.regime)]
        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        x = np.arange(len(regs)); k = len(FIG6_SERIES)
        for j, (lab, name, col, ls, mk) in enumerate(FIG6_SERIES):
            g = s[s.model == name].set_index("regime")
            if g.empty:
                continue
            g = g.loc[[r for r in regs if r in g.index]]
            xx = np.array([regs.index(r) for r in g.index]) + 0.36 * (j - (k - 1) / 2) / (k - 1)
            ax.errorbar(xx, g.E_traj_mean, yerr=_log_ci(g.E_traj_mean, g.E_traj_ci), capsize=2, marker=mk, ms=4,
                        color=col, ls=ls, lw=1.1, label=lab)
        ax.set_xticks(x); ax.set_xticklabels(regs, rotation=12, fontsize=7)
        ax.set(yscale="log", ylabel=r"$E_{\mathrm{traj}}$", title="Out-of-distribution generalization")
        lo, hi = ax.get_ylim(); ax.set_ylim(lo / 10**3.2, hi)   # room below the data for the legend
        ax.legend(fontsize=6.3, loc="lower left", ncol=2)
        fig.tight_layout(); _save(fig, out, "ood", fmt)
        return
    f = RESULTS_DIR / "ood_generalization" / "ood_generalization_summary.csv"
    if not f.exists():
        return print("skip fig6")
    s = pd.read_csv(f); fig, ax = plt.subplots(figsize=(4.2, 2.8))
    regs = list(dict.fromkeys(s.regime))
    for model, g in s.groupby("model"):
        g = g.set_index("regime").loc[regs]
        _errbar(ax, np.arange(len(regs)), g.E_traj_mean, g.E_traj_ci, log=True, label=model)
    fg = RESULTS_DIR / "gp_offdistribution" / "gp_ood_summary.csv"
    if fg.exists():   # Gaussian process, zero prior mean, from run_gp_offdistribution.py
        g = pd.read_csv(fg); g = g[g.model == "GP, zero mean"].set_index("regime").loc[regs]
        _errbar(ax, np.arange(len(regs)), g.E_traj_mean, g.E_traj_ci, log=True, label="Gaussian process")
    ax.set_xticks(np.arange(len(regs))); ax.set_xticklabels(regs, rotation=15, fontsize=7)
    ax.set(yscale="log", ylabel=r"$E_{\mathrm{traj}}$", title="Out-of-distribution generalization")
    lo, hi = ax.get_ylim(); ax.set_ylim(lo / 10**1.7, hi)   # free space below the data so the legend hides no point
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout(); _save(fig, out, "ood", fmt)


def fig7(out, fmt):
    f = RESULTS_DIR / "pi_gradient_analysis" / "pi_gradient_summary.csv"
    if not f.exists():
        return print("skip fig7")
    s = pd.read_csv(f); fig, axes = plt.subplots(1, 3, figsize=(8.5, 2.6))
    labels = [m.split(",")[-1].strip() for m in s.model]
    for ax, (col, title) in zip(axes, [("L_phys_test", r"(a) $L_{\mathrm{phys}}$ at convergence"),
                                       ("C_ODE_offgrid", r"(b) ODE consistency (off-grid)"), ("E_traj", r"(c) $E_{\mathrm{traj}}$ (test)")]):
        ax.bar(labels, s[f"{col}_mean"], yerr=_log_ci(s[f"{col}_mean"], s[f"{col}_ci"]), capsize=3,
               color=["C0", "C2", "C1", "C3"][:len(s)])
        ax.set(yscale="log", title=title); ax.tick_params(axis="x", labelsize=7)
        _minor_log_labels(ax)
    fig.tight_layout(); _save(fig, out, "physics_informed", fmt)


def fig9(out, fmt):
    """Transfer system: (top) trajectories for the held-out pair with the largest overshoot;
    (bottom) pointwise error |x1_hat - x1| on a log scale, including the Gaussian process."""
    f = RESULTS_DIR / "transfer_system" / "transfer_predictions_seed0.csv"
    if not f.exists():
        return print("skip fig9")
    df = pd.read_csv(f)
    fgp = RESULTS_DIR / "gp_offdistribution" / "gp_transfer_predictions_seed0.csv"
    if fgp.exists():
        gp = pd.read_csv(fgp)[["pair_id", "t", "Gaussian process"]]
        df = df.merge(gp, on=["pair_id", "t"], how="left")
    g = df.groupby("pair_id").first()
    g = g[g.T_cross_true.notna()] if g.T_cross_true.notna().any() else g
    over = df.groupby("pair_id").y_clean.max() - df.groupby("pair_id")["level"].first()
    pid = over.loc[g.index].idxmax()
    d = df[df.pair_id == pid]
    models = [("Residual MLP", "--", "C0"), ("Direct MLP", ":", "C1"), ("Residual Neural ODE", "-.", "C2"),
              ("Gaussian process", (0, (1, 1)), "C3")]
    fig, (a, b) = plt.subplots(2, 1, figsize=(4.4, 4.4), sharex=True, gridspec_kw={"height_ratios": [1.2, 1]})
    a.plot(d.t, d.y_clean, "k", lw=1.6, label="true")
    for name, st, colr in models[:3]:
        if name in d:
            a.plot(d.t, d[name], ls=st, color=colr, label=name)
    a.axhline(d.level.iloc[0], ls=":", color="gray")
    a.set(ylabel="$x_1(t)$", title=rf"Non-normal transfer system ($c={d.c.iloc[0]:.2f}$, $\gamma={d.gamma.iloc[0]:.2f}$)")
    a.legend(fontsize=7)
    for name, st, colr in models:
        if name in d:
            b.plot(d.t, np.abs(d[name] - d.y_clean).clip(lower=1e-16), ls=st, color=colr, label=name)
    b.set(xlabel="$t$", ylabel=r"$|\hat{x}_1-x_1|$", yscale="log")
    lo, hi = b.get_ylim(); b.set_ylim(lo / 10**2.5, hi)   # free space below the curves so the legend hides none
    b.legend(fontsize=6.5, ncol=2, loc="lower right")
    fig.tight_layout(); _save(fig, out, "transfer", fmt)


def fig8(out, fmt):
    base = RESULTS_DIR / "imperfect_baseline"
    files = [("imperfect_A_pde_ratios.csv", "stratum", "(A) PDE target, reduced-ODE baseline"),
             ("imperfect_B_euler_ratios.csv", "h", "(B) transfer, forward-Euler baseline"),
             ("imperfect_C_kappa_ratios.csv", "kappa", r"(C) Fisher, misspecified $\kappa$")]
    have = [(f, c, t) for f, c, t in files if (base / f).exists()]
    if not have:
        return print("skip imperfect-baseline figure")
    fig, axes = plt.subplots(1, len(have), figsize=(2.9 * len(have), 2.7), squeeze=False)
    for ax, (f, col, title) in zip(axes[0], have):
        r = pd.read_csv(base / f)
        summ = pd.read_csv(base / f.replace("_ratios.csv", "_summary.csv"))
        if col == "stratum":   # order by baseline error, not alphabetically
            order = ["exact", "small", "medium", "large"]
            r = r.set_index("stratum").loc[[o for o in order if o in r.stratum.values]].reset_index()
        x = np.arange(len(r))
        for m, lab in [("E_traj", r"$E_{\mathrm{traj}}$"), ("E_cross", r"$E_{\mathrm{cross}}$"), ("E_eq", r"$E_{\mathrm{eq}}$")]:
            # ratio = residual/direct; 95% interval in log space, log r +/- sqrt((ci_q/q)^2 + (ci_d/d)^2)
            rel = []
            for v in r[col]:
                d = summ[(summ[col] == v) & (summ.model == "Direct MLP")].iloc[0]
                q = summ[(summ[col] == v) & (summ.model == "Residual MLP")].iloc[0]
                rel.append(np.sqrt((q[f"{m}_ci"] / q[f"{m}_mean"]) ** 2 + (d[f"{m}_ci"] / d[f"{m}_mean"]) ** 2)
                           if (q[f"{m}_mean"] and d[f"{m}_mean"]) else 0.0)
            ratio = r[f"ratio_{m}"].to_numpy(float); rel = np.nan_to_num(np.asarray(rel))
            ax.errorbar(x, ratio, yerr=np.vstack([ratio - ratio * np.exp(-rel), ratio * np.exp(rel) - ratio]),
                        marker="o", capsize=2, label=lab)
        ax.axhline(1, color="k", lw=0.7); ax.set_xticks(x); ax.set_xticklabels(r[col].astype(str), fontsize=7)
        ax.set(yscale="log", title=title, ylabel="residual / direct" if ax is axes[0][0] else "")
    axes[0][0].legend(fontsize=7); fig.tight_layout(); _save(fig, out, "residual_advantage", fmt)


def figD10(out, fmt):
    f = RESULTS_DIR / "pi_gradient_analysis" / "data_efficiency_summary.csv"
    if not f.exists():
        return print("skip data-efficiency figure")
    s = pd.read_csv(f); fig, ax = plt.subplots(figsize=(4.5, 2.9))
    for lam, g in s.groupby("lambda"):
        n = g.fraction.str.extract(r"\((\d+)\)")[0].astype(int)
        ax.errorbar(n, g.E_traj_mean, yerr=_log_ci(g.E_traj_mean, g.E_traj_ci), marker="o", capsize=3,
                    label=rf"$\lambda={lam}$")
    ax.set(xlabel="number of training trajectories", ylabel=r"$E_{\mathrm{traj}}$ (log scale)", yscale="log",
           title="Data efficiency"); ax.legend(fontsize=7); fig.tight_layout(); _save(fig, out, "data_efficiency", fmt)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=RESULTS_DIR / "figures")
    p.add_argument("--fmt", default="pdf")
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    for fn in (fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, figD10):
        fn(a.out, a.fmt)
    print("figures in", a.out)


if __name__ == "__main__":
    main()
