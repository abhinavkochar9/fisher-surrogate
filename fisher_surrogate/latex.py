"""LaTeX table rendering for result DataFrames."""
from __future__ import annotations

import numpy as np

from .metrics import METRICS

HEADER = r"$E_{\mathrm{traj}}$ & $E_{\mathrm{cross}}$ & $E_{\mathrm{over}}$ & $E_{\mathrm{eq}}$"


def sci(x, digits=4):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    if x == 0:
        return "$0$"
    e = int(np.floor(np.log10(abs(x))))
    return rf"${x / 10**e:.{digits}f}\times 10^{{{e}}}$"


def sig(x, sf):
    """``x`` to ``sf`` significant figures in the manuscript's number format, e.g. ``1.268{\\times}10^{-7}``."""
    if x == 0:
        return "0"
    e = int(np.floor(np.log10(abs(x))))
    m = round(x / 10**e, sf - 1)
    if abs(m) >= 10:
        m /= 10
        e += 1
    return f"{m:.{sf - 1}f}{{\\times}}10^{{{e}}}"


def cell(mean, ci=None, sf=4, ci_sf=2):
    """Manuscript table cell: mean to 4 and 95% CI to 2 significant figures, e.g.
    ``$1.268{\\times}10^{-7} \\pm 2.8{\\times}10^{-8}$``."""
    if mean is None or np.isnan(mean):
        return "--"
    if ci is None or np.isnan(ci):
        return f"${sig(mean, sf)}$"
    return f"${sig(mean, sf)} \\pm {sig(ci, ci_sf)}$"


def ratio(r):
    """Residual/direct ratio to 3 significant figures (scientific below 0.01, integer from 100)."""
    if r is None or np.isnan(r):
        return "--"
    if r < 0.01:
        return f"${sig(r, 3)}$"
    return f"${r:.0f}$" if r >= 100 else f"${r:#.3g}$"


def sci_ci(mean, ci):
    if np.isnan(mean):
        return "--"
    if np.isnan(ci):
        return sci(mean, 3)
    return rf"${mean:.3e} \pm {ci:.1e}$"


def table_point(df, label_col, caption, label, size=r"\small", resize=True):
    """Single-value table (deterministic or single-seed results)."""
    lines = [r"\begin{table}[ht!]", r"\centering", size, rf"\caption{{{caption}}}", rf"\label{{{label}}}"]
    if resize:
        lines.append(r"\resizebox{\textwidth}{!}{%")
    lines += [r"\begin{tabular}{lcccc}", r"\hline", f"{label_col} & {HEADER} \\\\", r"\hline"]
    for _, r in df.iterrows():
        lines.append(f"{r[label_col]} & " + " & ".join(sci(r[m]) for m in METRICS) + r" \\")
    lines += [r"\hline", r"\end{tabular}" + ("%" if resize else "")]
    if resize:
        lines.append("}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def table_ci(summary, label_cols, caption, label, order=None, size=r"\scriptsize"):
    """Mean +/- CI table from ``metrics.summarize_seeds`` output."""
    label_cols = list(label_cols)
    if order is not None:
        summary = summary.set_index(label_cols).loc[order].reset_index()
    ncol = len(label_cols)
    lines = [r"\begin{table}[t]", r"\centering", size, rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{" + "l" * ncol + "cccc}", r"\hline",
             " & ".join(label_cols) + f" & {HEADER} \\\\", r"\hline"]
    for _, r in summary.iterrows():
        cells = [str(r[c]) for c in label_cols] + [sci_ci(r[f"{m}_mean"], r[f"{m}_ci"]) for m in METRICS]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\hline", r"\end{tabular}%", "}", r"\end{table}"]
    return "\n".join(lines)
