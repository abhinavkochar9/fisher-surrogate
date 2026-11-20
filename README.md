# fisher-surrogate

Code and results for

> M. Farmer, A. Kochar, Y. Lee. *Beyond Pointwise Error: Mechanism-Level Evaluation of Structure-Aware
> Surrogates for Fisher-Regularized Probability Flows.* Neurocomputing (manuscript NEUCOM-D-26-14165), 2026.

Release `v1.0` is the version cited in the paper. It contains the code and the final results behind
every table and figure; table and figure numbers below are those of the published manuscript.

```
fisher_surrogate/        library: reduced dynamics, datasets, models, unified metrics, PDE solver
experiments/             one script per experiment; each writes results/<experiment>/
results/                 the published results: per-seed CSVs, summaries, LaTeX tables, configs, figures
tables/final/            Tables 3 (time column), 11 and 19 in the final layout; revision-2 rows (round2_*.tex)
tests/                   unit tests (a few seconds)
archive/original_colab_notebooks/   the six notebooks behind the submitted manuscript (provenance only)
run_all.sh               regenerates every table and figure
run_legacy_audit.sh      re-runs the model comparisons under the pre-revision metric definitions
```

## Install

```bash
git clone https://github.com/abhinavkochar9/fisher-surrogate.git
cd fisher-surrogate
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt      # the exact versions used for the paper (Python 3.12)
python -m pytest -q tests            # 13 tests
```

Scripts are run from the repository root; installing the package is not required
(`pip install -e .` also works).

## Reproducing a table or figure

Each command regenerates one manuscript item and overwrites its files in `results/`, so `git status`
and `git diff` show any difference from the published numbers. Times are wall-clock on the machine
used for the paper (Intel Core i7-11800H, NVIDIA GeForce RTX 3070 Laptop GPU, Ubuntu 24.04). Every
experiment script also accepts `--quick`, a smoke test of a few minutes that writes to
`results_quick/` and leaves `results/` untouched.

| Manuscript item | Command | Output in `results/` | Time |
|---|---|---|---|
| Table 1 | none (definitions only) | | |
| Table 2 | `python experiments/run_pde_nongaussian.py` | `pde_nongaussian/table2_*` | 15 min (with Table C.21) |
| Table 3 | `python experiments/collect_model_specs.py` | `model_specs/` (training times read from the per-seed CSVs); Time/seed column in `tables/final/table3_timing.tex` | seconds |
| Tables 4, 5, 6 | `python experiments/run_sklearn_tables.py` | `sklearn_tables/table4_*`, `table5_*`, `table6_*` | 1.5 h |
| Tables 7, 9 | `python experiments/run_physics_informed.py` | `physics_informed/` | 1.5 h |
| Table 8 | `python experiments/run_eps_ablation.py` | `eps_ablation/` | 30 min |
| Table 10 | `python experiments/run_feature_matched.py` | `feature_matched/` | 2.5 h |
| Table 11 | `python experiments/run_offgrid_time.py` | `offgrid_time/`, `tables/final/table11_offgrid.tex` | 2 h |
| Table 12 | `python experiments/run_deeponet.py` | `deeponet/` | 1 h |
| Table 13 | `python experiments/run_neural_ode_gp.py` | `neural_ode_gp/` | 3.5 h |
| Table 14, Fig. 5 | `python experiments/run_noise_sensitivity.py` | `noise_sensitivity/` | 1.5 h |
| Table 15 | none (regime definitions, `OOD_REGIMES` in `fisher_surrogate/data.py`) | | |
| Table 16, Fig. 6 | `python experiments/run_ood_generalization.py` and, for the GP rows, `python experiments/run_gp_offdistribution.py --axes ood` | `ood_generalization/`, `gp_offdistribution/gp_ood_*` | 30 min + 1 h |
| Table 17, Fig. 7 | `python experiments/run_pi_gradient_analysis.py` | `pi_gradient_analysis/pi_gradient_*` | 25 min (with Tables D.22, D.23) |
| Table 18 | `python experiments/run_mechanism_loss.py` | `mechanism_loss/` | 5 min |
| Table 19, Fig. 8 | `python experiments/run_imperfect_baseline.py` (axes A, B, C; `--axes A` etc. for one) | `imperfect_baseline/`, `pde_bimodal/`, `tables/final/table19_baseline_error.tex` | 8 h (A alone: 1.2 h) |
| Table 20, Fig. 9 | `python experiments/run_transfer_system.py` and, for the GP rows, `python experiments/run_gp_offdistribution.py --axes transfer` | `transfer_system/`, `gp_offdistribution/gp_transfer_*` | 1.5 h + 1 h |
| Table C.21 | `python experiments/run_pde_nongaussian.py` | `pde_nongaussian/pde_convergence_*`, `pde_richardson_intercepts.csv`, `pde_initial_modes.csv` | same run as Table 2 |
| Table D.22 | `python experiments/run_pi_gradient_analysis.py` | `pi_gradient_analysis/pi_gradient_summary.csv` (`C_ODE_offgrid`) | same run as Table 17 |
| Table D.23, Fig. D.10 | `python experiments/run_pi_gradient_analysis.py` | `pi_gradient_analysis/data_efficiency_*` | same run as Table 17 |
| Table S1 (supplementary) | `python experiments/run_pilot_sweep.py` | `pilot_sweep/` | 35 min |
| Figs. 1, 2 | `python experiments/make_figures.py` (computed from the reduced dynamics) | `figures/fig1-*`, `figures/fig2-*` | seconds |
| Figs. 3, 5–9, D.10 | `python experiments/make_figures.py` (drawn from the result files above) | `figures/` | seconds |
| Fig. 4 | `python experiments/make_figures.py` (drawn from the saved test-set predictions of the published residual MLP in `feature_matched_ood/`; the figure of the original submission came from the archived notebook `fisher_residual_mlp_surrogate_generalization.ipynb`) | `figures/fig4-*` | seconds |

`python experiments/make_figures.py` redraws every figure from `results/`; the file names are those
of the manuscript's figure files. `bash run_all.sh` runs everything above in sequence (about a day);
`bash run_all.sh --quick` smoke-tests the whole pipeline in `results_quick/`.

The PDE dataset of Table 19 (A) is cached in `results/pde_bimodal/`; delete the file to regenerate it
from the PDE solver (about 1 h). `python experiments/run_pde_nongaussian.py --tables-only` rebuilds
the Table 2 and Table C.21 files from the saved simulation results in a few seconds.

## Revision 2 additions (NEUCOM-D-26-14165R1)

Added on `main` after `v1.0` for the second review round. Nothing published in `v1.0` changes: every
model below that also appears in a `v1.0` table is retrained with the same code path and seeds, and
its per-seed metrics reproduce the published files (bit-identical for the neural models, within
1e-5 relative for the GP; the check is written next to the results).

| Question | Command | Output in `results/` |
|---|---|---|
| Output parameterization vs. loss weighting: {direct, unscaled residual, scaled residual} x {unweighted, 1/eps^2-weighted trajectory MSE}, Table 7 setup | `python experiments/run_objective_ablation.py --part factorial` | `objective_ablation/factorial_*` |
| Physics-informed residual, lambda in {0.01, 0.1, 1}, under each data objective (lambda = 0 is the matching factorial cell) | `python experiments/run_objective_ablation.py --part pi` | `objective_ablation/pi_*` |
| Table 14 under the common unweighted loss (same noise draws) | `python experiments/run_objective_ablation.py --part noise` | `objective_ablation/noise_*` |
| The six factorial models on the OOD regimes of Table 16 and at T = 5, 10, 20 | `python experiments/run_objective_ablation.py --part ood` | `objective_ablation/ood_*`, `ood_horizon_*` |
| Table 8 (error by eps range) with the scaled residual also trained with the unweighted loss | `python experiments/run_objective_ablation.py --part eps` | `objective_ablation/eps_*` |
| Feature-matched baselines of Table 10 (and the Neural ODEs of Table 13) on the OOD regimes of Table 16; terminal error and horizon study | `python experiments/run_feature_matched_ood.py --family mlp` (then `gp`, `node`, `node_tight`) | `feature_matched_ood/` |
| Terminal error for the models of Table 20 | `python experiments/run_transfer_system.py --out results/transfer_terminal` and `python experiments/run_gp_offdistribution.py --axes transfer --out results/transfer_terminal` | `transfer_terminal/` |
| Sensitivity of T_acc and T_x to the output-time sampling and the initial-density background | `python experiments/run_pde_event_sensitivity.py` | `pde_event_sensitivity/` |

The OOD rows use the same per-seed regime datasets as Table 16. `--seeds` and `--no-summary` split a
part into parallel shards (`shard_*_seed_results.csv`); `--tables-only` merges the shards and writes
the summaries and the LaTeX rows. The runs shared the machine with each other, so their `train_time_s`
columns are not comparable with Table 3. Figures 5 and 6 are drawn from `objective_ablation/` and
`feature_matched_ood/` when those directories exist.

The metrics module now also records, for every model, the terminal error
`E_term = |u_hat(T_max) - u(T_max)|`, the true trajectory's distance to equilibrium
`D_term = |u(T_max) - u*|`, and `E_over` split into rising (`u0 < u*`) and falling trajectories
(`E_over_rise`, `E_over_fall`, `n_fall`). The four published metrics are unchanged.

## Manuscript table files

`tables/final/` holds three tables in the manuscript's final layout, and the revision-2 rows. Each is written by the script
that produces its data, and can be rebuilt from the saved results in seconds:

| file | manuscript item | rebuild from `results/` |
|---|---|---|
| `table3_timing.tex` | Table 3, Time/seed column | `python experiments/collect_model_specs.py` |
| `table11_offgrid.tex` | Table 11 | `python experiments/run_offgrid_time.py --tables-only` |
| `table19_baseline_error.tex` | Table 19 | `python experiments/run_imperfect_baseline.py --tables-only` |
| `round2_objective_{factorial,pi,noise,ood,eps}.tex` | revision 2: objective ablation | `python experiments/run_objective_ablation.py --tables-only` |
| `round2_table16_feature_matched.tex`, `round2_terminal_error.tex` | revision 2: Table 16 rows, terminal error | `python experiments/run_feature_matched_ood.py --tables-only` |
| `round2_pde_event_sensitivity.tex` | revision 2: event-time sensitivity | `python experiments/run_pde_event_sensitivity.py --tables-only` |

Running the same script without `--tables-only` regenerates the results first, then the table. The
LaTeX of Tables 2 and C.21 is in `results/pde_nongaussian/` (`table2_table.tex`,
`pde_convergence_table.tex`).

## Result files

Every `results/<experiment>/` directory holds

* `*_seed_results.csv`: one row per model and seed, with all four metrics, crossing counts, and
  training time;
* `*_summary.csv`: mean and 95 % t-interval over seeds, and crossing-failure counts;
* `*_table.tex`: the LaTeX table;
* `*_config.json`: every hyper-parameter, seed, solver tolerance, and data setting of the run.

`results/CONFIG_ALL.json` collects all configuration files in one place
(`python experiments/collect_configs.py`).

## Unified metric definitions

All tables use `fisher_surrogate.metrics.UNIFIED_METRIC_KW`:

| metric | definition |
|---|---|
| `E_traj` | mean squared error on the evaluation grid |
| `E_cross` | first upward crossing of the level (u = 1 on the Fisher benchmark, x1* on the transfer system): grid bracketing and linear interpolation, then Brent root-finding on the continuous surrogate inside the bracketing interval; the true crossing from the Eq. (13) quadrature or a fine reference trajectory. A surrogate with no crossing gets `T_hat = T_max` and is counted in `n_cross_fail`; spurious crossings are counted in `n_cross_spurious`. Applied identically to every model, including the GP. |
| `E_over` | `|max_t u_hat − max_t u|` (finite-horizon excursion) |
| `E_eq` | `|u_hat(T_max) − u_eq|`, with `u_eq = u*_eps` or the numerically obtained attractor |

Grid time resolution: 5/79 = 0.0633 for the PyTorch experiments (80 points) and 5/200 = 0.025 for
the scikit-learn experiments (201 points); crossing times of continuous surrogates are not limited
by it.

## Conventions

| setting | scikit-learn experiments (Tables 4–6) | PyTorch experiments |
|---|---|---|
| parameter pairs | 600 (70/15/15 split by pair) | 300 (70/15/15 by pair, seed 999); 400 for the transfer system; 80 for the physics-informed gradient analysis |
| time grid | 201 points on [0, 5] | 80 points on [0, 5] |
| integrator (data) | `solve_ivp` RK45, rtol 1e-9, atol 1e-11 | fixed-step RK4 on the grid |
| sampling | u0 ~ U[0.2, 2.5], eps ~ U[0.01, 0.5], i.i.d. | same |
| MLP | (64, 64) tanh, Adam, batch 512, early stopping | (64, 64) tanh, Adam lr 2e-3, batch 512, at most 1500 epochs, patience 150 |
| Neural ODE | none | same network as vector field, dopri5 (torchdiffeq), full batch over pairs, backpropagation through the solver |
| GP | none | RBF-ARD + white noise, L-BFGS marginal likelihood, 2 restarts, n_sub = 2000 |
| seeds | 11+i (ablation), 12+i (direct), 16+i (residual) | base 2026+i; offsets: residual +5000, PI +9000+10^4·lambda, DeepONet +12000, Neural ODE +15000, GP +17000 |

The GP is deterministic given its subsample seed (`random_state = seed`). PyTorch training is not
bit-for-bit reproducible across hardware and library builds, so a re-run on a different machine can
differ in the trailing digits; compare such re-runs against the reported 95 % intervals.

## Pre-revision metric definitions (audit)

`results/legacy_metrics_audit/` holds every model-comparison table re-run under the metric
definitions of the submitted manuscript, with the same code, seeds, and data.
`python experiments/audit_legacy_ordering.py` compares the within-table model orderings under the
two sets of definitions (`ordering_cells.csv`, `ordering_by_table.csv`); `bash run_legacy_audit.sh`
regenerates the audit tables themselves (about a day).

## Provenance

The six Colab notebooks behind the submitted manuscript are kept unchanged in
`archive/original_colab_notebooks/`; no published result is produced from them. They differed from
one another in (1) the handling of missing crossings (excluded vs. `T_max`), (2) the definition of
`E_over` (|max − max| vs. excess above u*), (3) the scikit-learn presets, and (4) the high-eps
extrapolation range ([0.5, 0.8] in Table 5, [0.5, 1.0] in Table 16). The revision keeps (3) and (4)
and removes (1) and (2) by regenerating every table with the unified definitions above. Tables 2,
13, 17, 20, D.22, and D.23 had no notebook source; they are implemented here from the paper's
description (`pde.py`, `models/gp_models.py`, `models/neural_ode.py`, `run_pi_gradient_analysis.py`,
`transfer.py`).

## Citation and license

Please cite the paper; `CITATION.cff` gives the reference for this repository. MIT license
(`LICENSE`).
