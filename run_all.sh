#!/usr/bin/env bash
# Regenerate every table and figure of the revised manuscript (NEUCOM-D-26-14165, revision 1).
# Table and figure numbers are those of the compiled manuscript.
# Pass --quick to smoke-test the whole pipeline in a few minutes (outputs in results_quick/); the full run
# takes about a day on one GPU workstation (the Neural ODE and PDE experiments dominate).
set -e
Q=${1:-}
cd "$(dirname "$0")"
if [ "$Q" = "--quick" ]; then export FISHER_SURROGATE_RESULTS="$PWD/results_quick"; fi   # never overwrite results/
run() { echo; echo "=================== $*"; python experiments/"$@" $Q; }
run run_sklearn_tables.py           # Tables 4, 5, 6
run run_physics_informed.py         # Tables 7, 9
run run_eps_ablation.py             # Table 8
run run_feature_matched.py          # Table 10
run run_offgrid_time.py             # Table 11
run run_deeponet.py                 # Table 12
run run_neural_ode_gp.py            # Table 13
run run_noise_sensitivity.py        # Table 14, Fig. 5
run run_ood_generalization.py       # Table 16, Fig. 6 (regimes of Table 15)
run run_pi_gradient_analysis.py     # Table 17, Tables D.22 and D.23, Figs. 7 and D.10
run run_mechanism_loss.py           # Table 18
run run_imperfect_baseline.py       # Table 19, Fig. 8
run run_transfer_system.py          # Table 20, Fig. 9
run run_gp_offdistribution.py       # GP rows of Tables 16 and 20, GP error curve of Fig. 9
run run_pde_nongaussian.py          # Table 2, Table C.21, Fig. 3
run run_pilot_sweep.py              # supplementary Table S1
python experiments/collect_model_specs.py   # Table 3
python experiments/collect_configs.py       # results/CONFIG_ALL.json
python experiments/make_figures.py          # Figs. 1-3, 5-9, D.10 -> results/figures/
