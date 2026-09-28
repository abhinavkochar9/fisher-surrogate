#!/usr/bin/env bash
# Metric-definition audit (unified-definitions paragraph): regenerate every model-comparison table under
# the PRE-REVISION metric definitions with the same code, seeds and data, then compare model orderings.
# Outputs go to results/legacy_metrics_audit/ (results_quick/ with --quick); the published tables are not touched.
set -e
Q=${1:-}
cd "$(dirname "$0")"
if [ "$Q" = "--quick" ]; then export FISHER_SURROGATE_RESULTS="$PWD/results_quick"; fi   # never overwrite results/
A=${FISHER_SURROGATE_RESULTS:-results}/legacy_metrics_audit
run() { name=$1; shift; echo; echo "=================== legacy: $name"; python experiments/"$@" --legacy-metrics --out $A/$name $Q; }
run sklearn_tables        run_sklearn_tables.py
run physics_informed      run_physics_informed.py
run deeponet              run_deeponet.py
run noise_sensitivity     run_noise_sensitivity.py
run ood_generalization    run_ood_generalization.py
run feature_matched       run_feature_matched.py
run neural_ode_gp         run_neural_ode_gp.py
run transfer_system       run_transfer_system.py
run gp_offdistribution    run_gp_offdistribution.py
run eps_ablation          run_eps_ablation.py
run pi_gradient_analysis  run_pi_gradient_analysis.py
python experiments/audit_legacy_ordering.py
