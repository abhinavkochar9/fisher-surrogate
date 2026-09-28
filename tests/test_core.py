"""Fast correctness checks for dynamics, metrics and the sklearn residual surrogate."""
import numpy as np

from fisher_surrogate.data import FEATURES, make_dataset, residual_to_u, xy_residual
from fisher_surrogate.dynamics import (crossing_time_exact, integrate_adaptive, integrate_rk4, rhs,
                                       u_ou, u_star)
from fisher_surrogate.metrics import compute_metrics, crossing_time, trajectory_metrics
from fisher_surrogate.models.sklearn_models import ResidualOUSurrogate


def test_equilibrium_is_fixed_point():
    for eps in [0.01, 0.1, 0.5, 1.0]:
        assert abs(rhs(u_star(eps), eps)) < 1e-12
        assert u_star(eps) > 1.0


def test_ou_limit_matches_solver():
    t = np.linspace(0, 5, 101)
    assert np.allclose(integrate_adaptive(0.4, 0.0, t), u_ou(t, 0.4), atol=1e-8)
    assert np.allclose(integrate_rk4(0.4, 0.0, t), u_ou(t, 0.4), atol=1e-8)


def test_integrators_agree():
    t = np.linspace(0, 5, 201)
    assert np.allclose(integrate_rk4(0.25, 0.3, t), integrate_adaptive(0.25, 0.3, t), atol=1e-7)


def test_crossing_time_matches_closed_form():
    t = np.linspace(0, 5, 2001)
    u = integrate_adaptive(0.25, 0.1, t)
    assert abs(crossing_time(t, u) - crossing_time_exact(0.25, 0.1)) < 1e-5


def test_metrics_zero_for_perfect_prediction():
    t = np.linspace(0, 5, 81)
    u = integrate_rk4(0.3, 0.2, t)
    m = trajectory_metrics(t, u, u, 0.2)
    # the true crossing time is the closed-form quadrature, so a grid-interpolated crossing of
    # the (perfect) prediction differs by O(dt^2); with a continuous predictor it is refined away
    assert m["E_traj"] == 0 and m["E_over"] == 0 and m["E_cross"] < 1e-3
    fine = lambda tt: integrate_adaptive(0.3, 0.2, np.concatenate([[0.0], np.atleast_1d(tt)]))[1:]
    assert trajectory_metrics(t, u, u, 0.2, pred_fn=fine)["E_cross"] < 1e-6
    assert m["E_eq"] == abs(u[-1] - u_star(0.2))


def test_residual_roundtrip():
    df = make_dataset(5, 20, seed=1)
    x, r = xy_residual(df)
    assert np.allclose(residual_to_u(df, r), df["u_clean"])


def test_residual_surrogate_trains():
    df = make_dataset(40, 41, seed=2, integrator="adaptive")
    model = ResidualOUSurrogate(random_state=0, max_iter=200).fit(df[FEATURES].to_numpy(), df["u_clean"].to_numpy())
    pred = df.copy()
    pred["u_pred"] = model.predict(df[FEATURES].to_numpy())
    m = compute_metrics(pred)
    assert m["E_traj"] < 1e-4


# ---------------------------------------------------------------------------
# revision additions
# ---------------------------------------------------------------------------

def test_missing_crossing_is_penalized_and_counted():
    t = np.linspace(0, 5, 81)
    u = integrate_rk4(0.3, 0.2, t)
    flat = np.full_like(u, 0.9)  # never crosses
    m = trajectory_metrics(t, u, flat, 0.2, cross_missing="penalize")
    assert m["cross_fail"] == 1 and abs(m["E_cross"] - (5.0 - m["T_cross_true"])) < 1e-12


def test_crossing_refinement_recovers_exact_time():
    from fisher_surrogate.metrics import crossing_time
    from fisher_surrogate.dynamics import crossing_time_exact
    t = np.linspace(0, 5, 41)  # coarse grid
    fine = lambda tt: integrate_adaptive(0.25, 0.1, np.concatenate([[0.0], np.atleast_1d(tt)]))[1:]
    u = integrate_adaptive(0.25, 0.1, t)
    coarse = crossing_time(t, u)
    refined = crossing_time(t, u, pred_fn=fine)
    exact = crossing_time_exact(0.25, 0.1)
    assert abs(refined - exact) < 1e-6 < abs(coarse - exact)


def test_eps_zero_has_no_crossing():
    from fisher_surrogate.dynamics import crossing_time_exact
    assert np.isnan(crossing_time_exact(0.3, 0.0))


def test_pde_gaussian_matches_reduced_ode():
    from fisher_surrogate import pde
    res = pde.run_family("Gaussian", eps=0.1, u0=0.36, nx=128, dt=4e-4, t_max=1.5, t_long=None, n_out=31)
    u_ode = integrate_adaptive(res.u[0], 0.1, res.t)
    assert np.abs(res.u - u_ode).max() < 5e-3
    assert np.abs(res.mass - 1).max() < 1e-10
    from fisher_surrogate.dynamics import cross_dissipation_gaussian
    assert np.abs(res.Dx[5:] - cross_dissipation_gaussian(np.sqrt(u_ode[5:]))).max() < 2e-2


def test_stationary_gaussian_is_fixed_point_of_pde():
    from fisher_surrogate import pde
    grid = pde.Grid(6.0, 256)
    eps = 0.3
    rho = pde.gaussian(grid.x, 0.0, float(u_star(eps)))
    drho = pde.rhs(grid.normalise(rho), grid.x, eps, grid.dx)
    assert np.abs(drho).max() < 1e-3   # Proposition 1 at the discrete level (2nd-order truncation)


def test_transfer_equilibrium_and_overshoot():
    from fisher_surrogate import transfer
    df = transfer.make_dataset(6, n_time=41, seed=1)
    st = transfer.overshoot_stats(df)
    g = df.groupby("pair_id").first()
    assert np.allclose(g["y_eq"], 1 + g["gamma"] / 2)
    assert st["n_overshoot"] >= 1
