"""Direct simulation of the one-dimensional Fisher-regularized Fokker--Planck
equation (paper Eq. 7, Sec. 5.3, Appendix C; revision items R8.2, R5.3, R5.4):

    d_t rho = d_x [ rho d_x ( V + log rho + eps mu_F ) ],
    V = x^2 / 2,   mu_F = - d_xx sqrt(rho) / sqrt(rho),

on [-L, L] with no-flux boundaries.

Discretisation.  Cell-centred finite volumes with N cells (dx = 2L/N).  The
chemical potential mu = V + log rho + eps mu_F is evaluated at cell centres
(mu_F by the centred second-difference of sqrt(rho) with zero-gradient
ghost cells); the flux through the face i+1/2 is

    F_{i+1/2} = rho_{i+1/2} (mu_{i+1} - mu_i) / dx,   rho_{i+1/2} = (rho_i + rho_{i+1}) / 2,

and d_t rho_i = (F_{i+1/2} - F_{i-1/2}) / dx with F = 0 on the two boundary
faces, so mass is conserved to solver tolerance.  The cross-dissipation

    D_x(t) = - int rho d_x mu_0 d_x mu_F dx

is evaluated with exactly the same face stencils (paper Sec. 5.3), which is
what makes T_x (first zero of D_x) comparable to T_acc (first crossing of
u = 1) at the discrete level.

Time integration.  The Fisher term is fourth order, so an explicit scheme
would need dt ~ dx^4 / (4 eps).  The default integrator is therefore the
implicit BDF method of ``scipy.integrate.solve_ivp`` with a banded
finite-difference Jacobian (``jac_sparsity``); the ``dt`` argument is
enforced as the maximum step on the transient window [0, T_max], which is
what the (N_x, dt) refinement study varies.  ``scheme="rk4"`` gives a
fixed-step explicit reference that is only stable for very small dt.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import diags

from .dynamics import u_star
from .metrics import crossing_time

RHO_FLOOR = 1e-10   # floor inside log / division in the diagnostics only
SQRT_SMOOTH = 1e-14  # smooth positive part used for sqrt(rho) so the RHS stays C^1 at rho = 0
BACKGROUND = 1e-8    # uniform background added to every initial density before normalisation.
                     # Keeps the tails away from rho = 0, where the implicit solver otherwise spends
                     # ~10^5 evaluations on narrow initial densities (u0 <~ 0.4); perturbs the initial
                     # variance by ~1e-7 and is reported in the config of every PDE experiment.


# ----------------------------------------------------------------------------
# initial densities (all normalised on the grid)
# ----------------------------------------------------------------------------

def gaussian(x, mean, var):
    return np.exp(-(x - mean) ** 2 / (2.0 * var)) / np.sqrt(2.0 * np.pi * var)


def density_gaussian(x, u0=0.36):
    return gaussian(x, 0.0, u0)


def density_symmetric_bimodal(x, a=0.5, sigma2=None, u0=0.36):
    """(1/2)N(-a, s^2) + (1/2)N(a, s^2); s^2 chosen so the variance equals u0 unless given."""
    if sigma2 is None:
        sigma2 = u0 - a**2
        if sigma2 <= 0:
            raise ValueError("separation too large for the requested variance")
    return 0.5 * gaussian(x, -a, sigma2) + 0.5 * gaussian(x, a, sigma2)


SEPARATION_RATIO = 1.0 / np.sqrt(0.11)   # d/sigma of the symmetric family (a = 0.5, sigma^2 = 0.11 at u0 = 0.36)


def density_asymmetric_bimodal(x, w=0.3, a1=None, sigma1_2=None, u0=0.36):
    """Weight-w component at -a1, weight-(1-w) component at a2 = w*a1/(1-w) (mean 0), total variance u0.

    Default (revision): the same construction as the symmetric family -- equal component variances and the
    same separation-to-width ratio d/sigma = SEPARATION_RATIO (~3.015), with the submitted weight w = 0.3.
    This gives two modes of rho_0 on every grid used in the paper (L = 6, 8; N_x = 256 ... 1024)."""
    if a1 is None or sigma1_2 is None:
        s2 = u0 / (1.0 + w * (1.0 - w) * SEPARATION_RATIO**2)
        a1 = (1.0 - w) * SEPARATION_RATIO * np.sqrt(s2)
        sigma1_2 = s2
    """w N(-a1, s1^2) + (1-w) N(a2, s2^2), with a2 and s2^2 fixed by zero mean and variance u0."""
    a2 = w * a1 / (1.0 - w)
    sigma2_2 = (u0 - w * (a1**2 + sigma1_2) - (1.0 - w) * a2**2) / (1.0 - w)
    if sigma2_2 <= 0:
        raise ValueError("inconsistent asymmetric-bimodal parameters")
    return w * gaussian(x, -a1, sigma1_2) + (1.0 - w) * gaussian(x, a2, sigma2_2)


def density_laplace(x, b=None, u0=0.36):
    """(1/2b) exp(-|x|/b); variance 2 b^2 = u0 unless b is given."""
    if b is None:
        b = np.sqrt(u0 / 2.0)
    return np.exp(-np.abs(x) / b) / (2.0 * b)


INITIAL_DENSITIES = {
    "Gaussian": density_gaussian,
    "Symmetric bimodal": density_symmetric_bimodal,
    "Asymmetric bimodal": density_asymmetric_bimodal,
    "Laplace": density_laplace,
}


# ----------------------------------------------------------------------------
# spatial discretisation
# ----------------------------------------------------------------------------

@dataclass
class Grid:
    L: float = 6.0
    nx: int = 512

    def __post_init__(self):
        self.dx = 2.0 * self.L / self.nx
        self.x = -self.L + (np.arange(self.nx) + 0.5) * self.dx  # cell centres

    def normalise(self, rho):
        rho = np.maximum(rho, 0.0)
        return rho / (rho.sum() * self.dx)


def _sqrt_with_ghosts(rho):
    # smooth positive part: 0.5 (rho + sqrt(rho^2 + delta^2)) -> max(rho, 0) as delta -> 0,
    # keeps the vector field differentiable where the solver drives rho through zero in the tails
    s = np.sqrt(0.5 * (rho + np.sqrt(rho**2 + SQRT_SMOOTH**2)))
    return np.concatenate([s[:1], s, s[-1:]])  # zero-gradient ghost cells


def bohm_potential(rho, dx):
    """mu_F = - d_xx sqrt(rho) / sqrt(rho) at cell centres (diagnostics only; the flux
    never divides by rho)."""
    sp = _sqrt_with_ghosts(rho)
    s = np.maximum(sp[1:-1], np.sqrt(RHO_FLOOR))
    d2 = (sp[2:] - 2.0 * sp[1:-1] + sp[:-2]) / dx**2
    return -d2 / s


def potentials(rho, x, eps, dx):
    """(mu_0, mu_F) at cell centres."""
    mu0 = 0.5 * x**2 + np.log(np.maximum(rho, RHO_FLOOR))
    muF = bohm_potential(rho, dx) if eps > 0 else np.zeros_like(rho)
    return mu0, muF


def face_fluxes(rho, x, eps, dx):
    """Baseline flux j0 = rho d_x mu_0 and Fisher flux jF = rho d_x mu_F at the interior faces.

    The Fisher flux uses the identity rho d_x mu_F = -(s d_xxx s - d_x s d_xx s) with
    s = sqrt(rho), which involves no division by rho and is therefore regular in the
    tails where rho -> 0.
    """
    rho_face = 0.5 * (rho[1:] + rho[:-1])
    x_face = 0.5 * (x[1:] + x[:-1])
    j0 = rho_face * x_face + (rho[1:] - rho[:-1]) / dx
    if eps <= 0:
        return j0, np.zeros_like(j0)
    sp = _sqrt_with_ghosts(rho)
    s = sp[1:-1]
    s2 = (sp[2:] - 2.0 * sp[1:-1] + sp[:-2]) / dx**2   # d_xx s at centres
    s_face = 0.5 * (s[1:] + s[:-1])
    s1_face = (s[1:] - s[:-1]) / dx
    s2_face = 0.5 * (s2[1:] + s2[:-1])
    s3_face = (s2[1:] - s2[:-1]) / dx
    jF = -(s_face * s3_face - s1_face * s2_face)
    return j0, jF


def rhs(rho, x, eps, dx):
    """d_t rho = d_x (j0 + eps jF) with zero flux through the boundary faces."""
    j0, jF = face_fluxes(rho, x, eps, dx)
    flux = j0 + eps * jF
    drho = np.zeros_like(rho)
    drho[:-1] += flux / dx
    drho[1:] -= flux / dx
    return drho


def cross_dissipation(rho, x, eps, dx):
    """D_x = - int (rho d_x mu_0) (d_x mu_F) dx with the same face stencils as the flux."""
    j0, _ = face_fluxes(rho, x, eps, dx)
    _, muF = potentials(rho, x, eps, dx)
    dmuF = (muF[1:] - muF[:-1]) / dx
    return float(-np.sum(j0 * dmuF) * dx)


def variance(rho, x, dx):
    return float(np.sum(x**2 * rho) * dx)


def mean(rho, x, dx):
    return float(np.sum(x * rho) * dx)


def l1_distance(rho, target, dx):
    return float(np.sum(np.abs(rho - target)) * dx)


def _jac_sparsity(nx, bandwidth=4):
    offs = list(range(-bandwidth, bandwidth + 1))
    return diags([np.ones(nx - abs(o)) for o in offs], offs, shape=(nx, nx), format="csr")


# ----------------------------------------------------------------------------
# simulation
# ----------------------------------------------------------------------------

@dataclass
class PDEResult:
    t: np.ndarray
    u: np.ndarray            # variance  int x^2 rho
    m: np.ndarray            # mean      int x rho
    Dx: np.ndarray           # cross-dissipation
    mass: np.ndarray
    rho_final: np.ndarray
    grid: Grid
    eps: float
    T_acc: float = np.nan    # first crossing of u = 1
    T_x: float = np.nan      # first zero of D_x (from negative to positive)
    max_excess: float = np.nan  # max_t u(t) - 1
    info: dict = field(default_factory=dict)


def _first_zero(t, y):
    """First sign change of y from negative to positive, by linear interpolation."""
    y = np.asarray(y, float)
    idx = np.where((y[:-1] < 0.0) & (y[1:] >= 0.0))[0]
    if len(idx) == 0:
        return np.nan
    k = idx[0]
    return float(t[k] + (0.0 - y[k]) * (t[k + 1] - t[k]) / (y[k + 1] - y[k]))


def simulate(rho0, eps, grid: Grid, t_max=5.0, dt=1e-4, n_out=501, scheme="bdf",
             rtol=1e-8, atol=1e-11, t_long=None, background=BACKGROUND):
    """Integrate the Fisher-regularized Fokker--Planck equation.

    Returns a ``PDEResult`` with the diagnostics on the output grid of ``n_out``
    points over [0, t_max].  If ``t_long`` is given the simulation is continued
    (adaptively, no max-step) to ``t_long`` and ``rho_final`` is the density at
    ``t_long``; the long-time variance is stored in ``info["u_long"]``.
    """
    x, dx = grid.x, grid.dx
    rho0 = grid.normalise(np.asarray(rho0, float) + background)
    t_eval = np.linspace(0.0, t_max, n_out)
    f = lambda t, r: rhs(r, x, eps, dx)

    if scheme == "bdf":
        sol = solve_ivp(f, (0.0, t_max), rho0, method="BDF", t_eval=t_eval, max_step=dt,
                        rtol=rtol, atol=atol, jac_sparsity=_jac_sparsity(grid.nx))
        if not sol.success:
            raise RuntimeError(sol.message)
        R = sol.y.T
        n_steps = int(sol.t.size) if sol.t is not None else -1
        info = {"scheme": "bdf", "max_step": dt, "rtol": rtol, "atol": atol, "background": background,
                "nfev": int(sol.nfev), "njev": int(sol.njev), "nlu": int(sol.nlu)}
    elif scheme == "rk4":
        R = _rk4_fixed(f, rho0, t_eval, dt)
        info = {"scheme": "rk4", "dt": dt}
    else:
        raise ValueError(scheme)

    u = np.array([variance(r, x, dx) for r in R])
    mm = np.array([mean(r, x, dx) for r in R])
    Dx = np.array([cross_dissipation(r, x, eps, dx) for r in R])
    mass = np.array([r.sum() * dx for r in R])
    res = PDEResult(t_eval, u, mm, Dx, mass, R[-1], grid, eps, info=info)
    res.T_acc = crossing_time(t_eval, u, 1.0)
    res.T_x = _first_zero(t_eval, Dx)
    res.max_excess = float(np.max(u) - 1.0)

    if t_long is not None and t_long > t_max:
        sol2 = solve_ivp(f, (t_max, t_long), R[-1], method="BDF", rtol=rtol, atol=atol,
                         jac_sparsity=_jac_sparsity(grid.nx))
        if not sol2.success:
            raise RuntimeError(sol2.message)
        rho_T = sol2.y[:, -1]
        res.rho_final = rho_T
        res.info["t_long"] = t_long
        res.info["u_long"] = variance(rho_T, x, dx)
        res.info["mass_long"] = float(rho_T.sum() * dx)
    return res


def _rk4_fixed(f, y0, t_eval, dt):
    out = [y0.copy()]
    y, t = y0.copy(), 0.0
    for t_next in t_eval[1:]:
        while t < t_next - 1e-15:
            h = min(dt, t_next - t)
            k1 = f(t, y)
            k2 = f(t + 0.5 * h, y + 0.5 * h * k1)
            k3 = f(t + 0.5 * h, y + 0.5 * h * k2)
            k4 = f(t + h, y + h * k3)
            y = y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            t += h
        out.append(y.copy())
    return np.array(out)


def stationary_check(res: PDEResult):
    """Compare the long-time density with the exact stationary Gaussian N(0, u*_eps)
    (Proposition 1).  Returns L1 distance, |u_long - u*|, and u_long."""
    ue = float(u_star(res.eps))
    target = gaussian(res.grid.x, 0.0, ue)
    u_long = res.info.get("u_long", res.u[-1])
    return {"u_star": ue, "u_inf": u_long, "L1_to_stationary": l1_distance(res.rho_final, target, res.grid.dx),
            "abs_u_inf_minus_u_star": abs(u_long - ue)}


def run_family(name, eps=0.1, u0=0.36, L=6.0, nx=512, dt=1e-4, t_max=5.0, t_long=20.0, n_out=501,
               density_kw=None, **kw):
    grid = Grid(L, nx)
    rho0 = INITIAL_DENSITIES[name](grid.x, u0=u0, **(density_kw or {}))
    return simulate(rho0, eps, grid, t_max=t_max, dt=dt, n_out=n_out, t_long=t_long, **kw)


REFINEMENT_LEVELS = {"coarse": (256, 2e-4), "ref.": (512, 1e-4), "fine": (1024, 5e-5)}
DOMAINS = (6.0, 8.0)


def observed_order(coarse, ref, fine, ratio=2.0):
    """Richardson-type observed order p = log(|c - r| / |r - f|) / log(ratio)."""
    num, den = abs(coarse - ref), abs(ref - fine)
    if den == 0 or num == 0:
        return np.nan
    return float(np.log(num / den) / np.log(ratio))


def count_modes(rho):
    """Number of local maxima of a sampled density (plateaus counted once)."""
    r = np.asarray(rho, float); inner = r[1:-1]
    cand = np.where((inner >= r[:-2]) & (inner >= r[2:]) & ((inner > r[:-2]) | (inner > r[2:])))[0] + 1
    return int(np.sum(np.diff(np.r_[-10, cand]) > 1))
