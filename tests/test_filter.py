"""Tests of the full splitting filter against exact or independent solutions."""
import numpy as np
from scipy import stats

from jdfilter import Grid, SplittingFilter, bootstrap_particle_filter, ou_jump_model, simulate
from jdfilter.exact import kalman_bucy, ou_jump_moments
from jdfilter.models import sigmoid_intensity


def test_no_observations_reproduces_signal_moments():
    """Without observations the filter is the law of x_t; compare with the exact moments."""
    theta, sigma, lam, sJ, x0s = 1.0, 0.3, 1.0, 1.5, 0.1
    model = ou_jump_model(theta, sigma, lam, sJ, h_gain=None, x0_std=x0s)
    path = simulate(model, T=1.0, n_steps=200, rng=np.random.default_rng(1))
    grid = Grid(-8, 8, 401)
    for scheme in ("lie", "strang"):
        res = SplittingFilter(model, grid, scheme=scheme).run(path)
        m, v = ou_jump_moments(theta, sigma, lam, 0.0, sJ**2, 0.0, x0s**2, res.t)
        assert np.abs(res.mean - m).max() < 1e-3
        # implicit Euler is first order in dt; dt = 0.005 here
        assert np.abs(res.var / v - 1).max() < 0.02


def test_matches_kalman_bucy_without_jumps():
    """Linear-Gaussian case without jumps: the exact filter is Kalman-Bucy."""
    theta, sigma, c = 1.0, 0.5, 2.0
    model = ou_jump_model(theta, sigma, jump_rate=0.0, h_gain=c, x0_std=0.3)
    path = simulate(model, T=3.0, n_steps=3000, rng=np.random.default_rng(2))
    res = SplittingFilter(model, Grid(-4, 4, 401), scheme="strang").run(path)
    m, v = kalman_bucy(path, -theta, sigma, c, 0.0, 0.3**2)
    sd = np.sqrt(v)
    assert np.abs(res.mean - m).max() < 0.05 * sd.min()
    assert np.abs(res.var / v - 1).max() < 0.05


def test_agrees_with_particle_filter_jumps_gauss_and_cox():
    """Jump-diffusion with both observation types: compare with a large particle filter."""
    model = ou_jump_model(cox_intensity=sigmoid_intensity())
    path = simulate(model, T=4.0, n_steps=800, rng=np.random.default_rng(3))
    grid_res = SplittingFilter(model, Grid(-7, 7, 281)).run(path)
    pf_res = bootstrap_particle_filter(model, path, 20000, np.random.default_rng(4))
    assert np.mean(np.abs(grid_res.mean - pf_res.mean)) < 0.05
    assert abs(grid_res.log_norm[-1] - pf_res.log_norm[-1]) < 0.5


def test_split_and_exact_prediction_agree():
    model = ou_jump_model(cox_intensity=sigmoid_intensity())
    path = simulate(model, T=1.0, n_steps=1024, rng=np.random.default_rng(5))
    grid = Grid(-6, 6, 121)
    a = SplittingFilter(model, grid, prediction="exact").run(path)
    b = SplittingFilter(model, grid, prediction="split").run(path)
    assert np.abs(a.final - b.final).sum() < 0.02


def test_cox_count_moves_mass_toward_high_intensity():
    model = ou_jump_model(h_gain=None, cox_intensity=sigmoid_intensity())
    grid = Grid(-6, 6, 241)
    f = SplittingFilter(model, grid)
    p0 = grid.discretize(stats.norm(0, 1))
    p1 = p0 * np.exp(f.loglik(None, 1, 0.01))
    assert grid.mean(p1) > grid.mean(p0)
