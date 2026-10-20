"""Tests of the 2D splitting filter."""
import numpy as np

from jdfilter.exact import ou_jump_moments
from jdfilter.models import sigmoid_intensity
from jdfilter.twod import Grid2D, SplittingFilter2D, coupled_ou_2d, particle_filter_2d, simulate_2d


def test_2d_no_observations_reproduces_signal_moments():
    """For b = -(theta I + c J) x with J skew, the covariance stays v(t) I with the 1D v(t)."""
    theta, sigma, lam, sJ, x0s = 1.0, 0.3, 1.0, 1.0, 0.3
    model = coupled_ou_2d(theta, 0.8, sigma, lam, sJ, h_gain=None, x0_std=x0s)
    path = simulate_2d(model, T=1.0, n_steps=100, rng=np.random.default_rng(0))
    errs = []
    for n in (49, 97):
        res = SplittingFilter2D(model, Grid2D(-6, 6, n)).run(path)
        _, v = ou_jump_moments(theta, sigma, lam, 0.0, sJ**2, 0.0, x0s**2, res["t"])
        assert np.abs(res["mean"]).max() < 1e-3
        errs.append(np.abs(res["var"] / v[:, None] - 1).max())
    # upwinding adds numerical diffusion ~ |b| dx / 2, so the variance error is O(dx):
    # it must roughly halve when dx halves
    assert errs[1] < 0.1 and errs[1] < 0.6 * errs[0]


def test_2d_agrees_with_particle_filter():
    model = coupled_ou_2d(cox_intensity=sigmoid_intensity())
    path = simulate_2d(model, T=2.0, n_steps=200, rng=np.random.default_rng(1))
    g = SplittingFilter2D(model, Grid2D(-6, 6, 81)).run(path)
    pf = particle_filter_2d(model, path, 20000, np.random.default_rng(2))
    assert np.mean(np.abs(g["mean"] - pf["mean"])) < 0.05
