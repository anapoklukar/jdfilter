"""Unit tests of the individual sub-steps and the simulator."""
import numpy as np
from scipy import stats

from jdfilter import Grid, ou_jump_model, simulate
from jdfilter.diffusion import DiffusionStep
from jdfilter.jumps import JumpStepFFT, JumpStepMatrix


def _setup():
    model = ou_jump_model(jump_rate=2.0, jump_std=0.8)
    grid = Grid(-8, 8, 321)
    p = grid.discretize(stats.norm(0.5, 0.7))
    return model, grid, p


def test_diffusion_step_preserves_mass_and_positivity():
    model, grid, p = _setup()
    for method in ("implicit", "exact"):
        q = DiffusionStep(grid, model, method)(p, 0.05)
        assert abs(q.sum() - 1.0) < 1e-12
        assert q.min() >= 0.0


def test_jump_step_fft_matches_matrix_and_keeps_mass():
    model, grid, p = _setup()
    q_fft = JumpStepFFT(grid, model)(p, 0.1)
    q_mat = JumpStepMatrix(grid, model)(p, 0.1)
    assert np.abs(q_fft - q_mat).sum() < 1e-9          # differ only by mass near the boundary
    # a few consecutive jumps carry ~1e-9 of the mass past the domain edge: genuine truncation loss
    assert 1.0 - 1e-8 < q_fft.sum() <= 1.0 + 1e-12
    assert q_fft.min() >= 0.0


def test_jump_step_poisson_signal_is_exact():
    """Unit jumps on the integer grid: after time t the law of x_0 + N_t is Poisson(lam t)."""
    from jdfilter.models import JumpDiffusionModel
    lam, t = 3.0, 2.0
    model = JumpDiffusionModel(drift=lambda x: 0 * x, diffusion=lambda x: 0 * x,
                               initial_dist=stats.randint(0, 1), jump_rate=lam,
                               jump_dist=stats.randint(1, 2))
    grid = Grid(0, 40, 41)
    p = grid.discretize(model.initial_dist)
    q = JumpStepFFT(grid, model)(p, t)
    assert np.abs(q - stats.poisson(lam * t).pmf(grid.x)).max() < 1e-12


def test_coarsen_sums_increments():
    model = ou_jump_model(cox_intensity=lambda x: 1.0 + 0 * x)
    path = simulate(model, T=1.0, n_steps=64, rng=np.random.default_rng(0))
    coarse = path.coarsen(8)
    assert coarse.n == 8 and np.isclose(coarse.dt, 1 / 8)
    assert np.isclose(coarse.dy.sum(), path.dy.sum())
    assert coarse.dn.sum() == path.dn.sum()
    assert np.allclose(coarse.x, path.x[::8])
