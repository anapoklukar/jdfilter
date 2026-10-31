"""U0 checks of the periodic cell-mass filter (NAVODILA_ZA_ANO.md, Section 7)."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.linalg import expm

from jdfilter import periodic as pe


def _small(model=None, R=4.0, J=128):
    model = model or pe.model_C("mixed")
    return pe.Setup.build(pe.PeriodicGrid(R, J), model)


def _paths(K=4, eps=2.0**-8, seed=1):
    return pe.q_increments(range(K), eps, 1.0, seed)


@pytest.mark.parametrize("kind", ["grid_exp", "practical"])
def test_fft_matches_direct_matrix(kind):
    s = _small()
    A, Jm = pe.generator_matrices(s)
    A = A.toarray()
    r = 0.07
    if kind == "grid_exp":
        M = expm(r * (A + Jm))
    else:
        M = expm(r * Jm) @ np.linalg.inv(np.eye(s.grid.J) - r * A)
    w = s.w0
    direct = M @ w
    fft = pe.FFTPredictor(s, kind)(w, r)
    mat = pe.MatrixPredictor(s, kind)(w, r)
    assert np.abs(fft - direct).max() / np.abs(direct).max() < 1e-11
    assert np.abs(mat - direct).max() / np.abs(direct).max() < 1e-11


def test_zero_frequency_is_one():
    s = pe.Setup.build(pe.PeriodicGrid(8.0, 1024), pe.model_C())
    for kind in ("grid_exp", "practical"):
        f = pe.FFTPredictor(s, kind)
        for r in (2.0**-12, 2.0**-3, 1.0):
            assert abs(f.factor(r)[0] - 1) < 1e-14


def test_orientation_of_jumps_and_drift():
    g = pe.PeriodicGrid(8.0, 1024)
    w = np.zeros(g.J)
    w[g.J // 2] = 1.0                                   # point mass near x = 0
    jumps = pe.Setup.build(g, replace(pe.model_C(), b=0.0, sigma=1e-8))
    jumps.w0[:] = w
    drift = pe.Setup.build(g, replace(pe.model_C(), lam=0.0, sigma=1e-8))
    r = 0.5
    shift_jumps = g.x @ pe.FFTPredictor(jumps, "grid_exp")(w, r) - g.x @ w
    shift_drift = g.x @ pe.FFTPredictor(drift, "grid_exp")(w, r) - g.x @ w
    assert shift_jumps == pytest.approx(0.8 * 0.6 * r, rel=1e-3)   # positive jumps move mass right
    assert shift_drift == pytest.approx(0.15 * r, rel=1e-8)        # positive drift moves mass right


@pytest.mark.parametrize("kind", ["grid_exp", "practical"])
def test_prediction_conserves_mass_and_positivity(kind):
    s = pe.Setup.build(pe.PeriodicGrid(8.0, 1024), pe.model_C())
    q = pe.FFTPredictor(s, kind)(s.w0, 0.25)
    assert abs(q.sum() - 1) < 1e-13
    assert q.min() > -1e-15


def test_uninformative_observation_is_identity():
    s = _small(replace(pe.model_C(), h=lambda x: 0 * x, ell=lambda x: 1 + 0 * x))
    obs = pe.Observation(s.grid, s.model)
    G = obs.G(np.array([0.7, -1.3]), np.array([0.0, 3.0]), 0.1)
    assert np.all(G == 0.0)


def test_successive_updates_multiply():
    s = _small()
    obs = pe.Observation(s.grid, s.model)
    dy, dn, dt = np.array([0.3, -0.1, 0.25]), np.array([1.0, 0.0, 2.0]), 0.05
    p, L = s.w0[None], np.zeros(1)
    for k in range(3):
        p, L = pe.observe(p, L, obs.G(dy[k:k + 1], dn[k:k + 1], dt))
    p1, L1 = pe.observe(s.w0[None], np.zeros(1), obs.G(dy.sum()[None], dn.sum()[None], 3 * dt))
    assert np.abs(p - p1).max() < 1e-14 and abs(L - L1)[0] < 1e-13


@pytest.mark.parametrize("dn", [0, 1, 2, 3])
def test_multiple_counts(dn):
    s = _small()
    obs = pe.Observation(s.grid, s.model)
    dy, dt = 0.2, 0.1
    x = s.grid.x
    lik = np.exp(np.tanh(x) * dy - 0.5 * np.tanh(x) ** 2 * dt) * pe.ell0(x) ** dn * np.exp(-(pe.ell0(x) - 1) * dt)
    p, L = pe.observe(s.w0[None], np.zeros(1), obs.G(np.array([dy]), np.array([float(dn)]), dt))
    w = s.w0 * lik
    assert np.abs(np.exp(L[0]) * p[0] - w).max() < 1e-15
    assert L[0] == pytest.approx(np.log(w.sum()), abs=1e-14)


def test_lie_equals_symmetric_for_constant_observation_functions():
    s = _small(replace(pe.model_C(), h=lambda x: 0.4 + 0 * x, ell=lambda x: 1.3 + 0 * x))
    dY, dN = _paths()
    pred = pe.FFTPredictor(s, "grid_exp")
    Pl, Ll, _ = pe.run(s, pred, "lie", pe.coarsen(dY, 8), pe.coarsen(dN, 8), 2.0**-5)
    Ps, Ls, _ = pe.run(s, pred, "sym", pe.coarsen(dY, 8), pe.coarsen(dN, 8), 2.0**-5)
    assert np.abs(Pl - Ps).max() < 1e-12 and np.abs(Ll - Ls).max() < 1e-12


def test_grid_exp_semigroup():
    s = pe.Setup.build(pe.PeriodicGrid(8.0, 1024), pe.model_C())
    f = pe.FFTPredictor(s, "grid_exp")
    assert np.abs(f(f(s.w0, 0.05), 0.05) - f(s.w0, 0.1)).max() < 1e-15


def test_forward_density_control_converges_in_dx():
    """h = 0, ell = 1: the grid solution exp(T(A + J)) w0 approaches the analytic law."""
    model = replace(pe.model_C(), h=None, ell=None)
    errs = []
    for J in (128, 256, 512, 1024):
        s = pe.Setup.build(pe.PeriodicGrid(8.0, J), model)
        w = pe.FFTPredictor(s, "grid_exp")(s.w0, 1.0)
        exact = pe.forward_density_masses(s.grid, 1.0, model)
        errs.append(np.sqrt(s.grid.norm2(w - exact)))
    rates = np.log2(np.array(errs[:-1]) / errs[1:])
    assert np.all(rates > 0.8)
    # moments on a domain where wrapping is negligible: mean 0.63 t, variance 0.25 + 0.72 t
    s = pe.Setup.build(pe.PeriodicGrid(16.0, 4096), model)
    w = pe.FFTPredictor(s, "grid_exp")(s.w0, 1.0)
    mean = s.grid.x @ w
    var = (s.grid.x - mean) ** 2 @ w
    assert mean == pytest.approx(0.63, abs=1e-4)
    assert var == pytest.approx(0.97, abs=5 * s.grid.dx)        # upwind adds b dx t


def test_mean_mass_under_Q():
    """E_Q of the unnormalized masses equals the deterministic prediction R(T) w0."""
    s = pe.Setup.build(pe.PeriodicGrid(8.0, 256), pe.model_C())
    K = 400
    dY, dN = pe.q_increments(range(K), 2.0**-5, 1.0, 7)
    pred = pe.FFTPredictor(s, "grid_exp")
    P, L, _ = pe.run(s, pred, "lie", dY, dN, 2.0**-5, save_every=32)
    Z = np.exp(L[:, -1])
    assert abs(Z.mean() - 1) < 4 * Z.std(ddof=1) / np.sqrt(K)
    w = np.exp(L[:, -1, None]) * P[:, -1]
    mean_w, se = w.mean(0), w.std(0, ddof=1) / np.sqrt(K)
    target = pred(s.w0, 1.0)
    assert np.sqrt(np.mean(((mean_w - target) / np.maximum(se, 1e-300)) ** 2)) < 2.0


def test_matrix_predictor_variable_coefficients():
    s = _small(pe.model_V("mixed"))
    w = s.w0
    for kind in ("grid_exp", "practical"):
        q = pe.MatrixPredictor(s, kind)(np.stack([w, w]), 0.1)
        assert np.abs(q.sum(axis=-1) - 1).max() < 1e-13 and q.min() > -1e-15


def test_path_coupling():
    dY, dN = pe.q_increments(range(3), 2.0**-6, 1.0, 5)
    assert np.allclose(pe.coarsen(dY, 4).sum(1), dY.sum(1))
    assert np.all(pe.coarsen(dN, 4).sum(1) == dN.sum(1))
    fY, fN = pe.refine(dY, dN, 2.0**-6, np.random.default_rng(0))
    assert np.allclose(pe.coarsen(fY, 2), dY) and np.all(pe.coarsen(fN, 2) == dN)
    again = pe.q_increments([2], 2.0**-6, 1.0, 5)
    assert np.array_equal(again[0][0], dY[2])                   # path j is reproducible alone
