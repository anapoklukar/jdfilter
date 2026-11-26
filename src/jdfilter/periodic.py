"""Periodic cell-mass splitting filter used for the numerical section of the paper.

Conventions (as in the numerical section of the manuscript):
  * grid: [-R, R) with J cells of width dx = 2R/J, centres x_i = -R + (i + 1/2) dx;
  * the state is a vector of cell masses w (density w_i/dx on cell i);
    the L2 norm of the reconstructed density is sqrt(sum w_i^2 / dx);
  * forward upwind generator A (mass moves right at rate r+, left at rate r-) and jump
    generator J = lam (P - I), where P convolves with the wrapped rounded jump law;
  * predictors: "grid_exp" = exp(r (A + J)), "practical" = exp(r J) (I - r A)^{-1};
  * the state is stored as normalized masses p and the accumulated log mass L (w = e^L p).

For constant coefficients A and J are circulant and the predictors are applied by FFT,
with DFT convention w_hat_k = sum_j w_j exp(-i theta_k j), theta_k = 2 pi k / J.
Variable coefficients use MatrixPredictor (sparse cyclic solve, or a dense exponential).
"""
from dataclasses import dataclass
from typing import Callable, Optional, Union

import numpy as np
import scipy.fft as sfft
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.linalg import expm
from scipy.stats import norm, poisson

Coef = Union[float, Callable[[np.ndarray], np.ndarray]]
WORKERS = -1                                     # threads used by scipy.fft


# -- model ------------------------------------------------------------------------
@dataclass(frozen=True)
class Model:
    """dX = b dt + sigma dW + compound Poisson(lam, N(jump_mean, jump_sd^2)) jumps,
    X_0 ~ N(u0_mean, u0_sd^2); observations dY = h(X) dt + dB and a Cox process with
    intensity ell(X). h = None means no continuous channel, ell = None means no counts."""
    b: Coef = 0.15
    sigma: Coef = 0.6
    lam: float = 0.8
    jump_mean: float = 0.6
    jump_sd: float = 0.3
    u0_mean: float = 0.0
    u0_sd: float = 0.5
    h: Optional[Callable] = None
    ell: Optional[Callable] = None
    T: float = 1.0
    name: str = "C"

    @property
    def constant(self) -> bool:
        return not (callable(self.b) or callable(self.sigma))


def h0(x):
    return np.tanh(x)


def ell0(x):
    return 1.0 + 0.5 * np.tanh(x)


OBS_CASES = {"gaussian": (h0, None), "cox": (None, ell0), "mixed": (h0, ell0)}


def model_C(obs: str = "mixed") -> Model:
    h, ell = OBS_CASES[obs]
    return Model(h=h, ell=ell, name=f"C-{obs}")


def model_V(obs: str = "mixed") -> Model:
    h, ell = OBS_CASES[obs]
    return Model(b=lambda x: -np.tanh(x), sigma=lambda x: 0.6 + 0.1 * np.tanh(x),
                 h=h, ell=ell, name=f"V-{obs}")


def _coef(c: Coef, x: np.ndarray) -> np.ndarray:
    return c(x) if callable(c) else np.full_like(x, float(c))


# -- grid and wrapped laws -------------------------------------------------------
@dataclass(frozen=True)
class PeriodicGrid:
    R: float
    J: int

    @property
    def dx(self) -> float:
        return 2 * self.R / self.J

    @property
    def edges(self) -> np.ndarray:
        return -self.R + self.dx * np.arange(self.J + 1)

    @property
    def x(self) -> np.ndarray:
        return -self.R + self.dx * (np.arange(self.J) + 0.5)

    def norm2(self, w: np.ndarray) -> np.ndarray:
        """Squared L2 norm of the piecewise constant density with cell masses w."""
        return np.sum(w * w, axis=-1) / self.dx

    def lift(self, w: np.ndarray, fine: "PeriodicGrid") -> np.ndarray:
        """Cell masses on a nested finer grid (same R) of the same piecewise constant density."""
        m = fine.J // self.J
        assert fine.R == self.R and m * self.J == fine.J
        return np.repeat(w / m, m, axis=-1)

    def embed(self, w: np.ndarray, big: "PeriodicGrid") -> np.ndarray:
        """Zero extension to a larger domain with the same cell width."""
        off = round((big.R - self.R) / self.dx)
        assert np.isclose(big.dx, self.dx) and off >= 0
        out = np.zeros(w.shape[:-1] + (big.J,))
        out[..., off:off + self.J] = w
        return out


def normal_interval_prob(lo, hi, mean: float, sd: float) -> np.ndarray:
    """P(lo <= Z < hi) for Z ~ N(mean, sd^2); upper-tail formula right of the mean."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    right = lo >= mean
    return np.where(right, norm.sf(lo, mean, sd) - norm.sf(hi, mean, sd),
                    norm.cdf(hi, mean, sd) - norm.cdf(lo, mean, sd))


def wrapped_normal_masses(grid: PeriodicGrid, mean: float, sd: float, tol: float = 1e-13):
    """Cell masses of N(mean, sd^2) wrapped onto [-R, R): cell i collects all of its
    translates by 2mR. Whole periods are summed until the omitted tails are below tol.
    Returns (masses, omitted probability)."""
    period = 2 * grid.R
    a, b = norm.ppf(tol / 2, mean, sd), norm.isf(tol / 2, mean, sd)
    m_lo, m_hi = int(np.floor((a + grid.R) / period)), int(np.floor((b + grid.R) / period))
    w = np.zeros(grid.J)
    for m in range(m_lo, m_hi + 1):
        e = grid.edges + m * period
        w += normal_interval_prob(e[:-1], e[1:], mean, sd)
    omitted = norm.cdf(-grid.R + m_lo * period, mean, sd) + norm.sf(-grid.R + (m_hi + 1) * period, mean, sd)
    return w, float(omitted)


def wrapped_jump_kernel(grid: PeriodicGrid, mean: float, sd: float, tol: float = 1e-13):
    """Rounded jump law p_k = P((k - 1/2) dx <= Z < (k + 1/2) dx), wrapped modulo J.
    Index 0 is no shift, index 1 a shift by +dx, index J-1 a shift by -dx.
    Returns (kernel, omitted probability); the kernel is not renormalized here."""
    dx = grid.dx
    a, b = norm.ppf(tol / 2, mean, sd), norm.isf(tol / 2, mean, sd)
    k = np.arange(int(np.floor(a / dx + 0.5)), int(np.floor(b / dx + 0.5)) + 1)
    pk = normal_interval_prob((k - 0.5) * dx, (k + 0.5) * dx, mean, sd)
    kernel = np.zeros(grid.J)
    np.add.at(kernel, k % grid.J, pk)
    omitted = norm.cdf((k[0] - 0.5) * dx, mean, sd) + norm.sf((k[-1] + 0.5) * dx, mean, sd)
    return kernel, float(omitted)


@dataclass
class Setup:
    """Initial masses and jump kernel on one grid, normalized to probability vectors.
    The omitted tail probabilities and the normalizing sums are kept as diagnostics."""
    grid: PeriodicGrid
    model: Model
    tol: float
    w0: np.ndarray
    kernel: np.ndarray
    diag: dict

    @classmethod
    def build(cls, grid: PeriodicGrid, model: Model, tol: float = 1e-13) -> "Setup":
        w0, om0 = wrapped_normal_masses(grid, model.u0_mean, model.u0_sd, tol)
        ker, omk = wrapped_jump_kernel(grid, model.jump_mean, model.jump_sd, tol)
        diag = dict(tol=tol, u0_omitted=om0, u0_sum=float(w0.sum()),
                    jump_omitted=omk, jump_sum=float(ker.sum()))
        return cls(grid, model, tol, w0 / w0.sum(), ker / ker.sum(), diag)


# -- generators and predictors -----------------------------------------------------
def upwind_rates(grid: PeriodicGrid, model: Model):
    x = grid.x
    a, b = _coef(model.sigma, x) ** 2, _coef(model.b, x)
    rp = a / (2 * grid.dx**2) + np.maximum(b, 0) / grid.dx
    rm = a / (2 * grid.dx**2) + np.maximum(-b, 0) / grid.dx
    return rp, rm


def generator_matrices(setup: Setup):
    """Sparse periodic A and dense J acting on cell masses:
    (A w)_i = r+_{i-1} w_{i-1} + r-_{i+1} w_{i+1} - (r+_i + r-_i) w_i,
    (J w)_i = lam (sum_k p_k w_{i-k} - w_i)."""
    J = setup.grid.J
    rp, rm = upwind_rates(setup.grid, setup.model)
    i = np.arange(J)
    A = sp.csr_matrix((np.concatenate([rp[(i - 1) % J], rm[(i + 1) % J], -(rp + rm)]),
                       (np.concatenate([i, i, i]), np.concatenate([(i - 1) % J, (i + 1) % J, i]))),
                      shape=(J, J))
    P = setup.kernel[(i[:, None] - i[None, :]) % J]
    return A, setup.model.lam * (P - np.eye(J))


class FFTPredictor:
    """Circulant predictor for constant coefficients; kind "grid_exp" or "practical"."""

    def __init__(self, setup: Setup, kind: str):
        if not setup.model.constant:
            raise ValueError("FFTPredictor needs constant coefficients; use MatrixPredictor")
        if kind not in ("grid_exp", "practical"):
            raise ValueError(kind)
        g = setup.grid
        theta = 2 * np.pi * np.arange(g.J // 2 + 1) / g.J
        rp, rm = (r[0] for r in upwind_rates(g, setup.model))
        self.zA = rp * (np.exp(-1j * theta) - 1) + rm * (np.exp(1j * theta) - 1)
        self.zJ = setup.model.lam * (sfft.rfft(setup.kernel) - 1)
        self.J, self.kind, self._cache = g.J, kind, {}

    def factor(self, r: float) -> np.ndarray:
        if r not in self._cache:
            if self.kind == "grid_exp":
                self._cache[r] = np.exp(r * (self.zA + self.zJ))
            else:
                self._cache[r] = np.exp(r * self.zJ) / (1 - r * self.zA)
        return self._cache[r]

    def __call__(self, w: np.ndarray, r: float) -> np.ndarray:
        what = sfft.rfft(w, axis=-1, workers=WORKERS)
        return sfft.irfft(what * self.factor(r), n=self.J, axis=-1, workers=WORKERS)


class MatrixPredictor:
    """Predictor from the assembled matrices (any coefficients).
    "practical": sparse LU of the cyclic tridiagonal I - rA, then exp(rJ) by FFT;
    "grid_exp":  dense exp(r (A + J)) (moderate J only)."""

    def __init__(self, setup: Setup, kind: str):
        if kind not in ("grid_exp", "practical"):
            raise ValueError(kind)
        self.A, self.Jm = generator_matrices(setup)
        self.zJ = setup.model.lam * (sfft.rfft(setup.kernel) - 1)
        self.J, self.kind, self._cache = setup.grid.J, kind, {}

    def __call__(self, w: np.ndarray, r: float) -> np.ndarray:
        if r not in self._cache:
            if self.kind == "grid_exp":
                self._cache[r] = expm(r * (self.A.toarray() + self.Jm))
            else:
                self._cache[r] = spla.splu(sp.csc_matrix(sp.identity(self.J) - r * self.A))
        op = self._cache[r]
        if self.kind == "grid_exp":
            return w @ op.T
        v = op.solve(np.ascontiguousarray(np.atleast_2d(w).T)).T.reshape(w.shape)
        return sfft.irfft(sfft.rfft(v, axis=-1) * np.exp(r * self.zJ), n=self.J, axis=-1)


def make_predictor(setup: Setup, kind: str):
    return FFTPredictor(setup, kind) if setup.model.constant else MatrixPredictor(setup, kind)


# -- observation update ----------------------------------------------------------
class Observation:
    """G_i = h(x_i) dY - h(x_i)^2 dt / 2 + dN log ell(x_i) - (ell(x_i) - 1) dt."""

    def __init__(self, grid: PeriodicGrid, model: Model):
        x = grid.x
        self.hx = model.h(x) if model.h is not None else None
        self.ellx = model.ell(x) if model.ell is not None else None
        self.logl = np.log(self.ellx) if self.ellx is not None else None
        self.J = grid.J

    def G(self, dy: np.ndarray, dn: np.ndarray, dt: float) -> np.ndarray:
        dy, dn = np.atleast_1d(dy), np.atleast_1d(dn)
        G = np.zeros((dy.shape[0], self.J))
        if self.hx is not None:
            G += dy[:, None] * self.hx - 0.5 * dt * self.hx**2
        if self.ellx is not None:
            G += dn[:, None] * self.logl - dt * (self.ellx - 1)
        return G


def observe(q: np.ndarray, L: np.ndarray, G: np.ndarray):
    """p+ = q e^{G - log gamma}, L+ = L + log gamma with log gamma = LSE(log q + G).
    Computed with the max of G as shift, which equals the LSE formula for q >= 0
    (log 0 = -inf) and does not take logarithms of roundoff-level negative masses."""
    s = G.max(axis=-1, keepdims=True)
    t = q * np.exp(G - s)
    tot = t.sum(axis=-1, keepdims=True)
    return t / tot, L + s[..., 0] + np.log(tot[..., 0])


class Filter:
    """A batch of K filters advanced together: state p (K, J) normalized masses, L (K,)."""

    def __init__(self, setup: Setup, predictor, scheme: str, K: int):
        if scheme not in ("lie", "sym"):
            raise ValueError(scheme)
        self.pred, self.scheme = predictor, scheme
        self.obs = Observation(setup.grid, setup.model)
        self.p = np.tile(setup.w0, (K, 1))
        self.L = np.zeros(K)
        self.neg_mass = 0.0             # largest total negative mass seen (no repair is made)
        self.min_mass = np.inf          # smallest cell mass seen
        self.mass_defect = 0.0          # largest |mass after - mass before| of a prediction

    def _predict(self, r: float):
        before = self.p.sum(axis=-1)
        self.p = self.pred(self.p, r)
        self.mass_defect = max(self.mass_defect, float(np.abs(self.p.sum(axis=-1) - before).max()))
        self.neg_mass = max(self.neg_mass, float(np.maximum(-self.p, 0).sum(axis=-1).max()))
        self.min_mass = min(self.min_mass, float(self.p.min()))

    def step(self, dy: np.ndarray, dn: np.ndarray, dt: float):
        G = self.obs.G(dy, dn, dt)
        if self.scheme == "lie":
            self._predict(dt)
            self.p, self.L = observe(self.p, self.L, G)
        else:
            self._predict(dt / 2)
            self.p, self.L = observe(self.p, self.L, G)
            self._predict(dt / 2)

    def diagnostics(self) -> dict:
        return dict(neg_mass=self.neg_mass, min_mass=self.min_mass, mass_defect=self.mass_defect)


def run(setup: Setup, predictor, scheme: str, dY: np.ndarray, dN: np.ndarray, dt: float,
        save_every: int = 1):
    """Run K paths with increments dY, dN of shape (K, M) on step dt.
    Returns p (K, S, J) and L (K, S) at t = 0, save_every dt, ..., T, and diagnostics."""
    K, M = dY.shape
    assert M % save_every == 0
    f = Filter(setup, predictor, scheme, K)
    S = M // save_every + 1
    P, Ls = np.empty((K, S, setup.grid.J)), np.empty((K, S))
    P[:, 0], Ls[:, 0] = f.p, f.L
    for n in range(M):
        f.step(dY[:, n], dN[:, n], dt)
        if (n + 1) % save_every == 0:
            s = (n + 1) // save_every
            P[:, s], Ls[:, s] = f.p, f.L
    return P, Ls, f.diagnostics()


# -- observation paths under Q -----------------------------------------------------
def path_rng(seed: int, j: int) -> np.random.Generator:
    """Independent stream of path j: SeedSequence(seed, spawn_key=(j,))."""
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(j,)))


def q_increments(path_ids, eps: float, T: float, seed: int):
    """Fine increments under Q: dY = sqrt(eps) xi, dN ~ Poisson(eps), independent.
    For each path, xi is drawn first and then the counts, from the path's own stream."""
    M = round(T / eps)
    dY, dN = np.empty((len(path_ids), M)), np.empty((len(path_ids), M))
    for r, j in enumerate(path_ids):
        rng = path_rng(seed, j)
        dY[r] = np.sqrt(eps) * rng.standard_normal(M)
        dN[r] = rng.poisson(eps, M)
    return dY, dN


def coarsen(d: np.ndarray, m: int) -> np.ndarray:
    """Sum blocks of m consecutive increments along the last axis."""
    K, M = d.shape
    return d.reshape(K, M // m, m).sum(axis=-1)


def refine(dY: np.ndarray, dN: np.ndarray, eps: float, rng: np.random.Generator):
    """Halve the observation grid of an existing path, keeping its coarse increments:
    Brownian bridge midpoint B1 = B/2 + sqrt(eps/4) eta and N1 ~ Binomial(N, 1/2)."""
    B1 = dY / 2 + np.sqrt(eps / 4) * rng.standard_normal(dY.shape)
    N1 = rng.binomial(dN.astype(np.int64), 0.5)
    return (np.stack([B1, dY - B1], axis=-1).reshape(dY.shape[0], -1),
            np.stack([N1, dN - N1], axis=-1).reshape(dN.shape[0], -1).astype(float))


# -- error measures ------------------------------------------------------------------
def sq_error(p, L, p_ref, L_ref, dx: float) -> np.ndarray:
    """dx^{-1} sum_i |e^L p_i - e^{L_ref} p_ref,i|^2 with the common scale m = max(L, L_ref)."""
    m = np.maximum(L, L_ref)
    d = np.exp(L - m)[..., None] * p - np.exp(L_ref - m)[..., None] * p_ref
    return np.exp(2 * m) * np.sum(d * d, axis=-1) / dx


def log_cosh(x: np.ndarray) -> np.ndarray:
    ax = np.abs(x)
    return ax + np.log1p(np.exp(-2 * ax)) - np.log(2.0)


def cell_averages(grid: PeriodicGrid) -> dict:
    """Exact cell averages of the bounded test functions tanh x and 1/(1+x^2)."""
    e = grid.edges
    return {"tanh": np.diff(log_cosh(e)) / grid.dx, "lorentz": np.diff(np.arctan(e)) / grid.dx}


def forward_density_masses(grid: PeriodicGrid, t: float, model: Model, tol: float = 1e-15):
    """Wrapped cell masses of the exact law of X_t for constant coefficients:
    a Poisson(lam t) mixture of N(u0_mean + b t + k mu, u0_sd^2 + sigma^2 t + k s^2)."""
    lt = model.lam * t
    kmax = int(poisson.isf(tol, lt)) + 1 if lt > 0 else 0
    w = np.zeros(grid.J)
    for k in range(kmax + 1):
        m = model.u0_mean + model.b * t + k * model.jump_mean
        v = model.u0_sd**2 + model.sigma**2 * t + k * model.jump_sd**2
        w += poisson.pmf(k, lt) * wrapped_normal_masses(grid, m, np.sqrt(v), tol)[0]
    return w
