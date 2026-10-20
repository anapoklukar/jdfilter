"""Two-dimensional splitting filter on a tensor-product grid.

Same scheme as the 1D ``SplittingFilter``: implicit upwind diffusion step, exact jump
step by 2D FFT (independent jump components), multiplicative Bayes update. Restrictions
of this first version: constant diagonal diffusion, independent jump components and
initial components, a scalar diffusive observation and a scalar Cox channel.
Densities are stored as (n, n) arrays of cell masses, index [i, j] <-> (x[i], x[j]).
"""
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import numpy as np
from scipy import fft, sparse
from scipy.sparse.linalg import splu

from .jumps import discretized_jump_kernel
from .particle import systematic_resample
from .update import cox_loglik, gaussian_loglik


@dataclass(frozen=True)
class JumpDiffusionModel2D:
    """dx = b(x) dt + diag(sigma) dW + dJ in R^2, with J compound Poisson (rate ``jump_rate``)
    and jumps rho = (rho_1, rho_2) with independent components ``jump_dists``.

    ``drift(x1, x2)`` returns (b1, b2); ``obs_h(x1, x2)`` and ``cox_intensity(x1, x2)`` are scalar.
    """

    drift: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]
    sigma: Tuple[float, float]
    initial_dists: Tuple[Any, Any]
    jump_rate: float = 0.0
    jump_dists: Optional[Tuple[Any, Any]] = None
    obs_h: Optional[Callable] = None
    cox_intensity: Optional[Callable] = None

    @property
    def has_jumps(self) -> bool:
        return self.jump_rate > 0 and self.jump_dists is not None


class Grid2D:
    """Square tensor grid [lower, upper]^2 with n points per axis."""

    def __init__(self, lower: float, upper: float, n: int):
        self.x = np.linspace(lower, upper, n)
        self.n = n
        self.dx = self.x[1] - self.x[0]
        self.X1, self.X2 = np.meshgrid(self.x, self.x, indexing="ij")

    def discretize(self, dists) -> np.ndarray:
        edges = np.concatenate([[-np.inf], 0.5 * (self.x[1:] + self.x[:-1]), [np.inf]])
        return np.outer(np.diff(dists[0].cdf(edges)), np.diff(dists[1].cdf(edges)))

    def mean(self, p: np.ndarray) -> np.ndarray:
        s = p.sum()
        return np.array([(p * self.X1).sum() / s, (p * self.X2).sum() / s])

    def var(self, p: np.ndarray) -> np.ndarray:
        m, s = self.mean(p), p.sum()
        return np.array([(p * (self.X1 - m[0]) ** 2).sum() / s, (p * (self.X2 - m[1]) ** 2).sum() / s])


def local_generator_2d(grid: Grid2D, model: JumpDiffusionModel2D) -> sparse.csr_matrix:
    """Upwind Markov-chain generator for the local part, flattened index i*n + j (reflecting boundary)."""
    n, dx = grid.n, grid.dx
    b1, b2 = model.drift(grid.X1, grid.X2)
    s1, s2 = model.sigma
    rates = {  # (index shift, rate array, mask of cells where the move stays on the grid)
        +n: (s1**2 / (2 * dx**2) + np.maximum(b1, 0) / dx, np.arange(n)[:, None] < n - 1),
        -n: (s1**2 / (2 * dx**2) + np.maximum(-b1, 0) / dx, np.arange(n)[:, None] > 0),
        +1: (s2**2 / (2 * dx**2) + np.maximum(b2, 0) / dx, np.arange(n)[None, :] < n - 1),
        -1: (s2**2 / (2 * dx**2) + np.maximum(-b2, 0) / dx, np.arange(n)[None, :] > 0),
    }
    idx = np.arange(n * n).reshape(n, n)
    rows, cols, vals = [], [], []
    out = np.zeros((n, n))
    for shift, (r, mask) in rates.items():
        mask = np.broadcast_to(mask, (n, n))
        rows.append(idx[mask]); cols.append(idx[mask] + shift); vals.append(r[mask])
        out += np.where(mask, r, 0.0)
    rows.append(idx.ravel()); cols.append(idx.ravel()); vals.append(-out.ravel())
    return sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                             shape=(n * n, n * n))


class SplittingFilter2D:
    def __init__(self, model: JumpDiffusionModel2D, grid: Grid2D, scheme: str = "lie",
                 jump_tol: float = 1e-12):
        if scheme not in ("lie", "strang"):
            raise ValueError(scheme)
        self.model, self.grid, self.scheme = model, grid, scheme
        self.A = local_generator_2d(grid, model).T.tocsc()
        self._lu, self._mult = {}, {}
        self.hx = model.obs_h(grid.X1, grid.X2) if model.obs_h is not None else None
        self.lx = model.cox_intensity(grid.X1, grid.X2) if model.cox_intensity is not None else None
        if model.has_jumps:
            o1, k1 = discretized_jump_kernel(model.jump_dists[0], grid.dx, grid.n - 1, jump_tol)
            o2, k2 = discretized_jump_kernel(model.jump_dists[1], grid.dx, grid.n - 1, jump_tol)
            reach = int(max(np.abs(o1).max(), np.abs(o2).max()))
            self.L = fft.next_fast_len(grid.n + 2 * reach)      # single jumps never wrap onto the grid
            kbuf = np.zeros((self.L, self.L))
            np.add.at(kbuf, ((o1 % self.L)[:, None], (o2 % self.L)[None, :]), np.outer(k1, k2))
            self.khat = fft.rfft2(kbuf)

    def _diffuse(self, p, dt):
        if dt not in self._lu:
            N = self.grid.n ** 2
            self._lu[dt] = splu((sparse.identity(N, format="csc") - dt * self.A).tocsc())
        return self._lu[dt].solve(p.ravel()).reshape(p.shape)

    def _jump(self, p, dt):
        if dt not in self._mult:
            self._mult[dt] = np.exp(self.model.jump_rate * dt * (self.khat - 1.0))
        L, n = self.L, self.grid.n
        q = fft.irfft2(fft.rfft2(p, s=(L, L)) * self._mult[dt], s=(L, L))[:n, :n]
        return np.maximum(q, 0.0)

    def predict(self, p, dt, reverse=False):
        steps = [self._diffuse] + ([self._jump] if self.model.has_jumps else [])
        for step in (reversed(steps) if reverse else steps):
            p = step(p, dt)
        return p

    def loglik(self, dy, dn, dt):
        ll = np.zeros((self.grid.n, self.grid.n))
        if self.hx is not None and dy is not None:
            ll += gaussian_loglik(self.hx, dy, dt)
        if self.lx is not None and dn is not None:
            ll += cox_loglik(self.lx, dn, dt)
        return ll

    def run(self, path):
        grid, dt, n = self.grid, path.dt, path.n
        p = grid.discretize(self.model.initial_dists)
        mean, var, log_norm = np.empty((n, 2)), np.empty((n, 2)), np.empty(n)
        acc = 0.0
        for k in range(n):
            dy = path.dy[k] if path.dy is not None else None
            dn = path.dn[k] if path.dn is not None else None
            ll = self.loglik(dy, dn, dt)
            p = self.predict(p, dt if self.scheme == "lie" else dt / 2)
            shift = ll.max()
            p = p * np.exp(ll - shift)
            s = p.sum(); acc += shift + np.log(s); p = p / s
            if self.scheme == "strang":
                p = self.predict(p, dt / 2, reverse=True); p = p / p.sum()
            mean[k], var[k], log_norm[k] = grid.mean(p), grid.var(p), acc
        return dict(t=path.t[1:], mean=mean, var=var, log_norm=log_norm, final=p)


@dataclass(frozen=True)
class ObservationPath2D:
    dt: float
    x: np.ndarray                    # (n+1, 2)
    dy: Optional[np.ndarray] = None
    dn: Optional[np.ndarray] = None

    @property
    def n(self):
        return len(self.x) - 1

    @property
    def t(self):
        return np.arange(self.n + 1) * self.dt

    def coarsen(self, factor: int) -> "ObservationPath2D":
        m = self.n // factor
        agg = lambda a: None if a is None else a.reshape(m, factor).sum(axis=1)
        return ObservationPath2D(self.dt * factor, self.x[::factor], agg(self.dy), agg(self.dn))


def _jumps_2d(model, size, dt, rng):
    out = np.zeros((size, 2))
    if not model.has_jumps:
        return out
    counts = rng.poisson(model.jump_rate * dt, size)
    total = counts.sum()
    if total:
        who = np.repeat(np.arange(size), counts)
        for c in range(2):
            np.add.at(out[:, c], who, model.jump_dists[c].rvs(size=total, random_state=rng))
    return out


def _step_2d(model, z, dt, rng):
    b1, b2 = model.drift(z[:, 0], z[:, 1])
    noise = rng.standard_normal(z.shape) * np.sqrt(dt) * np.asarray(model.sigma)
    return z + np.column_stack([b1, b2]) * dt + noise + _jumps_2d(model, len(z), dt, rng)


def simulate_2d(model: JumpDiffusionModel2D, T: float, n_steps: int, rng) -> ObservationPath2D:
    dt = T / n_steps
    x = np.empty((n_steps + 1, 2))
    x[0] = [d.rvs(random_state=rng) for d in model.initial_dists]
    dy = np.empty(n_steps) if model.obs_h is not None else None
    dn = np.empty(n_steps) if model.cox_intensity is not None else None
    for k in range(n_steps):
        if dy is not None:
            dy[k] = model.obs_h(x[k, 0], x[k, 1]) * dt + np.sqrt(dt) * rng.standard_normal()
        if dn is not None:
            dn[k] = rng.poisson(model.cox_intensity(x[k, 0], x[k, 1]) * dt)
        x[k + 1] = _step_2d(model, x[k:k + 1], dt, rng)[0]
    return ObservationPath2D(dt, x, dy, dn)


def particle_filter_2d(model: JumpDiffusionModel2D, path: ObservationPath2D, n_particles: int,
                       rng, resample_threshold: float = 0.5):
    dt, n, N = path.dt, path.n, n_particles
    z = np.column_stack([d.rvs(size=N, random_state=rng) for d in model.initial_dists]).astype(float)
    logw = np.zeros(N)
    mean, var = np.empty((n, 2)), np.empty((n, 2))
    for k in range(n):
        z = _step_2d(model, z, dt, rng)
        ll = np.zeros(N)
        if model.obs_h is not None:
            ll += gaussian_loglik(model.obs_h(z[:, 0], z[:, 1]), path.dy[k], dt)
        if model.cox_intensity is not None:
            ll += cox_loglik(model.cox_intensity(z[:, 0], z[:, 1]), path.dn[k], dt)
        logw += ll
        w = np.exp(logw - logw.max()); w /= w.sum()
        mean[k] = w @ z
        var[k] = w @ (z - mean[k]) ** 2
        if 1.0 / (w @ w) < resample_threshold * N:
            z = z[systematic_resample(w, rng)]; logw = np.zeros(N)
    return dict(t=path.t[1:], mean=mean, var=var)


def coupled_ou_2d(theta: float = 1.0, coupling: float = 0.8, sigma: float = 0.3,
                  jump_rate: float = 1.0, jump_std: float = 1.0, h_gain: Optional[float] = 2.0,
                  cox_intensity: Optional[Callable] = None, x0_std: float = 0.3):
    """b(x) = -[[theta, -coupling], [coupling, theta]] x (damped rotation); h observes x1,
    the Cox intensity (if given, a function of one variable) depends on x2."""
    from scipy import stats
    return JumpDiffusionModel2D(
        drift=lambda x1, x2: (-theta * x1 + coupling * x2, -coupling * x1 - theta * x2),
        sigma=(sigma, sigma),
        initial_dists=(stats.norm(0, x0_std), stats.norm(0, x0_std)),
        jump_rate=jump_rate,
        jump_dists=(stats.norm(0, jump_std), stats.norm(0, jump_std)) if jump_rate > 0 else None,
        obs_h=(lambda x1, x2: h_gain * x1) if h_gain is not None else None,
        cox_intensity=(lambda x1, x2: cox_intensity(x2)) if cox_intensity is not None else None,
    )
