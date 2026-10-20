"""Simulation of the signal and the observations on a fine time grid."""
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

from .models import JumpDiffusionModel


@dataclass(frozen=True)
class ObservationPath:
    """Observations on a uniform time grid t_k = k dt, k = 0..n.

    ``dy[k]`` and ``dn[k]`` are the increments over [t_k, t_{k+1}]; ``x`` is the true
    signal at t_0..t_n (kept for computing errors, never used by the filters).
    """

    dt: float
    x: np.ndarray
    dy: Optional[np.ndarray] = None
    dn: Optional[np.ndarray] = None

    @property
    def n(self) -> int:
        return len(self.x) - 1

    @property
    def t(self) -> np.ndarray:
        return np.arange(self.n + 1) * self.dt

    def coarsen(self, factor: int) -> "ObservationPath":
        """The same observations seen with step factor*dt (increments are summed).

        Use this to compare different step sizes on the *same* path.
        """
        if self.n % factor:
            raise ValueError(f"number of steps {self.n} is not divisible by {factor}")
        m = self.n // factor
        agg = lambda a: None if a is None else a.reshape(m, factor).sum(axis=1)
        return replace(self, dt=self.dt * factor, x=self.x[::factor],
                       dy=agg(self.dy), dn=agg(self.dn))


def compound_poisson_increments(model: JumpDiffusionModel, size: int, dt: float,
                                rng: np.random.Generator) -> np.ndarray:
    """Sum of the jumps of the compound Poisson process over one step, for ``size`` copies."""
    out = np.zeros(size)
    if not model.has_jumps:
        return out
    counts = rng.poisson(model.jump_rate * dt, size)
    total = counts.sum()
    if total:
        sizes = model.jump_dist.rvs(size=total, random_state=rng)
        np.add.at(out, np.repeat(np.arange(size), counts), sizes)
    return out


def simulate(model: JumpDiffusionModel, T: float, n_steps: int,
             rng: np.random.Generator) -> ObservationPath:
    """Euler-Maruyama for the signal with exact compound Poisson jumps per step.

    Observation increments over [t_k, t_{k+1}] use the signal at the left point t_k.
    Simulate on a fine grid and ``coarsen`` for the step sizes you want to test.
    """
    dt = T / n_steps
    x = np.empty(n_steps + 1)
    x[0] = model.initial_dist.rvs(random_state=rng)
    dy = np.empty(n_steps) if model.obs_h is not None else None
    dn = np.empty(n_steps) if model.cox_intensity is not None else None
    for k in range(n_steps):
        xk = x[k]
        if dy is not None:
            dy[k] = model.obs_h(xk) * dt + np.sqrt(dt) * rng.standard_normal()
        if dn is not None:
            dn[k] = rng.poisson(model.cox_intensity(xk) * dt)
        x[k + 1] = (xk + model.drift(xk) * dt
                    + model.diffusion(xk) * np.sqrt(dt) * rng.standard_normal()
                    + compound_poisson_increments(model, 1, dt, rng)[0])
    return ObservationPath(dt=dt, x=x, dy=dy, dn=dn)
