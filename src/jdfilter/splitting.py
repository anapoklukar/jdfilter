"""Splitting-up filter for the Zakai equation of a 1D jump-diffusion.

One step of size dt:
    Lie:     p <- U(dt) P(dt) p
    Strang:  p <- P(dt/2) U(dt) P(dt/2) p
where P is the prediction (diffusion + jumps) and U the Bayes update (multiplication
by the observation likelihood). The density is renormalized after every update and the
log of the normalizing constants is accumulated, so ``log_norm`` approximates log sigma_t(1).
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.linalg import expm

from .diffusion import DiffusionStep, local_generator
from .grid import Grid
from .jumps import JumpStepFFT, JumpStepMatrix, jump_generator_matrix
from .models import JumpDiffusionModel
from .simulate import ObservationPath
from .update import cox_loglik, gaussian_loglik


@dataclass
class FilterResult:
    t: np.ndarray                    # t_1 .. t_n
    mean: np.ndarray                 # E[x_t | observations], at t_1 .. t_n
    var: np.ndarray
    log_norm: np.ndarray             # accumulated log normalizing constants
    final: np.ndarray                # cell masses at t_n (normalized)
    densities: Optional[np.ndarray] = None   # cell masses at t_1 .. t_n, if requested


class SplittingFilter:
    """
    Parameters
    ----------
    scheme:     "lie" or "strang"
    prediction: "split" -> diffusion step, then jump step (the practical scheme);
                "exact" -> exp(dt (A + J_D)) with a dense matrix (reference; small grids only).
    diffusion:  "implicit" or "exact" (only for prediction="split").
    jumps:      "fft" or "matrix"     (only for prediction="split").
    """

    def __init__(self, model: JumpDiffusionModel, grid: Grid, scheme: str = "lie",
                 prediction: str = "split", diffusion: str = "implicit", jumps: str = "fft"):
        if scheme not in ("lie", "strang") or prediction not in ("split", "exact"):
            raise ValueError("unknown scheme or prediction")
        self.model, self.grid, self.scheme, self.prediction = model, grid, scheme, prediction
        self.hx = model.obs_h(grid.x) if model.obs_h is not None else None
        self.lx = model.cox_intensity(grid.x) if model.cox_intensity is not None else None

        if prediction == "exact":
            G = local_generator(grid, model).T.toarray()
            if model.has_jumps:
                G = G + jump_generator_matrix(grid, model)
            self._G, self._E = G, {}
        else:
            self.diff = DiffusionStep(grid, model, diffusion)
            self.jump = None
            if model.has_jumps:
                self.jump = JumpStepFFT(grid, model) if jumps == "fft" else JumpStepMatrix(grid, model)

    # -- prediction -------------------------------------------------------------
    def predict(self, p: np.ndarray, dt: float, reverse: bool = False) -> np.ndarray:
        """Advance the signal law by dt. ``reverse`` swaps the order of the two sub-steps
        (used for the second half-step of Strang, to keep the scheme symmetric)."""
        if self.prediction == "exact":
            if dt not in self._E:
                self._E[dt] = expm(dt * self._G)
            return np.maximum(self._E[dt] @ p, 0.0)
        steps = [self.diff] + ([self.jump] if self.jump is not None else [])
        for step in (reversed(steps) if reverse else steps):
            p = step(p, dt)
        return p

    # -- update -----------------------------------------------------------------
    def loglik(self, dy: Optional[float], dn: Optional[float], dt: float) -> np.ndarray:
        ll = np.zeros(self.grid.n)
        if self.hx is not None and dy is not None:
            ll += gaussian_loglik(self.hx, dy, dt)
        if self.lx is not None and dn is not None:
            ll += cox_loglik(self.lx, dn, dt)
        return ll

    # -- full run ---------------------------------------------------------------
    def run(self, path: ObservationPath, store_densities: bool = False) -> FilterResult:
        grid, dt, n = self.grid, path.dt, path.n
        p = grid.discretize(self.model.initial_dist)
        mean, var, log_norm = np.empty(n), np.empty(n), np.empty(n)
        dens = np.empty((n, grid.n)) if store_densities else None
        acc = 0.0
        for k in range(n):
            dy = path.dy[k] if path.dy is not None else None
            dn = path.dn[k] if path.dn is not None else None
            ll = self.loglik(dy, dn, dt)
            if self.scheme == "lie":
                p = self.predict(p, dt)
            else:
                p = self.predict(p, dt / 2)
            shift = ll.max()
            p = p * np.exp(ll - shift)
            s = p.sum()
            acc += shift + np.log(s)
            p = p / s
            if self.scheme == "strang":
                p = self.predict(p, dt / 2, reverse=True)
                p = p / p.sum()
            mean[k], var[k], log_norm[k] = grid.mean(p), grid.var(p), acc
            if dens is not None:
                dens[k] = p
        return FilterResult(t=path.t[1:], mean=mean, var=var, log_norm=log_norm,
                            final=p, densities=dens)
