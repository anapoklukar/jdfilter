"""Jump step: the nonlocal part J p = lam (k * p - p) of the Fokker-Planck operator.

Additive jumps x -> x + rho are discretized to multiples of dx: a jump moves mass
by j cells with probability k_j = P(rho in ((j - 1/2) dx, (j + 1/2) dx]).
For a finite jump rate J is a bounded operator, so exp(dt J) can be applied exactly.

Two implementations (they differ only in how mass leaving the grid is treated):
  JumpStepMatrix: exp(dt J_D) with J restricted to the grid; jumps landing outside are lost.
  JumpStepFFT:    exp(dt J) on the infinite lattice via FFT, then restricted to the grid
                  (mass may leave and come back within one step). O(n log n) per step.
"""
import numpy as np
from scipy.linalg import expm

from .grid import Grid
from .models import JumpDiffusionModel


def discretized_jump_kernel(dist, dx: float, max_cells: int, tol: float = 1e-15):
    """Offsets j (in cells, |j| <= max_cells) and probabilities k_j = P(rho in ((j-1/2)dx, (j+1/2)dx])."""
    j = np.arange(-max_cells, max_cells + 1)
    edges = (np.arange(-max_cells, max_cells + 2) - 0.5) * dx
    k = np.diff(dist.cdf(edges))
    keep = np.nonzero(k > tol)[0]
    lo, hi = keep[0], keep[-1] + 1
    return j[lo:hi], k[lo:hi]


def jump_kernel(grid: Grid, model: JumpDiffusionModel, tol: float = 1e-15):
    """Offsets j (in cells) and probabilities k_j of the discretized jump distribution."""
    return discretized_jump_kernel(model.jump_dist, grid.dx, grid.n - 1, tol)


def jump_generator_matrix(grid: Grid, model: JumpDiffusionModel) -> np.ndarray:
    """Dense forward operator J_D (acting on cell masses): J_D[i, l] = lam k_{i-l} - lam [i == l]."""
    offsets, k = jump_kernel(grid, model)
    n = grid.n
    K = np.zeros((n, n))
    for off, w in zip(offsets, k):
        K += w * np.eye(n, k=-off)          # mass at l moves to l + off
    return model.jump_rate * (K - np.eye(n))


class JumpStepMatrix:
    def __init__(self, grid: Grid, model: JumpDiffusionModel):
        self.J = jump_generator_matrix(grid, model)
        self._E = {}

    def __call__(self, p: np.ndarray, dt: float) -> np.ndarray:
        if dt not in self._E:
            self._E[dt] = expm(dt * self.J)
        return np.maximum(self._E[dt] @ p, 0.0)


class JumpStepFFT:
    def __init__(self, grid: Grid, model: JumpDiffusionModel):
        offsets, k = jump_kernel(grid, model)
        self.n = grid.n
        reach = int(np.abs(offsets).max())
        self.L = int(2 ** np.ceil(np.log2(3 * grid.n + 4 * reach)))   # room so wrap-around stays off-grid
        kbuf = np.zeros(self.L)
        np.add.at(kbuf, offsets % self.L, k)
        self.khat = np.fft.rfft(kbuf)
        self.lam = model.jump_rate
        self._mult = {}

    def __call__(self, p: np.ndarray, dt: float) -> np.ndarray:
        if dt not in self._mult:
            self._mult[dt] = np.exp(self.lam * dt * (self.khat - 1.0))
        q = np.fft.irfft(np.fft.rfft(p, self.L) * self._mult[dt], self.L)[: self.n]
        return np.maximum(q, 0.0)
