"""Diffusion step: the local (drift + diffusion) part of the Fokker-Planck operator.

The local generator is discretized as a continuous-time Markov chain on the grid
(Kushner's upwind approximation): from cell i the chain moves to i+1 with rate
sigma^2/(2 dx^2) + b^+/dx and to i-1 with rate sigma^2/(2 dx^2) + b^-/dx.
No rate leaves the grid (reflecting boundary), so the step preserves mass.
The forward operator acting on cell masses is A = Q^T.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm_multiply, splu

from .grid import Grid
from .models import JumpDiffusionModel


def local_generator(grid: Grid, model: JumpDiffusionModel) -> sparse.csr_matrix:
    """Generator Q (rows sum to zero) of the upwind Markov chain for b(x) d/dx + sigma(x)^2/2 d^2/dx^2."""
    x, dx = grid.x, grid.dx
    b = model.drift(x)
    s2 = model.diffusion(x) ** 2
    up = s2 / (2 * dx**2) + np.maximum(b, 0) / dx
    down = s2 / (2 * dx**2) + np.maximum(-b, 0) / dx
    up[-1] = 0.0
    down[0] = 0.0
    Q = sparse.diags([down[1:], -(up + down), up[:-1]], [-1, 0, 1], format="csr")
    return Q


class DiffusionStep:
    """p -> p(t + dt) for the local part, on cell masses.

    method="implicit": implicit Euler, (I - dt A) p_new = p. First order in dt,
                       unconditionally stable, preserves positivity and mass.
    method="exact":    p_new = exp(dt A) p (exact in time for the semi-discretized problem).
    """

    def __init__(self, grid: Grid, model: JumpDiffusionModel, method: str = "implicit"):
        if method not in ("implicit", "exact"):
            raise ValueError(method)
        self.method = method
        self.A = local_generator(grid, model).T.tocsc()
        self._lu = {}

    def __call__(self, p: np.ndarray, dt: float) -> np.ndarray:
        if self.method == "exact":
            return np.maximum(expm_multiply(dt * self.A, p), 0.0)
        if dt not in self._lu:
            n = self.A.shape[0]
            self._lu[dt] = splu((sparse.identity(n, format="csc") - dt * self.A).tocsc())
        return self._lu[dt].solve(p)
