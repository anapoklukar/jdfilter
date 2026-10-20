"""Uniform 1D grid. Distributions on the grid are stored as cell masses p_i (sum = 1)."""
import numpy as np


class Grid:
    def __init__(self, lower: float, upper: float, n: int):
        self.x = np.linspace(lower, upper, n)
        self.n = n
        self.dx = self.x[1] - self.x[0]

    def discretize(self, dist) -> np.ndarray:
        """Cell masses of a scipy distribution; the tails are put into the two boundary cells."""
        edges = np.concatenate([[-np.inf], 0.5 * (self.x[1:] + self.x[:-1]), [np.inf]])
        return np.diff(dist.cdf(edges))

    def mean(self, p: np.ndarray) -> float:
        return float(p @ self.x / p.sum())

    def var(self, p: np.ndarray) -> float:
        m = self.mean(p)
        return float(p @ (self.x - m) ** 2 / p.sum())

    def density(self, p: np.ndarray) -> np.ndarray:
        """Cell masses -> density values (for plotting)."""
        return p / (p.sum() * self.dx)
