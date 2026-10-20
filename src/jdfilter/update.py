"""Bayes update: log-likelihood of one observation increment, evaluated on the grid.

Both are relative to the reference measure of the Zakai equation (standard Brownian
motion / unit-rate Poisson process), so their sum over steps is log sigma_t(1).
"""
import numpy as np


def gaussian_loglik(hx: np.ndarray, dy: float, dt: float) -> np.ndarray:
    """log of exp(h dy - h^2 dt / 2) for dy = h(x) dt + dB."""
    return hx * dy - 0.5 * hx**2 * dt


def cox_loglik(lx: np.ndarray, dn: float, dt: float) -> np.ndarray:
    """log of lt^dn exp(-(lt - 1) dt) for counts dn with intensity lt(x)."""
    return dn * np.log(lx) - (lx - 1.0) * dt
