"""Exact or classical reference solutions used in tests and experiments."""
import numpy as np

from .simulate import ObservationPath


def kalman_bucy(path: ObservationPath, a: float, s: float, c: float, m0: float, v0: float):
    """Kalman-Bucy filter for dx = a x dt + s dW, dy = c x dt + dB (Euler in time on the path grid).

    Returns (mean, var) at t_1 .. t_n.
    """
    dt, n = path.dt, path.n
    m, v = m0, v0
    mean, var = np.empty(n), np.empty(n)
    for k in range(n):
        m = m + a * m * dt + v * c * (path.dy[k] - c * m * dt)
        v = v + (2 * a * v + s**2 - c**2 * v**2) * dt
        mean[k], var[k] = m, v
    return mean, var


def ou_jump_moments(theta: float, sigma: float, jump_rate: float, jump_mean: float,
                    jump_var: float, m0: float, v0: float, t):
    """Mean and variance of dx = -theta x dt + sigma dW + compound Poisson jumps (no observations).

    dm/dt = -theta m + lam E[rho],   dv/dt = -2 theta v + sigma^2 + lam E[rho^2].
    """
    t = np.asarray(t, dtype=float)
    e1, e2 = np.exp(-theta * t), np.exp(-2 * theta * t)
    mean = m0 * e1 + jump_rate * jump_mean * (1 - e1) / theta
    var = v0 * e2 + (sigma**2 + jump_rate * (jump_var + jump_mean**2)) * (1 - e2) / (2 * theta)
    return mean, var
