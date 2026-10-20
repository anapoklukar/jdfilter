"""Bootstrap particle filter (baseline and independent check of the splitting filter)."""
import numpy as np

from .models import JumpDiffusionModel
from .simulate import ObservationPath, compound_poisson_increments
from .splitting import FilterResult
from .update import cox_loglik, gaussian_loglik


def systematic_resample(w: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(w)
    positions = (rng.random() + np.arange(n)) / n
    return np.minimum(np.searchsorted(np.cumsum(w), positions), n - 1)


def bootstrap_particle_filter(model: JumpDiffusionModel, path: ObservationPath, n_particles: int,
                              rng: np.random.Generator, resample_threshold: float = 0.5
                              ) -> FilterResult:
    """Propagate particles with Euler-Maruyama + compound Poisson jumps, weight by the
    observation likelihood, resample (systematic) when the effective sample size drops
    below ``resample_threshold * n_particles``."""
    dt, n, N = path.dt, path.n, n_particles
    z = model.initial_dist.rvs(size=N, random_state=rng).astype(float)
    logw = np.zeros(N)
    mean, var, log_norm = np.empty(n), np.empty(n), np.empty(n)
    acc = 0.0
    for k in range(n):
        z = (z + model.drift(z) * dt + model.diffusion(z) * np.sqrt(dt) * rng.standard_normal(N)
             + compound_poisson_increments(model, N, dt, rng))
        ll = np.zeros(N)
        if model.obs_h is not None and path.dy is not None:
            ll += gaussian_loglik(model.obs_h(z), path.dy[k], dt)
        if model.cox_intensity is not None and path.dn is not None:
            ll += cox_loglik(model.cox_intensity(z), path.dn[k], dt)
        # incremental normalizing constant: sum_i w_i exp(ll_i) with normalized previous weights
        w_prev = np.exp(logw - logw.max()); w_prev /= w_prev.sum()
        logw = logw + ll
        shift = ll.max()
        acc += shift + np.log(w_prev @ np.exp(ll - shift))
        w = np.exp(logw - logw.max()); w /= w.sum()
        mean[k] = w @ z
        var[k] = w @ (z - mean[k]) ** 2
        log_norm[k] = acc
        if 1.0 / (w @ w) < resample_threshold * N:
            z = z[systematic_resample(w, rng)]
            logw = np.zeros(N)
    return FilterResult(t=path.t[1:], mean=mean, var=var, log_norm=log_norm, final=z)
