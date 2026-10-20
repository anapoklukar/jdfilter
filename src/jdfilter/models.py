"""Model definitions: a 1D jump-diffusion signal and its observation channels."""
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
from scipy import stats

Array = np.ndarray


@dataclass(frozen=True)
class JumpDiffusionModel:
    """Signal and observation model.

    Signal:
        dx = b(x) dt + sigma(x) dW + dJ,
    where J is compound Poisson with rate ``jump_rate`` and additive jumps x -> x + rho,
    rho ~ ``jump_dist``.

    Observations (either or both may be present):
        diffusive:  dy = h(x) dt + dB              (``obs_h``)
        Cox:        counts with intensity lt(x)    (``cox_intensity``)

    ``jump_dist`` and ``initial_dist`` are frozen ``scipy.stats`` distributions
    (they must provide ``cdf`` and ``rvs``).
    """

    drift: Callable[[Array], Array]
    diffusion: Callable[[Array], Array]
    initial_dist: Any
    jump_rate: float = 0.0
    jump_dist: Any = None
    obs_h: Optional[Callable[[Array], Array]] = None
    cox_intensity: Optional[Callable[[Array], Array]] = None

    @property
    def has_jumps(self) -> bool:
        return self.jump_rate > 0 and self.jump_dist is not None


def sigmoid_intensity(low: float = 0.5, high: float = 4.5, slope: float = 2.0):
    """Bounded, strictly positive Cox intensity lt(x) = low + (high - low) / (1 + exp(-slope x))."""
    def lt(x):
        return low + (high - low) / (1.0 + np.exp(-slope * np.asarray(x)))
    return lt


def ou_jump_model(theta: float = 1.0, sigma: float = 0.3, jump_rate: float = 1.0,
                  jump_std: float = 1.5, h_gain: Optional[float] = 2.0,
                  cox_intensity: Optional[Callable] = None, x0_std: float = 0.1
                  ) -> JumpDiffusionModel:
    """Ornstein-Uhlenbeck signal with N(0, jump_std^2) jumps; linear observation h(x) = h_gain x."""
    return JumpDiffusionModel(
        drift=lambda x: -theta * np.asarray(x),
        diffusion=lambda x: sigma * np.ones_like(np.asarray(x, dtype=float)),
        initial_dist=stats.norm(0.0, x0_std),
        jump_rate=jump_rate,
        jump_dist=stats.norm(0.0, jump_std) if jump_rate > 0 else None,
        obs_h=(lambda x: h_gain * np.asarray(x)) if h_gain is not None else None,
        cox_intensity=cox_intensity,
    )
