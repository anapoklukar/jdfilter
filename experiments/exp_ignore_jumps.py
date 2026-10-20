"""E6: what is lost by a filter that ignores or approximates the jumps.

The jump variance per unit time lam * E[rho^2] is kept fixed, while the jumps go from
frequent and small (large lam) to rare and large (small lam). Filters compared:
  correct:          the full model;
  no jumps:         the jump part is dropped (diffusion sigma only);
  moment-matched:   jumps replaced by extra diffusion, sigma_eff^2 = sigma^2 + lam E[rho^2]
                    (the same second moments, but Gaussian increments).
Metrics: MSE of the filter mean against the true signal / signal variance, and the
log-likelihood of the observations (log_norm at T) relative to the correct model.

Output: results/ignore_jumps.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from jdfilter import Grid, SplittingFilter, ou_jump_model, simulate
from jdfilter.models import sigmoid_intensity

OUT = Path(__file__).resolve().parents[1] / "results"
THETA, SIGMA, JUMP_VAR_RATE = 1.0, 0.3, 2.25        # lam * E[rho^2] fixed
RATES = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
T, N_STEPS, N_PATHS = 20.0, 2000, 8
GRID = Grid(-16, 16, 641)
OBS = {"diffusive (dy = 2x dt + dB)": dict(h_gain=2.0, cox_intensity=None),
       "Cox counts (intensity 1 to 30)": dict(h_gain=None, cox_intensity=sigmoid_intensity(1.0, 30.0, 1.5))}
STYLES = {"correct": "o-", "no jumps": "s--", "moment-matched": "^:"}


def main():
    rng = np.random.default_rng(5)
    sig_eff = np.sqrt(SIGMA**2 + JUMP_VAR_RATE)
    var_signal = (SIGMA**2 + JUMP_VAR_RATE) / (2 * THETA)
    fig, axes = plt.subplots(2, len(OBS), figsize=(11, 7), sharex=True)
    for col, (obs_name, obs_kw) in enumerate(OBS.items()):
        filters_fixed = {
            "no jumps": SplittingFilter(ou_jump_model(THETA, SIGMA, 0.0, **obs_kw), GRID),
            "moment-matched": SplittingFilter(ou_jump_model(THETA, sig_eff, 0.0, **obs_kw), GRID),
        }
        rel_mse = {k: [] for k in STYLES}
        dloglik = {k: [] for k in STYLES}
        for lam in RATES:
            true_model = ou_jump_model(THETA, SIGMA, lam, np.sqrt(JUMP_VAR_RATE / lam), **obs_kw)
            filters = {"correct": SplittingFilter(true_model, GRID), **filters_fixed}
            mse = {k: 0.0 for k in filters}
            ll = {k: 0.0 for k in filters}
            for _ in range(N_PATHS):
                path = simulate(true_model, T, N_STEPS, rng)
                for k, f in filters.items():
                    r = f.run(path)
                    mse[k] += np.mean((r.mean - path.x[1:]) ** 2) / N_PATHS
                    ll[k] += r.log_norm[-1] / N_PATHS
            for k in filters:
                rel_mse[k].append(mse[k] / var_signal)
                dloglik[k].append(ll[k] - ll["correct"])
            print(f"{obs_name:28s} lam = {lam:4.2f}  rel MSE: " +
                  ", ".join(f"{k} {mse[k] / var_signal:.3f}" for k in filters) +
                  "   log-lik vs correct: " + ", ".join(f"{k} {ll[k] - ll['correct']:+.1f}" for k in filters))
        for k, style in STYLES.items():
            axes[0, col].semilogx(RATES, rel_mse[k], style, label=k)
            axes[1, col].semilogx(RATES, dloglik[k], style, label=k)
        axes[0, col].set_title(f"observations: {obs_name}")
        axes[1, col].set_xlabel(r"jump rate $\lambda$  (rare, large jumps $\leftarrow$ $\rightarrow$ frequent, small)")
    axes[0, 0].set_ylabel("MSE / signal variance")
    axes[1, 0].set_ylabel(f"log-likelihood minus correct (T = {T:g})")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "ignore_jumps.png", dpi=150)


if __name__ == "__main__":
    main()
