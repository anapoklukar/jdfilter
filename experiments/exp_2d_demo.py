"""E4a: 2D example. A damped rotation with jumps; the diffusive channel observes x1 only,
the Cox channel observes x2 only, so each coordinate is learned from a different channel
(and from the coupling in the dynamics).

Output: results/2d_demo.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from jdfilter.models import sigmoid_intensity
from jdfilter.twod import Grid2D, SplittingFilter2D, coupled_ou_2d, particle_filter_2d, simulate_2d

OUT = Path(__file__).resolve().parents[1] / "results"


def main():
    model = coupled_ou_2d(theta=1.0, coupling=0.8, sigma=0.3, jump_rate=1.0, jump_std=1.0,
                          h_gain=2.0, cox_intensity=sigmoid_intensity(1.0, 30.0, 1.5))
    path = simulate_2d(model, T=10.0, n_steps=1000, rng=np.random.default_rng(11))
    g = SplittingFilter2D(model, Grid2D(-6, 6, 97), scheme="strang").run(path)
    pf = particle_filter_2d(model, path, 20000, np.random.default_rng(12))

    t = g["t"]
    fig, ax = plt.subplots(3, 1, figsize=(10, 6.5), sharex=True, gridspec_kw={"height_ratios": [3, 3, 1]})
    for c, name in enumerate(["x1 (observed by dy = 2 x1 dt + dB)", "x2 (observed by Cox counts, intensity 1 to 30)"]):
        sd = np.sqrt(g["var"][:, c])
        ax[c].plot(path.t, path.x[:, c], color="0.25", lw=1.1, label="true signal")
        ax[c].plot(t, g["mean"][:, c], color="#1f77b4", lw=1.3, label="splitting filter (mean)")
        ax[c].fill_between(t, g["mean"][:, c] - 2 * sd, g["mean"][:, c] + 2 * sd, color="#1f77b4",
                           alpha=0.15, label="±2 sd")
        ax[c].plot(t, pf["mean"][:, c], color="#ff7f0e", lw=1.0, ls="--", label="particle filter (20000)")
        ax[c].set_ylabel(name, fontsize=8)
    ax[0].legend(frameon=False, fontsize=8, ncol=4, loc="upper left")
    ax[0].set_title("2D example: damped rotation with jumps, two observation channels")
    ax[2].vlines(path.t[:-1][path.dn > 0], 0, 1, color="0.4", lw=0.6)
    ax[2].set_yticks([]); ax[2].set_ylabel("counts", fontsize=8); ax[2].set_xlabel("t")
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "2d_demo.png", dpi=150)
    mse = lambda m: np.mean(np.sum((m - path.x[1:]) ** 2, axis=1))
    print(f"MSE splitting = {mse(g['mean']):.4f}, particle = {mse(pf['mean']):.4f}, "
          f"mean |splitting - particle| = {np.mean(np.abs(g['mean'] - pf['mean'])):.4f}")


if __name__ == "__main__":
    main()
