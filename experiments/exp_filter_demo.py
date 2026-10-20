"""Demo: one path of an OU signal with jumps, observed by dy = 2x dt + dB and Cox counts.

Plots the true signal, the splitting filter (mean +- 2 sd) and a particle filter.
Output: results/filter_demo.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from jdfilter import Grid, SplittingFilter, bootstrap_particle_filter, ou_jump_model, simulate
from jdfilter.models import sigmoid_intensity

OUT = Path(__file__).resolve().parents[1] / "results"


def main():
    model = ou_jump_model(theta=1.0, sigma=0.3, jump_rate=1.0, jump_std=1.5, h_gain=2.0,
                          cox_intensity=sigmoid_intensity())
    path = simulate(model, T=10.0, n_steps=2000, rng=np.random.default_rng(6))
    grid_res = SplittingFilter(model, Grid(-7, 7, 281), scheme="strang").run(path)
    pf_res = bootstrap_particle_filter(model, path, 5000, np.random.default_rng(7))

    t = grid_res.t
    sd = np.sqrt(grid_res.var)
    fig, ax = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True, gridspec_kw={"height_ratios": [4, 1]})
    ax[0].plot(path.t, path.x, color="0.25", lw=1.1, label="true signal")
    ax[0].plot(t, grid_res.mean, color="#1f77b4", lw=1.3, label="splitting filter (mean)")
    ax[0].fill_between(t, grid_res.mean - 2 * sd, grid_res.mean + 2 * sd, color="#1f77b4", alpha=0.15,
                       label="splitting filter (±2 sd)")
    ax[0].plot(t, pf_res.mean, color="#ff7f0e", lw=1.0, ls="--", label="particle filter (5000)")
    ax[0].set_ylabel("x"); ax[0].legend(frameon=False, fontsize=8, ncol=2, loc="upper left")
    ax[0].set_title("OU signal with jumps; observations dy = 2x dt + dB and Cox counts")
    events = path.t[:-1][path.dn > 0]
    ax[1].vlines(events, 0, 1, color="0.4", lw=0.6)
    ax[1].set_yticks([]); ax[1].set_ylabel("counts"); ax[1].set_xlabel("t")
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "filter_demo.png", dpi=150)
    mse = lambda m: np.mean((m - path.x[1:]) ** 2)
    print(f"MSE splitting = {mse(grid_res.mean):.4f}, particle = {mse(pf_res.mean):.4f}")


if __name__ == "__main__":
    main()
