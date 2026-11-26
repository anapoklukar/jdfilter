"""E4b: empirical order in dt for the 2D filter (diffusive + Cox observations).

Same method as exp_convergence_dt.py: one fine observation path per sample, coarsened
for each dt; reference = Strang at the finest dt; error = L1 distance at T.

Output: results/2d_convergence_dt.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from jdfilter.models import sigmoid_intensity
from jdfilter.twod import Grid2D, SplittingFilter2D, coupled_ou_2d, simulate_2d

OUT = Path(__file__).resolve().parents[1] / "results"
T, N_FINE, N_PATHS = 1.0, 1024, 10
COARSE = [8, 16, 32, 64, 128]


def main():
    rng = np.random.default_rng(3)
    model = coupled_ou_2d(h_gain=2.0, cox_intensity=sigmoid_intensity(slope=3.0))
    grid = Grid2D(-6, 6, 61)
    filters = {s: SplittingFilter2D(model, grid, scheme=s) for s in ("lie", "strang")}
    errs = {s: np.zeros(len(COARSE)) for s in filters}
    for _ in range(N_PATHS):
        fine = simulate_2d(model, T, N_FINE, rng)
        ref = filters["strang"].run(fine.coarsen(2))["final"]
        for s, f in filters.items():
            for i, m in enumerate(COARSE):
                errs[s][i] += np.abs(f.run(fine.coarsen(N_FINE // m))["final"] - ref).sum() / N_PATHS
    dts = T / np.array(COARSE)
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    for s, e in errs.items():
        q = np.polyfit(np.log(dts), np.log(e), 1)[0]
        print(f"{s:7s} errors {np.array2string(e, precision=4)}  order {q:.2f}")
        ax.loglog(dts, e, "o-", ms=4, label=f"{s.capitalize()} ({q:.2f})")
    e0 = errs["lie"][0]
    ax.loglog(dts, e0 * (dts / dts[0]) ** 0.5, "k:", lw=1, label="slope 1/2")
    ax.loglog(dts, e0 * dts / dts[0], "k--", lw=1, label="slope 1")
    ax.set_xlabel(r"$\Delta t$"); ax.set_ylabel(r"mean $L^1$ error at $T$")
    ax.set_title("2D: convergence in $\\Delta t$ (diffusive + Cox)"); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "2d_convergence_dt.png", dpi=150)
    fig.savefig(OUT / "2d_convergence_dt.pdf")


if __name__ == "__main__":
    main()
