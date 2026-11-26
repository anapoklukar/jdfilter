"""Empirical convergence order of the splitting filter in the time step dt.

For each random path, the observations are simulated once on a fine grid and then
coarsened, so every dt sees the same observations (as in Higham's emstrong.m).
The error is the L1 distance between the normalized filter at T and a reference
(Strang with the exact prediction at the finest dt), averaged over paths.

Output: results/convergence_dt.png and the fitted orders printed to stdout.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from jdfilter import Grid, SplittingFilter, ou_jump_model, simulate
from jdfilter.models import sigmoid_intensity

OUT = Path(__file__).resolve().parents[1] / "results"
T, N_FINE, N_PATHS = 1.0, 2**12, 20
COARSE = [8, 16, 32, 64, 128, 256]                 # numbers of steps tested
GRID = Grid(-6, 6, 241)
SCHEMES = {                                       # label -> SplittingFilter options
    "Lie, exact prediction": dict(scheme="lie", prediction="exact"),
    "Strang, exact prediction": dict(scheme="strang", prediction="exact"),
    "Lie, implicit diffusion + FFT jumps": dict(scheme="lie", prediction="split"),
}
OBS = {
    "diffusive": dict(h_gain=2.0, cox_intensity=None),
    "Cox": dict(h_gain=None, cox_intensity=sigmoid_intensity()),
    "diffusive + Cox": dict(h_gain=2.0, cox_intensity=sigmoid_intensity()),
}


def main():
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, len(OBS), figsize=(14, 4.2), sharey=True)
    dts = T / np.array(COARSE)
    for ax, (obs_name, obs_kw) in zip(axes, OBS.items()):
        model = ou_jump_model(theta=1.0, sigma=0.3, jump_rate=1.0, jump_std=1.5, **obs_kw)
        filters = {lbl: SplittingFilter(model, GRID, **kw) for lbl, kw in SCHEMES.items()}
        reference = SplittingFilter(model, GRID, scheme="strang", prediction="exact")
        errs = {lbl: np.zeros(len(COARSE)) for lbl in SCHEMES}
        for _ in range(N_PATHS):
            fine = simulate(model, T, N_FINE, rng)
            ref = reference.run(fine.coarsen(2)).final
            for lbl, f in filters.items():
                for i, m in enumerate(COARSE):
                    errs[lbl][i] += np.abs(f.run(fine.coarsen(N_FINE // m)).final - ref).sum() / N_PATHS
        for lbl, e in errs.items():
            q = np.polyfit(np.log(dts), np.log(e), 1)[0]
            print(f"{obs_name:16s} {lbl:38s} errors {np.array2string(e, precision=4)}  order {q:.2f}")
            ax.loglog(dts, e, "o-", ms=4, label=f"{lbl} ({q:.2f})")
        e0 = np.mean([e[0] for e in errs.values()])
        ax.loglog(dts, e0 * (dts / dts[0]) ** 0.5, "k:", lw=1, label="slope 1/2")
        ax.loglog(dts, e0 * (dts / dts[0]), "k--", lw=1, label="slope 1")
        ax.set_title(f"observations: {obs_name}"); ax.set_xlabel(r"$\Delta t$")
        ax.legend(frameon=False, fontsize=7)
    axes[0].set_ylabel(r"mean $L^1$ error of filter at $T$")
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "convergence_dt.png", dpi=150)
    fig.savefig(OUT / "convergence_dt.pdf")


if __name__ == "__main__":
    main()
