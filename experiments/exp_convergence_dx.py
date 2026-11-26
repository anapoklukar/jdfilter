"""Empirical convergence order of the splitting filter in the grid spacing dx.

The time step is fixed; grids on [-6, 6] are nested (each coarse point is also a point of
the reference grid), so filter densities can be compared at the coarse points.
Error = L1 distance between the filter density at T and the reference (finest grid,
same dt), averaged over paths.

Output: results/convergence_dx.png and the fitted orders printed to stdout.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from jdfilter import Grid, SplittingFilter, ou_jump_model, simulate
from jdfilter.models import sigmoid_intensity

OUT = Path(__file__).resolve().parents[1] / "results"
T, N_STEPS, N_PATHS, R = 1.0, 256, 20, 6.0
CELLS = [48, 96, 192, 384, 768]          # number of cells on [-R, R]
CELLS_REF = 3072


def main():
    rng = np.random.default_rng(1)
    model = ou_jump_model(theta=1.0, sigma=0.3, jump_rate=1.0, jump_std=1.5, h_gain=2.0,
                          cox_intensity=sigmoid_intensity(), x0_std=0.5)
    grids = [Grid(-R, R, c + 1) for c in CELLS]
    ref_grid = Grid(-R, R, CELLS_REF + 1)
    filters = [SplittingFilter(model, g) for g in grids]
    ref_filter = SplittingFilter(model, ref_grid)
    errs = np.zeros(len(CELLS))
    for _ in range(N_PATHS):
        path = simulate(model, T, 4 * N_STEPS, rng).coarsen(4)
        ref = ref_grid.density(ref_filter.run(path).final)
        for i, (g, f) in enumerate(zip(grids, filters)):
            stride = CELLS_REF // CELLS[i]
            errs[i] += np.abs(g.density(f.run(path).final) - ref[::stride]).sum() * g.dx / N_PATHS
    dxs = np.array([g.dx for g in grids])
    q = np.polyfit(np.log(dxs), np.log(errs), 1)[0]
    print(f"dx     {np.array2string(dxs, precision=4)}")
    print(f"errors {np.array2string(errs, precision=5)}   fitted order in dx: {q:.2f}")

    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.loglog(dxs, errs, "o-", label=f"splitting filter (order {q:.2f})")
    ax.loglog(dxs, errs[0] * dxs / dxs[0], "k--", lw=1, label="slope 1")
    ax.loglog(dxs, errs[0] * (dxs / dxs[0]) ** 2, "k:", lw=1, label="slope 2")
    ax.set_xlabel(r"$\Delta x$"); ax.set_ylabel(r"mean $L^1$ error of filter density at $T$")
    ax.set_title(f"Spatial convergence (diffusive + Cox obs., $\\Delta t$ = {T / N_STEPS:.4f})")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "convergence_dx.png", dpi=150)
    fig.savefig(OUT / "convergence_dx.pdf")


if __name__ == "__main__":
    main()
