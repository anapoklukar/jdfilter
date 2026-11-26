"""Localization error: effect of truncating the real line to the domain [-R, R].

Grids share the spacing dx, so the points of a small domain are also points of the
reference domain [-R_REF, R_REF]. Error = L1 distance between the filter densities at T
on [-R, R] plus the reference mass outside [-R, R], averaged over paths.
Expected: fast (roughly exponential or faster) decay in R.

Output: results/domain_size.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from jdfilter import Grid, SplittingFilter, ou_jump_model, simulate
from jdfilter.models import sigmoid_intensity

OUT = Path(__file__).resolve().parents[1] / "results"
T, N_STEPS, N_PATHS, DX = 1.0, 256, 20, 0.05
RS = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
R_REF = 12.0


def make_grid(R):
    return Grid(-R, R, int(round(2 * R / DX)) + 1)


def main():
    rng = np.random.default_rng(2)
    model = ou_jump_model(theta=1.0, sigma=0.3, jump_rate=1.0, jump_std=1.5, h_gain=2.0,
                          cox_intensity=sigmoid_intensity(), x0_std=0.5)
    ref_grid = make_grid(R_REF)
    ref_filter = SplittingFilter(model, ref_grid)
    grids = [make_grid(R) for R in RS]
    filters = [SplittingFilter(model, g) for g in grids]
    errs = np.zeros(len(RS))
    for _ in range(N_PATHS):
        path = simulate(model, T, 4 * N_STEPS, rng).coarsen(4)
        ref = ref_filter.run(path).final                       # cell masses, sum = 1
        for i, (R, g, f) in enumerate(zip(RS, grids, filters)):
            inside = np.abs(ref_grid.x) <= R + 1e-9
            p = f.run(path).final
            errs[i] += (np.abs(p - ref[inside]).sum() + ref[~inside].sum()) / N_PATHS
    for R, e in zip(RS, errs):
        print(f"R = {R:4.1f}   mean L1 error = {e:.3e}")

    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.semilogy(RS, errs, "o-")
    ax.set_xlabel("domain half-width R"); ax.set_ylabel(r"mean $L^1$ error of filter at $T$")
    ax.set_title(f"Localization error (diffusive + Cox obs., $\\Delta x$ = {DX})")
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "domain_size.png", dpi=150)
    fig.savefig(OUT / "domain_size.pdf")


if __name__ == "__main__":
    main()
