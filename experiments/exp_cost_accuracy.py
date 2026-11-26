"""E5: cost versus accuracy, splitting filter versus bootstrap particle filter, in 1D and 2D.

Accuracy: RMSE over time of the filter mean against a reference filter (fine splitting).
Cost: wall-clock time of one filter run (including set-up). Averaged over a few paths.
Splitting settings refine dt and dx together (both are first order).

Output: results/cost_accuracy.png
"""
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from jdfilter import Grid, SplittingFilter, bootstrap_particle_filter, ou_jump_model, simulate
from jdfilter.models import sigmoid_intensity
from jdfilter.twod import Grid2D, SplittingFilter2D, coupled_ou_2d, particle_filter_2d, simulate_2d

OUT = Path(__file__).resolve().parents[1] / "results"
N_PATHS = 3
PARTICLES = [100, 300, 1000, 3000, 10000, 30000]


def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


def rmse(a, b):
    return float(np.sqrt(np.mean(np.sum((np.atleast_2d(a.T).T - np.atleast_2d(b.T).T) ** 2, axis=-1))))


def study_1d(rng):
    T, n_fine, R = 5.0, 4096, 7.0
    model = ou_jump_model(cox_intensity=sigmoid_intensity())
    grid_settings = [(35, 128), (70, 256), (140, 512), (280, 1024), (560, 2048)]   # (cells, steps)
    pf_steps = 1024
    res = {"split": np.zeros((len(grid_settings), 2)), "pf": np.zeros((len(PARTICLES), 2))}
    for _ in range(N_PATHS):
        fine = simulate(model, T, n_fine, rng)
        ref = SplittingFilter(model, Grid(-R, R, 1121), scheme="strang").run(fine).mean
        for i, (cells, steps) in enumerate(grid_settings):
            path = fine.coarsen(n_fine // steps)
            out, sec = timed(lambda: SplittingFilter(model, Grid(-R, R, cells + 1)).run(path))
            res["split"][i] += [rmse(out.mean, ref[n_fine // steps - 1::n_fine // steps]), sec]
        path = fine.coarsen(n_fine // pf_steps)
        for i, N in enumerate(PARTICLES):
            out, sec = timed(lambda: bootstrap_particle_filter(model, path, N, rng))
            res["pf"][i] += [rmse(out.mean, ref[n_fine // pf_steps - 1::n_fine // pf_steps]), sec]
    return {k: v / N_PATHS for k, v in res.items()}


def study_2d(rng):
    T, n_fine, R = 2.0, 1024, 6.0
    model = coupled_ou_2d(cox_intensity=sigmoid_intensity(slope=3.0))
    grid_settings = [(16, 64), (32, 128), (64, 256), (96, 512)]
    pf_steps = 256
    res = {"split": np.zeros((len(grid_settings), 2)), "pf": np.zeros((len(PARTICLES), 2))}
    for _ in range(N_PATHS):
        fine = simulate_2d(model, T, n_fine, rng)
        ref = SplittingFilter2D(model, Grid2D(-R, R, 161), scheme="strang").run(fine)["mean"]
        for i, (cells, steps) in enumerate(grid_settings):
            path = fine.coarsen(n_fine // steps)
            out, sec = timed(lambda: SplittingFilter2D(model, Grid2D(-R, R, cells + 1)).run(path))
            res["split"][i] += [rmse(out["mean"], ref[n_fine // steps - 1::n_fine // steps]), sec]
        path = fine.coarsen(n_fine // pf_steps)
        for i, N in enumerate(PARTICLES):
            out, sec = timed(lambda: particle_filter_2d(model, path, N, rng))
            res["pf"][i] += [rmse(out["mean"], ref[n_fine // pf_steps - 1::n_fine // pf_steps]), sec]
    return {k: v / N_PATHS for k, v in res.items()}


def main():
    rng = np.random.default_rng(4)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (name, study) in zip(axes, [("1D", study_1d), ("2D", study_2d)]):
        r = study(rng)
        for key, label, style in [("split", "splitting filter", "o-"), ("pf", "particle filter", "s--")]:
            ax.loglog(r[key][:, 1], r[key][:, 0], style, label=label)
            print(f"{name} {label:17s} (error, seconds): " +
                  ", ".join(f"({e:.4f}, {s:.2f})" for e, s in r[key]))
        ax.set_xlabel("wall-clock time per run [s]"); ax.set_ylabel("RMSE of filter mean vs reference")
        ax.set_title(f"{name}: cost versus accuracy"); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "cost_accuracy.png", dpi=150)
    fig.savefig(OUT / "cost_accuracy.pdf")


if __name__ == "__main__":
    main()
