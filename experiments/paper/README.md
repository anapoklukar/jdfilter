# Numerical experiments of the paper

Code for the numerical experiments of D. Rupnik Poklukar and A. Poklukar, *First-order time
splitting for the Zakai equation with a jump-diffusion signal and Gaussian and Cox
observations* (submitted to Advances in Computational Mathematics). Table 1, the
sensitivity numbers in Section 4 and Fig. 1 are produced by `acm_table.py` from the T1, V1
and D1 runs below; the remaining experiments are additional checks not reported in the
paper.

Implementation: `src/jdfilter/periodic.py` (periodic cell-mass grid, `grid_exp` and
`practical` predictors, observation update with log-mass bookkeeping, observation paths
under Q). Checks: `tests/test_periodic.py`.

| ID | Script | What it computes | Default size |
|---|---|---|---|
| U0 | `u0_checks.py` | operator, update, mass and analytic-density checks with recorded numbers | - |
| T1, T2 | `t1_time.py` | time refinement, three observation cases; T2 = practical Lie (mixed) | K = 256 |
| D1 | `d1_sensitivity.py` | dx/2, R = 6 and 12, tail tolerance 1e-16 (reference check is in T1) | K = 256 |
| S1 | `s1_space.py` | spatial refinement of the practical Lie scheme | K = 128 |
| V1 | `v1_variable.py` | variable-coefficient model V (optional) | K = 128 |
| N1 | `report.py` | physical-measure normalized errors, computed from the T1 and D1 runs | - |

The reported results use `--K 1024` for T1 and D1 and `--K 256` for S1 and V1:

```sh
.venv/bin/python -m pytest -q
.venv/bin/python experiments/paper/u0_checks.py
for o in mixed gaussian cox; do .venv/bin/python experiments/paper/t1_time.py --K 1024 --obs $o & done
for v in dx_half R6 R12 tol_1e-16; do .venv/bin/python experiments/paper/d1_sensitivity.py --K 1024 --variants $v & done
wait
.venv/bin/python experiments/paper/s1_space.py --K 256 & .venv/bin/python experiments/paper/v1_variable.py --K 256 & wait
.venv/bin/python experiments/paper/report.py
.venv/bin/python experiments/paper/latex_tables.py
.venv/bin/python experiments/paper/acm_table.py      # Table 1, sensitivity numbers, Fig. 1
```

Each process uses one FFT thread (`JDF_FFT_WORKERS`, default 1), so independent runs can
go in parallel. Path j is generated from `SeedSequence(20261029, spawn_key=(j,))` on the
grid eps = 2^-13 (normals for dY, then Poisson counts), so it is the same realization in
every experiment; coarser increments are block sums.

Output in `results/paper/` (not in git):
* `<ID>/*.npz` per-path arrays (squared errors `D|scheme|predictor|level` at every time of
  the level's grid, normalized L1 and test-function errors, log masses), `<ID>/*.json`
  settings, tail tolerances, negative-mass and mass diagnostics, timings, code revision;
* `tables/*.csv`, `figures/*.pdf|png`, `summary.json` from `report.py`. Bootstrap intervals
  (1000 resamples of whole paths, the same resamples for all schemes and resolutions) can
  be recomputed from the `.npz` files without rerunning any filter.

## Table 1, sensitivity numbers and Fig. 1 of the paper

Table 1 (8 rows: grid exponential, Lie and symmetric, three observation cases and model V),
the reference/grid/domain sensitivity numbers of Section 4 and Fig. 1 are computed from the
saved T1, V1 and D1 runs, without rerunning any filter:

```sh
.venv/bin/python experiments/paper/acm_table.py
```

Output in `results/paper/acm/`: `table.csv` (unrounded errors, slopes and bootstrap
intervals), `table.tex` (table rows), `sensitivity.json`, `time_errors.pdf|png`.
