"""U0: implementation checks with recorded numbers (the pass/fail versions are in
tests/test_periodic.py). Output: results/paper/U0/u0.json.

    .venv/bin/python experiments/paper/u0_checks.py
"""
from dataclasses import replace

import numpy as np
from scipy.linalg import expm

from common import OUT, metadata, save_json, pe


def main():
    rep = {}
    # 1. FFT against the assembled matrices on a small grid
    s = pe.Setup.build(pe.PeriodicGrid(4.0, 128), pe.model_C())
    A, Jm = pe.generator_matrices(s)
    A = A.toarray()
    for kind, M in (("grid_exp", expm(0.07 * (A + Jm))),
                    ("practical", expm(0.07 * Jm) @ np.linalg.inv(np.eye(s.grid.J) - 0.07 * A))):
        d = M @ s.w0
        rep[f"fft_vs_matrix_{kind}"] = float(np.abs(pe.FFTPredictor(s, kind)(s.w0, 0.07) - d).max() / np.abs(d).max())

    # 2-3, 8. orientation, conservation, positivity, semigroup on the main grid
    s = pe.Setup.build(pe.PeriodicGrid(8.0, 1024), pe.model_C())
    rep["setup_main_grid"] = s.diag
    g = s.grid
    for kind in ("grid_exp", "practical"):
        f = pe.FFTPredictor(s, kind)
        rep[f"zero_frequency_{kind}"] = float(max(abs(f.factor(r)[0] - 1) for r in (2.0**-12, 2.0**-3, 1.0)))
        q = f(s.w0, 0.25)
        rep[f"mass_defect_{kind}"] = float(abs(q.sum() - 1))
        rep[f"min_mass_{kind}"] = float(q.min())
    f = pe.FFTPredictor(s, "grid_exp")
    rep["semigroup_defect"] = float(np.abs(f(f(s.w0, 0.05), 0.05) - f(s.w0, 0.1)).max())
    w = np.zeros(g.J)
    w[g.J // 2] = 1.0
    for name, mod, expect in (("jumps", replace(pe.model_C(), b=0.0, sigma=1e-8), 0.8 * 0.6),
                              ("drift", replace(pe.model_C(), lam=0.0, sigma=1e-8), 0.15)):
        sj = pe.Setup.build(g, mod)
        shift = g.x @ pe.FFTPredictor(sj, "grid_exp")(w, 0.5) - g.x @ w
        rep[f"mean_shift_{name}_per_unit_time"] = dict(computed=float(shift / 0.5), expected=expect)

    # 4-6. observation update
    obs = pe.Observation(g, s.model)
    G = obs.G(np.array([0.3, -0.1, 0.25]), np.array([1.0, 0.0, 2.0]), 0.05)
    p, L = s.w0[None], np.zeros(1)
    for k in range(3):
        p, L = pe.observe(p, L, G[k:k + 1])
    p1, L1 = pe.observe(s.w0[None], np.zeros(1), obs.G(np.array([0.45]), np.array([3.0]), 0.15))
    rep["successive_updates_defect"] = dict(p=float(np.abs(p - p1).max()), logZ=float(abs(L - L1)[0]))
    x = g.x
    for dn in range(4):
        lik = np.exp(np.tanh(x) * 0.2 - 0.5 * np.tanh(x) ** 2 * 0.1) * pe.ell0(x) ** dn * np.exp(-(pe.ell0(x) - 1) * 0.1)
        p, L = pe.observe(s.w0[None], np.zeros(1), obs.G(np.array([0.2]), np.array([float(dn)]), 0.1))
        rep[f"count_{dn}_defect"] = float(np.abs(np.exp(L[0]) * p[0] - s.w0 * lik).max())
    null = pe.Observation(g, replace(pe.model_C(), h=lambda x: 0 * x, ell=lambda x: 1 + 0 * x))
    rep["identity_update_max_G"] = float(np.abs(null.G(np.array([0.7]), np.array([3.0]), 0.1)).max())

    # 7. constant h and ell: Lie and symmetric coincide
    sc = pe.Setup.build(g, replace(pe.model_C(), h=lambda x: 0.4 + 0 * x, ell=lambda x: 1.3 + 0 * x))
    dY, dN = pe.q_increments(range(4), 2.0**-8, 1.0, 1)
    dY, dN = pe.coarsen(dY, 8), pe.coarsen(dN, 8)
    fc = pe.FFTPredictor(sc, "grid_exp")
    Pl, Ll, _ = pe.run(sc, fc, "lie", dY, dN, 2.0**-5)
    Ps, Ls, _ = pe.run(sc, fc, "sym", dY, dN, 2.0**-5)
    rep["lie_minus_sym_constant_obs"] = float(max(np.abs(Pl - Ps).max(), np.abs(Ll - Ls).max()))

    # continuous control: h = 0, ell = 1, analytic wrapped cell masses at T = 1
    model = replace(pe.model_C(), h=None, ell=None)
    rows = []
    for J in (128, 256, 512, 1024, 2048):
        sJ = pe.Setup.build(pe.PeriodicGrid(8.0, J), model)
        exact = pe.forward_density_masses(sJ.grid, 1.0, model)
        row = dict(dx=sJ.grid.dx)
        for kind, dt in (("grid_exp", 1.0), ("practical", 2.0**-11)):
            fJ = pe.FFTPredictor(sJ, kind)
            wJ = sJ.w0
            for _ in range(round(1 / dt)):
                wJ = fJ(wJ, dt)
            row[f"L2_{kind}"] = float(np.sqrt(sJ.grid.norm2(wJ - exact)))
            row[f"L1_{kind}"] = float(np.abs(wJ - exact).sum())
        rows.append(row)
    rep["forward_density_control"] = rows
    sb = pe.Setup.build(pe.PeriodicGrid(16.0, 4096), model)
    wb = pe.FFTPredictor(sb, "grid_exp")(sb.w0, 1.0)
    mean = sb.grid.x @ wb
    rep["moments_T1_R16_dx1over128"] = dict(mean=float(mean), mean_expected=0.63,
                                            var=float((sb.grid.x - mean) ** 2 @ wb), var_expected=0.97,
                                            var_expected_with_upwind=0.97 + 0.15 * sb.grid.dx)

    # Monte Carlo control under Q: E_Q Z_T = 1 and E_Q w_T = R(T) w0 (Lie, grid_exp)
    K = 1000
    dY, dN = pe.q_increments(range(K), 2.0**-5, 1.0, 7)
    P, L, d = pe.run(s, f, "lie", dY, dN, 2.0**-5, save_every=32)
    Z = np.exp(L[:, -1])
    wT = Z[:, None] * P[:, -1]
    se = wT.std(0, ddof=1) / np.sqrt(K)
    z = (wT.mean(0) - f(s.w0, 1.0)) / np.where(se > 0, se, np.inf)
    rep["mc_mass_under_Q"] = dict(K=K, dt=2.0**-5, mean_Z=float(Z.mean()), se_Z=float(Z.std(ddof=1) / np.sqrt(K)),
                                  rms_zscore_masses=float(np.sqrt(np.mean(z[se > 1e-12] ** 2))),
                                  filter_diagnostics=d, note="separate paths (seed 7), not the main paths")
    save_json(OUT / "U0" / "u0.json", dict(checks=rep, metadata=metadata()))
    for k, v in rep.items():
        print(k, v)


if __name__ == "__main__":
    main()
