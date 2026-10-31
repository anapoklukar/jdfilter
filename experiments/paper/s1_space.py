"""S1: spatial refinement of the practical Lie scheme, constant coefficients, mixed
observations, R = 8. dx in {1/8, 1/16, 1/32, 1/64} against dx_ref = 1/256 at a common
dt = 2^-11, repeated at dt = 2^-12; at dt = 2^-11 the reference is checked against 1/512.
Coarse masses are lifted to the fine grid (each fine cell gets w_i/m), so the comparison
is between piecewise constant reconstructions. All grids advance in lockstep on the same
paths and errors are recorded at every time step.
Output: results/paper/S1/s1.npz and .json.

    .venv/bin/python experiments/paper/s1_space.py [--K 128]
"""
import argparse
import time

import numpy as np

from common import EPS, OUT, SEED, T, metadata, pe, save_json

R = 8.0
J_TEST = (128, 256, 512, 1024)
J_REF, J_CHECK = 4096, 8192
LEVELS = (11, 12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=128)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    model = pe.model_C("mixed")
    setups = {J: pe.Setup.build(pe.PeriodicGrid(R, J), model) for J in J_TEST + (J_REF, J_CHECK)}
    preds = {J: pe.FFTPredictor(s, "practical") for J, s in setups.items()}
    out, diag = {}, {}
    t_paths = t_run = 0.0
    for b0 in range(0, args.K, args.batch):
        rows = slice(b0, min(b0 + args.batch, args.K))
        Kb = rows.stop - rows.start
        t0 = time.perf_counter()
        dY, dN = pe.q_increments(range(rows.start, rows.stop), EPS, T, SEED)
        t1 = time.perf_counter()
        for lev in LEVELS:
            m = round(1 / EPS / 2**lev)
            dy, dn, dt = pe.coarsen(dY, m), pe.coarsen(dN, m), 2.0**-lev
            grids = J_TEST + (J_REF,) + ((J_CHECK,) if lev == 11 else ())
            filt = {J: pe.Filter(setups[J], preds[J], "lie", Kb) for J in grids}
            fine = max(grids)
            acc = {}

            def record(n):
                w = {J: setups[J].grid.lift(np.exp(f.L)[:, None] * f.p, setups[fine].grid)
                     for J, f in filt.items()}
                pairs = [(J, J_REF) for J in J_TEST]
                if lev == 11:
                    pairs += [(J, J_CHECK) for J in J_TEST + (J_REF,)]
                for J, Jr in pairs:
                    d = w[J] - w[Jr]
                    for name, val in (("D", setups[fine].grid.norm2(d)), ("L1", np.abs(d).sum(-1))):
                        acc.setdefault(f"{name}|{lev}|{J}|{Jr}", np.zeros((Kb, dy.shape[1] + 1)))[:, n] = val

            record(0)
            for n in range(dy.shape[1]):
                for f in filt.values():
                    f.step(dy[:, n], dn[:, n], dt)
                record(n + 1)
            for key, val in acc.items():
                out.setdefault(key, np.empty((args.K, val.shape[1])))[rows] = val
            for J, f in filt.items():
                d = diag.setdefault(f"{lev}|{J}", dict(neg_mass=0.0, min_mass=np.inf, mass_defect=0.0))
                fd = f.diagnostics()
                d.update(neg_mass=max(d["neg_mass"], fd["neg_mass"]), min_mass=min(d["min_mass"], fd["min_mass"]),
                         mass_defect=max(d["mass_defect"], fd["mass_defect"]))
        t2 = time.perf_counter()
        t_paths, t_run = t_paths + t1 - t0, t_run + t2 - t1
        print(f"  S1: paths {rows.stop}/{args.K}  ({t_run:.0f} s)", flush=True)
    path = OUT / "S1" / "s1"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path.with_suffix(".npz"), **out)
    save_json(path.with_suffix(".json"), dict(
        spec=dict(R=R, J_test=J_TEST, J_ref=J_REF, J_check=J_CHECK, levels=LEVELS, K=args.K,
                  scheme="lie", predictor="practical", obs="mixed", model="C"),
        setup={J: s.diag for J, s in setups.items()}, diagnostics=diag,
        time_paths_s=t_paths, time_filters_s=t_run, metadata=metadata()))


if __name__ == "__main__":
    main()
