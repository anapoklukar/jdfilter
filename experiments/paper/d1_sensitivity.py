"""D1: sensitivity of the mixed-observation T1 errors to the spatial grid, the domain and
the tail tolerance (the reference check sym 2^-11 vs 2^-12 is part of the T1 run).
Each variant repeats the T1 computation on the same observation paths.
Output: results/paper/D1/<variant>.npz and .json; compared with T1 in report.py.

    .venv/bin/python experiments/paper/d1_sensitivity.py [--K 256]
"""
import argparse

from common import OUT, Spec, save_study, time_study

VARIANTS = {
    "dx_half": dict(J=2048),                # dx = 1/128 at R = 8
    "R6": dict(R=6.0, J=768),               # dx = 1/64
    "R12": dict(R=12.0, J=1536),            # dx = 1/64
    "tol_1e-16": dict(tol=1e-16),           # stricter tails of the wrapped laws
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=256)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    args = ap.parse_args()
    for name in args.variants:
        spec = Spec(obs="mixed", K=args.K, **VARIANTS[name])
        save_study(OUT / "D1" / name, time_study(spec))


if __name__ == "__main__":
    main()
