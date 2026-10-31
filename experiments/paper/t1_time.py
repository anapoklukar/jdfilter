"""T1 (and T2): temporal refinement under Q on the periodic grid R = 8, dx = 1/64.

T1: Lie and symmetric splitting with the grid-exponential predictor, three observation cases.
T2: Lie splitting with the practical predictor, mixed observations (same run as T1 mixed).
Reference: symmetric grid_exp at dt = 2^-11; alternative references sym 2^-12 and Lie 2^-11
are run on the same paths (reference check of D1). The per-path arrays also feed N1.
Output: results/paper/T1/<obs>.npz and .json.

    .venv/bin/python experiments/paper/t1_time.py [--K 256] [--obs mixed gaussian cox]
"""
import argparse

from common import OUT, Spec, save_study, time_study


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=256)
    ap.add_argument("--obs", nargs="+", default=["mixed", "gaussian", "cox"])
    args = ap.parse_args()
    for obs in args.obs:
        methods = [("lie", "grid_exp"), ("sym", "grid_exp")]
        if obs == "mixed":
            methods.append(("lie", "practical"))
        spec = Spec(obs=obs, K=args.K, methods=tuple(methods),
                    alt_refs=(("sym", "grid_exp", 12), ("lie", "grid_exp", 11)))
        save_study(OUT / "T1" / obs, time_study(spec))


if __name__ == "__main__":
    main()
