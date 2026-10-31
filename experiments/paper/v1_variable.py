"""V1 (optional): temporal refinement for the variable-coefficient model V,
b(x) = -tanh x, sigma(x) = 0.6 + 0.1 tanh x, mixed observations, R = 8, dx = 1/64.
grid_exp is a dense matrix exponential, practical a sparse cyclic tridiagonal solve
followed by the FFT jump factor. Same paths, reference and error measures as T1.
Output: results/paper/V1/mixed.npz and .json.

    .venv/bin/python experiments/paper/v1_variable.py [--K 128]
"""
import argparse

from common import OUT, Spec, save_study, time_study


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=128)
    args = ap.parse_args()
    spec = Spec(obs="mixed", model="V", K=args.K,
                methods=(("lie", "grid_exp"), ("sym", "grid_exp"), ("lie", "practical")),
                alt_refs=(("sym", "grid_exp", 12),))
    save_study(OUT / "V1" / "mixed", time_study(spec))


if __name__ == "__main__":
    main()
