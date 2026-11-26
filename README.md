# jdfilter

Code for the paper

> D. Rupnik Poklukar, A. Poklukar. *First-order time splitting for the Zakai equation with a
> jump-diffusion signal and Gaussian and Cox observations.* Submitted to Advances in
> Computational Mathematics.

It implements Lie and symmetric splitting schemes for nonlinear filtering of a jump-diffusion
signal observed through a continuous (Gaussian) channel, a Cox counting process, or both.

## Install

```sh
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest -q
```

## Reproduce the paper

The experiments behind Table 1 and Fig. 1, and how to run them, are described in
[`experiments/paper/README.md`](experiments/paper/README.md).

## Contents

- `src/jdfilter/`: filters (`periodic.py` is the grid filter used in the paper), particle filter, models
- `experiments/paper/`: experiments of the paper
- `experiments/`: earlier exploratory experiments (convergence, 2D, cost vs. particle filter)
- `tests/`: unit tests

## License

MIT, see [`LICENSE`](LICENSE).
