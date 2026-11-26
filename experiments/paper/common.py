"""Shared machinery for the paper experiments (IDs as in experiments/paper/README.md).

Every experiment uses observation paths under Q generated on the common fine grid
EPS = 2^-13 from SeedSequence(SEED, spawn_key=(j,)) for path j, so path j is the same
realization in every experiment, scheme and resolution.
"""
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from jdfilter import periodic as pe  # noqa: E402

pe.WORKERS = int(os.environ.get("JDF_FFT_WORKERS", "1"))     # one thread; run experiments in parallel

OUT = ROOT / "results" / "paper"
SEED = 20261029
EPS = 2.0**-13
T = 1.0
LEVELS = (3, 4, 5, 6, 7)              # dt = 2^-l
B_BOOT = 1000


def metadata() -> dict:
    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        except OSError:
            return ""
    cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    return dict(git_revision=git("rev-parse", "HEAD"), git_dirty=bool(git("status", "--porcelain", "src")),
                python=sys.version.split()[0], numpy=np.__version__, scipy=scipy.__version__,
                platform=platform.platform(), cpu=cpu or platform.processor(), cpu_count=os.cpu_count(),
                fft_workers=pe.WORKERS, dtype="float64", date=time.strftime("%Y-%m-%d %H:%M:%S"),
                seed=SEED, seed_rule="path j: numpy SeedSequence(seed, spawn_key=(j,)); "
                                     "standard normals for dY first, then Poisson(eps) counts",
                eps=EPS, sampling_measure="Q (reference measure)")


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o)))


# -- coupled time-refinement study ----------------------------------------------------
@dataclass
class Spec:
    obs: str = "mixed"
    model: str = "C"
    R: float = 8.0
    J: int = 1024
    tol: float = 1e-13
    methods: tuple = (("lie", "grid_exp"), ("sym", "grid_exp"))
    levels: tuple = LEVELS
    ref: tuple = ("sym", "grid_exp", 11)          # scheme, predictor, level
    alt_refs: tuple = ()                          # further references compared with ref
    K: int = 256
    batch: int = 16
    first_path: int = 0
    keep_times: tuple = (0.25, 0.5, 0.75, 1.0)    # times at which solutions are kept (D1)

    def label(self, method) -> str:
        return f"{method[0]}|{method[1]}"


def build_setup(spec: Spec) -> "pe.Setup":
    model = pe.model_C(spec.obs) if spec.model == "C" else pe.model_V(spec.obs)
    return pe.Setup.build(pe.PeriodicGrid(spec.R, spec.J), model, spec.tol)


def time_study(spec: Spec, verbose: bool = True) -> dict:
    """Run every method at every level and the reference(s) on the same paths.

    Per-path arrays (K rows) returned in the dict:
      D|<scheme>|<pred>|<l>       squared L2 errors of unnormalized masses at t_n = n 2^-l
      L1|..., tanh|..., lorentz|... normalized L1 error and test-function errors (for N1)
      logZ|...                    accumulated log masses of the method
      logZref                     reference log mass on the 2^-max(levels) grid
      Dalt|<alt>                  squared L2 distance of an alternative reference to ref
      Dvs|<alt>|<method>|<l>      method errors measured against the alternative reference
      logZalt|<alt>, L1vs|...     its log mass and normalized L1 errors against it (N1 checks)
      keep|ref, keep|alt|<alt>    reference solutions (w = e^L p) at keep_times, for D1
    """
    setup = build_setup(spec)
    g = setup.grid
    tf = pe.cell_averages(g)
    lmax = max(spec.levels)
    stride_t = [round(t * 2**lmax) for t in spec.keep_times]
    preds = {}

    def pred(kind):
        if kind not in preds:
            preds[kind] = pe.make_predictor(setup, kind)
        return preds[kind]

    def run_level(scheme, kind, level, dY, dN, save_level):
        m = round(EPS**-1 / 2**level)
        return pe.run(setup, pred(kind), scheme, pe.coarsen(dY, m), pe.coarsen(dN, m), 2.0**-level,
                      save_every=2 ** (level - save_level))

    out, diag = {}, dict(neg_mass=0.0, min_mass=np.inf, mass_defect=0.0)

    def put(key, rows, value):
        if key not in out:
            out[key] = np.empty((spec.K,) + value.shape[1:], dtype=value.dtype)
        out[key][rows] = value

    def merge(d):
        diag["neg_mass"] = max(diag["neg_mass"], d["neg_mass"])
        diag["min_mass"] = min(diag["min_mass"], d["min_mass"])
        diag["mass_defect"] = max(diag["mass_defect"], d["mass_defect"])

    t_paths = t_run = 0.0
    for b0 in range(0, spec.K, spec.batch):
        rows = slice(b0, min(b0 + spec.batch, spec.K))
        ids = range(spec.first_path + rows.start, spec.first_path + rows.stop)
        t0 = time.perf_counter()
        dY, dN = pe.q_increments(ids, EPS, T, SEED)
        t1 = time.perf_counter()
        Pr, Lr, d = run_level(spec.ref[0], spec.ref[1], spec.ref[2], dY, dN, lmax)
        merge(d)
        put("logZref", rows, Lr)
        put("keep|ref", rows, np.exp(Lr[:, stride_t, None]) * Pr[:, stride_t])
        alts = {}
        for alt in spec.alt_refs:
            Pa, La, d = run_level(alt[0], alt[1], alt[2], dY, dN, lmax)
            merge(d)
            name = f"{alt[0]}|{alt[1]}|{alt[2]}"
            alts[name] = (Pa, La)
            put(f"Dalt|{name}", rows, pe.sq_error(Pa, La, Pr, Lr, g.dx))
            put(f"logZalt|{name}", rows, La)
            put(f"keep|alt|{name}", rows, np.exp(La[:, stride_t, None]) * Pa[:, stride_t])
        for method in spec.methods:
            lab = spec.label(method)
            for level in spec.levels:
                P, L, d = run_level(method[0], method[1], level, dY, dN, level)
                merge(d)
                s = 2 ** (lmax - level)
                Pref, Lref = Pr[:, ::s], Lr[:, ::s]
                key = f"{lab}|{level}"
                put(f"D|{key}", rows, pe.sq_error(P, L, Pref, Lref, g.dx))
                put(f"logZ|{key}", rows, L)
                diff = P - Pref
                put(f"L1|{key}", rows, np.abs(diff).sum(-1))
                for name, phi in tf.items():
                    put(f"{name}|{key}", rows, np.abs(diff @ phi))
                for name, (Pa, La) in alts.items():
                    put(f"Dvs|{name}|{key}", rows, pe.sq_error(P, L, Pa[:, ::s], La[:, ::s], g.dx))
                    put(f"L1vs|{name}|{key}", rows, np.abs(P - Pa[:, ::s]).sum(-1))
        t2 = time.perf_counter()
        t_paths, t_run = t_paths + t1 - t0, t_run + t2 - t1
        if verbose:
            print(f"  {spec.obs} R={spec.R} J={spec.J}: paths {rows.stop}/{spec.K}  ({t_run:.0f} s)", flush=True)
    info = dict(spec=asdict(spec), setup=setup.diag, dx=g.dx, diagnostics=diag,
                time_paths_s=t_paths, time_filters_s=t_run)
    return dict(arrays=out, info=info)


def save_study(path: Path, res: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path.with_suffix(".npz"), **res["arrays"])
    save_json(path.with_suffix(".json"), dict(res["info"], metadata=metadata()))


def load_study(path: Path) -> dict:
    with np.load(path.with_suffix(".npz")) as z:
        arrays = {k: z[k] for k in z.files}
    return dict(arrays=arrays, info=json.loads(path.with_suffix(".json").read_text()))


# -- statistics -----------------------------------------------------------------------
def boot_counts(K: int, B: int = B_BOOT, seed: int = SEED + 1) -> np.ndarray:
    """Multinomial resampling counts (B, K) of whole paths; reused for every quantity."""
    rng = np.random.default_rng(seed)
    return rng.multinomial(K, np.full(K, 1.0 / K), size=B).astype(float)


def rms_errors(D: np.ndarray, C: np.ndarray):
    """E_T = sqrt(mean_j D[j, -1]) and E_grid = max_n sqrt(mean_j D[j, n]),
    for the sample and for each bootstrap resample (rows of C)."""
    K = D.shape[0]
    m, mb = D.mean(0), C @ D / K
    return (np.sqrt(m[-1]), np.sqrt(mb[:, -1])), (np.sqrt(m.max()), np.sqrt(mb.max(1)))


def weighted_mean(X: np.ndarray, logZ: np.ndarray, C: np.ndarray):
    """K^-1 sum_j Z_j X_j (estimates E_P X from paths sampled under Q), per time column."""
    K = X.shape[0]
    Y = np.exp(logZ) * X
    return Y.mean(0), C @ Y / K


def ci(samples: np.ndarray, level: float = 0.95):
    a = (1 - level) / 2
    return np.quantile(samples, [a, 1 - a], axis=0)


def rates(levels, E: np.ndarray, Eb: np.ndarray, fit=None):
    """Pairwise rates log2(E(dt)/E(dt/2)) and the least-squares slope of log E vs log dt
    over the levels in `fit`, with bootstrap intervals."""
    levels = np.asarray(levels)
    fit = levels if fit is None else np.asarray(fit)
    pw = np.log2(E[:-1] / E[1:])
    pwb = np.log2(Eb[:, :-1] / Eb[:, 1:])
    sel = np.isin(levels, fit)
    x = -levels[sel] * np.log(2.0)
    slope = np.polyfit(x, np.log(E[sel]), 1)[0]
    slopeb = np.array([np.polyfit(x, np.log(e[sel]), 1)[0] for e in Eb])
    return dict(pairwise=pw, pairwise_ci=ci(pwb), slope=slope, slope_ci=ci(slopeb))


def ess(logZ: np.ndarray) -> float:
    """(sum Z)^2 / sum Z^2, computed from logarithms."""
    from scipy.special import logsumexp
    return float(np.exp(2 * logsumexp(logZ) - logsumexp(2 * logZ)))
