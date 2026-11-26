"""Table 1 (8 rows), the sensitivity checks and Fig. 1 of the manuscript submitted to Advances
in Computational Mathematics, recomputed from the saved runs in results/paper (no filter is rerun).

    .venv/bin/python experiments/paper/acm_table.py

Statistics follow the manuscript: D[j, n] = d^{-1} sum_i (w - w_ref)^2 with unnormalized cell
masses, E = max_n sqrt(mean_j D[j, n]), slope = least-squares fit of log E on log dt over the five
unrounded E, and 95% percentile intervals from 1000 bootstrap resamples of whole paths. In each
resample the same path counts are used for all five steps and both schemes of a study.
Output: results/paper/acm/{table.csv, table.tex, sensitivity.json, time_errors.pdf|png}.
"""
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import LEVELS, OUT, boot_counts, ci, load_study, metadata, pe

ACM = OUT / "acm"
DTS = 2.0 ** -np.array(LEVELS)
SCHEMES = (("lie|grid_exp", "Lie"), ("sym|grid_exp", "Symmetric"))
CASES = (("T1/gaussian", "Gaussian"), ("T1/cox", "Cox"), ("T1/mixed", "Mixed"), ("V1/mixed", "V, mixed"))


def errors(D, C):
    """E = max_n sqrt(mean_j D[j, n]) for the sample and for each bootstrap resample."""
    K = D.shape[0]
    return np.sqrt(D.mean(0).max()), np.sqrt((C @ D / K).max(1))


def slope(E):
    return np.polyfit(np.log(DTS), np.log(E), 1)[0]


def study_errors(a, C, prefix="D|"):
    out = {}
    for m, _ in SCHEMES:
        E, Eb = zip(*(errors(a[f"{prefix}{m}|{lev}"], C) for lev in LEVELS))
        E, Eb = np.array(E), np.stack(Eb, 1)
        sb = np.array([slope(e) for e in Eb])
        out[m] = dict(E=E, Eb=Eb, slope=slope(E), slope_ci=ci(sb))
    return out


def main():
    ACM.mkdir(parents=True, exist_ok=True)
    rows, res, info = [], {}, {}
    for path, label in CASES:
        st = load_study(OUT / path)
        a = st["arrays"]
        K = a["logZref"].shape[0]
        C = boot_counts(K)
        res[path] = study_errors(a, C)
        info[path] = dict(K=K, git_revision=st["info"]["metadata"]["git_revision"],
                          git_dirty=st["info"]["metadata"]["git_dirty"], date=st["info"]["metadata"]["date"])
        for m, name in SCHEMES:
            e = res[path][m]
            rows.append(dict(case=label, scheme=name, K=K, **{f"E_2^-{lev}": e["E"][i] for i, lev in enumerate(LEVELS)},
                             slope=e["slope"], slope_lo=e["slope_ci"][0], slope_hi=e["slope_ci"][1]))

    with open(ACM / "table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lines = []
    for i, r in enumerate(rows):
        data = r["case"] if i % 2 == 0 else ""
        vals = "&".join(f"{1e3 * r[f'E_2^-{lev}']:.2f}" for lev in LEVELS)
        lines.append(f"{data:9s}& {r['scheme']:10s}&{vals}&{r['slope']:.2f} [{r['slope_lo']:.2f}, {r['slope_hi']:.2f}]\\\\")
        if i == 5:
            lines.append("\\addlinespace")
    (ACM / "table.tex").write_text("\n".join(lines) + "\n")

    # -- sensitivity checks (mixed observations, primary model, K = 1024) ------------
    a = load_study(OUT / "T1" / "mixed")["arrays"]
    C = boot_counts(a["logZref"].shape[0])
    base = res["T1/mixed"]
    smallest = min(base[m]["E"][-1] for m, _ in SCHEMES)
    sens = dict(smallest_mixed_error=smallest)
    D = a["Dalt|sym|grid_exp|12"]
    dist = np.sqrt(D.mean(0).max())
    sens["ref_2^-11_vs_2^-12"] = dict(distance=dist, ratio_to_smallest=dist / smallest, K=D.shape[0],
                                      note="max over the 129 common times of sqrt(mean_j D_j)")
    alt = study_errors(a, C, prefix="Dvs|sym|grid_exp|12|")
    sens["finer_reference_rel_change"] = {m: float(np.max(np.abs(alt[m]["E"] / base[m]["E"] - 1))) for m, _ in SCHEMES}
    sens["finer_reference_slopes"] = {m: alt[m]["slope"] for m, _ in SCHEMES}

    g8, fine, big = pe.PeriodicGrid(8.0, 1024), pe.PeriodicGrid(8.0, 2048), pe.PeriodicGrid(12.0, 1536)
    for name, transform, g in (("dx_half", lambda w: g8.lift(w, fine), fine),
                               ("R6", lambda w: g8.embed(w, big), big), ("R12", lambda w: g8.embed(w, big), big)):
        v = load_study(OUT / "D1" / name)
        av, spec = v["arrays"], v["info"]["spec"]
        gv = pe.PeriodicGrid(spec["R"], spec["J"])
        ev = study_errors(av, C)
        vmap = (lambda w: gv.embed(w, big)) if name == "R6" else (lambda w: w)
        ref_change = np.sqrt(g.norm2(vmap(av["keep|ref"]) - transform(a["keep|ref"])).mean(0)).max()
        sens[name] = dict(R=spec["R"], J=spec["J"], dx=gv.dx, K=av["logZref"].shape[0],
                          rel_change={m: float(np.max(np.abs(ev[m]["E"] / base[m]["E"] - 1))) for m, _ in SCHEMES},
                          slopes={m: ev[m]["slope"] for m, _ in SCHEMES},
                          reference_change=float(ref_change),
                          reference_change_note="max over t = 0.25, 0.5, 0.75, 1 of sqrt(mean_j ||ref_variant - ref_base||^2), "
                                                "coarse masses split equally (dx_half) or zero-extended to [-12,12) (R6, R12)")
    sens["runs"] = info
    sens["report"] = metadata()
    (ACM / "sensitivity.json").write_text(json.dumps(sens, indent=2, default=float))

    # -- figure: one panel per case, both schemes ---------------------------------------
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "legend.frameon": False, "savefig.bbox": "tight", "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 4, figsize=(11, 3), sharey=True)
    style = {"lie|grid_exp": dict(color="#2a78d6", marker="o", ls="-"),
             "sym|grid_exp": dict(color="#eb6834", marker="s", ls="--")}
    for ax, (path, label) in zip(axes, CASES):
        for m, name in SCHEMES:
            e = res[path][m]
            lo, hi = ci(e["Eb"])
            ax.errorbar(DTS, e["E"], yerr=np.clip([e["E"] - lo, hi - e["E"]], 0, None), capsize=2, label=name, **style[m])
        y0 = 0.8 * res[path]["sym|grid_exp"]["E"][0]
        ax.plot(DTS, y0 * DTS / DTS[0], ls=":", color="#52514e", lw=1, label="slope 1")
        ax.set(xscale="log", yscale="log", title="Model V, mixed" if path.startswith("V1") else label, xlabel=r"$\Delta t$")
        ax.set_xticks(DTS, [f"$2^{{-{lev}}}$" for lev in LEVELS])
        ax.minorticks_off()
    axes[0].set_ylabel(r"$\widehat E$")
    axes[0].legend(fontsize=8)
    fig.savefig(ACM / "time_errors.pdf")
    fig.savefig(ACM / "time_errors.png", dpi=200)

    print(open(ACM / "table.tex").read())
    print(json.dumps({k: v for k, v in sens.items() if k not in ("runs", "report")}, indent=1, default=float))
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
