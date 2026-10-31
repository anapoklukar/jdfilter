"""Tables and figures for Section 9 from the saved runs (no filter is rerun here).
Output: results/paper/tables/*.csv, results/paper/figures/*.pdf|png, results/paper/summary.json.

    .venv/bin/python experiments/paper/report.py
"""
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import (LEVELS, OUT, boot_counts, ci, ess, load_study, metadata, rates, rms_errors,
                    save_json, weighted_mean, pe)

OBS = ("gaussian", "cox", "mixed")
OBS_TITLE = {"gaussian": "Gaussian", "cox": "Cox", "mixed": "mixed"}
NAME = {"lie|grid_exp": "Lie, grid exponential", "sym|grid_exp": "symmetric, grid exponential",
        "lie|practical": "Lie, practical predictor"}
STYLE = {"lie|grid_exp": dict(color="#2a78d6", marker="o", ls="-"),
         "sym|grid_exp": dict(color="#eb6834", marker="s", ls="--"),
         "lie|practical": dict(color="#1baf7a", marker="^", ls="-.")}
GUIDE = dict(color="#52514e", lw=1.0)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
                     "legend.frameon": False, "savefig.bbox": "tight", "lines.linewidth": 1.5,
                     "lines.markersize": 5, "pdf.fonttype": 42})
TAB, FIG = OUT / "tables", OUT / "figures"
DTS = 2.0 ** -np.array(LEVELS)


def write_csv(name, rows):
    TAB.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))      # union of keys, first-seen order
    with open(TAB / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)


def save_fig(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=200)
    plt.close(fig)


def methods_of(a, prefix="D|"):
    return sorted({"|".join(k.split("|")[1:3]) for k in a if k.startswith(prefix) and k.count("|") == 3},
                  key=list(NAME).index)


def level_errors(a, C, method, prefix="D|"):
    """E_T, E_grid per level, with bootstrap samples (B, levels)."""
    ET, EG, ETb, EGb = [], [], [], []
    for lev in LEVELS:
        (et, etb), (eg, egb) = rms_errors(a[f"{prefix}{method}|{lev}"], C)
        ET.append(et), EG.append(eg), ETb.append(etb), EGb.append(egb)
    return dict(ET=np.array(ET), EG=np.array(EG), ETb=np.stack(ETb, 1), EGb=np.stack(EGb, 1))


def time_rows(tag, e, method):
    rows, rate_rows = [], []
    lo_T, hi_T = ci(e["ETb"])
    lo_G, hi_G = ci(e["EGb"])
    for i, lev in enumerate(LEVELS):
        rows.append(dict(case=tag, method=method, dt=f"2^-{lev}", E_T=e["ET"][i], E_T_lo=lo_T[i], E_T_hi=hi_T[i],
                         E_grid=e["EG"][i], E_grid_lo=lo_G[i], E_grid_hi=hi_G[i]))
    for which in ("T", "G"):
        r = rates(LEVELS, e[f"E{which}"], e[f"E{which}b"])
        row = dict(case=tag, method=method, error="E_T" if which == "T" else "E_grid",
                   slope=r["slope"], slope_lo=r["slope_ci"][0], slope_hi=r["slope_ci"][1])
        for i, lev in enumerate(LEVELS[:-1]):
            row[f"rate_{lev}_{lev + 1}"] = r["pairwise"][i]
            row[f"rate_{lev}_{lev + 1}_ci"] = f"[{r['pairwise_ci'][0][i]:.2f}, {r['pairwise_ci'][1][i]:.2f}]"
        rate_rows.append(row)
    return rows, rate_rows


def plot_time(ax, e, method, key="EG"):
    lo, hi = ci(e[f"{key}b"])
    st = STYLE[method]
    ax.errorbar(DTS, e[key], yerr=np.clip([e[key] - lo, hi - e[key]], 0, None), color=st["color"], marker=st["marker"],
                ls=st["ls"], capsize=2, label=NAME[method])


def slope_guide(ax, y0, label="slope 1"):
    ax.loglog(DTS, y0 * DTS / DTS[0], ls=":", label=label, **GUIDE)


# -- T1, T2, V1 ---------------------------------------------------------------------
def report_time(summary):
    rows, rate_rows, studies = [], [], {}
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
    for ax, obs in zip(axes, OBS):
        st = load_study(OUT / "T1" / obs)
        studies[obs] = st
        a = st["arrays"]
        C = boot_counts(a["logZref"].shape[0])
        y0 = []
        for m in methods_of(a):
            e = level_errors(a, C, m)
            r, rr = time_rows(obs, e, m)
            rows += r
            rate_rows += rr
            if m != "lie|practical":
                plot_time(ax, e, m)
                y0.append(e["EG"][0])
        slope_guide(ax, 0.8 * min(y0))
        ax.set(xscale="log", yscale="log", title=f"{OBS_TITLE[obs]} observations", xlabel=r"$\Delta t$")
        ax.set_xticks(DTS, [f"$2^{{-{lev}}}$" for lev in LEVELS])
        ax.minorticks_off()
    axes[0].set_ylabel(r"$\max_n\,(\mathbb{E}_Q\|w^n-w^n_{\rm ref}\|^2)^{1/2}$")
    axes[0].legend(loc="upper left", fontsize=8)
    save_fig(fig, "t1_time")

    a = studies["mixed"]["arrays"]
    C = boot_counts(a["logZref"].shape[0])
    fig, ax = plt.subplots(figsize=(4, 3.2))
    for m in ("lie|grid_exp", "lie|practical"):
        plot_time(ax, level_errors(a, C, m), m)
    slope_guide(ax, 0.8 * level_errors(a, C, "lie|grid_exp")["EG"][0])
    ax.set(xscale="log", yscale="log", title="mixed observations, Lie splitting", xlabel=r"$\Delta t$",
           ylabel=r"$\max_n\,(\mathbb{E}_Q\|w^n-w^n_{\rm ref}\|^2)^{1/2}$")
    ax.set_xticks(DTS, [f"$2^{{-{lev}}}$" for lev in LEVELS])
    ax.minorticks_off()
    ax.legend(fontsize=8)
    save_fig(fig, "t2_practical")

    v1 = OUT / "V1" / "mixed"
    if v1.with_suffix(".npz").exists():
        st = load_study(v1)
        av = st["arrays"]
        Cv = boot_counts(av["logZref"].shape[0])
        for m in methods_of(av):
            r, rr = time_rows("V-mixed", level_errors(av, Cv, m), m)
            rows += r
            rate_rows += rr
        studies["V1"] = st
    write_csv("time_errors.csv", rows)
    write_csv("time_rates.csv", rate_rows)
    summary["time_rates"] = rate_rows
    summary["runs"] = {k: {kk: v["info"][kk] for kk in ("setup", "diagnostics", "time_paths_s", "time_filters_s")}
                       for k, v in studies.items()}
    return studies


# -- D1 -----------------------------------------------------------------------------
def rms_change(X, Y, g):
    """sqrt(mean_paths ||X - Y||^2) on grid g, maximized over the leading time axis if present."""
    d = g.norm2(X - Y)
    return float(np.sqrt(d.mean(0)).max()) if d.ndim > 1 else float(np.sqrt(d.mean()))


def report_d1(studies, summary):
    base = studies["mixed"]
    a = base["arrays"]
    K = a["logZref"].shape[0]
    C = boot_counts(K)
    meth = ("lie|grid_exp", "sym|grid_exp")
    e_base = {m: level_errors(a, C, m) for m in methods_of(a)}
    smallest = min(e["EG"][-1] for m, e in e_base.items() if m in meth)
    rows = []
    for alt in ("sym|grid_exp|12", "lie|grid_exp|11"):
        D = a[f"Dalt|{alt}"]
        dist_T, dist_G = np.sqrt(D[:, -1].mean()), np.sqrt(D.mean(0).max())
        row = dict(check=f"reference sym 2^-11 vs {alt.replace('|', ' ')}", distance_T=dist_T, distance_grid=dist_G,
                   ratio_to_smallest_error=dist_G / smallest)
        if alt.startswith("sym"):
            for m in meth:
                e_alt = level_errors(a, C, m, prefix=f"Dvs|{alt}|")
                row[f"max_rel_change_E_grid_{m}"] = float(np.max(np.abs(e_alt["EG"] / e_base[m]["EG"] - 1)))
                row[f"max_rel_change_E_T_{m}"] = float(np.max(np.abs(e_alt["ET"] / e_base[m]["ET"] - 1)))
        rows.append(row)
    for obs in ("gaussian", "cox"):
        ao = studies[obs]["arrays"]
        D = ao["Dalt|sym|grid_exp|12"]
        Co = boot_counts(ao["logZref"].shape[0])
        sm = min(level_errors(ao, Co, m)["EG"][-1] for m in meth)
        rows.append(dict(check=f"[{obs}] reference sym 2^-11 vs sym 2^-12", distance_T=np.sqrt(D[:, -1].mean()),
                         distance_grid=np.sqrt(D.mean(0).max()), ratio_to_smallest_error=np.sqrt(D.mean(0).max()) / sm))
    g8 = pe.PeriodicGrid(8.0, 1024)
    big = pe.PeriodicGrid(12.0, 1536)
    fine = pe.PeriodicGrid(8.0, 2048)
    for name, transform, g in (("dx_half", lambda w: g8.lift(w, fine), fine),
                               ("R6", lambda w: g8.embed(w, big), big), ("R12", lambda w: g8.embed(w, big), big),
                               ("tol_1e-16", lambda w: w, g8)):
        path = OUT / "D1" / name
        if not path.with_suffix(".npz").exists():
            continue
        v = load_study(path)
        av = v["arrays"]
        gv = pe.PeriodicGrid(v["info"]["spec"]["R"], v["info"]["spec"]["J"])
        vmap = (lambda w: gv.embed(w, big)) if name == "R6" else (lambda w: w)
        row = dict(check=f"variant {name} (R={gv.R:g}, dx=1/{round(1 / gv.dx)}, tol={v['info']['spec']['tol']:g})")
        for m in meth:
            ev = level_errors(av, C, m)
            row[f"max_rel_change_E_grid_{m}"] = float(np.max(np.abs(ev["EG"] / e_base[m]["EG"] - 1)))
            row[f"max_rel_change_E_T_{m}"] = float(np.max(np.abs(ev["ET"] / e_base[m]["ET"] - 1)))
            row[f"slope_E_grid_{m}"] = rates(LEVELS, ev["EG"], ev["EGb"])["slope"]
        row["reference_solution_change"] = rms_change(vmap(av["keep|ref"]), transform(a["keep|ref"]), g)
        row["reference_solution_change_ratio_to_smallest_error"] = row["reference_solution_change"] / smallest
        rows.append(row)
        studies[f"D1-{name}"] = v
    for m in meth:
        rows[0][f"slope_E_grid_{m}"] = rates(LEVELS, e_base[m]["EG"], e_base[m]["EGb"])["slope"]
    write_csv("d1_sensitivity.csv", rows)
    summary["d1"] = rows
    summary["d1_smallest_error"] = smallest


# -- S1 -----------------------------------------------------------------------------
def report_s1(summary):
    path = OUT / "S1" / "s1"
    if not path.with_suffix(".npz").exists():
        return
    st = load_study(path)
    a, sp_ = st["arrays"], st["info"]["spec"]
    J_test, J_ref, J_chk = sp_["J_test"], sp_["J_ref"], sp_["J_check"]
    K = sp_["K"]
    C = boot_counts(K)
    dxs = 16.0 / np.array(J_test)
    rows, res = [], {}
    for lev in sp_["levels"]:
        ET, EG, ETb, EGb = [], [], [], []
        for J in J_test:
            (et, etb), (eg, egb) = rms_errors(a[f"D|{lev}|{J}|{J_ref}"], C)
            ET.append(et), EG.append(eg), ETb.append(etb), EGb.append(egb)
        e = dict(ET=np.array(ET), EG=np.array(EG), ETb=np.stack(ETb, 1), EGb=np.stack(EGb, 1))
        res[lev] = e
        for i, J in enumerate(J_test):
            lo, hi = ci(e["EGb"][:, i])
            loT, hiT = ci(e["ETb"][:, i])
            Dm = a[f"D|{lev}|{J}|{J_ref}"].mean(0)
            row = dict(dt=f"2^-{lev}", dx=f"1/{round(1 / dxs[i])}", E_T=e["ET"][i], E_T_lo=loT, E_T_hi=hiT,
                       E_grid=e["EG"][i], E_grid_lo=lo, E_grid_hi=hi, argmax_t=int(Dm.argmax()) * 2.0**-lev)
            if i + 1 < len(J_test):
                for which in ("T", "G"):
                    row[f"rate_E_{which}"] = np.log2(e[f"E{which}"][i] / e[f"E{which}"][i + 1])
                    pw = np.log2(e[f"E{which}b"][:, i] / e[f"E{which}b"][:, i + 1])
                    row[f"rate_E_{which}_ci"] = f"[{ci(pw)[0]:.2f}, {ci(pw)[1]:.2f}]"
            if lev == 11:
                (etc, _), (egc, _) = rms_errors(a[f"D|11|{J}|{J_chk}"], C)
                row["E_T_vs_ref_1/512"], row["E_grid_vs_ref_1/512"] = etc, egc
            rows.append(row)
        x = np.log(dxs)
        for which in ("T", "G"):
            res[lev][f"slope_{which}"] = np.polyfit(x, np.log(e[f"E{which}"]), 1)[0]
            res[lev][f"slope_{which}_ci"] = ci(np.array([np.polyfit(x, np.log(b), 1)[0] for b in e[f"E{which}b"]]))
    (ref_chk_T, _), (ref_chk, _) = rms_errors(a[f"D|11|{J_ref}|{J_chk}"], C)
    write_csv("s1_space.csv", rows)
    summary["s1"] = dict(rows=rows, slopes={lev: {k: res[lev][k] for k in ("slope_T", "slope_T_ci", "slope_G", "slope_G_ci")}
                                            for lev in res},
                         reference_1_256_vs_1_512=dict(E_T=ref_chk_T, E_grid=ref_chk),
                         max_rel_change_dt_halving=dict(E_T=float(np.max(np.abs(res[12]["ET"] / res[11]["ET"] - 1))),
                                                        E_grid=float(np.max(np.abs(res[12]["EG"] / res[11]["EG"] - 1)))),
                         max_rel_change_ref_1_512=dict(
                             E_T=max(abs(r["E_T_vs_ref_1/512"] / r["E_T"] - 1) for r in rows if "E_T_vs_ref_1/512" in r),
                             E_grid=max(abs(r["E_grid_vs_ref_1/512"] / r["E_grid"] - 1) for r in rows
                                        if "E_grid_vs_ref_1/512" in r)),
                         info={k: st["info"][k] for k in ("diagnostics", "time_filters_s")})
    fig, ax = plt.subplots(figsize=(4, 3.2))
    e = res[11]
    for key, lab, st_ in (("EG", r"$\max_n$ (attained at $t=0$)", dict(color="#2a78d6", marker="o", ls="-")),
                          ("ET", r"$t=T$", dict(color="#eb6834", marker="s", ls="--"))):
        lo, hi = ci(e[f"{key}b"])
        ax.errorbar(dxs, e[key], yerr=np.clip([e[key] - lo, hi - e[key]], 0, None), capsize=2, label=lab, **st_)
    ax.loglog(dxs, 0.8 * res[11]["ET"][0] * dxs / dxs[0], ls=":", label="slope 1", **GUIDE)
    ax.set(xscale="log", yscale="log", xlabel=r"$\Delta x$", title=r"practical Lie scheme, $\Delta t=2^{-11}$",
           ylabel=r"$(\mathbb{E}_Q\|I w^n-I w^n_{\rm ref}\|^2)^{1/2}$")
    ax.set_xticks(dxs, [f"1/{round(1 / d)}" for d in dxs])
    ax.minorticks_off()
    ax.legend(fontsize=8)
    save_fig(fig, "s1_space")


# -- N1 -----------------------------------------------------------------------------
def n1_curves(a, C, method, err="L1", logZ_key="logZref"):
    """Physical first moments K^-1 sum_j Z_ref,j(t_n) err_{j,n}: max over n and at T."""
    out = dict(max=[], maxb=[], T=[], Tb=[])
    for lev in LEVELS:
        X = a[f"{err}|{method}|{lev}"]
        s = 2 ** (max(LEVELS) - lev)
        m, mb = weighted_mean(X, a[logZ_key][:, ::s], C)
        out["max"].append(m.max()), out["maxb"].append(mb.max(1))
        out["T"].append(m[-1]), out["Tb"].append(mb[:, -1])
    return {k: np.array(v) if not k.endswith("b") else np.stack(v, 1) for k, v in out.items()}


def report_n1(studies, summary):
    rows, diag_rows, sens = [], [], []
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
    for ax, obs in zip(axes, OBS):
        a = studies[obs]["arrays"]
        K = a["logZref"].shape[0]
        C = boot_counts(K)
        lz = a["logZref"]
        ZT = np.exp(lz[:, -1])
        diag_rows.append(dict(case=obs, K=K, mean_Z_T=ZT.mean(), se_Z_T=ZT.std(ddof=1) / np.sqrt(K),
                              logZ_T_min=lz[:, -1].min(), logZ_T_q05=np.quantile(lz[:, -1], 0.05),
                              logZ_T_median=np.median(lz[:, -1]), logZ_T_q95=np.quantile(lz[:, -1], 0.95),
                              logZ_T_max=lz[:, -1].max(), ESS_T=ess(lz[:, -1]),
                              ESS_min_over_t=min(ess(lz[:, n]) for n in range(lz.shape[1]))))
        y0 = []
        for m in ("lie|grid_exp", "sym|grid_exp"):
            for err in ("L1", "tanh", "lorentz"):
                c = n1_curves(a, C, m, err)
                lo, hi = ci(c["maxb"])
                q = rates(LEVELS, c["max"], c["maxb"])
                for i, lev in enumerate(LEVELS):
                    rows.append(dict(case=obs, method=m, error=err, dt=f"2^-{lev}", E_P_max=c["max"][i],
                                     E_P_max_lo=lo[i], E_P_max_hi=hi[i], E_P_T=c["T"][i],
                                     slope=q["slope"] if i == 0 else "",
                                     slope_ci=f"[{q['slope_ci'][0]:.2f}, {q['slope_ci'][1]:.2f}]" if i == 0 else ""))
                if err == "L1":
                    st = STYLE[m]
                    ax.errorbar(DTS, c["max"], yerr=np.clip([c["max"] - lo, hi - c["max"]], 0, None), capsize=2, color=st["color"],
                                marker=st["marker"], ls=st["ls"], label=NAME[m])
                    y0.append(c["max"][0])
            # sensitivity to the reference: errors and weights from sym 2^-12
            c0 = n1_curves(a, C, m)
            c1 = n1_curves(a, C, m, err="L1vs|sym|grid_exp|12", logZ_key="logZalt|sym|grid_exp|12")
            sens.append(dict(case=obs, method=m, change="reference 2^-11 -> 2^-12",
                             max_rel_change=float(np.max(np.abs(c1["max"] / c0["max"] - 1)))))
            if obs == "mixed":
                for name in ("dx_half", "R6", "R12", "tol_1e-16"):
                    key = f"D1-{name}"
                    if key in studies:
                        cv = n1_curves(studies[key]["arrays"], C, m)
                        sens.append(dict(case=obs, method=m, change=name,
                                         max_rel_change=float(np.max(np.abs(cv["max"] / c0["max"] - 1)))))
        g = min(y0) * 0.8
        ax.loglog(DTS, g * DTS / DTS[0], ls=":", label=r"$C\,\Delta t$", **GUIDE)
        lg = DTS * np.sqrt(1 + np.abs(np.log(DTS)))
        ax.loglog(DTS, g * lg / lg[0], ls="--", label=r"$C\,\Delta t\sqrt{1+|\log\Delta t|}$", **GUIDE)
        ax.set(title=f"{OBS_TITLE[obs]} observations", xlabel=r"$\Delta t$")
        ax.set_xticks(DTS, [f"$2^{{-{lev}}}$" for lev in LEVELS])
        ax.minorticks_off()
    axes[0].set_ylabel(r"$\max_n\,\widehat{\mathbb{E}}_P\|p^n-p^n_{\rm ref}\|_{L^1}$")
    axes[0].legend(loc="upper left", fontsize=7)
    save_fig(fig, "n1_normalized")
    write_csv("n1_normalized.csv", rows)
    write_csv("n1_weights.csv", diag_rows)
    write_csv("n1_sensitivity.csv", sens)
    summary["n1_weights"] = diag_rows
    summary["n1_sensitivity"] = sens
    summary["n1_slopes"] = [r for r in rows if r["slope"] != ""]


def main():
    summary = {}
    studies = report_time(summary)
    report_d1(studies, summary)
    report_s1(summary)
    report_n1(studies, summary)
    save_json(OUT / "summary.json", dict(summary, metadata=metadata()))
    print((TAB / "time_rates.csv").read_text())


if __name__ == "__main__":
    main()
