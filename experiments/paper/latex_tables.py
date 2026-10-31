"""LaTeX tables for Section 9, written from the CSV files of report.py.
Output: results/paper/tables/*.tex (pasted into sections/09_numerics.tex).

    .venv/bin/python experiments/paper/latex_tables.py
"""
import csv

from common import LEVELS, OUT

TAB = OUT / "tables"
CASE = {"gaussian": "Gaussian", "cox": "Cox", "mixed": "mixed", "V-mixed": "mixed, model V"}
METHOD = {"lie|grid_exp": r"Lie, $R^{\rm exp}$", "sym|grid_exp": r"symmetric, $R^{\rm exp}$",
          "lie|practical": r"Lie, $R^{\rm pr}$"}


def read(name):
    with open(TAB / name) as f:
        return list(csv.DictReader(f))


def time_table():
    err = read("time_errors.csv")
    rate = {(r["case"], r["method"]): r for r in read("time_rates.csv") if r["error"] == "E_grid"}
    lines = [r"\begin{tabular}{ll" + "r" * len(LEVELS) + "c}", r"\toprule",
             r"observations & scheme & " + " & ".join(rf"$2^{{-{lev}}}$" for lev in LEVELS)
             + r" & fitted rate \\", r"\midrule"]
    prev = None
    for case in CASE:
        for method in METHOD:
            rows = [r for r in err if r["case"] == case and r["method"] == method]
            if not rows:
                continue
            if prev is not None and case != prev:
                lines.append(r"\addlinespace")
            prev = case
            vals = " & ".join(f"{1e3 * float(r['E_grid']):.2f}" for r in rows)
            q = rate[(case, method)]
            lines.append(f"{CASE[case]} & {METHOD[method]} & {vals} & "
                         f"{float(q['slope']):.2f} [{float(q['slope_lo']):.2f}, {float(q['slope_hi']):.2f}]" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def weights_table():
    lines = [r"\begin{tabular}{lcccc}", r"\toprule",
             r"observations & $K^{-1}\sum_jZ_{j,\rm ref}(T)$ & "
             r"quantiles $5\%,50\%,95\%$ of $\log Z_{\rm ref}(T)$ & ESS at $T$ & smallest ESS \\", r"\midrule"]
    for r in read("n1_weights.csv"):
        lines.append(f"{CASE[r['case']]} & ${float(r['mean_Z_T']):.3f}\\pm{float(r['se_Z_T']):.3f}$ & "
                     f"${float(r['logZ_T_q05']):.2f},\\ {float(r['logZ_T_median']):.2f},\\ {float(r['logZ_T_q95']):.2f}$ & "
                     f"{float(r['ESS_T']):.0f} & {float(r['ESS_min_over_t']):.0f}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def n1_rates():
    rows = [r for r in read("n1_normalized.csv") if r["slope"] != ""]
    out = {}
    for r in rows:
        out[(r["case"], r["method"], r["error"])] = f"{float(r['slope']):.2f} {r['slope_ci']}"
    return out


def main():
    (TAB / "time_table.tex").write_text(time_table() + "\n")
    (TAB / "weights_table.tex").write_text(weights_table() + "\n")
    print(time_table())
    print(weights_table())
    for k, v in n1_rates().items():
        print(k, v)


if __name__ == "__main__":
    main()
