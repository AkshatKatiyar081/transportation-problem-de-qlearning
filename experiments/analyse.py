"""Tables, statistics and figures from results/*.csv.

    python experiments/analyse.py [--format png pdf]

Outputs (results/): table1_linear, table2_discount, table3_fe, table4_sens_de, table5_sens_hybrid
(.csv + .md), stats.csv, summary.md, figures/*.png
"""
from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import BUDGET, RESULTS  # noqa: E402
from stats import compare, effect_label, holm  # noqa: E402

FIG = RESULTS / "figures"
LINEAR = ["A1", "A3", "A5", "L5", "L10"]
DISCOUNT = ["N5", "N10"]
METHODS = ["de", "de_random", "hybrid"]
LABEL = {"de": "DE", "de_random": "DE-Random", "hybrid": "Hybrid DE+QL", "modi": "MODI", "lp": "LP",
         "milp": "MILP", "naive": "Naive"}
COLORS = {"de": "#2a78d6", "de_random": "#8a8a8a", "hybrid": "#d6552a"}


def save_table(df, name):
    df.to_csv(RESULTS / f"{name}.csv", index=False)
    try:
        (RESULTS / f"{name}.md").write_text(df.to_markdown(index=False))
    except ImportError:
        (RESULTS / f"{name}.md").write_text(df.to_string(index=False))


def fe_censored(df):
    """FE to reach the optimum; runs that never reach it get budget + 1 (censored)."""
    return df.apply(lambda r: r["fe_to_opt"] if pd.notna(r["fe_to_opt"]) else BUDGET[r["instance"]] + 1, axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--format", nargs="*", default=["png"])
    a = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    runs = pd.read_csv(RESULTS / "runs.csv")
    exact = pd.read_csv(RESULTS / "exact.csv")
    main_runs = runs[runs["study"] == "main"].copy()
    main_runs["fe_c"] = fe_censored(main_runs)

    def savefig(fig, name):
        for ext in a.format:
            fig.savefig(FIG / f"{name}.{ext}", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ------------------------------------------------------------- statistics
    stat_rows = []
    for inst in LINEAR + DISCOUNT:
        d = main_runs[main_runs["instance"] == inst]
        for metric in ("cost", "fe_c"):
            block = []
            for other in ("de", "de_random"):
                h = d[d["method"] == "hybrid"].sort_values("seed")[metric].to_numpy()
                o = d[d["method"] == other].sort_values("seed")[metric].to_numpy()
                if len(h) == 0 or len(h) != len(o):
                    continue
                c = compare(h, o)
                block.append({"instance": inst, "metric": "final cost" if metric == "cost" else "FE to optimum",
                              "comparison": f"Hybrid vs {LABEL[other]}", **c})
            for r, p in zip(block, holm([b["p"] for b in block])):
                r["p_holm"] = p
                r["effect"] = effect_label(r["r_rb"])
                r["verdict"] = ("not informative (ties)" if not r["informative"] else
                                ("Hybrid better" if r["p_holm"] < 0.05 and r["r_rb"] < 0 else
                                 "Hybrid worse" if r["p_holm"] < 0.05 else "no significant difference"))
            stat_rows += block
    stats = pd.DataFrame(stat_rows)
    stats.to_csv(RESULTS / "stats.csv", index=False)

    # ---------------------------------------------------------------- tables
    def summ(d):
        return {"mean": d["cost"].mean(), "sd": d["cost"].std(ddof=1), "best": d["cost"].min(),
                "success": d["success"].mean(), "gap": d["gap_pct"].mean(), "time": d["time_s"].mean()}

    t1 = []
    for inst in LINEAR:
        e = exact[exact["instance"] == inst].set_index("method")
        row = {"Instance": inst, "Optimum (MODI = LP)": e.loc["modi", "cost"] if "modi" in e.index else None,
               "MODI time (ms)": round(1000 * e.loc["modi", "time_mean_s"], 2) if "modi" in e.index else None}
        for m in METHODS:
            s = summ(main_runs[(main_runs["instance"] == inst) & (main_runs["method"] == m)])
            row[f"{LABEL[m]} mean±sd"] = f"{s['mean']:.1f} ± {s['sd']:.1f}"
            row[f"{LABEL[m]} success"] = f"{100 * s['success']:.0f}%"
            row[f"{LABEL[m]} time (s)"] = round(s["time"], 2)
        t1.append(row)
    save_table(pd.DataFrame(t1), "table1_linear")

    t2 = []
    for inst in DISCOUNT:
        e = exact[exact["instance"] == inst].set_index("method")
        opt = e.loc["milp", "cost"]
        row = {"Instance": inst, "MILP optimum": round(opt, 2), "MILP status": e.loc["milp", "status"],
               "MILP time (s)": round(e.loc["milp", "time_mean_s"], 3),
               "Naive plan gap %": round(100 * (e.loc["naive", "cost"] - opt) / opt, 3)}
        for m in METHODS:
            s = summ(main_runs[(main_runs["instance"] == inst) & (main_runs["method"] == m)])
            row[f"{LABEL[m]} mean gap %"] = round(s["gap"], 3)
            row[f"{LABEL[m]} success"] = f"{100 * s['success']:.0f}%"
            row[f"{LABEL[m]} time (s)"] = round(s["time"], 2)
        for other in ("de", "de_random"):
            r = stats[(stats["instance"] == inst) & (stats["metric"] == "final cost") &
                      (stats["comparison"] == f"Hybrid vs {LABEL[other]}")]
            if len(r):
                r = r.iloc[0]
                row[f"Hyb vs {LABEL[other]} (cost) p_Holm / r_rb"] = f"{r['p_holm']:.3g} / {r['r_rb']:+.2f}"
        t2.append(row)
    save_table(pd.DataFrame(t2), "table2_discount")

    t3 = []
    for inst in LINEAR + DISCOUNT:
        row = {"Instance": inst, "Budget (FE)": BUDGET[inst]}
        for m in METHODS:
            d = main_runs[(main_runs["instance"] == inst) & (main_runs["method"] == m)]
            ok = d["fe_to_opt"].dropna()
            row[f"{LABEL[m]} FE to optimum (mean, n)"] = f"{ok.mean():.0f} ({len(ok)}/{len(d)})" if len(ok) else f"- (0/{len(d)})"
            tg = d["fe_to_target"].dropna()
            row[f"{LABEL[m]} FE to 0.5%"] = f"{tg.mean():.0f}" if len(tg) else "-"
        r = stats[(stats["instance"] == inst) & (stats["metric"] == "FE to optimum")]
        for _, rr in r.iterrows():
            row[f"{rr['comparison']} (FE) p_Holm / r_rb"] = f"{rr['p_holm']:.3g} / {rr['r_rb']:+.2f}"
        t3.append(row)
    save_table(pd.DataFrame(t3), "table3_fe")

    # ----------------------------------------------------------- sensitivity
    sd = runs[runs["study"] == "sens_de"].copy()
    if len(sd):
        sd["fe_c"] = fe_censored(sd)
        g = sd.groupby(["F", "CR"]).agg(mean_gap=("gap_pct", "mean"), success=("success", "mean"),
                                         mean_fe_to_opt=("fe_c", "mean")).reset_index()
        save_table(g.round(4), "table4_sens_de")
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
        for ax, col, title in ((axes[0], "mean_gap", "Mean gap % (lower is better)"),
                               (axes[1], "mean_fe_to_opt", "Mean FE to optimum (censored)")):
            piv = g.pivot(index="F", columns="CR", values=col)
            im = ax.imshow(piv.values, cmap="viridis_r", aspect="auto")
            ax.set_xticks(range(len(piv.columns)), [f"{c:g}" for c in piv.columns])
            ax.set_yticks(range(len(piv.index)), [f"{f:g}" for f in piv.index])
            ax.set_xlabel("CR")
            ax.set_ylabel("F")
            ax.set_title(title, fontsize=10)
            for i in range(piv.shape[0]):
                for j in range(piv.shape[1]):
                    ax.text(j, i, f"{piv.values[i, j]:.3g}", ha="center", va="center", color="white", fontsize=8)
            fig.colorbar(im, ax=ax, shrink=0.8)
        fig.suptitle("DE parameter sensitivity on N10 (10 seeds)")
        savefig(fig, "fig_sens_de_N10")
    sh = runs[runs["study"] == "sens_hybrid"].copy()
    if len(sh):
        sh["fe_c"] = fe_censored(sh)
        g = sh.groupby(["gamma", "alpha", "eps_frac"]).agg(mean_gap=("gap_pct", "mean"),
                                                            success=("success", "mean"),
                                                            mean_fe_to_opt=("fe_c", "mean")).reset_index()
        save_table(g.round(4), "table5_sens_hybrid")

    # --------------------------------------------------------------- figures
    hist = pd.read_csv(RESULTS / "history.csv")
    hist = hist[hist["study"] == "main"]
    for inst in LINEAR + DISCOUNT:
        ref = main_runs[main_runs["instance"] == inst]["reference"].iloc[0]
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        for m in METHODS:
            h = hist[(hist["instance"] == inst) & (hist["method"] == m)]
            if h.empty:
                continue
            h = h.assign(gap=100 * (h["best_cost"] - ref) / abs(ref))
            q = h.groupby("fe")["gap"].quantile([0.25, 0.5, 0.75]).unstack()
            ax.plot(q.index, q[0.5], color=COLORS[m], label=LABEL[m], lw=1.8)
            ax.fill_between(q.index, q[0.25], q[0.75], color=COLORS[m], alpha=0.15)
        ax.set_yscale("symlog", linthresh=0.1)
        ax.axhline(0, color="black", ls="--", lw=1, label="exact optimum")
        ax.set_xlabel("Fitness evaluations")
        ax.set_ylabel("Gap to optimum (%)")
        ax.set_title(f"{inst}: median best-so-far, IQR band (30 runs)")
        ax.legend(frameon=False)
        savefig(fig, f"fig_convergence_{inst}")

    q = pd.read_csv(RESULTS / "qtables.csv")
    q = q[q["study"] == "main"]
    log = pd.read_csv(RESULTS / "hybrid_log.csv")
    log = log[log["study"] == "main"]
    from algorithms.de import ACTIONS
    alabels = [f"a{k}\nF{f}/CR{c}" for k, (f, c) in enumerate(ACTIONS)]
    for inst in ("N5", "N10", "L10"):
        qi = q[q["instance"] == inst]
        if qi.empty:
            continue
        piv = qi.groupby(["state", "action"])["q"].mean().unstack()
        fig, ax = plt.subplots(figsize=(8, 2.8))
        im = ax.imshow(piv.values, cmap="magma", aspect="auto")
        ax.set_yticks(range(3), ["0 productive", "1 slowing", "2 stagnant"])
        ax.set_xticks(range(9), alabels, fontsize=7)
        ax.set_title(f"{inst}: mean final Q-table over 30 runs (compare within a row)")
        fig.colorbar(im, ax=ax)
        savefig(fig, f"fig_qtable_{inst}")
        li = log[log["instance"] == inst].copy()
        G = li["generation"].max() + 1
        li["phase"] = pd.cut(li["generation"], bins=10, labels=False)
        freq = li.groupby(["phase", "action"]).size().unstack(fill_value=0)
        freq = freq.div(freq.sum(axis=1), axis=0)
        fig, ax = plt.subplots(figsize=(7, 3.6))
        ax.stackplot((freq.index + 0.5) * G / 10, freq.T.values, labels=[f"a{k}" for k in freq.columns],
                     colors=plt.cm.tab10(np.arange(9)))
        ax.set_xlabel("Generation")
        ax.set_ylabel("Share of runs choosing action")
        ax.set_title(f"{inst}: Hybrid action usage over the run")
        ax.legend(ncol=9, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False)
        savefig(fig, f"fig_actions_{inst}")

    x = np.arange(0, 151)
    from core.problem import load_instance
    p = load_instance("N5")
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(x, x, "--", color="gray", label="flat rate (c = 1)")
    ax.plot(x, p.effective_units(x.reshape(1, -1))[0], color="#d6552a", lw=2, label="incremental discount")
    ax.set_xlabel("Units shipped on a route")
    ax.set_ylabel("Route cost / c")
    ax.set_title("N5/N10 cost function g(x): 30 units x1.0, next 50 x0.7, rest x0.5")
    ax.legend(frameon=False)
    savefig(fig, "fig_tiers")

    # ----------------------------------------------------------------- summary
    lines = ["# Results summary (auto-generated by analyse.py)", ""]
    mi = json.loads((RESULTS / "machine.json").read_text())
    lines.append(f"Machine: {mi.get('cpu')}, {mi.get('cores')} cores, {mi.get('ram_gb')} GB RAM, {mi.get('os')}, "
                 f"Python {mi.get('python')}, NumPy {mi.get('numpy')}, SciPy {mi.get('scipy')}")
    lines.append(f"FE budgets: {mi.get('budgets')}")
    lines.append("")
    for name in ("table1_linear", "table2_discount", "table3_fe"):
        lines += [f"## {name}", "", (RESULTS / f"{name}.md").read_text(), ""]
    lines += ["## Statistical tests (two-sided Wilcoxon, Holm within instance x metric)", "",
              stats[["instance", "metric", "comparison", "ties", "p", "p_holm", "r_rb", "A12", "effect",
                     "verdict"]].round(4).to_markdown(index=False), ""]
    (RESULTS / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
