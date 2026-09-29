"""Tab renderers for the dashboard."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from algorithms.de import ACTIONS
from algorithms.registry import DE_METHODS, METHODS, applicable, run
from core.decoder import decode
from core.problem import Problem
from ui.common import (COLORS, convergence_figure, fmt, load_csv, load_json, matrix_df, pid, plan_table,
                       problem_table, reference_of, show_plan, tier_figure)

S = st.session_state
STATE_NAMES = ["0 productive", "1 slowing", "2 stagnant"]
ACTION_NAMES = [f"a{k} F={f} CR={c}" for k, (f, c) in enumerate(ACTIONS)]


# ======================================================================= Problem
def render_problem(prob: Problem, is_custom: bool):
    if not is_custom:
        st.subheader(f"{prob.name}: {prob.meta.get('description', '')}")
    feats = prob.features()
    st.markdown("**Type:** " + ("Standard transportation problem" if prob.is_standard and prob.is_balanced
                                 else "Standard (unbalanced)" if prob.is_standard
                                 else "Non-standard: " + ", ".join(feats)))
    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown("**Unit costs, supplies and demands**")
        st.dataframe(problem_table(prob), width="stretch")
    with c2:
        st.metric("Total supply", fmt(prob.supply.sum()))
        st.metric("Total demand", fmt(prob.demand.sum()))
        ref = reference_of(prob)
        if ref is not None:
            st.metric("Exact optimum", fmt(ref), help=f"computed by {prob.meta.get('reference_method', '').upper()}")
    if not prob.is_balanced:
        side = "destination" if prob.supply.sum() > prob.demand.sum() else "source"
        st.info(f"Unbalanced: a zero-cost dummy {side} of {abs(int(prob.supply.sum() - prob.demand.sum()))} "
                "units is added automatically by every solver.")
    if prob.meta.get("notes"):
        st.caption("Notes: " + prob.meta["notes"])
    if prob.discount is not None:
        c1, c2 = st.columns([1, 2])
        with c1:
            t = prob.discount["tiers"]
            lo = 0
            rows = []
            for up, r in t:
                if prob.discount["type"] == "incremental":
                    rng = f"units {lo + 1}-{up}" if up is not None else f"units above {lo}"
                else:
                    rng = f"whole shipment if x <= {up}" if up is not None else f"whole shipment if x > {lo}"
                rows.append({"Tier": rng, "Rate": f"{r:g} x c"})
                lo = up if up is not None else lo
            st.markdown(f"**{prob.discount['type'].replace('_', '-').title()} discount**")
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        with c2:
            st.plotly_chart(tier_figure(prob), width="stretch")
    extras = [("Fixed charge per used route", prob.fixed_charge),
              ("Route capacity (inf = unlimited)", prob.capacity),
              ("Minimum shipment", prob.min_ship),
              ("Forbidden routes (1 = forbidden)", None if prob.forbidden is None else prob.forbidden.astype(int))]
    if prob.objective2 is not None:
        extras.append((f"Second objective: {prob.objective2['name']} (weight {prob.objective2['weight']:g})",
                       prob.objective2["matrix"]))
    if prob.noise is not None:
        extras.append((f"Cost standard deviation (risk aversion k = {prob.noise['k']:g})", prob.noise["sd"]))
    for title, mat in extras:
        if mat is not None:
            with st.expander(title, expanded=True):
                st.dataframe(matrix_df(prob, mat), width="stretch")
    st.markdown("**What each method can do here**")
    st.dataframe(pd.DataFrame([{"Method": METHODS[m], "Applicable": "yes" if applicable(m, prob)[0] else "no",
                                "Why not": applicable(m, prob)[1]} for m in METHODS]),
                 hide_index=True, width="stretch")


# ========================================================================= Solve
def _modi_stepper(prob: Problem, trace, key):
    if not trace:
        return
    bal = prob.balanced()
    k = st.slider("MODI iteration", 0, len(trace) - 1, len(trace) - 1, key=key) if len(trace) > 1 else 0
    t = trace[k]
    basis = set(map(tuple, t["basis"]))
    X = t["X"]
    labels_r, labels_c = bal.row_labels, bal.col_labels
    disp = pd.DataFrame("", index=labels_r, columns=labels_c)
    for i in range(bal.m):
        for j in range(bal.n):
            if (i, j) in basis:
                disp.iat[i, j] = str(int(X[i, j])) if X[i, j] > 0 else "ε"
            else:
                disp.iat[i, j] = f"Δ={t['delta'][i, j]:+g}"
    disp["u"] = [f"{v:g}" for v in t["u"]]
    disp.loc["v"] = [f"{v:g}" for v in t["v"]] + [""]

    def sty(_):
        s = pd.DataFrame("", index=disp.index, columns=disp.columns)
        for i in range(bal.m):
            for j in range(bal.n):
                if (i, j) in basis:
                    s.iat[i, j] = "background-color: rgba(42,120,214,0.25); font-weight:600"
                elif t["delta"][i, j] < -1e-9:
                    s.iat[i, j] = "color: #c0392b; font-weight:600"
        if "entering" in t:
            s.iat[t["entering"][0], t["entering"][1]] += "; outline: 2px solid #d6552a"
        return s

    st.markdown(f"Iteration {k}: cost = **{fmt(t['cost'])}**. Blue = basic cells (ε = degenerate zero), "
                "Δ = c - u - v for the others.")
    st.dataframe(disp.style.apply(sty, axis=None), width="stretch")
    if "entering" in t:
        name = lambda c: f"({labels_r[c[0]]},{labels_c[c[1]]})"  # noqa: E731
        loop = " → ".join(f"{name(c)}{'+' if i % 2 == 0 else '−'}" for i, c in enumerate(t["loop"]))
        st.markdown(f"Most negative Δ: {name(t['entering'])} enters (Δ = {t['delta'][t['entering']]:+g}). "
                    f"Loop: {loop}. θ = {t['theta']}, {name(t['leaving'])} leaves.")
    else:
        st.success("All Δ ≥ 0, so this plan is optimal.")


def render_result(prob: Problem, res, key):
    if res.status == "not_applicable":
        st.warning(f"{METHODS[res.method]} is not applicable: {res.message}")
        return
    if res.status in ("error", "infeasible") or res.matrix is None:
        st.error(f"{METHODS[res.method]}: {res.status}. {res.message.splitlines()[0] if res.message else ''}")
        if res.status == "error":
            with st.expander("details"):
                st.code(res.message)
        return
    ref = reference_of(prob)
    c = st.columns(5)
    c[0].metric("Objective", fmt(res.objective, 4 if prob.objective2 is not None else 2))
    c[1].metric("Transport cost", fmt(res.cost))
    if ref is not None and prob.objective2 is None:
        c[2].metric("Gap to exact optimum", f"{100 * (res.objective - ref) / abs(ref):.3f}%")
    elif "obj2" in res.components:
        c[2].metric(prob.objective2["name"], fmt(res.components["obj2"]))
    c[3].metric("Time", f"{res.time_s:.3f} s")
    c[4].metric("Fitness evaluations", fmt(res.fe_used) if res.fe_used else "-")
    if "robust_cost" in res.components:
        st.caption(f"Expected cost {fmt(res.cost)} ± sd {fmt(res.components['cost_sd'])} → robust objective "
                   f"{fmt(res.components['robust_cost'])} (k = {prob.noise['k']:g})")
    if res.status == "time_limit":
        st.warning(f"MILP stopped at its time limit: best plan shown, MIP gap {res.diagnostics.get('mip_gap')}")
    if not res.feasible:
        v = {k: v for k, v in res.diagnostics.get("violations", {}).items() if v > 1e-9}
        st.error(f"Plan violates constraints: {v}. (DE could not find a fully feasible plan - "
                 "try a larger budget or the polish option.)")
    else:
        st.success(f"Feasible plan. Status: {res.status}.")
    if res.diagnostics.get("dummy_units"):
        kind = "unused supply" if res.diagnostics["dummy"] == "col" else "unmet demand"
        st.info(f"{res.diagnostics['dummy_units']} units of {kind} (sent to the dummy).")
    if res.message:
        st.caption(res.message)
    st.markdown("**Shipment plan** (highlighted = routes used)")
    show_plan(prob, res.matrix)
    csv = plan_table(prob, res.matrix).to_csv().encode()
    st.download_button("Download plan (.csv)", csv, file_name=f"{prob.name}_{res.method}_plan.csv", key=f"dl_{key}")
    if res.history:
        xs, ys = zip(*res.history)
        fig = convergence_figure([(METHODS[res.method], COLORS.get(res.method), xs, ys)],
                                 reference=ref if prob.objective2 is None else None,
                                 title="Best objective so far")
        st.plotly_chart(fig, width="stretch", key=f"conv_{key}")
        d = res.diagnostics
        st.caption(f"Generations: {d.get('generations')}, mean neutral fraction (trial decoded to the same plan "
                   f"as its parent): {d.get('neutral_frac_mean', 0):.2f}, distinct plans in final population: "
                   f"{d.get('distinct_plans_final')}")
        if "polish_improvements" in d:
            st.caption(f"Local-search polish: {d['polish_improvements']} improving moves "
                       f"(objective {fmt(d['objective_before_polish'])} → {fmt(res.objective)}).")
    if res.method == "modi":
        st.markdown(f"**MODI details** (start: {res.diagnostics.get('start')}, "
                    f"{res.diagnostics.get('iterations')} iterations)")
        for n in res.diagnostics.get("notes", []):
            st.caption(n)
        _modi_stepper(prob, res.trace, key=f"modistep_{key}")
    if res.method == "milp":
        d = res.diagnostics
        st.caption(f"MILP: {d.get('n_vars')} variables ({d.get('n_binaries')} binary), {d.get('n_rows')} "
                   f"constraints; dual bound {d.get('bound')}, MIP gap {d.get('mip_gap')}")
    if res.method == "naive":
        st.caption(f"Cost of this plan under the simplified (linear) model: {fmt(res.diagnostics.get('linear_cost'))}")
    if res.method == "hybrid" and res.actions:
        _hybrid_live(res, key)


def render_solve(prob: Problem, method: str):
    key = pid(prob)
    res = S.results.get((key, method))
    if res is None:
        st.info(f"Press **RUN** in the sidebar to solve with {METHODS[method]}.")
    else:
        render_result(prob, res, key=f"{key}_{method}")
    runs = [(m, r) for (k, m), r in S.results.items() if k == key and r.objective is not None]
    if len(runs) > 1:
        st.markdown("**Runs on this problem in this session**")
        st.dataframe(pd.DataFrame([{"Method": METHODS[m], "Objective": r.objective, "Cost": r.cost,
                                    "Feasible": r.feasible, "Time (s)": round(r.time_s, 3), "Status": r.status}
                                   for m, r in runs]), hide_index=True, width="stretch")


# ================================================================ Hybrid (live)
def _hybrid_live(res, key):
    st.markdown("**Hybrid controller (this run)**")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.imshow(res.q_table, x=[f"a{k}" for k in range(9)], y=STATE_NAMES, color_continuous_scale="Magma",
                        aspect="auto", title="Final Q-table (compare within a row)")
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, width="stretch", key=f"q_{key}")
    with c2:
        df = pd.DataFrame({"generation": range(len(res.actions)), "action": res.actions, "state": res.states})
        df["phase"] = pd.cut(df["generation"], bins=min(10, len(df)), labels=False)
        f = df.groupby(["phase", "action"]).size().unstack(fill_value=0)
        f = f.div(f.sum(1), axis=0)
        fig = go.Figure()
        for a in f.columns:
            fig.add_trace(go.Scatter(x=f.index, y=f[a], stackgroup="one", name=ACTION_NAMES[a]))
        fig.update_layout(title="Action usage over the run (10 phases)", height=300,
                          xaxis_title="phase of run", yaxis_title="share", margin=dict(l=10, r=10, t=50, b=10),
                          showlegend=False)
        st.plotly_chart(fig, width="stretch", key=f"a_{key}")
    share = pd.Series(res.states).value_counts(normalize=True).reindex([0, 1, 2], fill_value=0)
    st.caption("Time in each state: " + ", ".join(f"{STATE_NAMES[s]} {100 * share[s]:.0f}%" for s in range(3)))


def render_hybrid(prob: Problem, is_custom: bool):
    st.markdown("The Q-learning agent watches the **success rate** (share of trial vectors that beat their parent) "
                "and picks one of 9 (F, CR) actions each generation.")
    acts = pd.DataFrame({f"a{k}": [f"{f:g}", f"{c:g}"] for k, (f, c) in enumerate(ACTIONS)}, index=["F", "CR"])
    st.dataframe(acts, width="stretch")
    res = S.results.get((pid(prob), "hybrid"))
    if res is not None and res.actions:
        _hybrid_live(res, "tabh")
    else:
        st.info("Run **Hybrid DE + Q-Learning** on this problem to see its live Q-table here.")
    if is_custom:
        return
    log, q = load_csv("hybrid_log.csv"), load_csv("qtables.csv")
    if log is None or q is None:
        return
    qi = q[(q["study"] == "main") & (q["instance"] == prob.name)]
    if qi.empty:
        return
    st.markdown(f"**Study results for {prob.name} (30 runs, pre-computed)**")
    c1, c2 = st.columns(2)
    with c1:
        piv = qi.groupby(["state", "action"])["q"].mean().unstack()
        fig = px.imshow(piv.values, x=[f"a{k}" for k in piv.columns], y=STATE_NAMES, aspect="auto",
                        color_continuous_scale="Magma", title="Mean final Q-table over 30 runs")
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, width="stretch")
    with c2:
        li = log[(log["study"] == "main") & (log["instance"] == prob.name)].copy()
        li["phase"] = pd.cut(li["generation"], bins=10, labels=False)
        f = li.groupby(["phase", "action"]).size().unstack(fill_value=0)
        f = f.div(f.sum(1), axis=0)
        fig = go.Figure()
        for a in f.columns:
            fig.add_trace(go.Scatter(x=f.index, y=f[a], stackgroup="one", name=ACTION_NAMES[a]))
        fig.update_layout(title="Action usage over the run (all 30 runs)", height=300, showlegend=False,
                          xaxis_title="phase of run", margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, width="stretch")
    share = li["state"].value_counts(normalize=True).reindex([0, 1, 2], fill_value=0)
    st.caption("Time in each state: " + ", ".join(f"{STATE_NAMES[s]} {100 * share[s]:.0f}%" for s in range(3)))
    pc = load_json("pilot_choice.json")
    if pc:
        st.caption(f"State thresholds from the pilot study: productive if SR > {pc['sr_hi']}, "
                   f"stagnant if SR ≤ {pc['sr_lo']} ({pc['rule']}).")


# ======================================================================= Compare
def render_compare(prob: Problem, is_custom: bool):
    runs, exact = load_csv("runs.csv"), load_csv("exact.csv")
    have_study = (not is_custom and runs is not None and
                  ((runs["study"] == "main") & (runs["instance"] == prob.name)).any())
    if have_study:
        _study_compare(prob, runs, exact)
        st.divider()
    _quick_compare(prob)
    _study_report()


def _study_report():
    from ui.common import RESULTS
    if not (RESULTS / "summary.md").exists():
        return
    st.divider()
    with st.expander("📑 Full study report (all instances, pre-computed)"):
        mi = load_json("machine.json") or {}
        st.caption(f"Run on {mi.get('cpu')} ({mi.get('cores')} cores), Python {mi.get('python')}, {mi.get('date')}. "
                   f"FE budgets: {mi.get('budgets')}")
        for name, title in (("table1_linear", "Linear instances"), ("table2_discount", "Discount instances"),
                            ("table3_fe", "Fitness evaluations needed to reach the optimum"),
                            ("table4_sens_de", "DE sensitivity (F x CR) on N10, 10 seeds"),
                            ("table5_sens_hybrid", "Hybrid sensitivity on N10, 10 seeds")):
            df = load_csv(f"{name}.csv")
            if df is not None:
                st.markdown(f"**{title}**")
                st.dataframe(df, hide_index=True, width="stretch")
        figs = sorted((RESULTS / "figures").glob("*.png"))
        if figs:
            pick = st.selectbox("Figure", [f.name for f in figs], key="rep_fig")
            st.image(str(RESULTS / "figures" / pick))


def _study_compare(prob, runs, exact):
    name = prob.name
    st.markdown(f"### Pre-computed study: {name} (30 paired seeds per method)")
    d = runs[(runs["study"] == "main") & (runs["instance"] == name)]
    ref = d["reference"].iloc[0]
    rows = []
    e = exact[exact["instance"] == name]
    for _, r in e.iterrows():
        rows.append({"Method": METHODS[r["method"]], "Best": r["cost"], "Mean": r["cost"], "Sd": 0.0,
                     "Mean gap %": 100 * (r["cost"] - ref) / abs(ref), "Success": "exact" if r["method"] != "naive"
                     else ("yes" if abs(r["cost"] - ref) < 1e-6 else "no"),
                     "FE to optimum (mean)": "-", "Time (s)": r["time_mean_s"]})
    for m in DE_METHODS:
        x = d[d["method"] == m]
        ok = x["fe_to_opt"].dropna()
        rows.append({"Method": METHODS[m], "Best": x["cost"].min(), "Mean": x["cost"].mean(), "Sd": x["cost"].std(),
                     "Mean gap %": x["gap_pct"].mean(), "Success": f"{int(x['success'].sum())}/{len(x)}",
                     "FE to optimum (mean)": f"{ok.mean():.0f}" if len(ok) else "-",
                     "Time (s)": x["time_s"].mean()})
    st.dataframe(pd.DataFrame(rows).round(4), hide_index=True, width="stretch")
    hist = load_csv("history.csv")
    if hist is not None:
        h = hist[(hist["study"] == "main") & (hist["instance"] == name)]
        st.plotly_chart(_gap_fig(h, ref), width="stretch")
    stats = load_csv("stats.csv")
    if stats is not None:
        s = stats[stats["instance"] == name]
        if len(s):
            st.markdown("**Statistical tests** (two-sided Wilcoxon signed-rank on paired seeds, Holm-corrected; "
                        "r_rb < 0 means Hybrid is lower/better)")
            st.dataframe(s[["metric", "comparison", "ties", "p", "p_holm", "r_rb", "A12", "effect", "verdict"]]
                         .round(4), hide_index=True, width="stretch")


def _rgba(hex_color, a):
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


def _gap_fig(h, ref):
    fig = go.Figure()
    for m in DE_METHODS:
        hm = h[h["method"] == m].assign(gap=lambda z: 100 * (z["best_cost"] - ref) / abs(ref))
        if hm.empty:
            continue
        q = hm.groupby("fe")["gap"].quantile([0.25, 0.5, 0.75]).unstack()
        col = COLORS[m]
        fig.add_trace(go.Scatter(x=q.index, y=q[0.75], line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=q.index, y=q[0.25], fill="tonexty", line=dict(width=0), showlegend=False,
                                 fillcolor=_rgba(col, 0.15), hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=q.index, y=q[0.5], name=METHODS[m], line=dict(color=col, width=2)))
    fig.update_layout(title="Median gap to the exact optimum (%) with interquartile band, 30 runs",
                      xaxis_title="Fitness evaluations", yaxis_title="gap %", height=380,
                      margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.2))
    return fig


def _quick_compare(prob: Problem):
    st.markdown("### Quick comparison on this problem (runs now)")
    ms = [m for m in METHODS if applicable(m, prob)[0]]
    c1, c2, c3 = st.columns([3, 1, 1])
    chosen = c1.multiselect("Methods", ms, default=ms, format_func=lambda m: METHODS[m], key="qc_methods")
    seeds = c2.number_input("Seeds per DE method", 1, 30, 5, key="qc_seeds")
    budget = c3.number_input("FE budget", 1000, 200000, 10000 if prob.m * prob.n <= 30 else 30000, step=1000,
                             key="qc_budget")
    if st.button("Run comparison", key="qc_go", type="primary"):
        rows, curves = [], []
        bar = st.progress(0.0)
        todo = [(m, s) for m in chosen for s in (range(int(seeds)) if m in DE_METHODS else [None])]
        for k, (m, s) in enumerate(todo):
            r = run(m, prob, {"fe_budget": int(budget), "polish": not prob.is_standard}, seed=s)
            rows.append({"method": m, "seed": s, "objective": r.objective, "cost": r.cost, "feasible": r.feasible,
                         "time": r.time_s, "status": r.status})
            if r.history:
                curves.append((m, r.history))
            bar.progress((k + 1) / len(todo))
        S.quick[pid(prob)] = (rows, curves)
    out = S.quick.get(pid(prob))
    if not out:
        return
    rows, curves = out
    df = pd.DataFrame(rows)
    ok = df[df["objective"].notna()]
    if ok.empty:
        st.error("No method produced a plan.")
        return
    best = ok[ok["feasible"]]["objective"].min() if ok["feasible"].any() else ok["objective"].min()
    g = ok.groupby("method").agg(best=("objective", "min"), mean=("objective", "mean"), sd=("objective", "std"),
                                 feasible=("feasible", "mean"), time=("time", "mean"), runs=("objective", "size"))
    g["gap to best found %"] = 100 * (g["mean"] - best) / abs(best) if best else np.nan
    g.index = [METHODS[m] for m in g.index]
    st.dataframe(g.round(4), width="stretch")
    if curves:
        series = []
        for m in DE_METHODS:
            hs = [h for mm, h in curves if mm == m]
            if not hs:
                continue
            L = min(len(h) for h in hs)
            ys = np.median(np.array([[b for _, b in h[:L]] for h in hs]), axis=0)
            xs = [fe for fe, _ in hs[0][:L]]
            series.append((METHODS[m] + " (median)", COLORS[m], xs, ys))
        st.plotly_chart(convergence_figure(series, reference=best, title="Median best objective so far"),
                        width="stretch")


# ======================================================================= Decoder
def render_decoder(prob: Problem):
    bal = prob.balanced()
    D = bal.m * bal.n
    st.markdown("DE does not search over shipment amounts. Each candidate is a list of **priority keys**, one per "
                "cell. The decoder fills cells from the highest key down, each time shipping "
                "`min(remaining supply, remaining demand, capacity)`. That always gives a valid integer plan.")
    c1, c2, c3 = st.columns([1, 1, 2])
    src = c1.radio("Keys", ["random", "best keys from last DE run", "type my own"], key="dec_src")
    seed = c2.number_input("seed", 0, 10_000, 0, key="dec_seed")
    keys = np.random.default_rng(int(seed)).random(D)
    if src == "best keys from last DE run":
        cands = [r for (k, m), r in S.results.items() if k == pid(prob) and m in DE_METHODS and
                 r.diagnostics.get("keys") is not None]
        if cands:
            keys = np.asarray(cands[-1].diagnostics["keys"])
        else:
            st.info("No DE run on this problem yet - showing random keys.")
    elif src == "type my own":
        txt = c3.text_input(f"{D} numbers separated by commas (cells row by row)",
                            ", ".join(f"{v:.2f}" for v in keys), key="dec_txt")
        try:
            vals = [float(v) for v in txt.split(",") if v.strip()]
            if len(vals) == D:
                keys = np.array(vals)
            else:
                st.warning(f"need exactly {D} numbers")
        except ValueError:
            st.warning("could not read the numbers")
    X, unmet, steps = decode(keys, bal, return_steps=True)
    names = [f"({bal.row_labels[k // bal.n]},{bal.col_labels[k % bal.n]})" for k in range(D)]
    kmat = pd.DataFrame(keys.reshape(bal.m, bal.n).round(3), index=bal.row_labels, columns=bal.col_labels)
    st.markdown("**Keys per cell**")
    st.dataframe(kmat, width="stretch")
    upto = st.slider("Show the first k steps", 1, D, D, key="dec_k")
    rows = []
    Xk = np.zeros((bal.m, bal.n), dtype=int)
    for n_, (k, q, s, d, note) in enumerate(steps[:upto], 1):
        Xk[k // bal.n, k % bal.n] = q
        rows.append({"Step": n_, "Cell": names[k], "Key": round(float(keys[k]), 3), "Ship": q,
                     "Supply left": str(s), "Demand left": str(d), "Note": note})
    c1, c2 = st.columns([3, 2])
    c1.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=min(420, 38 + 35 * len(rows)))
    with c2:
        st.markdown(f"**Plan after {upto} steps**")
        st.dataframe(pd.DataFrame(Xk, index=bal.row_labels, columns=bal.col_labels), width="stretch")
        if upto == D:
            ev = prob.evaluate(bal.strip_dummy(X))
            st.metric("Objective of this plan", fmt(ev["objective"], 4 if prob.objective2 is not None else 2))
            if unmet:
                st.error(f"{unmet} units could not be shipped (route constraints) - DE adds a penalty.")


# ================================================================ Check a plan
def render_plan_check(prob: Problem):
    st.markdown("Enter any shipment plan to check feasibility and cost. For standard problems, MODI can start "
                "from your plan and show every improvement step (as in Lab Activity 5).")
    key = pid(prob)
    if S.get("pc_key") != key:
        S.pc_key = key
        S.pc_ver = S.get("pc_ver", 0) + 1
        start = prob.meta.get("proposed_plan")
        S.pc_base = pd.DataFrame(np.array(start) if start is not None else np.zeros((prob.m, prob.n), dtype=int),
                                 index=prob.row_labels, columns=prob.col_labels)
    c1, c2, c3 = st.columns(3)
    if c1.button("Load the last solver plan", key="pc_last"):
        rs = [r for (k, m), r in S.results.items() if k == key and r.matrix is not None]
        if rs:
            S.pc_base = pd.DataFrame(rs[-1].matrix, index=prob.row_labels, columns=prob.col_labels)
            S.pc_ver += 1
    if c2.button("Clear", key="pc_clear"):
        S.pc_base = pd.DataFrame(np.zeros((prob.m, prob.n), dtype=int), index=prob.row_labels,
                                 columns=prob.col_labels)
        S.pc_ver += 1
    if prob.meta.get("proposed_plan") is not None and c3.button("Load the lab's proposed plan", key="pc_prop"):
        S.pc_base = pd.DataFrame(np.array(prob.meta["proposed_plan"]), index=prob.row_labels,
                                 columns=prob.col_labels)
        S.pc_ver += 1
    df = st.data_editor(S.pc_base, key=f"pc_ed_{S.pc_ver}", width="stretch")
    try:
        X = df.fillna(0).to_numpy(dtype=float)
    except ValueError:
        st.error("All cells must be numbers.")
        return
    if not X.any():
        st.info("Type a plan into the table (or load one with the buttons above).")
        return
    ev = prob.evaluate(X)
    c = st.columns(3)
    c[0].metric("Objective", fmt(ev["objective"], 4 if prob.objective2 is not None else 2))
    c[1].metric("Transport cost", fmt(ev["cost"]))
    ref = reference_of(prob)
    if ref is not None and prob.objective2 is None:
        c[2].metric("Gap to optimum", f"{100 * (ev['objective'] - ref) / abs(ref):.2f}%")
    bad = {k: v for k, v in ev["violations"].items() if v > 1e-9}
    if bad:
        st.error("Not feasible: " + ", ".join(f"{k} off by {v:g}" for k, v in bad.items()))
        show_plan(prob, X)
        return
    st.success("Feasible plan.")
    if np.any(np.abs(X - np.round(X)) > 1e-9):
        st.warning("Plan has fractional shipments.")
        return
    if prob.is_standard:
        if st.button("Test optimality with MODI from this plan", type="primary", key="pc_modi"):
            S.pc_modi = (key, run("modi", prob, {"initial": X.astype(int), "trace": True}))
        r = S.get("pc_modi")
        if r and r[0] == key:
            res = r[1]
            if res.status == "error":
                st.error(res.message)
            else:
                t0 = res.trace[0]["cost"]
                st.markdown(f"Start cost **{fmt(t0)}** → optimum **{fmt(res.cost)}** in "
                            f"{res.diagnostics['iterations']} MODI iterations.")
                for n in res.diagnostics.get("notes", []):
                    st.caption(n)
                _modi_stepper(prob, res.trace, key="pc_step")
    else:
        st.caption("MODI optimality test needs a standard problem. Use MILP on the Solve tab for the exact optimum.")


# =============================================================== Pareto & Risk
def _variant(prob, **changes):
    d = prob.to_dict()
    d.update(changes)
    return Problem.from_dict(d)


def render_pareto_risk(prob: Problem):
    if prob.objective2 is None and prob.noise is None:
        st.info("This tab is for problems with a **second objective** or **uncertain costs**. Try examples X3 / X4, "
                "or add these features in the custom builder.")
        return
    if prob.objective2 is not None:
        st.markdown(f"### Trade-off: cost vs {prob.objective2['name']}")
        c1, c2, c3 = st.columns(3)
        npts = c1.slider("Number of weights", 3, 21, 11, key="par_n")
        ok_milp = applicable("milp", prob)[0]
        how = c2.selectbox("Solver per weight", (["milp"] if ok_milp else []) + ["hybrid"],
                           format_func=lambda m: METHODS[m], key="par_method")
        budget = c3.number_input("FE budget (DE)", 1000, 100000, 10000, step=1000, key="par_budget")
        if st.button("Draw Pareto front", type="primary", key="par_go"):
            pts = []
            bar = st.progress(0.0)
            for k, w in enumerate(np.linspace(0, 1, npts)):
                o2 = dict(prob.to_dict()["objective2"], weight=float(w))
                pv = _variant(prob, objective2=o2)
                pv._norms = prob.norms()
                r = run(how, pv, {"fe_budget": int(budget), "polish": True}, seed=k)
                if r.matrix is not None:
                    pts.append({"w": w, "cost": r.cost, "obj2": r.components["obj2"], "feasible": r.feasible,
                                "plan": r.matrix})
                bar.progress((k + 1) / npts)
            S.pareto[pid(prob)] = pts
        pts = S.pareto.get(pid(prob))
        if pts:
            df = pd.DataFrame([{k: v for k, v in p.items() if k != "plan"} for p in pts])
            nd = []
            for i, a in df.iterrows():
                dom = ((df["cost"] <= a["cost"]) & (df["obj2"] <= a["obj2"]) &
                       ((df["cost"] < a["cost"]) | (df["obj2"] < a["obj2"]))).any()
                nd.append(not dom)
            df["non-dominated"] = nd
            front = df[df["non-dominated"]].sort_values("cost")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=front["cost"], y=front["obj2"], mode="lines+markers", name="Pareto front",
                                     line=dict(color="#d6552a"), text=[f"w={w:.2f}" for w in front["w"]]))
            fig.add_trace(go.Scatter(x=df["cost"], y=df["obj2"], mode="markers", name="all weights",
                                     marker=dict(color="gray", size=6), text=[f"w={w:.2f}" for w in df["w"]]))
            fig.update_layout(xaxis_title="Transport cost", yaxis_title=prob.objective2["name"], height=400,
                              margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, width="stretch")
            st.dataframe(df.round(3), hide_index=True, width="stretch")
            pick = st.selectbox("Show the plan for weight", list(range(len(pts))),
                                format_func=lambda i: f"w = {pts[i]['w']:.2f}", key="par_pick")
            show_plan(prob, pts[pick]["plan"])
    if prob.noise is not None:
        st.markdown("### Risk: how uncertain costs affect each plan")
        st.caption("Unit costs are sampled from Normal(mean, sd) truncated at 0; each plan's total cost is "
                   "recomputed for every sample.")
        c1, c2 = st.columns(2)
        nsim = c1.number_input("Simulations", 200, 20000, 2000, step=200, key="risk_n")
        budget = c2.number_input("FE budget (DE)", 1000, 100000, 15000, step=1000, key="risk_budget")
        if st.button("Compare plans under uncertainty", type="primary", key="risk_go"):
            nz = prob.to_dict()["noise"]
            mean_prob = _variant(prob, noise=None)
            m_method = "milp" if applicable("milp", mean_prob)[0] else "hybrid"
            plans = {"Mean-cost plan (ignores risk)": run(m_method, mean_prob, {"fe_budget": int(budget)}, seed=0)}
            plans[f"Robust plan (k = {prob.noise['k']:g})"] = run("hybrid", prob,
                                                                 {"fe_budget": int(budget), "polish": True}, seed=0)
            hi_k = max(2 * prob.noise["k"], 2.0)
            plans[f"Very cautious plan (k = {hi_k:g})"] = run("hybrid", _variant(prob, noise=dict(nz, k=hi_k)),
                                                             {"fe_budget": int(budget), "polish": True}, seed=0)
            rng = np.random.default_rng(0)
            samples = np.clip(rng.normal(prob.cost, prob.noise["sd"], size=(int(nsim),) + prob.cost.shape), 0, None)
            out = {}
            for name, r in plans.items():
                if r.matrix is None:
                    continue
                h = prob.effective_units(r.matrix)
                tot = (samples * h).sum(axis=(1, 2))
                if prob.fixed_charge is not None:
                    tot += prob.fixed_charge[r.matrix > 0].sum()
                out[name] = (r.matrix, tot)
            S.risk[pid(prob)] = out
        out = S.risk.get(pid(prob))
        if out:
            fig = go.Figure()
            rows = []
            for (name, (X, tot)), col in zip(out.items(), ["#8a8a8a", "#2a78d6", "#d6552a"]):
                fig.add_trace(go.Histogram(x=tot, name=name, opacity=0.5, nbinsx=60, marker_color=col))
                rows.append({"Plan": name, "Mean cost": tot.mean(), "Sd": tot.std(), "5th pct": np.percentile(tot, 5),
                             "95th pct": np.percentile(tot, 95), "Worst": tot.max()})
            fig.update_layout(barmode="overlay", xaxis_title="Realised total cost", yaxis_title="count", height=380,
                              margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(fig, width="stretch")
            st.dataframe(pd.DataFrame(rows).round(1), hide_index=True, width="stretch")
            pick = st.selectbox("Show plan", list(out.keys()), key="risk_pick")
            show_plan(prob, out[pick][0])
