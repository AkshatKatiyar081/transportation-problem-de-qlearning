"""Tab renderers for the dashboard: Problem, Solve, and DE vs DE-Random vs Hybrid."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from algorithms.de import ACTIONS
from algorithms.registry import DE_METHODS, METHODS, applicable, run
from core.decoder import decode
from core.problem import Problem
from experiments.stats import compare, holm
from ui.common import (COLORS, convergence_figure, fmt, load_csv, matrix_df, pid, plan_table, problem_table,
                       reference_of, show_plan, tier_figure)

S = st.session_state
SHORT = {"de": "Standard DE", "de_random": "DE-Random", "hybrid": "Hybrid DE + QL"}


def _ref_label(prob):
    return "Optimum (MODI)" if prob.meta.get("reference_status") == "optimal" else "Best-known value"


# ======================================================================= Problem
def render_problem(prob: Problem, is_custom: bool):
    if not is_custom:
        st.subheader(prob.meta.get("description", prob.name))
    kind = ("Standard transportation problem" if prob.is_standard else
            "Non-standard: " + ", ".join(prob.features()))
    if prob.is_standard and not prob.is_balanced:
        kind += " (unbalanced - a dummy row/column is added automatically)"
    st.markdown(f"**Type:** {kind}")
    c1, c2 = st.columns([4, 1])
    with c1:
        st.dataframe(problem_table(prob), width="stretch")
    with c2:
        ref = reference_of(prob)
        if ref is not None:
            st.metric(_ref_label(prob), fmt(ref, 4 if prob.objective2 else 2))
        ok = [METHODS[m] for m in METHODS if applicable(m, prob)[0]]
        st.caption("Methods that apply: " + ", ".join(ok))
    if prob.meta.get("notes"):
        st.caption(prob.meta["notes"])
    if prob.discount is not None:
        c1, c2 = st.columns([1, 2])
        rows, lo = [], 0
        for up, r in prob.discount["tiers"]:
            if prob.discount["type"] == "incremental":
                rng = f"units {lo + 1}-{up}" if up is not None else f"units above {lo}"
            else:
                rng = f"whole shipment if x <= {up}" if up is not None else f"whole shipment if x > {lo}"
            rows.append({"Tier": rng, "Rate": f"{r:g} x base cost"})
            lo = up if up is not None else lo
        c1.markdown(f"**{prob.discount['type'].replace('_', '-').title()} discount**")
        c1.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        c2.plotly_chart(tier_figure(prob), width="stretch")
    extras = [("Fixed charge per used route", prob.fixed_charge),
              ("Route capacity (inf = unlimited)", prob.capacity),
              ("Minimum shipment per used route", prob.min_ship),
              ("Forbidden routes (1 = forbidden)", None if prob.forbidden is None else prob.forbidden.astype(int))]
    if prob.objective2 is not None:
        extras.append((f"Second objective: {prob.objective2['name']} (weight {prob.objective2['weight']:g})",
                       prob.objective2["matrix"]))
    if prob.noise is not None:
        extras.append((f"Cost standard deviation (risk aversion k = {prob.noise['k']:g})", prob.noise["sd"]))
    extras = [(t, m) for t, m in extras if m is not None]
    if extras:
        with st.expander("Extra data: " + ", ".join(t.split(" (")[0].lower() for t, _ in extras)):
            for title, mat in extras:
                st.markdown(f"**{title}**")
                st.dataframe(matrix_df(prob, mat), width="stretch")


# ========================================================================= Solve
def _modi_stepper(prob: Problem, trace, key):
    bal = prob.balanced()
    k = st.slider("Iteration", 0, len(trace) - 1, len(trace) - 1, key=key) if len(trace) > 1 else 0
    t = trace[k]
    basis = set(map(tuple, t["basis"]))
    X = t["X"]
    disp = pd.DataFrame("", index=bal.row_labels, columns=bal.col_labels)
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
        return s

    st.markdown(f"Iteration {k}: cost = **{fmt(t['cost'])}**. Blue = allocated cells (ε = degenerate zero); "
                "other cells show Δ = c − u − v.")
    st.dataframe(disp.style.apply(sty, axis=None), width="stretch")
    if "entering" in t:
        nm = lambda c: f"({bal.row_labels[c[0]]},{bal.col_labels[c[1]]})"  # noqa: E731
        loop = " → ".join(f"{nm(c)}{'+' if i % 2 == 0 else '−'}" for i, c in enumerate(t["loop"]))
        st.markdown(f"Most negative Δ: **{nm(t['entering'])}** enters (Δ = {t['delta'][t['entering']]:+g}). "
                    f"Loop: {loop}. θ = {t['theta']}; {nm(t['leaving'])} leaves.")
    else:
        st.success("All Δ ≥ 0 → this plan is optimal.")


def render_solve(prob: Problem, method: str):
    key = pid(prob)
    res = S.results.get((key, method))
    if res is None:
        st.info(f"Press **RUN** in the sidebar to solve with {METHODS[method]}.")
    elif res.status == "not_applicable":
        st.warning(f"{METHODS[method]} is not applicable: {res.message}")
    elif res.matrix is None:
        st.error(f"{METHODS[method]}: {res.status}. {(res.message or '').splitlines()[0] if res.message else ''}")
    else:
        _show_result(prob, res, key)
    if prob.objective2 is not None:
        with st.expander(f"⚖️ Trade-off: cost vs {prob.objective2['name']} (Pareto front)"):
            _pareto(prob)
    if prob.noise is not None:
        with st.expander("🎲 Risk: how uncertain costs affect each plan"):
            _risk(prob)


def _show_result(prob, res, key):
    ref = reference_of(prob)
    c = st.columns(4)
    two = prob.objective2 is not None
    c[0].metric("Weighted objective" if two else "Robust cost" if prob.noise is not None else "Total cost",
                fmt(res.objective, 4 if two else 2))
    if two:
        c[1].metric(f"Cost / {prob.objective2['name']}", f"{fmt(res.cost)} / {fmt(res.components['obj2'])}")
    elif ref is not None:
        c[1].metric(f"Gap to {_ref_label(prob).lower()}", f"{100 * (res.objective - ref) / abs(ref):.2f}%")
    c[2].metric("Time", f"{res.time_s:.2f} s")
    if res.fe_used:
        c[3].metric("Fitness evaluations", fmt(res.fe_used))
    if "robust_cost" in res.components:
        st.caption(f"Expected cost {fmt(res.cost)}, standard deviation {fmt(res.components['cost_sd'])} "
                   f"→ robust cost = expected + {prob.noise['k']:g} × sd.")
    if not res.feasible:
        v = {k: v for k, v in res.diagnostics.get("violations", {}).items() if v > 1e-9}
        st.error(f"The plan breaks some constraints {v}. Try a larger budget or tick 'polish' in the sidebar.")
    if res.diagnostics.get("dummy_units"):
        kind = "unused supply" if res.diagnostics["dummy"] == "col" else "unmet demand"
        st.info(f"{res.diagnostics['dummy_units']} units of {kind} go to the dummy.")
    if res.method == "naive":
        st.caption(res.message)
    st.markdown("**Shipment plan** (highlighted = routes used)")
    show_plan(prob, res.matrix)
    st.download_button("Download plan (.csv)", plan_table(prob, res.matrix).to_csv().encode(),
                       file_name=f"{prob.name}_{res.method}_plan.csv", key=f"dl_{key}")
    if res.history:
        xs, ys = zip(*res.history)
        st.plotly_chart(convergence_figure([(METHODS[res.method], COLORS.get(res.method), xs, ys)],
                                           reference=ref if not two else None, title="Best objective so far"),
                        width="stretch", key=f"conv_{key}")
    if res.method == "modi" and res.trace:
        with st.expander(f"🧮 MODI step by step (start: {res.diagnostics.get('start')}, "
                         f"{res.diagnostics.get('iterations')} improvements)", expanded=len(res.trace) > 1):
            for n in res.diagnostics.get("notes", []):
                st.caption(n)
            _modi_stepper(prob, res.trace, key=f"modi_{key}")
    if res.method == "hybrid" and res.actions:
        with st.expander("🤖 How the Hybrid chose F and CR during this run"):
            _action_chart(res.actions)


def _action_chart(actions, title="Share of generations using each (F, CR) setting"):
    df = pd.DataFrame({"g": range(len(actions)), "a": actions})
    df["phase"] = pd.cut(df["g"], bins=min(10, len(df)), labels=False)
    f = df.groupby(["phase", "a"]).size().unstack(fill_value=0)
    f = f.div(f.sum(axis=1), axis=0)
    fig = go.Figure()
    for a in f.columns:
        F, CR = ACTIONS[a]
        fig.add_trace(go.Scatter(x=(f.index + 1) * 10, y=f[a], stackgroup="one", name=f"F={F}, CR={CR}"))
    fig.update_layout(title=title, xaxis_title="% of the run", yaxis_title="share", height=320,
                      margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, width="stretch")
    st.caption("Early on the agent explores all 9 settings; later it mostly uses the ones that produced "
               "improvements (usually a low crossover rate for this encoding).")


# ================================================================ Pareto / Risk
def _variant(prob, **changes):
    d = prob.to_dict()
    d.update(changes)
    return Problem.from_dict(d)


def _pareto(prob: Problem):
    st.caption("The weight w is swept from 0 (cost only) to 1 (second objective only); each weight is "
               "solved with Hybrid DE. Plans that no other plan beats on both objectives form the Pareto front.")
    c1, c2 = st.columns(2)
    npts = c1.slider("Number of weights", 3, 21, 11, key="par_n")
    budget = c2.number_input("Budget per weight", 1000, 100000, 10000, step=1000, key="par_budget")
    if st.button("Draw Pareto front", key="par_go"):
        pts, bar = [], st.progress(0.0)
        for k, w in enumerate(np.linspace(0, 1, npts)):
            pv = _variant(prob, objective2=dict(prob.to_dict()["objective2"], weight=float(w)))
            r = run("hybrid", pv, {"fe_budget": int(budget), "polish": True}, seed=k)
            if r.matrix is not None:
                pts.append({"w": round(float(w), 2), "cost": r.cost, "obj2": r.components["obj2"], "plan": r.matrix})
            bar.progress((k + 1) / npts)
        S.pareto[pid(prob)] = pts
    pts = S.pareto.get(pid(prob))
    if not pts:
        return
    df = pd.DataFrame([{k: v for k, v in p.items() if k != "plan"} for p in pts])
    df["on front"] = [not ((df["cost"] <= a.cost) & (df["obj2"] <= a.obj2) &
                           ((df["cost"] < a.cost) | (df["obj2"] < a.obj2))).any() for a in df.itertuples()]
    front = df[df["on front"]].sort_values("cost")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=front["cost"], y=front["obj2"], mode="lines+markers", name="Pareto front",
                             line=dict(color="#d6552a"), text=[f"w={w}" for w in front["w"]]))
    fig.update_layout(xaxis_title="Transport cost", yaxis_title=prob.objective2["name"], height=360,
                      margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, width="stretch")
    pick = st.selectbox("Show the plan for weight", list(range(len(pts))),
                        format_func=lambda i: f"w = {pts[i]['w']}  (cost {fmt(pts[i]['cost'])}, "
                                              f"{prob.objective2['name']} {fmt(pts[i]['obj2'])})", key="par_pick")
    show_plan(prob, pts[pick]["plan"])


def _risk(prob: Problem):
    st.caption("Three plans are built: one that ignores risk (mean costs only), the robust plan for your k, and a "
               "very cautious plan. Then unit costs are simulated 2,000 times and each plan's real total cost "
               "is recorded.")
    budget = st.number_input("Budget per plan", 1000, 100000, 15000, step=1000, key="risk_budget")
    if st.button("Simulate", key="risk_go"):
        nz = prob.to_dict()["noise"]
        mean_prob = _variant(prob, noise=None)
        plans = {"Ignores risk": run("modi" if mean_prob.is_standard else "hybrid", mean_prob,
                                     {"fe_budget": int(budget), "polish": True}, seed=0),
                 f"Robust (k = {prob.noise['k']:g})": run("hybrid", prob, {"fe_budget": int(budget), "polish": True},
                                                          seed=0)}
        hi_k = max(2 * prob.noise["k"], 2.0)
        plans[f"Very cautious (k = {hi_k:g})"] = run("hybrid", _variant(prob, noise=dict(nz, k=hi_k)),
                                                    {"fe_budget": int(budget), "polish": True}, seed=0)
        rng = np.random.default_rng(0)
        samples = np.clip(rng.normal(prob.cost, prob.noise["sd"], size=(2000,) + prob.cost.shape), 0, None)
        out = {}
        for name, r in plans.items():
            if r.matrix is None:
                continue
            tot = (samples * prob.effective_units(r.matrix)).sum(axis=(1, 2))
            if prob.fixed_charge is not None:
                tot += prob.fixed_charge[r.matrix > 0].sum()
            out[name] = (r.matrix, tot)
        S.risk[pid(prob)] = out
    out = S.risk.get(pid(prob))
    if not out:
        return
    fig, rows = go.Figure(), []
    for (name, (X, tot)), col in zip(out.items(), ["#8a8a8a", "#2a78d6", "#d6552a"]):
        fig.add_trace(go.Histogram(x=tot, name=name, opacity=0.5, nbinsx=60, marker_color=col))
        rows.append({"Plan": name, "Average cost": round(tot.mean(), 1), "Spread (sd)": round(tot.std(), 1),
                     "Bad case (95th pct)": round(np.percentile(tot, 95), 1)})
    fig.update_layout(barmode="overlay", xaxis_title="Simulated total cost", yaxis_title="count", height=340,
                      margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig, width="stretch")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("A robust plan pays a little more on average but its bad cases are much less bad.")


# ======================================================== DE vs DE-Random vs Hybrid
def render_race(prob: Problem, is_custom: bool):
    st.markdown(
        "All three use the same DE engine and the same budget. **Only the choice of F (step size) and CR "
        "(crossover rate) differs:**\n"
        "- **Standard DE**: fixed F = 0.5, CR = 0.9 every generation.\n"
        "- **DE-Random**: a random (F, CR) from a 3×3 grid every generation, with no learning.\n"
        "- **Hybrid DE + QL**: a Q-learning agent picks (F, CR) from the same grid, learning from which "
        "settings produced improvements.")
    ref = reference_of(prob)
    runs = None if is_custom else load_csv("runs.csv")
    if runs is not None:
        d = runs[(runs["study"] == "showcase") & (runs["instance"] == prob.name)]
        if len(d):
            _showcase_result(prob, d, ref)
    if runs is None or not ((runs["study"] == "showcase") & (runs["instance"] == prob.name)).any():
        st.info("Tip: the **Showcase** problems S1 and S2 are designed so the three variants end clearly apart; "
                "their 30-run results are pre-computed.")
    st.markdown("### Run the race now")
    c1, c2 = st.columns(2)
    nrun = c1.number_input("Runs per method", 1, 10, 3, key="race_n")
    budget = c2.number_input("Evaluation budget per run", 1000, 100000,
                             int(prob.meta.get("fe_budget", 10000 if prob.m * prob.n <= 30 else 25000)),
                             step=1000, key=f"race_budget_{pid(prob)}")
    if st.button("🏁 Start race", type="primary", key="race_go"):
        out, bar = {m: [] for m in DE_METHODS}, st.progress(0.0, text="running ...")
        todo = [(m, s) for s in range(int(nrun)) for m in DE_METHODS]
        for k, (m, s) in enumerate(todo):
            r = run(m, prob, {"fe_budget": int(budget)}, seed=100 + s)
            out[m].append((r.objective, r.history, r.time_s))
            bar.progress((k + 1) / len(todo), text=f"{SHORT[m]}, run {s + 1}")
        bar.empty()
        S.race[pid(prob)] = out
    out = S.race.get(pid(prob))
    if out:
        best = min(v for m in out for v, _, _ in out[m])
        base = ref if ref is not None and ref <= best + 1e-9 else best
        rows = []
        series = []
        for m in DE_METHODS:
            vals = np.array([v for v, _, _ in out[m]])
            rows.append({"Method": SHORT[m], "Average final cost": round(vals.mean(), 2),
                         "Average gap %": round(100 * (vals.mean() - base) / abs(base), 2),
                         "Best run": round(vals.min(), 2), "Avg time (s)": round(np.mean([t for _, _, t in out[m]]), 2)})
            L = min(len(h) for _, h, _ in out[m])
            ys = np.median(np.array([[100 * (b - base) / abs(base) for _, b in h[:L]] for _, h, _ in out[m]]), axis=0)
            series.append((SHORT[m], COLORS[m], [fe for fe, _ in out[m][0][1][:L]], ys))
        fig = convergence_figure(series, title="Median gap during the run (%)", ylab="gap %")
        st.plotly_chart(fig, width="stretch")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption(f"Gap measured against {'the ' + _ref_label(prob).lower() if base == ref else 'the best cost found in this race'}"
                   f" ({fmt(base)}). With only a few runs, luck matters - the 30-run result above is the fair test.")
    with st.expander("🔢 How DE reads a solution (decoder explainer)"):
        _decoder(prob)


def _showcase_result(prob, d, ref):
    st.markdown(f"### Result over 30 runs (pre-computed, {int(d['fe_used'].max()):,} evaluations each)")
    c1, c2 = st.columns(2)
    box = go.Figure()
    for m in DE_METHODS:
        x = d[d["method"] == m]["gap_pct"]
        box.add_trace(go.Box(y=x, name=SHORT[m], marker_color=COLORS[m], boxpoints="all", jitter=0.4,
                             pointpos=0, boxmean=True))
    box.update_layout(title="Final gap to the best-known cost (%) - each dot is one run", yaxis_title="gap %",
                      height=380, showlegend=False, margin=dict(l=10, r=10, t=50, b=10))
    c1.plotly_chart(box, width="stretch")
    hist = load_csv("history.csv")
    if hist is not None:
        h = hist[(hist["study"] == "showcase") & (hist["instance"] == prob.name)]
        fig = go.Figure()
        for m in DE_METHODS:
            hm = h[h["method"] == m]
            g = 100 * (hm.groupby("fe")["best_cost"].median() - ref) / ref
            fig.add_trace(go.Scatter(x=g.index, y=g.values, name=SHORT[m], line=dict(color=COLORS[m], width=3)))
        fig.update_layout(title="Median gap during the run (log scale)", xaxis_title="Fitness evaluations",
                          yaxis_title="gap %", yaxis_type="log", height=380, margin=dict(l=10, r=10, t=50, b=10),
                          legend=dict(orientation="h", y=-0.25))
        c2.plotly_chart(fig, width="stretch")
    rows = []
    for m in DE_METHODS:
        x = d[d["method"] == m]
        rows.append({"Method": SHORT[m], "Average gap %": round(x["gap_pct"].mean(), 2),
                     "Worst gap %": round(x["gap_pct"].max(), 2),
                     "Runs within 1% of best-known": f"{int((x['gap_pct'] <= 1).sum())}/{len(x)}",
                     "Avg time (s)": round(x["time_s"].mean(), 2)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    h = d[d["method"] == "hybrid"].sort_values("seed")["cost"].to_numpy()
    tests = [compare(h, d[d["method"] == o].sort_values("seed")["cost"].to_numpy()) for o in ("de", "de_random")]
    ps = holm([t["p"] for t in tests])
    parts = []
    for o, t, p in zip(("Standard DE", "DE-Random"), tests, ps):
        word = "better" if t["mean_diff"] < 0 else "worse"
        parts.append(f"**{'significantly' if p < 0.05 else 'not significantly'} {word}** than {o} (p = {p:.1g})")
    msg = "**Verdict:** Hybrid DE + QL is " + " and ".join(parts) + "."
    (st.success if all(p < 0.05 for p in ps) else st.info)(msg)
    st.caption("Wilcoxon signed-rank test on the 30 paired runs (same seeds), Holm-corrected. "
               "p < 0.05 means the difference is very unlikely to be luck.")
    log = load_csv("hybrid_log.csv")
    if log is not None:
        li = log[(log["study"] == "showcase") & (log["instance"] == prob.name)]
        if len(li):
            with st.expander("🤖 What the Hybrid learned (all 30 runs)"):
                _action_chart(_phase_actions(li))


def _phase_actions(li):
    """Flatten the 30 runs into one action list ordered by relative position in the run."""
    li = li.copy()
    li["pos"] = li["generation"] / (li.groupby("seed")["generation"].transform("max") + 1)
    return li.sort_values("pos")["action"].tolist()


# ======================================================================= Decoder
def _decoder(prob: Problem):
    bal = prob.balanced()
    D = bal.m * bal.n
    st.markdown("DE does not search over shipment amounts directly. Each candidate is a list of **priority keys** "
                "(one number per route). Routes are filled from the highest key down, each time shipping "
                "`min(supply left, demand left)`. This always gives a valid whole-number plan.")
    c1, c2 = st.columns([1, 1])
    src = c1.radio("Keys", ["random", "best keys from my last DE run"], key="dec_src", horizontal=True)
    seed = c2.number_input("random seed", 0, 10_000, 0, key="dec_seed")
    keys = np.random.default_rng(int(seed)).random(D)
    if src != "random":
        cands = [r for (k, m), r in S.results.items() if k == pid(prob) and m in DE_METHODS and
                 r.diagnostics.get("keys") is not None]
        if cands:
            keys = np.asarray(cands[-1].diagnostics["keys"])
        else:
            st.info("No DE run on this problem yet (Solve tab) - showing random keys.")
    X, unmet, steps = decode(keys, bal, return_steps=True)
    nm = [f"({bal.row_labels[k // bal.n]},{bal.col_labels[k % bal.n]})" for k in range(D)]
    rows = [{"Step": i, "Route": nm[k], "Key": round(float(keys[k]), 3), "Ship": q,
             "Supply left": str(s), "Demand left": str(d), "Note": note}
            for i, (k, q, s, d, note) in enumerate(steps, 1) if q > 0 or note]
    c1, c2 = st.columns([3, 2])
    c1.caption("Only steps that ship something are listed (later routes get 0 because supply/demand is used up).")
    c1.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=min(420, 38 + 35 * len(rows)))
    with c2:
        st.markdown("**Resulting plan**")
        st.dataframe(pd.DataFrame(X, index=bal.row_labels, columns=bal.col_labels), width="stretch")
        st.metric("Cost of this plan", fmt(prob.evaluate(bal.strip_dummy(X))["objective"],
                                          4 if prob.objective2 is not None else 2))
        if unmet:
            st.warning(f"{unmet} units could not be shipped because of route constraints - DE adds a penalty.")
