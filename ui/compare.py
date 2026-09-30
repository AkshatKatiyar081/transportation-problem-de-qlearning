"""Algorithm Comparison tab: compares every run made on the current problem."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from algorithms.registry import DE_METHODS, METHODS, applicable, run
from core.problem import Problem
from ui.common import COLORS, first_fe, fmt, log_run, pid, reference_of

S = st.session_state
SHORT = {"modi": "MODI", "naive": "Naive plan", "de": "Standard DE", "de_random": "DE-Random",
         "hybrid": "Hybrid DE + QL"}


def _label(k, e):
    seed = f", seed {e['seed']}" if e["seed"] is not None else ""
    return f"#{k} {SHORT[e['res'].method]}{seed}"


def _ref_text(prob):
    return "MODI optimum" if prob.meta.get("reference_status") == "optimal" else "best-known value"


# ------------------------------------------------------------------ entry point
def render_comparison(prob: Problem):
    key = pid(prob)
    st.markdown("Compares every run made on the current problem: single runs from the sidebar **RUN** button "
                "and batches started below. Clearing the table in the Solve tab also clears this page.")
    _batch_runner(prob, expanded=len(S.runlog.get(key, [])) < 2)
    log = [e for e in S.runlog.get(key, []) if e["res"].objective is not None]
    if not log:
        st.info("No runs yet. Press RUN in the sidebar, or run several seeds above.")
        return
    two = prob.objective2 is not None
    st.subheader("1. Final cost and convergence")
    _cost_and_convergence(prob, log, two)
    st.subheader("2. Solution quality vs computation time")
    _quality_vs_time(prob, log, two)
    st.subheader("3. Repeated-run statistics")
    _repeated_stats(prob, log, two)
    st.subheader("4. Plan comparison")
    _plan_comparison(prob, log)


# ------------------------------------------------------------------ batch runner
def _batch_runner(prob, expanded):
    with st.expander("Run several seeds", expanded=expanded):
        st.caption("DE is random, so one run can be lucky or unlucky. Running each method with several seeds "
                   "gives a fair comparison. MODI and the naive plan are deterministic and run once.")
        ms = [m for m in METHODS if applicable(m, prob)[0]]
        default = [m for m in ("naive", *DE_METHODS) if m in ms]
        c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
        chosen = c1.multiselect("Methods", ms, default=default, format_func=lambda m: METHODS[m],
                                key=f"cmp_methods_{pid(prob)}")
        nseed = c2.number_input("Seeds per method", 1, 30, 5, key="cmp_n")
        first = c3.number_input("First seed", 0, 10_000, 0, key="cmp_first")
        size = prob.m * prob.n
        budget = c4.number_input("Evaluation budget", 1000, 200000,
                                 int(prob.meta.get("fe_budget", 5000 if size <= 20 else 10000 if size <= 30 else 25000)),
                                 step=1000, key=f"cmp_budget_{pid(prob)}")
        polish = st.checkbox("Polish the final DE plan with local search", value=not prob.is_standard,
                             key=f"cmp_polish_{pid(prob)}")
        if st.button("Run", type="primary", key="cmp_go", disabled=not chosen):
            todo = [(m, None) for m in chosen if m not in DE_METHODS] + \
                   [(m, int(first) + i) for i in range(int(nseed)) for m in chosen if m in DE_METHODS]
            bar = st.progress(0.0)
            for k, (m, seed) in enumerate(todo):
                params = {"fe_budget": int(budget), "polish": polish} if m in DE_METHODS else {"trace": True}
                if m == "de":
                    params.update(F=0.5, CR=0.9)
                res = run(m, prob, params, seed=seed)
                log_run(prob, m, params, seed, res)
                S.results[(pid(prob), m)] = res
                bar.progress((k + 1) / len(todo), text=f"{METHODS[m]}" + (f", seed {seed}" if seed is not None else ""))
            bar.empty()
            st.rerun()


# ------------------------------------------------------------ 1. cost + convergence
def _cost_and_convergence(prob, log, two):
    ref = reference_of(prob)
    labels = [_label(k, e) for k, e in enumerate(log, 1)]
    vals = [e["res"].objective for e in log]
    feas = [e["res"].feasible for e in log]
    best = min((v for v, f in zip(vals, feas) if f), default=None)
    bar = go.Figure()
    for m in dict.fromkeys(e["res"].method for e in log):
        idx = [i for i, e in enumerate(log) if e["res"].method == m]
        bar.add_trace(go.Bar(
            x=[f"#{i + 1}" for i in idx], y=[vals[i] for i in idx], name=SHORT[m], marker_color=COLORS.get(m, "#555"),
            marker_line_color=["black" if (best is not None and feas[i] and abs(vals[i] - best) < 1e-9)
                               else "rgba(0,0,0,0)" for i in idx], marker_line_width=2.5,
            text=[fmt(vals[i], 4 if two else 0) for i in idx] if len(log) <= 10 else None, textposition="outside",
            customdata=[labels[i] for i in idx], hovertemplate="%{customdata}<br>%{y:,.2f}<extra></extra>"))
    bar.update_xaxes(type="category", categoryorder="array", categoryarray=[f"#{i + 1}" for i in range(len(log))],
                     title="run")
    if ref is not None:
        bar.add_hline(y=ref, line_dash="dash", line_color="black", annotation_text=_ref_text(prob),
                      annotation_position="bottom right")
    lo = min(vals + ([ref] if ref is not None else []))
    hi = max(vals)
    pad = (hi - lo) * 0.25 or abs(hi) * 0.02 or 1
    bar.update_layout(title="Final result of each run (outlined = best)", height=380,
                      yaxis=dict(range=[lo - pad, hi + pad], title="weighted objective" if two else "total cost"),
                      margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.25))
    curves = [(lab, e) for lab, e in zip(labels, log) if e["res"].history]
    c1, c2 = st.columns(2)
    c1.plotly_chart(bar, width="stretch")
    if curves:
        fig = go.Figure()
        shown = set()
        for lab, e in curves:
            m = e["res"].method
            xs, ys = zip(*e["res"].history)
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=SHORT[m], legendgroup=m,
                                     showlegend=m not in shown, line=dict(color=COLORS[m], width=1.6),
                                     hovertemplate=f"{lab}<br>%{{x:,}} evaluations<br>%{{y:,.2f}}<extra></extra>"))
            shown.add(m)
        for e in log:
            if e["res"].method in ("modi", "naive"):
                fig.add_hline(y=e["res"].objective, line_dash="dot", line_color=COLORS[e["res"].method],
                              annotation_text=SHORT[e["res"].method], annotation_position="bottom right")
        if ref is not None:
            fig.add_hline(y=ref, line_dash="dash", line_color="black")
        early = []
        for _, e in curves:
            h = e["res"].history
            early.append(h[min(len(h) - 1, max(1, len(h) // 10))][1])   # value after ~10% of the run
        top = max(early) if early else None
        fig.update_layout(title="Best objective so far (all DE runs)", xaxis_title="Fitness evaluations",
                          yaxis_title="objective", height=380, margin=dict(l=10, r=10, t=50, b=10),
                          legend=dict(orientation="h", y=-0.25))
        if top is not None:
            low = min(vals + ([ref] if ref is not None else []))
            fig.update_yaxes(range=[low - 0.05 * (top - low), top])
        c2.plotly_chart(fig, width="stretch")
    else:
        c2.info("Convergence curves appear once a DE method has been run.")


# ---------------------------------------------------------- 2. quality vs time
def _quality_vs_time(prob, log, two):
    fig = go.Figure()
    shown = set()
    for k, e in enumerate(log, 1):
        r = e["res"]
        fig.add_trace(go.Scatter(
            x=[max(r.time_s, 1e-4)], y=[r.objective], mode="markers", name=SHORT[r.method], legendgroup=r.method,
            showlegend=r.method not in shown, marker=dict(size=12, color=COLORS.get(r.method, "#555"),
                                                          symbol="circle" if r.feasible else "x"),
            hovertemplate=f"{_label(k, e)}<br>%{{x:.3f}} s<br>%{{y:,.2f}}<extra></extra>"))
        shown.add(r.method)
    ref = reference_of(prob)
    if ref is not None:
        fig.add_hline(y=ref, line_dash="dash", line_color="black", annotation_text=_ref_text(prob))
    fig.update_layout(xaxis_title="Computation time (s, log scale)", xaxis_type="log",
                      yaxis_title="weighted objective" if two else "total cost", height=360,
                      margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h", y=-0.25))
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(fig, width="stretch")
    c2.markdown("Lower is better on both axes: points near the **bottom-left** give good plans quickly.")
    c2.caption("Circles are feasible plans, crosses break a constraint. MODI (standard problems only) is exact "
               "and almost instant. DE methods trade time for flexibility: they also work when costs are "
               "nonlinear or extra constraints exist.")


# ----------------------------------------------------- 3. repeated-run statistics
def _repeated_stats(prob, log, two):
    ref = reference_of(prob)
    rows, groups = [], {}
    for e in log:
        groups.setdefault((e["res"].method, e["config"]), []).append(e["res"])
    for (m, cfg), rs in groups.items():
        v = np.array([r.objective for r in rs])
        row = {"Method": METHODS[m], "Settings": cfg, "Runs": len(v), "Mean": v.mean(), "Best": v.min(),
               "Worst": v.max(), "Std": v.std(ddof=1) if len(v) > 1 else 0.0,
               "Feasible": f"{sum(r.feasible for r in rs)}/{len(rs)}"}
        if ref is not None:
            gaps = 100 * (v - ref) / abs(ref)
            row["Mean gap %"] = gaps.mean()
            row["Runs within 1%"] = f"{int((gaps <= 1 + 1e-9).sum())}/{len(v)}"
            if m in DE_METHODS:
                hits = [first_fe(r, ref) for r in rs]
                hits = [h for h in hits if h is not None]
                row["Reached best-known"] = f"{len(hits)}/{len(rs)}"
                row["Median evals to best-known"] = f"{int(np.median(hits)):,}" if hits else "-"
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("Mean")
    for c in ("Reached best-known", "Median evals to best-known"):
        if c in df.columns:
            df[c] = df[c].fillna("-")
    st.dataframe(df.style.format({c: "{:,.2f}" for c in ("Mean", "Best", "Worst", "Std", "Mean gap %")
                                  if c in df.columns} if not two else
                                 {c: "{:,.4f}" for c in ("Mean", "Best", "Worst", "Std")}),
                 hide_index=True, width="stretch")
    if ref is not None:
        st.caption("*Reached best-known* counts runs whose DE search reached the reference value (polish not "
                   "counted); the median is over those runs only.")
    multi = {k: rs for k, rs in groups.items() if len(rs) > 1}
    if not multi:
        st.caption("Run a method with several seeds (Run several seeds, above) to see its spread here.")
        return
    fig = go.Figure()
    for (m, cfg), rs in multi.items():
        fig.add_trace(go.Box(y=[r.objective for r in rs], name=f"{SHORT[m]}<br><sub>{cfg}</sub>",
                             marker_color=COLORS.get(m, "#555"), boxpoints="all", jitter=0.4, pointpos=0,
                             boxmean=True))
    if ref is not None:
        fig.add_hline(y=ref, line_dash="dash", line_color="black", annotation_text=_ref_text(prob))
    fig.update_layout(title="Spread of final results over seeds (dashed line in box = mean)", showlegend=False,
                      yaxis_title="weighted objective" if two else "total cost", height=380,
                      margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, width="stretch")
    st.caption("A method is more reliable when its box is low and narrow: good results regardless of the seed.")


# ------------------------------------------------------------- 4. plan comparison
def _breakdown(prob: Problem, X):
    X = np.asarray(X, dtype=float)
    base = float((prob.cost * X).sum())
    after = float((prob.cost * prob.effective_units(X)).sum())
    out = {"Routes used": int((X > 0).sum())}
    if prob.discount is not None:
        out["Cost before discount"] = base
        out["Discount saved"] = base - after
    out["Transport cost"] = after
    if prob.fixed_charge is not None:
        out["Fixed charges"] = float(prob.fixed_charge[X > 0].sum())
    out["Total cost"] = prob.components(X)["cost"]
    if prob.objective2 is not None:
        out[prob.objective2["name"]] = float((prob.objective2["matrix"] * X).sum())
        out["Weighted objective"] = prob.evaluate(X)["objective"]
    if prob.noise is not None:
        c = prob.components(X)
        out["Cost std. deviation"] = c["cost_sd"]
        out["Robust cost"] = c["robust_cost"]
    return out


def _plan_comparison(prob, log):
    if len(log) < 2:
        st.caption("Make at least two runs to compare their plans.")
        return
    labels = [_label(k, e) for k, e in enumerate(log, 1)]
    order = np.argsort([e["res"].objective for e in log])
    c1, c2 = st.columns(2)
    a = c1.selectbox("Run A", range(len(log)), index=int(order[0]), format_func=lambda i: labels[i], key="pc_a")
    b = c2.selectbox("Run B", range(len(log)), index=int(order[-1]), format_func=lambda i: labels[i], key="pc_b")
    XA, XB = np.asarray(log[a]["res"].matrix), np.asarray(log[b]["res"].matrix)
    ba, bb = _breakdown(prob, XA), _breakdown(prob, XB)
    tab = pd.DataFrame({"Run A": ba, "Run B": bb})
    tab["B − A"] = tab["Run B"] - tab["Run A"]
    st.dataframe(tab.style.format(lambda v: f"{v:,.0f}" if float(v).is_integer() else f"{v:,.2f}"), width="stretch")
    diff = [(prob.row_labels[i], prob.col_labels[j], int(XA[i, j]), int(XB[i, j]))
            for i in range(prob.m) for j in range(prob.n) if XA[i, j] != XB[i, j]]
    if not diff:
        st.success("Both runs produced exactly the same shipment plan.")
    else:
        same = int(((XA > 0) & (XB > 0) & (XA == XB)).sum())
        st.markdown(f"**{len(diff)} routes differ**; {same} routes carry the same amount in both plans.")
    c1, c2 = st.columns(2)
    for col, X, other, title in ((c1, XA, XB, "Plan A"), (c2, XB, XA, "Plan B")):
        df = pd.DataFrame(X.astype(int), index=prob.row_labels, columns=prob.col_labels)

        def sty(_, X=X, other=other):
            s = pd.DataFrame("", index=df.index, columns=df.columns)
            for i in range(prob.m):
                for j in range(prob.n):
                    if X[i, j] != other[i, j]:
                        s.iat[i, j] = "background-color: rgba(214,85,42,0.30); font-weight:600"
                    elif X[i, j] > 0:
                        s.iat[i, j] = "background-color: rgba(42,120,214,0.15)"
            return s

        col.markdown(f"**{title}**")
        col.dataframe(df.style.apply(sty, axis=None), width="stretch")
    st.caption("Orange = route where the two plans differ; blue = route used identically by both.")
