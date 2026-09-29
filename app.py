"""Transportation Problem Lab - Streamlit dashboard.

    streamlit run app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402

from algorithms.registry import DE_METHODS, METHODS, applicable, run  # noqa: E402
from ui.builder import _set_from_problem, render_builder  # noqa: E402
from ui.common import CUSTOM, get_instance, instance_catalog, load_json, pid  # noqa: E402
from ui.tabs import (render_compare, render_decoder, render_hybrid, render_pareto_risk,  # noqa: E402
                     render_plan_check, render_problem, render_solve)

st.set_page_config(page_title="Transportation Problem Lab", page_icon="🚚", layout="wide")
S = st.session_state
for k, v in (("results", {}), ("quick", {}), ("pareto", {}), ("risk", {})):
    S.setdefault(k, v)
if "source_pending" in S:
    S.source = S.pop("source_pending")

# ------------------------------------------------------------------ sidebar: problem
cat = instance_catalog()
names = [r["name"] for r in cat] + [CUSTOM]
info = {r["name"]: r for r in cat}
with st.sidebar:
    st.title("🚚 Transportation Lab")
    st.caption("VAM + MODI · LP / MILP · Differential Evolution · Hybrid DE + Q-Learning")
    source = st.selectbox(
        "Problem", names, key="source",
        format_func=lambda n: "✎ Custom problem (build your own)" if n == CUSTOM else
        f"{n} · {info[n]['group'].split()[0]} · {info[n]['description'][:48]}")

is_custom = source == CUSTOM
tabs = st.tabs(["📋 Problem", "▶️ Solve", "📊 Compare", "🤖 Hybrid insight", "🔢 Decoder", "✅ Check a plan",
                "⚖️ Pareto & Risk"])

with tabs[0]:
    if is_custom:
        prob, errors = render_builder()
        for e in errors:
            st.error(e)
        if prob is not None:
            with st.expander("Preview of the problem as the solvers see it", expanded=False):
                render_problem(prob, True)
    else:
        prob = get_instance(source)
        if st.button("✎ Edit a copy in the custom builder"):
            _set_from_problem(prob)
            S.source_pending = CUSTOM
            st.rerun()
        render_problem(prob, False)

# ------------------------------------------------------------------ sidebar: method
with st.sidebar:
    if prob is None:
        st.error("Fix the problem inputs (Problem tab) to enable solving.")
        method = "hybrid"
    else:
        default = "modi" if prob.is_standard else ("milp" if applicable("milp", prob)[0] else "hybrid")
        if S.get("method_for") != pid(prob):
            S.method = default
            S.method_for = pid(prob)
        method = st.radio("Method", list(METHODS), key="method",
                          format_func=lambda m: METHODS[m] + ("" if applicable(m, prob)[0] else "  (n/a)"))
        ok, why = applicable(method, prob)
        if not ok:
            st.warning(why)
        params = {}
        seed = None
        if method in DE_METHODS:
            size = prob.m * prob.n
            params["NP"] = st.slider("Population size NP", 10, 200, 50, 10)
            params["fe_budget"] = int(st.number_input("Fitness-evaluation budget", 1000, 200000,
                                                      5000 if size <= 20 else 10000 if size <= 30 else 50000,
                                                      step=1000))
            if method == "de":
                params["F"] = st.slider("Mutation factor F", 0.1, 1.0, 0.5, 0.05)
                params["CR"] = st.slider("Crossover rate CR", 0.0, 1.0, 0.9, 0.05)
            if method == "hybrid":
                pc = load_json("pilot_choice.json") or {"sr_hi": 0.10, "sr_lo": 0.05}
                with st.expander("Q-learning settings"):
                    params["alpha"] = st.slider("learning rate α", 0.01, 0.5, 0.1, 0.01)
                    params["gamma"] = st.slider("discount γ", 0.0, 0.99, 0.9, 0.01)
                    params["eps_frac"] = st.slider("ε decays to 0.05 over this share of the run", 0.1, 1.0, 0.4, 0.05)
                    params["sr_hi"] = st.number_input("state 'productive' if SR >", 0.0, 1.0, float(pc["sr_hi"]), 0.01)
                    params["sr_lo"] = st.number_input("state 'stagnant' if SR ≤", 0.0, 1.0, float(pc["sr_lo"]), 0.01)
            params["polish"] = st.checkbox("Local-search polish of the final plan", value=not prob.is_standard,
                                           help="Cycle moves that can reach non-basic optima (all-units discounts, "
                                                "minimum shipments). Off in the research study.")
            c1, c2 = st.columns(2)
            rnd = c2.checkbox("random seed", value=False)
            seed = None if rnd else int(c1.number_input("seed", 0, 10_000, 0))
        elif method == "milp":
            params["time_limit"] = st.slider("Time limit (s)", 5, 600, 60, 5)
        elif method == "modi" and prob.meta.get("proposed_plan") is not None:
            if st.radio("Start from", ["VAM", "lab's proposed plan"], horizontal=True) != "VAM":
                params["initial"] = np.array(prob.meta["proposed_plan"])
        if method == "modi":
            params["trace"] = True
        if st.button("▶ RUN", type="primary", width="stretch", disabled=not ok):
            bar = st.progress(0.0, text="running ...")
            cb = (lambda f, b: bar.progress(min(1.0, f), text=f"best so far {b:,.4g}")) if method in DE_METHODS else None
            with st.spinner(f"{METHODS[method]} ..."):
                res = run(method, prob, params, seed=seed, callback=cb)
            bar.empty()
            S.results[(pid(prob), method)] = res
            if res.objective is not None:
                st.success(f"{METHODS[method]}: objective {res.objective:,.4g} in {res.time_s:.2f} s")
            else:
                st.error(f"{METHODS[method]}: {res.status}")
    mi = load_json("machine.json")
    if mi:
        st.caption(f"Pre-computed study results: {mi.get('date')}, {mi.get('cpu')}")

# ------------------------------------------------------------------ other tabs
if prob is not None:
    with tabs[1]:
        render_solve(prob, method)
    with tabs[2]:
        render_compare(prob, is_custom)
    with tabs[3]:
        render_hybrid(prob, is_custom)
    with tabs[4]:
        render_decoder(prob)
    with tabs[5]:
        render_plan_check(prob)
    with tabs[6]:
        render_pareto_risk(prob)
