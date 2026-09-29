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
from ui.tabs import render_problem, render_race, render_solve  # noqa: E402

st.set_page_config(page_title="Transportation Problem Lab", page_icon="🚚", layout="wide")
S = st.session_state
for k in ("results", "pareto", "risk", "race"):
    S.setdefault(k, {})
if "source_pending" in S:
    S.source = S.pop("source_pending")

# ------------------------------------------------------------------ sidebar: problem
cat = [r for r in instance_catalog() if r["group"] != "Study instances"]   # study data lives in results/
names = [r["name"] for r in cat] + [CUSTOM]
info = {r["name"]: r for r in cat}
with st.sidebar:
    st.title("🚚 Transportation Lab")
    st.caption("MODI · Differential Evolution · Hybrid DE + Q-Learning")
    source = st.selectbox(
        "Problem", names, key="source",
        format_func=lambda n: "✎ My own problem" if n == CUSTOM else
        f"{n} · {info[n]['group']} · {info[n]['description'][:40]}")

is_custom = source == CUSTOM
tabs = st.tabs(["📋 Problem", "▶️ Solve", "🏁 DE vs DE-Random vs Hybrid"])

with tabs[0]:
    if is_custom:
        prob, errors = render_builder()
        for e in errors:
            st.error(e)
    else:
        prob = get_instance(source)
        render_problem(prob, False)
        if st.button("✎ Edit a copy of this problem"):
            _set_from_problem(prob)
            S.source_pending = CUSTOM
            st.rerun()

# ------------------------------------------------------------------ sidebar: method
with st.sidebar:
    if prob is None:
        st.error("Fix the problem inputs (Problem tab) to enable solving.")
        method = "hybrid"
    else:
        if S.get("method_for") != pid(prob):
            S.method = "modi" if prob.is_standard else "hybrid"
            S.method_for = pid(prob)
        method = st.radio("Method", list(METHODS), key="method",
                          format_func=lambda m: METHODS[m] + ("" if applicable(m, prob)[0] else "  (n/a)"))
        ok, why = applicable(method, prob)
        if not ok:
            st.warning(why)
        params, seed = {}, None
        if method in DE_METHODS:
            default_budget = int(prob.meta.get("fe_budget", 5000 if prob.m * prob.n <= 20 else 10000
                                               if prob.m * prob.n <= 30 else 25000))
            params["fe_budget"] = int(st.number_input("Evaluation budget", 1000, 200000, default_budget, step=1000))
            if method == "de":
                params["F"] = st.slider("F (step size)", 0.1, 1.0, 0.5, 0.05)
                params["CR"] = st.slider("CR (crossover rate)", 0.0, 1.0, 0.9, 0.05)
            seed = int(st.number_input("Random seed", 0, 10_000, 0))
            with st.expander("Advanced"):
                params["NP"] = st.slider("Population size", 10, 200, 50, 10)
                params["polish"] = st.checkbox("Polish the final plan with local search", value=not prob.is_standard,
                                               help="Helps with all-units discounts and minimum shipments.")
                if method == "hybrid":
                    pc = load_json("pilot_choice.json") or {"sr_hi": 0.10, "sr_lo": 0.05}
                    params["alpha"] = st.slider("Q-learning rate α", 0.01, 0.5, 0.1, 0.01)
                    params["gamma"] = st.slider("Q-learning discount γ", 0.0, 0.99, 0.9, 0.01)
                    params["sr_hi"], params["sr_lo"] = float(pc["sr_hi"]), float(pc["sr_lo"])
        elif method == "modi":
            params["trace"] = True
            if prob.meta.get("proposed_plan") is not None:
                if st.radio("Start from", ["VAM", "lab's proposed plan"], horizontal=True) != "VAM":
                    params["initial"] = np.array(prob.meta["proposed_plan"])
        if st.button("▶ RUN", type="primary", width="stretch", disabled=not ok):
            bar = st.progress(0.0, text="running ...")
            cb = (lambda f, b: bar.progress(min(1.0, f), text=f"best so far {b:,.4g}")) if method in DE_METHODS else None
            res = run(method, prob, params, seed=seed, callback=cb)
            bar.empty()
            S.results[(pid(prob), method)] = res
            if res.objective is not None:
                st.success(f"Done: {res.objective:,.2f} in {res.time_s:.2f} s - see the Solve tab")
            else:
                st.error(f"{METHODS[method]}: {res.status}")

if prob is not None:
    with tabs[1]:
        render_solve(prob, method)
    with tabs[2]:
        render_race(prob, is_custom)
