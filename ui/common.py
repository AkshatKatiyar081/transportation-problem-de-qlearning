"""Shared helpers for the Streamlit dashboard."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.problem import Problem, list_instances, load_instance

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
CUSTOM = "__custom__"
COLORS = {"de": "#2a78d6", "de_random": "#8a8a8a", "hybrid": "#d6552a", "milp": "#1b9e77", "lp": "#1b9e77",
          "modi": "#1b9e77", "naive": "#7570b3"}


@st.cache_data
def instance_catalog():
    return list_instances()


@st.cache_data
def cached_instance(name) -> dict:
    return load_instance(name).to_dict()


def get_instance(name) -> Problem:
    return Problem.from_dict(cached_instance(name))


def pid(prob: Problem) -> str:
    return hashlib.md5(prob.to_json().encode()).hexdigest()[:12]


@st.cache_data
def load_csv(name):
    p = RESULTS / name
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception:  # noqa: BLE001 - partial / malformed file: behave as if missing
        return None


@st.cache_data
def load_json(name):
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


def fmt(v, digits=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    if isinstance(v, (int, np.integer)) or float(v).is_integer():
        return f"{int(round(v)):,}"
    return f"{v:,.{digits}f}"


def matrix_df(prob: Problem, M, fmt_int=False):
    M = np.asarray(M)
    return pd.DataFrame(M, index=prob.row_labels[:M.shape[0]], columns=prob.col_labels[:M.shape[1]])


def problem_table(prob: Problem):
    """Cost matrix with Supply column and Demand row (textbook layout)."""
    df = pd.DataFrame(prob.cost, index=prob.row_labels, columns=prob.col_labels)
    df = df.apply(lambda c: c.map(lambda v: int(v) if float(v).is_integer() else v))
    df["Supply"] = prob.supply
    dem = list(prob.demand) + [f"{int(prob.supply.sum())} / {int(prob.demand.sum())}"]
    df.loc["Demand"] = dem
    return df.astype(str)


def plan_table(prob: Problem, X):
    X = np.asarray(X).astype(int)
    df = pd.DataFrame(X, index=prob.row_labels, columns=prob.col_labels)
    df["Shipped"] = X.sum(1)
    df["Supply"] = prob.supply
    tot = pd.DataFrame([list(X.sum(0)) + [int(X.sum()), int(prob.supply.sum())],
                        list(prob.demand) + [int(prob.demand.sum()), ""]],
                       index=["Received", "Demand"], columns=df.columns)
    return pd.concat([df, tot])


def style_plan(df, m, n):
    def color(v, r, c):
        try:
            v = float(v)
        except (TypeError, ValueError):
            return ""
        if r < m and c < n and v > 0:
            return "background-color: rgba(214,85,42,0.25); font-weight: 600"
        return ""

    styles = pd.DataFrame("", index=df.index, columns=df.columns)
    for r in range(df.shape[0]):
        for c in range(df.shape[1]):
            styles.iat[r, c] = color(df.iat[r, c], r, c)
        if r >= m:
            styles.iloc[r, :] = styles.iloc[r, :] + "; color: #666"
    return df.style.apply(lambda _: styles, axis=None)


def show_plan(prob: Problem, X, key=None):
    df = plan_table(prob, X).astype(str)
    st.dataframe(style_plan(df, prob.m, prob.n), width="stretch")


def convergence_figure(series, reference=None, title="Convergence", ylab="Best objective"):
    """series: list of (label, color, xs, ys)."""
    fig = go.Figure()
    for label, color, xs, ys in series:
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=label, line=dict(color=color, width=2)))
    if reference is not None:
        fig.add_hline(y=reference, line_dash="dash", line_color="black",
                      annotation_text="exact optimum", annotation_position="top right")
    fig.update_layout(title=title, xaxis_title="Fitness evaluations", yaxis_title=ylab, height=380,
                      margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.2))
    return fig


def tier_figure(prob: Problem, xmax=None):
    xmax = xmax or int(max(prob.supply.max(), prob.demand.max()))
    x = np.arange(0, xmax + 1)
    h = prob.effective_units(x.reshape(1, -1))[0]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=x, name="flat rate", line=dict(dash="dash", color="gray")))
    fig.add_trace(go.Scatter(x=x, y=h, name=prob.discount["type"].replace("_", "-") + " discount",
                             line=dict(color="#d6552a", width=2), line_shape="hv" if prob.discount["type"] == "all_units" else "linear"))
    fig.update_layout(title="Route cost per unit of base rate c:  g(x) / c", xaxis_title="Units shipped on a route",
                      yaxis_title="g(x) / c", height=320, margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation="h", y=-0.25))
    return fig


def reference_of(prob: Problem):
    v = prob.meta.get("reference_optimum")
    return None if v is None else float(v)
