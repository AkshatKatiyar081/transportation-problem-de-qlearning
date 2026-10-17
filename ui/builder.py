"""Custom problem builder: standard or non-standard transportation problems."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import streamlit as st

from core.problem import Problem
from ui.common import get_instance, instance_catalog

S = st.session_state
MATS = ("fc", "cap", "minship", "forb", "obj2", "sd")


# ------------------------------------------------------------------ state
def _labels(k, prefix):
    return [f"{prefix}{i + 1}" for i in range(k)]


def _blank(m, n, fill=0.0, dtype=float):
    return np.full((m, n), fill, dtype=dtype)


def _set_from_problem(p: Problem):
    """Load a Problem into the builder base tables."""
    m, n = p.m, p.n
    rows, cols = list(p.row_labels), list(p.col_labels)
    base = {}
    df = pd.DataFrame(p.cost, index=rows, columns=cols)
    df["Supply"] = p.supply.astype(int)
    base["main"] = df
    base["demand"] = pd.DataFrame([p.demand.astype(int)], index=["Demand"], columns=cols)
    tiers = p.discount["tiers"] if p.discount else [[10, 1.0], [50, 0.9], [None, 0.8]]
    base["tiers"] = pd.DataFrame({"Up to (units)": [t[0] for t in tiers], "Rate x base cost": [t[1] for t in tiers]})

    def mk(a, fill=0.0):
        return pd.DataFrame(a if a is not None else _blank(m, n, fill), index=rows, columns=cols)

    base["fc"] = mk(p.fixed_charge)
    cap = None if p.capacity is None else np.where(np.isinf(p.capacity), np.nan, p.capacity)
    base["cap"] = mk(cap, np.nan)
    base["minship"] = mk(p.min_ship)
    base["forb"] = pd.DataFrame(p.forbidden if p.forbidden is not None else _blank(m, n, False, bool),
                                index=rows, columns=cols)
    base["obj2"] = mk(None if p.objective2 is None else p.objective2["matrix"])
    base["sd"] = mk(None if p.noise is None else p.noise["sd"])
    S.cust_base = base
    S.cust_name = p.name
    S.cust_disc = p.cost_type if p.discount else "linear"
    S.cust_use_fc = p.fixed_charge is not None
    S.cust_use_cap = p.capacity is not None
    S.cust_use_min = p.min_ship is not None
    S.cust_use_forb = p.forbidden is not None
    S.cust_use_obj2 = p.objective2 is not None
    S.cust_obj2_name = (p.objective2 or {}).get("name", "CO2 (kg/unit)")
    S.cust_obj2_w = float((p.objective2 or {}).get("weight", 0.5))
    S.cust_use_noise = p.noise is not None
    S.cust_k = float((p.noise or {}).get("k", 1.0))
    S.cust_m, S.cust_n = m, n
    S.cust_rows, S.cust_cols = ", ".join(rows), ", ".join(cols)
    S.cust_latest = {}
    S.cust_ver = S.get("cust_ver", 0) + 1


def init_builder():
    if "cust_base" not in S:
        p = Problem("Custom", [[4, 6, 8], [5, 3, 7]], [30, 40], [20, 25, 25],
                    row_labels=["S1", "S2"], col_labels=["D1", "D2", "D3"])
        _set_from_problem(p)


def _current(name):
    return S.cust_latest.get(name, S.cust_base[name])


def _resize():
    m, n = int(S.cust_m), int(S.cust_n)
    rows = [r.strip() for r in S.cust_rows.split(",") if r.strip()]
    cols = [c.strip() for c in S.cust_cols.split(",") if c.strip()]
    rows = (rows + _labels(m, "S"))[:m] if len(rows) < m else rows[:m]
    cols = (cols + _labels(n, "D"))[:n] if len(cols) < n else cols[:n]
    if len(set(rows)) < m:
        rows = _labels(m, "S")
    if len(set(cols)) < n:
        cols = _labels(n, "D")

    def fit(df, fill):
        a = df.to_numpy()
        out = np.full((m, n), fill, dtype=a.dtype if a.dtype != object else float)
        mm, nn = min(m, a.shape[0]), min(n, a.shape[1])
        out[:mm, :nn] = a[:mm, :nn]
        return pd.DataFrame(out, index=rows, columns=cols)

    main = _current("main")
    new_main = fit(main[main.columns[:-1]], 0.0)
    sup = main["Supply"].to_numpy()
    new_main["Supply"] = [int(sup[i]) if i < len(sup) and not pd.isna(sup[i]) else 0 for i in range(m)]
    dem = _current("demand").to_numpy()[0]
    base = {"main": new_main,
            "demand": pd.DataFrame([[int(dem[j]) if j < len(dem) and not pd.isna(dem[j]) else 0 for j in range(n)]],
                                   index=["Demand"], columns=cols),
            "tiers": _current("tiers")}
    for k in MATS:
        fill = False if k == "forb" else (np.nan if k == "cap" else 0.0)
        base[k] = fit(_current(k), fill)
    S.cust_base = base
    S.cust_rows, S.cust_cols = ", ".join(rows), ", ".join(cols)
    S.cust_latest = {}
    S.cust_ver += 1


def _fill(name, value):
    df = _current(name).copy()
    df.loc[:, :] = value
    S.cust_base[name] = df
    S.cust_latest.pop(name, None)
    S.cust_ver += 1


def _sd_percent(pct):
    main = _current("main")
    cost = main[main.columns[:-1]].to_numpy(dtype=float)
    S.cust_base["sd"] = pd.DataFrame(np.round(cost * pct / 100, 2), index=main.index, columns=main.columns[:-1])
    S.cust_latest.pop("sd", None)
    S.cust_ver += 1


def _editor(name, **kw):
    df = st.data_editor(S.cust_base[name], key=f"ed_{name}_{S.cust_ver}", width="stretch", **kw)
    S.cust_latest[name] = df
    return df


def _random_problem(m, n, seed, kind):
    rng = np.random.default_rng(seed)
    cost = rng.integers(5, 51, (m, n))
    sup = rng.integers(40, 151, m)
    dem = 20 + rng.multinomial(int(sup.sum()) - 20 * n, rng.dirichlet(np.ones(n)))
    kw = {}
    if kind == "unbalanced":
        dem = np.maximum(10, (dem * 0.8).astype(int))
    elif kind == "discount":
        kw["discount"] = {"type": "incremental", "tiers": [[30, 1.0], [80, 0.7], [None, 0.5]]}
    return Problem("Random", cost, sup, dem, **kw)


# ---------------------------------------------------------------- render
def render_builder():
    """Draw the builder; return (Problem or None, list of errors)."""
    init_builder()
    st.markdown("Build your own problem. Tables accept typing or **paste from Excel** (select a cell, Ctrl+V).")
    with st.expander("Start from ...", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            names = [r["name"] for r in instance_catalog()]
            pick = st.selectbox("Pre-loaded example", names, key="cust_pick")
            if st.button("Load into builder", key="cust_load"):
                _set_from_problem(get_instance(pick))
                st.rerun()
        with c2:
            up = st.file_uploader("Import problem (.json)", type=["json"], key="cust_upload")
            if up is not None and S.get("cust_upload_id") != up.file_id:
                try:
                    _set_from_problem(Problem.from_dict(json.loads(up.getvalue())))
                    S.cust_upload_id = up.file_id
                    st.rerun()
                except Exception as e:  # noqa: BLE001
                    st.error(f"Could not read file: {e}")
        c3, c4, c5, c6 = st.columns(4)
        rm = c3.number_input("rows", 2, 10, 4, key="rnd_m")
        rn = c4.number_input("cols", 2, 10, 5, key="rnd_n")
        kind = c5.selectbox("type", ["standard", "unbalanced", "discount"], key="rnd_kind")
        rs = c6.number_input("seed", 0, 10_000, 1, key="rnd_seed")
        if st.button("Generate random problem", key="rnd_go"):
            _set_from_problem(_random_problem(int(rm), int(rn), int(rs), kind))
            st.rerun()

    c1, c2, c3 = st.columns([1, 1, 2])
    c1.number_input("Sources (rows)", 2, 10, key="cust_m", on_change=_resize)
    c2.number_input("Destinations (cols)", 2, 10, key="cust_n", on_change=_resize)
    c3.text_input("Problem name", key="cust_name")
    c1, c2 = st.columns(2)
    c1.text_input("Source names (comma separated)", key="cust_rows", on_change=_resize)
    c2.text_input("Destination names (comma separated)", key="cust_cols", on_change=_resize)

    st.markdown("**Unit costs and supplies**")
    main = _editor("main")
    st.markdown("**Demands**")
    dem = _editor("demand")

    st.markdown("#### Non-standard features (optional)")
    with st.expander("Nonlinear cost: quantity discounts / fixed charges", expanded=S.cust_disc != "linear" or S.cust_use_fc):
        opts = ["linear", "incremental", "all_units"]
        st.radio("Cost model", opts, horizontal=True, key="cust_disc",
                                 format_func=lambda v: {"linear": "Linear (constant unit cost)",
                                                        "incremental": "Incremental discount (like tax slabs)",
                                                        "all_units": "All-units discount (discontinuous)"}[v])
        if S.cust_disc != "linear":
            st.caption("Each row is a tier: units up to the limit are charged at rate x base cost. "
                       "Leave the last limit empty (= and above).")
            _editor("tiers", num_rows="dynamic",
                    column_config={"Up to (units)": st.column_config.NumberColumn(min_value=1, step=1),
                                   "Rate x base cost": st.column_config.NumberColumn(min_value=0.0, max_value=5.0,
                                                                                     step=0.05)})
        st.checkbox("Fixed charge per used route (paid once if x > 0)", key="cust_use_fc")
        if S.cust_use_fc:
            v = st.number_input("Fill all cells with", 0.0, value=100.0, key="fill_fc")
            st.button("Fill", key="fill_fc_btn", on_click=_fill, args=("fc", v))
            _editor("fc")

    with st.expander("Route constraints: forbidden routes / capacity / minimum shipment",
                     expanded=S.cust_use_forb or S.cust_use_cap or S.cust_use_min):
        st.checkbox("Forbidden routes (tick = not allowed)", key="cust_use_forb")
        if S.cust_use_forb:
            _editor("forb")
        st.checkbox("Route capacity (max units; empty = unlimited)", key="cust_use_cap")
        if S.cust_use_cap:
            v = st.number_input("Fill all cells with", 0.0, value=50.0, key="fill_cap")
            st.button("Fill", key="fill_cap_btn", on_click=_fill, args=("cap", v))
            _editor("cap")
        st.checkbox("Minimum shipment (a used route carries x = 0 or x >= L)", key="cust_use_min")
        if S.cust_use_min:
            v = st.number_input("Fill all cells with", 0.0, value=10.0, key="fill_min")
            st.button("Fill", key="fill_min_btn", on_click=_fill, args=("minship", v))
            _editor("minship")

    with st.expander("Multiple objectives: second objective (e.g. CO2, time, risk)", expanded=S.cust_use_obj2):
        st.checkbox("Add a second objective", key="cust_use_obj2")
        if S.cust_use_obj2:
            st.text_input("Name (per unit shipped)", key="cust_obj2_name")
            st.slider("Weight on the second objective (0 = cost only, 1 = second only)", 0.0, 1.0,
                      step=0.05, key="cust_obj2_w")
            st.caption("Both objectives are normalised, then combined: (1-w)*cost/Nc + w*obj2/N2. "
                       "The Solve tab's 'Trade-off' panel sweeps w to draw the trade-off curve.")
            _editor("obj2")

    with st.expander("Noisy / uncertain costs", expanded=S.cust_use_noise):
        st.checkbox("Unit costs are uncertain (give a standard deviation per route)", key="cust_use_noise")
        if S.cust_use_noise:
            st.slider("Risk aversion k (objective = expected cost + k * sd of total cost)", 0.0, 3.0,
                      step=0.1, key="cust_k")
            pct = st.number_input("Set every sd to this % of the unit cost", 0.0, 200.0, 20.0, key="sd_pct")
            st.button("Apply %", key="sd_pct_btn", on_click=_sd_percent, args=(pct,))
            _editor("sd")

    # --------------------------------------------------------- build Problem
    errors = []
    cost_df = main[main.columns[:-1]]
    try:
        cost = cost_df.to_numpy(dtype=float)
        sup = main["Supply"].to_numpy(dtype=float)
        dvec = dem.to_numpy(dtype=float)[0]
    except (TypeError, ValueError):
        return None, ["All cost, supply and demand cells must be numbers."]
    if np.isnan(cost).any() or np.isnan(sup).any() or np.isnan(dvec).any():
        errors.append("Some cost, supply or demand cells are empty.")
    if (np.abs(sup - np.round(sup)) > 1e-9).any() or (np.abs(dvec - np.round(dvec)) > 1e-9).any():
        errors.append("Supplies and demands must be whole numbers.")
    if errors:
        return None, errors
    kw = {}
    try:
        if S.cust_disc != "linear":
            t = S.cust_latest.get("tiers", S.cust_base["tiers"]).dropna(how="all")
            tiers = [[None if pd.isna(u) else int(u), float(r)] for u, r in t.to_numpy()]
            if any(pd.isna(r) for _, r in tiers):
                raise ValueError("every tier needs a rate")
            if not tiers:
                raise ValueError("add at least one discount tier")
            kw["discount"] = {"type": S.cust_disc, "tiers": tiers}
        if S.cust_use_fc:
            kw["fixed_charge"] = _current("fc").to_numpy(dtype=float)
        if S.cust_use_cap:
            c = _current("cap").to_numpy(dtype=float)
            kw["capacity"] = np.where(np.isnan(c), np.inf, c)
        if S.cust_use_min:
            kw["min_ship"] = np.nan_to_num(_current("minship").to_numpy(dtype=float))
        if S.cust_use_forb:
            kw["forbidden"] = _current("forb").fillna(False).to_numpy(dtype=bool)
        if S.cust_use_obj2:
            kw["objective2"] = {"name": S.cust_obj2_name, "weight": S.cust_obj2_w,
                                "matrix": np.nan_to_num(_current("obj2").to_numpy(dtype=float))}
        if S.cust_use_noise:
            kw["noise"] = {"sd": np.nan_to_num(_current("sd").to_numpy(dtype=float)), "k": S.cust_k}
        p = Problem(S.cust_name or "Custom", cost, sup.astype(int), dvec.astype(int),
                    row_labels=list(main.index), col_labels=list(cost_df.columns), **kw)
    except Exception as e:  # noqa: BLE001
        return None, [f"Invalid problem: {e}"]
    if p.forbidden is not None:
        U = p.balanced().upper_bounds()
        if (U.sum(1)[:p.m] == 0).any() or (U.sum(0)[:p.n] == 0).any():
            errors.append("Every source and destination needs at least one allowed route.")
            return None, errors
    st.download_button("Download this problem (.json)", p.to_json(), file_name=f"{p.name}.json",
                       mime="application/json", key="cust_dl")
    return p, []
