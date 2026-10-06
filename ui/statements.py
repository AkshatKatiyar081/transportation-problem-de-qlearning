"""Problem statements for the Problem tab: scenario text, standard-TP checklist, formulation, network diagram.

Only the problems in SHOWN appear in the dashboard; the others (A1, A3, U1, study instances) stay in
data/instances/ for the tests and the experiment study.  Scenario texts are illustrative stories; every
number in them is read from the problem data, so they always match the tables.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

SHOWN = ["A5", "L10", "S1", "S2", "S3", "X1", "X2", "X3", "X4"]

# name -> short dropdown label, category, title, scenario (str.format fields: see _facts), task
STATEMENTS = {
    "A5": {
        "label": "Lab Exp. 5 - is the proposed plan optimal?",
        "category": "Lab example",
        "title": "Lab Exp. 5, Activity 5: is the manager's plan optimal?",
        "scenario": (
            "A manufacturer ships one product from **{m} factories ({rows})** to **{n} warehouses ({cols})**. "
            "The factories can supply {supply} units and the warehouses need {demand} units, so total supply "
            "equals total demand ({total} units). Each unit shipped on a route costs between Rs {c_lo} and "
            "Rs {c_hi} (cost table below), and this cost per unit is the same however many units travel on it.\n\n"
            "The transport manager has already proposed the shipping plan shown below, which costs "
            "**Rs {plan_cost}**."),
        "task": "Test whether the proposed plan is optimal with the MODI (u-v) method. If it is not, improve it "
                "step by step until no cheaper plan exists.",
    },
    "L10": {
        "label": "10x10 standard problem - plants to cities",
        "category": "Standard (large)",
        "title": "10 x 10 standard transportation problem: plants to city distribution centres",
        "scenario": (
            "A packaged-food company has **{m} plants ({rows})** and **{n} city distribution centres ({cols})**. "
            "This month the plants can supply {total} cartons in total and the centres need exactly the same "
            "amount. Moving one carton from a plant to a centre costs between Rs {c_lo} and Rs {c_hi}, the same "
            "for every carton on that route, and all {routes} routes are open with no limit on quantity."),
        "task": "Decide how many cartons to send on each route so that every centre gets its demand and the "
                "total transport cost is as small as possible.",
    },
    "S1": {
        "label": "12x12 bulk discounts + fixed truck cost",
        "category": "Showcase",
        "title": "12 x 12 distribution with volume discounts and a fixed truck cost per route",
        "scenario": (
            "An FMCG company moves {total} cartons from **{m} warehouses ({rows})** to **{n} markets ({cols})**. "
            "The carrier's base rate on a route is Rs {c_lo}-{c_hi} per carton, but it is discounted with "
            "volume: {tiers}.\n\n"
            "On top of that, using a route at all means hiring a dedicated truck for it: a **fixed cost of "
            "Rs {f_lo}-{f_hi} per used route**, whether it carries 1 carton or 200."),
        "task": "Choose the shipments that meet every market's demand at the lowest total cost "
                "(discounted transport cost + truck costs).",
    },
    "S2": {
        "label": "12x12 fixed-charge problem",
        "category": "Showcase",
        "title": "12 x 12 fixed-charge transportation problem: factories to construction sites",
        "scenario": (
            "A building-materials company delivers {total} tonnes from **{m} factories ({rows})** to "
            "**{n} construction customers ({cols})**. Each tonne costs Rs {c_lo}-{c_hi} to move, depending on "
            "the route.\n\n"
            "Every delivery lane that is used also needs a one-off setup (truck mobilisation, permit, loading "
            "crew): a **fixed charge of Rs {f_lo}-{f_hi} per used lane**, paid in full whether 1 tonne or 300 "
            "tonnes move on it."),
        "task": "Decide which lanes to open and how much to send on each, so that all customers are supplied at "
                "the lowest total cost (transport + setup charges).",
    },
    "S3": {
        "label": "10x10 high fixed costs + steep discounts",
        "category": "Showcase",
        "title": "10 x 10 distribution with high fixed route costs and steep volume discounts",
        "scenario": (
            "A beverage distributor supplies **{n} retailers ({cols})** from **{m} depots ({rows})**; "
            "{total} crates must be delivered. The base rate is Rs {c_lo}-{c_hi} per crate, with steep volume "
            "discounts: {tiers}.\n\n"
            "Each route that is used needs its own vehicle and driver for the month: a **fixed cost of "
            "Rs {f_lo}-{f_hi} per used route**, large compared with the per-crate rates."),
        "task": "Find the delivery plan with the lowest total cost. Cheap-looking routes are not automatically "
                "good: concentrating crates on a few large routes earns the discounts and saves fixed costs.",
    },
    "X1": {
        "label": "All-units discount + fixed charge",
        "category": "Non-standard",
        "title": "All-units quantity discount with a fixed charge per route",
        "scenario": (
            "**{m} plants ({rows})** supply **{n} customers ({cols})** with {total} units in total. The base "
            "rate is Rs {c_lo}-{c_hi} per unit. The carrier uses **all-units pricing**: the size of a shipment "
            "decides the rate for the whole shipment - {tiers}. So sending a few more units can make a "
            "shipment cheaper in total (for example 61 units can cost less than 60).\n\n"
            "Each used route also pays a **fixed charge of Rs {f_lo}-{f_hi}**."),
        "task": "Choose the shipments with the lowest total cost, taking the price breaks and fixed charges into "
                "account.",
    },
    "X2": {
        "label": "Closed routes, capacities, minimum loads",
        "category": "Non-standard",
        "title": "Transportation with closed routes, route capacities and minimum loads",
        "scenario": (
            "**{m} plants ({rows})** supply **{n} sites ({cols})** with {total} units; the rate is "
            "Rs {c_lo}-{c_hi} per unit. Three kinds of route rules apply:\n\n"
            "- **Closed routes:** {forbidden} cannot be used (for example a bridge under repair).\n"
            "- **Capacity:** on {n_cap} routes the available vehicles can carry at most {cap} units.\n"
            "- **Minimum load:** the carrier refuses small loads, so a route is either not used or carries at "
            "least {min_ship} units."),
        "task": "Find the cheapest plan that meets all demand and obeys every route rule.",
    },
    "X3": {
        "label": "Cost vs CO2 emissions",
        "category": "Non-standard",
        "title": "Two objectives: transport cost versus CO2 emissions",
        "scenario": (
            "**{m} plants ({rows})** supply **{n} customers ({cols})** with {total} units. Every route has a "
            "cost (Rs {c_lo}-{c_hi} per unit) and an emission (**{e_lo}-{e_hi} kg CO2 per unit**). The cheapest "
            "routes use older trucks and long road hauls, so they pollute the most: the cheapest plan is the "
            "dirtiest one.\n\n"
            "Management wants a plan that balances cost and emissions with weight w = {w} on CO2. Rupees and "
            "kilograms cannot be added directly, so each is divided by a reference value (the cost of a simple "
            "cheapest-first plan and the CO2 of a simple cleanest-first plan) before they are combined."),
        "task": "Find the plan with the best weighted balance of cost and CO2. The Solve tab also shows the "
                "whole trade-off curve (Pareto front).",
    },
    "X4": {
        "label": "Uncertain freight rates (risk)",
        "category": "Non-standard",
        "title": "Uncertain freight rates: choosing a plan that is safe, not just cheap on average",
        "scenario": (
            "**{m} plants ({rows})** supply **{n} customers ({cols})** with {total} units. Freight rates are "
            "not fixed: each route has an **expected rate** (Rs {c_lo}-{c_hi} per unit) and a **standard "
            "deviation** (Rs {sd_lo}-{sd_hi}). The cheapest routes are booked on the spot market and are the "
            "most volatile, with a standard deviation of 80-120% of their mean.\n\n"
            "A risk-averse manager judges a plan by *expected cost + {k} x standard deviation of the total "
            "cost* (route rates vary independently)."),
        "task": "Find the plan with the lowest risk-adjusted cost. The Solve tab can simulate the real cost of "
                "the plan under random rates.",
    },
}


# ------------------------------------------------------------------ facts used in the texts
def _rng(a):
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    lo, hi = a.min(), a.max()
    f = (lambda v: f"{v:,.0f}") if np.allclose(a, np.round(a)) else (lambda v: f"{v:,.1f}")
    return f(lo), f(hi)


def _span(labels):
    return f"{labels[0]}-{labels[-1]}" if len(labels) > 2 else ", ".join(labels)


def tier_text(prob):
    """Plain-English description of the discount tiers."""
    d = prob.discount
    parts, lo = [], 0
    for up, r in d["tiers"]:
        rate = "the full rate" if r == 1 else f"{r:.0%} of the base rate"
        if d["type"] == "incremental":
            if lo == 0:
                parts.append(f"the first {up} units on a route pay {rate}")
            elif up is not None:
                parts.append(f"units {lo + 1}-{up} pay {rate}")
            else:
                parts.append(f"every unit above {lo} pays {rate}")
        else:
            who = f"up to {up} units" if lo == 0 else (f"{lo + 1}-{up} units" if up is not None else f"more than {lo} units")
            parts.append(f"{who}: whole shipment at {rate}")
        lo = up if up is not None else lo
    return "; ".join(parts)


def _facts(prob):
    rows, cols = prob.row_labels, prob.col_labels
    c_lo, c_hi = _rng(prob.cost)
    f = {"m": prob.m, "n": prob.n, "rows": _span(rows), "cols": _span(cols), "routes": prob.m * prob.n,
         "total": f"{int(prob.supply.sum()):,}", "supply": ", ".join(str(int(v)) for v in prob.supply),
         "demand": ", ".join(str(int(v)) for v in prob.demand), "c_lo": c_lo, "c_hi": c_hi}
    if prob.fixed_charge is not None:
        f["f_lo"], f["f_hi"] = _rng(prob.fixed_charge)
    if prob.discount is not None:
        f["tiers"] = tier_text(prob)
    if prob.forbidden is not None:
        cells = [f"{rows[i]} -> {cols[j]}" for i, j in zip(*np.nonzero(prob.forbidden))]
        f["forbidden"] = ", ".join(cells[:-1]) + " and " + cells[-1] if len(cells) > 1 else cells[0]
    if prob.capacity is not None:
        cap = prob.capacity[np.isfinite(prob.capacity)]
        f["n_cap"], f["cap"] = int(cap.size), _rng(cap)[1] if cap.min() == cap.max() else "-".join(_rng(cap))
    if prob.min_ship is not None:
        f["min_ship"] = _rng(prob.min_ship[prob.min_ship > 0])[1]
    if prob.objective2 is not None:
        f["e_lo"], f["e_hi"] = _rng(prob.objective2["matrix"])
        f["w"] = f"{prob.objective2['weight']:g}"
    if prob.noise is not None:
        f["sd_lo"], f["sd_hi"] = _rng(prob.noise["sd"])
        f["k"] = f"{prob.noise['k']:g}"
    plan = prob.meta.get("proposed_plan")
    if plan is not None:
        f["plan_cost"] = f"{prob.evaluate(np.array(plan))['cost']:,.0f}"
    return f


def statement(name, prob):
    """(title, scenario markdown, task) or None for problems without a written statement."""
    s = STATEMENTS.get(name)
    if s is None:
        return None
    return s["title"], s["scenario"].format(**_facts(prob)), s["task"]


def dropdown_label(name):
    s = STATEMENTS.get(name)
    return f"{name} · {s['category']} · {s['label']}" if s else name


# ------------------------------------------------------------------ standard-TP checklist
def checklist(prob):
    """Assumptions of the standard (textbook) transportation problem: [(assumption, holds, explanation)]."""
    rows = []
    why = []
    if prob.fixed_charge is not None:
        why.append("every used route pays a fixed charge, so the first unit on a route is very expensive and "
                   "each further unit is cheap")
    if prob.discount is not None:
        why.append("the rate falls with volume (" + ("the whole shipment is re-priced at each price break, "
                                                     "so the cost even jumps down" if prob.discount["type"] == "all_units"
                                                     else "later units on a route are cheaper") + ")")
    rows.append(("Cost per unit is constant on every route (route cost = c x units)", not why,
                 "Yes: doubling a shipment doubles its cost." if not why else "No: " + "; ".join(why) + "."))
    rules = []
    if prob.forbidden is not None:
        rules.append(f"{int(prob.forbidden.sum())} routes are closed")
    if prob.capacity is not None:
        rules.append(f"{int(np.isfinite(prob.capacity).sum())} routes have a capacity limit")
    if prob.min_ship is not None:
        rules.append("a used route must carry a minimum load (so x = 0 or x >= L, with a gap in between)")
    rows.append(("Every route is open and can carry any amount", not rules,
                 "Yes: no route limits." if not rules else "No: " + "; ".join(rules) + "."))
    two = prob.objective2 is not None
    rows.append(("There is one goal: minimise total cost", not two,
                 "Yes." if not two else f"No: cost and {prob.objective2['name']} are both minimised, "
                                        "and they pull in opposite directions."))
    noisy = prob.noise is not None
    rows.append(("All costs are known exactly in advance", not noisy,
                 "Yes." if not noisy else "No: each route rate is uncertain (mean and standard deviation), "
                                          "and the plan must be judged by its risk too."))
    bal = prob.is_balanced
    rows.append(("Total supply equals total demand", True,
                 f"Yes ({int(prob.supply.sum()):,} = {int(prob.demand.sum()):,})." if bal else
                 f"No ({int(prob.supply.sum()):,} vs {int(prob.demand.sum()):,}), but this is still a standard "
                 "problem: a dummy row or column with zero cost absorbs the difference."))
    return rows


def consequence(prob):
    """Markdown: what the broken assumptions mean for MODI / the naive plan / DE."""
    if prob.is_standard:
        return ("**All assumptions hold, so this is a standard transportation problem.** VAM gives a good "
                "starting plan and the MODI (u-v) test proves when a plan is optimal. MODI is exact and takes "
                "milliseconds, so it is the right method here. The DE methods also work, but they are slower "
                "and do not always reach the exact optimum; that makes this problem a good reference for when "
                "*not* to use a metaheuristic.")
    txt = ("**At least one assumption is broken, so this is not a standard transportation problem.** MODI's "
           "u-v test is only valid when every route has a constant cost per unit, one objective, no route "
           "limits and known costs, so MODI cannot prove a plan optimal here.")
    return txt


# ------------------------------------------------------------------ mathematical formulation
def _g_cases(prob):
    """LaTeX cases for the discounted quantity g(x)."""
    d = prob.discount
    lines, lo, acc = [], 0.0, 0.0
    for up, r in d["tiers"]:
        cond = (rf"{lo:g} < x \le {up:g}" if lo > 0 else rf"x \le {up:g}") if up is not None else rf"x > {lo:g}"
        if d["type"] == "incremental":
            expr = (rf"{acc:g} + {r:g}\,(x - {lo:g})" if lo > 0 else (rf"{r:g}\,x" if r != 1 else "x"))
            acc += r * ((up - lo) if up is not None else 0)
        else:
            expr = rf"{r:g}\,x" if r != 1 else "x"
        lines.append(rf"{expr} & {cond}")
        lo = float(up) if up is not None else lo
    return r"g(x) = \begin{cases} " + r" \\ ".join(lines) + r" \end{cases}"


def formulation(prob):
    """List of (caption, LaTeX) blocks describing the model of this problem."""
    unit = r"c_{ij}\,g(x_{ij})" if prob.discount is not None else r"c_{ij}\,x_{ij}"
    mean = r"\mu_{ij}" if prob.noise is not None else "c_{ij}"
    cost = rf"\sum_{{i=1}}^{{m}}\sum_{{j=1}}^{{n}} {unit.replace('c_{ij}', mean)}"
    if prob.fixed_charge is not None:
        cost += r" + \sum_{i=1}^{m}\sum_{j=1}^{n} f_{ij}\,y_{ij}"
    blocks = []
    if prob.noise is not None:
        k = f"{prob.noise['k']:g}"
        cost += rf" + {k}\sqrt{{\sum_{{i,j}} \sigma_{{ij}}^2\,x_{{ij}}^2}}"
    if prob.objective2 is not None:
        w = prob.objective2["weight"]
        blocks.append(("Objective: weighted balance of cost C(x) and emissions E(x), divided by reference values "
                       "C0 (cost of a cheapest-first plan) and E0 (CO2 of a cleanest-first plan)",
                       rf"\min\; {1 - w:g}\,\frac{{C(x)}}{{C_0}} + {w:g}\,\frac{{E(x)}}{{E_0}},\qquad "
                       rf"C(x) = {cost},\quad E(x) = \sum_{{i,j}} e_{{ij}}\,x_{{ij}}"))
    else:
        blocks.append(("Objective" + (" (expected cost + risk term)" if prob.noise is not None else ""),
                       rf"\min\; Z = {cost}"))
    sup = "=" if prob.is_balanced or prob.supply.sum() < prob.demand.sum() else r"\le"
    dem = "=" if prob.is_balanced or prob.supply.sum() > prob.demand.sum() else r"\le"
    cons = [rf"\sum_{{j=1}}^{{n}} x_{{ij}} {sup} s_i \quad i = 1,\dots,{prob.m} \;\text{{(supply)}}",
            rf"\sum_{{i=1}}^{{m}} x_{{ij}} {dem} d_j \quad j = 1,\dots,{prob.n} \;\text{{(demand)}}"]
    if prob.fixed_charge is not None:
        cons.append(r"y_{ij} = 1 \text{ if } x_{ij} > 0, \text{ else } 0 \;\text{(route used)}")
    if prob.capacity is not None:
        cons.append(r"x_{ij} \le u_{ij} \;\text{on capacity-limited routes}")
    if prob.forbidden is not None:
        cons.append(r"x_{ij} = 0 \;\text{on closed routes}")
    if prob.min_ship is not None:
        cons.append(r"x_{ij} = 0 \;\text{ or }\; x_{ij} \ge L_{ij} \;\text{(minimum load)}")
    cons.append(r"x_{ij} \ge 0 \text{ and integer}")
    blocks.append(("Subject to", r"\begin{aligned} " + r" \\ ".join(cons) + r" \end{aligned}"))
    if prob.discount is not None:
        kind = "incremental: each tier's rate applies to the units inside that tier" \
            if prob.discount["type"] == "incremental" else "all-units: the tier of x sets the rate for all x units"
        blocks.append((f"Discounted quantity g(x) ({kind})", _g_cases(prob)))
    return blocks


def symbols(prob):
    """One-line legend of the symbols used in the formulation."""
    s = [f"x_ij = units shipped from {prob.row_labels[0]}...{prob.row_labels[-1]} (i) to "
         f"{prob.col_labels[0]}...{prob.col_labels[-1]} (j)", "s_i = supply", "d_j = demand",
         "μ_ij / σ_ij = expected rate / its standard deviation" if prob.noise is not None else "c_ij = cost per unit"]
    if prob.fixed_charge is not None:
        s.append("f_ij = fixed charge of a used route")
    if prob.objective2 is not None:
        s.append("e_ij = kg CO2 per unit")
    if prob.capacity is not None:
        s.append("u_ij = route capacity")
    if prob.min_ship is not None:
        s.append("L_ij = minimum load")
    return "; ".join(s) + "."


# ------------------------------------------------------------------ network diagram
def network_figure(prob):
    """Sources on the left, destinations on the right, every allowed route as a line."""
    m, n = prob.m, prob.n
    ys = lambda k: np.linspace(1, 0, k) if k > 1 else np.array([0.5])  # noqa: E731
    yl, yr = ys(m), ys(n)
    closed = prob.forbidden if prob.forbidden is not None else np.zeros((m, n), bool)
    capped = np.isfinite(prob.capacity) if prob.capacity is not None else np.zeros((m, n), bool)
    fig = go.Figure()

    def edges(mask, name, color, width, dash=None):
        xs, yv = [], []
        for i, j in zip(*np.nonzero(mask)):
            xs += [0, 1, None]
            yv += [yl[i], yr[j], None]
        if xs:
            fig.add_trace(go.Scatter(x=xs, y=yv, mode="lines", name=name, hoverinfo="skip",
                                     line=dict(color=color, width=width, dash=dash)))

    alpha = 0.18 if m * n > 60 else 0.35
    edges(~closed & ~capped, "route", f"rgba(120,120,120,{alpha})", 1)
    edges(capped & ~closed, "capacity-limited route","rgba(230,145,56,0.9)", 2.5)
    edges(closed, "closed route", "rgba(200,40,40,0.9)", 2, "dot")
    size = lambda v: 14 + 26 * np.sqrt(np.asarray(v, float) / max(prob.supply.max(), prob.demand.max()))  # noqa: E731
    fig.add_trace(go.Scatter(x=np.zeros(m), y=yl, mode="markers+text", name="source (supply)",
                             marker=dict(size=size(prob.supply), color="#2a78d6"),
                             text=[f"{r}  {int(v):,}" for r, v in zip(prob.row_labels, prob.supply)],
                             textposition="middle left", hovertemplate="%{text}<extra>supply</extra>"))
    fig.add_trace(go.Scatter(x=np.ones(n), y=yr, mode="markers+text", name="destination (demand)",
                             marker=dict(size=size(prob.demand), color="#1b9e77"),
                             text=[f"{c}  {int(v):,}" for c, v in zip(prob.col_labels, prob.demand)],
                             textposition="middle right", hovertemplate="%{text}<extra>demand</extra>"))
    h = max(300, 34 * max(m, n) + 90)
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=30, b=10), showlegend=True,
                      legend=dict(orientation="h", y=-0.02, yanchor="top"),
                      xaxis=dict(visible=False, range=[-0.55, 1.55]), yaxis=dict(visible=False, range=[-0.08, 1.08]),
                      title=dict(text="Network: node size = supply / demand", font=dict(size=13)))
    return fig
