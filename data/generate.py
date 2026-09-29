"""Create the frozen instance files in data/instances/.

    python data/generate.py            # write any missing instance files
    python data/generate.py --force    # overwrite all instance files
    python data/generate.py --check    # compare generator output with committed files
    python data/generate.py --verify   # (re)compute reference optima with LP / MILP / MODI

The committed JSON files are the reference.  Experiments always load them and never regenerate.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "data" / "instances"
GEN_VERSION = 1
TIERS_INC = [[30, 1.0], [80, 0.7], [None, 0.5]]


def random_tp(m, n, seed, cost_lo=5, cost_hi=50, s_lo=40, s_hi=150, d_min=20):
    rng = np.random.default_rng(seed)
    cost = rng.integers(cost_lo, cost_hi + 1, size=(m, n))
    supply = rng.integers(s_lo, s_hi + 1, size=m)
    total = int(supply.sum())
    extra = rng.multinomial(total - d_min * n, rng.dirichlet(np.ones(n)))
    demand = d_min + extra
    return rng, cost.tolist(), supply.tolist(), demand.tolist()


def base(name, group, order, desc, cost, supply, demand, **kw):
    d = {"name": name, "group": group, "order": order, "description": desc, "cost": cost,
         "supply": supply, "demand": demand, "unit": "Rs", "reference_optimum": None,
         "reference_method": None, "seed": None, "generator_version": None, "notes": ""}
    d.update(kw)
    return d


def build_all():
    I = []
    # ------------------------------------------------------------- lab examples
    dist = [[50, 30, 220], [90, 45, 170], [50, 200, 50]]
    I.append(base("A1", "Lab examples", 1, "Lab Exp. 5, Activity 1 - 8 loads, cost Rs 10 per load per km",
                  [[10 * v for v in r] for r in dist], [1, 3, 4], [3, 3, 2],
                  row_labels=["X", "Y", "Z"], col_labels=["A", "B", "C"],
                  notes="Distance matrix (km) = " + json.dumps(dist) + "; optimum 385 load-km = Rs 3,850. "
                        "Optimal plan is degenerate (4 positive cells < m+n-1 = 5)."))
    I.append(base("A3", "Lab examples", 2, "Lab Exp. 5, Activity 3 - warehouses to markets",
                  [[6, 3, 5, 4], [5, 9, 2, 7], [5, 7, 8, 6]], [22, 15, 8], [7, 12, 17, 9],
                  row_labels=["A", "B", "C"], col_labels=["P", "Q", "R", "S"],
                  notes="The lab manual states 150; the LP optimum is 149."))
    I.append(base("A5", "Lab examples", 3, "Lab Exp. 5, Activity 5 - test the proposed plan for optimality",
                  [[12, 4, 9, 5, 9], [8, 1, 6, 6, 7], [1, 12, 4, 7, 7], [10, 15, 6, 9, 1]],
                  [55, 45, 30, 50], [40, 20, 50, 30, 40],
                  proposed_plan=[[25, 0, 0, 30, 0], [0, 20, 25, 0, 0], [15, 0, 15, 0, 0], [0, 0, 10, 0, 40]],
                  notes="Supplies/demands recovered from the proposed plan (cost 795)."))
    # ---------------------------------------------------------- study instances
    for nm, size, seed, order in (("5", 5, 42, 10), ("10", 10, 43, 12)):
        _, c, s, d = random_tp(size, size, seed)
        I.append(base("L" + nm, "Study instances", order, f"Generated {size}x{size} linear TP (seed {seed})",
                      c, s, d, seed=seed, generator_version=GEN_VERSION))
        I.append(base("N" + nm, "Study instances", order + 1,
                      f"L{nm} with incremental quantity discounts (first 30 units x1.0, next 50 x0.7, rest x0.5)",
                      c, s, d, discount={"type": "incremental", "tiers": TIERS_INC}, seed=seed,
                      generator_version=GEN_VERSION))
    _, c, s, d = random_tp(5, 5, 99)
    I.append(base("N5-pilot", "Study instances", 14, "Pilot instance for tuning Q-learning thresholds (seed 99)",
                  c, s, d, discount={"type": "incremental", "tiers": TIERS_INC}, seed=99,
                  generator_version=GEN_VERSION))
    # ----------------------------------------------------- non-standard examples
    rng, c, s, d = random_tp(4, 5, 7, s_lo=60, s_hi=160)
    fc = rng.integers(10, 41, size=(4, 5)) * 10
    I.append(base("X1", "Non-standard examples", 20,
                  "All-units discount (discontinuous) + fixed charge per used route",
                  c, s, d, discount={"type": "all_units", "tiers": [[20, 1.0], [60, 0.85], [None, 0.7]]},
                  fixed_charge=fc.tolist(), seed=7, generator_version=GEN_VERSION,
                  notes="All-units: the whole shipment on a route is charged at the rate of its tier "
                        "(<=20: 100%, 21-60: 85%, >60: 70%)."))
    rng, c, s, d = random_tp(4, 5, 8, s_lo=60, s_hi=140)
    forb = np.zeros((4, 5), dtype=int)
    for (i, j) in [(0, 1), (2, 3), (3, 0)]:
        forb[i, j] = 1
    cap = [[60 if (i + j) % 2 == 0 else None for j in range(5)] for i in range(4)]
    I.append(base("X2", "Non-standard examples", 21,
                  "Forbidden routes, route capacities (60) and minimum shipment of 15 units per used route",
                  c, s, d, forbidden=forb.tolist(), capacity=cap, min_ship=[[15] * 5 for _ in range(4)],
                  seed=8, generator_version=GEN_VERSION,
                  notes="Minimum shipment makes the feasible set discontinuous: x = 0 or x >= 15."))
    rng, c, s, d = random_tp(4, 5, 9)
    ca = np.array(c)
    co2 = np.clip(60 - ca + rng.integers(-8, 9, size=ca.shape), 3, None)
    I.append(base("X3", "Non-standard examples", 22, "Bi-objective: cost vs CO2 emissions (cheap routes pollute more)",
                  c, s, d, objective2={"name": "CO2 (kg/unit)", "matrix": co2.tolist(), "weight": 0.5},
                  seed=9, generator_version=GEN_VERSION))
    rng, c, s, d = random_tp(4, 5, 10)
    ca = np.array(c, dtype=float)
    rel = rng.uniform(0.05, 0.2, size=ca.shape)
    cheap = ca <= np.quantile(ca, 0.35)
    rel[cheap] = rng.uniform(0.4, 0.7, size=cheap.sum())       # cheap routes are volatile
    sd = np.round(ca * rel, 1)
    I.append(base("X4", "Non-standard examples", 23, "Uncertain unit costs (mean +- sd); robust objective mean + k*sd",
                  c, s, d, noise={"sd": sd.tolist(), "k": 1.5}, seed=10, generator_version=GEN_VERSION,
                  notes="The cheapest routes have the most volatile prices, so the robust plan differs "
                        "from the plan built on mean costs."))
    I.append(base("U1", "Non-standard examples", 24, "Unbalanced problem (supply 160 > demand 130)",
                  [[4, 8, 8], [16, 24, 16], [8, 16, 24]], [76, 82, 2], [72, 30, 28],
                  notes="A dummy destination absorbs the 30 units of unused supply."))
    return I


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    fresh = build_all()
    if a.check:
        bad = 0
        keys = ("cost", "supply", "demand", "discount", "fixed_charge", "capacity", "forbidden", "min_ship",
                "objective2", "noise")
        for d in fresh:
            p = OUT / f"{d['name']}.json"
            if not p.exists():
                print(f"MISSING {p.name}")
                bad += 1
                continue
            old = json.loads(p.read_text())
            diff = [k for k in keys if json.dumps(old.get(k)) != json.dumps(d.get(k))]
            if diff:
                print(f"DIFFERS {p.name}: {diff}")
                bad += 1
        print("check OK" if not bad else f"{bad} problem(s)")
        return
    for d in fresh:
        p = OUT / f"{d['name']}.json"
        if p.exists() and not a.force:
            continue
        p.write_text(json.dumps(d, indent=2))
        print("wrote", p.name)
    if a.verify:
        from core.problem import load_instance
        from algorithms.registry import reference_solution
        for d in fresh:
            p = OUT / f"{d['name']}.json"
            obj = json.loads(p.read_text())
            prob = load_instance(d["name"])
            val, method, status = reference_solution(prob)
            obj["reference_optimum"] = None if val is None else round(val, 6)
            obj["reference_method"] = method
            obj["reference_status"] = status
            p.write_text(json.dumps(obj, indent=2))
            print(f"{d['name']:9s} reference = {val} ({method}, {status})")


if __name__ == "__main__":
    main()
