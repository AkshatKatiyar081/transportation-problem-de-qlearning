"""Tune the Q-learning state thresholds on the pilot instance (separate from the test data).

    python experiments/pilot.py [--instance N5-pilot] [--seeds 100-104] [--jobs 2]

Rule (spec B.6): keep pairs where every state is >= 10% of generations, choose the lowest mean
gap; ties (< 0.1 percentage points) -> lowest mean FE-to-optimum -> closest to (0.10, 0.02).
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool

import numpy as np
import pandas as pd

from common import RESULTS, parse_seeds, run_task

HI = (0.05, 0.10, 0.20)
LO = (0.01, 0.02, 0.05)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="N5-pilot")
    ap.add_argument("--seeds", default="100-104")
    ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args()
    tasks = [{"study": "pilot", "instance": a.instance, "method": "hybrid", "seed": s,
              "params": {"sr_hi": hi, "sr_lo": lo}, "config_id": f"{hi}_{lo}"}
             for hi in HI for lo in LO for s in parse_seeds(a.seeds)]
    with Pool(a.jobs) as pool:
        outs = pool.map(run_task, tasks)
    rows = []
    for o in outs:
        r = o["run"]
        st = pd.Series([x["state"] for x in o["log"]])
        frac = st.value_counts(normalize=True).reindex([0, 1, 2], fill_value=0.0)
        rows.append({"sr_hi": r["sr_hi"], "sr_lo": r["sr_lo"], "seed": r["seed"], "cost": r["cost"],
                     "gap_pct": r["gap_pct"], "fe_to_opt": r["fe_to_opt"],
                     "frac_state0": frac[0], "frac_state1": frac[1], "frac_state2": frac[2]})
    df = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "pilot.csv", index=False)
    g = df.groupby(["sr_hi", "sr_lo"]).mean(numeric_only=True).reset_index()
    g["min_share"] = g[["frac_state0", "frac_state1", "frac_state2"]].min(axis=1)
    g["fe_to_opt"] = g["fe_to_opt"].fillna(np.inf)
    ok = g[g["min_share"] >= 0.10]
    rule = "all states >= 10%"
    if ok.empty:
        ok = g[g["min_share"] == g["min_share"].max()]
        rule = "no pair reached 10% in every state - chose largest minimum state share"
    best_gap = ok["gap_pct"].min()
    cand = ok[ok["gap_pct"] <= best_gap + 0.1].copy()
    cand["dist"] = (cand["sr_hi"] - 0.10).abs() + (cand["sr_lo"] - 0.02).abs()
    cand = cand.sort_values(["fe_to_opt", "dist"])
    ch = cand.iloc[0]
    choice = {"sr_hi": float(ch["sr_hi"]), "sr_lo": float(ch["sr_lo"]), "rule": rule,
              "mean_gap_pct": float(ch["gap_pct"]), "mean_fe_to_opt": float(ch["fe_to_opt"]),
              "state_shares": [float(ch["frac_state0"]), float(ch["frac_state1"]), float(ch["frac_state2"])]}
    (RESULTS / "pilot_choice.json").write_text(json.dumps(choice, indent=2))
    print(g.round(3).to_string(index=False))
    print("chosen:", choice)


if __name__ == "__main__":
    main()
