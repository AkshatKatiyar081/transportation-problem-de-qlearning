"""Run the experiment batch.

    python experiments/run_all.py --study main          # 30 seeds x 3 DE variants x 7 instances
    python experiments/run_all.py --study sens_de       # F x CR grid on N10 (10 seeds)
    python experiments/run_all.py --study sens_hybrid   # alpha x gamma x eps_frac on N10 (10 seeds)
    python experiments/run_all.py --study all
    python experiments/run_all.py --estimate            # time one run per pair, print projected total

Exact / deterministic methods (MODI, LP, MILP, naive) are run once per instance (+5 timing repeats)
into results/exact.csv.  Results are appended and finished runs are skipped, so the script can resume.
"""
from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from common import BUDGET, RESULTS, git_commit, machine_info, parse_seeds, run_task
from algorithms.registry import applicable, run
from core.problem import load_instance

CORE = ["A1", "A3", "A5", "L5", "N5", "L10", "N10"]
DE_METHODS = ["de", "de_random", "hybrid"]
SENS_INSTANCE = "N10"
SENS_SEEDS = "0-9"


def hybrid_thresholds():
    p = RESULTS / "pilot_choice.json"
    if p.exists():
        c = json.loads(p.read_text())
        return {"sr_hi": c["sr_hi"], "sr_lo": c["sr_lo"]}
    print("WARNING: results/pilot_choice.json not found - using defaults (run pilot.py first)")
    return {"sr_hi": 0.10, "sr_lo": 0.05}


def build_tasks(study, instances, methods, seeds):
    th = hybrid_thresholds()
    tasks = []
    if study == "main":
        for inst in instances:
            for m in methods:
                for s in seeds:
                    params = dict(th) if m == "hybrid" else {}
                    tasks.append({"study": "main", "instance": inst, "method": m, "seed": s, "params": params})
    elif study == "sens_de":
        for F in (0.3, 0.5, 0.7, 0.9):
            for CR in (0.1, 0.3, 0.5, 0.7, 0.9):
                for s in parse_seeds(SENS_SEEDS):
                    tasks.append({"study": "sens_de", "instance": SENS_INSTANCE, "method": "de", "seed": s,
                                  "params": {"F": F, "CR": CR}, "config_id": f"F{F}_CR{CR}"})
    elif study == "sens_hybrid":
        for g in (0.5, 0.9):
            for al in (0.05, 0.1, 0.3):
                for ef in (0.2, 0.4, 0.6):
                    for s in parse_seeds(SENS_SEEDS):
                        tasks.append({"study": "sens_hybrid", "instance": SENS_INSTANCE, "method": "hybrid",
                                      "seed": s, "params": {**th, "gamma": g, "alpha": al, "eps_frac": ef},
                                      "config_id": f"g{g}_a{al}_e{ef}"})
    return tasks


def _key(r):
    return (r["study"], str(r.get("config_id") or ""), r["instance"], r["method"], int(r["seed"]))


RUN_COLS = ["study", "config_id", "instance", "method", "seed", "params_json", "cost", "reference", "gap_pct",
            "success", "fe_used", "fe_to_target", "fe_to_opt", "time_s", "neutral_frac_mean", "distinct_plans_final",
            "status", "F", "CR", "alpha", "gamma", "eps_frac", "sr_hi", "sr_lo", "git_commit", "timestamp"]


def append(path, rows, cols=None):
    """Append rows to a CSV with a fixed column order (so later rows never shift columns)."""
    if not rows:
        return
    df = pd.DataFrame(rows)
    if cols is not None:
        df = df.reindex(columns=cols)
    elif path.exists():
        df = df.reindex(columns=pd.read_csv(path, nrows=0).columns)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def run_exact(instances):
    path = RESULTS / "exact.csv"
    done = set()
    if path.exists():
        old = pd.read_csv(path)
        done = set(zip(old["instance"], old["method"]))
    rows = []
    for inst in instances:
        prob = load_instance(inst)
        for m in ("modi", "lp", "milp", "naive"):
            if (inst, m) in done or not applicable(m, prob)[0]:
                continue
            if m == "milp" and prob.is_linear_program:
                continue
            times = []
            for _ in range(5):
                r = run(m, prob, {"time_limit": 300.0 if inst != "N10" else 1800.0})
                times.append(r.time_s)
            d = r.diagnostics
            rows.append({"instance": inst, "method": m, "cost": r.objective, "time_mean_s": float(np.mean(times)),
                         "time_sd_s": float(np.std(times)), "status": r.status,
                         "bound": d.get("bound"), "mip_gap": d.get("mip_gap"),
                         "iterations": d.get("iterations"), "feasible": r.feasible})
            print(f"exact {inst:4s} {m:6s} {r.objective:.2f} {r.status} {np.mean(times)*1000:.1f} ms")
    append(path, rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", default="main", choices=["main", "sens_de", "sens_hybrid", "all"])
    ap.add_argument("--instances", nargs="*", default=CORE)
    ap.add_argument("--methods", nargs="*", default=DE_METHODS)
    ap.add_argument("--seeds", default="0-29")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--estimate", action="store_true")
    a = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    studies = ["main", "sens_de", "sens_hybrid"] if a.study == "all" else [a.study]
    seeds = parse_seeds(a.seeds)
    if a.estimate:
        total = 0.0
        for st in studies:
            tasks = build_tasks(st, a.instances, a.methods, seeds)
            pairs = {}
            for t in tasks:
                pairs.setdefault((t["instance"], t["method"], t.get("config_id", "")), []).append(t)
            for (inst, m, cfg), ts in pairs.items():
                t0 = time.perf_counter()
                run_task(ts[0])
                dt = time.perf_counter() - t0
                total += dt * len(ts)
            print(f"{st}: {len(tasks)} runs")
        print(f"projected total: {total/60:.1f} CPU-minutes  (~{total/60/max(1,a.jobs):.1f} min with {a.jobs} jobs)")
        return
    (RESULTS / "machine.json").write_text(json.dumps({**machine_info(), "git_commit": git_commit(),
                                                       "budgets": BUDGET}, indent=2))
    run_exact(a.instances if a.study in ("main", "all") else [])
    runs_path = RESULTS / "runs.csv"
    done = set()
    if runs_path.exists():
        done = {_key(r) for r in pd.read_csv(runs_path, keep_default_na=False).to_dict("records")}
    for st in studies:
        tasks = [t for t in build_tasks(st, a.instances, a.methods, seeds)
                 if _key({**t, "config_id": t.get("config_id", "")}) not in done]
        print(f"{st}: {len(tasks)} runs to do")
        t0 = time.time()
        commit = git_commit()
        buf = {"run": [], "history": [], "log": [], "q": []}
        with Pool(a.jobs) as pool:
            for k, out in enumerate(pool.imap_unordered(run_task, tasks), 1):
                out["run"]["git_commit"] = commit
                out["run"]["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
                buf["run"].append(out["run"])
                buf["history"] += out["history"]
                buf["log"] += out["log"]
                buf["q"] += out["q"]
                if k % 30 == 0 or k == len(tasks):
                    append(runs_path, buf["run"], RUN_COLS)
                    append(RESULTS / "history.csv", buf["history"])
                    append(RESULTS / "hybrid_log.csv", buf["log"])
                    append(RESULTS / "qtables.csv", buf["q"])
                    buf = {"run": [], "history": [], "log": [], "q": []}
                    print(f"  {k}/{len(tasks)}  {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
